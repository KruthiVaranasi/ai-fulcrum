"""Agentic task-completion test: what does it cost to FINISH a multi-step coding job, Haiku vs Opus?

    python -m hybrid.agentic --check                 # make each task, confirm its tests fail (no AI calls)
    python -m hybrid.agentic --yes --repeats 2       # run the agents (Claude subscription quota)
    python -m hybrid.agentic --report

Each task starts from a small, working food-ordering backend (agentic/base) with one piece broken or
missing. A Claude Code agent works in a fresh temporary copy: it may read and edit files and run the
test suite, nothing else. Afterwards the suite is re-run here, and a run only counts as solved if every
test passes AND the tests were left untouched. Cost is Claude Code's own figure for the whole session
(all turns): list prices for each model that ran, with that model's prompt-caching rates. It is the cost
of finishing, not of one request.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from router.backends import find_claude_cli

ROOT = Path(__file__).parent.parent
BASE = ROOT / "agentic" / "base"
RUNS = ROOT / "results" / "agentic_runs.jsonl"
REPORT = ROOT / "results" / "agentic_report.json"
VENV_SCRIPTS = ROOT / ".venv" / "Scripts"
PROMPT = ("This repository is a small food-ordering backend in Python. Some tests in tests/ are failing. "
          "Find out why and change the code in orders/ so that the whole test suite passes. "
          "Run the tests with: python -m pytest -q. Do not modify anything in tests/.")
# Exact model ids: the "haiku" alias moved from Haiku 4.5 (the cheap model in the rest of this study) to Haiku 5.5.
MODELS = {"haiku-4.5": ("claude-haiku-4-5", {"MAX_THINKING_TOKENS": "0"}),
          "haiku-5.5": ("claude-haiku-5-5", {"MAX_THINKING_TOKENS": "0"}),
          "opus": ("claude-opus-5-5", {})}


def _edit(path, old, new):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) == 1, (path.name, old[:40])
    path.write_text(s.replace(old, new, 1), encoding="utf-8")


def t_discount(d):  # two bugs in one function
    _edit(d / "orders/cart.py", "round(price(item) * qty * 0.10)", "round(price(item) * 0.10)")
    _edit(d / "orders/cart.py", "if qty >= 5)", "if qty > 5)")


def t_peak(d):  # boundary bug
    _edit(d / "orders/delivery.py", "start <= hour < end", "start < hour <= end")


def t_coupons(d):  # missing feature across two files
    (d / "orders/coupons.py").unlink()
    _edit(d / "orders/cart.py", "from .coupons import coupon_discount\n", "")
    _edit(d / "orders/cart.py", "    def total(self, coupon: str | None = None) -> int:\n        amount = self.subtotal() - self.bulk_discount()\n"
          "        if coupon:\n            amount -= coupon_discount(coupon, amount)\n        return max(amount, 0)",
          "    def total(self) -> int:\n        return max(self.subtotal() - self.bulk_discount(), 0)")


def t_report(d):  # missing feature
    _edit(d / "orders/report.py", '''        "early": sum(o["actual_min"] < o["promised_min"] for o in delivered),
        "on_time": sum(o["actual_min"] == o["promised_min"] for o in delivered),
        "late": len(late),
        "average_delay": round(sum(o["actual_min"] - o["promised_min"] for o in late) / len(late), 1) if late else 0.0,
''', "")


def t_units(d):  # data stored in the wrong unit, found through other modules
    _edit(d / "orders/menu.py", '"""The menu. Prices are whole rupees."""', '"""The menu. Prices are stored in paise (1 rupee = 100 paise)."""')
    for item, rupees in {"idli": 40, "dosa": 60, "thali": 150, "biryani": 220, "samosa": 15, "chai": 20}.items():
        _edit(d / "orders/menu.py", f'"{item}": {rupees},', f'"{item}": {rupees * 100},')


def t_status(d):  # whole module missing
    (d / "orders/status.py").unlink()


TASKS = {"discount": t_discount, "peak": t_peak, "coupons": t_coupons, "report": t_report, "units": t_units, "status": t_status}


def tests_hash(d):
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted((d / "tests").rglob("*.py")))).hexdigest()


def run_tests(d):
    out = subprocess.run([str(VENV_SCRIPTS / "python.exe"), "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d,
                         capture_output=True, text=True, timeout=120)
    last = (out.stdout.strip().splitlines() or [""])[-1]
    return out.returncode == 0, last


def make(task, d):
    shutil.copytree(BASE, d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    TASKS[task](d)


def check():
    for task in TASKS:
        with tempfile.TemporaryDirectory(prefix=f"agentic-{task}-") as tmp:
            d = Path(tmp) / "repo"
            make(task, d)
            ok, last = run_tests(d)
            print(f"{task:9} tests before the agent: {'PASS (task is broken!)' if ok else 'fail, as intended'}  [{last}]")


def run_agent(task, model, timeout=1200):
    alias, extra_env = MODELS[model]
    with tempfile.TemporaryDirectory(prefix=f"agentic-{task}-") as tmp:
        d = Path(tmp) / "repo"
        make(task, d)
        before = tests_hash(d)
        env = dict(os.environ, **extra_env)
        env["PATH"] = str(VENV_SCRIPTS) + os.pathsep + env.get("PATH", "")  # `python` = the venv's, with pytest
        cmd = [find_claude_cli(), "-p", PROMPT, "--model", alias, "--output-format", "json",
               "--permission-mode", "acceptEdits",
               "--allowedTools", "Read,Edit,Write,Glob,Grep,Bash(python -m pytest:*)",
               "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence", "--max-budget-usd", "3"]
        t0 = time.time()
        try:
            out = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
            data = json.loads(out.stdout or "{}")
        except subprocess.TimeoutExpired:
            data = {"is_error": True, "result": "timeout"}
        except json.JSONDecodeError:
            data = {"is_error": True, "result": (out.stdout or out.stderr)[:300]}
        wall = time.time() - t0
        ok, last = run_tests(d)
        untouched = tests_hash(d) == before
        u = data.get("usage") or {}
        return {"task": task, "model": model, "solved": ok and untouched, "tests_pass": ok, "tests_untouched": untouched,
                "pytest": last, "turns": data.get("num_turns"), "cost_usd": data.get("total_cost_usd"),
                "duration_s": round((data.get("duration_ms") or wall * 1000) / 1000, 1), "wall_s": round(wall, 1),
                "error": data.get("result") if data.get("is_error") else None, "subtype": data.get("subtype"),
                "input_tokens": u.get("input_tokens"), "cache_read_tokens": u.get("cache_read_input_tokens"),
                "cache_write_tokens": u.get("cache_creation_input_tokens"), "output_tokens": u.get("output_tokens"),
                "models_used": sorted((data.get("modelUsage") or {}).keys())}


def load_runs():
    return [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()] if RUNS.exists() else []


def live(repeats, max_runs, only_task=None, only_model=None):
    runs = load_runs()
    done = {}
    for r in runs:
        done[(r["task"], r["model"])] = done.get((r["task"], r["model"]), 0) + 1
    n = 0
    for rep in range(repeats):
        for task in TASKS:
            if only_task and task != only_task:
                continue
            for model in MODELS:
                if only_model and model != only_model:
                    continue
                if done.get((task, model), 0) > rep:
                    continue
                if n >= max_runs:
                    print("Run budget reached; re-run to continue."); return
                r = run_agent(task, model)
                n += 1
                with RUNS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
                done[(task, model)] = done.get((task, model), 0) + 1
                cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "?"
                print(f"{task:9} {model:14} {'SOLVED' if r['solved'] else 'not solved'}  turns {r['turns']}  {cost}  {r['duration_s']}s  "
                      f"[{r['pytest']}]{'  ERROR ' + str(r['error'])[:120] if r['error'] else ''}", flush=True)
                if r["error"] and any(w in str(r["error"]).lower() for w in ("limit", "quota", "rate")):
                    print("Usage limit reported; stopping. Re-run later to resume."); return


def report():
    runs = [r for r in load_runs() if r["cost_usd"] is not None]
    if not runs:
        sys.exit("No runs yet.")
    out = {"runs": len(runs), "models": {}}
    for model in MODELS:
        rs = [r for r in runs if r["model"] == model]
        if not rs:
            continue
        solved = [r for r in rs if r["solved"]]
        out["models"][model] = {
            "runs": len(rs), "solved_rate": len(solved) / len(rs),
            "avg_turns": sum(r["turns"] or 0 for r in rs) / len(rs),
            "avg_cost_per_task": sum(r["cost_usd"] for r in rs) / len(rs),
            "cost_per_solved_task": sum(r["cost_usd"] for r in rs) / max(len(solved), 1),
            "avg_minutes": sum(r["duration_s"] for r in rs) / len(rs) / 60,
            "tests_tampered": sum(not r["tests_untouched"] for r in rs),
        }
    m = out["models"]
    for h in ("haiku-4.5", "haiku-5.5"):
        if h in m and "opus" in m:
            out[f"opus_over_{h}_cost_per_solved_task"] = m["opus"]["cost_per_solved_task"] / m[h]["cost_per_solved_task"]
    out["per_task"] = {}
    for task in TASKS:
        out["per_task"][task] = {model: {"solved": sum(r["solved"] for r in runs if r["task"] == task and r["model"] == model),
                                         "runs": sum(1 for r in runs if r["task"] == task and r["model"] == model),
                                         "avg_cost": (lambda c: sum(c) / len(c) if c else None)([r["cost_usd"] for r in runs if r["task"] == task and r["model"] == model]),
                                         "avg_turns": (lambda t: sum(t) / len(t) if t else None)([r["turns"] or 0 for r in runs if r["task"] == task and r["model"] == model])}
                                 for model in MODELS}
    REPORT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n| | Solved | Avg turns | Cost per task | Cost per solved task | Minutes per task |\n|---|---|---|---|---|---|")
    for model, s in m.items():
        print(f"| {model} | {s['solved_rate']:.0%} ({s['runs']} runs) | {s['avg_turns']:.1f} | ${s['avg_cost_per_task']:.4f} | "
              f"${s['cost_per_solved_task']:.4f} | {s['avg_minutes']:.1f} |")
    for k, v in out.items():
        if k.startswith("opus_over_"):
            print(f"{k}: {v:.1f}x")
    print("\n| Task | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for task, d in out["per_task"].items():
        cells = [f"{v['solved']}/{v['runs']} solved, {v['avg_turns']:.0f} turns, ${v['avg_cost']:.4f}" if v["runs"] else "—" for v in d.values()]
        print(f"| {task} | " + " | ".join(cells) + " |")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=24)
    ap.add_argument("--task"); ap.add_argument("--model")
    args = ap.parse_args()
    if args.check:
        check(); return
    if args.yes:
        live(args.repeats, args.max_runs, args.task, args.model)
    elif not args.report:
        sys.exit("Runs Claude Code agents on your subscription quota. Add --yes (or --check / --report).")
    report()


if __name__ == "__main__":
    main()
