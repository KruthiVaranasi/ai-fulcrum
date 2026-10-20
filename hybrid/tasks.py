"""Task-completion test: cost to FINISH a task, not cost per request.

    python -m hybrid.tasks --check                      # run the reference solutions, print expected values
    python -m hybrid.tasks --yes --max-claude-calls 80  # live run (resumes; re-run after a usage limit)
    python -m hybrid.tasks --report                     # tables from what has run so far

Each task (hybrid/task_specs.py) is attempted up to MAX_ATTEMPTS times. After a failed attempt the model
gets its previous code and the failing test cases (input, expected, got), like CI output, and tries again.
A task is done when every hidden test passes. Every attempt is a separate request, so the cost of a task
is the sum of its attempts; later attempts are bigger because they carry the previous code and failures.

Strategies compared on the same tasks:
  - Haiku alone (up to 4 attempts), Opus alone (up to 4 attempts);
  - Haiku first, escalate to a fresh Opus run after k failed Haiku attempts (computed from the two runs:
    cost = Haiku attempts spent + the Opus run, so nothing is double counted or left out).
Results go to results/tasks_runs.jsonl (task code and costs only; the tasks were written for this project).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from router.backends import ClaudeCodeBackend

from .grading import extract_code
from .task_specs import TASKS

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "results" / "tasks_runs.jsonl"
REPORT = ROOT / "results" / "tasks_report.json"
MAX_ATTEMPTS = 4
MODELS = {"claude/haiku-nothink": lambda: ClaudeCodeBackend("haiku", thinking=False),
          "claude/opus": lambda: ClaudeCodeBackend("opus")}
MARK = "@@RESULTS@@"


def run_cases(code, t, timeout=15):
    """Evaluate every test expression against `code` in a separate process. Returns a list of reprs,
    an exception class name per case, or None for every case if the program could not run."""
    program = "\n\n".join([code, t["setup"], "import json as _json",
                           f"_CASES = _json.loads({json.dumps(json.dumps(t['cases']))})",
                           "_out = []\nfor _e in _CASES:\n    try:\n        _out.append(repr(eval(_e)))\n"
                           "    except Exception as _x:\n        _out.append(type(_x).__name__)",
                           f"print({MARK!r} + _json.dumps(_out))"])
    with tempfile.TemporaryDirectory(prefix="router-task-") as d:
        path = Path(d) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            out = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timed out (possible infinite loop)"
    for line in reversed(out.stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):]), None
    err = (out.stderr.strip().splitlines() or ["program did not run"])[-1]
    return None, err[:300]


def expected(t):
    res, err = run_cases(t["reference"], t)
    if err:
        raise RuntimeError(f"reference for {t['id']} failed: {err}")
    return res


def feedback(t, got, err, exp):
    if got is None:
        return f"The program failed before any test ran: {err}", 0
    fails = [(e, g, x) for e, g, x in zip(t["cases"], got, exp) if g != x]
    lines = [f"{len(t['cases']) - len(fails)}/{len(t['cases'])} tests passed. Failing cases:"]
    lines += [f"- {e}  ->  got {g}, expected {x}" for e, g, x in fails[:5]]
    if len(fails) > 5:
        lines.append(f"- ... and {len(fails) - 5} more")
    return "\n".join(lines), len(t["cases"]) - len(fails)


def prompt_for(t, prev_code=None, fb=None):
    p = t["spec"] + "\n\nReturn only the complete Python code in one ```python block, no explanation."
    if prev_code is not None:
        p = (t["spec"] + f"\n\nYour previous code:\n```python\n{prev_code}\n```\n\nRunning the hidden tests gave:\n{fb}\n\n"
             "Fix the code. Return only the complete corrected Python code in one ```python block, no explanation.")
    return p


def load_runs():
    runs = {}
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                runs[(r["model"], r["task"])] = r
    return runs


def live(max_calls):
    runs, calls = load_runs(), 0
    RUNS.parent.mkdir(exist_ok=True)
    for model, make in MODELS.items():
        backend = make()
        for t in TASKS:
            if (model, t["id"]) in runs:
                continue
            exp, attempts, code, fb = expected(t), [], None, None
            for a in range(1, MAX_ATTEMPTS + 1):
                if calls >= max_calls:
                    print("Call budget reached; re-run to continue (unfinished tasks restart from attempt 1).")
                    return
                r = backend.ask(prompt_for(t, code, fb))
                calls += 1
                if "error" in r:
                    print(f"  {model} {t['id']} attempt {a}: error {r['error'][:160]}")
                    if any(w in r["error"].lower() for w in ("limit", "rate", "quota", "overloaded")):
                        print("Usage limit reported; stopping. Re-run later to resume.")
                        return
                    break
                code = extract_code(r["text"])
                got, err = run_cases(code, t)
                fb, passed = feedback(t, got, err, exp)
                ok = passed == len(t["cases"])
                attempts.append({"attempt": a, "passed": passed, "total": len(t["cases"]), "ok": ok,
                                 "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                                 "latency_s": r["latency_s"], "cost_usd": r["equiv_cost_usd"], "code": code})
                print(f"  {model:22} {t['id']:9} attempt {a}: {passed}/{len(t['cases'])}  ${r['equiv_cost_usd']:.4f}  {r['latency_s']:.0f}s")
                if ok:
                    break
            if len(attempts) == MAX_ATTEMPTS or (attempts and attempts[-1]["ok"]):
                with RUNS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": model, "task": t["id"], "solved": attempts[-1]["ok"],
                                        "attempts": attempts}) + "\n")
    print(f"Claude calls this run: {calls}")


def summarise(runs, model, tasks):
    rs = [runs[(model, t)] for t in tasks]
    n_att = [len(r["attempts"]) for r in rs]
    cost = [sum(a["cost_usd"] for a in r["attempts"]) for r in rs]
    lat = [sum(a["latency_s"] for a in r["attempts"]) for r in rs]
    first = [r["attempts"][0] for r in rs]
    solved = [r["solved"] for r in rs]
    return {"tasks": len(rs), "first_try_pass": sum(a["ok"] for a in first) / len(rs),
            "first_try_cost_per_request": sum(a["cost_usd"] for a in first) / len(rs),
            "solved_within_4": sum(solved) / len(rs), "avg_attempts": sum(n_att) / len(rs),
            "cost_per_task": sum(cost) / len(rs), "time_per_task_s": sum(lat) / len(rs),
            "cost_per_solved_task": sum(cost) / max(sum(solved), 1)}


def cascade(runs, tasks, k, cheap="claude/haiku-nothink", strong="claude/opus"):
    """Haiku first; after k failed Haiku attempts, start a fresh Opus run."""
    cost = lat = calls = solved = 0
    for t in tasks:
        h, s = runs[(cheap, t)]["attempts"], runs[(strong, t)]["attempts"]
        used = h[:k]
        cost += sum(a["cost_usd"] for a in used); lat += sum(a["latency_s"] for a in used); calls += len(used)
        if any(a["ok"] for a in used):
            solved += 1
            continue
        cost += sum(a["cost_usd"] for a in s); lat += sum(a["latency_s"] for a in s); calls += len(s)
        solved += runs[(strong, t)]["solved"]
    n = len(tasks)
    return {"tasks": n, "solved": solved / n, "avg_calls": calls / n, "cost_per_task": cost / n,
            "time_per_task_s": lat / n, "cost_per_solved_task": cost / max(solved, 1)}


def report():
    runs = load_runs()
    tasks = [t["id"] for t in TASKS if all((m, t["id"]) in runs for m in MODELS)]
    if not tasks:
        print("No task has results for both models yet.")
        return
    out = {"tasks_compared": tasks, "max_attempts": MAX_ATTEMPTS,
           "models": {m: summarise(runs, m, tasks) for m in MODELS},
           "cascades": {f"haiku_x{k}_then_opus": cascade(runs, tasks, k) for k in (1, 2, 3)}}
    h, o = out["models"]["claude/haiku-nothink"], out["models"]["claude/opus"]
    out["per_request_ratio_opus_over_haiku"] = o["first_try_cost_per_request"] / h["first_try_cost_per_request"]
    out["per_task_ratio_opus_over_haiku"] = o["cost_per_task"] / h["cost_per_task"]
    out["per_solved_task_ratio_opus_over_haiku"] = o["cost_per_solved_task"] / h["cost_per_solved_task"]
    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nTasks compared: {len(tasks)} (up to {MAX_ATTEMPTS} attempts each)\n")
    print("| Strategy | First-try pass | Solved | Avg calls | Cost per task | Cost per solved task | Time per task |")
    print("|---|---|---|---|---|---|---|")
    for m, label in (("claude/haiku-nothink", "Haiku alone"), ("claude/opus", "Opus alone")):
        s = out["models"][m]
        print(f"| {label} | {s['first_try_pass']:.0%} | {s['solved_within_4']:.0%} | {s['avg_attempts']:.2f} | "
              f"${s['cost_per_task']:.4f} | ${s['cost_per_solved_task']:.4f} | {s['time_per_task_s']:.0f} s |")
    for k, c in out["cascades"].items():
        print(f"| Haiku ×{k.split('_x')[1][0]}, then Opus | — | {c['solved']:.0%} | {c['avg_calls']:.2f} | "
              f"${c['cost_per_task']:.4f} | ${c['cost_per_solved_task']:.4f} | {c['time_per_task_s']:.0f} s |")
    print(f"\nOpus / Haiku cost: per request {out['per_request_ratio_opus_over_haiku']:.1f}×, "
          f"per task {out['per_task_ratio_opus_over_haiku']:.1f}×, per solved task {out['per_solved_task_ratio_opus_over_haiku']:.1f}×")
    print("\n| Task | Haiku attempts | Haiku solved | Opus attempts | Opus solved |\n|---|---|---|---|---|")
    for t in tasks:
        h_, o_ = runs[("claude/haiku-nothink", t)], runs[("claude/opus", t)]
        print(f"| {t} | {len(h_['attempts'])} | {'yes' if h_['solved'] else 'no'} | {len(o_['attempts'])} | {'yes' if o_['solved'] else 'no'} |")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-claude-calls", type=int, default=80)
    args = ap.parse_args()
    if args.check:
        for t in TASKS:
            for e, x in zip(t["cases"], expected(t)):
                print(f"{t['id']:9} {e}  ->  {x}")
        print(f"\n{len(TASKS)} tasks, {sum(len(t['cases']) for t in TASKS)} test cases")
        return
    if args.yes:
        live(args.max_claude_calls)
    elif not args.report:
        raise SystemExit("Uses Claude subscription quota. Add --yes (or --check / --report).")
    report()


if __name__ == "__main__":
    main()
