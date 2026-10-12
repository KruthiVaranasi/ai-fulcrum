"""Grade cached answers: objective checks for benchmark items, a blind batched judge for the rest.

    python -m hybrid.grade --pilot                       # objective grading only (free)
    python -m hybrid.grade --pilot --judge --yes --max-claude-calls 30
    python -m hybrid.grade --judge --reliability --yes   # + position-swap and second-judge checks

Each judged item yields one comparison per cheap model: cheap answer vs the strong (Opus) answer.
Results are cached in results/hybrid_grades.jsonl; re-running only grades what's missing.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from router.backends import ClaudeCodeBackend

from .grading import build_judge_prompt, grade_objective, parse_judge_output, strong_is_a, to_cheap_strong
from .run_models import load_answers, load_dataset, pilot_ids

GRADES = Path(__file__).parent.parent / "results" / "hybrid_grades.jsonl"
STRONG = "claude/opus"
CHEAP_MODELS = ["claude/haiku-nothink", "ollama/qwen2.5:3b"]
BATCH = 4


def in_subset(key: str, frac: float, salt: str) -> bool:
    return int(hashlib.sha1(f"{salt}|{key}".encode()).hexdigest(), 16) % 1000 < frac * 1000


def load_grades():
    out = {}
    # Full log locally; the public subset (results/public/) when the full log isn't present
    src = GRADES if GRADES.exists() else GRADES.parent / "public" / GRADES.name
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["kind"], r["id"], r.get("model") or r.get("cheap_model"), r.get("check", ""))] = r
    return out


def grade_objective_items(data, answers, grades, f):
    n = 0
    for item in data:
        if item["eval_type"] == "judge":
            continue
        for (model, item_id), ans in answers.items():
            if item_id != item["id"] or ("objective", item_id, model, "") in grades:
                continue
            res = grade_objective(item, ans["text"])
            row = {"kind": "objective", "id": item_id, "model": model, **res}
            f.write(json.dumps(row) + "\n")
            grades[("objective", item_id, model, "")] = row
            n += 1
    return n


def judge_pairs(pairs, judge, check, f, grades, budget):
    """Judge pairs in batches; returns number of judge calls made."""
    calls = 0
    for i in range(0, len(pairs), BATCH):
        if calls >= budget:
            print(f"Judge call budget reached ({budget}). Re-run later to continue.")
            break
        batch = pairs[i:i + BATCH]
        r = judge.ask(build_judge_prompt(batch))
        calls += 1
        if "error" in r:
            print(f"  judge error: {r['error'][:200]}")
            err = r["error"].lower()
            if "limit" in err or "rate" in err or "quota" in err or "overloaded" in err:
                print("Usage limit or overload reported; stopping. Re-run later to resume.")
                return budget  # use up the budget so the caller stops too
            continue
        try:
            verdicts = parse_judge_output(r["text"], len(batch))
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  could not parse judge output ({e}); will retry on next run")
            continue
        for p, v in zip(batch, verdicts):
            res = to_cheap_strong(v, p["strong_first"])
            row = {"kind": "judge", "id": p["item"]["id"], "cheap_model": p["cheap_model"], "check": check,
                   "judge": judge.name, "strong_first": p["strong_first"], **res,
                   "judge_cost_usd": r["equiv_cost_usd"] / len(batch)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            grades[("judge", row["id"], row["cheap_model"], check)] = row
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--judge", action="store_true", help="run the LLM judge (uses subscription quota)")
    ap.add_argument("--reliability", action="store_true", help="also run position-swap (10%%) and second-judge (15%%) checks")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-claude-calls", type=int, default=40)
    args = ap.parse_args()

    data = load_dataset()
    if args.pilot:
        keep = pilot_ids(data)
        data = [r for r in data if r["id"] in keep]
    answers = load_answers()
    grades = load_grades()
    GRADES.parent.mkdir(exist_ok=True)

    with GRADES.open("a", encoding="utf-8") as f:
        n = grade_objective_items(data, answers, grades, f)
        print(f"Objective grades added: {n}")
        if not args.judge:
            return
        if not args.yes:
            raise SystemExit("Judging uses Claude subscription quota. Add --yes.")

        def pairs_for(check, frac=1.0, flip=False, salt=""):
            out = []
            for item in data:
                if item["eval_type"] != "judge":
                    continue
                strong = answers.get((STRONG, item["id"]))
                for cm in CHEAP_MODELS:
                    cheap = answers.get((cm, item["id"]))
                    if not strong or not cheap or ("judge", item["id"], cm, check) in grades:
                        continue
                    if frac < 1 and not in_subset(f"{item['id']}|{cm}", frac, salt):
                        continue
                    first = strong_is_a(item["id"], cm)
                    out.append({"item": item, "cheap_model": cm, "cheap_text": cheap["text"],
                                "strong_text": strong["text"], "strong_first": (not first) if flip else first})
            return out

        budget = args.max_claude_calls
        main_pairs = pairs_for("")
        print(f"Main judge: {len(main_pairs)} pairs to judge (~{-(-len(main_pairs) // BATCH)} calls)")
        budget -= judge_pairs(main_pairs, ClaudeCodeBackend("opus"), "", f, grades, budget)

        if args.reliability and budget > 0:
            swap = pairs_for("swap", 0.10, flip=True, salt="swap")
            print(f"Position-swap check: {len(swap)} pairs")
            budget -= judge_pairs(swap, ClaudeCodeBackend("opus"), "swap", f, grades, budget)
            second = pairs_for("second_judge", 0.15, salt="second")
            print(f"Second-judge (Sonnet) check: {len(second)} pairs")
            budget -= judge_pairs(second, ClaudeCodeBackend("sonnet"), "second_judge", f, grades, budget)
        print(f"Judge calls this run: {args.max_claude_calls - budget}")


if __name__ == "__main__":
    main()
