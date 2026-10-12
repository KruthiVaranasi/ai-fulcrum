"""Verify-then-escalate: a cheap check on the cheap model's ANSWER decides whether to escalate.

    python -m hybrid.verify --yes --max-claude-calls 80

Predicting from the prompt alone proved weak. Here the cheap answer already exists, so a cheap
verifier (Haiku, thinking off) reads the request and the answer and says whether it fully meets the
need. If not, the request is escalated to the frontier model. Coding items are checked with their unit
tests instead (no verifier call), mirroring a CI check.

The verifier never sees the frontier answer or the judge's verdict. Results go to
results/hybrid_verify.jsonl and are used by hybrid.decide.
"""

import argparse
import json
from pathlib import Path

from router.backends import ClaudeCodeBackend

from .grading import MAX_PROMPT_CHARS, _clip
from .run_models import load_answers, load_dataset

OUT = Path(__file__).parent.parent / "results" / "hybrid_verify.jsonl"
CHEAP = "claude/haiku-nothink"
BATCH = 5

INSTRUCTIONS = """You are a quick quality check. For each item, decide whether the ANSWER fully meets the
user's request: correct, responsive to what was asked, and not missing anything important. Minor style
issues do not matter. Be strict about factual or reasoning errors and about answers that dodge the request.

Return ONLY a JSON array of booleans, one per item in order: true = the answer meets the need.
"""


def load_verdicts():
    out = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["id"], r["cheap_model"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-claude-calls", type=int, default=80)
    args = ap.parse_args()
    if not args.yes:
        raise SystemExit("Uses Claude subscription quota. Add --yes.")

    data, answers, done = load_dataset(), load_answers(), load_verdicts()
    todo = [it for it in data if it["eval_type"] != "tests" and (CHEAP, it["id"]) in answers
            and (it["id"], CHEAP) not in done]
    print(f"{len(todo)} answers to verify (~{-(-len(todo) // BATCH)} calls)")

    verifier = ClaudeCodeBackend("haiku", thinking=False)
    calls = 0
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            if calls >= args.max_claude_calls:
                print("Call budget reached; re-run to continue.")
                break
            batch = todo[i:i + BATCH]
            body = "\n\n".join(
                f"=== ITEM {n} ===\nREQUEST:\n{_clip(it['prompt'], MAX_PROMPT_CHARS)}\n--- ANSWER ---\n"
                f"{_clip(answers[(CHEAP, it['id'])]['text'], 4000)}" for n, it in enumerate(batch, 1))
            r = verifier.ask(INSTRUCTIONS + "\n" + body)
            calls += 1
            if "error" in r:
                print(f"  error: {r['error'][:160]}")
                if any(w in r["error"].lower() for w in ("limit", "rate", "quota", "overloaded")):
                    print("Usage limit reported; stopping. Re-run later to resume.")
                    break
                continue
            t = r["text"]
            try:
                votes = json.loads(t[t.find("["): t.rfind("]") + 1])
                assert len(votes) == len(batch) and all(isinstance(v, bool) for v in votes)
            except (ValueError, AssertionError):
                print("  could not parse verifier output; will retry on next run")
                continue
            for it, ok in zip(batch, votes):
                f.write(json.dumps({"id": it["id"], "cheap_model": CHEAP, "verified_ok": ok,
                                    "verify_cost_usd": r["equiv_cost_usd"] / len(batch)}) + "\n")
            f.flush()
    print(f"Verifier calls this run: {calls}")


if __name__ == "__main__":
    main()
