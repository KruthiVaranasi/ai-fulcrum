"""LLM-as-router baseline: a cheap model reads each prompt and decides if it needs a frontier model.

    python -m hybrid.llm_router --yes --max-claude-calls 40      # 10 prompts per call, cached

Results go to results/hybrid_llm_router.jsonl and are picked up by hybrid.analyze.
"""

import argparse
import json
from pathlib import Path

from router.backends import ClaudeCodeBackend

from .run_models import load_dataset, pilot_ids

OUT = Path(__file__).parent.parent / "results" / "hybrid_llm_router.jsonl"
BATCH = 10

INSTRUCTIONS = """You are a routing assistant. For each user request below, decide whether a small, fast, cheap
AI model would answer it as well as a frontier model, or whether it needs the frontier model
(hard reasoning, deep expertise, nuanced judgement, long multi-part answers).

Return ONLY a JSON array of booleans, one per request in order: true = needs the frontier model.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-claude-calls", type=int, default=40)
    args = ap.parse_args()
    if not args.yes:
        raise SystemExit("Uses Claude subscription quota. Add --yes.")

    data = load_dataset()
    if args.pilot:
        keep = pilot_ids(data)
        data = [r for r in data if r["id"] in keep]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in data if r["id"] not in done]
    print(f"{len(todo)} prompts to classify (~{-(-len(todo) // BATCH)} calls)")

    router = ClaudeCodeBackend("haiku", thinking=False)
    calls = 0
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            if calls >= args.max_claude_calls:
                print("Call budget reached; re-run to continue.")
                break
            batch = todo[i:i + BATCH]
            body = "\n\n".join(f"### REQUEST {n}\n{r['prompt'][:2000]}" for n, r in enumerate(batch, 1))
            r = router.ask(INSTRUCTIONS + "\n" + body)
            calls += 1
            if "error" in r:
                print(f"  error: {r['error']}")
                continue
            text = r["text"]
            try:
                votes = json.loads(text[text.find("["): text.rfind("]") + 1])
                assert len(votes) == len(batch) and all(isinstance(v, bool) for v in votes)
            except (ValueError, AssertionError):
                print("  could not parse router output; will retry on next run")
                continue
            for row, v in zip(batch, votes):
                f.write(json.dumps({"id": row["id"], "needs_strong": v, "cost_usd": r["equiv_cost_usd"] / len(batch)}) + "\n")
            f.flush()
    print(f"Router calls this run: {calls}")


if __name__ == "__main__":
    main()
