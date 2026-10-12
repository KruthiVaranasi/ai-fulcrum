"""Collect answers from each model for the hybrid dataset (cached; never asks the same thing twice).

    python -m hybrid.run_models --pilot --models ollama:qwen2.5:3b
    python -m hybrid.run_models --pilot --models claude:haiku-nothink,claude:opus --yes --max-claude-calls 80
    python -m hybrid.run_models --models claude:opus --yes --max-claude-calls 150      # full set, in sittings
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from live_free import backend
from router.backends import ClaudeCodeBackend

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
ANSWERS = HERE.parent / "results" / "hybrid_answers.jsonl"


REAL = HERE.parent / "data" / "real_prep.jsonl"  # private real-user slice (gitignored), if present


SESSIONS = HERE.parent / "data" / "session_requests.jsonl"  # your own Claude Code requests (gitignored)


def _session_rows():
    """Text and lookup requests from your Claude Code sessions (actions aren't comparable as text)."""
    if not SESSIONS.exists():
        return []
    rows = [json.loads(line) for line in SESSIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] in ("text", "lookup")]
    # Second variant of each request with ~3k tokens of recent conversation, to test the effect of context.
    longer = [{**r, "id": r["id"] + "-ctx", "category": "own_sessions_ctx", "prompt": r["prompt_long_context"]}
              for r in rows if r.get("prompt_long_context")]
    return rows + longer


def load_dataset(include_real: bool = True):
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if include_real and REAL.exists():
        rows += [json.loads(line) for line in REAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if include_real:
        rows += _session_rows()
    return rows


def pilot_ids(data, per_category: int = 5):
    """A fixed, evenly spread subset: per_category items from each category, mixing sources."""
    by_cat = defaultdict(list)
    for row in data:
        by_cat[row["category"]].append(row)
    ids = []
    for rows in by_cat.values():
        step = max(1, len(rows) // per_category)
        ids += [r["id"] for r in rows[::step][:per_category]]
    return set(ids)


def load_answers():
    out = {}
    # Full log locally; the public subset (results/public/) when the full log isn't present
    src = ANSWERS if ANSWERS.exists() else ANSWERS.parent / "public" / ANSWERS.name
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # two runs append to this file; skip a torn line
                continue
            out[(r["model"], r["id"])] = r
    # Opus already answered your session requests: use its real replies (and real token cost).
    for s in _session_rows():
        out.setdefault(("claude/opus", s["id"]), {
            "id": s["id"], "model": "claude/opus", "text": s["opus_reply"], "equiv_cost_usd": s["opus_equiv_cost_usd"],
            "input_tokens": s["opus_input_tokens"], "output_tokens": s["opus_output_tokens"],
            "latency_s": float("nan"), "from_transcript": True})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="e.g. ollama:qwen2.5:3b,claude:haiku-nothink,claude:opus")
    ap.add_argument("--pilot", action="store_true", help="only the 40-prompt pilot subset")
    ap.add_argument("--categories", help="comma-separated categories to include")
    ap.add_argument("--yes", action="store_true", help="allow Claude calls (uses subscription quota)")
    ap.add_argument("--max-claude-calls", type=int, default=100)
    args = ap.parse_args()

    data = load_dataset()
    if args.pilot:
        keep = pilot_ids(data)
        data = [r for r in data if r["id"] in keep]
    if args.categories:
        cats = set(args.categories.split(","))
        data = [r for r in data if r["category"] in cats]
    done = load_answers()
    ANSWERS.parent.mkdir(exist_ok=True)

    claude_calls = 0
    with ANSWERS.open("a", encoding="utf-8") as f:
        backends = []
        for spec in args.models.split(","):
            b = backend(spec.strip())
            if isinstance(b, ClaudeCodeBackend) and not args.yes:
                print(f"Skipping {b.name}: Claude calls use subscription quota. Add --yes.")
                continue
            backends.append(b)
        for b in backends:
            todo = sum((b.name, r["id"]) not in done for r in data)
            print(f"{b.name}: {todo} prompts to answer ({len(data) - todo} already cached)")

        # Go prompt by prompt across models, so a run cut short by the call cap still leaves
        # complete pairs to compare.
        new = {b.name: 0 for b in backends}
        broken = set()
        for row in data:
            needed = [b for b in backends if (b.name, row["id"]) not in done and b.name not in broken]
            n_claude = sum(isinstance(b, ClaudeCodeBackend) for b in needed)
            if n_claude and claude_calls + n_claude > args.max_claude_calls:
                print(f"Reached --max-claude-calls={args.max_claude_calls}. Re-run later to continue.")
                break
            for b in needed:
                r = b.ask(row["prompt"])
                claude_calls += isinstance(b, ClaudeCodeBackend) and not r.get("not_sent")
                if "error" in r:
                    print(f"  {b.name} {row['id']}: {r['error']}")
                    err = r["error"].lower()
                    if "unreachable" in err or "not found" in err:
                        broken.add(b.name)
                    if "limit" in err or "rate" in err or "quota" in err or "overloaded" in err:
                        print("Usage limit or overload reported; stopping this run. Re-run later to resume.")
                        broken.update(x.name for x in backends if isinstance(x, ClaudeCodeBackend))
                    continue
                f.write(json.dumps({"id": row["id"], "model": b.name, **r}, ensure_ascii=False) + "\n")
                f.flush()
                new[b.name] += 1
        print("Answered this run:", new)
    print(f"Claude calls this run: {claude_calls}")


if __name__ == "__main__":
    main()
