"""Live test on today's Claude models, designed to keep spend tiny and visible.

Step 1, one call, prints exact tokens and cost, plus a projection:
    python live_check.py --one

Step 2, after you choose a sample size (refuses to spend without --yes, stops at --max-usd):
    python live_check.py --models haiku,sonnet,opus --n 30 --yes --max-usd 1.00

Uses RouterBench multiple-choice prompts (MMLU, ARC, HellaSwag, Winogrande) so answers can be
graded automatically. The reference answer comes from RouterBench models that scored 1.0.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from router.data import load
from router.live import ALIASES, PRICES, ask, extract_choice, load_api_key

RESULTS = Path(__file__).parent / "results"
MC_PREFIXES = ("mmlu-", "arc-challenge", "hellaswag", "winogrande")


def mc_sample(n: int, seed: int = 7):
    """Stratified sample of multiple-choice prompts with a known correct letter."""
    bench = load()
    rng = np.random.default_rng(seed)
    pool = []
    for i, task in enumerate(bench.tasks):
        if not task.startswith(MC_PREFIXES):
            continue
        pool.append(i)
    rng.shuffle(pool)
    import pandas as pd

    raw = pd.read_pickle(Path(__file__).parent / "data" / "routerbench_0shot.pkl")
    out = []
    for i in pool:
        gold = None
        for m in bench.models:
            if bench.scores[i, bench.col(m)] == 1.0:
                gold = extract_choice(str(raw.iloc[i][m + "|model_response"]))
                if gold:
                    break
        if gold:
            out.append({"idx": int(i), "task": str(bench.tasks[i]), "prompt": bench.prompts[i], "gold": gold})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="make exactly one Haiku call and project costs")
    ap.add_argument("--models", default="haiku")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--effort", default="low", help="effort for Sonnet/Opus: low | medium | high")
    ap.add_argument("--max-usd", type=float, default=1.0, help="hard stop once this much is spent")
    ap.add_argument("--yes", action="store_true", help="confirm you want to spend money")
    args = ap.parse_args()

    import anthropic

    client = anthropic.Anthropic(api_key=load_api_key())

    if args.one:
        item = mc_sample(1)[0]
        r = ask(client, "claude-haiku-4-5", item["prompt"])
        if "error" in r:
            raise SystemExit(r["error"])
        pred = extract_choice(r["text"])
        print(f"Task: {item['task']}  |  answer {pred!r}, correct {item['gold']!r}  ->  {'RIGHT' if pred == item['gold'] else 'WRONG'}")
        print(f"Tokens: {r['input_tokens']} in / {r['output_tokens']} out   Cost: ${r['cost']:.6f}   Latency: {r['latency_s']}s")
        print("\nProjection for 30 prompts (Haiku measured; Sonnet/Opus also bill thinking tokens, so a range):")
        for name, model in ALIASES.items():
            pin, pout = PRICES[model]
            low = 30 * (r["input_tokens"] * pin + r["output_tokens"] * pout) / 1e6
            high = 30 * (r["input_tokens"] * pin + (r["output_tokens"] + 600) * pout) / 1e6 if name != "haiku" else low
            print(f"  {name:7} ${low:.4f}" + (f" to ${high:.4f}" if high != low else ""))
        return

    models = [ALIASES.get(m.strip(), m.strip()) for m in args.models.split(",")]
    items = mc_sample(args.n)
    if not args.yes:
        raise SystemExit(f"Would call {len(models)} model(s) on {len(items)} prompts. Re-run with --yes to spend (cap ${args.max_usd}).")

    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / "live_results.jsonl"
    spent = 0.0
    with log.open("a", encoding="utf-8") as f:
        for model in models:
            right = 0
            for k, item in enumerate(items, 1):
                if spent >= args.max_usd:
                    print(f"Budget cap ${args.max_usd} reached; stopping.")
                    return
                r = ask(client, model, item["prompt"], effort=args.effort)
                if "error" in r:
                    print(f"  {model} #{k}: {r['error']}")
                    continue
                spent += r["cost"]
                pred = extract_choice(r["text"])
                ok = pred == item["gold"]
                right += ok
                f.write(json.dumps({**{k2: item[k2] for k2 in ("idx", "task", "gold")}, "model": model,
                                    "effort": args.effort if model != "claude-haiku-4-5" else None,
                                    "pred": pred, "correct": ok, **{k2: r[k2] for k2 in
                                    ("input_tokens", "output_tokens", "cost", "latency_s", "stop_reason")}}) + "\n")
            print(f"{model:20} accuracy {right}/{len(items)}   running spend ${spent:.4f}")
    print(f"Total spent: ${spent:.4f}. Log: {log}")


if __name__ == "__main__":
    main()
