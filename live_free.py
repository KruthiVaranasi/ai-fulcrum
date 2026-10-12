"""Free live test: a local open-source model vs Claude (via the Claude subscription), then routing.

    python live_free.py --n 5 --models ollama:qwen2.5:3b              # local only, free
    python live_free.py --n 5 --models claude:haiku,claude:opus --yes  # uses Claude subscription quota
    python live_free.py --analyze --cheap ollama/qwen2.5:3b --strong claude/opus

Every answer is cached in results/live_free.jsonl, so re-running never repeats a call.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from live_check import mc_sample
from router.backends import ClaudeCodeBackend, OllamaBackend
from router.data import STRONG, load
from router.evaluate import curve
from router.live import extract_choice
from router.routers import CostAwareRouter

LOG = Path(__file__).parent / "results" / "live_free.jsonl"


def backend(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaBackend(rest)
    if kind == "claude":
        nothink = rest.endswith("-nothink")
        alias, _, effort = rest.removesuffix("-nothink").partition("@")
        return ClaudeCodeBackend(alias, effort or None, thinking=not nothink)
    raise SystemExit(f"Unknown model spec {spec!r}; use ollama:<model> or claude:<haiku|sonnet|opus>[@effort][-nothink]")


def load_log():
    rows = []
    if LOG.exists():
        rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def run(args):
    items = mc_sample(args.n)
    done = {(r["model"], r["idx"]) for r in load_log()}
    LOG.parent.mkdir(exist_ok=True)
    claude_calls = 0
    with LOG.open("a", encoding="utf-8") as f:
        for spec in args.models.split(","):
            b = backend(spec.strip())
            is_claude = isinstance(b, ClaudeCodeBackend)
            if is_claude and not args.yes:
                print(f"Skipping {b.name}: Claude calls use your subscription quota. Add --yes to run them.")
                continue
            right = total = 0
            for item in items:
                if (b.name, item["idx"]) in done:
                    continue
                if is_claude and claude_calls >= args.max_claude_calls:
                    print(f"Reached --max-claude-calls={args.max_claude_calls}; stopping Claude calls.")
                    break
                r = b.ask(item["prompt"])
                claude_calls += is_claude and not r.get("not_sent")
                if "error" in r:
                    print(f"  {b.name} idx={item['idx']}: {r['error']}")
                    if "unreachable" in r["error"] or "cli" in r["error"]:
                        break
                    continue
                pred = extract_choice(r["text"])
                ok = pred == item["gold"]
                right += ok
                total += 1
                f.write(json.dumps({"model": b.name, "idx": item["idx"], "task": item["task"], "gold": item["gold"],
                                    "pred": pred, "correct": ok, **{k: v for k, v in r.items() if k != "text"}}) + "\n")
                f.flush()
            if total:
                print(f"{b.name:28} new answers {total:3}   accuracy {right}/{total}")
    print(f"Claude calls this run: {claude_calls}")


LABELS = {
    "ollama/qwen2.5:3b": "Qwen 2.5 3B\n(local, free)",
    "claude/haiku-nothink": "Haiku 4.5\nthinking off",
    "claude/haiku": "Haiku 4.5\nthinking on",
    "claude/opus": "Opus 5.5",
}


def plot_models(by_model, path=LOG.parent / "live_models.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for m, d in by_model.items():
        acc = np.mean([r["correct"] for r in d.values()]) * 100
        cost = np.mean([r["equiv_cost_usd"] for r in d.values()]) * 1000
        ax.scatter(cost, acc, s=90, zorder=3)
        ax.annotate(LABELS.get(m, m), (cost, acc), textcoords="offset points", xytext=(8, -4), fontsize=8.5)
    ax.set_xlabel("Equivalent API cost per 1,000 prompts (USD, list prices)")
    ax.set_ylabel(f"Accuracy on live prompts (%)")
    ax.set_title("Price per token isn't price per task", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(args):
    rows = load_log()
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["idx"]] = r
    print(f"{'Model':28} {'n':>4} {'accuracy':>9} {'avg latency':>12} {'equiv API cost':>15}")
    for m, d in sorted(by_model.items()):
        acc = np.mean([r["correct"] for r in d.values()])
        lat = np.mean([r["latency_s"] for r in d.values()])
        cost = sum(r["equiv_cost_usd"] for r in d.values())
        print(f"{m:28} {len(d):4} {acc:9.1%} {lat:11.1f}s {'$%.4f' % cost:>15}")

    plot_models(by_model)
    if args.cheap not in by_model or args.strong not in by_model:
        return
    common = sorted(set(by_model[args.cheap]) & set(by_model[args.strong]))
    if len(common) < 5:
        print("Need at least 5 prompts answered by both models to evaluate routing.")
        return

    # Train the cost-aware router on RouterBench, excluding the live prompts to avoid leakage.
    bench = load()
    keep = np.setdiff1d(np.arange(len(bench)), common)
    train = bench.subset(keep)
    cheap_proxy = "zero-one-ai/Yi-34B-Chat"
    y = (train.score(STRONG) > train.score(cheap_proxy)).astype(int)
    router = CostAwareRouter().fit(train.prompts, y, train.cost(STRONG) - train.cost(cheap_proxy))
    p = router.predict([bench.prompts[i] for i in common])

    c, s = by_model[args.cheap], by_model[args.strong]
    s_cheap = np.array([float(c[i]["correct"]) for i in common])
    s_strong = np.array([float(s[i]["correct"]) for i in common])
    c_cheap = np.array([c[i]["equiv_cost_usd"] for i in common])
    c_strong = np.array([s[i]["equiv_cost_usd"] for i in common])
    cv = curve("Cost-aware router (trained offline)", p, s_cheap, s_strong, c_cheap, c_strong)
    oracle = curve("Oracle", s_strong - s_cheap, s_cheap, s_strong, c_cheap, c_strong)
    rand = curve("Random", np.random.default_rng(0).random(len(common)), s_cheap, s_strong, c_cheap, c_strong)

    print(f"\nRouting {args.cheap} (cheap) vs {args.strong} (strong) on {len(common)} live prompts")
    print(f"  cheap only : accuracy {s_cheap.mean():.1%}, equiv cost ${c_cheap.sum():.4f}")
    print(f"  strong only: accuracy {s_strong.mean():.1%}, equiv cost ${c_strong.sum():.4f}")
    for k in (cv, rand, oracle):
        sm = k.summary
        at95 = sm["share_strong_at_95pct_of_strong_quality"]
        sv = sm["cost_saving_at_95pct_of_strong_quality"]
        print(f"  {k.name:38} cost-APGR {sm['cost_APGR']:.2f}   95% of strong quality at "
              f"{'—' if at95 is None else f'{at95:.0%}'} strong calls, saving {'—' if sv is None else f'{sv:.0%}'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--models", default="ollama:qwen2.5:3b")
    ap.add_argument("--yes", action="store_true", help="allow Claude calls (uses subscription quota)")
    ap.add_argument("--max-claude-calls", type=int, default=40)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--cheap", default="ollama/qwen2.5:3b")
    ap.add_argument("--strong", default="claude/opus")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
