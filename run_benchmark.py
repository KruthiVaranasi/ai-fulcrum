"""Offline router benchmark on RouterBench (no API calls, no spend).

    python run_benchmark.py              # TF-IDF and baselines only (fast)
    python run_benchmark.py --embeddings # also embedding routers (first run embeds ~36k prompts on CPU)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from router.data import CHEAP, STRONG, load, needs_strong, random_split, task_split
from router.evaluate import curve
from router.routers import (
    Embedder,
    EmbeddingKNNRouter,
    EmbeddingLRRouter,
    LengthRouter,
    RandomRouter,
    TfidfRouter,
    CostAwareRouter,
)

RESULTS = Path(__file__).parent / "results"


def run_split(label, bench, train, test, emb, use_embeddings, cheap=CHEAP):
    y = needs_strong(bench, cheap=cheap)
    test_b = bench.subset(test)
    args = (test_b.score(cheap), test_b.score(STRONG), test_b.cost(cheap), test_b.cost(STRONG))
    texts_tr = [bench.prompts[i] for i in train]
    texts_te = test_b.prompts

    curves = []
    # Upper bound: a router that knows the answers in advance.
    curves.append(curve("Oracle (upper bound)", args[1] - args[0], *args))
    routers = [RandomRouter(), LengthRouter(), TfidfRouter(), CostAwareRouter()]
    if use_embeddings:
        routers += [EmbeddingLRRouter(), EmbeddingKNNRouter()]
    for r in routers:
        t0 = time.time()
        if getattr(r, "uses_cost", False):
            extra = bench.cost(STRONG) - bench.cost(cheap)
            r.fit(texts_tr, y[train], extra[train])
            t1 = time.time()
            p = r.predict(texts_te)
        elif getattr(r, "uses_embeddings", False):
            r.fit(emb[train], y[train])
            t1 = time.time()
            p = r.predict(emb[test])
        else:
            r.fit(texts_tr, y[train])
            t1 = time.time()
            p = r.predict(texts_te)
        t2 = time.time()
        cv = curve(r.name, p, *args)
        cv.summary["train_seconds"] = round(t1 - t0, 2)
        cv.summary["route_ms_per_prompt"] = round((t2 - t1) / len(test) * 1000, 4)
        curves.append(cv)

    print(f"\n=== {label}: train {len(train):,} / test {len(test):,} prompts ===")
    print(f"Cheap ({cheap}) quality {curves[0].q_cheap:.3f}, cost ${curves[0].c_cheap:.2f}")
    print(f"Strong ({STRONG}) quality {curves[0].q_strong:.3f}, cost ${curves[0].c_strong:.2f}")
    hdr = f"{'Router':36} {'APGR':>6} {'costAPGR':>8} {'50% gap':>8} {'80% gap':>8} {'95%-of-GPT-4 at':>16} {'saving':>7}"
    print(hdr)
    for cv in curves:
        s = cv.summary
        fmt = lambda v: f"{v:.0%}" if v is not None else "—"
        print(
            f"{cv.name:36} {s['APGR']:6.3f} {s['cost_APGR']:8.3f} {fmt(s['share_strong_for_50pct_gap']):>8} "
            f"{fmt(s['share_strong_for_80pct_gap']):>8} {fmt(s['share_strong_at_95pct_of_strong_quality']):>16} "
            f"{fmt(s['cost_saving_at_95pct_of_strong_quality']):>7}"
        )
    return curves


def plot(curves, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    styles = {"Oracle (upper bound)": ":", "Random": "--"}
    for cv in curves:
        ax.plot(cv.cost / cv.c_strong * 100, cv.quality * 100, styles.get(cv.name, "-"), label=cv.name, lw=1.8)
    c0 = curves[0]
    ax.axhline(c0.q_strong * 100, color="grey", lw=0.8)
    ax.text(101, c0.q_strong * 100, "GPT-4 only", va="center", fontsize=8, color="grey")
    ax.set_xlabel("Cost (% of sending everything to GPT-4)")
    ax.set_ylabel("Average quality (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--cheap", default=CHEAP, help="cheap-tier model column in RouterBench")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    bench = load()
    print(f"Loaded {len(bench):,} prompts, {len(bench.models)} models")
    emb = Embedder().encode(bench.prompts, cache_name="minilm_all") if args.embeddings else None

    out = {}
    tr, te = random_split(len(bench))
    c_in = run_split("In-distribution (random split)", bench, tr, te, emb, args.embeddings, args.cheap)
    tag = args.cheap.split("/")[-1]
    plot(c_in, RESULTS / f"curve_{tag}_in_distribution.png", f"{tag} vs GPT-4: in-distribution test set")
    out["in_distribution"] = {cv.name: cv.summary for cv in c_in}

    tr, te, held = task_split(bench.tasks)
    c_ood = run_split(f"Out-of-distribution ({len(held)} tasks held out)", bench, tr, te, emb, args.embeddings, args.cheap)
    plot(c_ood, RESULTS / f"curve_{tag}_out_of_distribution.png", f"{tag} vs GPT-4: unseen task types")
    out["out_of_distribution"] = {cv.name: cv.summary for cv in c_ood}
    out["out_of_distribution_tasks"] = held

    (RESULTS / f"offline_results_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved results and charts to {RESULTS}")


if __name__ == "__main__":
    main()
