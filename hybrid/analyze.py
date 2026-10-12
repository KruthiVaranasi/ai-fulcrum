"""Analyse the hybrid run: per-category results, routing experiments, judge reliability.

    python -m hybrid.analyze            # everything graded so far
    python -m hybrid.analyze --pilot    # pilot subset only
"""

import sys
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from router.evaluate import curve
from router.routers import CostAwareRouter, TfidfRouter

from .grade import CHEAP_MODELS, STRONG, load_grades
from .grading import routing_scores
from .run_models import load_answers, load_dataset, pilot_ids

RESULTS = Path(__file__).parent.parent / "results"
CATEGORY_ORDER = ["simple", "general_qa", "reasoning", "software", "aiml", "pm_explain", "pm_decide", "interview", "real_interview"]


def build_table(data, answers, grades, cheap_model):
    """One row per item that has a strong and cheap result: scores, costs, category, prompt."""
    rows = []
    for item in data:
        sa, ca = answers.get((STRONG, item["id"])), answers.get((cheap_model, item["id"]))
        if not sa or not ca:
            continue
        if item["eval_type"] == "judge":
            g = grades.get(("judge", item["id"], cheap_model, ""))
            if not g:
                continue
            s_cheap, s_strong = routing_scores(g)
            outcome = "tie" if s_cheap == s_strong else ("cheap_better" if s_cheap > s_strong else "strong_better")
        else:
            gs, gc = grades.get(("objective", item["id"], STRONG, "")), grades.get(("objective", item["id"], cheap_model, ""))
            if not gs or not gc:
                continue
            s_cheap, s_strong = float(gc["correct"]), float(gs["correct"])
            outcome = "tie" if s_cheap == s_strong else ("cheap_better" if s_cheap > s_strong else "strong_better")
        rows.append({"id": item["id"], "category": item["category"], "prompt": item["prompt"], "outcome": outcome,
                     "objective": item["eval_type"] != "judge", "s_cheap": s_cheap, "s_strong": s_strong,
                     "c_cheap": ca["equiv_cost_usd"], "c_strong": sa["equiv_cost_usd"]})
    return rows


def category_report(rows, cheap_model):
    print(f"\n=== {cheap_model} vs {STRONG}: is the cheap answer as good as Opus? ===")
    print(f"{'category':12} {'n':>4} {'as good as Opus':>16} {'Opus better':>12} {'cheap better':>13} {'Opus $/1k':>10} {'cheap $/1k':>11}")
    out = {}
    for cat in CATEGORY_ORDER + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        oc = Counter(r["outcome"] for r in rs)
        good = (oc["tie"] + oc["cheap_better"]) / len(rs)
        cs = np.mean([r["c_strong"] for r in rs]) * 1000
        cc = np.mean([r["c_cheap"] for r in rs]) * 1000
        out[cat] = {"n": len(rs), "good_enough": good, "strong_better": oc["strong_better"] / len(rs),
                    "cheap_better": oc["cheap_better"] / len(rs), "strong_cost_per_1k": cs, "cheap_cost_per_1k": cc}
        print(f"{cat:12} {len(rs):4} {good:16.0%} {oc['strong_better'] / len(rs):12.0%} "
              f"{oc['cheap_better'] / len(rs):13.0%} {cs:10.2f} {cc:11.2f}")
    return out


def cv_predict(make, texts, y, extra, groups=None, folds=5):
    """Out-of-fold scores: k-fold, or leave-one-group-out when groups are given."""
    texts, y, extra = list(texts), np.asarray(y), np.asarray(extra)
    p = np.zeros(len(texts))
    if groups is None:
        splits = KFold(folds, shuffle=True, random_state=0).split(texts)
    else:
        groups = np.asarray(groups)
        splits = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
    for tr, te in splits:
        if len(set(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        r = make()
        if isinstance(r, CostAwareRouter):
            r.fit([texts[i] for i in tr], y[tr], extra[tr])
        else:
            r.fit([texts[i] for i in tr], y[tr])
        p[te] = r.predict([texts[i] for i in te])
    return p


def category_rule_scores(rows, folds=5):
    """Route by task type: P(strong needed) = share of that category needing Opus in the other folds."""
    cats = np.array([r["category"] for r in rows])
    need = np.array([float(r["s_strong"] > r["s_cheap"]) for r in rows])
    p = np.zeros(len(rows))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(rows):
        overall = need[tr].mean()
        for i in te:
            same = tr[cats[tr] == cats[i]]
            p[i] = need[same].mean() if len(same) else overall
    return p


def routing_report(rows, cheap_model):
    if len(rows) < 40:
        print(f"\n(Routing experiments need more graded items; have {len(rows)}.)")
        return {}
    texts = [r["prompt"] for r in rows]
    s_c = np.array([r["s_cheap"] for r in rows]); s_s = np.array([r["s_strong"] for r in rows])
    c_c = np.array([r["c_cheap"] for r in rows]); c_s = np.array([r["c_strong"] for r in rows])
    y = (s_s > s_c).astype(int)
    extra = np.clip(c_s - c_c, 1e-7, None)
    cats = [r["category"] for r in rows]

    from router.policy import load_policy

    scores = {
        "Random": np.random.default_rng(0).random(len(rows)),
        "Oracle (upper bound)": s_s - s_c,
        "Transfer: RouterBench-trained policy": load_policy().router.predict(texts),
        "Category rule (route by task type)": category_rule_scores(rows),
        "Retrained classifier (5-fold CV)": cv_predict(TfidfRouter, texts, y, extra),
        "Retrained cost-aware (5-fold CV)": cv_predict(CostAwareRouter, texts, y, extra),
        "Retrained cost-aware (unseen category)": cv_predict(CostAwareRouter, texts, y, extra, groups=cats),
    }
    llm = RESULTS / "hybrid_llm_router.jsonl"
    if llm.exists():
        votes = {}
        for line in llm.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                votes[v["id"]] = v["needs_strong"]
        if all(r["id"] in votes for r in rows):
            scores["LLM-as-router (Haiku decides)"] = np.array([float(votes[r["id"]]) for r in rows])

    print(f"\n=== Routing {cheap_model} vs {STRONG} on {len(rows)} items ===")
    print(f"cheap only: quality {s_c.mean():.1%}, cost ${c_c.sum():.3f}   |   Opus only: quality {s_s.mean():.1%}, cost ${c_s.sum():.3f}")
    print(f"{'router':42} {'cost-APGR':>9} {'95%-of-Opus at':>15} {'saving':>7}")
    out, curves = {}, []
    for name, p in scores.items():
        cv = curve(name, p, s_c, s_s, c_c, c_s)
        curves.append(cv)
        sm = cv.summary
        out[name] = sm
        share, saving = sm["share_strong_at_95pct_of_strong_quality"], sm["cost_saving_at_95pct_of_strong_quality"]
        print(f"{name:42} {sm['cost_APGR']:9.2f} {'—' if share is None else f'{share:.0%} Opus':>15} "
              f"{'—' if saving is None else f'{saving:.0%}':>7}")
    plot_curves(curves, RESULTS / f"hybrid_routing_{cheap_model.split('/')[-1].replace(':', '-')}.png",
                f"Real-world prompts: {cheap_model.split('/')[-1]} vs Opus 5.5")
    return out


def plot_curves(curves, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for cv in curves:
        style = ":" if cv.name.startswith("Oracle") else "--" if cv.name == "Random" else "-"
        ax.plot(cv.cost / cv.c_strong * 100, cv.quality * 100, style, label=cv.name, lw=1.6)
    ax.axhline(curves[0].q_strong * 100, color="grey", lw=0.8)
    ax.set_xlabel("Cost (% of sending everything to Opus)")
    ax.set_ylabel("Quality (% of prompts with a best-or-tied answer)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability_report(grades):
    main = {(k[1], k[2]): v for k, v in grades.items() if k[0] == "judge" and k[3] == ""}
    out = {}
    for check, label in (("swap", "Position-swap agreement (same judge, answers reversed)"),
                         ("second_judge", "Cross-judge agreement (Sonnet vs Opus)")):
        pairs = [(main[(k[1], k[2])]["outcome"], v["outcome"]) for k, v in grades.items()
                 if k[0] == "judge" and k[3] == check and (k[1], k[2]) in main]
        if pairs:
            agree = np.mean([a == b for a, b in pairs])
            out[check] = {"n": len(pairs), "agreement": float(agree)}
            print(f"{label}: {agree:.0%} on {len(pairs)} pairs")
    judged = list(main.values())
    if judged:
        print(f"Judge verdicts so far: {dict(Counter(v['outcome'] for v in judged))}")
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):  # ₹ and ≈ crash the default Windows console encoding
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    data = load_dataset()
    if args.pilot:
        keep = pilot_ids(data)
        data = [r for r in data if r["id"] in keep]
    answers, grades = load_answers(), load_grades()

    report = {"categories": {}, "routing": {}}
    for cm in CHEAP_MODELS:
        rows = build_table(data, answers, grades, cm)
        if not rows:
            print(f"\nNo graded comparisons yet for {cm}.")
            continue
        report["categories"][cm] = category_report(rows, cm)
        report["routing"][cm] = routing_report(rows, cm)
    print()
    report["reliability"] = reliability_report(grades)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ("hybrid_pilot_report.json" if args.pilot else "hybrid_report.json")).write_text(
        json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
