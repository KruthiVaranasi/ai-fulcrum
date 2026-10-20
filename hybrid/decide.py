"""Turn measured model behaviour into a routing decision per business use case.

    python -m hybrid.decide                 # writes results/policy_*.json/png and docs/DECISIONS.md
    python -m hybrid.decide --cheap ollama/qwen2.5:3b

Decision rule (per prompt): send it to the frontier model when

    P(cheap answer is bad) x cost of a bad answer  >=  extra cost of the frontier answer

so the escalation threshold is  extra cost / cost of a bad answer : it moves with the business stakes.
If the resulting quality is below the use case's floor, the effective cost of a bad answer is raised
until the floor is met (or reported as unreachable with these models).

P(cheap answer is bad) and the extra cost are predicted from the prompt alone, out-of-fold, so every
number here comes from prompts the predictors did not see.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from .grade import STRONG, load_grades
from .policies import USE_CASES
from .run_models import load_answers, load_dataset

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
INR_PER_USD = 88.0


def verifier_single_cost_inr():
    """Mean cost of checking ONE answer per call (no batching), or None if not measured."""
    path = RESULTS / "overhead.json"
    if path.exists():
        v = json.loads(path.read_text(encoding="utf-8")).get("verifier_single")
        if v:
            return v["cost_usd"]["mean"] * INR_PER_USD
    return None


def verifier_latency_s():
    """Median latency of one single-request verifier call, or None if hybrid.overhead has not measured it."""
    path = RESULTS / "overhead.json"
    if path.exists():
        v = json.loads(path.read_text(encoding="utf-8")).get("verifier_single")
        if v:
            return v["latency_s"]["median"]
    return None


def item_rows(cheap_model, include_real=True):
    """Per item: good/bad for cheap and strong under both quality bars, and costs in ₹."""
    data, answers, grades = load_dataset(include_real), load_answers(), load_grades()
    rows = []
    for it in data:
        sa, ca = answers.get((STRONG, it["id"])), answers.get((cheap_model, it["id"]))
        if not sa or not ca:
            continue
        if it["eval_type"] == "judge":
            g = grades.get(("judge", it["id"], cheap_model, ""))
            if not g:
                continue
            acc = (g["cheap_meets_need"], g["strong_meets_need"])
            best = (g["cheap_meets_need"] and g["outcome"] != "strong_better",
                    g["strong_meets_need"] and g["outcome"] != "cheap_better")
        else:
            gs, gc = grades.get(("objective", it["id"], STRONG, "")), grades.get(("objective", it["id"], cheap_model, ""))
            if not gs or not gc:
                continue
            acc = best = (gc["correct"], gs["correct"])
        rows.append({
            "id": it["id"], "category": it["category"], "prompt": it["prompt"], "eval_type": it["eval_type"],
            "good": {"acceptable": tuple(map(bool, acc)), "best": tuple(map(bool, best))},
            "c_cheap": ca["equiv_cost_usd"] * INR_PER_USD, "c_strong": sa["equiv_cost_usd"] * INR_PER_USD,
            "l_cheap": ca["latency_s"], "l_strong": sa["latency_s"],
        })
    # Verify-then-escalate inputs: unit tests for code (free, like CI); a Haiku verifier otherwise.
    verdicts = {}
    vpath = RESULTS / "hybrid_verify.jsonl"
    if not vpath.exists():
        vpath = RESULTS / "public" / "hybrid_verify.jsonl"
    if vpath.exists():
        for line in vpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                verdicts[(v["id"], v["cheap_model"])] = v
    # The verifier ran in batches of 5, so its per-request time comes from single-request calls
    # measured by hybrid.overhead (median); unit tests run locally and add no model time.
    verify_s = verifier_latency_s()
    for r in rows:
        if r["eval_type"] == "tests":
            r["verified_ok"], r["verify_cost"], r["verify_latency"] = r["good"]["acceptable"][0], 0.0, 0.0
        elif (r["id"], cheap_model) in verdicts:
            v = verdicts[(r["id"], cheap_model)]
            r["verified_ok"], r["verify_cost"] = v["verified_ok"], v["verify_cost_usd"] * INR_PER_USD
            r["verify_latency"] = verify_s or 0.0
        else:
            r["verified_ok"], r["verify_cost"], r["verify_latency"] = None, 0.0, 0.0
    return rows


def out_of_fold_predictions(rows, bar, folds=5):
    """P(cheap is bad) and predicted extra cost (₹), each predicted without seeing that prompt."""
    texts = [r["prompt"] for r in rows]
    y = np.array([0 if r["good"][bar][0] else 1 for r in rows])
    extra = np.array([max(r["c_strong"] - r["c_cheap"], 1e-6) for r in rows])
    p, dc = np.zeros(len(rows)), np.zeros(len(rows))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(texts):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=20_000)
        Xtr, Xte = vec.fit_transform([texts[i] for i in tr]), vec.transform([texts[i] for i in te])
        # Calibrated probabilities matter here: the rule multiplies P(bad) by a ₹ amount, so an
        # over- or under-confident model moves the threshold. (Class-balanced weights would inflate P(bad).)
        if len(set(y[tr])) > 1 and min(np.bincount(y[tr])) >= 3:
            clf = CalibratedClassifierCV(LogisticRegression(C=2.0, max_iter=2000), method="sigmoid", cv=3)
            p[te] = clf.fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        else:
            p[te] = y[tr].mean()
        dc[te] = np.exp(Ridge(alpha=1.0).fit(Xtr, np.log(extra[tr])).predict(Xte))
    return p, dc


def evaluate(rows, idx, to_strong, bar):
    """Realised share of good answers and ₹ cost per 1,000 requests for a routing decision."""
    good = np.array([rows[i]["good"][bar][1] if s else rows[i]["good"][bar][0] for i, s in zip(idx, to_strong)])
    cost = np.array([rows[i]["c_strong"] if s else rows[i]["c_cheap"] for i, s in zip(idx, to_strong)])
    return good.mean(), cost.mean() * 1000, good


def bootstrap_ci(good, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(good, len(good)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def decide_use_case(uc, rows, preds, min_items=10):
    idx = [i for i, r in enumerate(rows) if r["category"] in uc.categories]
    if len(idx) < min_items:
        return None
    p, dc = preds[uc.quality_bar]
    bar = uc.quality_bar
    n = len(idx)

    def p95_latency(to_strong):
        return float(np.percentile([rows[i]["l_strong"] if s else rows[i]["l_cheap"] for i, s in zip(idx, to_strong)], 95))

    def policy(e_cost):
        to_strong = p[idx] * e_cost >= dc[idx]
        q, c, good = evaluate(rows, idx, to_strong, bar)
        return {"share_strong": float(to_strong.mean()), "quality": float(q), "cost_per_1k": float(c),
                "expected_total_per_1k": float(c + (1 - q) * e_cost * 1000), "p95_latency_s": p95_latency(to_strong),
                "good": good}

    cheap_q, cheap_c, _ = evaluate(rows, idx, np.zeros(n, bool), bar)
    strong_q, strong_c, _ = evaluate(rows, idx, np.ones(n, bool), bar)

    # Business-cost optimum, then enforce the quality floor by raising the effective error cost.
    e_eff, chosen = uc.error_cost_inr, policy(uc.error_cost_inr)
    while chosen["quality"] < uc.quality_floor and e_eff < 1e7:
        e_eff *= 2
        chosen = policy(e_eff)
    floor_met = chosen["quality"] >= uc.quality_floor

    out = {
        "use_case": uc.name, "key": uc.key, "categories": uc.categories, "n": n, "quality_bar": bar,
        "error_cost_inr": uc.error_cost_inr, "quality_floor": uc.quality_floor, "monthly_volume": uc.monthly_volume,
        "rationale": uc.rationale,
        "always_cheap": {"quality": float(cheap_q), "cost_per_1k": float(cheap_c), "p95_latency_s": p95_latency(np.zeros(n, bool))},
        "always_strong": {"quality": float(strong_q), "cost_per_1k": float(strong_c), "p95_latency_s": p95_latency(np.ones(n, bool))},
        "latency_budget_p95_s": uc.p95_latency_budget_s,
        "policy": {k: v for k, v in chosen.items() if k != "good"},
        "policy_quality_ci95": bootstrap_ci(chosen["good"]),
        "effective_error_cost_inr": float(e_eff), "floor_met": bool(floor_met),
        "floor_reachable_with_frontier": bool(strong_q >= uc.quality_floor),
    }
    out["monthly_cost_inr"] = {
        "always_strong": strong_c / 1000 * uc.monthly_volume,
        "policy": chosen["cost_per_1k"] / 1000 * uc.monthly_volume,
        "always_cheap": cheap_c / 1000 * uc.monthly_volume,
    }
    out["latency_ok"] = bool(chosen["p95_latency_s"] <= uc.p95_latency_budget_s)
    out["saving_vs_always_strong"] = float(1 - chosen["cost_per_1k"] / strong_c) if strong_c else 0.0

    # Sensitivity: how the routing changes as the cost of a bad answer changes.
    grid = np.logspace(-1, 5, 61)
    sens = [policy(e) for e in grid]
    out["sensitivity"] = [{"error_cost_inr": float(e), "share_strong": s["share_strong"], "quality": s["quality"],
                           "cost_per_1k": s["cost_per_1k"]} for e, s in zip(grid, sens)]
    flip = next((float(e) for e, s in zip(grid, sens) if s["share_strong"] >= 0.5), None)
    out["error_cost_where_majority_goes_frontier"] = flip

    # Verify-then-escalate cascade (only where answers can be checked automatically).
    if uc.allow_cascade:
        tidx = [i for i in idx if rows[i]["eval_type"] == "tests"]
        if tidx:
            good = np.array([rows[i]["good"][bar][0] or rows[i]["good"][bar][1] for i in tidx])
            cost = np.array([rows[i]["c_cheap"] + (0 if rows[i]["good"][bar][0] else rows[i]["c_strong"]) for i in tidx])
            s_cost = np.mean([rows[i]["c_strong"] for i in tidx]) * 1000
            out["cascade_tested_items"] = {"n": len(tidx), "quality": float(good.mean()), "cost_per_1k": float(cost.mean() * 1000),
                                           "always_strong_cost_per_1k": float(s_cost),
                                           "always_strong_quality": float(np.mean([rows[i]["good"][bar][1] for i in tidx]))}

    # Verify-then-escalate for every item: the cheap model answers, a cheap check (unit tests for code,
    # a Haiku verifier otherwise) reads the ANSWER, and only answers that fail the check are escalated.
    vidx = [i for i in idx if rows[i].get("verified_ok") is not None]
    if len(vidx) >= 0.9 * n:
        esc = np.array([not rows[i]["verified_ok"] for i in vidx])
        good = np.array([rows[i]["good"][bar][1] if e else rows[i]["good"][bar][0] for i, e in zip(vidx, esc)])
        cost = np.array([rows[i]["c_cheap"] + rows[i]["verify_cost"] + (rows[i]["c_strong"] if e else 0) for i, e in zip(vidx, esc)])
        lat = np.array([rows[i]["l_cheap"] + rows[i]["verify_latency"] + (rows[i]["l_strong"] if e else 0) for i, e in zip(vidx, esc)])
        cheap_bad = np.array([not rows[i]["good"][bar][0] for i in vidx])
        out["verify_cascade"] = {
            "n": len(vidx), "share_strong": float(esc.mean()), "quality": float(good.mean()),
            "cost_per_1k": float(cost.mean() * 1000), "p95_latency_s": float(np.percentile(lat, 95)),
            "quality_ci95": bootstrap_ci(good),
            # The verifier ran 5 answers per call; an interactive product checks one at a time.
            "cost_per_1k_unbatched_check": (float(np.mean([rows[i]["c_cheap"] + (verifier_single_cost_inr() if rows[i]["verify_cost"] else 0)
                                                           + (rows[i]["c_strong"] if e else 0) for i, e in zip(vidx, esc)]) * 1000)
                                            if verifier_single_cost_inr() else None),
            # How good is the check itself? catch rate = bad cheap answers it escalated;
            # false alarms = escalations of answers that were actually fine.
            "verifier_catch_rate": float(esc[cheap_bad].mean()) if cheap_bad.any() else None,
            "verifier_false_alarm_rate": float(esc[~cheap_bad].mean()) if (~cheap_bad).any() else None,
        }

    out["recommendation"] = recommend(out, uc.error_cost_inr, uc.quality_floor)
    # Robustness: does the recommendation survive reasonable changes to the business assumptions?
    variants = {"cost of a bad answer ×0.5": (uc.error_cost_inr * 0.5, uc.quality_floor),
                "cost of a bad answer ×2": (uc.error_cost_inr * 2, uc.quality_floor),
                "quality floor −5 pts": (uc.error_cost_inr, max(0.0, uc.quality_floor - 0.05)),
                "quality floor +5 pts": (uc.error_cost_inr, min(1.0, uc.quality_floor + 0.05))}
    changes = {name: recommend(out, e, f)["strategy"] for name, (e, f) in variants.items()}
    out["robustness"] = {"variants": changes,
                         "stable": all(s == out["recommendation"]["strategy"] for s in changes.values())}
    return out


STRATEGY_NAMES = {"always_cheap": "Always the cheap model", "always_strong": "Always the frontier model",
                  "policy": "Predict from the prompt, escalate risky ones", "verify_cascade": "Answer cheaply, verify, escalate failures"}


def recommend(d, error_cost, floor):
    """Cheapest expected total cost (API cost + bad answers × their cost) among strategies meeting the floor."""
    options = {k: d[k] for k in ("always_cheap", "always_strong", "policy", "verify_cascade") if k in d}
    scored = {k: v["cost_per_1k"] + (1 - v["quality"]) * error_cost * 1000 for k, v in options.items()}
    ok = [k for k, v in options.items() if v["quality"] >= floor]
    pick = min(ok, key=scored.get) if ok else max(options, key=lambda k: options[k]["quality"])
    return {"strategy": pick, "label": STRATEGY_NAMES[pick], "quality": options[pick]["quality"],
            "cost_per_1k": options[pick]["cost_per_1k"], "expected_total_per_1k": scored[pick], "meets_floor": pick in ok}


def calibration_report(rows, preds, path):
    """Do predicted failure probabilities match observed failure rates? (Brier score + reliability chart.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
    ax.plot([0, 1], [0, 1], ":", color="grey", label="perfectly calibrated")
    out = {}
    for bar, (p, _) in preds.items():
        y = np.array([0 if r["good"][bar][0] else 1 for r in rows])
        bins = np.clip((p * 5).astype(int), 0, 4)
        xs, ys, ns = [], [], []
        for b in range(5):
            m = bins == b
            if m.sum() >= 5:
                xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
        brier = float(np.mean((p - y) ** 2))
        out[bar] = {"brier": brier, "base_rate_brier": float(np.mean((y.mean() - y) ** 2)),
                    "bins": [{"predicted": float(a), "observed": float(b), "n": c} for a, b, c in zip(xs, ys, ns)]}
        ax.plot(xs, ys, "o-", label=f"{bar} bar (Brier {brier:.3f})")
    ax.set_xlabel("Predicted P(cheap answer is bad)")
    ax.set_ylabel("Observed share of bad cheap answers")
    ax.set_title("Calibration of the failure predictor", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return out


def plot(decisions, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    for d in decisions:
        xs = [s["error_cost_inr"] for s in d["sensitivity"]]
        ys = [s["share_strong"] * 100 for s in d["sensitivity"]]
        line, = ax.plot(xs, ys, lw=1.7, label=d["use_case"].split(" (")[0])
        ax.scatter([d["error_cost_inr"]], [np.interp(d["error_cost_inr"], xs, ys)], color=line.get_color(), zorder=3, s=30)
    ax.set_xscale("log")
    ax.set_xlabel("Cost of one bad answer to the business (₹, log scale)")
    ax.set_ylabel("Requests sent to the frontier model (%)")
    ax.set_title("The threshold follows the business: higher stakes, more escalation\n"
                 "(dots = each use case's assumed cost of a bad answer)", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def inr(x):
    return f"₹{x:,.0f}" if x >= 100 else f"₹{x:,.2f}"


def write_markdown(decisions, cheap_model, path, calib):
    rec = lambda d: d["recommendation"]
    total_strong = sum(d["monthly_cost_inr"]["always_strong"] for d in decisions)
    total_rec = sum(rec(d)["cost_per_1k"] / 1000 * d["monthly_volume"] for d in decisions)
    L = ["# Routing decisions by business use case", "",
         f"Generated by `python -m hybrid.decide` from measured answers of **{cheap_model}** (cheap) and **{STRONG}** "
         f"(frontier). Quality is judged against each use case's bar; costs are equivalent API list prices at "
         f"₹{INR_PER_USD:.0f}/$. The business inputs are stated assumptions, chosen to be argued with.", "",
         "**How a strategy is chosen:** among the strategies that meet the use case's quality floor, pick the one with "
         "the lowest *expected total cost* = API cost + (share of bad answers × business cost of a bad answer).", "",
         "| Strategy | How it works |", "|---|---|"]
    L += [f"| {STRATEGY_NAMES[k]} | {v} |" for k, v in {
        "always_cheap": "Every request goes to the cheap model.",
        "always_strong": "Every request goes to the frontier model.",
        "policy": "A classifier predicts from the prompt alone whether the cheap answer will be bad; escalate when "
                  "P(bad) × cost of a bad answer ≥ the extra cost.",
        "verify_cascade": "The cheap model answers; a cheap check (unit tests for code, a Haiku verifier otherwise) "
                          "reads the answer; only failures are escalated."}.items()]
    ov_path = RESULTS / "overhead.json"
    if ov_path.exists():
        ov = json.loads(ov_path.read_text(encoding="utf-8"))
        bits = []
        if ov.get("ml_router"):
            d = ov["ml_router"]["decision_ms"]
            bits.append(f"the prompt classifier decides in {d['median']:.1f} ms (median; p95 {d['p95']:.1f} ms) on a laptop CPU, at no API cost")
        if ov.get("verifier_single"):
            v = ov["verifier_single"]
            bits.append(f"the Haiku check adds {v['latency_s']['median']:.1f} s (median; p95 {v['latency_s']['p95']:.1f} s) "
                        f"and about ${v['cost_usd']['mean']:.4f} per checked answer; that time is included in the p95 latency below")
        if ov.get("llm_router_single"):
            v = ov["llm_router_single"]
            bits.append(f"for comparison, a Haiku *LLM router* would add {v['latency_s']['median']:.1f} s (median) and "
                        f"${v['cost_usd']['mean']:.4f} per request before any answer is written")
        if bits:
            L += ["", "**Router overhead (measured by `python -m hybrid.overhead`):** " + "; ".join(bits) + "."]
    L += ["", "## Summary", "",
          "| Use case | Bar | Cost of a bad answer | Floor | **Recommended** | Quality | Cost / 1k | Saving vs all-frontier | Robust? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in decisions:
        r = rec(d)
        sv = 1 - r["cost_per_1k"] / d["always_strong"]["cost_per_1k"]
        L.append(f"| {d['use_case']} | {d['quality_bar']} | {inr(d['error_cost_inr'])} | {d['quality_floor']:.0%} | "
                 f"**{r['label']}** | {r['quality']:.0%} | {inr(r['cost_per_1k'])} | {sv:.0%} | "
                 f"{'✅ stable' if d['robustness']['stable'] else '⚠️ fragile'} |")
    L += ["", f"**Company view** (all use cases at their stated volumes): all-frontier ≈ **{inr(total_strong)}/month**; "
          f"recommended strategies ≈ **{inr(total_rec)}/month** "
          f"(**{1 - total_rec / total_strong:.0%} lower**), with every use case at or above its quality floor"
          f"{'' if all(rec(d)['meets_floor'] for d in decisions) else ' except where flagged below'}.", "",
          "![Sensitivity](../results/policy_sensitivity.png)", ""]
    for bar, c in calib.items():
        L.append(f"- Prompt-only predictor calibration ({bar} bar): Brier {c['brier']:.3f} vs {c['base_rate_brier']:.3f} for "
                 f"always guessing the average ({'better' if c['brier'] < c['base_rate_brier'] else 'worse'} than the baseline).")
    L.append("")
    for d in decisions:
        r = rec(d)
        opts = [(k, d[k]) for k in ("always_cheap", "always_strong", "policy", "verify_cascade") if k in d]
        L += [f"## {d['use_case']}", "", f"*{d['rationale']}*", "",
              f"**Inputs:** bar = {d['quality_bar']} · cost of a bad answer = {inr(d['error_cost_inr'])} · floor = "
              f"{d['quality_floor']:.0%} · volume = {d['monthly_volume']:,}/month · latency budget p95 = "
              f"{d['latency_budget_p95_s']:.0f} s · n = {d['n']} prompts", "",
              "| Strategy | Quality | Sent to frontier | Cost / 1k | Expected total / 1k | p95 latency |", "|---|---|---|---|---|---|"]
        for k, v in opts:
            share = {"always_cheap": 0.0, "always_strong": 1.0}.get(k, v.get("share_strong", 0))
            exp_total = v["cost_per_1k"] + (1 - v["quality"]) * d["error_cost_inr"] * 1000
            mark = " ✅" if k == r["strategy"] else ""
            L.append(f"| {STRATEGY_NAMES[k]}{mark} | {v['quality']:.0%} | {share:.0%} | {inr(v['cost_per_1k'])} | "
                     f"{inr(exp_total)} | {v.get('p95_latency_s', float('nan')):.0f} s |")
        monthly = r["cost_per_1k"] / 1000 * d["monthly_volume"]
        L += ["", f"- **Decision:** {r['label']} → {r['quality']:.0%} quality at {inr(r['cost_per_1k'])}/1k; "
              f"about {inr(monthly)}/month vs {inr(d['monthly_cost_inr']['always_strong'])} all-frontier."
              + ("" if r["meets_floor"] else " ⚠️ No strategy meets the floor; this is the highest-quality option.")]
        if "verify_cascade" in d and d["verify_cascade"]["verifier_catch_rate"] is not None:
            v = d["verify_cascade"]
            L.append(f"- **The check itself:** caught {v['verifier_catch_rate']:.0%} of bad cheap answers; escalated "
                     f"{v['verifier_false_alarm_rate']:.0%} of answers that were actually fine."
                     + (f" Checking each answer on its own call (no batching, as an interactive product would) raises this "
                        f"strategy to {inr(v['cost_per_1k_unbatched_check'])}/1k, a "
                        f"{1 - v['cost_per_1k_unbatched_check'] / d['always_strong']['cost_per_1k']:.0%} saving vs all-frontier."
                        if v.get("cost_per_1k_unbatched_check") and rec(d)["strategy"] == "verify_cascade" else ""))
        rk = rec(d)["strategy"]
        if d[rk].get("p95_latency_s", 0) > d["latency_budget_p95_s"]:
            L.append(f"- **Latency:** ⚠️ the recommended strategy's p95 is {d[rk]['p95_latency_s']:.0f} s, over the "
                     f"{d['latency_budget_p95_s']:.0f} s budget.")
        flip = d["error_cost_where_majority_goes_frontier"]
        if flip:
            L.append(f"- **Threshold intuition:** with prompt-based routing, most traffic moves to the frontier model once a "
                     f"bad answer costs about {inr(flip)} or more.")
        rb = d["robustness"]["variants"]
        changed = [f"{k} → {STRATEGY_NAMES[v]}" for k, v in rb.items() if v != r["strategy"]]
        L.append("- **Robustness:** " + ("the recommendation holds under ×0.5/×2 error cost and ±5-point floors."
                                         if not changed else "changes when " + "; ".join(changed) + "."))
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):  # ₹ and ≈ crash the default Windows console encoding
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", default="claude/haiku-nothink")
    ap.add_argument("--min-items", type=int, default=10, help="minimum graded items per use case")
    ap.add_argument("--public", action="store_true", help="only the 350 public prompts (what docs/DECISIONS.md reports)")
    args = ap.parse_args()
    rows = item_rows(args.cheap, include_real=not args.public)
    if len(rows) < 30:
        raise SystemExit(f"Only {len(rows)} graded items for {args.cheap}; grade more first.")
    preds = {bar: out_of_fold_predictions(rows, bar) for bar in ("acceptable", "best")}
    decisions = [d for uc in USE_CASES if (d := decide_use_case(uc, rows, preds, args.min_items))]
    skipped = [uc.name for uc in USE_CASES if not any(d["key"] == uc.key for d in decisions)]

    tag = args.cheap.split("/")[-1]
    RESULTS.mkdir(exist_ok=True)
    calib = calibration_report(rows, preds, RESULTS / f"policy_calibration_{tag}.png")
    (RESULTS / f"policy_calibration_{tag}.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    (RESULTS / f"policy_decisions_{tag}.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    plot(decisions, RESULTS / "policy_sensitivity.png")
    DOCS.mkdir(exist_ok=True)
    write_markdown(decisions, args.cheap, DOCS / "DECISIONS.md", calib)

    print(f"{len(rows)} graded items. Decisions for {len(decisions)} use cases" + (f"; skipped (too few items): {skipped}" if skipped else ""))
    for bar, c in calib.items():
        print(f"Calibration ({bar} bar): Brier {c['brier']:.3f} vs {c['base_rate_brier']:.3f} for always predicting the base rate")
    print(f"{'use case':52} {'recommended':42} {'quality':>8} {'cost/1k':>9} {'saving':>7} robust")
    for d in decisions:
        r = d["recommendation"]
        sv = 1 - r["cost_per_1k"] / d["always_strong"]["cost_per_1k"]
        print(f"{d['use_case'][:52]:52} {r['label'][:42]:42} {r['quality']:8.0%} {inr(r['cost_per_1k']):>9} {sv:7.0%} "
              f"{'yes' if d['robustness']['stable'] else 'FRAGILE'}{'' if r['meets_floor'] else ' (floor not met)'}")
        if "verify_cascade" in d:
            v = d["verify_cascade"]
            print(f"{'':52}   verify cascade: {v['quality']:.0%} quality, {v['share_strong']:.0%} escalated, "
                  f"catch {v['verifier_catch_rate'] if v['verifier_catch_rate'] is None else round(v['verifier_catch_rate'], 2)}")


if __name__ == "__main__":
    main()
