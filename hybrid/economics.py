"""Business economics behind the routing strategy, computed from measured results.

    python -m hybrid.economics      # writes results/economics.json and two charts

Three questions:
A. Personal: what is my real AI usage worth at API prices, and what does routing buy on a flat
   subscription (capacity, not money)?
B. Company: does building a router pay back, and how fast?
C. Strategy over time: what happens to routing's value as frontier prices fall?

Every assumption is named in ASSUMPTIONS and printed, so the numbers can be challenged.
"""

import sys
import json
from pathlib import Path

import numpy as np

from .decide import INR_PER_USD, recommend

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

ASSUMPTIONS = {
    "subscription_inr_per_month": 2300,          # what Kruthi pays (corrected 2026-10-09)
    "active_days_per_month": [10, 20],           # light vs heavy months of building
    "sprint_active_days": 4,                     # 5-8 Oct 2026, the two sessions analysed
    # Company router project
    "build_engineer_weeks": 6,                   # gateway, policy engine, eval harness, dashboards
    "engineer_cost_inr_per_month": 250_000,      # fully loaded, mid-level engineer in India
    "maintenance_fte": 0.25,                     # ongoing owner for thresholds, evals, model updates
    "shadow_eval_share_of_traffic": 0.02,        # share of requests re-judged to watch quality
    "volume_multipliers": [0.25, 1, 4],          # smaller and larger companies than the base case
    "frontier_price_factors": [1.0, 0.75, 0.5, 0.25],  # frontier prices falling relative to today
}


def personal(sessions_path: Path):
    rows = [json.loads(l) for l in sessions_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_kind = {}
    for k in ("text", "lookup", "action"):
        rs = [r for r in rows if r["kind"] == k]
        tok = {t: sum(r["opus_tokens"][t] for r in rs) for t in ("fresh_input", "cache_write", "cache_read", "output")}
        by_kind[k] = {"n": len(rs), "equiv_usd": sum(r["opus_equiv_cost_usd"] for r in rs), "tokens": tok}
    total = sum(v["equiv_usd"] for v in by_kind.values())
    per_day = total / ASSUMPTIONS["sprint_active_days"]
    monthly = {d: per_day * d * INR_PER_USD for d in ASSUMPTIONS["active_days_per_month"]}
    share = {k: v["equiv_usd"] / total for k, v in by_kind.items()}

    # Capacity scenarios on a flat subscription, using API-equivalent cost as the proxy for limit use.
    action_cache = by_kind["action"]["tokens"]["cache_read"] * 0.20 / 1e6 / total  # share of all usage
    scenarios = {
        "Route question-style requests to Haiku (≈1/4 the price)": share["text"] * 0.75,
        "Run 30% of agentic sub-steps on Sonnet (½ the price)": share["action"] * 0.30 * 0.5,
        "Halve average context length (fresh sessions / compaction)": action_cache * 0.5,
    }
    scenarios["All three together"] = 1 - np.prod([1 - v for v in scenarios.values()])
    capacity = {k: {"usage_saved": v, "more_work_per_window": 1 / (1 - v) - 1} for k, v in scenarios.items()}
    return {"by_kind": by_kind, "share_of_usage": share, "equiv_usd_total": total, "equiv_usd_per_active_day": per_day,
            "equiv_api_inr_per_month": monthly, "subscription_inr": ASSUMPTIONS["subscription_inr_per_month"],
            "api_multiple_of_subscription": {d: v / ASSUMPTIONS["subscription_inr_per_month"] for d, v in monthly.items()},
            "capacity_scenarios": capacity}


def personal_or_published(path):
    """My own session log is private. Without it, reuse the published personal figures."""
    if path.exists():
        return personal(path)
    print("(data/session_requests.jsonl is private and not in the repo; using the published personal figures from results/economics.json)\n")
    return json.loads((RESULTS / "economics.json").read_text(encoding="utf-8"))["personal"]


def judge_cost_per_request_inr():
    path = RESULTS / "hybrid_grades.jsonl"
    if not path.exists():
        path = RESULTS / "public" / "hybrid_grades.jsonl"
    costs = []
    for l in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "judge" and r.get("judge_cost_usd"):
            costs.append(r["judge_cost_usd"])
    return float(np.mean(costs)) * INR_PER_USD if costs else 1.0


def company(decisions):
    a = ASSUMPTIONS
    all_frontier = sum(d["monthly_cost_inr"]["always_strong"] for d in decisions)
    routed = sum(d["recommendation"]["cost_per_1k"] / 1000 * d["monthly_volume"] for d in decisions)
    volume = sum(d["monthly_volume"] for d in decisions)
    judge = judge_cost_per_request_inr()
    build = a["build_engineer_weeks"] / 4.33 * a["engineer_cost_inr_per_month"]
    out = {"base_monthly_volume": volume, "judge_cost_per_check_inr": judge, "build_cost_inr": build, "scenarios": {}}
    for m in a["volume_multipliers"]:
        saving = (all_frontier - routed) * m
        run = a["maintenance_fte"] * a["engineer_cost_inr_per_month"] + volume * m * a["shadow_eval_share_of_traffic"] * judge
        net = saving - run
        out["scenarios"][f"{m}x volume"] = {
            "monthly_requests": volume * m, "all_frontier_inr": all_frontier * m, "routed_inr": routed * m,
            "gross_saving_inr": saving, "run_cost_inr": run, "net_saving_inr": net,
            "payback_months": build / net if net > 0 else None}
    return out


def price_scenarios(decisions):
    """Re-decide every use case with frontier costs scaled down; cheap-model costs unchanged."""
    out = {}
    for f in ASSUMPTIONS["frontier_price_factors"]:
        total_frontier = total_routed = 0.0
        routed_cases = []
        for d in decisions:
            s = d["always_strong"]["cost_per_1k"]
            c = d["always_cheap"]["cost_per_1k"]
            opts = {"always_cheap": dict(d["always_cheap"]), "always_strong": {**d["always_strong"], "cost_per_1k": s * f}}
            for k in ("policy", "verify_cascade"):
                if k in d:
                    share = d[k]["share_strong"]
                    other = d[k]["cost_per_1k"] - share * s          # cheap + verification part
                    opts[k] = {**d[k], "cost_per_1k": other + share * s * f}
            r = recommend(opts, d["error_cost_inr"], d["quality_floor"])
            total_frontier += s * f / 1000 * d["monthly_volume"]
            total_routed += r["cost_per_1k"] / 1000 * d["monthly_volume"]
            if r["strategy"] != "always_strong":
                routed_cases.append(d["key"])
        out[f"{int(f * 100)}% of today's frontier price"] = {
            "all_frontier_inr": total_frontier, "recommended_inr": total_routed,
            "saving_inr": total_frontier - total_routed, "saving_share": 1 - total_routed / total_frontier,
            "use_cases_not_all_frontier": routed_cases}
    return out


def charts(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = result["personal"]["capacity_scenarios"]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    names = list(cap)
    vals = [cap[n]["usage_saved"] * 100 for n in names]
    ax.barh(names[::-1], vals[::-1], color=["#2c7349", "#1d5c8a", "#1d5c8a", "#1d5c8a"])
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, y, f"{v:.0f}% less usage", va="center", fontsize=8)
    ax.set_xlabel("Share of my Claude usage saved (API-equivalent cost as proxy)")
    ax.set_title("What actually stretches a subscription: context and agentic steps, not questions", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.35)
    fig.tight_layout()
    fig.savefig(RESULTS / "economics_capacity.png")
    plt.close(fig)

    ps = result["price_scenarios"]
    xs = [int(k.split("%")[0]) for k in ps]
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.plot(xs, [ps[k]["saving_share"] * 100 for k in ps], "o-", color="#1d5c8a")
    for x, k in zip(xs, ps):
        ax.annotate(f"{len(ps[k]['use_cases_not_all_frontier'])} use cases\nnot all-frontier", (x, ps[k]["saving_share"] * 100),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Frontier price, % of today's")
    ax.set_ylabel("Company-wide saving from routing (%)")
    ax.set_title("Routing's value shrinks as frontier prices fall", fontsize=10)
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "economics_price_scenarios.png")
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):  # ₹ and ≈ crash the default Windows console encoding
        sys.stdout.reconfigure(encoding="utf-8")
    decisions = json.loads((RESULTS / "policy_decisions_haiku-nothink.json").read_text(encoding="utf-8"))
    result = {"assumptions": ASSUMPTIONS,
              "personal": personal_or_published(ROOT / "data" / "session_requests.jsonl"),
              "company": company(decisions),
              "price_scenarios": price_scenarios(decisions)}
    (RESULTS / "economics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    charts(result)

    p = result["personal"]
    print(f"PERSONAL: {p['equiv_usd_total']:.0f} USD API-equivalent over {ASSUMPTIONS['sprint_active_days']} active days "
          f"(≈ ₹{p['equiv_usd_per_active_day'] * INR_PER_USD:,.0f}/active day)")
    for d, v in p["equiv_api_inr_per_month"].items():
        print(f"  {d} active days/month → ₹{v:,.0f} at API prices = {p['api_multiple_of_subscription'][d]:.0f}× the ₹2,300 subscription")
    print("  usage share:", {k: f"{v:.0%}" for k, v in p["share_of_usage"].items()})
    for k, v in p["capacity_scenarios"].items():
        print(f"  {k}: {v['usage_saved']:.0%} less usage → {v['more_work_per_window']:.0%} more work per limit window")
    c = result["company"]
    print(f"\nCOMPANY (base {c['base_monthly_volume']:,} requests/month; build ≈ ₹{c['build_cost_inr']:,.0f}; "
          f"judge check ≈ ₹{c['judge_cost_per_check_inr']:.2f})")
    for k, v in c["scenarios"].items():
        pb = f"{v['payback_months']:.1f} months" if v["payback_months"] else "never"
        print(f"  {k:10}: all-frontier ₹{v['all_frontier_inr']:,.0f}/mo, routed ₹{v['routed_inr']:,.0f}/mo, "
              f"net saving ₹{v['net_saving_inr']:,.0f}/mo after run costs → payback {pb}")
    print("\nPRICE SCENARIOS")
    for k, v in result["price_scenarios"].items():
        print(f"  {k:28}: saving {v['saving_share']:.0%} (₹{v['saving_inr']:,.0f}/mo); routed use cases: {v['use_cases_not_all_frontier']}")


if __name__ == "__main__":
    main()
