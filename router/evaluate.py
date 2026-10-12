"""Turn router scores into the cost/quality trade-off curve and summary metrics.

For a share s of traffic, the router sends the s fraction of prompts it is most worried about
to the strong model and the rest to the cheap one. Sweeping s from 0 to 1 traces the curve.

Metrics:
- PGR (performance gap recovered) = (Q_router - Q_cheap) / (Q_strong - Q_cheap).
- APGR = average PGR over the sweep (area under the PGR curve), as used in the RouteLLM paper.
  Random routing scores ~0.5; a perfect router scores well above it.
- Strong-call share and cost saving needed to match the strong model's quality.
"""

from dataclasses import dataclass, field

import numpy as np

GRID = np.linspace(0, 1, 101)


@dataclass
class Curve:
    name: str
    share_strong: np.ndarray
    quality: np.ndarray
    cost: np.ndarray
    q_cheap: float
    q_strong: float
    c_cheap: float
    c_strong: float
    summary: dict = field(default_factory=dict)


def curve(name, p, s_cheap, s_strong, c_cheap, c_strong) -> Curve:
    n = len(p)
    order = np.argsort(-np.asarray(p), kind="stable")
    # Prefix sums let us evaluate every routing share in O(n).
    gain_q = np.concatenate([[0], np.cumsum((s_strong - s_cheap)[order])])
    gain_c = np.concatenate([[0], np.cumsum((c_strong - c_cheap)[order])])
    k = np.round(GRID * n).astype(int)
    quality = (s_cheap.sum() + gain_q[k]) / n
    cost = c_cheap.sum() + gain_c[k]
    cv = Curve(name, GRID, quality, cost, s_cheap.mean(), s_strong.mean(), c_cheap.sum(), c_strong.sum())
    cv.summary = summarize(cv)
    return cv


def summarize(cv: Curve) -> dict:
    gap = cv.q_strong - cv.q_cheap
    pgr = (cv.quality - cv.q_cheap) / gap
    matches = np.where(cv.quality >= cv.q_strong)[0]
    # Cost-based APGR: average PGR over the spend range from all-cheap to all-strong. Unlike APGR
    # (which averages over share of calls), this rewards routers for saving money, not calls.
    spend = (cv.cost - cv.c_cheap) / (cv.c_strong - cv.c_cheap)
    order = np.argsort(spend)
    grid = np.linspace(0, 1, 201)
    pgr_at_spend = np.interp(grid, spend[order], np.maximum.accumulate(pgr[order]))
    out = {
        "APGR": float(np.trapezoid(pgr, cv.share_strong)),
        "cost_APGR": float(np.trapezoid(pgr_at_spend, grid)),
        "best_quality": float(cv.quality.max()),
        "best_quality_share_strong": float(cv.share_strong[cv.quality.argmax()]),
    }
    if len(matches):
        i = matches[0]
        out["share_strong_to_match_strong"] = float(cv.share_strong[i])
        out["cost_saving_at_match"] = max(0.0, float(1 - cv.cost[i] / cv.c_strong))
    else:
        out["share_strong_to_match_strong"] = None
        out["cost_saving_at_match"] = None
    for rel in (0.90, 0.95):
        hit = np.where(cv.quality >= rel * cv.q_strong)[0]
        key = f"{int(rel * 100)}pct_of_strong_quality"
        out[f"share_strong_at_{key}"] = float(cv.share_strong[hit[0]]) if len(hit) else None
        out[f"cost_saving_at_{key}"] = max(0.0, float(1 - cv.cost[hit[0]] / cv.c_strong)) if len(hit) else None
    for target in (0.5, 0.8, 0.95):
        hit = np.where(pgr >= target)[0]
        out[f"share_strong_for_{int(target * 100)}pct_gap"] = float(cv.share_strong[hit[0]]) if len(hit) else None
    return out
