import numpy as np

from hybrid.decide import decide_use_case
from hybrid.policies import UseCase


def make_rows(n=40, seed=0):
    """Half the prompts are 'hard' (cheap fails, frontier succeeds); half are easy (both succeed)."""
    rng = np.random.default_rng(seed)
    rows, p = [], []
    for i in range(n):
        hard = i % 2 == 0
        good = (not hard, True)
        rows.append({"id": f"x{i}", "category": "simple", "prompt": "", "eval_type": "judge",
                     "good": {"acceptable": good, "best": good}, "c_cheap": 0.1, "c_strong": 1.0,
                     "l_cheap": 2.0, "l_strong": 8.0})
        p.append(0.9 if hard else 0.1 + rng.random() * 0.05)
    p, dc = np.array(p), np.full(n, 0.9)
    return rows, {"acceptable": (p, dc), "best": (p, dc)}


def uc(error_cost, floor=0.0):
    return UseCase("t", "test", ["simple"], "acceptable", error_cost, floor, 1000, "test")


def test_escalation_rises_with_cost_of_a_bad_answer():
    rows, preds = make_rows()
    cheap_world = decide_use_case(uc(error_cost=0.5), rows, preds)
    risky_world = decide_use_case(uc(error_cost=5.0), rows, preds)
    # At ₹0.5 per bad answer, nothing is worth escalating (0.9 x 0.5 < 0.9 extra cost).
    assert cheap_world["policy"]["share_strong"] == 0
    # At ₹5, only the prompts likely to fail (p = 0.9) are escalated: exactly the hard half.
    assert risky_world["policy"]["share_strong"] == 0.5
    assert risky_world["policy"]["quality"] == 1.0
    shares = [s["share_strong"] for s in risky_world["sensitivity"]]
    assert shares == sorted(shares)  # monotonic in the cost of a bad answer


def test_latency_budget_is_checked():
    rows, preds = make_rows()
    fast = decide_use_case(UseCase("t", "test", ["simple"], "acceptable", 0.5, 0.0, 1000, "test", p95_latency_budget_s=5), rows, preds)
    assert fast["policy"]["p95_latency_s"] == 2.0 and fast["latency_ok"]  # all cheap: within a 5 s budget
    slow = decide_use_case(UseCase("t", "test", ["simple"], "acceptable", 5.0, 0.0, 1000, "test", p95_latency_budget_s=5), rows, preds)
    assert slow["policy"]["p95_latency_s"] == 8.0 and not slow["latency_ok"]  # escalated half: p95 hits the 8 s tier


def test_quality_floor_forces_escalation():
    rows, preds = make_rows()
    d = decide_use_case(uc(error_cost=0.5, floor=0.95), rows, preds)
    assert d["floor_met"] and d["policy"]["quality"] >= 0.95
    assert d["effective_error_cost_inr"] > 0.5
    assert 0 < d["saving_vs_always_strong"] < 1


def test_recommend_picks_cheapest_strategy_that_meets_the_floor():
    from hybrid.decide import recommend

    d = {"always_cheap": {"quality": 0.80, "cost_per_1k": 100},
         "always_strong": {"quality": 0.99, "cost_per_1k": 1000},
         "policy": {"quality": 0.90, "cost_per_1k": 700},
         "verify_cascade": {"quality": 0.97, "cost_per_1k": 300}}
    # Floor 95%: only the frontier and the cascade qualify; the cascade is far cheaper.
    assert recommend(d, error_cost=10, floor=0.95)["strategy"] == "verify_cascade"
    # No floor, nearly free errors: the cheap model wins on expected total cost.
    assert recommend(d, error_cost=0.01, floor=0.0)["strategy"] == "always_cheap"
    # Floor nobody meets: fall back to the highest quality and say so.
    r = recommend(d, error_cost=10, floor=0.999)
    assert r["strategy"] == "always_strong" and not r["meets_floor"]
