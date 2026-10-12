import numpy as np

from router.data import clean_prompt, task_split
from router.evaluate import curve
from router.live import cost_usd, extract_choice


def make(n=100, seed=0):
    rng = np.random.default_rng(seed)
    s_cheap = rng.integers(0, 2, n).astype(float)
    s_strong = np.maximum(s_cheap, rng.integers(0, 2, n)).astype(float)  # strong never worse here
    return s_cheap, s_strong, np.full(n, 0.01), np.full(n, 0.25)


def test_curve_endpoints_match_single_models():
    s_c, s_s, c_c, c_s = make()
    cv = curve("x", np.zeros(len(s_c)), s_c, s_s, c_c, c_s)
    assert np.isclose(cv.quality[0], s_c.mean()) and np.isclose(cv.cost[0], c_c.sum())
    assert np.isclose(cv.quality[-1], s_s.mean()) and np.isclose(cv.cost[-1], c_s.sum())


def test_oracle_beats_random():
    s_c, s_s, c_c, c_s = make(1000)
    oracle = curve("oracle", s_s - s_c, s_c, s_s, c_c, c_s)
    rand = curve("random", np.random.default_rng(1).random(1000), s_c, s_s, c_c, c_s)
    needed = (s_s > s_c).mean()
    # PGR rises linearly to 1 until every prompt that needs the strong model is routed, then stays at 1.
    assert abs(oracle.summary["APGR"] - (1 - needed / 2)) < 0.01
    assert 0.4 < rand.summary["APGR"] < 0.6
    # The oracle reaches strong-model quality once it has routed every prompt the cheap model gets wrong.
    assert abs(oracle.summary["share_strong_to_match_strong"] - needed) <= 0.011
    assert oracle.summary["cost_saving_at_match"] > 0


def test_task_split_holds_out_whole_tasks():
    tasks = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20 + ["d"] * 10)
    train, test, held = task_split(tasks, holdout_frac=0.2)
    assert set(tasks[test]) == set(held)
    assert not set(tasks[train]) & set(held)


def test_clean_prompt_and_choice_parsing():
    assert clean_prompt("['Hello', 'world']") == "Hello world"
    assert clean_prompt("plain") == "plain"
    for text, want in [("B", "B"), ("['A)']", "A"), ("\nC", "C"), ("The answer is D.", "D"), ("", None)]:
        assert extract_choice(text) == want


def test_cost():
    assert np.isclose(cost_usd("claude-haiku-4-5", 1_000_000, 0), 1.0)
    assert np.isclose(cost_usd("claude-opus-5-5", 0, 1_000_000), 20.0)
