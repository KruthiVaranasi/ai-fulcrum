"""The task-completion test is only as good as its tests: every reference solution must run,
and a wrong solution must be caught."""
from hybrid.tasks import expected, feedback, run_cases
from hybrid.task_specs import TASKS


def test_every_reference_runs_and_ids_are_unique():
    assert len({t["id"] for t in TASKS}) == len(TASKS) == 20
    for t in TASKS:
        assert len(expected(t)) == len(t["cases"])


def test_a_wrong_solution_is_caught_and_explained():
    t = next(t for t in TASKS if t["id"] == "expr")
    got, err = run_cases("def eval_expr(s):\n    return int(eval(s.replace('/', '//')))", t)
    fb, passed = feedback(t, got, err, expected(t))
    assert passed < len(t["cases"]) and "7/-2" in fb
