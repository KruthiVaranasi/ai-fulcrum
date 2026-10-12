import json

import pytest

from hybrid.grading import (
    build_judge_prompt,
    extract_code,
    extract_number,
    grade_number,
    parse_judge_output,
    routing_scores,
    run_tests,
    strong_is_a,
    to_cheap_strong,
)


def test_extract_number_prefers_answer_line():
    assert extract_number("We get 12 then 18.\nAnswer: 18") == 18
    assert extract_number("The total is $1,250.") == 1250
    assert extract_number("**Answer:** 7.5") == 7.5
    assert extract_number("no numbers here") is None
    assert grade_number("Answer: 42", "42") and not grade_number("Answer: 41", "42")


def test_extract_code_takes_last_block():
    text = "Here:\n```python\ndef a(): return 1\n```\nBetter:\n```python\ndef a(): return 2\n```"
    assert extract_code(text).strip() == "def a(): return 2"
    assert extract_code("def b(): pass") == "def b(): pass"


def test_run_tests_pass_fail_timeout():
    assert run_tests("def add(a, b): return a + b", "assert add(2, 3) == 5")["passed"]
    fail = run_tests("def add(a, b): return a - b", "assert add(2, 3) == 5")
    assert not fail["passed"] and "AssertionError" in fail["error"]
    slow = run_tests("while True: pass", "", timeout=2)
    assert not slow["passed"] and slow["error"] == "timeout"


def verdict(mb="neither", a=True, b=True, sa=4, sb=4):
    return {"item": 1, "A_meets_need": a, "B_meets_need": b, "materially_better": mb, "score_A": sa, "score_B": sb}


def test_judge_parsing_and_mapping():
    out = "Sure.\n" + json.dumps([verdict("A", True, False, 5, 2)])
    v = parse_judge_output(out, 1)[0]
    as_strong_first = to_cheap_strong(v, strong_first=True)
    assert as_strong_first["outcome"] == "strong_better" and as_strong_first["cheap_meets_need"] is False
    as_cheap_first = to_cheap_strong(v, strong_first=False)
    assert as_cheap_first["outcome"] == "cheap_better" and as_cheap_first["score_cheap"] == 5
    with pytest.raises(ValueError):
        parse_judge_output(json.dumps([verdict("maybe")]), 1)
    with pytest.raises(ValueError):
        parse_judge_output(json.dumps([{**verdict(), "A_meets_need": "yes"}]), 1)
    with pytest.raises(ValueError):
        parse_judge_output("no json", 1)


def test_routing_scores_and_position_balance():
    both_fine = to_cheap_strong(verdict("neither"), strong_first=True)
    assert routing_scores(both_fine) == (1.0, 1.0)  # minor differences count as a tie
    cheap_fails = to_cheap_strong(verdict("neither", a=True, b=False), strong_first=True)
    assert routing_scores(cheap_fails) == (0.0, 1.0)
    strong_material = to_cheap_strong(verdict("A"), strong_first=True)
    assert routing_scores(strong_material) == (0.0, 1.0)
    firsts = [strong_is_a(f"item-{i}", "claude/haiku-nothink") for i in range(400)]
    assert 0.4 < sum(firsts) / len(firsts) < 0.6  # roughly balanced A/B positions


def test_judge_prompt_contains_key_points_and_both_answers():
    item = {"prompt": "Explain RAG", "reference": {"rubric": "Accurate.", "key_points": ["retrieval", "citations"]}}
    p = build_judge_prompt([{"item": item, "cheap_text": "CHEAP", "strong_text": "STRONG", "strong_first": False}])
    assert "ANSWER KEY" in p and "- citations" in p
    assert p.index("CHEAP") < p.index("STRONG")  # cheap shown as A when strong_first is False
