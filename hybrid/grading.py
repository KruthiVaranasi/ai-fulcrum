"""Graders for the hybrid dataset.

Objective (no judge): multiple-choice letter, numeric answer, unit tests run in a subprocess.
Judged: a blind pairwise comparison (cheap vs strong answer) by an LLM judge, several pairs per call.
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from router.live import extract_choice

# ---------------------------------------------------------------- objective graders

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str):
    """Prefer an explicit 'Answer: <number>' line; otherwise the last number in the text."""
    if not text:
        return None
    m = re.findall(r"answer\s*[:=]\s*\**\s*\$?\s*(-?\d[\d,]*\.?\d*)", text, flags=re.I)
    cand = m[-1] if m else (_NUM.findall(text) or [None])[-1]
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade_number(text: str, gold: str) -> bool:
    got = extract_number(text)
    return got is not None and abs(got - float(gold)) < 1e-6


def grade_choice(text: str, gold: str) -> bool:
    return extract_choice(text) == gold


def extract_code(text: str) -> str:
    """Last ```python (or plain ```) block; the whole text if there is none."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    return blocks[-1] if blocks else (text or "")


def run_tests(code: str, tests: str, setup: str = "", timeout: int = 10) -> dict:
    """Run model-written code plus unit tests in a separate Python process inside a temp directory.

    Limits damage from bad code with a timeout and an isolated working directory; it is not a full
    sandbox, so only use it on well-known benchmark tasks.
    """
    program = "\n\n".join([setup or "", code, tests])
    with tempfile.TemporaryDirectory(prefix="router-tests-") as d:
        path = Path(d) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            out = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True,
                                 text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
    if out.returncode == 0:
        return {"passed": True, "error": None}
    last = (out.stderr.strip().splitlines() or ["failed"])[-1]
    return {"passed": False, "error": last[:200]}


def grade_objective(item: dict, text: str) -> dict:
    ref = item["reference"]
    if item["eval_type"] == "choice":
        return {"correct": grade_choice(text, ref["answer"])}
    if item["eval_type"] == "number":
        return {"correct": grade_number(text, ref["answer"])}
    if item["eval_type"] == "tests":
        r = run_tests(extract_code(text), ref["tests"], ref.get("setup", ""))
        return {"correct": r["passed"], "error": r["error"]}
    raise ValueError(f"not an objective item: {item['eval_type']}")


# ---------------------------------------------------------------- pairwise judge

# v2. v1 asked "which answer is better?" and never returned a tie (45 of 49 pilot pairs went to the
# stronger model, including a grammar fix where both answers fixed every error). The routing question
# is "is the cheap answer good enough?", so v2 asks for binary, need-based judgements instead.
JUDGE_VERSION = "v2-meets-need"
JUDGE_INSTRUCTIONS = """You are a fair evaluator checking whether AI answers meet a user's need.

For each item below, using the item's criteria (and its answer key or reference answer when given):

1. For EACH answer, decide MEETS_NEED (true/false): it does what the user asked, with no significant
   errors and no important omissions. Minor style differences, wording choices, slightly less detail,
   or a missing nice-to-have do NOT fail an answer. Factual errors, wrong conclusions, ignoring the
   request, or missing key points that matter DO fail it.
2. Decide MATERIALLY_BETTER: "A" or "B" only if a typical user with this request would clearly notice
   and care about the difference; otherwise "neither". Do not reward length or formatting by itself;
   padding the user didn't ask for is a flaw.
3. Score each answer 1-5 (5 = excellent, 3 = acceptable, 1 = wrong or unhelpful).

Return ONLY a JSON array, one object per item, in order:
[{"item": 1, "A_meets_need": true, "B_meets_need": true, "materially_better": "A" | "B" | "neither",
  "score_A": 1-5, "score_B": 1-5, "reason": "<= 20 words"}]
"""

MAX_PROMPT_CHARS = 3000
MAX_ANSWER_CHARS = 6000


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n[... truncated ...]"


def strong_is_a(item_id: str, cheap_model: str) -> bool:
    """Deterministic, balanced A/B assignment per pair so position bias averages out and reruns match."""
    return int(hashlib.sha1(f"{item_id}|{cheap_model}".encode()).hexdigest(), 16) % 2 == 0


def build_judge_prompt(pairs: list) -> str:
    """pairs: dicts with item, cheap_text, strong_text, strong_first (bool)."""
    parts = [JUDGE_INSTRUCTIONS]
    for n, p in enumerate(pairs, 1):
        ref = p["item"]["reference"]
        a, b = (p["strong_text"], p["cheap_text"]) if p["strong_first"] else (p["cheap_text"], p["strong_text"])
        block = [f"\n=== ITEM {n} ===", f"USER REQUEST:\n{_clip(p['item']['prompt'], MAX_PROMPT_CHARS)}",
                 f"CRITERIA: {ref.get('rubric', 'Correct, complete and responsive.')}"]
        if ref.get("key_points"):
            block.append("ANSWER KEY (a good answer covers these):\n" + "\n".join(f"- {k}" for k in ref["key_points"]))
        if ref.get("reference_answer"):
            block.append(f"REFERENCE ANSWER (human-written, for guidance):\n{_clip(ref['reference_answer'], 1500)}")
        if ref.get("max_words"):
            block.append(f"LENGTH LIMIT: the user asked for at most about {ref['max_words']} words.")
        block += [f"--- ANSWER A ---\n{_clip(a, MAX_ANSWER_CHARS)}", f"--- ANSWER B ---\n{_clip(b, MAX_ANSWER_CHARS)}"]
        parts.append("\n".join(block))
    return "\n".join(parts)


def parse_judge_output(text: str, n: int) -> list:
    """Extract the JSON array; raise ValueError if it is missing or malformed."""
    start, end = (text or "").find("["), (text or "").rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in judge output")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list) or len(data) != n:
        raise ValueError(f"expected {n} verdicts, got {len(data) if isinstance(data, list) else 'non-list'}")
    for d in data:
        if d.get("materially_better") not in ("A", "B", "neither"):
            raise ValueError(f"bad materially_better {d.get('materially_better')!r}")
        if not isinstance(d.get("A_meets_need"), bool) or not isinstance(d.get("B_meets_need"), bool):
            raise ValueError("meets_need must be true/false for both answers")
    return data


def to_cheap_strong(verdict: dict, strong_first: bool) -> dict:
    """Translate an A/B verdict into cheap/strong terms."""
    mb = verdict["materially_better"]
    if mb == "neither":
        outcome = "tie"
    else:
        outcome = "strong_better" if (mb == "A") == strong_first else "cheap_better"
    a_meets, b_meets = verdict["A_meets_need"], verdict["B_meets_need"]
    sa, sb = verdict.get("score_A"), verdict.get("score_B")
    strong_meets, cheap_meets = (a_meets, b_meets) if strong_first else (b_meets, a_meets)
    s_strong, s_cheap = (sa, sb) if strong_first else (sb, sa)
    return {"outcome": outcome, "cheap_meets_need": cheap_meets, "strong_meets_need": strong_meets,
            "score_cheap": s_cheap, "score_strong": s_strong, "reason": verdict.get("reason", ""),
            "judge_version": JUDGE_VERSION}


def routing_scores(g: dict):
    """(s_cheap, s_strong): 1 if that answer meets the user's need and isn't materially worse than the other."""
    s_cheap = float(g["cheap_meets_need"] and g["outcome"] != "strong_better")
    s_strong = float(g["strong_meets_need"] and g["outcome"] != "cheap_better")
    return s_cheap, s_strong
