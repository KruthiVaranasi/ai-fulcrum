"""Human check of the AI judge: I grade a sample blind, then measure how often the judge agrees with me.

    python -m hybrid.human_review make            # writes human_review/grading_sheet.csv (+ a hidden key)
    python -m hybrid.human_review score           # after grading: agreement report -> results/human_review.json

The sheet asks the same three questions the AI judge answered (hybrid/grading.py JUDGE_INSTRUCTIONS):
does answer A meet the need, does answer B, and is one materially better. Answers are shown as A and B
in a random order per item, with no model names, and the judge's verdicts are not in the sheet.
The A/B order is kept in results/human_review_key.json: don't open it before grading.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SHEET = ROOT / "human_review" / "grading_sheet.csv"
KEY = ROOT / "results" / "human_review_key.json"
OUT = ROOT / "results" / "human_review.json"
CHEAP, STRONG = "claude/haiku-nothink", "claude/opus"
# 50 items across the 8 kinds of work, roughly in proportion to how many judged prompts each has.
PLAN = {"simple": 6, "general_qa": 5, "reasoning": 4, "software": 4, "aiml": 7, "pm_explain": 8, "pm_decide": 8, "interview": 8}
LABEL = {"simple": "Rewrites & simple tasks", "general_qa": "Knowledge Q&A", "reasoning": "Maths & reasoning", "software": "Coding",
         "aiml": "ML engineering", "pm_explain": "PM explanations", "pm_decide": "PM decisions", "interview": "Interview prep"}
YES = {"y", "yes", "true", "1"}
NO = {"n", "no", "false", "0"}


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def criteria(ref):
    parts = [ref.get("rubric", "Correct, complete and responsive.")]
    if ref.get("key_points"):
        parts.append("A good answer covers: " + "; ".join(ref["key_points"]))
    if ref.get("reference_answer"):
        parts.append("Reference answer (for guidance): " + ref["reference_answer"])
    if ref.get("max_words"):
        parts.append(f"The user asked for at most about {ref['max_words']} words.")
    return "\n".join(parts)


def make():
    data = {d["id"]: d for d in load_jsonl(ROOT / "hybrid" / "dataset.jsonl")}
    answers = {(a["model"], a["id"]): a["text"] for a in load_jsonl(ROOT / "results" / "public" / "hybrid_answers.jsonl")}
    grades = {g["id"]: g for g in load_jsonl(ROOT / "results" / "public" / "hybrid_grades.jsonl")
              if g.get("kind") == "judge" and g.get("cheap_model") == CHEAP and g.get("check", "") == ""
              and g.get("judge_version") == "v2-meets-need"}
    rng = random.Random(2026)
    picked = []
    for cat, n in PLAN.items():
        pool = sorted(i for i, d in data.items() if d["category"] == cat and d["eval_type"] == "judge" and i in grades
                      and (CHEAP, i) in answers and (STRONG, i) in answers)
        picked += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(picked)
    SHEET.parent.mkdir(exist_ok=True)
    key = {}
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "kind_of_work", "request", "what_a_good_answer_needs", "answer_A", "answer_B",
                    "A_meets_need (yes/no)", "B_meets_need (yes/no)", "materially_better (A/B/neither)", "notes"])
        for n, i in enumerate(picked, 1):
            d = data[i]
            cheap_first = rng.random() < 0.5
            a, b = (answers[(CHEAP, i)], answers[(STRONG, i)]) if cheap_first else (answers[(STRONG, i)], answers[(CHEAP, i)])
            item = f"R{n:02d}"
            key[item] = {"id": i, "A": "cheap" if cheap_first else "strong"}
            w.writerow([item, LABEL[d["category"]], d["prompt"], criteria(d.get("reference") or {}), a, b, "", "", "", ""])
    KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote {SHEET.relative_to(ROOT)} ({len(picked)} items) and the hidden key {KEY.relative_to(ROOT)}")
    print(Counter(data[k['id']]['category'] for k in key.values()))


def _yn(v):
    v = (v or "").strip().lower()
    return True if v in YES else False if v in NO else None


def _ab(v):
    v = (v or "").strip().lower()
    return {"a": "A", "b": "B", "neither": "neither", "none": "neither", "same": "neither", "n": "neither"}.get(v)


def kappa(pairs):
    """Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score():
    key = json.loads(KEY.read_text(encoding="utf-8"))
    grades = {g["id"]: g for g in load_jsonl(ROOT / "results" / "public" / "hybrid_grades.jsonl")
              if g.get("kind") == "judge" and g.get("cheap_model") == CHEAP and g.get("check", "") == ""
              and g.get("judge_version") == "v2-meets-need"}
    data = {d["id"]: d for d in load_jsonl(ROOT / "hybrid" / "dataset.jsonl")}
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    meets, better, by_cat, skipped, disagreements = [], [], defaultdict(list), [], []
    for r in rows:
        k = key[r["item"]]
        g = grades[k["id"]]
        hA, hB, hbetter = _yn(r["A_meets_need (yes/no)"]), _yn(r["B_meets_need (yes/no)"]), _ab(r["materially_better (A/B/neither)"])
        if hA is None or hB is None or hbetter is None:
            skipped.append(r["item"]); continue
        cheap_is_A = k["A"] == "cheap"
        h_cheap, h_strong = (hA, hB) if cheap_is_A else (hB, hA)
        h_pref = {"A": "cheap" if cheap_is_A else "strong", "B": "strong" if cheap_is_A else "cheap", "neither": "neither"}[hbetter]
        j_pref = {"cheap_better": "cheap", "strong_better": "strong"}.get(g["outcome"], "neither")
        meets += [(h_cheap, g["cheap_meets_need"]), (h_strong, g["strong_meets_need"])]
        better.append((h_pref, j_pref))
        same = h_cheap == g["cheap_meets_need"] and h_strong == g["strong_meets_need"] and h_pref == j_pref
        by_cat[data[k["id"]]["category"]].append(same)
        if not same:
            disagreements.append({"item": r["item"], "id": k["id"], "you": {"cheap_meets": h_cheap, "strong_meets": h_strong, "better": h_pref},
                                  "judge": {"cheap_meets": g["cheap_meets_need"], "strong_meets": g["strong_meets_need"], "better": j_pref},
                                  "judge_reason": g.get("reason", ""), "your_notes": r.get("notes", "")})
    n = len(better)
    if not n:
        sys.exit("No graded rows yet: fill in the three answer columns, then run this again.")
    # Does the judge favour the frontier model, compared with me?
    you_strong = sum(p == "strong" for p, _ in better) / n
    judge_strong = sum(p == "strong" for _, p in better) / n
    report = {
        "graded_items": n, "skipped_items": skipped,
        "meets_need_agreement": sum(a == b for a, b in meets) / len(meets), "meets_need_kappa": kappa(meets),
        "preference_agreement": sum(a == b for a, b in better) / n, "preference_kappa": kappa(better),
        "all_three_match": sum(v for vs in by_cat.values() for v in vs) / n,
        "share_frontier_better": {"you": you_strong, "judge": judge_strong},
        "cheap_meets_need_rate": {"you": sum(a for a, _ in meets[0::2]) / n, "judge": sum(b for _, b in meets[0::2]) / n},
        "by_kind_of_work": {LABEL[c]: {"items": len(v), "all_three_match": sum(v) / len(v)} for c, v in by_cat.items()},
        "disagreements": disagreements,
    }
    OUT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"{n} items graded ({len(skipped)} skipped)")
    print(f"Meets the need: you and the judge agree {report['meets_need_agreement']:.0%} (kappa {report['meets_need_kappa']:.2f})")
    print(f"Which is better: agree {report['preference_agreement']:.0%} (kappa {report['preference_kappa']:.2f})")
    print(f"Frontier answer materially better: you {you_strong:.0%}, judge {judge_strong:.0%}  (a big gap = the judge favours Opus)")
    print(f"Cheap answer meets the need: you {report['cheap_meets_need_rate']['you']:.0%}, judge {report['cheap_meets_need_rate']['judge']:.0%}")
    print(f"Disagreements written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "score"])
    args = ap.parse_args()
    make() if args.step == "make" else score()
