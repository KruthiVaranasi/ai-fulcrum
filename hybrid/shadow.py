"""Shadow test on my own Claude Code work: would a smaller model have done as well?

    python -m hybrid.shadow          # after run_models + grade --judge on the own_sessions category

Opus 5.5 did the real work across two projects; its replies were accepted and built on, so they are the
reference. The question-style requests (no tools) were replayed on smaller models and judged blind
against Opus's replies.

Fair cost comparison: Opus's recorded cost includes re-reading the whole conversation. The small
models only saw a short excerpt, so their measured cost would flatter them. Savings are therefore
estimated as "the small model in the same context": Opus's actual token counts priced at the small
model's rates.
"""

import json
from pathlib import Path

import numpy as np

from .decide import INR_PER_USD
from .grade import load_grades
from .grading import routing_scores
from .run_models import _session_rows

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# $ per million tokens: fresh input, cache write, cache read, output (list prices, Oct 2026)
PRICES = {"claude/opus": (4.0, 5.0, 0.20, 20.0), "claude/haiku-nothink": (1.0, 1.25, 0.10, 5.0),
          "ollama/qwen2.5:3b": (0.0, 0.0, 0.0, 0.0)}


def same_context_cost(tokens, model):
    p = PRICES[model]
    return (tokens["fresh_input"] * p[0] + tokens["cache_write"] * p[1] + tokens["cache_read"] * p[2]
            + tokens["output"] * p[3]) / 1e6


VARIANTS = {"own_sessions": "short excerpt (~700 chars of the previous reply)",
            "own_sessions_ctx": "recent conversation (~12,000 chars, both sides)"}


def main():
    grades = load_grades()
    report = {}
    for cat, label in VARIANTS.items():
        rows = {r["id"]: r for r in _session_rows() if r["category"] == cat}
        report[cat] = analyse(rows, grades, label)
    (RESULTS / "shadow_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def analyse(rows, grades, label):
    all_rows = [json.loads(l) for l in (ROOT / "data" / "session_requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    total_usage = sum(r["opus_equiv_cost_usd"] for r in all_rows)
    out = {"n_requests_total": len(all_rows), "n_comparable": len(rows),
           "comparable_share_of_usage": sum(r["opus_equiv_cost_usd"] for r in rows.values()) / total_usage, "models": {}}

    for cm in ("claude/haiku-nothink", "ollama/qwen2.5:3b"):
        judged = [(rows[i], grades[("judge", i, cm, "")]) for i in rows if ("judge", i, cm, "") in grades]
        if not judged:
            continue
        meets = np.array([g["cheap_meets_need"] for _, g in judged])
        # 'as good' requires the small model to meet the need (both failing is not a win)
        as_good = np.array([g["cheap_meets_need"] and g["outcome"] != "strong_better" for _, g in judged])
        opus_cost = np.array([r["opus_equiv_cost_usd"] for r, _ in judged])
        small_cost = np.array([same_context_cost(r["opus_tokens"], cm) for r, _ in judged])
        # Hindsight (oracle): use the small model exactly where it was good enough.
        res = {"n_judged": len(judged), "meets_need": float(meets.mean()), "as_good_as_opus": float(as_good.mean())}
        for bar, ok in (("acceptable", meets), ("best", as_good)):
            saved = float(((opus_cost - small_cost) * ok).sum())
            res[f"hindsight_saving_{bar}"] = {"share_of_these_requests": saved / opus_cost.sum(),
                                              "share_of_all_my_usage": saved / total_usage}
        # Realistic, no hindsight: send every question-style request to the small model.
        res["route_all_questions"] = {"quality_acceptable": float(meets.mean()), "quality_best": float(as_good.mean()),
                                      "share_of_these_requests_saved": float((opus_cost - small_cost).sum() / opus_cost.sum()),
                                      "share_of_all_my_usage_saved": float((opus_cost - small_cost).sum() / total_usage)}
        res["examples_not_good_enough"] = [{"request": r["prompt"].split("\n\n")[-1][:140], "reason": g["reason"]}
                                           for r, g in judged if not g["cheap_meets_need"]][:6]
        out["models"][cm] = res

    out["context_given"] = label
    print(f"\n##### Context given to the small model: {label}")
    print(f"{out['n_requests_total']} real requests; {out['n_comparable']} question-style ones are comparable "
          f"({out['comparable_share_of_usage']:.0%} of my usage).")
    for cm, r in out["models"].items():
        print(f"\n{cm} (n={r['n_judged']}): meets the need {r['meets_need']:.0%}, as good as Opus {r['as_good_as_opus']:.0%}")
        for bar in ("acceptable", "best"):
            h = r[f"hindsight_saving_{bar}"]
            print(f"  hindsight saving ({bar} bar): {h['share_of_these_requests']:.0%} of these requests' cost "
                  f"= {h['share_of_all_my_usage']:.1%} of all my usage")
        ra = r["route_all_questions"]
        print(f"  route all questions to it: quality {ra['quality_acceptable']:.0%} acceptable / {ra['quality_best']:.0%} best; "
              f"saves {ra['share_of_these_requests_saved']:.0%} of these = {ra['share_of_all_my_usage_saved']:.1%} of all usage")
    return out


if __name__ == "__main__":
    main()
