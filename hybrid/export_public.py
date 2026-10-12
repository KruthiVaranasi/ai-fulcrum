"""Write the public subset of the evaluation results to results/public/.

The full answer, grade and verification logs also contain rows from private slices
(my interview-prep notes and my own Claude Code sessions), so they stay out of git.
This keeps only rows for the 350 prompts in hybrid/dataset.jsonl, which are public
benchmark items or prompts written for this project. hybrid.decide and
hybrid.economics fall back to these files when the full logs are not present.

    python -m hybrid.export_public
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PUBLIC = RESULTS / "public"
FILES = ["hybrid_answers.jsonl", "hybrid_grades.jsonl", "hybrid_verify.jsonl"]


def public_ids():
    return {json.loads(l)["id"] for l in (ROOT / "hybrid" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    ids = public_ids()
    PUBLIC.mkdir(exist_ok=True)
    for name in FILES:
        src = RESULTS / name
        if not src.exists():
            print(f"skip {name}: not found")
            continue
        kept = dropped = 0
        with (PUBLIC / name).open("w", encoding="utf-8", newline="\n") as out:
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") in ids:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); kept += 1
                else:
                    dropped += 1
        print(f"{name}: kept {kept} public rows, left out {dropped} private rows")


if __name__ == "__main__":
    main()
