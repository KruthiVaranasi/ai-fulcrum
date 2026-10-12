"""Turn a real interview-prep chat export into router prompts (kept local, never committed).

    python -m hybrid.build_real data/interview_prep.txt

The source is a ChatGPT conversation exported to text ("user" / "ChatGPT" turn markers). Each
substantive user turn becomes a prompt, verbatim (typos included), prefixed with a short excerpt of
the assistant message before it, because real chat turns are answered in context.

Output: data/real_prep.jsonl (data/ is gitignored; the text contains personal details).
"""

import json
import re
import sys
from pathlib import Path

from .drafted import RUBRICS

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "real_prep.jsonl"
CONTEXT_CHARS = 700
MIN_WORDS = 4  # drops "yes", "continue", "next please"


FILLER = re.compile(
    r"give me \d+ ?min|will start with|^hi[, ]|\bq\d+\b|answer (the )?above|anwer the above|"
    r"i want to 1st prepare|^https?://", re.I)


def has_request(text: str) -> bool:
    """Keep real requests, however terse ('now hybrid indexing pls'); drop only navigation and filler
    such as greetings, 'give me 10 min', 'answer Q7 then Q3' or a bare link."""
    return len(text.split()) >= 30 or not FILLER.search(text.strip())


def split_turns(lines):
    turns, role, buf = [], None, []
    for line in lines:
        s = line.strip()
        if s in ("user", "ChatGPT"):
            if role and buf:
                turns.append((role, "\n".join(buf).strip()))
            role, buf = s, []
        else:
            buf.append(line)
    if role and buf:
        turns.append((role, "\n".join(buf).strip()))
    return turns


def build(path: Path):
    raw = path.read_text(encoding="utf-8").splitlines()
    lines = [l[9:] if l.startswith("[Normal] ") else l for l in raw]
    turns = split_turns(lines)
    rows, last_assistant = [], ""
    for role, text in turns:
        if role == "ChatGPT":
            last_assistant = text
            continue
        if len(text.split()) < MIN_WORDS or not has_request(text):
            continue
        context = last_assistant[-CONTEXT_CHARS:].strip()
        prompt = (f"(Earlier in this conversation, the assistant said: \"...{context}\")\n\n{text}"
                  if context else text)
        rows.append({
            "id": f"real-{len(rows):04d}", "source": "real-interview-prep", "category": "real_interview",
            "persona": "ML engineer preparing for a big-tech interview loop", "eval_type": "judge",
            "prompt": prompt,
            "reference": {"rubric": "Is the answer technically correct and genuinely useful for someone preparing for "
                                    "an ML/LLM engineering interview: it answers what was asked (in context), "
                                    "corrects misconceptions, and is clear?"},
        })
    return rows


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "interview_prep.txt"
    rows = build(src)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} real prompts to {OUT} (gitignored)")
