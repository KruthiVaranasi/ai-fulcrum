"""Build hybrid/dataset.jsonl: public benchmarks (with objective answers) + drafted real-world prompts.

    python -m hybrid.build_dataset

Public sources (Hugging Face datasets-server API), sampled with a fixed seed:
  GSM8K (MIT), HumanEval (MIT), MBPP (CC BY 4.0), MMLU (MIT), Dolly-15k (CC BY-SA 3.0).
"""

import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .drafted import RUBRICS, drafted

HERE = Path(__file__).parent
OUT = HERE / "dataset.jsonl"
API = "https://datasets-server.huggingface.co/rows"
SEED = 42

MMLU_GENERAL = ["global_facts", "high_school_geography", "miscellaneous", "nutrition", "management",
                "marketing", "high_school_world_history", "sociology", "computer_security", "high_school_statistics"]
# Benchmark labels verified by hand to be wrong. Keyed by the start of the question.
GOLD_FIXES = {
    # 430 (Alexandra) + 750 (Kim) + 300 (Sarah) + 700 (Maryam) = 2180; the dataset says 2280.
    "The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra": "2180",
}
DOLLY_SIMPLE ={"summarization", "information_extraction", "classification", "creative_writing", "brainstorming"}
DOLLY_QA = {"open_qa", "general_qa", "closed_qa"}


CACHE = HERE.parent / "data" / "hf_cache"  # raw pages, not committed


def rows(dataset, config, split, offset=0, length=100):
    """One page of rows, cached on disk so rebuilds are fast and immune to network hiccups."""
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    cache = CACHE / (hashlib.sha1(q.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                page = [x["row"] for x in json.load(r)["rows"]]
            CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(page), encoding="utf-8")
            return page
        except Exception:
            time.sleep(3 + attempt * 5)
    raise RuntimeError(f"Could not fetch {dataset}/{config}/{split} at offset {offset}")


def all_rows(dataset, config, split, total):
    out = []
    for off in range(0, total, 100):
        out += rows(dataset, config, split, off)
    return out


def mc_prompt(question, choices):
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
    return f"Answer with the letter of the correct option.\n\n{question}\n{opts}\nAnswer:"


def build():
    rng = random.Random(SEED)
    data = []

    # GSM8K: grade-school maths with a numeric answer.
    gsm = all_rows("openai/gsm8k", "main", "test", 1319)
    for r in rng.sample(gsm, 30):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        gold = next((v for k, v in GOLD_FIXES.items() if r["question"].startswith(k)), gold)
        data.append({"source": "gsm8k", "category": "reasoning", "persona": "student", "eval_type": "number",
                     "prompt": r["question"] + "\n\nGive the final answer on the last line as 'Answer: <number>'.",
                     "reference": {"answer": gold}})

    # HumanEval: complete a Python function; graded by unit tests.
    he = all_rows("openai/openai_humaneval", "openai_humaneval", "test", 164)
    for r in rng.sample(he, 25):
        data.append({"source": "humaneval", "category": "software", "persona": "software engineer", "eval_type": "tests",
                     "prompt": "Complete the following Python function. Return the complete function in a single "
                               "```python code block.\n\n```python\n" + r["prompt"] + "```",
                     "reference": {"tests": r["test"] + f"\n\ncheck({r['entry_point']})\n", "setup": "",
                                   "canonical": r["prompt"] + r["canonical_solution"]}})

    # MBPP: write a Python function from a description; graded by unit tests.
    mb = all_rows("google-research-datasets/mbpp", "full", "test", 500)
    for r in rng.sample(mb, 15):
        data.append({"source": "mbpp", "category": "software", "persona": "software engineer", "eval_type": "tests",
                     "prompt": f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}\n"
                               "Return the code in a single ```python code block.",
                     "reference": {"tests": "\n".join(r["test_list"]) + "\n", "setup": r.get("test_setup_code") or "",
                                   "canonical": r["code"]}})

    # MMLU machine learning: AI/ML knowledge with a gold letter.
    ml = all_rows("cais/mmlu", "machine_learning", "test", 112)
    for r in rng.sample(ml, 15):
        data.append({"source": "mmlu-machine_learning", "category": "aiml", "persona": "AI/ML engineer", "eval_type": "choice",
                     "prompt": mc_prompt(r["question"], r["choices"]), "reference": {"answer": "ABCD"[r["answer"]]}})

    # MMLU general subjects: 2 per subject.
    for subject in MMLU_GENERAL:
        sub = rows("cais/mmlu", subject, "test", 0, 100)
        for r in rng.sample(sub, 2):
            data.append({"source": f"mmlu-{subject}", "category": "general_qa", "persona": "general user", "eval_type": "choice",
                         "prompt": mc_prompt(r["question"], r["choices"]), "reference": {"answer": "ABCD"[r["answer"]]}})

    # Dolly-15k: human-written everyday instructions with a human reference response.
    simple, qa = [], []
    offsets = list(range(0, 15000, 100))
    rng.shuffle(offsets)
    for off in offsets:
        for r in rows("databricks/databricks-dolly-15k", "default", "train", off):
            if len(r["instruction"]) + len(r["context"]) > 2500:
                continue
            bucket = simple if r["category"] in DOLLY_SIMPLE else qa if r["category"] in DOLLY_QA else None
            if bucket is not None and len(bucket) < 20:
                bucket.append(r)
        if len(simple) >= 20 and len(qa) >= 20:
            break
    for bucket, cat in ((simple, "simple"), (qa, "general_qa")):
        for r in bucket:
            prompt = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
            data.append({"source": f"dolly-{r['category']}", "category": cat, "persona": "general user", "eval_type": "judge",
                         "prompt": prompt, "reference": {"reference_answer": r["response"], "rubric": RUBRICS[cat]}})

    # Drafted real-world prompts.
    for d in drafted():
        ref = {"rubric": RUBRICS[d["category"]]}
        if d["key_points"]:
            ref["key_points"] = d["key_points"]
        if d["max_words"]:
            ref["max_words"] = d["max_words"]
        data.append({"source": "drafted", "category": d["category"], "persona": d["persona"], "difficulty": d["difficulty"],
                     "eval_type": "judge", "prompt": d["prompt"], "reference": ref})

    for i, row in enumerate(data):
        slug = re.sub(r"[^a-z0-9]+", "-", row["source"].lower()).strip("-")
        row["id"] = f"{slug}-{i:04d}"
    return data


if __name__ == "__main__":
    data = build()
    with OUT.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"Wrote {len(data)} prompts to {OUT}")
    for cat, n in sorted(Counter(r["category"] for r in data).items()):
        print(f"  {cat:12} {n}")
    print("By grading:", dict(Counter(r["eval_type"] for r in data)))
