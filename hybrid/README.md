# Hybrid dataset: real-world prompts for routing decisions

`dataset.jsonl` holds 350 prompts across eight kinds of work, built by `python -m hybrid.build_dataset` (deterministic, seed 42).

| Category | Prompts | Sources | Grading |
|---|---|---|---|
| `simple` (rewrites, summaries, emails) | 40 | 20 Dolly-15k + 20 drafted | AI judge vs. frontier answer |
| `general_qa` | 40 | 20 MMLU (10 subjects) + 20 Dolly-15k | Answer key (MMLU), AI judge (Dolly) |
| `reasoning` | 40 | 30 GSM8K + 10 drafted estimates | Answer key, AI judge |
| `software` | 50 | 25 HumanEval + 15 MBPP + 10 drafted | **Unit tests** (40), AI judge (10) |
| `aiml` | 50 | 15 MMLU machine learning + 35 drafted | Answer key, AI judge with answer key |
| `pm_explain` | 45 | Drafted | AI judge with answer key |
| `pm_decide` | 45 | Drafted | AI judge with PM rubric |
| `interview` | 40 | Drafted | AI judge with rubric |

**Drafted prompts** (`drafted.py`) were written with AI assistance to reflect realistic requests from different personas (PMs, ML engineers, software engineers, job seekers), at mixed difficulty, each with key points or a rubric for the judge.

**Label fix:** one GSM8K answer key is wrong (430 + 750 + 300 + 700 = 2,180, not 2,280). It's corrected in `build_dataset.py` (`GOLD_FIXES`) with a note.

## Public sources and licences

| Dataset | Licence |
|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | MIT |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | CC BY 4.0 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT |
| [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 |

## Private slices (never committed)

Two further slices are built locally into `data/` (gitignored) because they contain personal details:
- `build_real.py`: real questions from an ML-interview-prep conversation export.
- `build_sessions.py`: my own Claude Code requests and Opus's replies, with secrets and emails redacted, used for the shadow test.

Only aggregate results from these appear in the repository.

## Grading pipeline

1. `run_models.py` collects answers (cached; Claude calls capped and opt-in).
2. `grade.py` applies answer keys and unit tests, then a blind pairwise judge (`grading.py`, v2: "does it meet the need / is the other answer materially better?") with position-swap and second-judge checks.
3. `decide.py` turns the grades into decisions per use case; `economics.py` into business impact; `shadow.py` into the own-work shadow test.
