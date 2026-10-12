"""Load RouterBench and turn it into router training data.

RouterBench (Martian, 2024) records, for ~36k prompts, the score and cost of 11 models.
Scores are in [0, 1] (most tasks are right/wrong; a few are graded).
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET = DATA_DIR / "routerbench_0shot.pkl"

CHEAP = "mistralai/mixtral-8x7b-chat"  # open-source tier
STRONG = "gpt-4-1106-preview"  # frontier tier


def clean_prompt(prompt) -> str:
    """Some prompts are stored as the string form of a list of message parts."""
    if isinstance(prompt, str) and prompt.startswith("["):
        try:
            parts = ast.literal_eval(prompt)
            if isinstance(parts, list):
                return " ".join(str(p) for p in parts)
        except (ValueError, SyntaxError):
            pass
    return str(prompt)


@dataclass
class Bench:
    prompts: list
    tasks: np.ndarray  # eval_name per prompt
    models: list
    scores: np.ndarray  # (N, M)
    costs: np.ndarray  # (N, M) in USD

    def col(self, model: str) -> int:
        return self.models.index(model)

    def score(self, model: str) -> np.ndarray:
        return self.scores[:, self.col(model)]

    def cost(self, model: str) -> np.ndarray:
        return self.costs[:, self.col(model)]

    def subset(self, idx: np.ndarray) -> "Bench":
        return Bench([self.prompts[i] for i in idx], self.tasks[idx], self.models, self.scores[idx], self.costs[idx])

    def __len__(self) -> int:
        return len(self.prompts)


def load(path: Path = DATASET) -> Bench:
    if not path.exists():
        raise SystemExit(f"RouterBench data not found at {path}. Run `python download_data.py` first (about 100 MB, once).")
    df = pd.read_pickle(path)
    models = [c for c in df.columns if c + "|total_cost" in df.columns]
    return Bench(
        prompts=[clean_prompt(p) for p in df["prompt"]],
        tasks=df["eval_name"].to_numpy(),
        models=models,
        scores=df[models].astype(float).to_numpy(),
        costs=df[[m + "|total_cost" for m in models]].astype(float).to_numpy(),
    )


def needs_strong(bench: Bench, cheap: str = CHEAP, strong: str = STRONG) -> np.ndarray:
    """Training label: 1 if the strong model scores higher than the cheap one on this prompt."""
    return (bench.score(strong) > bench.score(cheap)).astype(int)


def random_split(n: int, test_frac: float = 0.2, seed: int = 0):
    """In-distribution split: test prompts come from the same tasks seen in training."""
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - test_frac))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def task_split(tasks: np.ndarray, holdout_frac: float = 0.2, seed: int = 0):
    """Out-of-distribution split: whole tasks are held out, so the router meets new kinds of prompts."""
    names = np.unique(tasks)
    np.random.default_rng(seed).shuffle(names)
    held, count = [], 0
    for name in names:
        if count >= holdout_frac * len(tasks):
            break
        if (tasks == name).sum() > 0.5 * len(tasks) * holdout_frac:
            continue  # skip giant tasks (e.g. hellaswag) so one task doesn't dominate the test set
        held.append(name)
        count += (tasks == name).sum()
    test = np.isin(tasks, held)
    return np.where(~test)[0], np.where(test)[0], sorted(held)
