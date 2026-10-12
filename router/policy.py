"""A ready-to-use routing policy: the cost-aware router plus a calibrated decision threshold.

The threshold is chosen on held-out RouterBench prompts as the loosest cut-off that still keeps
quality at the target (default: 95% of the strong model's quality). Trained once, cached to disk.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .data import DATA_DIR, STRONG, load, random_split
from .evaluate import curve
from .routers import CostAwareRouter

CHEAP_PROXY = "zero-one-ai/Yi-34B-Chat"  # cheap tier used for training labels
POLICY_FILE = DATA_DIR / "policy.joblib"


@dataclass
class Decision:
    route: str  # "cheap" or "strong"
    p_strong_needed: float
    predicted_extra_cost: float  # USD, on RouterBench's price scale (relative signal only)
    score: float
    threshold: float


class Policy:
    def __init__(self, router: CostAwareRouter, threshold: float, target: float, share_strong: float):
        self.router = router
        self.threshold = threshold
        self.target = target
        self.share_strong = share_strong  # expected share of traffic sent to the strong model

    def decide(self, prompts) -> list:
        X = self.router.vec.transform(prompts)
        p = self.router.clf.predict_proba(X)[:, 1]
        extra = np.exp(self.router.reg.predict(X))
        score = p / extra
        return [
            Decision("strong" if s >= self.threshold else "cheap", float(pi), float(e), float(s), self.threshold)
            for pi, e, s in zip(p, extra, score)
        ]

    def save(self, path: Path = POLICY_FILE):
        joblib.dump(self, path)


def train_policy(target: float = 0.95, seed: int = 0) -> Policy:
    bench = load()
    tr, te = random_split(len(bench), seed=seed)
    y = (bench.score(STRONG) > bench.score(CHEAP_PROXY)).astype(int)
    extra = bench.cost(STRONG) - bench.cost(CHEAP_PROXY)
    texts = bench.prompts

    # Calibrate the threshold on held-out prompts.
    r = CostAwareRouter().fit([texts[i] for i in tr], y[tr], extra[tr])
    scores = r.predict([texts[i] for i in te])
    t = bench.subset(te)
    cv = curve("policy", scores, t.score(CHEAP_PROXY), t.score(STRONG), t.cost(CHEAP_PROXY), t.cost(STRONG))
    hit = np.where(cv.quality >= target * cv.q_strong)[0]
    share = float(cv.share_strong[hit[0]]) if len(hit) else 1.0
    k = int(round(share * len(scores)))
    threshold = float(np.sort(scores)[::-1][k - 1]) if k > 0 else float("inf")

    # Refit on all data for the shipped policy; keep the calibrated threshold.
    final = CostAwareRouter().fit(texts, y, extra)
    return Policy(final, threshold, target, share)


def load_policy() -> Policy:
    if POLICY_FILE.exists():
        return joblib.load(POLICY_FILE)
    policy = train_policy()
    policy.save()
    return policy
