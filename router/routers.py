"""Routers: each looks only at the prompt and returns P(the strong model is needed).

Ordered from cheapest to most complex, following the "climb the ladder only when the evals
say so" rule. Every router exposes fit(texts, y) and predict(texts) -> probabilities.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .data import DATA_DIR


class RandomRouter:
    """Baseline: routes a random share of prompts to the strong model."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, texts, y):
        return self

    def predict(self, texts) -> np.ndarray:
        return self.rng.random(len(texts))


class LengthRouter:
    """Rule-based baseline: longer prompts go to the strong model first."""

    name = "Prompt length (rule)"

    def fit(self, texts, y):
        return self

    def predict(self, texts) -> np.ndarray:
        return np.array([len(t) for t in texts], dtype=float)


class TfidfRouter:
    """Bag-of-words features + logistic regression. Trains in seconds, routes in microseconds."""

    name = "TF-IDF + logistic regression"

    def __init__(self):
        self.vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(C=1.0, max_iter=2000)

    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def predict(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


class CostAwareRouter:
    """Ranks prompts by expected quality gain per extra dollar: P(strong needed) / predicted extra cost.

    A plain classifier sends the hardest prompts to the strong model, but hard prompts are often
    the long, expensive ones. Dividing by a predicted cost difference (learned from the prompt alone,
    since output length is unknown before calling a model) spends the budget where it buys the most.
    """

    name = "Cost-aware (TF-IDF, gain per $)"
    uses_cost = True

    def __init__(self):
        from sklearn.linear_model import Ridge

        self.vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(C=1.0, max_iter=2000)
        self.reg = Ridge(alpha=1.0)

    def fit(self, texts, y, extra_cost):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, y)
        self.reg.fit(X, np.log(np.clip(extra_cost, 1e-7, None)))
        return self

    def predict(self, texts) -> np.ndarray:
        X = self.vec.transform(texts)
        return self.clf.predict_proba(X)[:, 1] / np.exp(self.reg.predict(X))


class Embedder:
    """Sentence embeddings with a small open model (all-MiniLM-L6-v2, 384-d), cached to disk."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def encode(self, texts, cache_name: str = None) -> np.ndarray:
        cache = DATA_DIR / f"emb_{cache_name}.npy" if cache_name else None
        if cache and cache.exists():
            emb = np.load(cache)
            if len(emb) == len(texts):
                return emb
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._model.max_seq_length = 256
        emb = self._model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        if cache:
            np.save(cache, emb)
        return emb


class EmbeddingLRRouter:
    """Prompt embedding + logistic regression."""

    name = "Embeddings + logistic regression"
    uses_embeddings = True

    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=2000)

    def fit(self, emb, y):
        self.clf.fit(emb, y)
        return self

    def predict(self, emb) -> np.ndarray:
        return self.clf.predict_proba(emb)[:, 1]


class EmbeddingKNNRouter:
    """Nearest neighbours: 'how often did the strong model win on similar prompts?'"""

    name = "Embeddings + k-nearest neighbours"
    uses_embeddings = True

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, emb, y):
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(emb)
        self.y = np.asarray(y, dtype=float)
        return self

    def predict(self, emb) -> np.ndarray:
        _, idx = self.nn.kneighbors(emb)
        return self.y[idx].mean(axis=1)
