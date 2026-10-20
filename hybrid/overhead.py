"""Router overhead: what deciding costs, per request, before any answer is generated.

    python -m hybrid.overhead                       # ML router timing only (local, free)
    python -m hybrid.overhead --yes --n 15          # plus single-request LLM-router and verifier calls

Three things sit in front of (or behind) the model call in this project:
  1. the ML router (TF-IDF + calibrated logistic regression + ridge cost model), timed on CPU;
  2. an LLM-as-router (Haiku, thinking off) deciding ONE request per call, as it would live;
  3. the verifier in verify-then-escalate (Haiku, thinking off) checking ONE answer per call.
The earlier runs batched 5-10 items per call to save quota, which hides per-request latency.
Only aggregate timings and costs are written (results/overhead.json); no prompt or answer text.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from router.backends import ClaudeCodeBackend

from .grading import MAX_PROMPT_CHARS, _clip
from .llm_router import INSTRUCTIONS as ROUTER_INSTRUCTIONS
from .run_models import load_answers
from .verify import CHEAP, INSTRUCTIONS as VERIFY_INSTRUCTIONS

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "overhead.json"


def public_items():
    return [json.loads(l) for l in (ROOT / "hybrid" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def stats(values):
    v = np.asarray(values, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()), "max": float(v.max())}


def time_ml_router(texts, repeats=3):
    """Per-request decision time of the router used in hybrid.decide, one prompt at a time."""
    rng = np.random.default_rng(0)
    y, extra = rng.integers(0, 2, len(texts)), rng.random(len(texts)) + 0.01  # labels don't affect speed
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20_000)
    X = vec.fit_transform(texts)
    clf = CalibratedClassifierCV(LogisticRegression(C=2.0, max_iter=2000), method="sigmoid", cv=3).fit(X, y)
    reg = Ridge(alpha=1.0).fit(X, np.log(extra))
    train_s = time.perf_counter() - t0
    ms = []
    for t in texts * repeats:
        a = time.perf_counter()
        x = vec.transform([t])
        _ = clf.predict_proba(x)[0, 1] * 150 >= np.exp(reg.predict(x)[0])
        ms.append((time.perf_counter() - a) * 1000)
    return {"train_s": train_s, "decision_ms": stats(ms), "cost_usd": 0.0}


def single_calls(prompts, label):
    backend = ClaudeCodeBackend("haiku", thinking=False)
    lat, cost = [], []
    for i, p in enumerate(prompts, 1):
        r = backend.ask(p)
        if "error" in r:
            print(f"  {label} {i}: error {r['error'][:120]}")
            if any(w in r["error"].lower() for w in ("limit", "rate", "quota", "overloaded")):
                break
            continue
        lat.append(r["latency_s"]); cost.append(r["equiv_cost_usd"])
        print(f"  {label} {i}/{len(prompts)}: {r['latency_s']:.1f} s")
    return {"latency_s": stats(lat), "cost_usd": stats(cost)} if lat else None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="also make live Claude calls (uses subscription quota)")
    ap.add_argument("--n", type=int, default=15, help="live calls per measurement")
    args = ap.parse_args()

    items = public_items()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out["ml_router"] = time_ml_router([it["prompt"] for it in items])
    d = out["ml_router"]["decision_ms"]
    print(f"ML router: median {d['median']:.2f} ms, p95 {d['p95']:.2f} ms per request (train {out['ml_router']['train_s']:.2f} s)")

    if args.yes:
        sample = [items[i] for i in np.random.default_rng(1).choice(len(items), args.n, replace=False)]
        out["llm_router_single"] = single_calls(
            [ROUTER_INSTRUCTIONS + f"\n### REQUEST 1\n{it['prompt'][:2000]}" for it in sample], "LLM router")
        answers = load_answers()
        checkable = [it for it in items if it["eval_type"] != "tests" and (CHEAP, it["id"]) in answers]
        vs = [checkable[i] for i in np.random.default_rng(2).choice(len(checkable), args.n, replace=False)]
        out["verifier_single"] = single_calls(
            [VERIFY_INSTRUCTIONS + f"\n=== ITEM 1 ===\nREQUEST:\n{_clip(it['prompt'], MAX_PROMPT_CHARS)}\n--- ANSWER ---\n"
             f"{_clip(answers[(CHEAP, it['id'])]['text'], 4000)}" for it in vs], "verifier")
        for k in ("llm_router_single", "verifier_single"):
            if out.get(k):
                print(f"{k}: median {out[k]['latency_s']['median']:.1f} s, p95 {out[k]['latency_s']['p95']:.1f} s, "
                      f"${out[k]['cost_usd']['mean']:.5f} per request")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
