"""Charts for the README and docs/EXPERIMENTS.md, computed from the stored results.

    python -m hybrid.charts          # writes results/charts/*.png

Uses the 350 public prompts only (the same basis as docs/DECISIONS.md), the RouterBench summaries,
the task-completion runs, the overhead measurements and the personal-usage aggregates.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from router.evaluate import curve

from .decide import INR_PER_USD, item_rows, out_of_fold_predictions, verifier_single_cost_inr
from .task_specs import TASKS

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "charts"
CHEAP = "claude/haiku-nothink"

INK, MUTED, GRID, BG = "#1b1b1b", "#6b6b6b", "#e6e1d8", "#fffdf8"
GREEN, BLUE, ORANGE, PURPLE, RED, GREY, GOLD = "#1f7a4d", "#2f6db5", "#d9822b", "#7a4fb3", "#c0392b", "#9a9a9a", "#b8912f"
CATEGORY_LABELS = {"simple": "Rewrites & simple tasks", "general_qa": "Knowledge Q&A", "reasoning": "Maths & reasoning",
                   "software": "Coding", "aiml": "ML engineering", "pm_explain": "PM explanations",
                   "pm_decide": "PM decisions", "interview": "Interview prep"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c2b6", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


# ---------------------------------------------------------------- the headline line chart

def saving_curve(cv, targets):
    """Best cost saving vs all-Opus that reaches each quality target (as a share of Opus quality)."""
    rel_q, saving = cv.quality / cv.q_strong, 1 - cv.cost / cv.c_strong
    return np.array([saving[rel_q >= t - 1e-9].max() if (rel_q >= t - 1e-9).any() else np.nan for t in targets])


def category_scores(rows, bar, folds=5):
    """Out-of-fold P(cheap is bad) per category: the 'route by task type' rule."""
    cats = np.array([r["category"] for r in rows])
    bad = np.array([0.0 if r["good"][bar][0] else 1.0 for r in rows])
    out = np.zeros(len(rows))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(rows):
        rate = {c: bad[tr][cats[tr] == c].mean() for c in set(cats[tr])}
        out[te] = [rate.get(c, bad[tr].mean()) for c in cats[te]]
    return out + np.random.default_rng(0).random(len(rows)) * 1e-6  # break ties randomly, not by file order


def headline_chart(rows):
    llm = {r["id"]: r for r in load_jsonl(RESULTS / "hybrid_llm_router.jsonl")}
    single_check = verifier_single_cost_inr()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"wspace": 0.16})
    panels = [("acceptable", "Quality bar: the answer meets the need"), ("best", "Quality bar: as good as Opus")]
    summary = {}
    for ax, (bar, title) in zip(axes, panels):
        sc = np.array([r["good"][bar][0] for r in rows], float)
        ss = np.array([r["good"][bar][1] for r in rows], float)
        cc = np.array([r["c_cheap"] for r in rows]); cs = np.array([r["c_strong"] for r in rows])
        p, dc = out_of_fold_predictions(rows, bar)
        q_cheap_rel = sc.mean() / ss.mean()
        lo = np.floor(q_cheap_rel * 50) / 50 - 0.01
        targets = np.linspace(lo, 1.0, 201)
        x = targets * 100

        # The oracle knows each outcome: it escalates only prompts Opus fixes, cheapest fixes first.
        methods = [("Oracle (perfect foresight, ceiling)", (ss - sc) / np.maximum(cs - cc, 1e-9), GOLD, ":", 2.2),
                   ("Prompt classifier, cost-aware", p / dc, GREEN, "-", 2.6),
                   ("Prompt classifier", p, BLUE, "-", 2.0),
                   ("Route by task type", category_scores(rows, bar), PURPLE, "-", 2.0)]
        curves = {}
        for name, score, color, ls, lw in methods:
            y = saving_curve(curve(name, score, sc, ss, cc, cs), targets) * 100
            curves[name] = y
            ax.plot(x, y, ls, color=color, lw=lw, label=name, zorder=3)
        # Random routing: quality and cost both move linearly with the share sent to Opus.
        share = np.clip((targets * ss.mean() - sc.mean()) / (ss.mean() - sc.mean()), 0, 1)
        y_rand = (1 - (cc.sum() + share * (cs.sum() - cc.sum())) / cs.sum()) * 100
        ax.plot(x, y_rand, "--", color=GREY, lw=1.8, label="Random routing (baseline)", zorder=2)
        ax.fill_between(x, y_rand, curves["Oracle (perfect foresight, ceiling)"], color=GOLD, alpha=0.07, zorder=1)

        # Single operating points.
        def point(q, c):
            return q / ss.mean() * 100, (1 - c / cs.sum()) * 100
        pts = {}
        pts["Always Haiku"] = point(sc.mean(), cc.sum())
        if llm and all(r["id"] in llm for r in rows):
            esc = np.array([llm[r["id"]]["needs_strong"] for r in rows])
            router_cost = np.array([llm[r["id"]]["cost_usd"] * INR_PER_USD for r in rows])
            q = np.where(esc, ss, sc).mean(); c = (np.where(esc, cs, cc) + router_cost).sum()
            pts["LLM router (Haiku decides), incl. its own cost"] = point(q, c)
        vrows = [i for i, r in enumerate(rows) if r.get("verified_ok") is not None]
        if len(vrows) == len(rows):
            esc = np.array([not r["verified_ok"] for r in rows])
            q = np.where(esc, ss, sc).mean()
            vc = np.array([r["verify_cost"] for r in rows])
            pts["Verify-then-escalate (checks batched)"] = point(q, (cc + vc + np.where(esc, cs, 0)).sum())
            if single_check:
                vc1 = np.where(vc > 0, single_check, 0.0)
                pts["Verify-then-escalate (one check per call)"] = point(q, (cc + vc1 + np.where(esc, cs, 0)).sum())
        styles = {"Always Haiku": ("o", INK, 70), "LLM router (Haiku decides), incl. its own cost": ("X", RED, 120),
                  "Verify-then-escalate (checks batched)": ("*", ORANGE, 300),
                  "Verify-then-escalate (one check per call)": ("*", "none", 300)}
        for name, (px, py) in pts.items():
            m, col, s = styles[name]
            if col == "none":
                ax.scatter(px, py, marker=m, s=s, facecolors=BG, edgecolors=ORANGE, linewidths=1.6, zorder=5, label=name)
            else:
                ax.scatter(px, py, marker=m, s=s, color=col, edgecolors=INK if m == "*" else None, linewidths=0.6, zorder=5, label=name)
        if "Verify-then-escalate (checks batched)" in pts and "Verify-then-escalate (one check per call)" in pts:
            (a, b), (c, d) = pts["Verify-then-escalate (checks batched)"], pts["Verify-then-escalate (one check per call)"]
            ax.plot([a, c], [b, d], color=ORANGE, lw=1, alpha=0.7, zorder=4)

        # Annotate the 95% target.
        if lo <= 0.95:
            i95 = np.argmin(abs(targets - 0.95))
            ax.axvline(95, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
            ax.text(95.3, 97, "95% of Opus quality", color=MUTED, fontsize=8.5, va="top")
            box = {"boxstyle": "round,pad=0.25", "fc": BG, "ec": "none", "alpha": 0.9}
            for name, col, dy in (("Oracle (perfect foresight, ceiling)", GOLD, 8), ("Prompt classifier, cost-aware", GREEN, 10)):
                v = curves[name][i95]
                if not np.isnan(v):
                    ax.scatter(95, v, s=28, color=col, zorder=6)
                    ax.annotate(f"{v:.0f}%", (95, v), xytext=(7, dy), textcoords="offset points", color=col,
                                fontsize=10, fontweight="bold", bbox=box, zorder=7)
            ax.scatter(95, y_rand[i95], s=28, color=GREY, zorder=6)
            ax.annotate(f"{y_rand[i95]:.0f}%", (95, y_rand[i95]), xytext=(7, -16), textcoords="offset points",
                        color=GREY, fontsize=10, fontweight="bold", bbox=box, zorder=7)
            summary[bar] = {k: float(v[i95]) for k, v in curves.items()} | {"Random": float(y_rand[i95])}
        summary.setdefault(bar, {})["points"] = {k: [float(a), float(b)] for k, (a, b) in pts.items()}

        ax.set_title(title, loc="left", color=INK, pad=10)
        ax.set_xlim(lo * 100, 101.5); ax.set_ylim(-2, 100)
        step = 2 if lo > 0.75 else 10
        ax.set_xticks(np.arange(np.ceil(lo * 100 / step) * step, 100.1, step))
        quiet = {"fontsize": 8.2, "color": MUTED, "zorder": 7}
        if "Always Haiku" in pts:
            px, py = pts["Always Haiku"]
            ax.annotate(f"Always Haiku: {py:.0f}% saving at {px:.0f}% quality", (px, py), xytext=(8, 6),
                        textcoords="offset points", **quiet)
        name = "LLM router (Haiku decides), incl. its own cost"
        if name in pts:
            px, py = pts[name]
            ax.annotate(f"LLM router, incl. its own cost:\n{py:.0f}% saving at {px:.0f}% quality", (px, py), xytext=(-12, -30),
                        textcoords="offset points", ha="right", **quiet)
        vb, vu = pts.get("Verify-then-escalate (checks batched)"), pts.get("Verify-then-escalate (one check per call)")
        if vb:
            extra = f"\n({vu[1]:.0f}% with one check per call)" if vu else ""
            ax.annotate(f"Verify-then-escalate\n{vb[1]:.0f}% saving at {vb[0]:.0f}% quality{extra}", vb,
                        xytext=(lo * 100 + (100 - lo * 100) * 0.04, 10), textcoords="data", va="center",
                        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.9, "shrinkB": 9}, fontsize=8.6,
                        color="#a35d16", fontweight="bold", zorder=7)
        ax.set_xlabel("Quality target (% of always-Opus quality)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("Cost saving vs sending everything to Opus")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("How much each routing method saves at a given quality target", x=0.125, ha="left",
                 fontsize=15, fontweight="bold", color=INK, y=1.02)
    fig.text(0.125, 0.955, f"{len(rows)} real-world prompts across 8 kinds of work · cheap model Claude Haiku 4.5, frontier "
             "model Claude Opus 5.5 · predictors scored out-of-fold · shaded = headroom a perfect router would add over random",
             fontsize=9, color=MUTED)
    save(fig, "savings_by_method.png")
    return summary


# ---------------------------------------------------------------- supporting charts

def donut(ax, counts, title, colors, total_label):
    labels, vals = zip(*counts)
    wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": BG, "linewidth": 2})
    ax.text(0, 0, total_label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_title(title, color=INK, pad=6)
    ax.legend(wedges, [f"{l}  {v} ({v / sum(vals):.0%})" for l, v in counts], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=8.6, ncol=1)


def dataset_mix():
    data = load_jsonl(ROOT / "hybrid" / "dataset.jsonl")
    cats = Counter(r["category"] for r in data)
    grading = Counter({"tests": "Unit tests (code)", "choice": "Answer key (multiple choice)", "number": "Answer key (numeric)",
                       "judge": "Blind AI judge"}[r["eval_type"]] for r in data)
    src = Counter("Written for this project" if r["source"] == "drafted" else
                  {"gsm": "GSM8K (maths)", "hum": "HumanEval (code)", "mbp": "MBPP (code)", "mml": "MMLU (knowledge)",
                   "dol": "Dolly (everyday tasks)"}[r["source"][:3]] for r in data)
    pal = [GREEN, BLUE, ORANGE, PURPLE, GOLD, RED, "#3c9d9b", GREY]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    donut(axes[0], [(CATEGORY_LABELS[k], v) for k, v in cats.most_common()], "Kind of work", pal, f"{len(data)}\nprompts")
    donut(axes[1], src.most_common(), "Where the prompts came from", pal, "5 public\nbenchmarks\n+ written")
    donut(axes[2], grading.most_common(), "How answers were graded", pal, f"{sum(v for k, v in grading.items() if 'judge' not in k)}\nobjective")
    fig.suptitle("The 350-prompt evaluation set", x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=0.93)
    save(fig, "dataset_mix.png")


def usage_pie():
    e = json.loads((RESULTS / "economics.json").read_text(encoding="utf-8"))["personal"]
    sh = e["share_of_usage"]; n = {k: v["n"] for k, v in e["by_kind"].items()}
    counts = [("Agentic build work (editing files, running commands)", sh["action"], n["action"]),
              ("Questions answered from the conversation", sh["text"], n["text"]),
              ("Look-ups (reading files to answer)", sh["lookup"], n["lookup"])]
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    wedges, _ = ax.pie([c[1] for c in counts], colors=[GREEN, BLUE, ORANGE], startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": BG, "linewidth": 2})
    ax.text(0, 0, f"{sh['action']:.0%}\nagentic", ha="center", va="center", fontsize=16, fontweight="bold", color=GREEN)
    ax.legend(wedges, [f"{l}: {s:.1%} of usage ({k} requests)" for l, s, k in counts], loc="upper center",
              bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=9)
    ax.set_title(f"Where my own Claude usage went: 4 days, 2 projects, ${e['equiv_usd_total']:.0f} at API prices",
                 color=INK, loc="left")
    save(fig, "own_usage.png")


def coding_tasks_pie():
    kinds = {"Parsing & validation": ["duration", "csv", "expr", "roman", "pointer", "shsplit", "template"],
             "Algorithms": ["ranges", "topo", "glob", "justify", "compress", "paths"],
             "Dates & scheduling": ["busdays", "cron", "rolldate"],
             "Business rules": ["ledger", "limiter", "words", "semver"]}
    assert sorted(sum(kinds.values(), [])) == sorted(t["id"] for t in TASKS)
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    hard = {t["id"] for t in TASKS if t.get("tier") == "hard"}
    vals = [len(v) for v in kinds.values()]
    wedges, _ = ax.pie(vals, colors=[GREEN, BLUE, ORANGE, PURPLE], startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": BG, "linewidth": 2})
    ax.text(0, 0, f"{len(TASKS)} tasks\n{sum(len(t['cases']) for t in TASKS)} tests", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.legend(wedges, [f"{k} ({len(v)}): {', '.join(v)}" + (f"  · hard: {', '.join(x for x in v if x in hard)}" if any(x in hard for x in v) else "")
                       for k, v in kinds.items()], loc="upper center", bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=8.4)
    ax.set_title("Task-completion test: the coding tasks", color=INK, loc="left")
    save(fig, "coding_tasks_mix.png")


def category_bars(rows):
    cats = [c for c in CATEGORY_LABELS if any(r["category"] == c for r in rows)]
    meets = [np.mean([r["good"]["acceptable"][0] for r in rows if r["category"] == c]) * 100 for c in cats]
    best = [np.mean([r["good"]["best"][0] for r in rows if r["category"] == c]) * 100 for c in cats]
    order = np.argsort(meets)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(cats))
    ax.barh(y + 0.2, [meets[i] for i in order], 0.38, color=GREEN, label="Meets the need")
    ax.barh(y - 0.2, [best[i] for i in order], 0.38, color=ORANGE, label="As good as Opus")
    for yy, i in zip(y, order):
        ax.text(meets[i] + 1, yy + 0.2, f"{meets[i]:.0f}%", va="center", fontsize=8.6, color=INK)
        ax.text(best[i] + 1, yy - 0.2, f"{best[i]:.0f}%", va="center", fontsize=8.6, color=INK)
    ax.set_yticks(y, [CATEGORY_LABELS[cats[i]] for i in order])
    ax.set_xlim(0, 108); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("How often the cheap model (Haiku 4.5) was good enough, by kind of work", loc="left", color=INK)
    save(fig, "cheap_model_by_category.png")


def benchmark_vs_real(summary):
    rb = {m: json.loads((RESULTS / f"offline_results_{m}.json").read_text(encoding="utf-8")) for m in ("mixtral-8x7b-chat", "Yi-34B-Chat")}
    key = "Cost-aware (TF-IDF, gain per $)"
    def s(m, split):
        return rb[m][split][key]["cost_saving_at_95pct_of_strong_quality"] * 100
    items = [("RouterBench\nfamiliar tasks\ncheap: Mixtral", s("mixtral-8x7b-chat", "in_distribution"), BLUE),
             ("RouterBench\nfamiliar tasks\ncheap: Yi-34B", s("Yi-34B-Chat", "in_distribution"), BLUE),
             ("RouterBench\nunseen tasks\ncheap: Mixtral", s("mixtral-8x7b-chat", "out_of_distribution"), "#7fa7d6"),
             ("RouterBench\nunseen tasks\ncheap: Yi-34B", s("Yi-34B-Chat", "out_of_distribution"), "#7fa7d6"),
             ("350 real prompts\nsame router\ncheap: Haiku", summary["best"]["Prompt classifier, cost-aware"], GREEN),
             ("350 real prompts\nrandom routing\ncheap: Haiku", summary["best"]["Random"], GREY)]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(items))
    bars = ax.bar(xs, [v for _, v, _ in items], color=[c for *_, c in items], width=0.62)
    for b, (_, v, _) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(xs, [l for l, *_ in items], fontsize=9)
    ax.text(0, -0.3, "Quality = a correct answer on RouterBench; as good as Opus on the 350 prompts.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.set_ylim(0, 95); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Cost saving at 95% of the strong model's quality: benchmark vs real prompts (prompt-only router)",
                 loc="left", color=INK)
    save(fig, "benchmark_vs_real.png")


def overhead_bars():
    ov = json.loads((RESULTS / "overhead.json").read_text(encoding="utf-8"))
    ans = load_jsonl(RESULTS / "public" / "hybrid_answers.jsonl")
    lat = lambda m: float(np.median([a["latency_s"] for a in ans if a["model"] == m]))
    cost = lambda m: float(np.mean([a["equiv_cost_usd"] for a in ans if a["model"] == m]))
    items = [("Prompt classifier (router)", ov["ml_router"]["decision_ms"]["median"] / 1000, 0.0, GREEN),
             ("LLM router (Haiku decides)", ov["llm_router_single"]["latency_s"]["median"], ov["llm_router_single"]["cost_usd"]["mean"], RED),
             ("Haiku checks an answer", ov["verifier_single"]["latency_s"]["median"], ov["verifier_single"]["cost_usd"]["mean"], ORANGE),
             ("For scale: a Haiku answer", lat("claude/haiku-nothink"), cost("claude/haiku-nothink"), GREY),
             ("For scale: an Opus answer", lat("claude/opus"), cost("claude/opus"), "#5d5d5d")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(items))[::-1]
    axes[0].barh(y, [i[1] for i in items], color=[i[3] for i in items])
    axes[0].set_xscale("log"); axes[0].set_xlim(0.001, 60)
    axes[0].set_xticks([0.001, 0.01, 0.1, 1, 10], ["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    axes[0].minorticks_off()
    for yy, i in zip(y, items):
        axes[0].text(i[1] * 1.15, yy, f"{i[1] * 1000:.1f} ms" if i[1] < 1 else f"{i[1]:.1f} s", va="center", fontsize=9)
    axes[0].set_yticks(y, [i[0] for i in items]); axes[0].set_title("Added time per request (median, log scale)", loc="left", color=INK)
    axes[1].barh(y, [i[2] * 1000 for i in items], color=[i[3] for i in items])
    for yy, i in zip(y, items):
        axes[1].text(i[2] * 1000 + 0.3, yy, "$0" if i[2] == 0 else f"${i[2] * 1000:.1f} per 1k", va="center", fontsize=9)
    axes[1].set_yticks(y, [i[0] for i in items]); axes[1].set_xlim(0, max(i[2] for i in items) * 1000 * 1.35)
    axes[1].set_title("Added cost per 1,000 requests (USD, list prices)", loc="left", color=INK)
    for ax in axes:
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    save(fig, "router_overhead.png")


def task_cost_bars():
    runs = {(r["model"], r["task"]): r for r in load_jsonl(RESULTS / "tasks_runs.jsonl")}
    ids = [t["id"] for t in TASKS if (CHEAP, t["id"]) in runs and ("claude/opus", t["id"]) in runs]
    if not ids:
        return
    hard = {t["id"] for t in TASKS if t.get("tier") == "hard"}
    c = lambda m, t: sum(a["cost_usd"] for a in runs[(m, t)]["attempts"]) * 1000
    ids.sort(key=lambda t: c("claude/opus", t))
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(ids))
    ax.bar(x - 0.2, [c(CHEAP, t) for t in ids], 0.4, color=GREEN, label="Haiku 4.5 (thinking off)")
    ax.bar(x + 0.2, [c("claude/opus", t) for t in ids], 0.4, color="#5d5d5d", label="Opus 5.5")
    for i, t in enumerate(ids):
        if c(CHEAP, t) > c("claude/opus", t):
            ax.annotate("Haiku cost more", (i - 0.2, c(CHEAP, t)), xytext=(-40, 10), textcoords="offset points",
                        fontsize=8.5, color=RED, arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8})
    ax.set_xticks(x, [t + (" ★" if t in hard else "") for t in ids], rotation=35, ha="right", fontsize=8.8)
    ax.set_ylabel("Cost to finish the task (USD per 1,000 tasks)")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    h = sum(c(CHEAP, t) for t in ids) / len(ids); o = sum(c("claude/opus", t) for t in ids) / len(ids)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Cost to finish each coding task (all solved on the first attempt) · average: Haiku \${h:.1f} vs Opus \${o:.1f} "
                 f"per 1k tasks ({o / h:.1f}×) · ★ = hard tier", loc="left", color=INK, fontsize=10.5)
    save(fig, "task_cost.png")


def per_request_vs_per_task():
    rep = json.loads((RESULTS / "tasks_report.json").read_text(encoding="utf-8"))
    ans = load_jsonl(RESULTS / "public" / "hybrid_answers.jsonl")
    cost = lambda m: float(np.mean([a["equiv_cost_usd"] for a in ans if a["model"] == m]))
    items = [("List price per token", 4.0), ("Cost per request\n(350 general prompts)", cost("claude/opus") / cost(CHEAP)),
             ("Cost per finished\ncoding task (20 tasks)", rep["per_task_ratio_opus_over_haiku"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(range(3), [v for _, v in items], color=[GREY, BLUE, GREEN], width=0.55)
    for b, (_, v) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}×", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.axhline(1, color=RED, lw=1, ls="--"); ax.text(2.33, 1.12, "same cost", color=RED, fontsize=8.5)
    ax.set_xticks(range(3), [l for l, _ in items]); ax.set_ylim(0, max(v for _, v in items) * 1.2)
    ax.set_ylabel("Opus cost ÷ Haiku cost"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("How much more Opus costs depends on what you count", loc="left", color=INK)
    save(fig, "price_gap.png")


def use_case_bars():
    ds = json.loads((RESULTS / "policy_decisions_haiku-nothink.json").read_text(encoding="utf-8"))
    colors = {"always_cheap": GREEN, "verify_cascade": ORANGE, "policy": BLUE, "always_strong": "#5d5d5d"}
    names = {"always_cheap": "Always the cheap model", "verify_cascade": "Answer cheaply, verify, escalate",
             "policy": "Predict from the prompt", "always_strong": "Always the frontier model"}
    rows = sorted(((d["use_case"], d["recommendation"]["strategy"],
                    max(0.0, 1 - d["recommendation"]["cost_per_1k"] / d["always_strong"]["cost_per_1k"]) * 100,
                    d["error_cost_inr"]) for d in ds), key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(rows))
    ax.barh(y, [max(r[2], 0.6) for r in rows], color=[colors[r[1]] for r in rows])
    for yy, r in zip(y, rows):
        ax.text(max(r[2], 0.6) + 1, yy, f"{r[2]:.0f}%  ·  bad answer costs ₹{r[3]:,.0f}", va="center", fontsize=8.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 115); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    handles = [matplotlib.patches.Patch(color=colors[k], label=names[k]) for k in colors if any(r[1] == k for r in rows)]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Recommended strategy and saving vs all-frontier, per business use case", loc="left", color=INK)
    save(fig, "use_case_savings.png")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = item_rows(CHEAP, include_real=False)
    summary = headline_chart(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    dataset_mix(); usage_pie(); coding_tasks_pie(); category_bars(rows); benchmark_vs_real(summary)
    overhead_bars(); task_cost_bars(); per_request_vs_per_task(); use_case_bars()


if __name__ == "__main__":
    main()
