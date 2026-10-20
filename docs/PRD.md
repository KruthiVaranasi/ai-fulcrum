# PRD: LLM Routing Policy

**Decide, per business use case, when a cheaper model is good enough, and prove the quality held.**

| | |
|---|---|
| **Author** | Kruthi Varanasi |
| **Status** | v2: evidence complete (benchmark, nine business use cases, own-work shadow test); strategy recommended |
| **Last updated** | October 2026 |
| **Evidence** | [Strategy](STRATEGY.md) · [Decision records](DECISIONS.md) · [Offline results](../results/) |

---

## 0. What changed between v1 and v2

v1 of this PRD proposed a **smarter cost router**. Testing it on realistic work changed the product:

| v1 assumption | What the evidence showed | v2 decision |
|---|---|---|
| A per-prompt predictor can route most traffic cheaply | On realistic prompts, the prompt-only predictor was worse than guessing the average | Route by **use-case policy**; per-prompt prediction is not the core |
| Cheap models are fine for most requests | Fine for routine work (69–92% met the need), poor for judgement (9–13% as good as Opus on PM and interview coaching) | The **quality bar per use case** is a product decision |
| Cheap answers can be self-checked | A cheap model caught only 0–57% of its own mistakes; unit tests worked (64% saving, no quality loss) | Escalate only on **independent, objective checks** |
| Savings are broad and lasting | Savings concentrate in high-volume, low-stakes use cases and fall to 0% if frontier prices drop to a quarter | Position on **governance, verification and capacity**, not only cost |
| It would work on my own usage | Shadow test on 36 real requests: Haiku met my need 39–47%; questions were ~7% of my usage | For heavy users, sell **capacity and context hygiene**, not per-message routing |

The strategy, economics, go-to-market and launch plan are in [STRATEGY.md](STRATEGY.md). The sections below are kept from v1 for the problem framing and are superseded where they conflict with v2.

---

## 1. TL;DR

Most AI requests don't need a frontier model, but teams send everything to one because choosing per request is hard. On RouterBench (36,497 prompts, 11 models), a perfect router would beat GPT-4's quality at 7% of its cost. A **cost-aware router** I built keeps **95% of GPT-4's quality while cutting cost 65–79%** on familiar task types and **43–48%** on unseen ones. A live check on today's Claude models agreed (68% saving on 30 prompts). On my own 350-prompt evaluation set, though, predicting from the prompt alone saved only 8–9% (random: 7%); checking the cheap *answer* and escalating failures is what carried over (see README).

Two findings shape the product:
1. **Routers that learn "which prompts are hard" save calls, not money.** Hard prompts are also the long, expensive ones. Ranking by *expected quality gain per extra dollar* fixes this.
2. **Price per token isn't price per task.** In the live test, Claude Opus 5.5 (4× Haiku's token price) was *cheaper* per answer than Haiku with thinking on, because Opus adapts how much it thinks. The router must decide on cost per task, not list price, and "which effort level" may matter as much as "which model".

---

## 2. Problem

### Origin
I pay about ₹2,300 a month for AI subscriptions and regularly hit usage limits. My requests are a mix: complex, strategic work alongside smaller, simpler ones (rewrites, explanations, quick questions). Because I use the same frontier model for everything, the simple requests consume the same scarce capacity as the hard ones. Switching some of them to a smaller model needs confidence that it's as good as the frontier model for that kind of task, which is what this project tests.

### The bigger problem
- Companies building on LLM APIs pay per token. Defaulting to a frontier model for every request is the simplest engineering choice and often the most expensive.
- Choosing a model per request by hand doesn't scale, and simple rules don't work: in our benchmark, "send long prompts to the big model" performed **no better than random** (APGR 0.54 vs 0.50).
- Cost has become a product constraint. A feature can be loved and still lose money on every user (see the cost model in my AI PM notes: an all-frontier ₹499/month assistant runs at about −48% gross margin).

### What we know vs. what we're assuming

| Claim | Status |
|---|---|
| A large share of prompts can be served by cheaper models without quality loss | **Validated** on RouterBench: cheap open-source Yi-34B matches GPT-4 on 76% of prompts |
| A router can capture a meaningful share of that, using only the prompt | **Partly validated**: 65–79% saving at 95% of GPT-4 quality on RouterBench (43–48% on unseen task types), but only 8–9% on my 350 real-world prompts (random 7%) |
| The result transfers to today's models | **Partly validated**: 30-prompt live test on Claude Haiku vs Opus, 68% saving. Small sample |
| Teams with LLM API spend will adopt a router to save money | **Assumption**: needs customer interviews (Section 12) |
| Consumers would switch from subscriptions to pay-per-use + routing | **Assumption, probably weak**: see Section 4 |

---

## 3. Why AI (and what isn't AI)

The routing decision has to happen **before** any model answers, from the prompt alone, in well under a millisecond. Rules (by prompt length, keywords or task type) are the obvious non-AI option, and we tested one: it barely beat random. The input is free text with no structured "difficulty" field, which is exactly where a learned model helps.

But the router itself is deliberately **small**: TF-IDF features with logistic regression and a ridge regressor for predicted cost. Embeddings didn't beat it, so I stopped climbing (no fine-tuned LLM router). Measured overhead: it trains in under a second and decides in about 2–3 ms per request on a laptop CPU (p95 2.6–6.3 ms across three runs), with no API cost, which is negligible next to a 5–15 s model call. An LLM router (Haiku) was also tested: it adds about 4 s and $0.0016 per request and routed no better than random (5% saving vs 7%).

---

## 4. Who pays: consumer vs. B2B

| | Consumers (power users) | **Teams with LLM API spend (primary)** |
|---|---|---|
| How they pay today | Flat subscriptions (₹1,700–4,500/month per app) | Pay per token, often ₹ lakhs per month |
| Does routing save money? | Only if they **cancel subscriptions** and move to pay-per-use. Subscriptions bundle apps, voice, memory and file tools that APIs don't | Yes, directly: every routed request is cheaper |
| Willingness to pay | Low; price-sensitive; churn-prone | Clear ROI; a buyer exists (eng lead / finance) |
| Verdict | Free **savings calculator** as top-of-funnel and portfolio demo | **Core product** |

**Decision:** target teams building on LLM APIs (AI startups, SaaS companies with AI features, support automation). Keep the consumer calculator (`savings.py`) as a free, private tool that shows people their own numbers.

---

## 5. Users and jobs to be done

- **Primary: the engineering lead at a company with an AI feature.** "When our model bill grows faster than revenue, I want to cut cost without a quality regression I can't see, so I can keep the feature profitable."
- **Secondary: the finance/ops owner.** "I want a trustworthy number for what we saved and what quality we kept."
- **Tertiary: the power user** comparing subscriptions with pay-per-use.

---

## 6. Solution

### MVP (what the POC proves)
- **Cost-aware routing policy:** P(strong model needed) ÷ predicted extra cost, with a threshold calibrated to a **quality target** (e.g. "keep 95% of the strong model's quality").
- **Offline evaluation harness:** baselines (random, rule, oracle), in-distribution and unseen-task tests, cost-based metrics.
- **Live validation** on current models.
- **Demos:** `route.py` (explain a single routing decision) and `savings.py` (estimate savings from your own chat export, privately).

### Product (next)
- **Drop-in gateway / SDK** with an OpenAI-compatible endpoint, so adoption is a one-line change.
- **Calibration on the customer's own traffic:** shadow mode first (route nothing, measure what routing *would* have done), then turn it on gradually.
- **Quality guardrail:** sample routed requests, grade them against the strong model with a validated LLM judge, and automatically tighten the threshold if quality drops.
- **Escalation:** if the cheap model's answer looks uncertain, retry on the strong model (a cascade).
- **Effort routing:** choose the thinking/effort level, not just the model (Finding 2).
- **Savings dashboard:** money saved, quality retained, share routed, by feature.

### Non-goals (for now)
Training our own models; hosting models; replacing provider SDKs beyond routing.

---

## 7. Success metrics

| Layer | Metric | Target | Current evidence |
|---|---|---|---|
| **North star** | Money saved per customer per month, at or above their quality target | — | — |
| Model quality | Quality retained vs. all-strong | ≥ 95% | 95% target met on benchmark |
| Model quality | Cost saving at the quality target | ≥ 40% on real traffic | Benchmark: 65–79% in-distribution, 43–48% unseen; live 68% (n=30); real-world 350 prompts: 8–9% from the prompt alone, 58% with verify-then-escalate on internal productivity |
| Model quality | Cost-based APGR | ≫ 0.5 (random) | 0.79–0.83 in-distribution; 0.70 unseen |
| Experience | Added latency per request | < 5 ms | 1.8–2.8 ms median, 2.6–6.3 ms p95 across three runs (laptop CPU); met at the median, borderline at p95 under load |
| Trust | Share of routed requests sampled and graded | ≥ 2% | Not built |
| **Guardrail** | Quality regressions reaching users | 0 unnoticed | Requires the guardrail (Section 6) |

---

## 8. Competitive landscape

| | What they offer | Our angle |
|---|---|---|
| **OpenRouter** | Marketplace for many models, with an auto-routing option | We focus on *proving* savings and quality on the customer's own traffic |
| **Martian** | Commercial model router (authors of RouterBench) | Same space; our cost-aware finding is the kind of insight to compete on |
| **Not Diamond** | Router that predicts the best model per prompt | Optimising for "best model" ≠ optimising for cost at a quality target |
| **RouteLLM (LMSYS)** | Open-source routing framework and research | Strong baseline; its standard metric (APGR) counts calls, which hides cost (Finding 1) |
| **Model providers** | Model families with effort/thinking controls; some products route internally | The biggest threat: adaptive models make "cheaper model" routing less valuable (Section 10) |

The space is crowded. A credible wedge is **cost-aware, quality-guaranteed routing with transparent savings reporting**, calibrated on the customer's own traffic.

---

## 9. Business model and sizing

- **Pricing:** start with a **share of measured savings** (e.g. 20%): no savings, no fee, which removes buying friction. Move to a platform fee plus usage once observability and guardrails add value beyond savings.
- **Unit economics:** the router costs a few milliseconds of CPU per request, so gross margin on the routing itself is very high. The real costs are the quality-grading samples (LLM judge calls) and support.
- **Illustrative sizing (assumptions, not market data):** 2,000 companies in India spending ₹5 lakh/month on LLM APIs; 30% average saving → ₹1.5 lakh/month saved each; 20% fee → ₹30,000/month per customer → ₹72 crore/year if all adopted. 5% adoption ≈ ₹3.6 crore/year. Worth validating, not yet venture-scale on India alone.

---

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Providers make routing less valuable** (adaptive thinking, effort settings, falling prices). Our own live test showed Opus 5.5 cheaper than Haiku-with-thinking on easy tasks | High | Route effort levels and modes, not just models; sell trust and reporting, not just price arbitrage |
| **Distribution shift:** savings drop on unfamiliar traffic (65–79% → 43–48% in our test) | High | Shadow-mode calibration on customer traffic; continuous sampled grading |
| **Hidden quality regressions** erode trust | High | Quality guardrail with a validated judge; auto-tighten threshold; per-feature reporting |
| **Privacy:** a gateway sees every prompt | Medium | Self-hosted SDK option; no prompt storage by default |
| **Training data age:** RouterBench models are from 2023 | Medium | Retrain on fresh labelled traffic; live checks on current models |
| **Wrong cheap tier:** a weak model fails unpredictably, so routing can't help (Qwen 2.5 3B live test) | Medium | Qualify the cheap tier first (accuracy and failure clustering); offer cascades for weak tiers |
| **Benchmark ≠ real work:** RouterBench coding tasks are simple, so the router sends code to cheap models | Medium | Per-task guardrails; customer-specific calibration |

---

## 11. Key decisions and trade-offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Router model | TF-IDF + logistic regression + ridge cost model | Embeddings, fine-tuned LLM | Embeddings didn't beat it; an LLM router (Haiku) added ~4 s and $0.0016 per request and routed no better than random |
| Ranking objective | Gain per extra dollar | P(strong needed) | Plain classifier was no better than random on cost |
| Decision rule | Threshold calibrated to a quality target | Fixed traffic share | Customers think in quality guarantees, not traffic shares |
| Live test models | Claude via subscription + local open-source | Paid APIs | ₹0 spend; results labelled as "equivalent API cost" |

---

## 12. Validation plan

1. **8–10 interviews** with engineering leads at companies with LLM spend: how they choose models, monthly spend, what quality evidence would make them trust routing. *Kill criterion:* if fewer than half see model cost as a top-3 problem, reposition as a free open-source tool rather than a product.
2. **Shadow-mode pilot** with 2 design partners: measure saving at their quality target on real traffic.
3. **Effort-routing experiment:** compare "route between models" with "route between effort levels of one model" on cost per task.

---

## 13. Roadmap

| Now | Next | Later |
|---|---|---|
| Publish POC, results and calculator | Gateway/SDK with shadow mode | Effort routing; cascades |
| Customer interviews | Quality guardrail with validated judge | Multi-provider routing; savings dashboard |
| Retrain on fresher labelled data | Design-partner pilots | Self-hosted enterprise version |

---

## Appendix: results

**Offline (RouterBench, 7,300 unseen prompts per test; 95% of GPT-4 quality):**

| Router | Saving (cheap = Mixtral) | Saving (cheap = Yi-34B) | Unseen task types (Mixtral / Yi) |
|---|---|---|---|
| Random | 16% | 32% | 19% / 16% |
| Classifier ("hard prompts") | 16% | 29% | 14% / 27% |
| **Cost-aware router** | **65%** | **79%** | **48% / 43%** |
| Oracle (ceiling) | 84% | 89% | 84% / 78% |

**Live (30 multiple-choice prompts, October 2026; equivalent API cost at list prices, nothing billed):**

| Model | Accuracy | Avg latency | Equivalent cost |
|---|---|---|---|
| Claude Haiku 4.5, thinking on | 93.3% | 11.4 s | $0.139 |
| Claude Haiku 4.5, thinking off | 86.7% | 5.0 s | $0.025 |
| Claude Opus 5.5 | 100% | 7.6 s | $0.129 |

| Qwen 2.5 3B (local, open-source) | 56.7% | 3.7 s | free |

Routing Haiku (thinking off) ↔ Opus: 95% of Opus quality at 16% Opus calls, **68% saving** (random 46%, oracle 71%).

Routing Qwen 2.5 3B ↔ Opus: **no router beat random** (cost-APGR 0.34–0.50 vs. 0.51; oracle needs Opus for 39% of prompts). Qwen's errors were spread across topics, so the prompt carried no signal. Product implication: choose a cheap tier that is competent and fails predictably; use cascades (answer, check confidence, escalate) for weak models; calibrate per model pair.
