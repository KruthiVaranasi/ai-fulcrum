# The AI Fulcrum

**When should an AI product use a cheaper model, and when isn't it worth the risk?** A decision per business use case, tested on 350 prompts and on my own real AI work.

Most teams frame model routing as "send easy prompts to a cheap model and save money". I treated it as a product decision instead: what does a bad answer cost this business, how good is good enough, and what does it really cost to get a task done? The answer is more useful than "routing saves X%".

## What I found

- **Savings are real but concentrated.** Only high-volume, low-stakes use cases (internal rewrites, a free consumer tier) should default to a cheap model. Because they carry most of the traffic, a company with nine AI use cases still saves **~66%** at today's prices. *Decision: validate the volume mix before promising savings.*
- **For judgement-heavy work, the frontier model is the rational choice.** A frontier answer costs about ₹1–3, so once a bad answer costs more than roughly ₹1–12, escalating wins. *Decision: default to the frontier model for anything customer-facing or high-stakes.*
- **Check the answer; don't guess from the prompt.** Code checked by its unit tests matched all-frontier quality at **64% lower cost**. Guessing from the prompt which requests need the big model barely beat random on real prompts (8–10% saving vs 7–8%), even though the same idea promised 65–79% on a public benchmark. *Decision: build routing around checks that can be verified, and don't trust benchmark savings until they're measured on your own traffic.*
- **Count the cost of a finished task, not of one request.** Opus is 4× Haiku's token price and 7.7× its cost per request, but only **1.7× per finished coding task**; on the hardest task the "cheap" model cost more. *Decision: compare models on cost per completed task.*
- **The arbitrage is shrinking.** If frontier prices fall to a quarter of today's, the best saving drops to **0%**. *Decision: the durable product is governance, verification and capacity, not price arbitrage.*

📄 **Read:** [Strategy & exec memo](docs/STRATEGY.md) · [Decisions per use case](docs/DECISIONS.md) · [Experiments](docs/EXPERIMENTS.md) · [PRD](docs/PRD.md)

![The threshold follows the business](results/policy_sensitivity.png)

## Why this exists

I pay about ₹2,300 a month for AI subscriptions and regularly hit usage limits. My requests are a real mix: complex, strategic work (product decisions, positioning, evaluation design) alongside smaller, simpler ones (rewrites, explanations, quick questions). Because I use the same frontier model for everything, the simple requests consume the same scarce capacity as the hard ones.

Switching to a smaller model only makes sense with **confidence that it's good enough for that kind of task**. Which kinds of task, at what risk, and how would a team decide? Companies face the same question at much larger scale on their API bills.

## The calls I made

The data can't make these decisions; a PM has to. Each one is stated openly in [`hybrid/policies.py`](hybrid/policies.py) so it can be argued with.

| Call | What I decided | Why |
|---|---|---|
| What "good enough" means | Two bars: the answer *meets the need*, or it's *as good as the frontier model* | A rewrite only needs to be usable; PM decision support needs the best answer |
| What a bad answer costs | A rupee figure per use case, from ₹2 (an employee retries) to ₹2,000 (a wrong product decision) | It turns "risk" into something you can weigh against API cost |
| The decision rule | Use the frontier model when *P(cheap answer is bad) × cost of a bad answer ≥ the extra cost* | The threshold then moves with the business stakes, not with a model leaderboard |
| Quality floor | A minimum share of good answers per use case (80–95%) | It's the input most likely to flip a decision (⚠️ below), so it must be set with stakeholders |
| Latency | A p95 budget per use case | Checking an answer adds about 5 seconds; for the internal assistant that breaks its 15-second budget, so it's flagged |
| Who judges quality | Answer keys and unit tests where possible; a blind AI judge otherwise, checked for consistency | My first judge design never called a tie, so I redesigned it before trusting it |

## Nine business use cases: the decisions

Every prompt was answered by Claude Opus 5.5, Claude Haiku 4.5 and Qwen 2.5 3B (free, on my laptop), then graded. The engine picks the strategy with the lowest *expected total cost*, API cost plus bad answers times their cost, among those that meet the floor.

| Use case | Cost of a bad answer | Floor | **Recommended** | Saving vs all-frontier |
|---|---|---|---|---|
| Internal productivity (rewrites, summaries) | ₹2 | 85% | Answer cheaply, verify, escalate failures | **58%** ⚠️ |
| Interview-prep app, free tier | ₹5 | 80% | Always the cheap model | **91%** ⚠️ |
| Customer-facing knowledge Q&A | ₹150 | 95% | Always frontier | 0% |
| Analytics and reasoning | ₹50 | 90% | Always frontier | 0% |
| Coding assistant | ₹80 | 90% | Always frontier (tests-based cascade: 64% saving on test-verifiable tasks) | 0% |
| ML engineering copilot | ₹400 | 95% | Always frontier | 0% |
| PM ramp-up tutor | ₹30 | 85% | Always frontier | 0% |
| PM decision support | ₹2,000 | 95% | Always frontier | 0% |
| Interview-prep app, paid tier | ₹60 | 95% | Always frontier | 0% |

⚠️ = **fragile**: the decision flips if the quality floor moves by 5 points.

Haiku usually *met the need* (69–92% depending on the work), but it was *as good as Opus* on PM decisions and interview coaching only **9–13%** of the time. The quality bar is the product decision. Full records, with every strategy's quality, cost, latency and confidence range: **[docs/DECISIONS.md](docs/DECISIONS.md)**.

## Business impact

| | Monthly cost (base case, 2.25M requests) |
|---|---|
| All-frontier | ₹48.8 lakh |
| Recommended policies | ₹16.4 lakh |
| **Net saving after running costs** | **₹31.2 lakh**, payback on a ~₹3.5 lakh build in days |

The savings come almost entirely from two high-volume, low-stakes use cases, and the case weakens as prices fall:

![Routing's value shrinks as frontier prices fall](results/economics_price_scenarios.png)

Strategy, build-vs-buy, go-to-market (the shadow test *is* the sales motion), launch plan and risks: **[docs/STRATEGY.md](docs/STRATEGY.md)**.

## What my own usage showed

I replayed my own work (111 real Claude Code tasks across two projects) to see whether routing would help someone like me:

- **93%** of my usage was agentic build work (editing files, running commands over long contexts); question-style requests were about 7%.
- On those questions, Haiku met my need 39–47% of the time and matched Opus 8%. My work is mostly judgement, so Opus was the right default.
- That 4-day sprint was worth about **$300** at API prices, 29–57× my monthly subscription. For heavy users, the subscription is the bargain; the lever is capacity. Keeping contexts short frees an estimated **+32%** more work per usage window, versus **+4%** from routing questions to a smaller model.

The session data is personal, so it stays on my machine; only these totals are published.

## Evidence and its limits

- **Data:** 350 prompts across eight kinds of work (public benchmarks with answer keys, plus 205 realistic prompts I wrote for PM, ML, interview-prep and software personas), a 36,000-prompt public benchmark (RouterBench), and my own sessions. Details: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- **Grading I trust:** unit tests and answer keys where possible; the AI judge gave the same verdict 93% of the time with answers swapped, and a second model agreed with it 87% of the time.
- **Limits:** about 35–50 prompts per use case, so results are directional (95% ranges in DECISIONS.md). Costs are list-price equivalents, not invoices. The business inputs are my assumptions, stated to be challenged. A human review sample is still pending. Multi-step agent work, where 93% of my own usage went, is the next thing to test.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest                      # includes privacy checks; no API calls needed

python -m hybrid.decide --public      # decisions per use case → docs/DECISIONS.md
python -m hybrid.economics            # business impact and price scenarios
python -m hybrid.charts               # charts in results/charts/
```

The public subset of answers and grades is in `results/public/`, so the decisions reproduce without any API calls. New answers are collected with Qwen 2.5 3B via [Ollama](https://ollama.com) and Claude via the Claude Code CLI; every Claude call needs `--yes`, runs are capped, and they stop on a usage-limit message. `tests/test_privacy.py` fails if a private row, secret, email address or local path ever reaches the repository.

## Credits

RouterBench (Martian, 2024); RouteLLM (LMSYS, 2024) for the APGR metric; GSM8K, HumanEval, MBPP, MMLU and Dolly-15k for public prompts (see `hybrid/README.md` for licences).

## License

MIT
