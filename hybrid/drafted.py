"""Hand-drafted prompts for the hybrid dataset.

Written with AI assistance to reflect realistic day-to-day usage by different personas, at mixed
difficulty. Each item has either key points (an answer key the judge checks against) or relies on
its category rubric. Swap in real prompts any time; keep the same fields.

Fields: category, persona, difficulty, prompt, key_points (optional list), max_words (optional).
"""

# Category rubrics used by the judge for pairwise comparison.
RUBRICS = {
    "simple": "Does the answer do exactly what was asked (meaning preserved, requested tone/format/length followed), "
              "with no errors or unnecessary additions?",
    "general_qa": "Is the answer correct, directly responsive and appropriately concise?",
    "reasoning": "Is the reasoning sound, are assumptions stated and sensible, and is the final answer clearly given?",
    "software": "Is the technical content correct, does it identify the real cause or right design, and is it actionable?",
    "aiml": "Is the technical content correct and practical for a working ML engineer, covering the key points?",
    "pm_explain": "Is the explanation accurate, clear for a PM without deep ML background, and does it connect to "
                  "product decisions (trade-offs, metrics, risks)?",
    "pm_decide": "Does the answer state assumptions, quantify where possible, weigh trade-offs, make a clear "
                 "recommendation, and give actionable next steps a PM could use?",
    "interview": "Is the advice or answer well structured, specific (not generic), realistic for the interview, and "
                 "actionable?",
    "real_interview": "Is the answer technically correct and genuinely useful for someone preparing for an ML/LLM "
                      "engineering interview: it answers what was asked (in context), corrects misconceptions, and is clear?",
}

D = []


def add(category, persona, difficulty, prompt, key_points=None, max_words=None):
    D.append({"category": category, "persona": persona, "difficulty": difficulty, "prompt": prompt,
              "key_points": key_points or [], "max_words": max_words})


# ---------------------------------------------------------------- 1. Simple workplace tasks (20)
P = "working professional"
add("simple", P, "easy", "Rewrite this to sound professional: 'hey, can u send me the deck by tmrw? need it for the client call'", max_words=40)
add("simple", P, "easy", "Make this sentence shorter without losing meaning: 'At this point in time we are not in a position to be able to confirm the delivery date of the order.'", max_words=20)
add("simple", P, "easy", "Turn these notes into 3 bullet points for a status update: login bug fixed, payments page still slow (investigating DB query), release moved to Friday because QA found 2 blockers.")
add("simple", P, "easy", "Write a polite two-sentence email declining a meeting on Thursday and suggesting Friday afternoon instead.", max_words=60)
add("simple", P, "easy", "Correct the grammar: 'Me and my team has completed the analysis and we will sharing the results tomorrow.'")
add("simple", P, "easy", "Convert this into a markdown table with columns Name, Role, City: Asha - Designer - Pune; Ravi - Engineer - Bengaluru; Meera - PM - Hyderabad.")
add("simple", P, "medium", "Rewrite this message so it's firm but friendly: 'You've missed the deadline three times now and it's affecting the whole team. This can't keep happening.'")
add("simple", P, "easy", "Translate to simple Hindi: 'The meeting has been moved to Monday at 11 am. Please bring your laptop.'")
add("simple", P, "medium", "Summarise in one sentence: 'Our Q3 revenue grew 12% year on year, driven mostly by the enterprise segment, while SMB revenue was flat. Churn increased slightly among SMB customers, mainly due to pricing changes in July. We plan to revisit SMB pricing in Q4.'", max_words=35)
add("simple", P, "easy", "Give me 5 subject lines for an email announcing a new office location in Bengaluru.")
add("simple", P, "easy", "Rewrite for a LinkedIn post in a warm tone, under 60 words: 'Completed 2 years at my company today. Learned a lot about requirements and stakeholders.'", max_words=60)
add("simple", P, "medium", "Extract all dates and amounts from this text as a list: 'Invoice #4521 issued on 3 March 2026 for ₹48,500, due on 2 April 2026. A late fee of ₹1,200 applies after 10 April 2026.'")
add("simple", P, "easy", "Make this Slack message more concise: 'Hi team, I just wanted to quickly check in and see if anyone has had a chance to look at the document I shared yesterday, because I would really appreciate any feedback you might have when you get a moment.'", max_words=30)
add("simple", P, "medium", "Rewrite this paragraph for a non-technical audience: 'We migrated the service from a monolith to microservices, which reduced p95 latency by 40% and allowed independent deployments.'")
add("simple", P, "easy", "Write a one-line out-of-office message for 14–18 October; urgent issues go to priya@company.com.")
add("simple", P, "medium", "Reorder these steps into a logical sequence and number them: test in staging, write the code, get code review, deploy to production, write unit tests, monitor dashboards.")
add("simple", P, "easy", "Change this to passive voice: 'The analytics team released the new dashboard last week.'")
add("simple", P, "medium", "Write a 3-sentence apology to a customer whose order arrived two days late, offering a 10% discount on the next order.", max_words=80)
add("simple", P, "easy", "Classify each feedback as positive, negative or neutral: 1) 'Delivery was quick!' 2) 'App crashed twice today.' 3) 'I used the coupon at checkout.'")
add("simple", P, "medium", "Turn this rambling note into a clear agenda with 4 items and time allocations for a 30-minute meeting: we need to talk about the pricing change, also the onboarding drop-off numbers, someone should demo the new search, and decide on the launch date.")

# ---------------------------------------------------------------- 3. Guesstimates and reasoning (10)
P = "student / job seeker"
add("reasoning", P, "medium", "Estimate how many food delivery orders are placed in Bengaluru on a typical day. Show your assumptions.",
    ["Uses a structured approach (population → households/users → order frequency)", "States assumptions explicitly with plausible numbers", "Arrives at a final estimate in a plausible range (roughly hundreds of thousands to low millions)", "Sanity-checks the result"])
add("reasoning", P, "medium", "Estimate the monthly cost of running an AI chatbot for 100,000 daily active users if each user sends 10 messages a day, each message is ~1,500 input and 300 output tokens, and the model costs $1 per million input tokens and $5 per million output tokens.",
    ["Computes messages per month (≈30 million)", "Input cost ≈ $45,000 and output cost ≈ $45,000 per month", "Total ≈ $90,000 per month", "Mentions levers to reduce cost (caching, shorter prompts, smaller models)"])
add("reasoning", P, "hard", "A product has 50,000 monthly active users, 4% convert to a ₹399/month plan, and monthly churn of paid users is 6%. What is the approximate customer lifetime value and monthly recurring revenue? Explain.",
    ["Paid users = 2,000; MRR ≈ ₹7.98 lakh", "Average lifetime ≈ 1/0.06 ≈ 16.7 months", "LTV ≈ ₹399 × 16.7 ≈ ₹6,650 (revenue basis), noting gross margin would reduce it", "States assumptions clearly"])
add("reasoning", P, "medium", "If an A/B test has a baseline conversion of 5% and I want to detect an increase to 5.5%, roughly how many users do I need per group? Explain the rule of thumb you use.",
    ["Uses n ≈ 16·p(1−p)/δ² (80% power, 5% significance) or equivalent", "p = 0.05, δ = 0.005 → ≈ 30,400 per group", "Explains that small effects need large samples", "Mentions running full weeks / not peeking"])
add("reasoning", P, "easy", "A meeting room fits 8 people. A team of 27 needs to meet in groups that are as equal as possible with the fewest rooms. How many rooms and how many people per room?",
    ["Needs 4 rooms (27/8 rounds up to 4)", "Groups of 7, 7, 7 and 6", "Clear reasoning"])
add("reasoning", P, "medium", "Three teams estimate a project at 6, 9 and 15 weeks. Using a PERT-style estimate where the middle value is most likely, what is the expected duration? Show the formula.",
    ["PERT: (optimistic + 4×most likely + pessimistic) / 6", "(6 + 36 + 15) / 6 = 9.5 weeks", "Notes it weights the most likely estimate"])
add("reasoning", P, "hard", "Estimate the market size (in ₹ per year) for an AI interview-practice app for Indian college students. Show a bottom-up calculation.",
    ["Bottom-up: number of relevant students × share who would pay × price × frequency", "States assumptions with plausible numbers", "Gives a final figure and distinguishes TAM/SAM/SOM or similar", "Names the most uncertain assumption"])
add("reasoning", P, "medium", "A server handles 200 requests per second at 60% CPU. Traffic is expected to grow 2.5× during a sale. How many servers do we need if we currently have 4 and want to stay under 70% CPU?",
    ["Per-server load scales: 60% × 2.5 = 150% of one server's capacity at current count", "Total CPU needed ≈ 4 × 150% = 600% of a server; at 70% max → 600/70 ≈ 8.6 → 9 servers", "States the linear-scaling assumption"])
add("reasoning", P, "easy", "If a model is 90% accurate and we process 2,000 support tickets a day, how many tickets per day are likely misclassified, and what would you do about it?",
    ["≈ 200 misclassified per day", "Suggests human review of low-confidence cases or routing by confidence", "Mentions measuring which errors are costly"])
add("reasoning", P, "hard", "A feature increased average session time by 8% but daily active users fell 2% in the same period. Lay out a reasoning process to decide whether the feature was a success.",
    ["Checks whether the two changes are causally linked (experiment vs. coincidence, seasonality)", "Segments users (who left vs. who engaged more)", "Ties back to the north-star/business goal rather than either metric alone", "Proposes an experiment or analysis to resolve it"])

# ---------------------------------------------------------------- 4. Software engineering, conceptual (10)
P = "software engineer"
add("software", P, "medium", "My Python API endpoint is fast locally but takes 4 seconds in production. It queries a list of orders and then, for each order, queries the customer. What's the likely problem and fix?",
    ["Identifies the N+1 query problem", "Fix: join or batch-load customers (eager loading / IN query)", "Suggests adding an index if needed and measuring with query logs or APM"])
add("software", P, "easy", "What's the difference between a process and a thread? When would I use each in Python?",
    ["Processes have separate memory; threads share memory within a process", "Python GIL limits CPU-bound threads; use multiprocessing for CPU-bound work", "Threads/async suit I/O-bound work"])
add("software", P, "hard", "Design a URL shortener that handles 10,000 writes/sec and 100,000 reads/sec. Cover the key components and trade-offs.",
    ["ID generation strategy (base62, counter/hash, collision handling)", "Storage choice and partitioning/sharding", "Caching for reads (e.g. Redis/CDN)", "Discusses trade-offs such as consistency, analytics, expiry"])
add("software", P, "medium", "Our React app re-renders a large list every time the user types in a search box, making typing laggy. How do I fix it?",
    ["Debounce the input or use deferred value/transition", "Memoise list items (React.memo, useMemo) and stable keys", "Virtualise long lists (windowing)"])
add("software", P, "medium", "Explain what a race condition is, with a simple example, and two ways to prevent it.",
    ["Definition: outcome depends on timing of concurrent operations", "Concrete example (e.g. two threads incrementing a counter / double booking)", "Prevention: locks/mutexes, atomic operations, transactions or idempotency"])
add("software", P, "hard", "We get duplicate charges when users double-click 'Pay'. The payment API sometimes times out and the client retries. How should we design this to be safe?",
    ["Idempotency keys per payment attempt", "Server-side deduplication / unique constraints", "Disable the button client-side but don't rely on it", "Handle timeouts by checking status before retrying"])
add("software", P, "easy", "What does HTTP status code 429 mean and how should a client handle it?",
    ["Too Many Requests / rate limited", "Respect Retry-After header", "Exponential backoff with jitter"])
add("software", P, "medium", "Our Docker image is 2.3 GB for a small Python app. How can we make it smaller?",
    ["Use a slim/alpine or distroless base image", "Multi-stage builds", "Avoid caching pip downloads, clean up, use .dockerignore", "Only install needed dependencies"])
add("software", P, "hard", "A service's memory usage grows steadily over days until it is restarted. How would you investigate and fix a suspected memory leak in a Python service?",
    ["Measure and reproduce: track memory over time", "Use profilers (tracemalloc, objgraph, memray) to find growing objects", "Common causes: unbounded caches, global lists, reference cycles, unclosed resources", "Fix and verify with a load test"])
add("software", P, "medium", "When should I use a message queue (like Kafka or RabbitMQ) instead of direct API calls between services?",
    ["Decoupling and asynchronous processing", "Smoothing spikes / buffering load", "Reliability: retries, durability", "Trade-offs: added complexity, eventual consistency, ordering"])

# ---------------------------------------------------------------- 5. AI/ML engineer work (35)
P = "AI/ML engineer"
A = lambda d, p, k: add("aiml", P, d, p, k)
A("medium", "My model's training loss keeps going down but validation loss started going up after epoch 5. What's happening and what should I try?",
  ["Overfitting", "Early stopping / checkpoint at best validation", "Regularisation (dropout, weight decay), data augmentation, more data", "Simplify the model"])
A("easy", "What's the difference between precision and recall, and when should I optimise for each?",
  ["Precision = TP/(TP+FP); recall = TP/(TP+FN)", "Optimise precision when false positives are costly", "Optimise recall when false negatives are costly", "Threshold controls the trade-off"])
A("hard", "Our fraud classifier has 99.2% accuracy but the business says it misses most fraud. Fraud is 0.5% of transactions. What's going on and how should we evaluate and improve it?",
  ["Class imbalance makes accuracy misleading", "Use precision/recall, PR-AUC, recall at a fixed precision", "Resampling or class weights; tune the threshold on business costs", "Evaluate on a time-based split"])
A("medium", "How do I choose chunk size for a RAG system over internal PDFs?",
  ["Trade-off: small chunks are precise but lose context; large chunks add noise", "Respect document structure (headings, paragraphs); use overlap", "Evaluate retrieval recall@k on a labelled question set to pick size", "Consider parent-document or hierarchical retrieval"])
A("hard", "Our RAG chatbot gives confident wrong answers even though the right document exists in the index. How do I debug this systematically?",
  ["Evaluate retrieval separately from generation", "Check if the right chunk is in top-k (recall@k); inspect parsing/chunking", "Try hybrid search (BM25 + vectors) and a reranker", "Instruct grounding/abstention and measure groundedness"])
A("medium", "When should I fine-tune an LLM versus using RAG or better prompting?",
  ["Prompting first; RAG for knowledge that is private or changes", "Fine-tune for consistent behaviour/format or to make a smaller model good at a narrow task", "Fine-tuning needs quality data and an eval to prove it helps", "Knowledge vs behaviour distinction"])
A("hard", "How would you build an evaluation pipeline for an LLM feature that summarises customer support tickets?",
  ["Start with error analysis on real outputs to find failure modes", "Build a labelled eval set covering common and edge cases", "Graders: code checks + LLM-as-judge validated against human labels", "Run on every change; track faithfulness/coverage; online signals after launch"])
A("medium", "Explain data drift vs concept drift and how to monitor each in production.",
  ["Data drift: input distribution changes", "Concept drift: relationship between inputs and labels changes", "Monitor input statistics (PSI, KS tests) and prediction distributions", "Monitor performance with delayed labels; retraining triggers"])
A("easy", "What does temperature do in LLM sampling, and what value should I use for structured data extraction?",
  ["Controls randomness of token selection", "Low temperature (around 0) for extraction/consistency", "Higher for creative tasks", "Use structured outputs/schema for extraction"])
A("medium", "My LLM-as-judge seems to favour longer answers. How do I make it more reliable?",
  ["Known verbosity/position biases", "Use specific binary criteria instead of vague scores", "Randomise answer order; control for length", "Validate against human labels and measure agreement"])
A("hard", "Inference cost for our LLM feature is too high. List the levers to cut cost by 50% without hurting quality, in the order you'd try them.",
  ["Prompt caching for stable prefixes", "Trim tokens: shorter prompts, less history, fewer retrieved chunks, concise outputs", "Batch processing for offline work", "Route easy requests to smaller models / lower effort; fine-tune a small model; always verify with evals"])
A("medium", "What's the difference between an embedding model and a generative LLM, and when do I need each?",
  ["Embedding models map text to vectors for similarity", "Generative LLMs produce text", "Embeddings for search, clustering, classification features, RAG retrieval", "LLMs for generation, reasoning, answering"])
A("medium", "How do I handle class imbalance when training a text classifier with 95% negative examples?",
  ["Class weights or resampling (oversample minority / undersample majority)", "Use appropriate metrics (F1, PR-AUC, per-class recall)", "Tune the decision threshold", "Collect more minority examples if possible"])
A("hard", "Design a system to detect prompt injection in an AI agent that reads user emails and can send replies.",
  ["Treat email content as untrusted data, separate from instructions", "Least-privilege tools; require human approval for sending", "Input/output filtering and classifiers for injection patterns", "Logging, red-teaming, limits on actions"])
A("easy", "What is the difference between batch inference and real-time inference? Give an example of each.",
  ["Batch: process many inputs offline on a schedule (e.g. nightly scoring)", "Real-time: low-latency per request (e.g. fraud check at checkout)", "Trade-offs: cost, latency, freshness"])
A("medium", "How do I know if my train/test split is leaking information?",
  ["Features unavailable at prediction time (target leakage)", "Duplicates or near-duplicates across splits", "Time-based data needs time-based splits; group splits for users/entities", "Suspiciously high scores are a warning sign"])
A("hard", "We want to A/B test a new recommendation model. What metrics, guardrails and pitfalls should we plan for?",
  ["Primary business metric (e.g. clicks that convert, retention) and offline-to-online gap", "Guardrails: latency, diversity, revenue, complaints", "Pitfalls: novelty effects, feedback loops, network effects, sample ratio mismatch", "Run long enough; pre-register the decision rule"])
A("medium", "Explain what a vector database does and when pgvector is enough versus a dedicated vector DB.",
  ["Stores embeddings and supports nearest-neighbour search (often ANN indexes like HNSW)", "pgvector is enough for moderate scale and when you already use Postgres", "Dedicated DBs for very large scale, high QPS, advanced filtering/ops", "Trade-off: extra infrastructure vs. performance"])
A("medium", "My model works well on English text but poorly on Hindi. What could cause this and how do I improve it?",
  ["Training data imbalance / tokenizer inefficiency for Hindi", "Evaluate per language with a Hindi test set", "Use multilingual models or add Hindi data / fine-tune", "Consider transliteration and code-mixed (Hinglish) inputs"])
A("hard", "How would you decide between a small open-source model self-hosted on GPUs and a frontier model API for a high-volume classification task?",
  ["Compare quality on your own eval set", "Total cost: API per-token cost vs GPU + ops cost at your volume", "Latency, data privacy and control", "Consider fine-tuning a small model if quality gap is small"])
A("easy", "What is overfitting? Give a one-paragraph explanation with an example.",
  ["Model learns noise in training data and generalises poorly", "Gap between training and test performance", "Example such as memorising training examples", "Mitigations mentioned briefly"])
A("medium", "How should I version datasets and models so experiments are reproducible?",
  ["Version data (DVC, lakeFS, or immutable snapshots with hashes)", "Track experiments: code commit, data version, hyperparameters, metrics (MLflow/W&B)", "Fix random seeds; record environment/dependencies", "Model registry with stages"])
A("hard", "Our model's offline AUC improved from 0.81 to 0.84 but the online A/B test showed no change in conversions. Why might that happen?",
  ["Offline metric not aligned with business outcome", "Improvement in segments that matter little / ranking positions users don't see", "Training-serving skew or feature differences online", "Test underpowered or novelty/feedback effects"])
A("medium", "What is quantisation and what trade-offs does it bring for serving LLMs?",
  ["Reducing numeric precision of weights (e.g. 16-bit to 8/4-bit)", "Benefits: less memory, faster/cheaper inference", "Costs: possible quality loss, especially at very low bits", "Evaluate quality after quantising"])
A("medium", "How do I set a threshold for a toxicity classifier used to auto-hide comments?",
  ["Depends on costs of false positives vs false negatives", "Use a validation set to plot precision/recall vs threshold", "Consider a middle band for human review", "Monitor by segment for bias; revisit over time"])
A("hard", "Explain how you would evaluate an AI agent that books meetings by reading calendars and sending invites.",
  ["Task success rate on realistic scenarios", "Evaluate the full trajectory (tool calls, intermediate steps), not just the final answer", "Safety checks: wrong invites, permissions, confirmations", "Cost, latency and intervention rate; regression tests from failures"])
A("easy", "What is a confusion matrix?",
  ["Table of true/false positives and negatives", "Rows actual vs columns predicted (or vice versa)", "Basis for precision, recall, accuracy"])
A("medium", "How do I reduce latency for an LLM chat feature where users complain responses feel slow?",
  ["Stream tokens to reduce perceived latency", "Use smaller/faster models for simple turns; reduce prompt length", "Prompt caching; parallelise retrieval/tool calls", "Measure time to first token and p95"])
A("hard", "We trained a churn model, but the retention team says the predicted 'high risk' users are ones they can't save anyway. How should we rethink the modelling?",
  ["Predicting churn ≠ predicting who responds to intervention", "Uplift modelling / treatment-effect estimation", "Run experiments to measure intervention impact", "Align the target with the business action"])
A("medium", "What's the purpose of a held-out test set if I already have a validation set?",
  ["Validation set is used for tuning and model selection, so it becomes optimistic", "Test set gives an unbiased final estimate", "Touch the test set only once at the end"])
A("medium", "How would you detect and handle duplicate or near-duplicate documents before building a RAG index?",
  ["Exact dedup with hashes", "Near-duplicate detection with MinHash/LSH or embedding similarity", "Keep the most recent/authoritative version; store metadata", "Re-run dedup on updates"])
A("hard", "How do you decide whether an LLM output quality regression after a model upgrade is real or noise?",
  ["Run the same eval set on both versions", "Look at per-category results and confidence intervals / significance", "Inspect failing examples manually (error analysis)", "Check for judge variance; repeat runs if sampling is stochastic"])
A("easy", "What is the difference between supervised and unsupervised learning? One example each.",
  ["Supervised learns from labelled examples (e.g. spam detection)", "Unsupervised finds structure without labels (e.g. customer clustering)"])
A("medium", "How do I write a good system prompt for a customer-support LLM?",
  ["Define role, audience and goal", "Give clear rules, tone and what to do when uncertain (escalate/ask)", "Provide examples and separate instructions from user data", "Evaluate prompt changes on an eval set"])
A("hard", "Our recommendation model keeps recommending the same popular items. How do we fix this feedback loop?",
  ["Popularity bias / feedback loop from training on its own exposures", "Exploration (bandits, epsilon-greedy), diversity constraints", "Debias training data (inverse propensity weighting)", "Measure diversity/coverage as guardrails"])

# ---------------------------------------------------------------- 6. PM ramp-up explanations (45)
P = "PM new to AI"
E = lambda d, p, k: add("pm_explain", P, d, p, k)
E("easy", "Explain what an LLM is to me as a product manager, in 5 bullet points, and what it means for the features I can build.",
  ["Trained to predict the next token on large text data", "Can generate, summarise, classify, extract; follows instructions", "Can be wrong confidently (hallucinations)", "Product implications: grounding, evals, cost per use"])
E("medium", "What is RAG, and when should my team use it instead of fine-tuning? Explain like I'm a PM deciding the roadmap.",
  ["Retrieve relevant documents and give them to the model as context", "Use RAG for private or frequently changing knowledge; citations", "Fine-tuning changes behaviour/format; needs data and retraining", "Decision depends on knowledge vs behaviour, cost, freshness"])
E("medium", "Our ML engineer said the model has 'high precision but low recall'. What does that mean for our product and users?",
  ["Most flagged items are correct, but many true cases are missed", "User impact depends on the cost of misses vs false alarms", "Threshold can trade one for the other", "PM decides based on error costs"])
E("easy", "What are tokens and why do they matter for cost and pricing of AI features?",
  ["Tokens are chunks of text (~¾ of an English word)", "APIs charge per input and output token; output usually costs more", "Long prompts, history and outputs drive cost", "Indian languages often use more tokens"])
E("medium", "Explain what an 'eval' is and why my ML team keeps insisting on building one before launch.",
  ["Repeatable test of AI quality with inputs, grader and score", "Prevents vibe-checking and catches regressions", "Defines what 'good' means", "Enables decisions about launch thresholds and model changes"])
E("hard", "My team wants to build an AI agent. Explain what agents are, their risks, and how I should scope the first version.",
  ["Model in a loop using tools to complete multi-step tasks", "Risks: compounding errors, cost/latency, unsafe actions, prompt injection", "Start narrow with clear tasks and human approval for consequential actions", "Evaluate full trajectories; log every step"])
E("medium", "What's a hallucination, why does it happen, and what product decisions reduce its impact?",
  ["Fluent but false output", "Model predicts plausible text, doesn't know what it doesn't know", "Grounding with sources, citations, abstention, tools", "UX: show sources, easy reporting, human review for high stakes"])
E("easy", "What's the difference between AI, machine learning, deep learning and generative AI? Keep it short.",
  ["AI is the broad field", "ML learns patterns from data", "Deep learning uses multi-layer neural networks", "Generative AI creates new content (text, images)"])
E("medium", "What does 'context window' mean and how should it affect my product design?",
  ["Max tokens the model can consider in one request", "Bigger windows allow long documents but cost more and may dilute attention", "Design choices: retrieval, summarising history", "Affects cost and latency"])
E("hard", "Explain the precision–recall threshold trade-off with a fraud example, and tell me how I as PM should choose the threshold.",
  ["Raising threshold increases precision and lowers recall (and vice versa)", "Assign costs to false positives and false negatives", "Choose the threshold minimising total expected cost", "Can differ by segment; revisit with data"])
E("medium", "What is an embedding, explained with a product example I can use in a meeting?",
  ["List of numbers representing meaning so similar items are close", "Example: semantic search or 'similar products'", "Enables search by meaning instead of keywords"])
E("medium", "Why can't our ML team give me a firm date for when the model will be 'good enough'?",
  ["ML work is experimental; performance isn't known in advance", "Progress depends on data and iterations", "Plan with checkpoints/decision gates and fallback options", "Agree on the eval and launch bar upfront"])
E("easy", "What does 'latency' mean for an AI feature and which numbers should I ask the team for?",
  ["Time taken to respond", "Time to first token and total response time", "Percentiles: p50 and p95", "Streaming affects perceived latency"])
E("hard", "Explain A/B testing for an AI model change: what's different from a normal UI A/B test?",
  ["Offline eval should gate online tests", "Shadow mode / canary before exposure", "Watch cost and latency guardrails, not just quality", "Model outputs vary; measure downstream behaviour and novelty effects"])
E("medium", "What is fine-tuning in simple terms, and what would I need to provide for it?",
  ["Further training a model on your examples", "Needs hundreds to thousands of high-quality examples", "Changes style/behaviour; can make small models good at a narrow task", "Ongoing cost: retraining and evaluation"])
E("medium", "What is model drift and why should a PM care about it after launch?",
  ["Model performance degrades as data or the world changes", "Product quality can drop silently", "Need monitoring and retraining plans", "Include in launch plan and dashboards"])
E("easy", "What's the difference between open-source and closed AI models for a business?",
  ["Closed: accessed by API, usually most capable, per-token pricing", "Open: downloadable, self-hostable, more control/privacy", "Trade-offs: capability, cost at scale, ops burden", "Many companies use both"])
E("hard", "Explain how AI products make money differently from traditional SaaS, and the risks to gross margin.",
  ["AI has real marginal cost per use (inference)", "Heavy users can be unprofitable under flat pricing", "Pricing options: usage-based, tiers/credits, outcome-based", "Gross margins can be lower than typical SaaS"])
E("medium", "What is prompt engineering, and is it a real product lever or just a hack?",
  ["Designing instructions/examples for a model", "Real lever: affects quality, format, safety, cost", "Should be versioned and evaluated like code", "Cheaper than fine-tuning; first step on the ladder"])
E("medium", "Explain 'LLM-as-a-judge' and whether I can trust it for measuring quality.",
  ["Using a model to grade outputs against criteria", "Scales evaluation cheaply", "Has biases (position, verbosity, self-preference)", "Must be validated against human labels"])
E("easy", "What does 'human in the loop' mean and when should my AI feature use it?",
  ["Humans review/approve AI outputs or actions", "Use when errors are costly or irreversible", "Also used for low-confidence cases and to collect labels"])
E("hard", "My team says we need 'more data' to improve the model. How do I evaluate that request and what questions should I ask?",
  ["What failure modes does more data fix (from error analysis)?", "What kind of data, how much, how labelled, cost and time", "Is the issue data quantity, quality, or coverage of segments?", "Expected improvement measured on an eval"])
E("medium", "What is a vector database and do we really need one for our search feature?",
  ["Stores embeddings and finds nearest neighbours", "Not always needed: small scale can use in-memory search or Postgres pgvector", "Needed for large scale/high QPS", "Decision depends on scale, ops cost"])
E("medium", "Explain what 'reasoning models' are and when they're worth the extra cost.",
  ["Models that think step by step before answering (extra tokens)", "Better at maths, code, multi-step analysis", "Slower and more expensive", "Use for hard tasks; not needed for simple ones"])
E("easy", "What is a confusion matrix, explained for a PM?",
  ["Table counting correct and incorrect predictions by type", "True/false positives/negatives", "Helps see which errors happen"])
E("hard", "Explain the EU AI Act and India's DPDP Act at a level I need to make product decisions.",
  ["EU AI Act: risk-based tiers; high-risk uses have strict obligations; transparency duties; phased timeline", "DPDP Act: consent-based processing of personal data, purpose limitation, user rights", "Implications: data minimisation, disclosure, human oversight", "Work with legal on specifics; dates change"])
E("medium", "What is a data flywheel and how could our product build one?",
  ["Usage generates data that improves the product, driving more usage", "Needs a quality signal (accepted suggestions, corrections)", "Concrete example for a product", "Not just more logs"])
E("medium", "What do 'p50' and 'p95' latency mean and why does p95 matter more for user experience?",
  ["Median vs 95th percentile response time", "p95 captures the slow tail many users still experience", "Averages hide outliers"])
E("easy", "What's the difference between training and inference?",
  ["Training: learning model weights from data (expensive, occasional)", "Inference: using the model to make predictions (per request)", "Inference cost scales with usage"])
E("hard", "How should I think about build vs buy for an AI capability like speech-to-text in our app?",
  ["Is it core differentiation?", "Quality on our languages/accents; evaluate vendors on our data", "Cost at scale, latency, privacy, vendor lock-in", "Team capability to build and maintain"])
E("medium", "Explain 'cold start' in recommendation systems and product ways to handle it.",
  ["New users/items have no history", "Use onboarding preferences, popularity, content-based features", "Exploration strategies", "Collect signals quickly"])
E("medium", "What is semantic search and how is it different from keyword search?",
  ["Searches by meaning using embeddings", "Keyword search matches exact terms", "Semantic handles synonyms/paraphrases; keyword better for exact codes/names", "Hybrid often best"])
E("easy", "What are guardrails in AI products?",
  ["Controls that keep AI behaviour safe and on-policy", "Input/output filters, rules, limits on actions", "Also refers to guardrail metrics that must not get worse"])
E("hard", "What does it mean that 'offline metrics are proxies'? Give me an example where a model looked better offline but didn't help users.",
  ["Offline eval scores don't always translate to user outcomes", "Example: higher accuracy on cases users rarely encounter / ranking positions not seen", "Need online metrics (task success, retention)", "Use both"])
E("medium", "What is prompt caching and how does it reduce costs?",
  ["Reusing a repeated prompt prefix across requests", "Cached tokens are much cheaper and faster", "Works when system prompts/documents are stable", "Design prompts with stable prefix first"])
E("medium", "Explain multimodal AI and two product opportunities it creates for Indian users.",
  ["Models handling text, images, audio, etc.", "Opportunities: voice in local languages, reading documents/receipts, visual search", "Considerations: accuracy, accessibility"])
E("easy", "What is an API rate limit and how does it affect my AI feature's launch?",
  ["Max requests allowed per time window", "Can cause errors under load", "Plan capacity, retries/backoff, request limit increases"])
E("hard", "My engineer says our AI feature costs ₹3 per request. How do I figure out whether that's acceptable?",
  ["Compare cost per request with value/revenue per request or per user", "Estimate usage per user and cost per user per month", "Gross margin impact; heavy users", "Levers to reduce cost; pricing options"])
E("medium", "Explain synthetic data and when it's useful for building AI features.",
  ["Artificially generated data (often by models)", "Useful when real data is scarce, sensitive, or to cover edge cases", "Risks: unrealistic distribution, bias", "Validate with real data"])
E("medium", "What is the difference between accuracy and F1 score? When does accuracy mislead?",
  ["Accuracy = share of correct predictions", "F1 balances precision and recall", "Accuracy misleads with class imbalance"])
E("easy", "What does 'grounding' mean for an LLM?",
  ["Basing answers on provided sources/data", "Reduces hallucinations", "Often with citations"])
E("hard", "Explain the trade-offs of letting an AI feature take actions automatically versus only suggesting actions.",
  ["Autonomy spectrum: inform, suggest, act with approval, act and report, fully autonomous", "Higher autonomy saves effort but raises cost of errors", "Start lower; increase autonomy as evals and trust grow", "Reversibility and stakes decide"])
E("medium", "What's the difference between a chatbot and a copilot from a product perspective?",
  ["Chatbot: conversational interface, often standalone", "Copilot: embedded in a workflow, assists the user in context", "Copilots use context of the task; measured by productivity"])
E("medium", "What is RLHF in simple terms and why did it make chatbots more useful?",
  ["Reinforcement learning from human feedback", "Humans compare responses; model learns preferred behaviour", "Made models more helpful, safer, better at following instructions"])
E("easy", "What is an AI 'model card'?",
  ["Document describing a model's intended use, performance, limitations, and risks", "Supports transparency and responsible use"])

# ---------------------------------------------------------------- 7. PM decisions: features, deadlines, thresholds, strategy (45)
P = "product manager"
M = lambda d, p: add("pm_decide", P, d, p)
M("medium", "We're launching an AI summary feature for support tickets. Engineering says accuracy is 88% on our eval but the target was 92%. Launch is in 2 weeks. What should I do?")
M("hard", "Our fraud model can block 80% of fraud at a 2% false-positive rate, or 90% of fraud at a 6% false-positive rate. A missed fraud costs ₹5,000 on average and a false positive costs ₹150 in support and lost goodwill. We process 100,000 transactions/day with 0.3% fraud. Which threshold should we pick?")
M("medium", "Define success metrics for an AI writing assistant inside our note-taking app.")
M("hard", "Should our mid-sized Indian SaaS company build its own LLM, fine-tune an open model, or use APIs? We have 40 engineers, ₹2 crore AI budget, and customers in BFSI who care about data privacy.")
M("medium", "Prioritise these 5 features for next quarter using RICE and explain any overrides: AI search, dark mode, bulk export, SSO, onboarding checklist. Make reasonable assumptions.")
M("easy", "Write a crisp problem statement for: users abandon our KYC flow at the document upload step.")
M("medium", "Our AI chatbot's thumbs-down rate went from 4% to 9% after a model update last week. Walk me through how to investigate and what to tell leadership.")
M("hard", "We charge ₹499/month flat for our AI assistant. Top 10% of users generate 60% of our AI costs and we're losing money on them. What pricing changes would you consider?")
M("medium", "Engineering estimates 8 weeks for a feature I promised sales in 4 weeks. How should I handle this?")
M("medium", "How should I set the launch bar for an AI feature that drafts replies to customer emails?")
M("hard", "A competitor launched a free AI feature similar to our paid one. What should our strategy be over the next 6 months?")
M("easy", "Write 3 user stories with acceptance criteria for a 'share playlist' feature.")
M("medium", "Our north-star metric is weekly active users, but leadership wants to focus on revenue this quarter. How do I align the roadmap?")
M("hard", "Design an experiment to test whether adding AI-generated product descriptions increases conversion on our e-commerce site, including sample size thinking and guardrails.")
M("medium", "How would you decide whether to make an AI feature opt-in or on by default?")
M("medium", "Write a one-paragraph status update to executives: AI search launch delayed 2 weeks due to quality issues with Hindi queries; English quality is above target.")
M("hard", "We have budget for one of these: improving model accuracy from 90% to 94%, or building a human review queue for low-confidence cases. How do I decide?")
M("medium", "What should be in the PRD for an AI feature that a normal PRD might miss?")
M("easy", "Give me 5 questions to ask in a customer interview to understand how small shop owners manage inventory.")
M("hard", "Our AI feature's cost per user is ₹180/month on a ₹299 plan. Leadership wants 70% gross margin. Lay out a plan.")
M("medium", "Should we use a Now/Next/Later roadmap or a dated roadmap for our AI product? Our sales team wants dates.")
M("medium", "How do I decide between launching to 5% of users vs a closed beta for a risky AI feature?")
M("hard", "Our recommendation feature increased clicks by 12% but customer complaints about 'irrelevant suggestions' doubled. Should we keep it?")
M("medium", "Write the 'goals and non-goals' section for a feature that lets users search their photos with natural language.")
M("medium", "How should I respond when a senior stakeholder insists on adding an AI chatbot to our product 'because competitors have one'?")
M("hard", "Estimate the ROI of an AI agent that resolves 30% of our 50,000 monthly support tickets automatically. Human agents cost ₹40 per ticket; the AI costs ₹6 per resolved ticket and ₹2 per attempted ticket. Assume it attempts 60% of tickets.")
M("easy", "Turn this feedback into themes: 'search is slow', 'can't find old invoices', 'love the new design', 'export to excel please', 'search results irrelevant'.")
M("medium", "What guardrail metrics would you set when launching an AI feature that auto-tags expenses for finance teams?")
M("hard", "We're entering the Indian SMB market with our AI accounting assistant. What's our go-to-market strategy?")
M("medium", "How do I prioritise fixing AI quality issues versus building new features on the roadmap?")
M("medium", "Write a decision memo recommending whether to switch our AI provider to save 30% on costs, given some quality uncertainty.")
M("hard", "Our AI resume screener ranks candidates. How do we make sure it's fair, and what would block us from launching?")
M("easy", "Suggest a north-star metric for a language-learning app and explain why.")
M("medium", "How should I plan the roadmap when the ML team isn't sure the model can reach the required accuracy?")
M("hard", "We can either serve all users with a cheaper model (85% quality) or premium users with a better model (95%) and free users with the cheap one. Which is better for a freemium AI app?")
M("medium", "Write a launch checklist for an AI feature going to 100% of users next week.")
M("medium", "How do I measure whether users trust our AI feature?")
M("hard", "Our AI meeting-notes feature has high adoption in week 1 but 70% of users stop using it by week 4. Diagnose and propose next steps.")
M("easy", "What are 3 risks of launching an AI chatbot for a bank's customers?")
M("medium", "Should our AI coding assistant show a confidence score with its suggestions? Make a recommendation.")
M("hard", "Leadership asks for an AI strategy for our food delivery app for the next 2 years. Outline it.")
M("medium", "How should I set OKRs for an AI team whose work is experimental?")
M("medium", "Our model's accuracy is 95% overall but 78% for users in tier-3 cities. What do we do before launch?")
M("hard", "Make the case for or against building an LLM router to cut our AI API bill of ₹12 lakh/month.")
M("medium", "Draft three options with trade-offs for handling user requests our AI assistant can't answer.")

# ---------------------------------------------------------------- 8. Interview prep (40)
P = "job seeker"
I = lambda d, p: add("interview", P, d, p)
I("medium", "I'm interviewing for an AI PM role. How should I answer 'Design an AI feature for Google Maps'? Give me a structure and a short sample answer.")
I("easy", "Give me a 60-second answer to 'Tell me about yourself' for a business analyst moving into product management.")
I("medium", "Critique my answer to 'Why do you want to be a PM?': 'I like technology and I want to lead teams and make decisions.'")
I("hard", "How do I answer 'Your model's accuracy dropped after launch, what do you do?' in an AI PM interview? Give a structured answer.")
I("medium", "What are the most common behavioural questions in PM interviews and how should I prepare stories for them?")
I("medium", "Help me turn this into a STAR story: 'I found that our reporting process wasted time, so I automated parts of it with Excel macros.'")
I("hard", "Mock interview: ask me one tough AI PM product sense question, then show what an excellent answer would cover.")
I("easy", "What should I ask the interviewer at the end of a PM interview?")
I("medium", "How do I explain my career switch from BA to PM without sounding like I lack experience?")
I("hard", "Give me a strong answer to 'How would you measure the success of ChatGPT?'")
I("medium", "How should I prepare for a guesstimate round? Give a method and one worked example.")
I("medium", "Critique this answer to 'How would you prioritise features?': 'I would use RICE and pick the highest score.'")
I("hard", "How do I answer 'Should we build our own LLM?' in a strategy round for an Indian e-commerce company?")
I("easy", "List 5 common mistakes candidates make in PM product design interviews.")
I("medium", "How do I talk about a personal side project (an AI photo search tool) in a PM interview so it sounds impressive but honest?")
I("hard", "Give me a model answer to 'Tell me about a time you disagreed with an engineer' for a PM interview.")
I("medium", "What's the difference between how Google and Amazon interview PMs, and how should I prepare for each?")
I("medium", "How should I answer 'What's your favourite AI product and how would you improve it?'")
I("hard", "Simulate an interviewer pushing back on my metric choice: I said 'DAU' as the north star for an AI writing tool. What would they ask and how should I respond?")
I("easy", "How long should my answers be in a behavioural interview?")
I("medium", "How do I prepare for a take-home assignment for a PM role in 5 days?")
I("hard", "How do I answer 'How would you evaluate an LLM feature before launch?' to impress an ML-heavy interview panel?")
I("medium", "I blanked on a question in my last interview. How do I recover in the moment next time?")
I("medium", "What technical topics should an AI PM be ready to explain in interviews, and to what depth?")
I("hard", "Give me a model answer to 'Design an AI feature for farmers in India'.")
I("easy", "How do I write a thank-you email after a PM interview?")
I("medium", "How should I answer 'What's a product you think is badly designed?'")
I("hard", "How do I answer 'Your AI feature is loved by users but loses money on each user. What do you do?'")
I("medium", "Help me prepare for an execution question: 'Uber ride requests dropped 10% in one city.'")
I("medium", "What does a strong PM resume bullet look like? Rewrite: 'Worked on requirement gathering for the payments module.'")
I("hard", "Give me a mock 'AI system design' question for a PM and what a strong answer includes.")
I("easy", "What's the STAR method?")
I("medium", "How should I negotiate my first PM offer in India?")
I("hard", "How do I answer 'When would you NOT use AI to solve a problem?' with a strong example?")
I("medium", "How do I handle a case interview where the interviewer gives almost no information?")
I("medium", "How should I explain precision and recall in an interview if asked, with an example?")
I("hard", "Critique this answer to 'Design a product for elderly users': 'I'd build an app with big buttons and voice commands.'")
I("easy", "How many mock interviews should I do before real PM interviews?")
I("medium", "How do I show 'customer obsession' in my interview answers if I haven't been a PM?")
I("hard", "Give me a strong answer for 'How would you price an AI feature in a B2B SaaS product?'")


def drafted():
    return D
