"""Business use cases and the policy inputs a PM sets for each.

The router is the same for every use case; what changes is the policy:
- quality_bar: "acceptable" (the answer meets the user's need) or "best" (also not materially worse
  than the frontier model's answer).
- error_cost_inr: what one bad answer costs the business (retries, churn, support, bad decisions).
- quality_floor: the minimum share of good answers the use case must deliver, whatever the cost.
- monthly_volume: requests per month, to express costs in business terms.

These numbers are illustrative assumptions for each scenario, stated openly so they can be argued
with. The point of the project is the method: given these inputs, the threshold follows.
"""

from dataclasses import dataclass, field


@dataclass
class UseCase:
    key: str
    name: str
    categories: list
    quality_bar: str  # "acceptable" | "best"
    error_cost_inr: float
    quality_floor: float
    monthly_volume: int
    rationale: str
    p95_latency_budget_s: float = 30.0  # measured via the CLI, which adds a few seconds of startup
    allow_cascade: bool = False  # only where answers can be verified automatically (e.g. unit tests)
    notes: list = field(default_factory=list)


USE_CASES = [
    UseCase(
        "productivity", "Internal productivity assistant (rewrites, summaries, emails)",
        ["simple"], "acceptable", error_cost_inr=2, quality_floor=0.85, monthly_volume=500_000,
        rationale="Employees can see a weak rewrite instantly and just retry. A bad answer costs a few "
                  "seconds, so cost per request dominates.",
    ),
    UseCase(
        "knowledge", "Customer-facing knowledge Q&A",
        ["general_qa"], "acceptable", error_cost_inr=150, quality_floor=0.95, monthly_volume=200_000,
        rationale="A wrong answer reaches a customer: a support ticket (~₹150) and some lost trust. "
                  "Acceptable answers are fine, but errors must be rare.",
    ),
    UseCase(
        "analytics", "Analytics and reasoning helper (maths, estimates)",
        ["reasoning"], "acceptable", error_cost_inr=50, quality_floor=0.90, monthly_volume=50_000,
        rationale="Wrong numbers can propagate into reports, but analysts usually sanity-check results.",
    ),
    UseCase(
        "coding", "Coding assistant",
        ["software"], "acceptable", error_cost_inr=80, quality_floor=0.90, monthly_volume=300_000,
        rationale="Broken code costs developer time to spot and fix. Code can be verified by tests, "
                  "so a verify-then-escalate cascade is possible.",
        allow_cascade=True,
    ),
    UseCase(
        "mlops", "ML engineering copilot",
        ["aiml"], "acceptable", error_cost_inr=400, quality_floor=0.95, monthly_volume=40_000,
        rationale="Wrong technical advice can waste days of engineering effort. Errors are expensive.",
    ),
    UseCase(
        "pm_tutor", "PM ramp-up tutor (explain AI concepts)",
        ["pm_explain"], "acceptable", error_cost_inr=30, quality_floor=0.85, monthly_volume=100_000,
        rationale="Learning content: an acceptable explanation does the job, and the learner can ask a follow-up.",
    ),
    UseCase(
        "pm_decisions", "PM decision support (thresholds, deadlines, strategy)",
        ["pm_decide"], "best", error_cost_inr=2_000, quality_floor=0.95, monthly_volume=10_000,
        rationale="Feeds real product decisions. A weaker analysis can cost far more than tokens, and "
                  "volume is low, so the bar is the best available answer.",
    ),
    UseCase(
        "interview_free", "Interview-prep app, free tier",
        ["interview"], "acceptable", error_cost_inr=5, quality_floor=0.80, monthly_volume=1_000_000,
        rationale="Huge free volume with no revenue per request. Acceptable advice keeps users engaged; "
                  "cost must stay near zero.",
    ),
    UseCase(
        "interview_paid", "Interview-prep app, paid tier",
        ["interview"], "best", error_cost_inr=60, quality_floor=0.95, monthly_volume=50_000,
        rationale="Paying users expect the best coaching; weaker answers drive churn of a ₹499/month plan.",
    ),
    UseCase(
        "real_mle_prep", "Real user: ML-engineer interview prep (private slice)",
        ["real_interview"], "best", error_cost_inr=300, quality_floor=0.90, monthly_volume=20_000,
        rationale="Real questions from someone preparing for a big-tech ML interview loop. A shallow or wrong "
                  "technical explanation the week before an interview is costly, so the bar is the best answer.",
    ),
]

# p95 latency budget per use case, in seconds. Measured through the Claude Code CLI, which adds a few
# seconds of startup per call, so read these as relative budgets rather than production SLAs.
LATENCY_BUDGETS_S = {
    "productivity": 15,     # people wait for a rewrite, but not long
    "knowledge": 10,        # customer-facing: must feel responsive
    "analytics": 30,
    "coding": 30,
    "mlops": 60,
    "pm_tutor": 30,
    "pm_decisions": 120,    # quality matters far more than speed
    "interview_free": 20,
    "interview_paid": 60,
    "real_mle_prep": 60,
}
for _uc in USE_CASES:
    _uc.p95_latency_budget_s = LATENCY_BUDGETS_S[_uc.key]
