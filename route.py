"""Ask the router where a prompt should go.

    python route.py "What's the capital of France?"
    python route.py "Prove that there are infinitely many primes, then generalise to arithmetic progressions."
"""

import sys

from router.policy import load_policy


def main():
    prompts = sys.argv[1:] or [input("Prompt: ")]
    policy = load_policy()
    print(f"Policy: cost-aware router, threshold set to keep {policy.target:.0%} of the strong model's quality "
          f"(~{policy.share_strong:.0%} of traffic goes to the strong model on the benchmark)\n")
    for prompt, d in zip(prompts, policy.decide(prompts)):
        label = "STRONG model (e.g. Claude Opus)" if d.route == "strong" else "CHEAP model (e.g. Haiku / local open-source)"
        print(f"> {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
        print(f"  -> {label}")
        print(f"     P(strong model needed) = {d.p_strong_needed:.2f}   relative extra cost = {d.predicted_extra_cost * 1000:.2f}"
              f"   gain-per-$ score = {d.score:.1f} (threshold {d.threshold:.1f})\n")


if __name__ == "__main__":
    main()
