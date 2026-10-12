"""What would *your* AI usage cost on pay-per-use APIs, with and without routing?

Runs entirely on your machine. Point it at the conversations.json from a ChatGPT export
(Settings > Data controls > Export data) or a Claude export (Settings > Privacy > Export data):

    python savings.py path/to/conversations.json
    python savings.py path/to/conversations.json --subscription-inr 2300 --inr-per-usd 88

How the estimate works:
- Every turn re-sends the conversation so far (that is how chat APIs bill), so input tokens grow
  through a conversation. Tokens are estimated as characters / 4.
- Output tokens come from the real assistant replies in your export.
- "Routed" sends each prompt to the cheap or strong tier using the cost-aware router.
- Prices: Claude list prices per million tokens (Oct 2026); cheap = Haiku 4.5, strong = Opus 5.5.
  Thinking tokens and prompt caching are not modelled; both would change the totals.
"""

import sys
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from router.live import PRICES

CHEAP, STRONG = "claude-haiku-4-5", "claude-opus-5-5"


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_export(data) -> list:
    """Return conversations as lists of (role, text, timestamp), for ChatGPT or Claude exports."""
    convs = []
    for conv in data:
        turns = []
        if "mapping" in conv:  # ChatGPT
            nodes = [n.get("message") for n in conv["mapping"].values() if n.get("message")]
            nodes.sort(key=lambda m: m.get("create_time") or 0)
            for m in nodes:
                role = (m.get("author") or {}).get("role")
                parts = (m.get("content") or {}).get("parts") or []
                text = "\n".join(p for p in parts if isinstance(p, str)).strip()
                if role in ("user", "assistant") and text:
                    turns.append((role, text, _ts(m.get("create_time"))))
        elif "chat_messages" in conv:  # Claude
            for m in conv["chat_messages"]:
                role = {"human": "user", "assistant": "assistant"}.get(m.get("sender"))
                text = (m.get("text") or "").strip()
                if not text and isinstance(m.get("content"), list):
                    text = "\n".join(c.get("text", "") for c in m["content"] if isinstance(c, dict)).strip()
                if role and text:
                    turns.append((role, text, _ts(m.get("created_at"))))
        if turns:
            convs.append(turns)
    return convs


def build_requests(convs) -> list:
    """One API request per user turn: (prompt, input_tokens incl. history, output_tokens, month)."""
    reqs = []
    for turns in convs:
        history = 0
        for i, (role, text, ts) in enumerate(turns):
            if role == "user":
                reply = next((t for r, t, _ in turns[i + 1:] if r == "assistant"), "")
                month = ts.strftime("%Y-%m") if ts else "unknown"
                reqs.append((text, history + tokens(text), tokens(reply), month))
            history += tokens(text)
    return reqs


def cost(model, tin, tout):
    pin, pout = PRICES[model]
    return tin * pin / 1e6 + tout * pout / 1e6


def main():
    if hasattr(sys.stdout, "reconfigure"):  # ₹ and ≈ crash the default Windows console encoding
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("--subscription-inr", type=float, default=2300, help="what you pay per month today")
    ap.add_argument("--inr-per-usd", type=float, default=88.0)
    args = ap.parse_args()

    convs = parse_export(json.loads(args.export.read_text(encoding="utf-8")))
    reqs = build_requests(convs)
    if not reqs:
        raise SystemExit("No user messages found. Is this a ChatGPT or Claude conversations.json?")

    from router.policy import load_policy

    decisions = load_policy().decide([r[0] for r in reqs])
    by_month = defaultdict(lambda: {"n": 0, "strong": 0.0, "cheap": 0.0, "routed": 0.0, "to_strong": 0})
    for (prompt, tin, tout, month), d in zip(reqs, decisions):
        b = by_month[month]
        b["n"] += 1
        b["strong"] += cost(STRONG, tin, tout)
        b["cheap"] += cost(CHEAP, tin, tout)
        b["routed"] += cost(STRONG if d.route == "strong" else CHEAP, tin, tout)
        b["to_strong"] += d.route == "strong"

    fx = args.inr_per_usd
    print(f"{len(convs)} conversations, {len(reqs)} prompts\n")
    print(f"{'Month':10} {'prompts':>8} {'all Opus':>10} {'all Haiku':>10} {'routed':>10} {'% to Opus':>10}")
    for month in sorted(by_month):
        b = by_month[month]
        print(f"{month:10} {b['n']:8} {'₹%.0f' % (b['strong'] * fx):>10} {'₹%.0f' % (b['cheap'] * fx):>10} "
              f"{'₹%.0f' % (b['routed'] * fx):>10} {b['to_strong'] / b['n']:10.0%}")
    months = [m for m in by_month if m != "unknown"] or list(by_month)
    avg = lambda key: sum(by_month[m][key] for m in months) / len(months) * fx
    print(f"\nAverage per month: all-Opus ₹{avg('strong'):.0f} · routed ₹{avg('routed'):.0f} · "
          f"your subscriptions ₹{args.subscription_inr:.0f}")
    routed = avg("routed")
    if routed < args.subscription_inr:
        print(f"Routing on pay-per-use APIs would cost about ₹{args.subscription_inr - routed:.0f}/month less than your subscriptions.")
    else:
        print("At your usage, subscriptions are cheaper than pay-per-use, even with routing.")
    print("Estimate only: tokens ≈ characters/4; thinking tokens, caching, images and files are not modelled.")


if __name__ == "__main__":
    main()
