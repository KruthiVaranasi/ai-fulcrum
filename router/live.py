"""Live calls to Claude models, with cost tracking.

Prices are Anthropic list prices per million tokens (October 2026). Check
https://www.anthropic.com/pricing before relying on them.
"""

import os
import re
import time
from pathlib import Path

PRICES = {  # model id: (input $/M tokens, output $/M tokens)
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
ALIASES = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_api_key() -> str:
    """Read ANTHROPIC_API_KEY from the environment, or from llm-router/.env (never committed)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit(
            "No Claude API key found. Create llm-router/.env with one line:\n"
            "ANTHROPIC_API_KEY=your-key-here\n"
            "(Get a key at console.anthropic.com. A Claude Pro subscription does not include API access.)"
        )
    return key


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES[model]
    return input_tokens * pin / 1e6 + output_tokens * pout / 1e6


def ask(client, model: str, prompt: str, effort: str = "low", max_tokens: int = 4096) -> dict:
    """One request. Thinking tokens on Sonnet/Opus are billed as output and included in usage."""
    import anthropic

    kwargs = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if model != "claude-haiku-4-5":
        # Effort trades depth of reasoning for cost and latency. Opus 5.5 defaults to "medium";
        # we set it explicitly so runs are comparable.
        kwargs["output_config"] = {"effort": effort}
    t0 = time.time()
    try:
        msg = client.messages.create(**kwargs)
    except anthropic.RateLimitError as e:
        return {"error": f"rate_limited: {e.message}"}
    except anthropic.APIStatusError as e:
        return {"error": f"api_error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        return {"error": "connection_error"}
    latency = time.time() - t0
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    return {
        "text": text,
        "stop_reason": msg.stop_reason,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost": cost_usd(model, u.input_tokens, u.output_tokens),
        "latency_s": round(latency, 2),
    }


_LETTER = re.compile(r"\b([A-E])\b")


def extract_choice(text: str):
    """First standalone answer letter in a response, e.g. 'B', 'B)', 'Answer: B'."""
    if not text:
        return None
    t = text.strip().lstrip("[('\"\\n ")
    if t[:1] in "ABCDE" and (len(t) == 1 or not t[1].isalpha()):
        return t[0]
    m = _LETTER.search(text)
    return m.group(1) if m else None
