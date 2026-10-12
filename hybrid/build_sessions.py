"""Turn your own Claude Code session transcripts into router data (kept local, never committed).

    python -m hybrid.build_sessions <session1.jsonl> [<session2.jsonl> ...]

For every message you typed, record:
- the request (verbatim, secrets redacted) and a short excerpt of the previous reply as context,
- Opus's actual reply text (the frontier answer we compare against),
- how Opus handled it: "text" (no tools), "lookup" (read-only tools such as web search or reading
  files), or "action" (edited files, ran commands). Only "text" requests are a fair text-to-text
  comparison with a small model,
- the real token usage of that turn, a proxy for how fast it used the Pro usage limit.

Output: data/session_requests.jsonl (data/ is gitignored).
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "session_requests.jsonl"
# Transcript files to read, given on the command line in the order to number them, e.g.
#   python -m hybrid.build_sessions ~/.claude/projects/<project>/<session-id>.jsonl ...
# (This case study used two sessions: the main project session and the SmartHire session.)
SESSION_FILES = [os.path.expanduser(p) for p in sys.argv[1:]]
CONTEXT_CHARS = 700
LONG_CONTEXT_CHARS = 12_000  # ~3k tokens: recent conversation, both sides, no tool output

ACTION_TOOLS = {"Edit", "Write", "Bash", "PowerShell", "NotebookEdit", "Artifact", "Agent", "TaskStop"}
LOOKUP_TOOLS = {"Read", "Grep", "Glob", "WebSearch", "WebFetch", "ToolSearch", "Skill", "AskUserQuestion"}

SECRET_PATTERNS = [
    re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}"),           # Google-style API keys
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),             # OpenAI / Anthropic style keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),         # GitHub tokens
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email addresses
]
NOISE = re.compile(r"<(system-reminder|ide_selection|ide_opened_file|command-[a-z-]+|local-command-[a-z-]+)>.*?</\1>", re.S)


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def user_text(msg) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        parts = [c]
    elif isinstance(c, list):
        parts = [x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"]
    else:
        parts = []
    return NOISE.sub("", "\n".join(parts)).strip()


# Claude Opus 5.5 list prices per million tokens: input $4, output $20, cache read $0.20,
# cache write about 1.25 x input. Used only to weight token types into an equivalent cost.
OPUS = {"fresh_input": 4.0, "cache_write": 5.0, "cache_read": 0.20, "output": 20.0}


def equiv_cost(r) -> float:
    return (r["fresh_input"] * OPUS["fresh_input"] + r["cache_write"] * OPUS["cache_write"]
            + r["cache_read"] * OPUS["cache_read"] + r["output_tokens"] * OPUS["output"]) / 1e6


def build():
    rows = []
    if not SESSION_FILES:
        raise SystemExit("Give one or more Claude Code transcript files: python -m hybrid.build_sessions <session.jsonl> ...")
    for path in SESSION_FILES:
        session = Path(path).stem[:8]
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        current, last_reply, transcript = None, "", ""
        for r in records:
            m = r.get("message") if isinstance(r.get("message"), dict) else {}
            if r.get("type") == "user" and not r.get("isMeta") and not r.get("isSidechain"):
                text = user_text(m)
                if text and not text.startswith("[Request interrupted") and len(text.split()) >= 2:
                    if current:
                        rows.append(current)
                        transcript += f"\n\nUSER: {current['request']}\n\nASSISTANT: {current['reply'].strip()}"
                    current = {"session": session, "request": redact(text), "context": redact(last_reply[-CONTEXT_CHARS:]),
                               "long_context": redact(transcript[-LONG_CONTEXT_CHARS:]),
                               "reply": "", "tools": [], "input_tokens": 0, "output_tokens": 0,
                               "fresh_input": 0, "cache_write": 0, "cache_read": 0}
                continue
            if r.get("type") == "assistant" and current and not r.get("isSidechain"):
                for block in m.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        current["reply"] += block.get("text", "") + "\n"
                    elif block.get("type") == "tool_use":
                        current["tools"].append(block.get("name", ""))
                u = m.get("usage") or {}
                current["input_tokens"] += (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) \
                    + (u.get("cache_read_input_tokens") or 0)
                current["output_tokens"] += u.get("output_tokens") or 0
                current["fresh_input"] += u.get("input_tokens") or 0
                current["cache_write"] += u.get("cache_creation_input_tokens") or 0
                current["cache_read"] += u.get("cache_read_input_tokens") or 0
                if current["reply"]:
                    last_reply = current["reply"]
        if current:
            rows.append(current)

    out, per_session = [], {}
    for r in rows:
        n = per_session[r["session"]] = per_session.get(r["session"], -1) + 1
        tools = set(r["tools"])
        kind = "action" if tools & ACTION_TOOLS else "lookup" if tools else "text"
        reply = redact(r["reply"].strip())
        if not reply:
            continue
        prompt = (f"(Earlier in this conversation, the assistant said: \"...{r['context'].strip()}\")\n\n{r['request']}"
                  if r["context"].strip() else r["request"])
        # ids are numbered within each session, so they stay stable as a session grows
        out.append({"id": f"session-{r['session']}-{n:03d}", "source": f"claude-code-session-{r['session']}", "category": "own_sessions",
                    "persona": "AI PM candidate working with Claude Code", "eval_type": "judge", "kind": kind,
                    "tools_used": sorted(tools), "prompt": prompt, "opus_reply": reply,
                    "prompt_long_context": (f"Recent conversation so far:\n{r['long_context'].strip()}\n\n"
                                            f"USER (new message): {r['request']}"
                                            if r["long_context"].strip() else r["request"]),
                    "opus_input_tokens": r["input_tokens"], "opus_output_tokens": r["output_tokens"],
                    "opus_tokens": {"fresh_input": r["fresh_input"], "cache_write": r["cache_write"],
                                    "cache_read": r["cache_read"], "output": r["output_tokens"]},
                    "opus_equiv_cost_usd": equiv_cost(r),
                    "reference": {"rubric": "Does the answer correctly and usefully address what the user asked, in the "
                                            "context of the conversation, at the depth they needed?"}})
    return out


if __name__ == "__main__":
    rows = build()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    kinds = Counter(r["kind"] for r in rows)
    print(f"Wrote {len(rows)} requests to {OUT} (gitignored): {dict(kinds)}")
    for k in ("text", "lookup", "action"):
        rs = [r for r in rows if r["kind"] == k]
        if rs:
            t = {x: sum(r["opus_tokens"][x] for r in rs) for x in ("fresh_input", "cache_write", "cache_read", "output")}
            print(f"  {k:7} n={len(rs):3}  cache-read {t['cache_read']:>12,}  cache-write {t['cache_write']:>11,}  "
                  f"fresh {t['fresh_input']:>9,}  output {t['output']:>10,}  equiv ${sum(r['opus_equiv_cost_usd'] for r in rs):8.2f}")
