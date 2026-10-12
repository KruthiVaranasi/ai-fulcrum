"""Zero-cost model backends for the live test.

- OllamaBackend: an open-source model running locally (no API cost).
- ClaudeCodeBackend: Claude models through the Claude Code CLI in print mode, which uses the
  user's Claude subscription instead of an API key. Calls count against subscription usage
  limits, so callers should keep volumes small.

Both return the same dict: text, input_tokens, output_tokens, latency_s, equiv_cost_usd.
`equiv_cost_usd` is what the request would cost on the pay-per-token API at list prices, so
cost comparisons stay meaningful even though nothing is billed.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from .live import PRICES

MINIMAL_SYSTEM_PROMPT = "You are a helpful assistant."


def find_claude_cli() -> str:
    """Path to the Claude Code executable. On Windows, npm installs a claude.cmd wrapper; calling the
    claude.exe behind it directly avoids cmd.exe mangling multi-line prompts."""
    found = shutil.which("claude")
    if not found:
        raise FileNotFoundError("Claude Code CLI not found on PATH")
    if found.lower().endswith((".cmd", ".ps1")) or not found.lower().endswith(".exe"):
        exe = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if exe.exists():
            return str(exe)
    return found


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://127.0.0.1:11434", max_tokens: int = 1024):
        self.model = model
        self.host = host
        self.max_tokens = max_tokens  # caps generation time on CPU for open-ended answers
        self.name = f"ollama/{model}"

    def ask(self, prompt: str) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.load(resp)
        except TimeoutError:
            return {"error": "ollama timeout (answer took too long)"}
        except OSError as e:
            if "timed out" in str(e):
                return {"error": "ollama timeout (answer took too long)"}
            return {"error": f"ollama unreachable: {e}"}
        return {
            "text": d["message"]["content"],
            "input_tokens": d.get("prompt_eval_count", 0),
            "output_tokens": d.get("eval_count", 0),
            "latency_s": round(time.time() - t0, 2),
            "equiv_cost_usd": 0.0,
        }


class ClaudeCodeBackend:
    """Runs `claude -p` with a minimal system prompt, no tools, no MCP servers and no saved session."""

    ALIASES = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}

    def __init__(self, alias: str = "haiku", effort: str = None, timeout: int = 300, thinking: bool = True):
        self.alias = alias
        self.thinking = thinking
        self.model = self.ALIASES[alias]
        self.effort = effort
        self.timeout = timeout
        self.name = f"claude/{alias}" + (f"@{effort}" if effort else "") + ("" if thinking else "-nothink")
        self.workdir = tempfile.mkdtemp(prefix="router-claude-")  # empty dir: no project files or CLAUDE.md

    def ask(self, prompt: str) -> dict:
        try:
            exe = find_claude_cli()
        except FileNotFoundError as e:
            return {"error": f"claude cli: {e}", "not_sent": True}
        # Long prompts (e.g. judge batches) go through stdin: Windows caps command lines at ~32k chars.
        use_stdin = len(prompt) > 8000
        cmd = [
            exe, "-p", *([] if use_stdin else [prompt]),
            "--model", self.alias,
            "--output-format", "json",
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", MINIMAL_SYSTEM_PROMPT,
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        t0 = time.time()
        try:
            env = dict(os.environ)
            if not self.thinking:
                env["MAX_THINKING_TOKENS"] = "0"  # Claude Code enables thinking by default; this turns it off
            out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True, encoding="utf-8",
                                 timeout=self.timeout, shell=False, env=env, input=prompt if use_stdin else None)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {"error": f"claude cli: {e}"}
        try:
            d = json.loads(out.stdout)
        except json.JSONDecodeError:
            return {"error": f"claude cli returned non-JSON: {(out.stdout or out.stderr)[:200]}"}
        if d.get("is_error"):
            return {"error": f"claude cli error: {d.get('result') or d.get('api_error_status')}"}
        u = d["usage"]
        input_tokens = u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        output_tokens = u["output_tokens"]  # includes thinking tokens, which the API bills as output
        pin, pout = PRICES[self.model]
        return {
            "text": d.get("result", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
            "latency_s": round(time.time() - t0, 2),
            "equiv_cost_usd": input_tokens * pin / 1e6 + output_tokens * pout / 1e6,
        }
