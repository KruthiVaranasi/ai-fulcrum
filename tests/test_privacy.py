"""Guards against publishing private data.

The real-usage slices (my interview-prep notes and my own Claude Code sessions) stay
on my machine. Only aggregate results and the public subset are published."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
PUBLIC_IDS = {json.loads(l)["id"] for l in (ROOT / "hybrid" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [ROOT / f for f in out.splitlines()]
    except Exception:  # not a git checkout: check everything outside private folders
        return [p for p in ROOT.rglob("*") if p.is_file() and not {"data", ".venv", ".git"} & set(p.parts)]


def test_public_results_only_contain_public_prompts():
    for f in (ROOT / "results" / "public").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)["id"] in PUBLIC_IDS, f"{f.name} contains a non-public row"


def test_private_files_are_not_tracked():
    names = {p.relative_to(ROOT).as_posix() for p in tracked_files()}
    for private in ["results/hybrid_answers.jsonl", "results/hybrid_grades.jsonl", "results/shadow_test.json"]:
        assert private not in names, f"{private} must stay local"
    assert not any(n.startswith("data/") for n in names), "data/ holds private and downloaded files"


def test_no_secrets_or_personal_identifiers_in_tracked_files():
    patterns = {
        "API key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}"),
        "private row id": re.compile(r'"id":\s*"(real-\d|session-)'),
        "local user path": re.compile(r"C:\\\\Users\\\\[A-Za-z]+|/c/Users/[A-Za-z]+|c--Users-[A-Za-z]+", re.I),
        "email address": re.compile(r"[\w.+-]+@(gmail|yahoo|outlook|hotmail)\.com", re.I),
    }
    for p in tracked_files():
        if p.suffix.lower() in {".png", ".jpg", ".pkl", ".npy", ".joblib"} or not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for label, rx in patterns.items():
            m = rx.search(text)
            assert not m, f"{label} found in {p.relative_to(ROOT)}: {m.group(0)[:40]}"
