"""
v3 Cascade routing — canonical constants and gate functions.

This is the single source of truth for model names, prompts, and routing logic.
Both the smart-ask CLI and all benchmark runners import from here so they
stay in perfect sync.

Architecture
------------
  Gate 1  Binary Gemini classifier: easy → EASY_MODEL, hard → HARD_MODEL
  Gate 2  Gemini non-interactive preflight (hermes -q) with self-check suffix.
          Model outputs ESCALATE_MARKER alone on its own line to escalate.

Cost profile (HumanEval 164 problems)
--------------------------------------
  pass@1  89.0%   |   cost $0.199   |   5.3x cheaper than always-Opus
"""

import json, re, subprocess
from openai import OpenAI

# ── Models ────────────────────────────────────────────────────────────────────

OR_BASE          = "https://openrouter.ai/api/v1"
CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"   # same model as EASY_MODEL
EASY_MODEL       = "google/gemini-2.5-flash-lite"
HARD_MODEL       = "anthropic/claude-opus-4.8"

# ── Gate 1: binary classifier prompt ─────────────────────────────────────────

CLASSIFY_PROMPT = """\
You are routing a coding/AI task to either a cheap model (easy) or an expert model (hard).
Label "hard" if ANY of these are true:
- Requires dynamic programming, graph traversal, or non-obvious algorithm
- Has subtle edge cases a junior programmer would likely miss
- Needs number theory, combinatorics, or careful mathematical reasoning
- Complex multi-system design or advanced architecture decisions
Label "easy" if:
- Solvable with basic loops, string ops, or simple math
- Straightforward Q&A, explanation, debug, or format task
- Edge cases are obvious and minimal
Reply ONLY with JSON: {"d":"easy"} or {"d":"hard"}
Task:\n"""

# ── Gate 2: self-check suffix ─────────────────────────────────────────────────
# Appended to the prompt in the hermes -q preflight call.
# ESCALATE_MARKER appears only once in these instructions to avoid false positives.

SELF_CHECK_SUFFIX = (
    "\n\n---\n[SMART-ASK SELF-CHECK]\n"
    "After your answer, check:\n"
    "1. Does your code have stubs (NotImplementedError / pass as placeholder / # TODO)?\n"
    "2. If the task has visible >>> examples, does your code pass them?\n"
    "3. Is your answer only a high-level outline with no working code?\n"
    "If YES to any of the above: output the token ESCALATE_NOW alone on its own line.\n"
    "If your answer is complete and correct: no action needed, output nothing extra."
)

ESCALATE_MARKER = "ESCALATE_NOW"  # distinct token, appears once — no false positives


# ── Gate 1 function ───────────────────────────────────────────────────────────

def gate1_classify(prompt: str, api_key: str):
    """
    Classify prompt as 'easy' or 'hard' using CLASSIFIER_MODEL.

    Returns (difficulty: str, usage: object | None).
    `usage` is the raw OpenAI CompletionUsage object for cost tracking.
    """
    try:
        client = OpenAI(base_url=OR_BASE, api_key=api_key)
        r = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT + prompt[:1200]}],
            max_tokens=20, temperature=0,
        )
        raw = (r.choices[0].message.content or "").strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        difficulty = json.loads(raw).get("d", "easy")
        return difficulty, r.usage
    except Exception:
        return "easy", None


# ── Gate 2 function ───────────────────────────────────────────────────────────

def gate2_preflight(prompt: str, provider: str = "openrouter") -> bool:
    """
    Run Gemini non-interactively via `hermes chat -q` with SELF_CHECK_SUFFIX.

    Gemini drafts the full answer and self-evaluates (can run its own code
    with the terminal tool). It outputs ESCALATE_MARKER alone on its own line
    to signal that the answer is insufficient and Opus should take over.

    Returns True if escalation was triggered.
    """
    cmd = [
        "hermes", "chat", "-q",
        prompt + SELF_CHECK_SUFFIX,
        "-m", EASY_MODEL,
        "--provider", provider,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        # Line-only match: ESCALATE_MARKER must be alone on its own line.
        # This prevents false positives when the model mentions the token in prose.
        return bool(re.search(
            rf'^\s*{re.escape(ESCALATE_MARKER)}\s*$', output, re.MULTILINE
        ))
    except Exception:
        return False   # on error, give benefit of doubt — don't escalate
