"""
Gate 2 — Gemini self-check preflight.

What it does
------------
Runs Gemini non-interactively via `hermes chat -q` with SELF_CHECK_SUFFIX
appended to the task prompt. Gemini:

  1. Drafts a full answer to the task (with tool access — it can run code)
  2. Self-evaluates against three quality checks:
       • No stub code (NotImplementedError / pass / # TODO)
       • Visible >>> examples produce correct output
       • Answer is actual working code, not a high-level outline
  3. If any check fails → outputs ESCALATE_MARKER alone on its own line
     If all checks pass  → outputs nothing extra

When ESCALATE_MARKER is detected, the CLI discards the draft and restarts
the session with HARD_MODEL (Opus), giving it a brief failure hint.

Why run via hermes instead of raw API?
---------------------------------------
In hermes -q mode, Gemini has access to the terminal tool. It can actually
run its own code, observe failures, and self-report accurately. A raw API
call without tools produces much weaker self-evaluation.

False positive prevention
--------------------------
ESCALATE_MARKER appears exactly once in SELF_CHECK_SUFFIX (as the action
token). Detection uses a line-only regex — the marker must be alone on its
own line. If the model mentions the token in prose ("I don't need to output
ESCALATE_NOW"), the regex does not match.

Only runs for Gate 1 "easy" tasks. Hard tasks skip straight to Opus.
"""

import re, subprocess

# ── Models ────────────────────────────────────────────────────────────────────

EASY_MODEL = "google/gemini-2.5-flash-lite"   # model that runs the preflight
HARD_MODEL = "anthropic/claude-opus-4.8"       # model that takes over on escalation

# ── Self-check suffix ─────────────────────────────────────────────────────────
# Appended to the prompt in the hermes -q call.
# ESCALATE_MARKER appears only once — prevents the model from echoing it
# in explanatory prose, which would cause a false positive.

SELF_CHECK_SUFFIX = (
    "\n\n---\n[SMART-ASK SELF-CHECK]\n"
    "After your answer, check:\n"
    "1. Does your code have stubs (NotImplementedError / pass as placeholder / # TODO)?\n"
    "2. If the task has visible >>> examples, does your code pass them?\n"
    "3. Is your answer only a high-level outline with no working code?\n"
    "If YES to any of the above: output the token ESCALATE_NOW alone on its own line.\n"
    "If your answer is complete and correct: no action needed, output nothing extra."
)

ESCALATE_MARKER = "ESCALATE_NOW"   # emitted by the model to trigger escalation


# ── Gate 2 function ───────────────────────────────────────────────────────────

def gate2_preflight(prompt: str, provider: str = "openrouter") -> bool:
    """
    Run Gemini via `hermes chat -q` with SELF_CHECK_SUFFIX appended.

    Returns True  → escalate to HARD_MODEL (Opus)
    Returns False → proceed with EASY_MODEL (Gemini)

    On subprocess error or timeout returns False (benefit of the doubt).
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
        # ESCALATE_MARKER must appear alone on its own line — not inside prose.
        return bool(re.search(
            rf'^\s*{re.escape(ESCALATE_MARKER)}\s*$', output, re.MULTILINE
        ))
    except Exception:
        return False
