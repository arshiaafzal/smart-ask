"""
Gate 1 — Binary task classifier.

What it does
------------
Sends the task prompt to a cheap Gemini model and gets back a single JSON
label that decides which model handles the session:

  {"d": "easy"} → route to EASY_MODEL (Gemini, cheap)
                   then run Gate 2 to verify the answer quality
  {"d": "hard"} → skip Gate 2, route directly to HARD_MODEL (Opus)

Why a pre-flight classifier?
-----------------------------
Most coding tasks are straightforward. Paying Opus prices for every request
wastes ~80% of the budget. Gate 1 filters out the simple work in ~100ms for
roughly $0.000001 per call — well under 1 cent per 1000 classifications.

Accuracy on HumanEval (164 problems)
--------------------------------------
  Classified easy → 135 / 164  (82%)
  Classified hard →  29 / 164  (18%)
  Gate 1 pass@1 accuracy matches the product benchmark at 89.0%
"""

import json

# ── Models ────────────────────────────────────────────────────────────────────

OR_BASE          = "https://openrouter.ai/api/v1"
CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"
EASY_MODEL       = "google/gemini-2.5-flash-lite"
HARD_MODEL       = "anthropic/claude-opus-4.8"

# ── Classifier prompt ─────────────────────────────────────────────────────────
# Tuned to minimise false-hard classifications (expensive) while catching
# genuine hard problems before Gemini fails at them.

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


# ── Gate 1 function ───────────────────────────────────────────────────────────

def gate1_classify(prompt: str, client):
    """
    Classify prompt as 'easy' or 'hard' using CLASSIFIER_MODEL.

    Parameters
    ----------
    client : openai.OpenAI
        Shared client — create once with OpenAI(base_url=OR_BASE, api_key=key)
        and reuse across calls (thread-safe, avoids redundant client construction).

    Returns (difficulty: str, usage: object | None).
    `usage` is the raw OpenAI CompletionUsage object — pass it directly to
    TokenTracker.record() for exact cost tracking.

    Never raises: on any error defaults to 'easy' with usage=None.
    """
    try:
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
