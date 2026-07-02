"""
Cascade orchestration — Gate 1 + Gate 2 pipeline.

cascade_solve() is the single entry point for benchmarks and the CLI.
It runs both gates and returns the result with all usage objects so
the caller can pass them to TokenTracker.record() for exact cost tracking.

Individual gate modules can still be imported directly when needed:
    from methods.gate1 import gate1_classify
    from methods.gate2 import gate2_preflight
    from methods.models import call_easy, call_hard
"""

from .gate1 import (
    OR_BASE,
    CLASSIFIER_MODEL,
    EASY_MODEL,
    HARD_MODEL,
    CLASSIFY_PROMPT,
    gate1_classify,
)

from .gate2 import (
    SELF_CHECK_SUFFIX,
    ESCALATE_MARKER,
    gate2_preflight,
)

from .models import (
    DEFAULT_EASY_SYSTEM,
    DEFAULT_HARD_SYSTEM,
    call_easy,
    call_hard,
)


def cascade_solve(
    prompt: str,
    client,
    easy_system: str = None,
    hard_system: str = None,
) -> dict:
    """
    Full two-gate cascade pipeline.

    Gate 1  gate1_classify       — routes to easy or hard
    Gate 2  embedded in call_easy — SELF_CHECK_SUFFIX + ESCALATE_MARKER detection

    Parameters
    ----------
    prompt       : task prompt sent to the model
    client       : shared OpenAI client (create once, reuse across calls)
    easy_system  : system prompt for Gemini; defaults to DEFAULT_EASY_SYSTEM
    hard_system  : system prompt for Opus;   defaults to DEFAULT_HARD_SYSTEM

    Returns dict
    ------------
    code      str   model output with fences stripped
    gate1     str   'easy' or 'hard'
    model     str   'gemini' | 'opus-G1' | 'opus-esc'
    escalated bool  True when Gate 2 triggered Opus escalation
    usages    list  [(model_id, role, usage_obj), ...]
                    pass each triple to TokenTracker.record() for exact costs
    """
    usages = []

    difficulty, g1_usage = gate1_classify(prompt, client)
    usages.append((CLASSIFIER_MODEL, "classifier", g1_usage))

    if difficulty == "hard":
        code, usage = call_hard(prompt, client, system_prompt=hard_system)
        usages.append((HARD_MODEL, "writer", usage))
        return {"code": code, "gate1": difficulty, "model": "opus-G1",
                "escalated": False, "usages": usages}

    code, usage, escalated = call_easy(prompt, client, system_prompt=easy_system)
    usages.append((EASY_MODEL, "generator", usage))

    if escalated:
        code, usage = call_hard(prompt, client, system_prompt=hard_system, escalated=True)
        usages.append((HARD_MODEL, "fixer", usage))
        return {"code": code, "gate1": difficulty, "model": "opus-esc",
                "escalated": True, "usages": usages}

    return {"code": code, "gate1": difficulty, "model": "gemini",
            "escalated": False, "usages": usages}
