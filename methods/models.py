"""
Model call functions — the only place in the codebase that calls model APIs.

Both the CLI and all benchmarks must import from here. There must be zero
model API calls in benchmarks — only evaluation harness code lives there.

Functions
---------
call_easy(prompt, client, system_prompt=None)
    Call EASY_MODEL (Gemini) with SELF_CHECK_SUFFIX appended.
    Returns (code: str, usage, escalated: bool).

call_hard(prompt, client, system_prompt=None, escalated=False)
    Call HARD_MODEL (Opus).
    Returns (code: str, usage).

Notes
-----
Both functions return the raw CompletionUsage object as `usage`.
Pass it directly to TokenTracker.record() — it contains exact token counts.
Code is returned with fences stripped and trailing whitespace removed.
"""

import re
from .gate1 import EASY_MODEL, HARD_MODEL
from .gate2 import SELF_CHECK_SUFFIX, ESCALATE_MARKER

DEFAULT_EASY_SYSTEM = (
    "You are an expert competitive programmer. "
    "Return ONLY the Python code — no explanation, no markdown fences, no extra text."
)
DEFAULT_HARD_SYSTEM = (
    "You are an expert competitive programmer. "
    "Write correct, complete Python code. "
    "Return ONLY the code — no explanation, no markdown fences, no extra text."
)


def call_easy(prompt: str, client, system_prompt: str = None, max_tokens: int = 1024):
    """
    Call EASY_MODEL (Gemini) with SELF_CHECK_SUFFIX appended.

    Returns (code: str, usage, escalated: bool).
      code      — model output with fences stripped, trailing whitespace removed,
                  leading whitespace preserved (critical for completion stitching)
      usage     — raw CompletionUsage; pass directly to TokenTracker.record()
      escalated — True when model emitted ESCALATE_MARKER alone on its own line
    """
    full = prompt + SELF_CHECK_SUFFIX
    r = client.chat.completions.create(
        model=EASY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_EASY_SYSTEM},
            {"role": "user",   "content": full},
        ],
        max_tokens=max_tokens, temperature=0.0,
    )
    raw = r.choices[0].message.content or ""
    escalated = bool(re.search(
        rf'^\s*{re.escape(ESCALATE_MARKER)}\s*$', raw, re.MULTILINE
    ))
    code_part = raw.split(ESCALATE_MARKER)[0] if escalated else raw
    return _strip_fences(code_part), r.usage, escalated


def call_hard(prompt: str, client, system_prompt: str = None, escalated: bool = False, max_tokens: int = 1024):
    """
    Call HARD_MODEL (Opus).

    Returns (code: str, usage).
      code  — model output with fences stripped
      usage — raw CompletionUsage; pass directly to TokenTracker.record()

    escalated=True prepends a brief failure hint so Opus knows a prior attempt
    was flagged as insufficient by the self-check gate.
    """
    user_msg = (
        "A previous attempt at this task was flagged as insufficient. "
        "Please solve this correctly and completely:\n\n" + prompt
        if escalated else prompt
    )
    r = client.chat.completions.create(
        model=HARD_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_HARD_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens, temperature=0.0,
    )
    return _strip_fences(r.choices[0].message.content or ""), r.usage


def call_hard_stream(prompt: str, client, system_prompt: str = None, escalated: bool = False, max_tokens: int = 4096):
    """
    Stream HARD_MODEL (Opus). Returns (generator, get_usage).

    generator  — yields str chunks as they arrive
    get_usage  — callable; returns CompletionUsage after generator is exhausted
                 (returns None if usage was not included in stream)

    escalated=True prepends a failure hint so Opus knows a prior attempt failed.
    max_tokens defaults to 4096 (larger than call_hard's 1024 — streaming is
    used from the CLI where real coding tasks need longer outputs).
    """
    user_msg = (
        "A previous attempt at this task was flagged as insufficient. "
        "Please solve this correctly and completely:\n\n" + prompt
        if escalated else prompt
    )
    usage_box = [None]

    def _gen():
        r = client.chat.completions.create(
            model=HARD_MODEL,
            messages=[
                {"role": "system", "content": system_prompt or DEFAULT_HARD_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=max_tokens, temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in r:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                usage_box[0] = chunk.usage

    return _gen(), lambda: usage_box[0]


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences (``` or ```python) from model output.
    Uses rstrip only — preserves leading indentation needed for completion stitching.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.rstrip()
