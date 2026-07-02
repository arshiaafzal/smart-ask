"""
Cascade orchestration — re-exports Gate 1 and Gate 2 in one place.

Import from here when you need both gates together:

    from methods.cascade import gate1_classify, gate2_preflight, EASY_MODEL, HARD_MODEL

Or import from the individual gate modules when you only need one:

    from methods.gate1 import gate1_classify, CLASSIFY_PROMPT
    from methods.gate2 import gate2_preflight, SELF_CHECK_SUFFIX, ESCALATE_MARKER
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
