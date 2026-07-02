"""Public response-escalation contracts and implementations."""

from .base import EscalationDecision, EscalationOutcome, EscalationPolicy
from .marker import (
    DEFAULT_ESCALATION_MARKER,
    DEFAULT_ESCALATION_PREFIX,
    DEFAULT_SELF_CHECK_SUFFIX,
    MarkerEscalationPolicy,
)

__all__ = [
    "DEFAULT_ESCALATION_MARKER",
    "DEFAULT_ESCALATION_PREFIX",
    "DEFAULT_SELF_CHECK_SUFFIX",
    "EscalationDecision",
    "EscalationOutcome",
    "EscalationPolicy",
    "MarkerEscalationPolicy",
]
