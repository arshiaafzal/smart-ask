"""Marker-based response escalation policy."""

import re

from ...domain import ModelResult, Task
from .base import EscalationDecision


DEFAULT_SELF_CHECK_SUFFIX = (
    "\n\n---\n[SMART-ASK SELF-CHECK]\n"
    "After your answer, check:\n"
    "1. Does your code have stubs (NotImplementedError / pass as placeholder / # TODO)?\n"
    "2. If the task has visible >>> examples, does your code pass them?\n"
    "3. Is your answer only a high-level outline with no working code?\n"
    "If YES to any of the above: output the token ESCALATE_NOW alone on its own line.\n"
    "If your answer is complete and correct: no action needed, output nothing extra."
)
DEFAULT_ESCALATION_MARKER = "ESCALATE_NOW"
DEFAULT_ESCALATION_PREFIX = (
    "A previous attempt at this task was flagged as insufficient. "
    "Please solve this correctly and completely:\n\n"
)


class MarkerEscalationPolicy:
    """Ask for a marker, detect it exactly, and prepare an escalation retry."""

    def __init__(
        self,
        marker: str = DEFAULT_ESCALATION_MARKER,
        self_check_suffix: str = DEFAULT_SELF_CHECK_SUFFIX,
        escalation_prefix: str = DEFAULT_ESCALATION_PREFIX,
    ):
        self.marker = marker
        self.self_check_suffix = self_check_suffix
        self.escalation_prefix = escalation_prefix

    def prepare_candidate_prompt(self, task: Task) -> str:
        """Append the self-check instructions understood by this policy."""

        return task.prompt + self.self_check_suffix

    def assess(self, response: ModelResult) -> EscalationDecision:
        """Escalate only when the raw response contains a standalone marker line."""

        response_text = response.raw_text if response.raw_text is not None else response.text
        escalated = bool(re.search(
            rf"^\s*{re.escape(self.marker)}\s*$",
            response_text,
            re.MULTILINE,
        ))
        return EscalationDecision(
            outcome="escalate" if escalated else "accept",
            reason=(
                f"Response emitted {self.marker}"
                if escalated
                else "Response passed the escalation check"
            ),
        )

    def prepare_escalation_prompt(self, task: Task) -> str:
        """Prefix the original task with context for the hard-model retry."""

        return self.escalation_prefix + task.prompt
