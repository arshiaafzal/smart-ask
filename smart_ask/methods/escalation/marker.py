"""Marker-based response escalation policy."""

import re

from ...domain import ModelResult, Task
from .base import EscalationDecision


class MarkerEscalationPolicy:
    """Ask for a marker, detect it exactly, and prepare an escalation retry."""

    def __init__(
        self,
        marker: str,
        self_check_suffix: str,
        escalation_prefix: str,
    ):
        if (
            not isinstance(marker, str)
            or not marker
            or marker != marker.strip()
            or "\n" in marker
            or "\r" in marker
        ):
            raise ValueError("marker must be a non-empty, trimmed, single-line string")
        for name, value in (
            ("self_check_suffix", self_check_suffix),
            ("escalation_prefix", escalation_prefix),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        self._marker = marker
        self._self_check_suffix = self_check_suffix
        self._escalation_prefix = escalation_prefix

    def prepare_candidate_prompt(self, task: Task) -> str:
        """Append the self-check instructions understood by this policy."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        return task.prompt + self._self_check_suffix

    def assess(self, response: ModelResult) -> EscalationDecision:
        """Escalate only when the raw response contains a standalone marker line."""

        if not isinstance(response, ModelResult):
            raise TypeError("response must be a ModelResult")
        response_text = response.raw_text if response.raw_text is not None else response.text
        escalated = bool(re.search(
            rf"^\s*{re.escape(self._marker)}\s*$",
            response_text,
            re.MULTILINE,
        ))
        return EscalationDecision(
            outcome="escalate" if escalated else "accept",
            reason=(
                f"Response emitted {self._marker}"
                if escalated
                else "Response passed the escalation check"
            ),
        )

    def prepare_escalation_prompt(self, task: Task) -> str:
        """Prefix the original task with context for the hard-model retry."""

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        return self._escalation_prefix + task.prompt
