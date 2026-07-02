"""Response-escalation contract used by cascade methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ...domain import ModelResult, RoutingEvent, Task


EscalationOutcome = Literal["accept", "escalate"]


@dataclass(frozen=True)
class EscalationDecision:
    """A response policy's typed accept/escalate assessment."""

    outcome: EscalationOutcome
    reason: str

    @property
    def should_escalate(self) -> bool:
        """Return whether the response should be retried on the hard model."""

        return self.outcome == "escalate"

    def to_routing_event(self) -> RoutingEvent:
        """Convert this assessment into a passive run-audit event."""

        return RoutingEvent(
            source="response-escalation",
            outcome=self.outcome,
            reason=self.reason,
        )


class EscalationPolicy(Protocol):
    """Prepare and assess the easy-model attempt in a cascade."""

    def prepare_candidate_prompt(self, task: Task) -> str:
        """Prepare the prompt whose response may later be escalated."""

        ...

    def assess(self, response: ModelResult) -> EscalationDecision:
        """Decide whether the candidate response is acceptable."""

        ...

    def prepare_escalation_prompt(self, task: Task) -> str:
        """Prepare the retry prompt for the hard model."""

        ...
