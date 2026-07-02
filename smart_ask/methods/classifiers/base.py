"""Difficulty-classification contract used by difficulty-based methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ...domain import RoutingEvent, Task


Difficulty = Literal["easy", "hard"]


@dataclass(frozen=True)
class DifficultyClassification:
    """A classifier's typed easy/hard assessment and call metadata."""

    difficulty: Difficulty
    reason: str
    model: str | None = None
    usage: Any | None = None

    def to_routing_event(self) -> RoutingEvent:
        """Convert this assessment into a passive run-audit event."""

        return RoutingEvent(
            source="difficulty-classifier",
            outcome=self.difficulty,
            reason=self.reason,
            model=self.model,
            role="classifier" if self.model else None,
            usage=self.usage,
        )


class DifficultyClassifier(Protocol):
    """A method collaborator that assesses task difficulty."""

    def classify(self, task: Task) -> DifficultyClassification:
        """Classify one task as easy or hard."""

        ...
