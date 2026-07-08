"""Difficulty-classification contract used by difficulty-based methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ...domain import RoutingEvent, Task


Difficulty = Literal["easy", "hard"]


@dataclass(frozen=True)
class DifficultyClassification:
    """A classifier's typed easy/hard assessment and call metadata."""

    difficulty: Difficulty
    reason: str
    model: str | None = None

    def __post_init__(self) -> None:
        if self.difficulty not in ("easy", "hard"):
            raise ValueError("difficulty must be easy or hard")
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason != self.reason.strip()
        ):
            raise ValueError("reason must be a non-empty trimmed string")
        if self.model is not None and (
            not isinstance(self.model, str)
            or not self.model
            or self.model != self.model.strip()
        ):
            raise ValueError("model must be a non-empty trimmed string or None")

    def to_routing_event(self) -> RoutingEvent:
        """Convert this assessment into a passive run-audit event."""

        return RoutingEvent(
            source="difficulty-classifier",
            outcome=self.difficulty,
            reason=self.reason,
            model=self.model,
        )


class DifficultyClassifier(Protocol):
    """A method collaborator that assesses task difficulty."""

    def classify(self, task: Task) -> DifficultyClassification:
        """Classify one task as easy or hard."""

        ...
