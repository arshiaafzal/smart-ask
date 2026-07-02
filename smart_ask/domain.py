"""Immutable values passed through one smart-ask request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


RouteAction = Literal["execute", "accept"]
RoutePhase = Literal["initial-easy", "initial-hard", "escalation", "fixed"]


@dataclass(frozen=True)
class Task:
    """One user request submitted to the routing system."""

    prompt: str
    task_id: str | None = None


@dataclass(frozen=True)
class ExecutionRequest:
    """One provider-neutral request sent to a model execution backend.

    Token and temperature settings are optional hints. Adapters that cannot
    control those settings may ignore them.
    """

    model: str
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ModelResult:
    """One response produced by a model execution backend."""

    model: str
    text: str
    usage: Any | None = None
    returncode: int | None = 0
    raw_text: str | None = None


@dataclass(frozen=True)
class RoutingEvent:
    """Passive audit record emitted while a routing method chooses a route."""

    source: str
    outcome: str
    reason: str
    model: str | None = None
    role: str | None = None
    usage: Any | None = None


@dataclass(frozen=True)
class RouteResult:
    """The next action selected by a routing method."""

    action: RouteAction
    model: str | None = None
    prompt: str | None = None
    role: str = "writer"
    phase: RoutePhase | None = None
    label: str = ""
    routing_events: tuple[RoutingEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in ("execute", "accept"):
            raise ValueError(f"Unknown route action: {self.action!r}")
        if self.action == "execute" and (not self.model or self.prompt is None):
            raise ValueError("An execute route requires both model and prompt")


@dataclass(frozen=True)
class Attempt:
    """A selected route paired with the model response it produced."""

    route: RouteResult
    result: ModelResult


@dataclass(frozen=True)
class Context:
    """Per-task history available while a routing method chooses its next action."""

    attempts: tuple[Attempt, ...] = ()
    routing_events: tuple[RoutingEvent, ...] = ()

    @property
    def previous_attempt(self) -> ModelResult | None:
        """Return the latest model response, if the task has been attempted."""

        return self.attempts[-1].result if self.attempts else None

    @property
    def previous_route(self) -> RouteResult | None:
        """Return the route used for the latest attempt, if one exists."""

        return self.attempts[-1].route if self.attempts else None


@dataclass(frozen=True)
class RunResult:
    """Complete audit trail for one user task and its final response."""

    task: Task
    attempts: tuple[Attempt, ...]
    routing_events: tuple[RoutingEvent, ...]

    @property
    def final_result(self) -> ModelResult:
        """Return the final model response shown to the caller."""

        if not self.attempts:
            raise RuntimeError("The task completed without a model attempt")
        return self.attempts[-1].result

    @property
    def final_route(self) -> RouteResult:
        """Return the route that produced the final model response."""

        if not self.attempts:
            raise RuntimeError("The task completed without a model attempt")
        return self.attempts[-1].route
