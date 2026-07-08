"""Immutable values passed through one smart-ask request."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

from ._numeric import is_finite_real


RouteAction = Literal["execute", "accept"]
RoutePhase = Literal["initial-easy", "initial-hard", "escalation", "fixed"]
FinishReason = Literal[
    "stop",
    "length",
    "refusal",
    "content_filter",
    "tool_call",
    "error",
    "unknown",
]
FINISH_REASONS: tuple[FinishReason, ...] = (
    "stop",
    "length",
    "refusal",
    "content_filter",
    "tool_call",
    "error",
    "unknown",
)
OutputStatus = Literal[
    "usable",
    "empty",
    "truncated",
    "refused",
    "unavailable",
]
OUTPUT_STATUSES: tuple[OutputStatus, ...] = (
    "usable",
    "empty",
    "truncated",
    "refused",
    "unavailable",
)


@dataclass(frozen=True)
class Task:
    """One user request submitted to the routing system."""

    prompt: str
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.task_id is not None and (
            not isinstance(self.task_id, str)
            or not self.task_id
            or self.task_id != self.task_id.strip()
        ):
            raise ValueError("task_id must be a non-empty trimmed string or None")


@dataclass(frozen=True)
class ExecutionRequest:
    """One provider-neutral request sent to a model execution backend.

    Token and temperature settings are optional hints. Adapters that cannot
    control those settings may ignore them.
    """

    model: str
    prompt: str
    role: str
    max_tokens: int | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model, str)
            or not self.model
            or self.model != self.model.strip()
        ):
            raise ValueError("model must be a non-empty trimmed string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if (
            not isinstance(self.role, str)
            or not self.role
            or self.role != self.role.strip()
        ):
            raise ValueError("role must be a non-empty trimmed string")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, Integral)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer or None")
        if self.max_tokens is not None:
            object.__setattr__(self, "max_tokens", int(self.max_tokens))
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not is_finite_real(self.temperature)
            or not 0 <= float(self.temperature) <= 2
        ):
            raise ValueError("temperature must be finite, between 0 and 2, or None")
        if self.temperature is not None:
            object.__setattr__(self, "temperature", float(self.temperature))


@dataclass(frozen=True)
class ModelResult:
    """One response produced by a model execution backend.

    Optional token counts and ``provider_cost_usd`` are provider evidence, not
    local estimates. ``None`` means the adapter could not observe that value.
    ``applied_max_tokens`` is the generation cap the adapter actually
    submitted, which can differ from the optional hint on
    :class:`ExecutionRequest`. ``output_status`` and ``max_tokens_reached`` are
    normalized from the visible response and finish metadata so callers cannot
    construct contradictory status fields.
    """

    model: str | None
    text: str
    usage: Any | None = None
    raw_text: str | None = None
    finish_reason: FinishReason = "unknown"
    native_finish_reason: str | None = None
    output_status: OutputStatus | None = None
    refusal: str | None = None
    applied_max_tokens: int | None = None
    max_tokens_reached: bool | None = None
    visible_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    provider_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.model is not None and (
            not isinstance(self.model, str)
            or not self.model
            or self.model != self.model.strip()
        ):
            raise ValueError("model must be a non-empty trimmed string or None")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise ValueError("raw_text must be a string or None")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"unknown finish_reason: {self.finish_reason!r}")
        if self.native_finish_reason is not None and (
            not isinstance(self.native_finish_reason, str)
            or not self.native_finish_reason
        ):
            raise ValueError(
                "native_finish_reason must be a non-empty string or None"
            )
        if self.refusal is not None and (
            not isinstance(self.refusal, str) or not self.refusal.strip()
        ):
            raise ValueError("refusal must be a non-empty string or None")

        for name in (
            "applied_max_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "applied_max_tokens" else 0
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                qualifier = "positive" if minimum else "non-negative"
                raise ValueError(f"{name} must be a {qualifier} integer or None")
            if value is not None:
                object.__setattr__(self, name, int(value))

        if self.provider_cost_usd is not None and (
            isinstance(self.provider_cost_usd, bool)
            or not isinstance(self.provider_cost_usd, Real)
            or not is_finite_real(self.provider_cost_usd)
            or self.provider_cost_usd < 0
        ):
            raise ValueError(
                "provider_cost_usd must be finite, non-negative, or None"
            )
        if self.provider_cost_usd is not None:
            object.__setattr__(
                self,
                "provider_cost_usd",
                float(self.provider_cost_usd),
            )

        expected_status: OutputStatus
        unavailable_output = self.output_status == "unavailable"
        if unavailable_output:
            if (
                self.text
                or self.raw_text is not None
                or self.refusal is not None
            ):
                raise ValueError(
                    "unavailable output_status contradicts response evidence"
                )
            expected_status = "unavailable"
        elif self.refusal is not None or self.finish_reason in (
            "refusal",
            "content_filter",
        ):
            expected_status = "refused"
        elif self.finish_reason == "length":
            expected_status = "truncated"
        elif self.text.strip():
            expected_status = "usable"
        else:
            expected_status = "empty"
        if self.output_status is None:
            object.__setattr__(self, "output_status", expected_status)
        elif self.output_status not in OUTPUT_STATUSES:
            raise ValueError(f"unknown output_status: {self.output_status!r}")
        elif self.output_status != expected_status:
            raise ValueError(
                "output_status contradicts text, refusal, or finish_reason"
            )
        if (
            self.output_status != "unavailable"
            and self.visible_output_tokens is not None
            and (self.visible_output_tokens == 0) is not self.output_empty
        ):
            raise ValueError(
                "visible_output_tokens contradict captured output emptiness"
            )

        expected_limit_status: bool | None
        if self.finish_reason == "length":
            expected_limit_status = True
        elif self.finish_reason == "unknown":
            expected_limit_status = None
        else:
            expected_limit_status = False
        if self.max_tokens_reached is None:
            object.__setattr__(
                self,
                "max_tokens_reached",
                expected_limit_status,
            )
        elif not isinstance(self.max_tokens_reached, bool):
            raise ValueError("max_tokens_reached must be a boolean or None")
        elif self.max_tokens_reached != expected_limit_status:
            raise ValueError("max_tokens_reached contradicts finish_reason")

    @property
    def output_empty(self) -> bool | None:
        """Whether captured visible text is empty, independently of termination."""

        if self.output_status == "unavailable":
            return None
        return not bool(self.text.strip())


@dataclass(frozen=True)
class RoutingEvent:
    """Passive audit record emitted while a routing method chooses a route."""

    source: str
    outcome: str
    reason: str
    model: str | None = None

    def __post_init__(self) -> None:
        for name in ("source", "outcome"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
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


@dataclass(frozen=True)
class RouteResult:
    """The next action selected by a routing method."""

    action: RouteAction
    model: str | None = None
    prompt: str | None = None
    role: str | None = None
    phase: RoutePhase | None = None
    label: str = ""
    routing_events: tuple[RoutingEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in ("execute", "accept"):
            raise ValueError(f"Unknown route action: {self.action!r}")
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        if not isinstance(self.routing_events, tuple):
            raise TypeError("routing_events must be a tuple")
        if any(not isinstance(event, RoutingEvent) for event in self.routing_events):
            raise TypeError("routing_events must contain RoutingEvent values")
        if self.action == "execute":
            if (
                not isinstance(self.model, str)
                or not self.model
                or self.model != self.model.strip()
            ):
                raise ValueError("An execute route requires a non-empty trimmed model")
            if not isinstance(self.prompt, str) or not self.prompt.strip():
                raise ValueError("An execute route requires a non-empty string prompt")
            if (
                not isinstance(self.role, str)
                or not self.role
                or self.role != self.role.strip()
            ):
                raise ValueError("An execute route requires an explicit semantic role")
            if self.phase not in (
                "initial-easy",
                "initial-hard",
                "escalation",
                "fixed",
            ):
                raise ValueError("An execute route requires an explicit phase")
        elif any(value is not None for value in (
            self.model,
            self.prompt,
            self.role,
            self.phase,
        )):
            raise ValueError("An accept route cannot contain execution fields")


@dataclass(frozen=True)
class Attempt:
    """A selected route paired with the model response it produced."""

    route: RouteResult
    result: ModelResult

    def __post_init__(self) -> None:
        if not isinstance(self.route, RouteResult) or self.route.action != "execute":
            raise ValueError("an attempt requires an execute RouteResult")
        if not isinstance(self.result, ModelResult):
            raise TypeError("result must be a ModelResult")


@dataclass(frozen=True)
class Context:
    """Per-task history available while a routing method chooses its next action."""

    attempts: tuple[Attempt, ...] = ()
    routing_events: tuple[RoutingEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(attempt, Attempt) for attempt in self.attempts
        ):
            raise TypeError("attempts must be a tuple of Attempt values")
        if not isinstance(self.routing_events, tuple) or any(
            not isinstance(event, RoutingEvent) for event in self.routing_events
        ):
            raise TypeError("routing_events must be a tuple of RoutingEvent values")

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

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            raise TypeError("task must be a Task")
        if not isinstance(self.attempts, tuple) or not self.attempts or any(
            not isinstance(attempt, Attempt) for attempt in self.attempts
        ):
            raise ValueError("attempts must be a non-empty tuple of Attempt values")
        if not isinstance(self.routing_events, tuple) or any(
            not isinstance(event, RoutingEvent) for event in self.routing_events
        ):
            raise TypeError("routing_events must be a tuple of RoutingEvent values")

    @property
    def final_result(self) -> ModelResult:
        """Return the final model response shown to the caller."""

        return self.attempts[-1].result

    @property
    def final_route(self) -> RouteResult:
        """Return the route that produced the final model response."""

        return self.attempts[-1].route
