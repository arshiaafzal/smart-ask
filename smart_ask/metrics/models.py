"""Immutable values and aggregation for call and run metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal, TYPE_CHECKING

from ..domain import (
    FINISH_REASONS,
    OUTPUT_STATUSES,
    FinishReason,
    OutputStatus,
)
from .cost import PriceCatalog, PriceQuote
from .._numeric import checked_fsum, is_finite_real

if TYPE_CHECKING:
    from ..domain import RunResult


CallStatus = Literal["ok", "error"]
UsageStatus = Literal["complete", "total_only", "partial", "unavailable", "error"]
TelemetryStatus = Literal["complete", "partial", "error"]
ErrorCategory = Literal[
    "timeout",
    "rate_limit",
    "authentication",
    "provider_5xx",
    "invalid_response",
    "unknown",
]
ERROR_CATEGORIES: tuple[ErrorCategory, ...] = (
    "timeout",
    "rate_limit",
    "authentication",
    "provider_5xx",
    "invalid_response",
    "unknown",
)
TaskOutcome = Literal[
    "passed",
    "incorrect",
    "routing_error",
    "execution_error",
    "evaluation_error",
    "unrated",
]

TASK_OUTCOMES: tuple[TaskOutcome, ...] = (
    "passed",
    "incorrect",
    "routing_error",
    "execution_error",
    "evaluation_error",
    "unrated",
)
OUTPUT_EMPTINESS: tuple[str, ...] = ("empty", "nonempty", "unknown")

_TIMING_TOLERANCE_MS = 1e-6


@dataclass(frozen=True)
class TokenUsage:
    """Provider-neutral token evidence for one model call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    visible_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
            if value is not None:
                object.__setattr__(self, name, int(value))

        prompt = self.prompt_tokens
        completion = self.completion_tokens
        total = self.total_tokens
        if prompt is not None and completion is not None:
            expected = prompt + completion
            if total is None:
                object.__setattr__(self, "total_tokens", expected)
            elif total != expected:
                raise ValueError("total_tokens must equal prompt + completion tokens")
        elif total is not None:
            known_component = prompt if prompt is not None else completion
            if known_component is not None and known_component > total:
                raise ValueError("a token component cannot exceed total_tokens")

        completion_capacity = (
            completion
            if completion is not None
            else total - prompt
            if total is not None and prompt is not None
            else total
        )
        if completion_capacity is not None:
            for name in ("visible_output_tokens", "reasoning_tokens"):
                value = getattr(self, name)
                if value is not None and value > completion_capacity:
                    raise ValueError(
                        f"{name} cannot exceed available completion tokens"
                    )
            if (
                self.visible_output_tokens is not None
                and self.reasoning_tokens is not None
                and self.visible_output_tokens + self.reasoning_tokens
                > completion_capacity
            ):
                raise ValueError(
                    "visible_output_tokens + reasoning_tokens cannot exceed "
                    "available completion tokens"
                )
        prompt_capacity = (
            prompt
            if prompt is not None
            else total - completion
            if total is not None and completion is not None
            else total
        )
        if prompt_capacity is not None:
            for name in ("cached_input_tokens", "cache_write_input_tokens"):
                value = getattr(self, name)
                if value is not None and value > prompt_capacity:
                    raise ValueError(
                        f"{name} cannot exceed available prompt tokens"
                    )
            if (
                self.cached_input_tokens is not None
                and self.cache_write_input_tokens is not None
                and self.cached_input_tokens + self.cache_write_input_tokens
                > prompt_capacity
            ):
                raise ValueError(
                    "cached_input_tokens + cache_write_input_tokens cannot "
                    "exceed available prompt tokens"
                )
        if total is not None:
            known_details = sum(
                int(getattr(self, name) or 0)
                for name in (
                    "visible_output_tokens",
                    "reasoning_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                )
            )
            if known_details > total:
                raise ValueError(
                    "known input and output token details cannot exceed "
                    "total_tokens"
                )

    @property
    def total_complete(self) -> bool:
        """Whether an authoritative total token count is known."""

        return self.total_tokens is not None

    @property
    def breakdown_complete(self) -> bool:
        """Whether both input and output counts are known."""

        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def any_known(self) -> bool:
        return any(value is not None for value in (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.visible_output_tokens,
            self.reasoning_tokens,
            self.cached_input_tokens,
            self.cache_write_input_tokens,
        ))

    def to_dict(self) -> dict[str, Any]:
        from .wire import _token_usage_to_dict

        return _token_usage_to_dict(self)


@dataclass(frozen=True)
class CallStats:
    """Accounting, timing, and telemetry state for one executor invocation."""

    run_id: str
    call_id: str
    ordinal: int
    channel: str
    role: str
    requested_model: str
    actual_model: str | None
    priced_model: str | None
    status: CallStatus
    latency_ms: float
    started_offset_ms: float
    usage: TokenUsage
    usage_status: UsageStatus
    price_quote: PriceQuote
    finish_reason: FinishReason
    output_status: OutputStatus | None
    output_empty: bool | None
    requested_max_tokens: int | None
    applied_max_tokens: int | None
    max_tokens_reached: bool | None
    provider_cost_usd: float | None
    native_finish_reason: str | None = None
    refusal: str | None = None
    usage_diagnostic: str | None = None
    error_category: ErrorCategory | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "call_id",
            "channel",
            "role",
            "requested_model",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, Integral):
            raise ValueError("ordinal must be a positive integer")
        if self.ordinal < 1:
            raise ValueError("ordinal must be a positive integer")
        object.__setattr__(self, "ordinal", int(self.ordinal))

        for name in ("latency_ms", "started_offset_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not is_finite_real(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))

        if not isinstance(self.price_quote, PriceQuote):
            raise TypeError("price_quote must be a PriceQuote")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage")

        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"unknown finish_reason: {self.finish_reason!r}")
        if (
            self.output_status is not None
            and self.output_status not in OUTPUT_STATUSES
        ):
            raise ValueError(f"unknown output_status: {self.output_status!r}")
        if self.output_empty is not None and not isinstance(
            self.output_empty,
            bool,
        ):
            raise ValueError("output_empty must be a boolean or None")
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
        for name in ("requested_max_tokens", "applied_max_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be positive or None")
            if value is not None:
                object.__setattr__(self, name, int(value))
        if self.max_tokens_reached is not None and not isinstance(
            self.max_tokens_reached,
            bool,
        ):
            raise ValueError("max_tokens_reached must be a boolean or None")
        if self.finish_reason == "length" and self.max_tokens_reached is not True:
            raise ValueError("length finish_reason requires max_tokens_reached=true")
        if (
            self.finish_reason not in ("length", "unknown")
            and self.max_tokens_reached is not False
        ):
            raise ValueError(
                "known non-length finish_reason requires max_tokens_reached=false"
            )
        if self.finish_reason == "unknown" and self.max_tokens_reached is not None:
            raise ValueError(
                "unknown finish_reason requires unknown max_tokens_reached"
            )
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

        if self.status == "ok":
            if self.actual_model is not None and (
                not isinstance(self.actual_model, str)
                or not self.actual_model
                or self.actual_model != self.actual_model.strip()
            ):
                raise ValueError("actual model must be non-empty, trimmed, or None")
            if (
                not isinstance(self.priced_model, str)
                or not self.priced_model
                or self.priced_model != self.priced_model.strip()
            ):
                raise ValueError("a successful call requires a trimmed priced model")
            expected_priced_model = self.actual_model or self.requested_model
            if self.priced_model != expected_priced_model:
                raise ValueError(
                    "priced_model must equal actual_model when available, "
                    "otherwise requested_model"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("a successful call cannot contain an execution error")
            if self.output_status is None:
                raise ValueError("a successful call requires an output_status")
            if self.error_category is not None:
                raise ValueError("a successful call cannot contain an error_category")
            if self.output_status == "unavailable":
                if self.output_empty is not None or self.refusal is not None:
                    raise ValueError(
                        "unavailable output requires unknown emptiness and no refusal text"
                    )
            else:
                if self.output_empty is None:
                    raise ValueError(
                        "captured output requires known output_empty evidence"
                    )
                refusal_evidenced = (
                    self.refusal is not None
                    or self.finish_reason in ("refusal", "content_filter")
                )
                expected_status = (
                    "refused"
                    if refusal_evidenced
                    else "truncated"
                    if self.finish_reason == "length"
                    else "empty"
                    if self.output_empty
                    else "usable"
                )
                if self.output_status != expected_status:
                    raise ValueError(
                        "output_status contradicts finish, refusal, or emptiness evidence"
                    )
                if (
                    self.usage.visible_output_tokens is not None
                    and (self.usage.visible_output_tokens == 0)
                    is not self.output_empty
                ):
                    raise ValueError(
                        "visible_output_tokens contradict output_empty evidence"
                    )
        elif self.status == "error":
            if self.actual_model is not None or self.priced_model is not None:
                raise ValueError("a failed call cannot contain actual or priced models")
            if (
                not isinstance(self.error_type, str)
                or not self.error_type
                or self.error_type != self.error_type.strip()
            ):
                raise ValueError("a failed call requires a trimmed error_type")
            if not isinstance(self.error_message, str):
                raise ValueError("a failed call requires error_message")
            if self.usage.any_known or self.usage_status != "unavailable":
                raise ValueError("a failed call cannot contain token usage")
            if self.price_quote.status != "unavailable":
                raise ValueError("a failed call cannot contain a priced cost state")
            if (
                self.finish_reason != "error"
                or self.output_status is not None
                or self.output_empty is not None
            ):
                raise ValueError(
                    "a failed call requires error finish_reason and no output evidence"
                )
            if self.max_tokens_reached is not False:
                raise ValueError(
                    "a failed call requires max_tokens_reached=false"
                )
            if self.native_finish_reason is not None or self.refusal is not None:
                raise ValueError("a failed call cannot contain response evidence")
            if self.provider_cost_usd is not None:
                raise ValueError("a failed call cannot contain provider cost")
            if self.applied_max_tokens is not None:
                raise ValueError(
                    "a failed call cannot claim an applied token limit"
                )
            if self.error_category not in ERROR_CATEGORIES:
                raise ValueError("a failed call requires an error_category")
        else:
            raise ValueError(f"unknown call status: {self.status!r}")

        if self.usage_status not in (
            "complete",
            "total_only",
            "partial",
            "unavailable",
            "error",
        ):
            raise ValueError(f"unknown usage status: {self.usage_status!r}")

        expected_usage_status: UsageStatus
        if not self.usage.any_known:
            if self.usage_status not in ("unavailable", "error"):
                raise ValueError(
                    f"{self.usage_status} usage requires token evidence"
                )
            expected_usage_status = self.usage_status
        elif self.usage.breakdown_complete:
            expected_usage_status = "complete"
        elif (
            self.usage.total_complete
            and self.usage.prompt_tokens is None
            and self.usage.completion_tokens is None
        ):
            expected_usage_status = "total_only"
        elif self.usage.any_known:
            expected_usage_status = "partial"
        else:
            expected_usage_status = "partial"

        if self.usage_status in ("unavailable", "error"):
            if self.usage.any_known:
                raise ValueError(
                    f"{self.usage_status} usage cannot contain token counts"
                )
        elif self.usage_status != expected_usage_status:
            raise ValueError(
                f"usage_status {self.usage_status!r} contradicts token evidence"
            )
        if self.usage_status == "complete":
            if self.usage_diagnostic is not None:
                raise ValueError("complete usage cannot contain a diagnostic")
        elif (
            not isinstance(self.usage_diagnostic, str)
            or not self.usage_diagnostic
            or self.usage_diagnostic != self.usage_diagnostic.strip()
        ):
            raise ValueError(
                f"{self.usage_status} usage requires a non-empty trimmed diagnostic"
            )

    @property
    def telemetry_status(self) -> TelemetryStatus:
        if self.usage_status == "error" or self.price_quote.status == "error":
            return "error"
        if self.usage_status == "complete" and self.price_quote.status == "priced":
            return "complete"
        return "partial"

    def to_dict(self) -> dict[str, Any]:
        from .wire import _call_stats_to_dict

        return _call_stats_to_dict(self)


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _normalized_counter_items(
    items: tuple[tuple[str, int], ...],
    name: str,
) -> dict[str, int]:
    try:
        raw = dict(items)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain key/count pairs") from exc
    counts: dict[str, int] = {}
    for key, count in raw.items():
        if (
            not isinstance(key, str)
            or not key
            or key != key.strip()
        ):
            raise TypeError(f"{name} keys must be non-empty trimmed strings")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise TypeError(f"{name} counts must be positive integers")
        counts[key] = int(count)
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class RunStats:
    """Immutable metrics snapshot for one task, turn, or benchmark case."""

    run_id: str
    task_id: str | None
    duration_ms: float
    calls: tuple[CallStats, ...]
    strategy_id: str | None = None
    generation_attempts: int = 0
    routing_events: int = 0
    outcome: TaskOutcome = "unrated"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or self.run_id != self.run_id.strip()
        ):
            raise ValueError("run_id must be a non-empty trimmed string")
        if self.task_id is not None and (
            not isinstance(self.task_id, str)
            or not self.task_id
            or self.task_id != self.task_id.strip()
        ):
            raise ValueError("task_id must be non-empty, trimmed, or None")
        if self.strategy_id is not None and (
            not isinstance(self.strategy_id, str)
            or not self.strategy_id
            or self.strategy_id != self.strategy_id.strip()
        ):
            raise ValueError("strategy_id must be non-empty, trimmed, or None")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, Real)
            or not is_finite_real(self.duration_ms)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be finite and non-negative")
        object.__setattr__(self, "duration_ms", float(self.duration_ms))
        if not isinstance(self.calls, tuple):
            raise TypeError("calls must be a tuple")
        if any(not isinstance(call, CallStats) for call in self.calls):
            raise TypeError("calls must contain CallStats values")
        ordinals = tuple(call.ordinal for call in self.calls)
        if ordinals != tuple(range(1, len(self.calls) + 1)):
            raise ValueError("call ordinals must be unique, ordered, and contiguous")
        call_ids = tuple(call.call_id for call in self.calls)
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("call_id values must be unique within a run")
        if any(call.run_id != self.run_id for call in self.calls):
            raise ValueError("every call run_id must match its containing run")
        if self.calls and max(
            call.started_offset_ms + call.latency_ms for call in self.calls
        ) > self.duration_ms + _TIMING_TOLERANCE_MS:
            raise ValueError("call timing cannot extend beyond run duration")
        for name in ("generation_attempts", "routing_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.generation_attempts > len(self.calls):
            raise ValueError("generation_attempts cannot exceed interactions")
        if self.outcome not in TASK_OUTCOMES:
            raise ValueError(f"unknown task outcome: {self.outcome!r}")

    @property
    def interaction_count(self) -> int:
        return len(self.calls)

    @property
    def failed_interactions(self) -> int:
        return sum(call.status == "error" for call in self.calls)

    @property
    def missing_total_usage_calls(self) -> int:
        return sum(not call.usage.total_complete for call in self.calls)

    @property
    def missing_usage_breakdown_calls(self) -> int:
        return sum(not call.usage.breakdown_complete for call in self.calls)

    @property
    def usage_error_calls(self) -> int:
        return sum(call.usage_status == "error" for call in self.calls)

    @property
    def total_usage_complete(self) -> bool:
        return self.missing_total_usage_calls == 0

    @property
    def usage_breakdown_complete(self) -> bool:
        return self.missing_usage_breakdown_calls == 0

    @property
    def known_prompt_tokens(self) -> int:
        return sum(int(call.usage.prompt_tokens or 0) for call in self.calls)

    @property
    def known_completion_tokens(self) -> int:
        return sum(int(call.usage.completion_tokens or 0) for call in self.calls)

    @property
    def known_total_tokens(self) -> int:
        return sum(int(call.usage.total_tokens or 0) for call in self.calls)

    @property
    def known_visible_output_tokens(self) -> int:
        return sum(
            int(call.usage.visible_output_tokens or 0) for call in self.calls
        )

    @property
    def known_reasoning_tokens(self) -> int:
        return sum(int(call.usage.reasoning_tokens or 0) for call in self.calls)

    @property
    def known_cached_input_tokens(self) -> int:
        return sum(
            int(call.usage.cached_input_tokens or 0) for call in self.calls
        )

    @property
    def known_cache_write_input_tokens(self) -> int:
        return sum(
            int(call.usage.cache_write_input_tokens or 0) for call in self.calls
        )

    def _missing_usage_detail_calls(self, field: str) -> int:
        return sum(getattr(call.usage, field) is None for call in self.calls)

    @property
    def missing_visible_output_usage_calls(self) -> int:
        return self._missing_usage_detail_calls("visible_output_tokens")

    @property
    def missing_reasoning_usage_calls(self) -> int:
        return self._missing_usage_detail_calls("reasoning_tokens")

    @property
    def missing_cached_input_usage_calls(self) -> int:
        return self._missing_usage_detail_calls("cached_input_tokens")

    @property
    def missing_cache_write_input_usage_calls(self) -> int:
        return self._missing_usage_detail_calls("cache_write_input_tokens")

    @property
    def total_tokens(self) -> int | None:
        return self.known_total_tokens if self.total_usage_complete else None

    @property
    def missing_cost_calls(self) -> int:
        return sum(call.price_quote.cost_usd is None for call in self.calls)

    @property
    def pricing_error_calls(self) -> int:
        return sum(call.price_quote.status == "error" for call in self.calls)

    @property
    def cost_complete(self) -> bool:
        return self.missing_cost_calls == 0

    @property
    def known_cost_usd(self) -> float:
        return checked_fsum(
            (
                float(call.price_quote.cost_usd)
                for call in self.calls
                if call.price_quote.cost_usd is not None
            ),
            name="run aggregate catalog cost",
        )

    @property
    def total_cost_usd(self) -> float | None:
        return self.known_cost_usd if self.cost_complete else None

    @property
    def missing_provider_cost_calls(self) -> int:
        return sum(call.provider_cost_usd is None for call in self.calls)

    @property
    def provider_cost_complete(self) -> bool:
        return self.missing_provider_cost_calls == 0

    @property
    def known_provider_cost_usd(self) -> float:
        return checked_fsum(
            (
                float(call.provider_cost_usd)
                for call in self.calls
                if call.provider_cost_usd is not None
            ),
            name="run aggregate provider-reported cost",
        )

    @property
    def total_provider_cost_usd(self) -> float | None:
        return (
            self.known_provider_cost_usd
            if self.provider_cost_complete
            else None
        )

    @property
    def interactions_by_channel(self) -> dict[str, int]:
        return _counts(call.channel for call in self.calls)

    @property
    def interactions_by_role(self) -> dict[str, int]:
        return _counts(call.role for call in self.calls)

    @property
    def interactions_by_requested_model(self) -> dict[str, int]:
        return _counts(call.requested_model for call in self.calls)

    @property
    def interactions_by_actual_model(self) -> dict[str, int]:
        return _counts(
            call.actual_model
            for call in self.calls
            if call.actual_model is not None
        )

    @property
    def interactions_by_priced_model(self) -> dict[str, int]:
        return _counts(
            call.priced_model
            for call in self.calls
            if call.priced_model is not None
        )

    @property
    def priced_calls_by_source(self) -> dict[str, int]:
        return _counts(
            call.price_quote.source
            for call in self.calls
            if (
                call.price_quote.status == "priced"
                and call.price_quote.source is not None
            )
        )

    @property
    def finish_reasons(self) -> dict[str, int]:
        return _counts(call.finish_reason for call in self.calls)

    @property
    def output_statuses(self) -> dict[str, int]:
        return _counts(
            call.output_status or "unavailable" for call in self.calls
        )

    @property
    def output_emptiness(self) -> dict[str, int]:
        return _counts(
            "unknown"
            if call.output_empty is None
            else "empty"
            if call.output_empty
            else "nonempty"
            for call in self.calls
        )

    @property
    def error_categories(self) -> dict[str, int]:
        return _counts(
            call.error_category
            for call in self.calls
            if call.error_category is not None
        )

    @property
    def max_tokens_reached_calls(self) -> int:
        return sum(call.max_tokens_reached is True for call in self.calls)

    @property
    def price_catalogs(self) -> tuple[PriceCatalog, ...]:
        catalogs: dict[str, PriceCatalog] = {}
        for call in self.calls:
            catalog = call.price_quote.catalog
            if catalog is None:
                continue
            previous = catalogs.setdefault(catalog.catalog_id, catalog)
            if previous != catalog:
                raise ValueError(
                    f"conflicting price catalogs share id {catalog.catalog_id!r}"
                )
        return tuple(catalogs[key] for key in sorted(catalogs))

    def with_run_result(self, run: "RunResult") -> "RunStats":
        """Attach routing-level counts without recounting model usage."""

        return replace(
            self,
            generation_attempts=len(run.attempts),
            routing_events=len(run.routing_events),
        )

    def with_routing_counts(
        self,
        *,
        generation_attempts: int,
        routing_events: int,
    ) -> "RunStats":
        return replace(
            self,
            generation_attempts=generation_attempts,
            routing_events=routing_events,
        )

    def with_outcome(self, outcome: TaskOutcome) -> "RunStats":
        """Return this immutable snapshot with an explicit task outcome."""

        return replace(self, outcome=outcome)

    def to_dict(self, *, include_calls: bool = True) -> dict[str, Any]:
        from .wire import _run_stats_to_dict

        return _run_stats_to_dict(self, include_calls=include_calls)


@dataclass(frozen=True)
class StatsSummary:
    """A standard aggregate with the same wire envelope as ``RunStats``."""

    runs: int
    interactions: int
    failed_interactions: int
    interactions_by_channel_items: tuple[tuple[str, int], ...]
    interactions_by_role_items: tuple[tuple[str, int], ...]
    interactions_by_requested_model_items: tuple[tuple[str, int], ...]
    interactions_by_actual_model_items: tuple[tuple[str, int], ...]
    interactions_by_priced_model_items: tuple[tuple[str, int], ...]
    generation_attempts: int
    routing_events: int
    known_prompt_tokens: int
    known_completion_tokens: int
    known_total_tokens: int
    known_visible_output_tokens: int
    known_reasoning_tokens: int
    known_cached_input_tokens: int
    known_cache_write_input_tokens: int
    missing_total_usage_calls: int
    missing_usage_breakdown_calls: int
    missing_visible_output_usage_calls: int
    missing_reasoning_usage_calls: int
    missing_cached_input_usage_calls: int
    missing_cache_write_input_usage_calls: int
    usage_error_calls: int
    known_cost_usd: float
    missing_cost_calls: int
    pricing_error_calls: int
    known_provider_cost_usd: float
    missing_provider_cost_calls: int
    priced_calls_by_source_items: tuple[tuple[str, int], ...]
    finish_reasons_items: tuple[tuple[str, int], ...]
    output_statuses_items: tuple[tuple[str, int], ...]
    output_emptiness_items: tuple[tuple[str, int], ...]
    error_categories_items: tuple[tuple[str, int], ...]
    outcome_counts_items: tuple[tuple[str, int], ...]
    max_tokens_reached_calls: int
    price_catalogs: tuple[PriceCatalog, ...]
    cumulative_run_duration_ms: float

    def __post_init__(self) -> None:
        integer_fields = (
            "runs",
            "interactions",
            "failed_interactions",
            "generation_attempts",
            "routing_events",
            "known_prompt_tokens",
            "known_completion_tokens",
            "known_total_tokens",
            "known_visible_output_tokens",
            "known_reasoning_tokens",
            "known_cached_input_tokens",
            "known_cache_write_input_tokens",
            "missing_total_usage_calls",
            "missing_usage_breakdown_calls",
            "missing_visible_output_usage_calls",
            "missing_reasoning_usage_calls",
            "missing_cached_input_usage_calls",
            "missing_cache_write_input_usage_calls",
            "usage_error_calls",
            "missing_cost_calls",
            "pricing_error_calls",
            "missing_provider_cost_calls",
            "max_tokens_reached_calls",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "known_cost_usd",
            "known_provider_cost_usd",
            "cumulative_run_duration_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not is_finite_real(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))

        counter_fields = (
            "interactions_by_channel_items",
            "interactions_by_role_items",
            "interactions_by_requested_model_items",
            "interactions_by_actual_model_items",
            "interactions_by_priced_model_items",
            "priced_calls_by_source_items",
            "finish_reasons_items",
            "output_statuses_items",
            "output_emptiness_items",
            "error_categories_items",
            "outcome_counts_items",
        )
        for name in counter_fields:
            items = getattr(self, name)
            if not isinstance(items, tuple):
                raise TypeError(f"{name} must be a tuple")
            normalized = _normalized_counter_items(items, name)
            if tuple(normalized.items()) != items:
                raise ValueError(f"{name} must be unique and sorted")
            object.__setattr__(self, name, tuple(normalized.items()))

        if self.failed_interactions > self.interactions:
            raise ValueError("failed_interactions cannot exceed interactions")
        for counts in (
            self.interactions_by_channel,
            self.interactions_by_role,
            self.interactions_by_requested_model,
        ):
            if sum(counts.values()) != self.interactions:
                raise ValueError(
                    "complete interaction counters must sum to interactions"
                )
        if sum(self.interactions_by_actual_model.values()) > (
            self.interactions - self.failed_interactions
        ):
            raise ValueError("actual-model interactions cannot exceed successful calls")
        if sum(self.interactions_by_priced_model.values()) != (
            self.interactions - self.failed_interactions
        ):
            raise ValueError("priced-model interactions must count successful calls")
        if sum(self.finish_reasons.values()) != self.interactions:
            raise ValueError("finish reasons must count every interaction")
        if sum(self.output_statuses.values()) != self.interactions:
            raise ValueError("output statuses must count every interaction")
        if sum(self.output_emptiness.values()) != self.interactions:
            raise ValueError("output emptiness must count every interaction")
        if sum(self.error_categories.values()) != self.failed_interactions:
            raise ValueError("error categories must count every failed interaction")
        if set(self.finish_reasons) - set(FINISH_REASONS):
            raise ValueError("finish reasons contain an unknown value")
        if set(self.output_statuses) - set(OUTPUT_STATUSES):
            raise ValueError("output statuses contain an unknown value")
        if set(self.output_emptiness) - set(OUTPUT_EMPTINESS):
            raise ValueError("output emptiness contains an unknown value")
        if set(self.error_categories) - set(ERROR_CATEGORIES):
            raise ValueError("error categories contain an unknown value")
        if self.finish_reasons.get("error", 0) < self.failed_interactions:
            raise ValueError("failed interactions require error finish reasons")
        if self.output_statuses.get("unavailable", 0) < self.failed_interactions:
            raise ValueError("failed interactions require unavailable output status")
        if self.output_statuses.get("unavailable", 0) != (
            self.output_emptiness.get("unknown", 0)
        ):
            raise ValueError(
                "unavailable output statuses must match unknown output emptiness"
            )
        if self.output_statuses.get("empty", 0) > self.output_emptiness.get(
            "empty",
            0,
        ):
            raise ValueError("empty output statuses require empty captured output")
        if self.output_statuses.get("usable", 0) > self.output_emptiness.get(
            "nonempty",
            0,
        ):
            raise ValueError("usable output statuses require nonempty captured output")
        if self.output_statuses.get("truncated", 0) > self.finish_reasons.get(
            "length",
            0,
        ):
            raise ValueError("truncated output statuses require length finish reasons")
        constrained_finishes = (
            self.finish_reasons.get("length", 0)
            + self.finish_reasons.get("refusal", 0)
            + self.finish_reasons.get("content_filter", 0)
            + self.failed_interactions
        )
        compatible_status_capacity = (
            self.output_statuses.get("truncated", 0)
            + self.output_statuses.get("refused", 0)
            + self.output_statuses.get("unavailable", 0)
        )
        if constrained_finishes > compatible_status_capacity:
            raise ValueError(
                "finish reasons cannot be assigned to compatible output statuses"
            )
        if self.max_tokens_reached_calls != self.finish_reasons.get("length", 0):
            raise ValueError(
                "max_tokens_reached_calls must match length finish reasons"
            )
        if set(self.outcome_counts) - set(TASK_OUTCOMES):
            raise ValueError("outcome counts contain an unknown task outcome")
        if sum(self.outcome_counts.values()) != self.runs:
            raise ValueError("task outcomes must count every run")
        if self.missing_total_usage_calls > self.missing_usage_breakdown_calls:
            raise ValueError("missing total usage must also lack a breakdown")
        if max(
            self.missing_total_usage_calls,
            self.missing_usage_breakdown_calls,
            self.missing_visible_output_usage_calls,
            self.missing_reasoning_usage_calls,
            self.missing_cached_input_usage_calls,
            self.missing_cache_write_input_usage_calls,
            self.usage_error_calls,
            self.missing_cost_calls,
            self.pricing_error_calls,
            self.missing_provider_cost_calls,
            self.max_tokens_reached_calls,
        ) > self.interactions:
            raise ValueError("completeness counts cannot exceed interactions")
        if self.usage_error_calls > self.missing_total_usage_calls:
            raise ValueError("usage errors must lack total-token evidence")
        if min(
            self.missing_total_usage_calls,
            self.missing_usage_breakdown_calls,
            self.missing_visible_output_usage_calls,
            self.missing_reasoning_usage_calls,
            self.missing_cached_input_usage_calls,
            self.missing_cache_write_input_usage_calls,
            self.missing_cost_calls,
            self.missing_provider_cost_calls,
        ) < self.failed_interactions:
            raise ValueError(
                "failed interactions require missing usage and cost evidence"
            )
        if (
            self.missing_total_usage_calls == self.interactions
            and self.known_total_tokens != 0
        ):
            raise ValueError("known total tokens require total-token evidence")
        if self.missing_total_usage_calls == 0:
            if (
                self.known_prompt_tokens + self.known_completion_tokens
                > self.known_total_tokens
            ):
                raise ValueError("known token components exceed known total")
            if (
                self.known_prompt_tokens
                + self.known_visible_output_tokens
                + self.known_reasoning_tokens
                > self.known_total_tokens
            ):
                raise ValueError(
                    "known output details exceed aggregate completion capacity"
                )
            if (
                self.known_completion_tokens
                + self.known_cached_input_tokens
                + self.known_cache_write_input_tokens
                > self.known_total_tokens
            ):
                raise ValueError(
                    "known cache details exceed aggregate prompt capacity"
                )
            if (
                self.known_visible_output_tokens
                + self.known_reasoning_tokens
                + self.known_cached_input_tokens
                + self.known_cache_write_input_tokens
                > self.known_total_tokens
            ):
                raise ValueError(
                    "known input and output details exceed aggregate total"
                )
        if (
            self.missing_usage_breakdown_calls == 0
            and self.known_total_tokens
            != self.known_prompt_tokens + self.known_completion_tokens
        ):
            raise ValueError("known token breakdown contradicts known total")
        for missing_name, known_name in (
            ("missing_visible_output_usage_calls", "known_visible_output_tokens"),
            ("missing_reasoning_usage_calls", "known_reasoning_tokens"),
            ("missing_cached_input_usage_calls", "known_cached_input_tokens"),
            (
                "missing_cache_write_input_usage_calls",
                "known_cache_write_input_tokens",
            ),
        ):
            if (
                getattr(self, missing_name) == self.interactions
                and getattr(self, known_name) != 0
            ):
                raise ValueError(f"{known_name} requires per-call evidence")
        if self.pricing_error_calls > self.missing_cost_calls:
            raise ValueError("pricing errors must have missing cost")
        if self.missing_cost_calls == self.interactions and self.known_cost_usd:
            raise ValueError("known catalog cost requires priced call evidence")
        if (
            self.missing_provider_cost_calls == self.interactions
            and self.known_provider_cost_usd
        ):
            raise ValueError("known provider cost requires provider evidence")
        if sum(self.priced_calls_by_source.values()) != (
            self.interactions - self.missing_cost_calls
        ):
            raise ValueError("cost sources must count priced calls")
        if self.generation_attempts > self.interactions:
            raise ValueError("generation_attempts cannot exceed interactions")
        if not isinstance(self.price_catalogs, tuple):
            raise TypeError("price_catalogs must be a tuple")
        catalog_ids: set[str] = set()
        for catalog in self.price_catalogs:
            if not isinstance(catalog, PriceCatalog):
                raise TypeError("price_catalogs must contain PriceCatalog values")
            if catalog.catalog_id in catalog_ids:
                raise ValueError("price_catalogs must have unique catalog IDs")
            catalog_ids.add(catalog.catalog_id)
        if self.interactions == 0 and (
            self.known_prompt_tokens
            or self.known_completion_tokens
            or self.known_total_tokens
            or self.known_visible_output_tokens
            or self.known_reasoning_tokens
            or self.known_cached_input_tokens
            or self.known_cache_write_input_tokens
            or self.known_cost_usd
            or self.known_provider_cost_usd
            or self.price_catalogs
        ):
            raise ValueError("zero interactions cannot contain usage or cost evidence")
        if self.runs == 0 and (
            self.interactions
            or self.generation_attempts
            or self.routing_events
            or self.cumulative_run_duration_ms
        ):
            raise ValueError("an empty summary cannot contain run evidence")

    @property
    def interactions_by_channel(self) -> dict[str, int]:
        return dict(self.interactions_by_channel_items)

    @property
    def interactions_by_role(self) -> dict[str, int]:
        return dict(self.interactions_by_role_items)

    @property
    def interactions_by_requested_model(self) -> dict[str, int]:
        return dict(self.interactions_by_requested_model_items)

    @property
    def interactions_by_actual_model(self) -> dict[str, int]:
        return dict(self.interactions_by_actual_model_items)

    @property
    def interactions_by_priced_model(self) -> dict[str, int]:
        return dict(self.interactions_by_priced_model_items)

    @property
    def priced_calls_by_source(self) -> dict[str, int]:
        return dict(self.priced_calls_by_source_items)

    @property
    def finish_reasons(self) -> dict[str, int]:
        return dict(self.finish_reasons_items)

    @property
    def output_statuses(self) -> dict[str, int]:
        return dict(self.output_statuses_items)

    @property
    def output_emptiness(self) -> dict[str, int]:
        return dict(self.output_emptiness_items)

    @property
    def error_categories(self) -> dict[str, int]:
        return dict(self.error_categories_items)

    @property
    def outcome_counts(self) -> dict[str, int]:
        observed = dict(self.outcome_counts_items)
        return {outcome: observed.get(outcome, 0) for outcome in TASK_OUTCOMES}

    @property
    def total_usage_complete(self) -> bool:
        return self.missing_total_usage_calls == 0

    @property
    def usage_breakdown_complete(self) -> bool:
        return self.missing_usage_breakdown_calls == 0

    @property
    def total_tokens(self) -> int | None:
        return self.known_total_tokens if self.total_usage_complete else None

    @property
    def cost_complete(self) -> bool:
        return self.missing_cost_calls == 0

    @property
    def total_cost_usd(self) -> float | None:
        return self.known_cost_usd if self.cost_complete else None

    @property
    def provider_cost_complete(self) -> bool:
        return self.missing_provider_cost_calls == 0

    @property
    def total_provider_cost_usd(self) -> float | None:
        return (
            self.known_provider_cost_usd
            if self.provider_cost_complete
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        from .wire import _stats_summary_to_dict

        return _stats_summary_to_dict(self)


def aggregate_stats(runs: Iterable[RunStats]) -> StatsSummary:
    """Return the default sum across independent run or turn snapshots."""

    if isinstance(runs, (str, bytes, Mapping)):
        raise TypeError("runs must be an iterable of RunStats values")
    try:
        snapshots = tuple(runs)
    except TypeError as exc:
        raise TypeError("runs must be an iterable of RunStats values") from exc
    for index, run in enumerate(snapshots, start=1):
        if not isinstance(run, RunStats):
            raise TypeError(f"runs[{index}] must be a RunStats value")
    duplicate_run_ids = sorted(
        run_id
        for run_id, count in Counter(run.run_id for run in snapshots).items()
        if count > 1
    )
    if duplicate_run_ids:
        raise ValueError(
            "run_id values must be unique; duplicates: "
            + ", ".join(duplicate_run_ids)
        )
    snapshots = tuple(sorted(snapshots, key=lambda run: run.run_id))

    def merge_counts(attribute: str) -> tuple[tuple[str, int], ...]:
        counts: Counter[str] = Counter()
        for run in snapshots:
            counts.update(getattr(run, attribute))
        return tuple(sorted(counts.items()))

    catalogs: dict[str, PriceCatalog] = {}
    for run in snapshots:
        for catalog in run.price_catalogs:
            previous = catalogs.setdefault(catalog.catalog_id, catalog)
            if previous != catalog:
                raise ValueError(
                    f"conflicting price catalogs share id {catalog.catalog_id!r}"
                )

    return StatsSummary(
        runs=len(snapshots),
        interactions=sum(run.interaction_count for run in snapshots),
        failed_interactions=sum(run.failed_interactions for run in snapshots),
        interactions_by_channel_items=merge_counts("interactions_by_channel"),
        interactions_by_role_items=merge_counts("interactions_by_role"),
        interactions_by_requested_model_items=merge_counts(
            "interactions_by_requested_model"
        ),
        interactions_by_actual_model_items=merge_counts(
            "interactions_by_actual_model"
        ),
        interactions_by_priced_model_items=merge_counts(
            "interactions_by_priced_model"
        ),
        generation_attempts=sum(run.generation_attempts for run in snapshots),
        routing_events=sum(run.routing_events for run in snapshots),
        known_prompt_tokens=sum(run.known_prompt_tokens for run in snapshots),
        known_completion_tokens=sum(
            run.known_completion_tokens for run in snapshots
        ),
        known_total_tokens=sum(run.known_total_tokens for run in snapshots),
        known_visible_output_tokens=sum(
            run.known_visible_output_tokens for run in snapshots
        ),
        known_reasoning_tokens=sum(
            run.known_reasoning_tokens for run in snapshots
        ),
        known_cached_input_tokens=sum(
            run.known_cached_input_tokens for run in snapshots
        ),
        known_cache_write_input_tokens=sum(
            run.known_cache_write_input_tokens for run in snapshots
        ),
        missing_total_usage_calls=sum(
            run.missing_total_usage_calls for run in snapshots
        ),
        missing_usage_breakdown_calls=sum(
            run.missing_usage_breakdown_calls for run in snapshots
        ),
        missing_visible_output_usage_calls=sum(
            run.missing_visible_output_usage_calls for run in snapshots
        ),
        missing_reasoning_usage_calls=sum(
            run.missing_reasoning_usage_calls for run in snapshots
        ),
        missing_cached_input_usage_calls=sum(
            run.missing_cached_input_usage_calls for run in snapshots
        ),
        missing_cache_write_input_usage_calls=sum(
            run.missing_cache_write_input_usage_calls for run in snapshots
        ),
        usage_error_calls=sum(run.usage_error_calls for run in snapshots),
        known_cost_usd=checked_fsum(
            (run.known_cost_usd for run in snapshots),
            name="summary aggregate catalog cost",
        ),
        missing_cost_calls=sum(run.missing_cost_calls for run in snapshots),
        pricing_error_calls=sum(run.pricing_error_calls for run in snapshots),
        known_provider_cost_usd=checked_fsum(
            (run.known_provider_cost_usd for run in snapshots),
            name="summary aggregate provider-reported cost",
        ),
        missing_provider_cost_calls=sum(
            run.missing_provider_cost_calls for run in snapshots
        ),
        priced_calls_by_source_items=merge_counts("priced_calls_by_source"),
        finish_reasons_items=merge_counts("finish_reasons"),
        output_statuses_items=merge_counts("output_statuses"),
        output_emptiness_items=merge_counts("output_emptiness"),
        error_categories_items=merge_counts("error_categories"),
        outcome_counts_items=tuple(sorted(Counter(
            run.outcome for run in snapshots
        ).items())),
        max_tokens_reached_calls=sum(
            run.max_tokens_reached_calls for run in snapshots
        ),
        price_catalogs=tuple(catalogs[key] for key in sorted(catalogs)),
        cumulative_run_duration_ms=checked_fsum(
            (run.duration_ms for run in snapshots),
            name="cumulative run duration",
        ),
    )
