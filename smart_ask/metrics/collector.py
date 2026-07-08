"""Context-scoped collection at the provider-neutral executor boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
import threading
import time
from typing import Any, Optional
from uuid import uuid4

from ..domain import ExecutionRequest, ModelResult
from ..executors.base import ModelExecutor
from .cost import (
    DEFAULT_PRICE_CATALOG,
    PriceCatalog,
    PriceQuote,
    price_usage,
)
from .models import CallStats, ErrorCategory, RunStats, TokenUsage, UsageStatus


UsageNormalizer = Callable[[Any], TokenUsage]
Pricer = Callable[[Optional[str], TokenUsage], PriceQuote]


def _exception_diagnostic(prefix: str, error: Exception) -> str:
    detail = str(error).strip()
    diagnostic = f"{prefix}: {type(error).__name__}"
    return f"{diagnostic}: {detail}" if detail else diagnostic


def _error_category(error: Exception) -> ErrorCategory:
    """Conservatively classify executor failures without provider coupling."""

    name = type(error).__name__.lower()
    detail = str(error).lower()
    status = getattr(error, "status_code", None)
    if "timeout" in name or "timed out" in detail:
        return "timeout"
    if "ratelimit" in name or "rate limit" in detail or status == 429:
        return "rate_limit"
    if (
        "authentication" in name
        or "unauthorized" in detail
        or status in (401, 403)
    ):
        return "authentication"
    if isinstance(status, Integral) and 500 <= int(status) <= 599:
        return "provider_5xx"
    if (
        "invalid response" in detail
        or "must return a modelresult" in detail
        or "responsevalidation" in name
        or (
            isinstance(error, (TypeError, ValueError))
            and "response" in detail
        )
    ):
        return "invalid_response"
    return "unknown"


def normalize_usage(raw: Any) -> TokenUsage:
    """Normalize common provider usage shapes without discarding total-only data."""

    if raw is None:
        return TokenUsage()

    sentinel = object()

    def read(*names: str) -> Any:
        for name in names:
            if isinstance(raw, Mapping) and name in raw:
                value = raw[name]
                if value is not None:
                    return value
                continue
            value = getattr(raw, name, sentinel)
            if value is not sentinel and value is not None:
                return value
        return sentinel

    def count(value: Any, field_name: str) -> int | None:
        if value is sentinel or value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be a non-negative integer")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return normalized

    return TokenUsage(
        prompt_tokens=count(
            read("prompt_tokens", "input_tokens"),
            "prompt_tokens",
        ),
        completion_tokens=count(
            read("completion_tokens", "output_tokens"),
            "completion_tokens",
        ),
        total_tokens=count(read("total_tokens"), "total_tokens"),
        visible_output_tokens=count(
            read("visible_output_tokens"),
            "visible_output_tokens",
        ),
        reasoning_tokens=count(
            read("reasoning_tokens"),
            "reasoning_tokens",
        ),
        cached_input_tokens=count(
            read("cached_input_tokens", "cached_tokens"),
            "cached_input_tokens",
        ),
        cache_write_input_tokens=count(
            read("cache_write_input_tokens", "cache_creation_input_tokens"),
            "cache_write_input_tokens",
        ),
    )


@dataclass(frozen=True)
class CallRecord:
    """Raw call evidence paired with its canonical immutable metrics."""

    request: ExecutionRequest
    result: ModelResult | None
    stats: CallStats

    def __post_init__(self) -> None:
        if not isinstance(self.request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        if self.result is not None and not isinstance(self.result, ModelResult):
            raise TypeError("result must be a ModelResult or None")
        if not isinstance(self.stats, CallStats):
            raise TypeError("stats must be CallStats")
        if self.request.model != self.stats.requested_model:
            raise ValueError("request model and call metrics disagree")
        if self.request.role != self.stats.role:
            raise ValueError("request role and call metrics disagree")
        if self.request.max_tokens != self.stats.requested_max_tokens:
            raise ValueError("request max_tokens and call metrics disagree")
        if self.stats.status == "ok":
            if self.result is None:
                raise ValueError("a successful call record requires a result")
            if self.result.model != self.stats.actual_model:
                raise ValueError("result model and call metrics disagree")
            for field in (
                "finish_reason",
                "output_status",
                "output_empty",
                "native_finish_reason",
                "refusal",
                "max_tokens_reached",
                "provider_cost_usd",
            ):
                if getattr(self.result, field) != getattr(self.stats, field):
                    raise ValueError(f"result {field} and call metrics disagree")
        elif self.result is not None:
            raise ValueError("a failed call record cannot contain a result")
        if self.result is not None and (
            self.result.applied_max_tokens != self.stats.applied_max_tokens
        ):
            raise ValueError(
                "result applied_max_tokens and call metrics disagree"
            )


class StatsCapture:
    """An encapsulated run scope with a stable snapshot after close."""

    __slots__ = (
        "_strategy_id",
        "_task_id",
        "_run_id",
        "_started_ns",
        "_clock",
        "_records",
        "_closed_calls",
        "_final_stats",
        "_next_ordinal",
        "_lock",
    )

    def __init__(
        self,
        *,
        strategy_id: str | None,
        task_id: str | None,
        run_id: str,
        started_ns: int,
        clock: Callable[[], int],
    ):
        self._strategy_id = strategy_id
        self._task_id = task_id
        self._run_id = run_id
        self._started_ns = started_ns
        self._clock = clock
        self._records: list[CallRecord] = []
        self._closed_calls: tuple[CallRecord, ...] | None = None
        self._final_stats: RunStats | None = None
        self._next_ordinal = 1
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._final_stats is not None

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        with self._lock:
            if self._closed_calls is not None:
                return self._closed_calls
            return tuple(sorted(
                self._records,
                key=lambda call: call.stats.ordinal,
            ))

    def _reserve_call(self) -> tuple[int, str]:
        with self._lock:
            if self._final_stats is not None:
                raise RuntimeError("metrics capture is closed")
            ordinal = self._next_ordinal
            self._next_ordinal += 1
        return ordinal, f"call-{ordinal}"

    def _append(self, call: CallRecord) -> None:
        with self._lock:
            if self._final_stats is not None:
                raise RuntimeError("metrics capture is closed")
            self._records.append(call)

    def _snapshot(self, duration_ms: float) -> RunStats:
        calls = tuple(sorted(
            self._records,
            key=lambda call: call.stats.ordinal,
        ))
        return RunStats(
            run_id=self._run_id,
            task_id=self._task_id,
            duration_ms=duration_ms,
            calls=tuple(call.stats for call in calls),
            strategy_id=self._strategy_id,
        )

    def _close(self, duration_ms: float) -> None:
        with self._lock:
            if self._final_stats is not None:
                raise RuntimeError("metrics capture is already closed")
            calls = tuple(sorted(
                self._records,
                key=lambda call: call.stats.ordinal,
            ))
            self._closed_calls = calls
            self._final_stats = RunStats(
                run_id=self._run_id,
                task_id=self._task_id,
                duration_ms=duration_ms,
                calls=tuple(call.stats for call in calls),
                strategy_id=self._strategy_id,
            )
            self._records.clear()

    @property
    def stats(self) -> RunStats:
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats
            duration_ms = (self._clock() - self._started_ns) / 1_000_000
            return self._snapshot(duration_ms)


class StatsCollector:
    """Context-local collector shared by normal calls and benchmark workers."""

    __slots__ = (
        "_price_catalog",
        "_require_active_capture",
        "_clock",
        "_usage_normalizer",
        "_pricer",
        "_run_id_factory",
        "_active",
    )

    def __init__(
        self,
        *,
        price_catalog: PriceCatalog | None = DEFAULT_PRICE_CATALOG,
        require_active_capture: bool = False,
        clock: Callable[[], int] = time.perf_counter_ns,
        run_id_factory: Callable[[], str] | None = None,
        usage_normalizer: UsageNormalizer = normalize_usage,
        pricer: Pricer | None = None,
    ):
        if price_catalog is not None and not isinstance(price_catalog, PriceCatalog):
            raise TypeError("price_catalog must be a PriceCatalog or None")
        if not isinstance(require_active_capture, bool):
            raise TypeError("require_active_capture must be a boolean")
        for name, value in (
            ("clock", clock),
            ("usage_normalizer", usage_normalizer),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        if run_id_factory is not None and not callable(run_id_factory):
            raise TypeError("run_id_factory must be callable or None")
        if pricer is not None and not callable(pricer):
            raise TypeError("pricer must be callable or None")
        self._price_catalog = price_catalog
        self._require_active_capture = require_active_capture
        self._clock = clock
        self._usage_normalizer = usage_normalizer
        if pricer is not None:
            self._pricer = pricer
        elif price_catalog is not None:
            self._pricer = lambda model, usage: price_usage(
                model,
                usage,
                price_catalog,
            )
        else:
            self._pricer = lambda model, usage: PriceQuote(
                None,
                "unavailable",
                diagnostic="no pricer or price catalog is configured",
            )
        self._run_id_factory = (
            (lambda: f"run-{uuid4().hex}")
            if run_id_factory is None
            else run_id_factory
        )
        self._active: ContextVar[StatsCapture | None] = ContextVar(
            f"smart_ask_stats_{id(self)}",
            default=None,
        )

    @property
    def price_catalog(self) -> PriceCatalog | None:
        return self._price_catalog

    @property
    def require_active_capture(self) -> bool:
        return self._require_active_capture

    @contextmanager
    def capture(
        self,
        strategy_id: str | None = None,
        task_id: str | None = None,
        *,
        run_id: str | None = None,
    ):
        if self._active.get() is not None:
            raise RuntimeError("metrics captures cannot be nested")
        for name, value in (("strategy_id", strategy_id), ("task_id", task_id)):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty trimmed string or None")
        resolved_run_id = run_id if run_id is not None else self._run_id_factory()
        if (
            not isinstance(resolved_run_id, str)
            or not resolved_run_id
            or resolved_run_id != resolved_run_id.strip()
        ):
            raise ValueError("run_id must be a non-empty trimmed string")
        capture = StatsCapture(
            strategy_id=strategy_id,
            task_id=task_id,
            run_id=resolved_run_id,
            started_ns=self._clock(),
            clock=self._clock,
        )
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            try:
                capture._close(
                    (self._clock() - capture._started_ns) / 1_000_000
                )
            finally:
                self._active.reset(token)

    def _active_capture(self) -> StatsCapture | None:
        return self._active.get()

    def wrap(self, executor: ModelExecutor, channel: str) -> ModelExecutor:
        """Instrument an executor; each call must carry its semantic role."""

        if (
            not isinstance(channel, str)
            or not channel
            or channel != channel.strip()
        ):
            raise ValueError("channel must be a non-empty trimmed string")
        if isinstance(executor, _InstrumentedExecutor):
            raise ValueError("executor is already instrumented")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor.execute must be callable")
        if not isinstance(getattr(executor, "captures_output", None), bool):
            raise TypeError("executor.captures_output must be a boolean")
        return _InstrumentedExecutor(executor, self, channel)

    def is_instrumented(
        self,
        executor: Any,
        *,
        channel: str | None = None,
    ) -> bool:
        """Return whether this collector owns an executor wrapper and channel."""

        if channel is not None and (
            not isinstance(channel, str)
            or not channel
            or channel != channel.strip()
        ):
            raise ValueError("channel must be a non-empty trimmed string or None")
        return (
            isinstance(executor, _InstrumentedExecutor)
            and executor._collector is self
            and (channel is None or executor._channel == channel)
        )


class _InstrumentedExecutor:
    """Private executor decorator used exclusively by ``StatsCollector``."""

    __slots__ = ("_delegate", "_collector", "_channel")

    def __init__(
        self,
        delegate: ModelExecutor,
        collector: StatsCollector,
        channel: str,
    ):
        self._delegate = delegate
        self._collector = collector
        self._channel = channel

    @property
    def captures_output(self) -> bool:
        return bool(getattr(self._delegate, "captures_output", False))

    def execute(self, request: ExecutionRequest) -> ModelResult:
        capture = self._collector._active_capture()
        if capture is None:
            if self._collector._require_active_capture:
                raise RuntimeError(
                    "instrumented executor called outside a metrics capture"
                )
            result = self._delegate.execute(request)
            if not isinstance(result, ModelResult):
                raise TypeError("instrumented executor must return a ModelResult")
            return result

        ordinal, call_id = capture._reserve_call()
        started_ns = self._collector._clock()
        offset_ms = (started_ns - capture._started_ns) / 1_000_000
        try:
            result = self._delegate.execute(request)
            if not isinstance(result, ModelResult):
                raise TypeError("instrumented executor must return a ModelResult")
        except Exception as exc:
            latency_ms = (self._collector._clock() - started_ns) / 1_000_000
            call_stats = CallStats(
                run_id=capture._run_id,
                call_id=call_id,
                ordinal=ordinal,
                channel=self._channel,
                role=request.role,
                requested_model=request.model,
                actual_model=None,
                priced_model=None,
                status="error",
                latency_ms=latency_ms,
                started_offset_ms=offset_ms,
                usage=TokenUsage(),
                usage_status="unavailable",
                usage_diagnostic="call failed before usage was returned",
                price_quote=PriceQuote(
                    None,
                    "unavailable",
                    diagnostic="call failed before pricing",
                ),
                finish_reason="error",
                output_status=None,
                output_empty=None,
                requested_max_tokens=request.max_tokens,
                applied_max_tokens=None,
                max_tokens_reached=False,
                provider_cost_usd=None,
                error_category=_error_category(exc),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            capture._append(CallRecord(request, None, call_stats))
            raise

        latency_ms = (self._collector._clock() - started_ns) / 1_000_000
        usage, usage_status, usage_diagnostic = self._normalize_usage(result.usage)
        usage, usage_status, usage_diagnostic = self._reconcile_usage(
            usage,
            usage_status,
            usage_diagnostic,
            result,
        )
        priced_model = (result.model or None) or request.model
        quote = self._price(priced_model, usage, usage_status)
        call_stats = CallStats(
            run_id=capture._run_id,
            call_id=call_id,
            ordinal=ordinal,
            channel=self._channel,
            role=request.role,
            requested_model=request.model,
            actual_model=result.model,
            priced_model=priced_model,
            status="ok",
            latency_ms=latency_ms,
            started_offset_ms=offset_ms,
            usage=usage,
            usage_status=usage_status,
            usage_diagnostic=usage_diagnostic,
            price_quote=quote,
            finish_reason=result.finish_reason,
            output_status=result.output_status,
            output_empty=result.output_empty,
            requested_max_tokens=request.max_tokens,
            applied_max_tokens=result.applied_max_tokens,
            max_tokens_reached=result.max_tokens_reached,
            provider_cost_usd=result.provider_cost_usd,
            native_finish_reason=result.native_finish_reason,
            refusal=result.refusal,
        )
        capture._append(CallRecord(request, result, call_stats))
        return result

    def _normalize_usage(
        self,
        raw_usage: Any,
    ) -> tuple[TokenUsage, UsageStatus, str | None]:
        try:
            usage = self._collector._usage_normalizer(raw_usage)
            if not isinstance(usage, TokenUsage):
                raise TypeError("usage normalizer must return TokenUsage")
        except Exception as exc:
            return (
                TokenUsage(),
                "error",
                _exception_diagnostic("usage normalization failed", exc),
            )

        if usage.breakdown_complete:
            return usage, "complete", None
        if (
            usage.total_complete
            and usage.prompt_tokens is None
            and usage.completion_tokens is None
        ):
            return usage, "total_only", "input/output token breakdown is unavailable"
        if usage.any_known:
            return usage, "partial", "token usage contains only a partial breakdown"
        if raw_usage is None:
            return usage, "unavailable", "provider did not return token usage"
        return usage, "unavailable", "usage contained no supported token counts"

    def _price(
        self,
        priced_model: str,
        usage: TokenUsage,
        usage_status: UsageStatus,
    ) -> PriceQuote:
        if usage_status == "error":
            return PriceQuote(
                None,
                "unavailable",
                diagnostic="pricing skipped because usage telemetry is invalid",
            )
        try:
            quote = self._collector._pricer(priced_model, usage)
            if not isinstance(quote, PriceQuote):
                raise TypeError("pricer must return PriceQuote")
            return quote
        except Exception as exc:
            # Telemetry remains non-fatal, but the failure is retained explicitly.
            return PriceQuote(
                None,
                "error",
                diagnostic=_exception_diagnostic("pricing failed", exc),
            )

    @staticmethod
    def _reconcile_usage(
        usage: TokenUsage,
        usage_status: UsageStatus,
        usage_diagnostic: str | None,
        result: ModelResult,
    ) -> tuple[TokenUsage, UsageStatus, str | None]:
        """Merge adapter-level detail without turning telemetry into call failure."""

        if usage_status == "error":
            return usage, usage_status, usage_diagnostic
        try:
            def detail(field: str) -> int | None:
                normalized = getattr(usage, field)
                adapter = getattr(result, field)
                if (
                    normalized is not None
                    and adapter is not None
                    and normalized != adapter
                ):
                    raise ValueError(
                        f"raw and adapter {field} evidence disagree"
                    )
                return adapter if adapter is not None else normalized

            reconciled = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                visible_output_tokens=detail("visible_output_tokens"),
                reasoning_tokens=detail("reasoning_tokens"),
                cached_input_tokens=detail("cached_input_tokens"),
                cache_write_input_tokens=detail("cache_write_input_tokens"),
            )
        except Exception as exc:
            return (
                TokenUsage(),
                "error",
                _exception_diagnostic("usage reconciliation failed", exc),
            )
        if usage_status == "unavailable" and reconciled.any_known:
            return (
                reconciled,
                "partial",
                "token usage contains only response-detail counts",
            )
        return reconciled, usage_status, usage_diagnostic
