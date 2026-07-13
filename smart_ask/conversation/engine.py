"""One asynchronous execution engine for all SmartAsk conversation callers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import json
import time
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from .domain import (
    ConversationEvent,
    thaw_value,
)
from .model import (
    CompletedRun,
    Conversation,
    DecisionDraft,
    DecisionId,
    DecisionRecord,
    InputTokenCount,
    ModelCallRecord,
    ModelCallResult,
    ModelCallSpec,
    OutputStatus,
    PreparedResponse,
    ProviderRequestRecord,
    RunMetadata,
    RunRecord,
    TokenCount,
    _PreparedPayload,
)


class ModelCallFailed(RuntimeError):
    """A logical model call did not produce a usable provider stream."""

    def __init__(self, call_id: str, message: str):
        super().__init__(message)
        self.call_id = call_id


class BufferedResponseLimitExceeded(ModelCallFailed):
    """A hidden response exceeded a configured buffering bound."""


class TokenCountUnavailable(RuntimeError):
    """A reachable transformed request could not be counted."""


class RunDeadlineExceeded(TimeoutError):
    """A complete strategy invocation exceeded its configured deadline."""


@runtime_checkable
class ModelCallExecutor(Protocol):
    """Resolve a trusted target and stream one structured logical call."""

    async def stream(
        self,
        spec: ModelCallSpec,
    ) -> AsyncIterator[ConversationEvent]: ...


@runtime_checkable
class TokenCountingModelCallExecutor(ModelCallExecutor, Protocol):
    """Optional token-counting capability of a model-call executor."""

    async def count_tokens(self, spec: ModelCallSpec) -> InputTokenCount | None: ...


@runtime_checkable
class StrategyMethod(Protocol):
    """A complete routing and response-selection algorithm."""

    async def respond(
        self,
        conversation: Conversation,
        run: "RunScope",
    ) -> PreparedResponse: ...

    def token_count_candidates(
        self,
        conversation: Conversation,
    ) -> tuple[ModelCallSpec, ...]: ...


@runtime_checkable
class RunObserver(Protocol):
    """Optional live, content-bearing observation of one engine invocation."""

    def run_started(
        self,
        run_id: str,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> None: ...

    def model_call_planned(
        self,
        run_id: str,
        call_id: str,
        sequence: int,
        spec: ModelCallSpec,
        caused_by: DecisionId | None,
    ) -> None: ...

    def model_event(
        self,
        run_id: str,
        call_id: str,
        event: ConversationEvent,
    ) -> None: ...

    def model_failed(
        self,
        run_id: str,
        call_id: str,
        error_type: str,
        message: str,
    ) -> None: ...

    def decision_recorded(
        self,
        run_id: str,
        decision: DecisionRecord,
    ) -> None: ...

    def run_finished(self, record: RunRecord) -> None: ...

    def run_failed(
        self,
        run_id: str,
        error_type: str,
        message: str,
    ) -> None: ...


@dataclass
class _CallState:
    call_id: str
    sequence: int
    spec: ModelCallSpec
    caused_by: DecisionId | None
    provider_request_ids: list[str]
    status: str = "planned"
    error: str | None = None


@dataclass
class _ProviderState:
    provider_request_id: str
    call_id: str
    sequence: int
    target_id: str
    requested_max_output_tokens: int | None
    started_at: float
    status: str = "running"
    selected_model: str | None = None
    actual_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    visible_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    provider_cost_usd: float | None = None
    stop_reason: str | None = None
    stream_complete: bool = False
    tool_call_count: int = 0
    error: str | None = None
    finished_at: float | None = None
    first_output_at: float | None = None
    evidence: "_EventEvidence | None" = None

    def record(self) -> ProviderRequestRecord:
        finished = self.finished_at or time.perf_counter()
        return ProviderRequestRecord(
            provider_request_id=self.provider_request_id,
            call_id=self.call_id,
            sequence=self.sequence,
            status=self.status,  # type: ignore[arg-type]
            target_id=self.target_id,
            requested_max_output_tokens=self.requested_max_output_tokens,
            selected_model=self.selected_model,
            actual_model=self.actual_model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            visible_output_tokens=self.visible_output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            provider_cost_usd=self.provider_cost_usd,
            stop_reason=self.stop_reason,
            stream_complete=self.stream_complete,
            tool_call_count=self.tool_call_count,
            visible_text_chars=(
                0
                if self.evidence is None
                else sum(len(value) for value in self.evidence.text)
            ),
            output_status=(
                "empty"
                if self.evidence is None
                else self.evidence.output_status
            ),
            time_to_first_output_ms=(
                None
                if self.first_output_at is None
                else (self.first_output_at - self.started_at) * 1000
            ),
            duration_ms=(finished - self.started_at) * 1000,
            error=self.error,
        )


class _EventEvidence:
    def __init__(self) -> None:
        self.selected_model: str | None = None
        self.actual_model: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.visible_output_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.provider_cost_usd: float | None = None
        self.stop_reason: str | None = None
        self.stream_complete = False
        self.tool_call_count = 0
        self.text: list[str] = []
        self.error: str | None = None
        self.first_output_at: float | None = None

    def observe(self, event: ConversationEvent) -> None:
        data = event.data
        if event.kind == "message_start":
            selected = data.get("selected_model")
            if isinstance(selected, str) and selected:
                self.selected_model = selected
            value = data.get("model")
            if isinstance(value, str) and value:
                self.actual_model = value
        elif event.kind == "content_start":
            block = data.get("block")
            block_type = block.get("type") if hasattr(block, "get") else None
            if block_type in ("text", "thinking", "tool_call"):
                if self.first_output_at is None:
                    self.first_output_at = time.perf_counter()
            if block_type == "tool_call":
                self.tool_call_count += 1
        elif event.kind == "content_delta":
            delta = data.get("delta")
            if hasattr(delta, "get") and delta.get("type") == "text":
                value = delta.get("text")
                if isinstance(value, str):
                    self.text.append(value)
        elif event.kind == "usage":
            for source, target in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("visible_output_tokens", "visible_output_tokens"),
                ("reasoning_tokens", "reasoning_tokens"),
                ("cache_read_tokens", "cache_read_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
            ):
                value = data.get(source)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    setattr(self, target, value)
            cost = data.get("provider_cost_usd")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and cost >= 0
            ):
                self.provider_cost_usd = float(cost)
        elif event.kind == "message_delta":
            value = data.get("stop_reason")
            if isinstance(value, str) and value:
                self.stop_reason = value
        elif event.kind == "message_stop":
            self.stream_complete = True
        elif event.kind == "error":
            value = data.get("message")
            self.error = value if isinstance(value, str) else "executor error"

    @property
    def output_status(self) -> OutputStatus:
        if self.stop_reason in ("refusal", "content_filter"):
            return "refused"
        if self.stop_reason in ("length", "max_tokens"):
            return "truncated"
        if self.text or self.tool_call_count:
            return "usable"
        return "empty"

    def copy_to(self, state: _ProviderState) -> None:
        state.selected_model = self.selected_model
        state.actual_model = self.actual_model
        state.input_tokens = self.input_tokens
        state.output_tokens = self.output_tokens
        state.visible_output_tokens = self.visible_output_tokens
        state.reasoning_tokens = self.reasoning_tokens
        state.cache_read_tokens = self.cache_read_tokens
        state.cache_write_tokens = self.cache_write_tokens
        state.provider_cost_usd = self.provider_cost_usd
        state.stop_reason = self.stop_reason
        state.stream_complete = self.stream_complete
        state.tool_call_count = self.tool_call_count
        state.first_output_at = self.first_output_at


class RunScope:
    """Bounded, instrumented model-call capability for one method invocation."""

    def __init__(
        self,
        executor: ModelCallExecutor,
        metadata: RunMetadata,
        *,
        observer: RunObserver | None = None,
        observer_error: Any = None,
        max_model_calls: int = 16,
        max_buffer_events: int = 10000,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        max_buffer_seconds: float = 600.0,
    ) -> None:
        if not callable(getattr(executor, "stream", None)):
            raise TypeError("executor must expose an async stream operation")
        if not isinstance(metadata, RunMetadata):
            raise TypeError("metadata must be RunMetadata")
        for value, name in (
            (max_model_calls, "max_model_calls"),
            (max_buffer_events, "max_buffer_events"),
            (max_buffer_bytes, "max_buffer_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_buffer_seconds <= 0:
            raise ValueError("max_buffer_seconds must be positive")
        self._executor = executor
        self.metadata = metadata
        self._observer = observer
        self._observer_error = observer_error
        self.run_id = uuid4().hex
        self._owner = object()
        self._state = "planning"
        self._started_wall = time.time()
        self._started = time.perf_counter()
        self._max_model_calls = max_model_calls
        self._max_buffer_events = max_buffer_events
        self._max_buffer_bytes = max_buffer_bytes
        self._max_buffer_seconds = float(max_buffer_seconds)
        self._calls: list[_CallState] = []
        self._calls_by_id: dict[str, _CallState] = {}
        self._providers: list[_ProviderState] = []
        self._decisions: list[DecisionRecord] = []
        self._decision_ids: set[DecisionId] = set()
        self._final_call_id: str | None = None
        self._final_decision_id: DecisionId | None = None
        self._finished_record: RunRecord | None = None
        self._active_call_tasks: set[asyncio.Task[object]] = set()
        self._event_sequence = 0
        self._success_operations: list[Callable[[], Awaitable[None]]] = []

    def _next_event_sequence(self) -> int:
        self._event_sequence += 1
        return self._event_sequence

    def _notify(self, operation: str, *args: object) -> None:
        if self._observer is None:
            return
        try:
            getattr(self._observer, operation)(*args)
        except Exception as exc:
            if callable(self._observer_error):
                self._observer_error(operation, exc)

    def _require_planning(self) -> None:
        if self._state != "planning":
            raise RuntimeError("the method-call phase for this run is closed")

    def _require_decision(self, decision_id: DecisionId) -> None:
        if decision_id not in self._decision_ids:
            raise ValueError("decision id does not belong to this run")

    def _reserve_call(
        self,
        spec: ModelCallSpec,
        caused_by: DecisionId | None,
    ) -> _CallState:
        self._require_planning()
        if not isinstance(spec, ModelCallSpec):
            raise TypeError("spec must be a ModelCallSpec")
        if caused_by is not None:
            self._require_decision(caused_by)
        if len(self._calls) >= self._max_model_calls:
            raise RuntimeError("model-call limit exceeded")
        call_number = len(self._calls) + 1
        state = _CallState(
            call_id=f"call-{call_number}",
            sequence=self._next_event_sequence(),
            spec=spec,
            caused_by=caused_by,
            provider_request_ids=[],
        )
        self._calls.append(state)
        self._calls_by_id[state.call_id] = state
        self._notify(
            "model_call_planned",
            self.run_id,
            state.call_id,
            state.sequence,
            spec,
            caused_by,
        )
        return state

    def record_decision(self, draft: DecisionDraft) -> DecisionId:
        self._require_planning()
        if not isinstance(draft, DecisionDraft):
            raise TypeError("draft must be a DecisionDraft")
        for call_id in draft.evidence_call_ids:
            call = self._calls_by_id.get(call_id)
            if call is None:
                raise ValueError(f"unknown evidence call id: {call_id}")
            if call.status != "completed":
                raise ValueError("decision evidence must be a completed model call")
        decision_number = len(self._decisions) + 1
        decision_id = DecisionId(f"decision-{decision_number}")
        record = DecisionRecord(
            decision_id=decision_id,
            gate=draft.gate,
            outcome=draft.outcome,
            reason_code=draft.reason_code,
            selected_profile_id=draft.selected_profile_id,
            evidence_call_ids=draft.evidence_call_ids,
            sequence=self._next_event_sequence(),
        )
        self._decisions.append(record)
        self._decision_ids.add(decision_id)
        self._notify("decision_recorded", self.run_id, record)
        return decision_id

    def defer_success(
        self,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        """Commit method state only after the selected response succeeds."""

        self._require_planning()
        if not callable(operation):
            raise TypeError("success operation must be callable")
        self._success_operations.append(operation)

    async def _commit_success(self) -> None:
        for operation in self._success_operations:
            result = operation()
            if not isinstance(result, Awaitable):
                raise TypeError("success operation must return an awaitable")
            await result

    async def call_buffered(
        self,
        spec: ModelCallSpec,
        *,
        caused_by: DecisionId | None = None,
    ) -> ModelCallResult:
        call = self._reserve_call(spec, caused_by)
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("buffered calls require an asyncio task")
        self._active_call_tasks.add(task)
        events: list[ConversationEvent] = []
        byte_count = 0
        started = time.perf_counter()
        source = self._execute(call).__aiter__()
        try:
            async with asyncio.timeout(self._max_buffer_seconds):
                async for event in source:
                    events.append(event)
                    byte_count += len(event.kind.encode("utf-8")) + len(
                        json.dumps(
                            thaw_value(event.data),
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                    if len(events) > self._max_buffer_events:
                        raise BufferedResponseLimitExceeded(
                            call.call_id,
                            "buffered response event limit exceeded",
                        )
                    if byte_count > self._max_buffer_bytes:
                        raise BufferedResponseLimitExceeded(
                            call.call_id,
                            "buffered response byte limit exceeded",
                        )
        except TimeoutError as exc:
            message = "buffered response duration limit exceeded"
            self._mark_call_error(call, message)
            raise BufferedResponseLimitExceeded(call.call_id, message) from exc
        except BufferedResponseLimitExceeded as exc:
            self._mark_call_error(call, str(exc))
            raise
        finally:
            closer = getattr(source, "aclose", None)
            if callable(closer):
                await closer()
            self._active_call_tasks.discard(task)
        evidence = self._provider_evidence(call)
        return ModelCallResult(
            call_id=call.call_id,
            events=tuple(events),
            selected_model=evidence.selected_model,
            actual_model=evidence.actual_model,
            text="".join(evidence.text),
            stop_reason=evidence.stop_reason,
            input_tokens=evidence.input_tokens,
            output_tokens=evidence.output_tokens,
            reasoning_tokens=evidence.reasoning_tokens,
            cache_read_tokens=evidence.cache_read_tokens,
            cache_write_tokens=evidence.cache_write_tokens,
            provider_cost_usd=evidence.provider_cost_usd,
            tool_call_count=evidence.tool_call_count,
            stream_complete=evidence.stream_complete,
            output_status=evidence.output_status,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def plan_live(
        self,
        spec: ModelCallSpec,
        *,
        caused_by: DecisionId,
    ) -> PreparedResponse:
        self._require_decision(caused_by)
        call = self._reserve_call(spec, caused_by)
        return PreparedResponse._create(
            self._owner,
            _PreparedPayload("live", call.call_id, caused_by),
        )

    def plan_replay(
        self,
        result: ModelCallResult,
        *,
        accepted_by: DecisionId,
    ) -> PreparedResponse:
        self._require_planning()
        self._require_decision(accepted_by)
        if not isinstance(result, ModelCallResult):
            raise TypeError("result must be a ModelCallResult")
        call = self._calls_by_id.get(result.call_id)
        if call is None or call.status != "completed":
            raise ValueError("result is not a completed call from this run")
        return PreparedResponse._create(
            self._owner,
            _PreparedPayload("replay", result.call_id, accepted_by, result),
        )

    def _provider_evidence(self, call: _CallState) -> _EventEvidence:
        if not call.provider_request_ids:
            raise RuntimeError("model call has no provider request evidence")
        provider_id = call.provider_request_ids[-1]
        provider = next(
            value
            for value in self._providers
            if value.provider_request_id == provider_id
        )
        evidence = provider.evidence
        if not isinstance(evidence, _EventEvidence):
            raise RuntimeError("provider request lost normalized evidence")
        return evidence

    async def _execute(
        self,
        call: _CallState,
    ) -> AsyncIterator[ConversationEvent]:
        if call.status != "planned":
            raise RuntimeError("model call was already executed")
        call.status = "running"
        request_number = len(self._providers) + 1
        requested_max = call.spec.conversation.parameters.get("max_tokens")
        if isinstance(requested_max, bool) or not isinstance(requested_max, int):
            requested_max = None
        provider = _ProviderState(
            provider_request_id=f"provider-request-{request_number}",
            call_id=call.call_id,
            sequence=self._next_event_sequence(),
            target_id=call.spec.target_id,
            requested_max_output_tokens=requested_max,
            started_at=time.perf_counter(),
        )
        evidence = _EventEvidence()
        provider.evidence = evidence
        self._providers.append(provider)
        call.provider_request_ids.append(provider.provider_request_id)
        iterator = self._executor.stream(call.spec).__aiter__()
        try:
            async for event in iterator:
                if not isinstance(event, ConversationEvent):
                    raise TypeError(
                        "conversation executors must emit ConversationEvent"
                    )
                evidence.observe(event)
                self._notify("model_event", self.run_id, call.call_id, event)
                if evidence.error is not None:
                    raise ModelCallFailed(call.call_id, evidence.error)
                yield event
        except asyncio.CancelledError:
            call.status = "cancelled"
            call.error = "model call cancelled"
            provider.status = "cancelled"
            provider.error = call.error
            raise
        except GeneratorExit:
            if call.status == "running":
                call.status = "cancelled"
                call.error = "model call stream closed"
                provider.status = "cancelled"
                provider.error = call.error
            raise
        except Exception as exc:
            diagnostic = type(exc).__name__
            self._notify(
                "model_failed",
                self.run_id,
                call.call_id,
                diagnostic,
                str(exc),
            )
            call.status = "error"
            call.error = diagnostic
            provider.status = "error"
            provider.error = diagnostic
            if isinstance(exc, ModelCallFailed):
                raise
            raise ModelCallFailed(call.call_id, str(exc)) from exc
        else:
            call.status = "completed"
            provider.status = "completed"
        finally:
            evidence.copy_to(provider)
            provider.finished_at = time.perf_counter()
            closer = getattr(iterator, "aclose", None)
            if callable(closer):
                await closer()

    def _mark_call_error(self, call: _CallState, message: str) -> None:
        call.status = "error"
        call.error = message
        if call.provider_request_ids:
            provider_id = call.provider_request_ids[-1]
            provider = next(
                value
                for value in self._providers
                if value.provider_request_id == provider_id
            )
            provider.status = "error"
            provider.error = message
            if provider.finished_at is None:
                provider.finished_at = time.perf_counter()

    async def _cancel_active_calls(self) -> None:
        current = asyncio.current_task()
        tasks = [
            task
            for task in self._active_call_tasks
            if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _begin_response(self, response: PreparedResponse) -> _PreparedPayload:
        if self._state != "planning":
            raise RuntimeError("run has already committed a response")
        if not isinstance(response, PreparedResponse):
            raise TypeError("method must return a PreparedResponse")
        unfinished = [
            call.call_id for call in self._calls if call.status == "running"
        ]
        if unfinished:
            raise RuntimeError(
                "method returned while model calls were still running: "
                + ", ".join(unfinished)
            )
        payload = response._consume(self._owner)
        self._require_decision(payload.selected_by)
        self._state = "responding"
        self._final_call_id = payload.call_id
        self._final_decision_id = payload.selected_by
        return payload

    async def _response_events(
        self,
        payload: _PreparedPayload,
    ) -> AsyncIterator[ConversationEvent]:
        if payload.kind == "replay":
            if payload.result is None:
                raise RuntimeError("replay response lost its buffered result")
            for event in payload.result.events:
                yield event
            return
        call = self._calls_by_id[payload.call_id]
        async for event in self._execute(call):
            yield event

    def _finish(
        self,
        status: str,
        error: str | None = None,
    ) -> RunRecord:
        if self._finished_record is not None:
            return self._finished_record
        self._state = "closed"
        finished = time.perf_counter()
        calls = tuple(ModelCallRecord(
            call_id=value.call_id,
            sequence=value.sequence,
            profile_id=value.spec.profile_id,
            target_id=value.spec.target_id,
            selected_model=(
                None
                if not value.provider_request_ids
                else next(
                    provider.selected_model
                    for provider in reversed(self._providers)
                    if provider.provider_request_id
                    in value.provider_request_ids
                )
            ),
            role=value.spec.role,
            phase=value.spec.phase,
            caused_by_decision_id=value.caused_by,
            provider_request_ids=tuple(value.provider_request_ids),
            status=value.status,  # type: ignore[arg-type]
            error=value.error,
        ) for value in self._calls)
        self._finished_record = RunRecord(
            run_id=self.run_id,
            metadata=self.metadata,
            status=status,  # type: ignore[arg-type]
            started_at=self._started_wall,
            duration_ms=(finished - self._started) * 1000,
            decisions=tuple(self._decisions),
            model_calls=calls,
            provider_requests=tuple(value.record() for value in self._providers),
            final_call_id=self._final_call_id,
            final_decision_id=self._final_decision_id,
            error=error,
        )
        self._notify("run_finished", self._finished_record)
        return self._finished_record


class RunHandle:
    """Single-consumer stream handle with an awaitable final run record."""

    def __init__(
        self,
        engine: "StrategyEngine",
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> None:
        self._engine = engine
        self._conversation = conversation
        self._scope = engine._new_scope(metadata)
        self._claimed = False
        self._record: asyncio.Future[RunRecord] = (
            asyncio.get_running_loop().create_future()
        )

    def events(self) -> AsyncIterator[ConversationEvent]:
        if self._claimed:
            raise RuntimeError("run events can only be consumed once")
        self._claimed = True
        return self._engine._run(self)

    async def result(self) -> RunRecord:
        if not self._claimed:
            try:
                async for _event in self.events():
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        return await asyncio.shield(self._record)

    def _set_record(self, record: RunRecord) -> None:
        if not self._record.done():
            self._record.set_result(record)


class StrategyEngine:
    """Execute one strategy method per complete conversation snapshot."""

    def __init__(
        self,
        method: StrategyMethod,
        executor: ModelCallExecutor,
        *,
        heartbeat_seconds: float = 15.0,
        max_model_calls: int = 16,
        max_buffer_events: int = 10000,
        max_buffer_bytes: int = 16 * 1024 * 1024,
        max_buffer_seconds: float = 600.0,
        deadline_seconds: float = 600.0,
        observer: RunObserver | None = None,
        owns_executor: bool = True,
    ) -> None:
        if not callable(getattr(method, "respond", None)):
            raise TypeError("method must expose an async respond operation")
        if not callable(getattr(executor, "stream", None)):
            raise TypeError("executor must expose an async stream operation")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        if not isinstance(owns_executor, bool):
            raise TypeError("owns_executor must be a boolean")
        if observer is not None:
            for operation in (
                "run_started",
                "model_call_planned",
                "model_event",
                "model_failed",
                "decision_recorded",
                "run_finished",
                "run_failed",
            ):
                if not callable(getattr(observer, operation, None)):
                    raise TypeError(
                        f"observer must expose callable {operation}()"
                    )
        self._method = method
        self._executor = executor
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._deadline_seconds = float(deadline_seconds)
        self._scope_options = {
            "max_model_calls": max_model_calls,
            "max_buffer_events": max_buffer_events,
            "max_buffer_bytes": max_buffer_bytes,
            "max_buffer_seconds": max_buffer_seconds,
        }
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_timeout_seconds = 5.0
        self._observer = observer
        self._owns_executor = owns_executor
        self._observer_errors: list[str] = []

    @property
    def observer_errors(self) -> tuple[str, ...]:
        return tuple(self._observer_errors)

    def _record_observer_error(self, operation: str, exc: Exception) -> None:
        self._observer_errors.append(
            f"{operation}: {type(exc).__name__}: {exc}"
        )

    def _new_scope(self, metadata: RunMetadata) -> RunScope:
        return RunScope(
            self._executor,
            metadata,
            observer=self._observer,
            observer_error=self._record_observer_error,
            **self._scope_options,
        )

    def start(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> RunHandle:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(metadata, RunMetadata):
            raise TypeError("metadata must be RunMetadata")
        handle = RunHandle(self, conversation, metadata)
        handle._scope._notify(
            "run_started",
            handle._scope.run_id,
            conversation,
            metadata,
        )
        return handle

    async def _run(
        self,
        handle: RunHandle,
    ) -> AsyncIterator[ConversationEvent]:
        scope = handle._scope
        preparation: asyncio.Task[PreparedResponse] | None = None
        source: AsyncIterator[ConversationEvent] | None = None
        pending_event: asyncio.Task[ConversationEvent] | None = None
        status = "cancelled"
        error: str | None = "response stream closed"
        deadline = asyncio.get_running_loop().time() + self._deadline_seconds

        def wait_interval() -> float:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RunDeadlineExceeded("strategy run deadline exceeded")
            return min(self._heartbeat_seconds, remaining)

        try:
            preparation = asyncio.create_task(
                self._method.respond(handle._conversation, scope)
            )
            while not preparation.done():
                done, _pending = await asyncio.wait(
                    {preparation},
                    timeout=wait_interval(),
                )
                if not done:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RunDeadlineExceeded(
                            "strategy run deadline exceeded"
                        )
                    yield ConversationEvent("heartbeat")
            response = await preparation
            payload = scope._begin_response(response)
            source = scope._response_events(payload).__aiter__()
            while True:
                if pending_event is None:
                    pending_event = asyncio.create_task(anext(source))
                try:
                    event = await asyncio.wait_for(
                        asyncio.shield(pending_event),
                        timeout=wait_interval(),
                    )
                except TimeoutError:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RunDeadlineExceeded(
                            "strategy run deadline exceeded"
                        )
                    yield ConversationEvent("heartbeat")
                    continue
                except StopAsyncIteration:
                    pending_event = None
                    break
                pending_event = None
                yield event
            await scope._commit_success()
        except asyncio.CancelledError:
            status = "cancelled"
            error = "run cancelled"
            raise
        except GeneratorExit:
            status = "cancelled"
            error = "response stream closed"
            raise
        except Exception as exc:
            status = "error"
            error = type(exc).__name__
            scope._notify(
                "run_failed",
                scope.run_id,
                type(exc).__name__,
                str(exc),
            )
            raise
        else:
            status = "completed"
            error = None
        finally:
            async def finalize() -> None:
                nonlocal status, error
                try:
                    async with asyncio.timeout(self._cleanup_timeout_seconds):
                        if preparation is not None and not preparation.done():
                            preparation.cancel()
                            await asyncio.gather(
                                preparation,
                                return_exceptions=True,
                            )
                        if pending_event is not None and not pending_event.done():
                            pending_event.cancel()
                            await asyncio.gather(
                                pending_event,
                                return_exceptions=True,
                            )
                        if source is not None:
                            closer = getattr(source, "aclose", None)
                            if callable(closer):
                                try:
                                    await closer()
                                except Exception as exc:
                                    if status == "completed":
                                        status = "error"
                                        error = type(exc).__name__
                                        scope._notify(
                                            "run_failed",
                                            scope.run_id,
                                            type(exc).__name__,
                                            str(exc),
                                        )
                        await scope._cancel_active_calls()
                except TimeoutError:
                    cleanup_error = "run cleanup timed out"
                    if status == "completed":
                        status = "error"
                        error = cleanup_error
                    elif error is None:
                        error = cleanup_error
                    else:
                        error = f"{error}; {cleanup_error}"
                finally:
                    handle._set_record(scope._finish(status, error))

            cleanup = asyncio.create_task(finalize())
            self._cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(self._cleanup_tasks.discard)
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # AnyIO cancel scopes may repeatedly cancel the response task.
                # The independent bounded cleanup continues and resolves the
                # RunHandle record before provider resources are discarded.
                pass

    async def stream(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> AsyncIterator[ConversationEvent]:
        handle = self.start(conversation, metadata)
        async for event in handle.events():
            yield event

    async def complete(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> CompletedRun:
        handle = self.start(conversation, metadata)
        events = tuple([event async for event in handle.events()])
        return CompletedRun(events=events, record=await handle.result())

    async def count_tokens(self, conversation: Conversation) -> TokenCount:
        """Count every reachable transformed request without executing policy.

        The method supplies a pure, conservative candidate set.  This path
        creates no ``RunScope`` and therefore cannot classify, generate,
        mutate route memory, or emit run metrics.
        """

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        candidates_fn = getattr(self._method, "token_count_candidates", None)
        if not callable(candidates_fn):
            raise TokenCountUnavailable(
                "strategy method does not declare token-count candidates"
            )
        candidates = candidates_fn(conversation)
        if not isinstance(candidates, tuple) or not candidates:
            raise TokenCountUnavailable(
                "strategy method must declare at least one token-count candidate"
            )
        if any(not isinstance(candidate, ModelCallSpec) for candidate in candidates):
            raise TypeError(
                "token-count candidates must contain ModelCallSpec values"
            )
        counter = getattr(self._executor, "count_tokens", None)
        if not callable(counter):
            raise TokenCountUnavailable(
                "model-call executor does not support exact token counting"
            )

        counts = await asyncio.gather(*(counter(candidate) for candidate in candidates))
        normalized: list[InputTokenCount] = []
        for candidate, value in zip(candidates, counts, strict=True):
            if value is None:
                raise TokenCountUnavailable(
                    "exact token count unavailable for target "
                    f"{candidate.target_id!r}"
                )
            if not isinstance(value, InputTokenCount):
                raise TypeError(
                    "model-call token counts must be InputTokenCount values or None"
                )
            normalized.append(value)
        exact = all(value.provenance == "exact" for value in normalized)
        multiple = len(normalized) > 1
        return TokenCount(
            value=max(value.value for value in normalized),
            provenance=(
                "upper_bound" if exact and multiple
                else "exact" if exact
                else "estimated_max" if multiple
                else "estimate"
            ),
            candidate_count=len(candidates),
        )

    async def aclose(self) -> None:
        if self._cleanup_tasks:
            await asyncio.gather(
                *tuple(self._cleanup_tasks),
                return_exceptions=True,
            )
        if self._owns_executor:
            closer = getattr(self._executor, "aclose", None)
            if callable(closer):
                await closer()
