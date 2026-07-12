"""SmartAsk-owned routing, execution, state, cascade, and metrics runtime."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
import time
from typing import Any

from ..domain import Attempt, Context, ModelResult, RouteResult, RoutingEvent, Task
from ..methods import CascadeRoutingMethod, DifficultyRoutingMethod
from ..routing import SmartRouter
from ..strategy.loader import LoadedStrategy
from ..strategy.schema import CascadeMethodConfig, FixedMethodConfig
from .domain import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationRequest,
    SessionContext,
    thaw_value,
)
from .executor import ConversationExecutor
from .metrics import (
    ConversationAttemptMeasurement,
    ConversationMetricsStore,
    ConversationRunMeasurement,
)


@dataclass(frozen=True)
class _StoredRoute:
    fingerprint: str
    route: RouteResult
    stored_at: float


class _TurnRoutes:
    def __init__(self, *, ttl_seconds: float, max_entries: int):
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, str], _StoredRoute] = OrderedDict()

    @staticmethod
    def key(context: SessionContext) -> tuple[str, str] | None:
        if context.session_id is None:
            return None
        return context.session_id, context.agent_id or context.parent_agent_id or "root"

    def get(
        self,
        key: tuple[str, str] | None,
        fingerprint: str | None,
    ) -> RouteResult | None:
        if key is None or fingerprint is None:
            return None
        now = time.monotonic()
        self._prune(now)
        value = self._entries.get(key)
        if value is None or value.fingerprint != fingerprint:
            return None
        self._entries.move_to_end(key)
        return value.route

    def put(
        self,
        key: tuple[str, str] | None,
        fingerprint: str | None,
        route: RouteResult,
    ) -> None:
        if key is None or fingerprint is None:
            return
        now = time.monotonic()
        self._prune(now)
        self._entries[key] = _StoredRoute(fingerprint, route, now)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, value in self._entries.items()
            if now - value.stored_at > self._ttl
        ]
        for key in expired:
            self._entries.pop(key, None)


def _routing_event(event: RoutingEvent) -> dict[str, str | None]:
    return {
        "source": event.source,
        "outcome": event.outcome,
        "reason": event.reason,
        "model": event.model,
    }


def _conversation_value(request: ConversationRequest) -> dict[str, Any]:
    return {
        "system": thaw_value(request.system),
        "messages": [
            {
                "role": message.role,
                "content": thaw_value(message.content),
                "extensions": thaw_value(message.extensions),
            }
            for message in request.messages
        ],
        "tools": thaw_value(request.tools),
        "parameters": thaw_value(request.parameters),
        "extensions": thaw_value(request.extensions),
    }


def _route_value(route: RouteResult) -> dict[str, Any]:
    return {
        "action": route.action,
        "model": route.model,
        "role": route.role,
        "phase": route.phase,
        "label": route.label,
        "routing_prompt": route.prompt,
        "events": [_routing_event(event) for event in route.routing_events],
    }


def _finish_reason(native: str | None) -> str:
    return {
        "max_tokens": "length",
        "length": "length",
        "refusal": "refusal",
        "content_filter": "content_filter",
        "tool_call": "tool_call",
        "tool_calls": "tool_call",
        "tool_use": "tool_call",
        "stop": "stop",
        "end_turn": "stop",
    }.get(native or "", "unknown")


class ConversationRuntime:
    """Route complete conversations through a strategy-configured executor."""

    def __init__(
        self,
        *,
        loaded_strategy: LoadedStrategy,
        router: SmartRouter,
        executor: ConversationExecutor,
        metrics: ConversationMetricsStore | None = None,
        decision_ttl_seconds: float = 3600.0,
        max_cached_turns: int = 10000,
        heartbeat_seconds: float = 15.0,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        if not isinstance(loaded_strategy, LoadedStrategy):
            raise TypeError("loaded_strategy must be a LoadedStrategy")
        if not isinstance(router, SmartRouter):
            raise TypeError("router must be a SmartRouter")
        if not callable(getattr(executor, "stream", None)):
            raise TypeError("executor must expose an async stream operation")
        if not callable(getattr(executor, "count_tokens", None)):
            raise TypeError("executor must expose an async count_tokens operation")
        self._loaded = loaded_strategy
        self._router = router
        self._executor = executor
        self._metrics = metrics or ConversationMetricsStore()
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if trace_sink is not None and not callable(trace_sink):
            raise TypeError("trace_sink must be callable or None")
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._trace_sink = trace_sink
        self._trace_errors: list[str] = []
        self._turn_routes = _TurnRoutes(
            ttl_seconds=decision_ttl_seconds,
            max_entries=max_cached_turns,
        )

    @property
    def strategy_name(self) -> str:
        return self._loaded.config.name

    @property
    def strategy_digest(self) -> str:
        return self._loaded.digest

    @property
    def metrics(self) -> ConversationMetricsStore:
        return self._metrics

    @property
    def trace_errors(self) -> tuple[str, ...]:
        return tuple(self._trace_errors)

    async def aclose(self) -> None:
        closer = getattr(self._executor, "aclose", None)
        if callable(closer):
            await closer()

    async def _select(
        self,
        request: ConversationRequest,
        session: SessionContext,
    ) -> tuple[Task, RouteResult, object, str | None, tuple[str, str] | None]:
        projected = request.latest_human_instruction()
        prompt = projected[0] if projected is not None else "Conversation continuation"
        fingerprint = projected[1] if projected is not None else None
        task = Task(prompt, task_id=session.session_id)
        key = self._turn_routes.key(session)
        method = self._router.method
        if isinstance(method, (DifficultyRoutingMethod, CascadeRoutingMethod)):
            cached = self._turn_routes.get(key, fingerprint)
            if cached is not None:
                with self._router.capture_stats(task_id=task.task_id) as capture:
                    empty_capture = capture
                return task, cached, empty_capture.stats, fingerprint, key
        route, stats = await asyncio.to_thread(self._router.plan_with_stats, task)
        if route.action != "execute":
            raise RuntimeError("fresh conversation routing must select an executor")
        if isinstance(method, (DifficultyRoutingMethod, CascadeRoutingMethod)):
            self._turn_routes.put(key, fingerprint, route)
        return task, route, stats, fingerprint, key

    def _prepare(
        self,
        request: ConversationRequest,
        route: RouteResult,
    ) -> ConversationRequest:
        prepared = request
        profile = next(
            profile
            for profile in self._loaded.config.model_profiles
            if profile.model == route.model
        )
        if profile.system_prompt is not None:
            prepared = prepared.with_system_text(
                self._loaded.resolve_prompt(profile.system_prompt)
            )
        parameter_updates = {}
        if profile.parameters.max_tokens is not None:
            parameter_updates["max_tokens"] = profile.parameters.max_tokens
        if profile.parameters.temperature is not None:
            parameter_updates["temperature"] = profile.parameters.temperature
        if profile.parameters.reasoning_effort is not None:
            parameter_updates["reasoning_effort"] = (
                profile.parameters.reasoning_effort
            )
        if parameter_updates:
            prepared = prepared.with_parameters(parameter_updates)
        method = self._loaded.config.method
        if isinstance(method, FixedMethodConfig):
            if method.prompt_prefix is not None:
                prepared = prepared.with_latest_human_text(
                    self._loaded.resolve_prompt(method.prompt_prefix),
                    before=True,
                )
            if method.prompt_suffix is not None:
                prepared = prepared.with_latest_human_text(
                    self._loaded.resolve_prompt(method.prompt_suffix),
                    before=False,
                )
        elif isinstance(method, CascadeMethodConfig):
            if route.phase == "initial-easy":
                prepared = prepared.with_latest_human_text(
                    self._loaded.resolve_prompt(
                        method.escalation.self_check_suffix
                    ),
                    before=False,
                )
            elif route.phase == "escalation":
                prepared = prepared.with_latest_human_text(
                    self._loaded.resolve_prompt(
                        method.escalation.escalation_prefix
                    ),
                    before=True,
                )
        return prepared

    async def _events_for_attempt(
        self,
        route: RouteResult,
        prepared: ConversationRequest,
        measurement: ConversationAttemptMeasurement,
        trace_events: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ConversationEvent]:
        if route.model is None or route.role is None:
            raise RuntimeError("execute route lost model or role")
        execution = ConversationExecutionRequest(
            model=route.model,
            role=route.role,
            conversation=prepared,
        )
        try:
            async for event in self._executor.stream(execution):
                if not isinstance(event, ConversationEvent):
                    raise TypeError("conversation executors must emit ConversationEvent")
                measurement.observe(event)
                if trace_events is not None and event.kind != "heartbeat":
                    trace_events.append({
                        "kind": event.kind,
                        "data": thaw_value(event.data),
                    })
                yield event
        except asyncio.CancelledError:
            measurement.finish(error="conversation cancelled", cancelled=True)
            raise
        except Exception as exc:
            measurement.finish(error=str(exc))
            raise
        else:
            measurement.finish()

    async def _with_heartbeats(
        self,
        source: AsyncIterator[ConversationEvent],
    ) -> AsyncIterator[ConversationEvent]:
        iterator = source.__aiter__()
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.create_task(anext(iterator))
                try:
                    event = await asyncio.wait_for(
                        asyncio.shield(pending),
                        timeout=self._heartbeat_seconds,
                    )
                except TimeoutError:
                    yield ConversationEvent("heartbeat")
                    continue
                except StopAsyncIteration:
                    return
                pending = None
                yield event
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            closer = getattr(iterator, "aclose", None)
            if callable(closer):
                await closer()

    async def stream(
        self,
        request: ConversationRequest,
        session: SessionContext | None = None,
    ) -> AsyncIterator[ConversationEvent]:
        """Execute one conversation request and always retain runtime evidence."""

        if not isinstance(request, ConversationRequest):
            raise TypeError("request must be a ConversationRequest")
        session = session or SessionContext()
        if not isinstance(session, SessionContext):
            raise TypeError("session must be a SessionContext or None")
        run = ConversationRunMeasurement(
            strategy_name=self.strategy_name,
            strategy_digest=self.strategy_digest,
            session_id=session.session_id,
            agent_id=session.agent_id,
            parent_agent_id=session.parent_agent_id,
        )
        trace = None
        if self._trace_sink is not None:
            trace = {
                "schema": "smart-ask.conversation-trace/v1",
                "run_id": run.run_id,
                "strategy": {
                    "name": self.strategy_name,
                    "digest": self.strategy_digest,
                },
                "session_id": session.session_id,
                "agent_id": session.agent_id,
                "parent_agent_id": session.parent_agent_id,
                "request": _conversation_value(request),
                "routing_task": None,
                "routes": [],
                "attempts": [],
            }
        recorded = False
        try:
            task, route, classifier_stats, fingerprint, key = await self._select(
                request,
                session,
            )
            if trace is not None:
                trace["routing_task"] = task.prompt
                trace["routes"].append(_route_value(route))
            run.classifier_stats = classifier_stats
            run.routing_events.extend(
                _routing_event(event) for event in route.routing_events
            )
            first = run.start_attempt(
                phase=route.phase,
                role=route.role or "generation",
                model=route.model or "unknown",
            )
            cascade_candidate = (
                isinstance(self._router.method, CascadeRoutingMethod)
                and route.phase == "initial-easy"
            )
            first_prepared = self._prepare(request, route)
            first_trace = None
            if trace is not None:
                first_trace = {
                    "phase": route.phase,
                    "role": route.role,
                    "selected_model": route.model,
                    "effective_context": _conversation_value(first_prepared),
                    "output_events": [],
                }
                trace["attempts"].append(first_trace)
            if not cascade_candidate:
                async for event in self._events_for_attempt(
                    route,
                    first_prepared,
                    first,
                    None if first_trace is None else first_trace["output_events"],
                ):
                    yield event
                return

            buffered: list[ConversationEvent] = []
            passthrough = False
            async for event in self._with_heartbeats(
                self._events_for_attempt(
                    route,
                    first_prepared,
                    first,
                    None if first_trace is None else first_trace["output_events"],
                )
            ):
                if event.kind == "heartbeat":
                    yield event
                    continue
                if passthrough:
                    yield event
                    continue
                buffered.append(event)
                if first.tool_call_count:
                    passthrough = True
                    for buffered_event in buffered:
                        yield buffered_event
                    buffered.clear()
            if passthrough:
                return

            result = ModelResult(
                model=first.actual_model or route.model,
                text=first.text,
                raw_text=first.text,
                usage={
                    "prompt_tokens": first.input_tokens,
                    "completion_tokens": first.output_tokens,
                    "total_tokens": (
                        first.input_tokens + first.output_tokens
                        if first.input_tokens is not None
                        and first.output_tokens is not None
                        else None
                    ),
                },
                finish_reason=_finish_reason(first.stop_reason),
                reasoning_tokens=first.reasoning_tokens,
                cached_input_tokens=first.cache_read_tokens,
                cache_write_input_tokens=first.cache_write_tokens,
                provider_cost_usd=first.provider_cost_usd,
            )
            context = Context(
                attempts=(Attempt(route, result),),
                routing_events=route.routing_events,
            )
            next_route = self._router.route(task, context)
            if trace is not None:
                trace["routes"].append(_route_value(next_route))
            run.routing_events.extend(
                _routing_event(event) for event in next_route.routing_events
            )
            if next_route.action == "accept":
                for event in buffered:
                    yield event
                return
            self._turn_routes.put(key, fingerprint, next_route)
            second = run.start_attempt(
                phase=next_route.phase,
                role=next_route.role or "generation",
                model=next_route.model or "unknown",
            )
            second_prepared = self._prepare(request, next_route)
            second_trace = None
            if trace is not None:
                second_trace = {
                    "phase": next_route.phase,
                    "role": next_route.role,
                    "selected_model": next_route.model,
                    "effective_context": _conversation_value(second_prepared),
                    "output_events": [],
                }
                trace["attempts"].append(second_trace)
            async for event in self._events_for_attempt(
                next_route,
                second_prepared,
                second,
                None if second_trace is None else second_trace["output_events"],
            ):
                yield event
        except asyncio.CancelledError:
            run.error = "conversation cancelled"
            run.cancelled = True
            raise
        except Exception as exc:
            run.error = str(exc)
            raise
        finally:
            if not recorded:
                self._metrics.record(run)
                recorded = True
            if trace is not None:
                trace["routing_events"] = list(run.routing_events)
                trace["error"] = run.error
                trace["cancelled"] = run.cancelled
                for attempt_trace, measurement in zip(
                    trace["attempts"],
                    run.attempts,
                ):
                    attempt_trace["actual_model"] = measurement.actual_model
                    attempt_trace["output_text"] = measurement.text
                    attempt_trace["stop_reason"] = measurement.stop_reason
                    attempt_trace["error"] = measurement.error
                    attempt_trace["cancelled"] = measurement.cancelled
                try:
                    self._trace_sink(trace)
                except Exception as exc:
                    self._trace_errors.append(f"{type(exc).__name__}: {exc}")

    async def count_tokens(
        self,
        request: ConversationRequest,
        session: SessionContext | None = None,
    ) -> int:
        """Count with the configured executor, falling back to a documented estimate."""

        session = session or SessionContext()
        method = self._loaded.config.method
        profile = method.model if isinstance(method, FixedMethodConfig) else method.hard
        route = RouteResult(
            action="execute",
            model=profile.model,
            prompt=(request.latest_human_instruction() or ("continuation", ""))[0],
            role="writer",
            phase="fixed" if isinstance(method, FixedMethodConfig) else "initial-hard",
            label="conversation token count",
        )
        execution = ConversationExecutionRequest(
            model=route.model,
            role=route.role,
            conversation=self._prepare(request, route),
        )
        count = await self._executor.count_tokens(execution)
        if count is not None:
            return count
        text = "".join(
            str(block.get("text", ""))
            for message in request.messages
            for block in message.content
        )
        return max(1, (len(text) + 3) // 4)
