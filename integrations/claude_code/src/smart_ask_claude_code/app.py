"""External ASGI adapter from Claude Code to SmartAsk's public runtime."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from math import fsum
import os
from typing import Any

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .auth import AdapterAuthenticator
from .catalog import StrategyCatalog
from smart_ask.conversation.domain import ConversationEvent, freeze_value, thaw_value
from smart_ask.conversation.model import RunMetadata, RunRecord
from smart_ask.methods.memory import route_memory_key
from .codec import (
    AnthropicEventEncoder,
    AnthropicMessageAssembler,
    decode_request,
)
from .config import AdapterConfig, AdapterConfigError


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=status,
    )


_YELLOW = "\x1b[33m"
_LIGHT_GREEN = "\x1b[92m"
_GREEN = "\x1b[32m"
_RESET = "\x1b[0m"


def _turn_footer(
    record: RunRecord,
    fallback_model: str,
    session_total_usd: float,
) -> str:
    """Build user-visible per-turn metadata from the canonical full run."""

    final_requests = [
        request
        for request in record.provider_requests
        if request.call_id == record.final_call_id
    ]
    final_request = final_requests[-1] if final_requests else None
    model = (
        (final_request.actual_model or final_request.selected_model)
        if final_request is not None
        else None
    ) or fallback_model
    known_costs = [
        request.provider_cost_usd
        for request in record.provider_requests
        if request.provider_cost_usd is not None
    ]
    if len(known_costs) == len(record.provider_requests):
        cost = f"${fsum(known_costs):.4f} this turn"
    elif known_costs:
        cost = f"${fsum(known_costs):.4f}+ this turn"
    else:
        cost = "cost unavailable this turn"
    return (
        f"\n\n---\n"
        f"{_YELLOW}via {model}{_RESET} · "
        f"{_LIGHT_GREEN}{cost}{_RESET} · "
        f"{_GREEN}${session_total_usd:.4f} total{_RESET}"
    )


def _session_total(envelope: Mapping[str, Any]) -> float:
    session = envelope.get("session")
    resources = session.get("resources") if isinstance(session, Mapping) else None
    overall = resources.get("overall") if isinstance(resources, Mapping) else None
    value = overall.get("known_cost_usd") if isinstance(overall, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


@dataclass
class _TurnSpend:
    requests: int = 0
    known_cost_usd: float = 0.0


class _TurnBudget:
    """Bound one human instruction without limiting the whole chat session."""

    def __init__(
        self,
        *,
        max_requests: int | None,
        max_cost_usd: float | None,
        max_entries: int = 10000,
    ) -> None:
        self._max_requests = max_requests
        self._max_cost_usd = max_cost_usd
        self._max_entries = max_entries
        self._turns: OrderedDict[str, _TurnSpend] = OrderedDict()

    def begin(self, key: str | None) -> str | None:
        if key is None:
            return None
        spend = self._turns.setdefault(key, _TurnSpend())
        self._turns.move_to_end(key)
        if (
            self._max_requests is not None
            and spend.requests >= self._max_requests
        ):
            return (
                "SmartAsk stopped this instruction after "
                f"{spend.requests} model responses to prevent a runaway loop. "
                "Send a new instruction to continue."
            )
        if (
            self._max_cost_usd is not None
            and spend.known_cost_usd >= self._max_cost_usd
        ):
            return (
                "SmartAsk stopped this instruction after spending "
                f"${spend.known_cost_usd:.4f} to prevent a runaway loop. "
                "Send a new instruction to continue."
            )
        spend.requests += 1
        while len(self._turns) > self._max_entries:
            self._turns.popitem(last=False)
        return None

    def finish(self, key: str | None, record: RunRecord) -> None:
        if key is None:
            return
        spend = self._turns.get(key)
        if spend is None:
            return
        spend.known_cost_usd += fsum(
            request.provider_cost_usd
            for request in record.provider_requests
            if request.provider_cost_usd is not None
        )


class _ConcurrencyMiddleware:
    def __init__(self, app, *, limit: int):
        self.app = app
        self.semaphore = anyio.Semaphore(limit)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in (
            "/v1/messages",
            "/v1/messages/count_tokens",
        ):
            await self.app(scope, receive, send)
            return
        try:
            self.semaphore.acquire_nowait()
        except anyio.WouldBlock:
            await _error(
                503,
                "overloaded_error",
                "adapter concurrency limit reached",
            )(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.semaphore.release()


async def _body(request: Request, config: AdapterConfig) -> dict[str, Any] | Response:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > config.limits.max_request_bytes:
                return _error(413, "invalid_request_error", "request body is too large")
        except ValueError:
            return _error(400, "invalid_request_error", "invalid content-length")
    raw = await request.body()
    if len(raw) > config.limits.max_request_bytes:
        return _error(413, "invalid_request_error", "request body is too large")
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _error(400, "invalid_request_error", "request body must be JSON")
    if not isinstance(value, dict):
        return _error(400, "invalid_request_error", "request body must be an object")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def create_app(
    config: AdapterConfig,
    catalog: StrategyCatalog,
    *,
    env: Mapping[str, str] | None = None,
) -> Starlette:
    if not isinstance(config, AdapterConfig):
        raise TypeError("config must be AdapterConfig")
    if not isinstance(catalog, StrategyCatalog):
        raise TypeError("catalog must be StrategyCatalog")
    resolved_env = dict(os.environ if env is None else env)
    auth_required = not config.listen.is_loopback or config.auth.required_on_loopback
    token = resolved_env.get(config.auth.token_env)
    if auth_required and not token:
        raise AdapterConfigError(
            f"required adapter credential {config.auth.token_env} is not set"
        )
    authenticator = AdapterAuthenticator(token, required=auth_required)
    turn_budget = _TurnBudget(
        max_requests=config.limits.max_requests_per_turn,
        max_cost_usd=config.limits.max_cost_per_turn_usd,
    )
    finalizers: set[asyncio.Task[None]] = set()
    finalizer_errors: list[str] = []

    def _finalizer_done(task: asyncio.Task[None]) -> None:
        finalizers.discard(task)
        try:
            task.result()
        except Exception as exc:
            finalizer_errors.append(f"{type(exc).__name__}: {exc}")

    def _record(record: RunRecord, budget_key: str | None):
        envelope = catalog.record(record)
        turn_budget.finish(budget_key, record)
        return envelope

    async def _record_stream_run(events, handle, budget_key) -> None:
        await events.aclose()
        record = await handle.result()
        _record(record, budget_key)

    async def root(_request: Request) -> Response:
        return Response(status_code=200)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def models(request: Request) -> Response:
        if not authenticator.authenticate(request.headers):
            return _error(401, "authentication_error", "invalid adapter credential")
        return JSONResponse(catalog.discovery_payload())

    async def messages(request: Request) -> Response:
        if not authenticator.authenticate(request.headers):
            return _error(401, "authentication_error", "invalid adapter credential")
        body = await _body(request, config)
        if isinstance(body, Response):
            return body
        model_id = body.get("model")
        if (
            not isinstance(model_id, str)
            or not model_id
            or model_id != model_id.strip()
        ):
            return _error(400, "invalid_request_error", "model must be non-empty text")
        stream_requested = body.get("stream", False)
        if not isinstance(stream_requested, bool):
            return _error(400, "invalid_request_error", "stream must be boolean")
        max_tokens = body.get("max_tokens")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            return _error(
                400,
                "invalid_request_error",
                "max_tokens must be a positive integer",
            )
        if (
            config.limits.max_output_tokens is not None
            and isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and max_tokens > config.limits.max_output_tokens
        ):
            return _error(
                400,
                "invalid_request_error",
                "requested max_tokens exceeds adapter limit",
            )
        try:
            entry = catalog.resolve(model_id)
            conversation, session = decode_request(body, request.headers)
        except KeyError:
            return _error(404, "not_found_error", f"unknown model: {model_id}")
        except (TypeError, ValueError) as exc:
            return _error(400, "invalid_request_error", str(exc))

        metadata = RunMetadata(
            strategy_name=entry.loaded.config.name,
            strategy_digest=entry.loaded.digest,
            session_id=session.session_id,
            agent_id=session.agent_id,
            parent_agent_id=session.parent_agent_id,
            request_id=request.headers.get("x-request-id"),
            extensions={"principal_id": "authenticated"},
        )
        budget_key = route_memory_key(conversation, metadata)
        budget_error = turn_budget.begin(budget_key)
        if budget_error is not None:
            return _error(400, "invalid_request_error", budget_error)

        if stream_requested:
            # Do not perform a second tokenizer request before opening the
            # stream. The dedicated count endpoint remains available, while
            # generation usage is authoritative for this response.
            active_engine = entry.engine
            encoder = AnthropicEventEncoder(model_id, input_tokens=0)
            handle = active_engine.start(conversation, metadata)
            events = handle.events()

            async def stream():
                recorded = False
                terminal_chunks: list[bytes] = []
                next_block_index = 0
                block_types: dict[int, str | None] = {}
                pending_text_stop: tuple[int, bytes] | None = None
                try:
                    async for event in events:
                        if event.kind == "message_start":
                            raw = event.data.get("model") if hasattr(event.data, "get") else None
                            if isinstance(raw, str) and raw:
                                encoder.requested_model = raw.split("/")[-1]
                        if event.kind == "content_start":
                            if pending_text_stop is not None:
                                yield pending_text_stop[1]
                                pending_text_stop = None
                            index = event.data.get("index")
                            if isinstance(index, int):
                                next_block_index = max(next_block_index, index + 1)
                                block = event.data.get("block")
                                block_types[index] = (
                                    block.get("type")
                                    if isinstance(block, Mapping)
                                    else None
                                )
                        encoded = encoder.encode(event)
                        if event.kind in ("message_delta", "message_stop"):
                            if encoded is not None:
                                terminal_chunks.append(encoded)
                            continue
                        if event.kind == "content_stop":
                            index = event.data.get("index")
                            if (
                                isinstance(index, int)
                                and block_types.get(index) == "text"
                                and encoded is not None
                            ):
                                pending_text_stop = (index, encoded)
                                continue
                        if encoded is not None:
                            yield encoded

                    record = await handle.result()
                    envelope = _record(record, budget_key)
                    recorded = True
                    footer = _turn_footer(
                        record,
                        encoder.requested_model,
                        _session_total(envelope),
                    )
                    footer_index = (
                        pending_text_stop[0]
                        if pending_text_stop is not None
                        else next_block_index
                    )
                    footer_events = []
                    if pending_text_stop is None:
                        footer_events.append(ConversationEvent("content_start", {
                            "index": footer_index,
                            "block": {"type": "text"},
                        }))
                    footer_events.append(ConversationEvent("content_delta", {
                        "index": footer_index,
                        "delta": {"type": "text", "text": footer},
                    }))
                    if pending_text_stop is None:
                        footer_events.append(ConversationEvent("content_stop", {
                            "index": footer_index,
                        }))
                    for event in footer_events:
                        encoded = encoder.encode(event)
                        if encoded is not None:
                            yield encoded
                    if pending_text_stop is not None:
                        yield pending_text_stop[1]
                    for chunk in terminal_chunks:
                        yield chunk

                except anyio.get_cancelled_exc_class():
                    raise
                except Exception as exc:
                    yield encoder.error("api_error", str(exc))
                finally:
                    # Closing the engine iterator produces its canonical
                    # cancellation record. Run that cleanup in an independent
                    # task because Starlette's response cancel scope has
                    # already cancelled this body iterator on disconnect.
                    if not recorded:
                        finalizer = asyncio.create_task(
                            _record_stream_run(events, handle, budget_key)
                        )
                        finalizers.add(finalizer)
                        finalizer.add_done_callback(_finalizer_done)
                        try:
                            await asyncio.shield(finalizer)
                        except asyncio.CancelledError:
                            pass

            return StreamingResponse(
                stream(),
                status_code=200,
                media_type="text/event-stream",
            )

        assembler = AnthropicMessageAssembler(model_id)
        active_engine_nb = entry.engine
        handle = active_engine_nb.start(conversation, metadata)
        try:
            async for event in handle.events():
                assembler.observe(event)
                if event.kind == "message_start" and hasattr(event.data, "get"):
                    raw = event.data.get("model")
                    if isinstance(raw, str) and raw:
                        assembler.requested_model = raw.split("/")[-1]
        except Exception as exc:
            _record(await handle.result(), budget_key)
            return _error(500, "api_error", str(exc))
        record = await handle.result()
        envelope = _record(record, budget_key)
        message = assembler.message()
        footer = _turn_footer(
            record,
            assembler.requested_model,
            _session_total(envelope),
        )
        text_blocks = [
            block
            for block in message["content"]
            if block.get("type") == "text"
        ]
        if text_blocks:
            text_blocks[-1]["text"] += footer
        else:
            message["content"].append({"type": "text", "text": footer})
        return JSONResponse(message)

    async def count_tokens(request: Request) -> Response:
        if not authenticator.authenticate(request.headers):
            return _error(401, "authentication_error", "invalid adapter credential")
        body = await _body(request, config)
        if isinstance(body, Response):
            return body
        model_id = body.get("model")
        if not isinstance(model_id, str) or not model_id:
            return _error(400, "invalid_request_error", "model must be non-empty text")
        try:
            entry = catalog.resolve(model_id)
            conversation, _session = decode_request(body, request.headers)
            count = await entry.engine.count_tokens(conversation)
        except KeyError:
            return _error(404, "not_found_error", f"unknown model: {model_id}")
        except (TypeError, ValueError) as exc:
            return _error(400, "invalid_request_error", str(exc))
        except Exception as exc:
            return _error(500, "api_error", str(exc))
        return JSONResponse({"input_tokens": count.value})

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            if finalizers:
                await asyncio.gather(*tuple(finalizers), return_exceptions=True)
            await catalog.aclose()

    app = Starlette(
        routes=[
            Route("/", root, methods=["HEAD"]),
            Route("/healthz", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
            Route("/v1/messages", messages, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        _ConcurrencyMiddleware,
        limit=config.limits.max_concurrent_requests,
    )
    app.state.strategy_catalog = catalog
    app.state.finalizer_errors = finalizer_errors
    return app
