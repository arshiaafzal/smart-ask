"""External ASGI adapter from Claude Code to SmartAsk's public runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
import json
import os
from typing import Any

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import sys

from .auth import AdapterAuthenticator
from .catalog import StrategyCatalog
from smart_ask.conversation.domain import ConversationEvent, freeze_value, thaw_value
from smart_ask.conversation.model import RunMetadata
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
    finalizers: set[asyncio.Task[None]] = set()
    finalizer_errors: list[str] = []

    def _finalizer_done(task: asyncio.Task[None]) -> None:
        finalizers.discard(task)
        try:
            task.result()
        except Exception as exc:
            finalizer_errors.append(f"{type(exc).__name__}: {exc}")

    async def _record_stream_run(events, handle) -> None:
        await events.aclose()
        record = await handle.result()
        catalog.record(record)

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

        print(f"[SA] stream_requested={stream_requested}", file=sys.stderr, flush=True)
        if stream_requested:
            # Do not perform a second tokenizer request before opening the
            # stream. The dedicated count endpoint remains available, while
            # generation usage is authoritative for this response.
            active_engine = entry.engine
            encoder = AnthropicEventEncoder(model_id, input_tokens=0)
            handle = active_engine.start(conversation, metadata)
            events = handle.events()

            def _raw_sse(event_name: str, value: dict[str, Any]) -> bytes:
                return f"event: {event_name}\ndata: {json.dumps(value, separators=(',', ':'))}\n\n".encode()

            async def stream():
                actual_model: str | None = None
                cost_usd: float | None = None
                input_tokens: int | None = None
                output_tokens: int | None = None
                last_text_block_index: int | None = None
                pending_end: list[bytes] = []
                try:
                    async for event in events:
                        # Capture the real model name and update Claude Code's
                        # model bar to show it instead of the strategy alias.
                        if event.kind == "message_start":
                            raw = event.data.get("model") if hasattr(event.data, "get") else None
                            if isinstance(raw, str) and raw:
                                actual_model = raw
                                encoder.requested_model = raw.split("/")[-1]

                        # Track the last text content block so we can
                        # append the routing footer to it before it closes.
                        if event.kind == "content_start" and hasattr(event.data, "get"):
                            block = event.data.get("block", {})
                            if hasattr(block, "get") and block.get("type") == "text":
                                idx = event.data.get("index")
                                if isinstance(idx, int):
                                    last_text_block_index = idx

                        # Capture cost and token counts from the usage event
                        # emitted at the end of the generator stream.
                        if event.kind == "usage" and hasattr(event.data, "get"):
                            d = event.data
                            c = d.get("provider_cost_usd")
                            if isinstance(c, (int, float)) and not isinstance(c, bool):
                                cost_usd = float(c)
                            for attr, key in (
                                ("input_tokens", "input_tokens"),
                                ("output_tokens", "output_tokens"),
                            ):
                                v = d.get(key)
                                if isinstance(v, int) and not isinstance(v, bool):
                                    if attr == "input_tokens":
                                        input_tokens = v
                                    else:
                                        output_tokens = v

                        # Buffer content_stop, message_delta, and message_stop
                        # so we can inject the footer before the block closes.
                        if event.kind in ("content_stop", "message_delta", "message_stop"):
                            encoded = encoder.encode(event)
                            if encoded is not None:
                                pending_end.append(encoded)
                            continue

                        encoded = encoder.encode(event)
                        if encoded is not None:
                            yield encoded

                    print(f"[SA] stream done. model={actual_model} cost={cost_usd} out_tok={output_tokens} last_text_block={last_text_block_index}", file=sys.stderr, flush=True)
                    # Append routing footer as a final delta to the last text
                    # block before it is closed.  Staying in the same block
                    # guarantees Claude Code renders it without special cases.
                    if last_text_block_index is not None and (
                        actual_model is not None
                        or cost_usd is not None
                        or output_tokens is not None
                    ):
                        total_tok = (input_tokens or 0) + (output_tokens or 0)
                        parts: list[str] = []
                        if actual_model is not None:
                            parts.append(f"via {actual_model.split('/')[-1]}")
                        if cost_usd is not None:
                            parts.append(f"${cost_usd:.6f}")
                        if total_tok:
                            parts.append(f"{total_tok:,} tok")
                        footer = "\n\n---\n*" + " · ".join(parts) + "*"
                        print(f"[SA] injecting footer to block {last_text_block_index}: {footer!r}", file=sys.stderr, flush=True)
                        yield _raw_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": last_text_block_index,
                            "delta": {"type": "text_delta", "text": footer},
                        })

                    for chunk in pending_end:
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
                    finalizer = asyncio.create_task(
                        _record_stream_run(events, handle)
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
        nb_actual_model: str | None = None
        nb_cost_usd: float | None = None
        nb_input_tokens: int | None = None
        nb_output_tokens: int | None = None
        try:
            async for event in handle.events():
                assembler.observe(event)
                if event.kind == "message_start" and hasattr(event.data, "get"):
                    raw = event.data.get("model")
                    if isinstance(raw, str) and raw:
                        nb_actual_model = raw
                if event.kind == "usage" and hasattr(event.data, "get"):
                    d = event.data
                    c = d.get("provider_cost_usd")
                    if isinstance(c, (int, float)) and not isinstance(c, bool):
                        nb_cost_usd = float(c)
                    v_in = d.get("input_tokens")
                    if isinstance(v_in, int) and not isinstance(v_in, bool):
                        nb_input_tokens = v_in
                    v_out = d.get("output_tokens")
                    if isinstance(v_out, int) and not isinstance(v_out, bool):
                        nb_output_tokens = v_out
        except Exception as exc:
            catalog.record(await handle.result())
            return _error(500, "api_error", str(exc))
        catalog.record(await handle.result())
        msg = assembler.message()
        print(f"[SA] non-stream done. model={nb_actual_model} cost={nb_cost_usd} out_tok={nb_output_tokens}", file=sys.stderr, flush=True)
        if nb_actual_model is not None or nb_cost_usd is not None or nb_output_tokens is not None:
            total_tok = (nb_input_tokens or 0) + (nb_output_tokens or 0)
            parts: list[str] = []
            if nb_actual_model is not None:
                parts.append(f"via {nb_actual_model.split('/')[-1]}")
            if nb_cost_usd is not None:
                parts.append(f"${nb_cost_usd:.6f}")
            if total_tok:
                parts.append(f"{total_tok:,} tok")
            footer = "\n\n---\n*" + " · ".join(parts) + "*"
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = block["text"] + footer
                    print(f"[SA] non-stream footer injected: {footer!r}", file=sys.stderr, flush=True)
                    break
        return JSONResponse(msg)

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
