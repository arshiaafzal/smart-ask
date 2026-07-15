"""Anthropic-compatible ASGI gateway to the SmartAsk engine."""

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

from .auth import GatewayAuthenticator
from .catalog import StrategyCatalog
from smart_ask.conversation.model import RunMetadata
from .codec import (
    AnthropicEventEncoder,
    AnthropicMessageAssembler,
    decode_request,
)
from .config import GatewayConfig, GatewayConfigError


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
                "gateway concurrency limit reached",
            )(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.semaphore.release()


async def _body(request: Request, config: GatewayConfig) -> dict[str, Any] | Response:
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
    config: GatewayConfig,
    catalog: StrategyCatalog,
    *,
    env: Mapping[str, str] | None = None,
) -> Starlette:
    if not isinstance(config, GatewayConfig):
        raise TypeError("config must be GatewayConfig")
    if not isinstance(catalog, StrategyCatalog):
        raise TypeError("catalog must be StrategyCatalog")
    resolved_env = dict(os.environ if env is None else env)
    auth_required = not config.listen.is_loopback or config.auth.required_on_loopback
    token = resolved_env.get(config.auth.token_env)
    if auth_required and not token:
        raise GatewayConfigError(
            f"required gateway credential {config.auth.token_env} is not set"
        )
    authenticator = GatewayAuthenticator(token, required=auth_required)
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
            return _error(401, "authentication_error", "invalid gateway credential")
        return JSONResponse(catalog.discovery_payload())

    async def messages(request: Request) -> Response:
        if not authenticator.authenticate(request.headers):
            return _error(401, "authentication_error", "invalid gateway credential")
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
                "requested max_tokens exceeds gateway limit",
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

        if stream_requested:
            # Do not perform a second tokenizer request before opening the
            # stream. The dedicated count endpoint remains available, while
            # generation usage is authoritative for this response.
            encoder = AnthropicEventEncoder(model_id, input_tokens=0)
            handle = entry.engine.start(conversation, metadata)
            events = handle.events()

            async def stream():
                try:
                    async for event in events:
                        encoded = encoder.encode(event)
                        if encoded is not None:
                            yield encoded
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
        handle = entry.engine.start(conversation, metadata)
        try:
            async for event in handle.events():
                assembler.observe(event)
        except Exception as exc:
            catalog.record(await handle.result())
            return _error(500, "api_error", str(exc))
        catalog.record(await handle.result())
        return JSONResponse(assembler.message())

    async def count_tokens(request: Request) -> Response:
        if not authenticator.authenticate(request.headers):
            return _error(401, "authentication_error", "invalid gateway credential")
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
