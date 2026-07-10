"""External ASGI adapter from Claude Code to SmartAsk's public runtime."""

from __future__ import annotations

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

from .auth import AdapterAuthenticator
from .catalog import StrategyCatalog
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
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _error(400, "invalid_request_error", "request body must be JSON")
    if not isinstance(value, dict):
        return _error(400, "invalid_request_error", "request body must be an object")
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

        if stream_requested:
            try:
                input_tokens = await entry.runtime.count_tokens(
                    conversation,
                    session,
                )
            except Exception:
                # Token counting is useful protocol metadata, but generation
                # remains authoritative when a backend cannot pre-count.
                input_tokens = 0
            encoder = AnthropicEventEncoder(
                model_id,
                input_tokens=input_tokens,
            )

            async def stream():
                try:
                    async for event in entry.runtime.stream(conversation, session):
                        encoded = encoder.encode(event)
                        if encoded is not None:
                            yield encoded
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception as exc:
                    yield encoder.error("api_error", str(exc))

            return StreamingResponse(
                stream(),
                status_code=200,
                media_type="text/event-stream",
            )

        assembler = AnthropicMessageAssembler(model_id)
        try:
            async for event in entry.runtime.stream(conversation, session):
                assembler.observe(event)
        except Exception as exc:
            return _error(500, "api_error", str(exc))
        return JSONResponse(assembler.message())

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
            conversation, session = decode_request(body, request.headers)
            count = await entry.runtime.count_tokens(conversation, session)
        except KeyError:
            return _error(404, "not_found_error", f"unknown model: {model_id}")
        except (TypeError, ValueError) as exc:
            return _error(400, "invalid_request_error", str(exc))
        except Exception as exc:
            return _error(500, "api_error", str(exc))
        return JSONResponse({"input_tokens": count})

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
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
    return app
