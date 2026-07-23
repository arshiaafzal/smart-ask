"""Native Anthropic Messages API transport.

Keeping a native Anthropic hop avoids lossy OpenAI-compatible translations for
thinking and tool-use blocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
import json
import math
from numbers import Integral
from typing import Any

import httpx

from ..conversation.domain import ConversationEvent, thaw_value
from ..conversation.model import InputTokenCount
from ._protocol import ProviderCall
from .ollama import UnsupportedConversationFeature


# Direct API list prices in USD per million tokens.  Cache writes are 1.25x
# input and cache reads are 0.1x input for the five-minute cache.
_MODEL_PRICES = {
    "claude-sonnet-4": (3.0, 15.0),
    "claude-opus-4": (5.0, 25.0),
}
_MAX_CACHE_BREAKPOINTS = 4


def _price(model: str) -> tuple[float, float] | None:
    normalized = model.lower()
    for fragment, prices in _MODEL_PRICES.items():
        if fragment in normalized:
            return prices
    return None


def _cost(model: str, usage: Mapping[str, Any]) -> float | None:
    prices = _price(model)
    if prices is None:
        return None
    input_price, output_price = prices
    input_tokens = _token(usage.get("input_tokens"))
    output_tokens = _token(usage.get("output_tokens"))
    cache_write = _token(usage.get("cache_creation_input_tokens"))
    cache_read = _token(usage.get("cache_read_input_tokens"))
    # Anthropic reports uncached input separately from cache tokens.
    return (
        input_tokens * input_price
        + cache_write * input_price * 1.25
        + cache_read * input_price * 0.1
        + output_tokens * output_price
    ) / 1_000_000


def _token(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _extensions(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("extensions")
    return thaw_value(raw) if isinstance(raw, Mapping) else {}


def _extra(value: Mapping[str, Any], canonical: set[str]) -> dict[str, Any]:
    """Preserve open-schema fields without allowing canonical overrides."""

    extra = _extensions(value)
    extra.update({
        key: thaw_value(item)
        for key, item in value.items()
        if key not in canonical and key != "extensions"
    })
    return extra


def _encode_block(block: Mapping[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    extra = _extra(block, {"type", "text", "thinking", "signature", "data",
                           "media_type", "source_type", "id", "name",
                           "arguments", "content", "is_error"})
    if block_type == "text":
        return {**extra, "type": "text", "text": block.get("text", "")}
    if block_type == "image":
        data = block.get("data")
        media_type = block.get("media_type")
        if not isinstance(data, str) or not isinstance(media_type, str):
            raise UnsupportedConversationFeature(
                "images require base64 data and a media type"
            )
        return {
            **extra,
            "type": "image",
            "source": {
                "type": block.get("source_type") or "base64",
                "media_type": media_type,
                "data": data,
            },
        }
    if block_type == "thinking":
        value = {
            **extra,
            "type": "thinking",
            "thinking": block.get("thinking", ""),
        }
        signature = block.get("signature")
        if isinstance(signature, str) and signature:
            value["signature"] = signature
        return value
    if block_type == "tool_call":
        call_id = block.get("id")
        name = block.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise UnsupportedConversationFeature("tool calls require a non-empty id")
        if not isinstance(name, str) or not name:
            raise UnsupportedConversationFeature("tool calls require a non-empty name")
        return {
            **extra,
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": thaw_value(block.get("arguments", {})),
        }
    if block_type == "tool_result":
        call_id = block.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise UnsupportedConversationFeature(
                "tool results require a non-empty tool-use id"
            )
        content = block.get("content", "")
        if isinstance(content, tuple):
            content = [_encode_block(item) for item in content]
        value = {
            **extra,
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": thaw_value(content),
        }
        if block.get("is_error") is True:
            value["is_error"] = True
        return value
    if block_type == "redacted_thinking":
        return thaw_value(block)
    raise UnsupportedConversationFeature(
        f"Anthropic execution cannot encode block type {block_type!r}"
    )


def _messages(request) -> list[dict[str, Any]]:
    messages = []
    for message in request.messages:
        if message.role in ("system", "developer"):
            continue
        # Some harnesses append an empty assistant placeholder before sending.
        # It is not conversation content and the Messages API rejects it.
        if not message.content:
            continue
        if message.role not in ("user", "assistant"):
            raise UnsupportedConversationFeature(
                f"unsupported Anthropic message role: {message.role!r}"
            )
        messages.append({
            **thaw_value(message.extensions),
            "role": message.role,
            "content": [_encode_block(block) for block in message.content],
        })
    return messages


def _system(blocks) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        if block.get("type") != "text":
            raise UnsupportedConversationFeature(
                f"unsupported Anthropic system block: {block.get('type')!r}"
            )
        result.append({
            **_extra(block, {"type", "text"}),
            "type": "text",
            "text": block.get("text", ""),
        })
    return result


def _tools(tools) -> list[dict[str, Any]]:
    values = []
    for tool in tools:
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
            raise UnsupportedConversationFeature(
                "tools require a non-empty name and input_schema mapping"
            )
        value = {
            **_extensions(tool),
            "name": name,
            "input_schema": thaw_value(schema),
        }
        description = tool.get("description")
        if isinstance(description, str):
            value["description"] = description
        values.append(value)
    return values


def _ensure_recent_cache_breakpoints(
    *,
    system: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> None:
    """Use spare Anthropic breakpoints on recent user/tool boundaries.

    Coding harnesses commonly move one ephemeral marker to the newest tool
    result. Keeping the immediately preceding user boundary marked as well
    gives the next request an exact lookup point for the prefix cached by the
    prior request. Existing caller markers are preserved and the provider's
    four-breakpoint limit is never exceeded.
    """

    blocks = [*system, *tools]
    blocks.extend(
        block
        for message in messages
        for block in message.get("content", [])
        if isinstance(block, dict)
    )
    count = sum("cache_control" in block for block in blocks)
    if count >= _MAX_CACHE_BREAKPOINTS:
        return

    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        block = next(
            (
                value
                for value in reversed(content)
                if isinstance(value, dict)
                and value.get("type") in ("text", "tool_result")
            ),
            None,
        )
        if block is None or "cache_control" in block:
            continue
        block["cache_control"] = {"type": "ephemeral"}
        count += 1
        if count >= _MAX_CACHE_BREAKPOINTS:
            return


def _usage(model: str, value: Mapping[str, Any]) -> dict[str, Any]:
    uncached_input = _token(value.get("input_tokens"))
    output_tokens = _token(value.get("output_tokens"))
    cache_read = _token(value.get("cache_read_input_tokens"))
    cache_write = _token(value.get("cache_creation_input_tokens"))
    input_tokens = uncached_input + cache_read + cache_write
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "provider_cost_usd": _cost(model, value),
    }


def _error_message(body: Any, fallback: str) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return error["message"]
    return fallback


def _retry_after(value: str | None) -> float | None:
    """Parse a delta-seconds Retry-After value and cap excessive waits."""

    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, 60.0)


class _TransientAnthropicError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AnthropicTransport:
    """Stream a neutral conversation through Anthropic's Messages API."""

    def __init__(self, client: httpx.AsyncClient, *, default_max_tokens: int):
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        if (
            isinstance(default_max_tokens, bool)
            or not isinstance(default_max_tokens, Integral)
            or default_max_tokens < 1
        ):
            raise ValueError("default_max_tokens must be a positive integer")
        self._client = client
        self._default_max_tokens = int(default_max_tokens)

    def _payload(self, request: ProviderCall, *, stream: bool) -> dict[str, Any]:
        conversation = request.conversation
        parameters = conversation.parameters
        messages = _messages(conversation)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": parameters.get("max_tokens", self._default_max_tokens),
            "stream": stream,
        }
        message_system = tuple(
            block
            for message in conversation.messages
            if message.role in ("system", "developer")
            for block in message.content
        )
        system = _system(conversation.system + message_system)
        if system:
            payload["system"] = system
        tools = _tools(conversation.tools)
        if tools:
            payload["tools"] = tools
        _ensure_recent_cache_breakpoints(
            system=system,
            tools=tools,
            messages=messages,
        )
        for source, target in (
            ("temperature", "temperature"),
            ("stop", "stop_sequences"),
            ("thinking", "thinking"),
            ("tool_choice", "tool_choice"),
        ):
            if source in parameters:
                payload[target] = thaw_value(parameters[source])
        return payload

    async def stream(
        self,
        request: ProviderCall,
    ) -> AsyncIterator[ConversationEvent]:
        """Retry transient failures only when no output has reached the caller."""

        for attempt in range(5):
            emitted = False
            try:
                async for event in self._stream_once(request):
                    emitted = True
                    yield event
                return
            except (_TransientAnthropicError, httpx.TransportError) as exc:
                if emitted or attempt == 4:
                    raise
                delay = (
                    exc.retry_after
                    if isinstance(exc, _TransientAnthropicError)
                    and exc.retry_after is not None
                    else 2**attempt
                )
                await asyncio.sleep(delay)

    async def _stream_once(
        self,
        request: ProviderCall,
    ) -> AsyncIterator[ConversationEvent]:
        if not isinstance(request, ProviderCall):
            raise TypeError("request must be a ProviderCall")
        started = False
        terminal = False
        actual_model = request.model
        usage: dict[str, Any] = {}
        open_blocks: set[int] = set()
        async with self._client.stream(
            "POST", "/v1/messages", json=self._payload(request, stream=True)
        ) as response:
            if response.is_error:
                raw = await response.aread()
                try:
                    body = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = None
                message = (
                    f"Anthropic API returned {response.status_code}: "
                    f"{_error_message(body, response.reason_phrase)}"
                )
                if response.status_code in (408, 409, 425, 429) or (
                    response.status_code >= 500
                ):
                    raise _TransientAnthropicError(
                        message,
                        retry_after=_retry_after(response.headers.get("retry-after")),
                    )
                raise RuntimeError(message)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                event = json.loads(raw)
                if not isinstance(event, Mapping):
                    raise ValueError("Anthropic stream events must be JSON objects")
                event_type = event.get("type")
                if event_type == "error":
                    message = (
                        "Anthropic stream failed: "
                        + _error_message(event, "unknown provider error")
                    )
                    error = event.get("error")
                    error_type = error.get("type") if isinstance(error, Mapping) else None
                    if error_type in (
                        "api_error",
                        "overloaded_error",
                        "rate_limit_error",
                    ):
                        raise _TransientAnthropicError(message)
                    raise RuntimeError(message)
                if event_type == "message_start":
                    message = event.get("message")
                    if not isinstance(message, Mapping):
                        raise ValueError("message_start requires a message object")
                    model = message.get("model")
                    if isinstance(model, str) and model:
                        actual_model = model
                    initial_usage = message.get("usage")
                    if isinstance(initial_usage, Mapping):
                        usage.update(initial_usage)
                    started = True
                    yield ConversationEvent("message_start", {"model": actual_model})
                    if usage:
                        yield ConversationEvent("usage", _usage(actual_model, usage))
                    continue
                if event_type == "content_block_start":
                    index = event.get("index")
                    block = event.get("content_block")
                    if not isinstance(index, int) or not isinstance(block, Mapping):
                        raise ValueError("invalid content_block_start event")
                    native_type = block.get("type")
                    if native_type == "text":
                        normalized = {"type": "text"}
                    elif native_type == "thinking":
                        normalized = {"type": "thinking"}
                    elif native_type == "tool_use":
                        normalized = {
                            "type": "tool_call",
                            "id": block.get("id"),
                            "name": block.get("name"),
                        }
                    elif native_type == "redacted_thinking":
                        normalized = thaw_value(block)
                    else:
                        raise UnsupportedConversationFeature(
                            f"unsupported Anthropic output block {native_type!r}"
                        )
                    open_blocks.add(index)
                    yield ConversationEvent(
                        "content_start", {"index": index, "block": normalized}
                    )
                    continue
                if event_type == "content_block_delta":
                    index = event.get("index")
                    delta = event.get("delta")
                    if not isinstance(index, int) or not isinstance(delta, Mapping):
                        raise ValueError("invalid content_block_delta event")
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        normalized = {"type": "text", "text": delta.get("text", "")}
                    elif delta_type == "thinking_delta":
                        normalized = {
                            "type": "thinking",
                            "text": delta.get("thinking", ""),
                        }
                    elif delta_type == "signature_delta":
                        normalized = {
                            "type": "signature",
                            "signature": delta.get("signature", ""),
                        }
                    elif delta_type == "input_json_delta":
                        normalized = {
                            "type": "tool_arguments_json",
                            "json": delta.get("partial_json", ""),
                        }
                    else:
                        continue
                    yield ConversationEvent(
                        "content_delta", {"index": index, "delta": normalized}
                    )
                    continue
                if event_type == "content_block_stop":
                    index = event.get("index")
                    if isinstance(index, int):
                        open_blocks.discard(index)
                        yield ConversationEvent("content_stop", {"index": index})
                    continue
                if event_type == "message_delta":
                    delta = event.get("delta")
                    final_usage = event.get("usage")
                    if isinstance(final_usage, Mapping):
                        usage.update(final_usage)
                        yield ConversationEvent("usage", _usage(actual_model, usage))
                    yield ConversationEvent("message_delta", {
                        "stop_reason": (
                            delta.get("stop_reason")
                            if isinstance(delta, Mapping)
                            else None
                        )
                    })
                    continue
                if event_type == "message_stop":
                    terminal = True
                    yield ConversationEvent("message_stop")
        if not started:
            raise RuntimeError("Anthropic stream ended before message_start")
        if open_blocks or not terminal:
            raise RuntimeError("Anthropic stream ended without terminal evidence")

    async def count_tokens(self, request: ProviderCall) -> InputTokenCount | None:
        if not isinstance(request, ProviderCall):
            raise TypeError("request must be a ProviderCall")
        payload = self._payload(request, stream=False)
        payload.pop("stream", None)
        payload.pop("max_tokens", None)
        response = await self._client.post("/v1/messages/count_tokens", json=payload)
        if response.is_error:
            return None
        body = response.json()
        value = body.get("input_tokens") if isinstance(body, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return InputTokenCount(value, "exact")
        return None

    async def aclose(self) -> None:
        await self._client.aclose()
