"""Provider-neutral Responses API conversation executor mechanics."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import json
from numbers import Integral
from typing import Any

import httpx

from ..conversation import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationRequest,
    thaw_value,
)
from .ollama import UnsupportedConversationFeature


_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


def _text(blocks: tuple[Mapping[str, Any], ...]) -> str:
    return "\n".join(
        block["text"]
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _tool_output(block: Mapping[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(thaw_value(content), separators=(",", ":"))


def _message_content(
    blocks: tuple[Mapping[str, Any], ...],
) -> str | list[dict[str, Any]]:
    images = [block for block in blocks if block.get("type") == "image"]
    if not images:
        return _text(blocks)
    content: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            content.append({"type": "input_text", "text": block["text"]})
        elif block_type == "image":
            data = block.get("data")
            media_type = block.get("media_type")
            if not isinstance(data, str) or not isinstance(media_type, str):
                raise UnsupportedConversationFeature(
                    "images require base64 data and a media type"
                )
            content.append({
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{media_type};base64,{data}",
            })
    return content


def _input(request: ConversationRequest) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role not in ("user", "assistant", "system", "developer"):
            raise UnsupportedConversationFeature(
                f"unsupported Responses API message role: {message.role!r}"
            )
        ordinary = tuple(
            block
            for block in message.content
            if block.get("type") in ("text", "image")
        )
        unsupported = [
            block.get("type")
            for block in message.content
            if block.get("type")
            not in ("text", "image", "thinking", "tool_call", "tool_result")
        ]
        if unsupported:
            raise UnsupportedConversationFeature(
                f"Responses API cannot encode block types: {unsupported}"
            )
        if ordinary:
            values.append({
                "type": "message",
                "role": message.role,
                "content": _message_content(ordinary),
            })
        for block in message.content:
            block_type = block.get("type")
            if block_type == "tool_call":
                call_id = block.get("id")
                name = block.get("name")
                if not isinstance(call_id, str) or not call_id:
                    raise UnsupportedConversationFeature(
                        "tool calls require a non-empty id"
                    )
                if not isinstance(name, str) or not name:
                    raise UnsupportedConversationFeature(
                        "tool calls require a non-empty name"
                    )
                values.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(
                        thaw_value(block.get("arguments", {})),
                        separators=(",", ":"),
                    ),
                })
            elif block_type == "tool_result":
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise UnsupportedConversationFeature(
                        "tool results require a non-empty call id"
                    )
                values.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _tool_output(block),
                })
    if not values:
        raise UnsupportedConversationFeature(
            "Responses API requires at least one encodable input item"
        )
    return values


def _instructions(request: ConversationRequest) -> str | None:
    if not request.system:
        return None
    unsupported = [
        block.get("type") for block in request.system if block.get("type") != "text"
    ]
    if unsupported:
        raise UnsupportedConversationFeature(
            f"system block types are unsupported: {unsupported}"
        )
    return _text(request.system)


def _tools(tools: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    encoded = []
    for tool in tools:
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
            raise UnsupportedConversationFeature(
                "tools require a non-empty name and input_schema mapping"
            )
        value: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": thaw_value(schema),
            "strict": False,
        }
        description = tool.get("description")
        if isinstance(description, str):
            value["description"] = description
        encoded.append(value)
    return encoded


def _tool_choice(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return thaw_value(value)
    kind = value.get("type")
    if kind in ("auto", "none"):
        return kind
    if kind == "any":
        return "required"
    if kind == "tool":
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise UnsupportedConversationFeature(
                "a forced tool choice requires a non-empty name"
            )
        return {"type": "function", "name": name}
    return thaw_value(value)


def _usage(value: Mapping[str, Any]) -> dict[str, Any]:
    input_details = value.get("input_tokens_details")
    output_details = value.get("output_tokens_details")
    result = {
        "input_tokens": value.get("input_tokens"),
        "output_tokens": value.get("output_tokens"),
        "total_tokens": value.get("total_tokens"),
        "provider_cost_usd": None,
    }
    if isinstance(input_details, Mapping):
        result["cache_read_tokens"] = input_details.get("cached_tokens")
        result["cache_write_tokens"] = input_details.get("cache_write_tokens")
    if isinstance(output_details, Mapping):
        result["reasoning_tokens"] = output_details.get("reasoning_tokens")
    return result


def _response_error(value: Mapping[str, Any], fallback: str) -> str:
    error = value.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return fallback


class ResponsesConversationExecutor:
    """Stream conversations through a Responses-compatible API."""

    _include_store = False

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        default_max_tokens: int,
        reasoning_effort: str,
    ):
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        if (
            isinstance(default_max_tokens, bool)
            or not isinstance(default_max_tokens, Integral)
            or default_max_tokens < 1
        ):
            raise ValueError("default_max_tokens must be a positive integer")
        if reasoning_effort not in _EFFORTS:
            raise ValueError("reasoning_effort is invalid")
        self._client = client
        self._default_max_tokens = int(default_max_tokens)
        self._reasoning_effort = reasoning_effort

    def _payload(self, request: ConversationExecutionRequest) -> dict[str, Any]:
        conversation = request.conversation
        parameters = conversation.parameters
        effort = parameters.get("reasoning_effort", self._reasoning_effort)
        if effort not in _EFFORTS:
            raise UnsupportedConversationFeature(
                "reasoning_effort must be a supported effort name"
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "input": _input(conversation),
            "max_output_tokens": parameters.get(
                "max_tokens",
                self._default_max_tokens,
            ),
            "reasoning": {"effort": effort},
            "stream": True,
        }
        if self._include_store:
            payload["store"] = False
        instructions = _instructions(conversation)
        if instructions:
            payload["instructions"] = instructions
        tools = _tools(conversation.tools)
        if tools:
            payload["tools"] = tools
        if "tool_choice" in parameters:
            raw_choice = parameters["tool_choice"]
            payload["tool_choice"] = _tool_choice(raw_choice)
            if (
                isinstance(raw_choice, Mapping)
                and raw_choice.get("disable_parallel_tool_use") is True
            ):
                payload["parallel_tool_calls"] = False
        if "stop" in parameters and parameters["stop"] not in (None, (), ""):
            raise UnsupportedConversationFeature(
                "Responses API does not support stop sequences"
            )
        return payload

    async def stream(
        self,
        request: ConversationExecutionRequest,
    ) -> AsyncIterator[ConversationEvent]:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
        started = False
        actual_model = request.model
        content_blocks: dict[tuple[str, int], int] = {}
        tool_blocks: dict[str, int] = {}
        next_index = 0
        finish_reason: str | None = None
        refusal_seen = False

        async with self._client.stream(
            "POST",
            "/responses",
            json=self._payload(request),
        ) as response:
            if response.is_error:
                raw = await response.aread()
                try:
                    body = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = {}
                message = _response_error(
                    body if isinstance(body, Mapping) else {},
                    response.reason_phrase,
                )
                raise RuntimeError(
                    f"Responses API returned {response.status_code}: {message}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                event = json.loads(raw)
                if not isinstance(event, Mapping):
                    raise ValueError("Responses API events must be JSON objects")
                event_type = event.get("type")
                event_response = event.get("response")
                if isinstance(event_response, Mapping):
                    model = event_response.get("model")
                    if isinstance(model, str) and model:
                        actual_model = model
                if not started:
                    started = True
                    yield ConversationEvent("message_start", {
                        "model": actual_model,
                    })

                if event_type == "response.output_item.added":
                    item = event.get("item")
                    if (
                        not isinstance(item, Mapping)
                        or item.get("type") != "function_call"
                    ):
                        continue
                    item_id = item.get("id")
                    call_id = item.get("call_id")
                    name = item.get("name")
                    if not isinstance(item_id, str) or not item_id:
                        raise ValueError("function calls require an item id")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("function calls require a call id")
                    if not isinstance(name, str) or not name:
                        raise ValueError("function calls require a name")
                    tool_blocks[item_id] = next_index
                    next_index += 1
                    yield ConversationEvent("content_start", {
                        "index": tool_blocks[item_id],
                        "block": {
                            "type": "tool_call",
                            "id": call_id,
                            "name": name,
                        },
                    })
                    continue

                if event_type == "response.function_call_arguments.delta":
                    item_id = event.get("item_id")
                    delta = event.get("delta")
                    if (
                        isinstance(item_id, str)
                        and item_id in tool_blocks
                        and isinstance(delta, str)
                        and delta
                    ):
                        yield ConversationEvent("content_delta", {
                            "index": tool_blocks[item_id],
                            "delta": {
                                "type": "tool_arguments_json",
                                "json": delta,
                            },
                        })
                    continue

                delta_kind = None
                if event_type == "response.output_text.delta":
                    delta_kind = "text"
                elif event_type in (
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                ):
                    delta_kind = "thinking"
                elif event_type == "response.refusal.delta":
                    delta_kind = "text"
                    refusal_seen = True
                if delta_kind is not None:
                    delta = event.get("delta")
                    output_index = event.get("output_index", 0)
                    if not isinstance(output_index, int):
                        output_index = 0
                    if isinstance(delta, str) and delta:
                        key = (delta_kind, output_index)
                        if key not in content_blocks:
                            content_blocks[key] = next_index
                            next_index += 1
                            yield ConversationEvent("content_start", {
                                "index": content_blocks[key],
                                "block": {"type": delta_kind},
                            })
                        yield ConversationEvent("content_delta", {
                            "index": content_blocks[key],
                            "delta": {"type": delta_kind, "text": delta},
                        })
                    continue

                if event_type in (
                    "response.completed",
                    "response.incomplete",
                ) and isinstance(event_response, Mapping):
                    usage = event_response.get("usage")
                    if isinstance(usage, Mapping):
                        yield ConversationEvent("usage", _usage(usage))
                    if refusal_seen:
                        finish_reason = "refusal"
                    elif tool_blocks:
                        finish_reason = "tool_call"
                    elif event_type == "response.incomplete":
                        details = event_response.get("incomplete_details")
                        reason = (
                            details.get("reason")
                            if isinstance(details, Mapping)
                            else None
                        )
                        finish_reason = (
                            "length" if reason == "max_output_tokens" else "unknown"
                        )
                    else:
                        finish_reason = "stop"
                    continue

                if event_type == "response.failed":
                    details = (
                        event_response if isinstance(event_response, Mapping) else {}
                    )
                    raise RuntimeError(
                        _response_error(details, "Responses API request failed")
                    )
                if event_type == "error":
                    raise RuntimeError(
                        _response_error(event, "Responses API stream failed")
                    )

        if not started:
            raise ValueError("Responses API stream ended without response events")
        for index in sorted((*content_blocks.values(), *tool_blocks.values())):
            yield ConversationEvent("content_stop", {"index": index})
        yield ConversationEvent("message_delta", {
            "stop_reason": finish_reason or "unknown",
        })
        yield ConversationEvent("message_stop")

    async def count_tokens(
        self,
        request: ConversationExecutionRequest,
    ) -> int | None:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
        serialized = json.dumps(
            _input(request.conversation),
            sort_keys=True,
            separators=(",", ":"),
        )
        return max(1, (len(serialized) + 3) // 4)

    async def aclose(self) -> None:
        await self._client.aclose()
