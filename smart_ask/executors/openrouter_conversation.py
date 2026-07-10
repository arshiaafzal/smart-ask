"""Structured conversation execution through an OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import json
from typing import Any

import httpx

from ..conversation import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationMessage,
    ConversationRequest,
    thaw_value,
)
from .ollama import UnsupportedConversationFeature


def _text(blocks: tuple[Mapping[str, Any], ...]) -> str:
    return "\n".join(
        block["text"]
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _tool_result_content(block: Mapping[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(thaw_value(content), separators=(",", ":"))


def _content_parts(blocks: tuple[Mapping[str, Any], ...]) -> str | list[dict[str, Any]]:
    images = [block for block in blocks if block.get("type") == "image"]
    if not images:
        return _text(blocks)
    parts: list[dict[str, Any]] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif block_type == "image":
            data = block.get("data")
            media_type = block.get("media_type")
            if not isinstance(data, str) or not isinstance(media_type, str):
                raise UnsupportedConversationFeature(
                    "images require base64 data and a media type"
                )
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
    return parts


def _ordinary_message(
    message: ConversationMessage,
    blocks: tuple[Mapping[str, Any], ...],
) -> dict[str, Any] | None:
    supported = {"text", "image", "thinking", "tool_call"}
    unsupported = [
        block.get("type") for block in blocks if block.get("type") not in supported
    ]
    if unsupported:
        raise UnsupportedConversationFeature(
            f"OpenAI-compatible execution cannot encode block types: {unsupported}"
        )
    if not blocks:
        return None
    encoded: dict[str, Any] = {
        "role": message.role,
        "content": _content_parts(blocks),
    }
    thinking = "\n".join(
        block.get("thinking", "")
        for block in blocks
        if block.get("type") == "thinking"
        and isinstance(block.get("thinking"), str)
    )
    if thinking:
        encoded["reasoning"] = thinking
    tool_calls = []
    for block in blocks:
        if block.get("type") != "tool_call":
            continue
        name = block.get("name")
        call_id = block.get("id")
        if not isinstance(name, str) or not name or not isinstance(call_id, str):
            raise UnsupportedConversationFeature(
                "tool calls require non-empty id and name values"
            )
        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    thaw_value(block.get("arguments", {})),
                    separators=(",", ":"),
                ),
            },
        })
    if tool_calls:
        encoded["tool_calls"] = tool_calls
    return encoded


def _messages(request: ConversationRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system:
        unsupported = [
            block.get("type") for block in request.system if block.get("type") != "text"
        ]
        if unsupported:
            raise UnsupportedConversationFeature(
                f"system block types are unsupported: {unsupported}"
            )
        messages.append({"role": "system", "content": _text(request.system)})
    for message in request.messages:
        ordinary = tuple(
            block for block in message.content if block.get("type") != "tool_result"
        )
        encoded = _ordinary_message(message, ordinary)
        if encoded is not None:
            messages.append(encoded)
        for block in message.content:
            if block.get("type") != "tool_result":
                continue
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise UnsupportedConversationFeature(
                    "tool results require a non-empty call id"
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _tool_result_content(block),
            })
    return messages


def _tools(tools: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    encoded = []
    for tool in tools:
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
            raise UnsupportedConversationFeature(
                "tools require a non-empty name and input_schema mapping"
            )
        function: dict[str, Any] = {
            "name": name,
            "parameters": thaw_value(schema),
        }
        description = tool.get("description")
        if isinstance(description, str):
            function["description"] = description
        encoded.append({"type": "function", "function": function})
    return encoded


def _reasoning(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return {"enabled": value}
    if isinstance(value, Mapping):
        kind = value.get("type")
        if kind == "disabled":
            return {"enabled": False}
        if kind in ("enabled", "adaptive"):
            result: dict[str, Any] = {"enabled": True}
            budget = value.get("budget_tokens")
            if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
                result["max_tokens"] = budget
            return result
    raise UnsupportedConversationFeature("unsupported structured thinking request")


def _usage(value: Mapping[str, Any]) -> dict[str, Any]:
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    details = value.get("completion_tokens_details")
    prompt_details = value.get("prompt_tokens_details")
    result = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": value.get("total_tokens"),
        "provider_cost_usd": value.get("cost"),
    }
    if isinstance(details, Mapping):
        result["reasoning_tokens"] = details.get("reasoning_tokens")
    if isinstance(prompt_details, Mapping):
        result["cache_read_tokens"] = prompt_details.get("cached_tokens")
    return result


class OpenRouterConversationExecutor:
    """Stream normalized conversations through an OpenAI-compatible API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        default_max_tokens: int,
        temperature: float,
    ):
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("client must be an httpx.AsyncClient")
        self._client = client
        self._default_max_tokens = int(default_max_tokens)
        self._temperature = float(temperature)

    def _payload(self, request: ConversationExecutionRequest) -> dict[str, Any]:
        conversation = request.conversation
        parameters = conversation.parameters
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _messages(conversation),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": parameters.get("max_tokens", self._default_max_tokens),
            "temperature": parameters.get("temperature", self._temperature),
        }
        tools = _tools(conversation.tools)
        if tools:
            payload["tools"] = tools
        for source, target in (("stop", "stop"), ("tool_choice", "tool_choice")):
            if source in parameters:
                payload[target] = thaw_value(parameters[source])
        reasoning = _reasoning(parameters.get("thinking"))
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return payload

    async def stream(
        self,
        request: ConversationExecutionRequest,
    ) -> AsyncIterator[ConversationEvent]:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
        started = False
        content_blocks: dict[str, int] = {}
        tool_blocks: dict[int, int] = {}
        next_index = 0
        finish_reason: str | None = None
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=self._payload(request),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                chunk = json.loads(raw)
                if not isinstance(chunk, Mapping):
                    raise ValueError("stream chunks must be JSON objects")
                if not started:
                    started = True
                    yield ConversationEvent("message_start", {
                        "model": chunk.get("model") or request.model,
                    })
                usage = chunk.get("usage")
                if isinstance(usage, Mapping):
                    yield ConversationEvent("usage", _usage(usage))
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    continue
                native_finish = choice.get("finish_reason")
                if isinstance(native_finish, str) and native_finish:
                    finish_reason = native_finish
                delta = choice.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                for block_type, field in (("thinking", "reasoning"), ("text", "content")):
                    value = delta.get(field)
                    if not isinstance(value, str) or not value:
                        continue
                    if block_type not in content_blocks:
                        content_blocks[block_type] = next_index
                        next_index += 1
                        yield ConversationEvent("content_start", {
                            "index": content_blocks[block_type],
                            "block": {"type": block_type},
                        })
                    yield ConversationEvent("content_delta", {
                        "index": content_blocks[block_type],
                        "delta": {"type": block_type, "text": value},
                    })
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, Mapping):
                            continue
                        provider_index = call.get("index")
                        if not isinstance(provider_index, int):
                            continue
                        function = call.get("function")
                        if not isinstance(function, Mapping):
                            function = {}
                        if provider_index not in tool_blocks:
                            tool_blocks[provider_index] = next_index
                            next_index += 1
                            yield ConversationEvent("content_start", {
                                "index": tool_blocks[provider_index],
                                "block": {
                                    "type": "tool_call",
                                    "id": call.get("id") or f"call_{provider_index}",
                                    "name": function.get("name"),
                                },
                            })
                        arguments = function.get("arguments")
                        if isinstance(arguments, str) and arguments:
                            yield ConversationEvent("content_delta", {
                                "index": tool_blocks[provider_index],
                                "delta": {
                                    "type": "tool_arguments_json",
                                    "json": arguments,
                                },
                            })
        if not started:
            raise ValueError("stream ended without any response chunks")
        for index in sorted((*content_blocks.values(), *tool_blocks.values())):
            yield ConversationEvent("content_stop", {"index": index})
        yield ConversationEvent("message_delta", {
            "stop_reason": finish_reason or "stop",
        })
        yield ConversationEvent("message_stop")

    async def count_tokens(
        self,
        request: ConversationExecutionRequest,
    ) -> int | None:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
        serialized = json.dumps(
            _messages(request.conversation),
            sort_keys=True,
            separators=(",", ":"),
        )
        return max(1, (len(serialized) + 3) // 4)

    async def aclose(self) -> None:
        await self._client.aclose()
