"""Native local Ollama structured conversation transport."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import json
from numbers import Integral
from typing import Any

import httpx

from ..conversation.domain import (
    ConversationEvent,
    thaw_value,
)
from ..conversation.model import Conversation, InputTokenCount
from ._protocol import ProviderCall


class UnsupportedConversationFeature(ValueError):
    """Raised when a backend cannot faithfully encode a conversation feature."""


def _non_negative_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return int(value)


def _text_from_blocks(blocks: tuple[Mapping[str, Any], ...]) -> str:
    return "\n".join(
        block["text"]
        for block in blocks
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _tool_result_text(block: Mapping[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, tuple):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "\n".join(texts)
    return json.dumps(thaw_value(content), separators=(",", ":"))


def _ollama_think(value: Any, default: bool) -> bool | str:
    """Translate a neutral thinking request into Ollama's accepted shape."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        kind = value.get("type")
        if kind == "disabled":
            return False
        if kind in ("enabled", "adaptive"):
            return True
    raise UnsupportedConversationFeature(
        "Ollama thinking must be boolean, a level, or an enabled/disabled request"
    )


def _ollama_messages(request: Conversation) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system:
        unsupported = [
            block.get("type") for block in request.system if block.get("type") != "text"
        ]
        if unsupported:
            raise UnsupportedConversationFeature(
                f"Ollama cannot encode system block types: {unsupported}"
            )
        messages.append({
            "role": "system",
            "content": _text_from_blocks(request.system),
        })

    for message in request.messages:
        tool_results = [
            block for block in message.content if block.get("type") == "tool_result"
        ]
        ordinary = [
            block for block in message.content if block.get("type") != "tool_result"
        ]
        if ordinary:
            encoded: dict[str, Any] = {
                "role": message.role,
                "content": _text_from_blocks(tuple(ordinary)),
            }
            images = [
                block.get("data")
                for block in ordinary
                if block.get("type") == "image"
                and isinstance(block.get("data"), str)
            ]
            if images:
                encoded["images"] = images
            thinking = "\n".join(
                block.get("thinking", "")
                for block in ordinary
                if block.get("type") == "thinking"
                and isinstance(block.get("thinking"), str)
            )
            if thinking:
                encoded["thinking"] = thinking
            tool_calls = []
            for block in ordinary:
                if block.get("type") != "tool_call":
                    continue
                name = block.get("name")
                arguments = block.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise UnsupportedConversationFeature(
                        "tool_call blocks require a non-empty name"
                    )
                tool_calls.append({
                    "function": {
                        "name": name,
                        "arguments": thaw_value(arguments),
                    }
                })
            if tool_calls:
                encoded["tool_calls"] = tool_calls
            supported_types = {
                "text",
                "image",
                "thinking",
                "tool_call",
            }
            unsupported = [
                block.get("type")
                for block in ordinary
                if block.get("type") not in supported_types
            ]
            if unsupported:
                raise UnsupportedConversationFeature(
                    f"Ollama cannot encode content block types: {unsupported}"
                )
            messages.append(encoded)
        for block in tool_results:
            messages.append({
                "role": "tool",
                "content": _tool_result_text(block),
            })
    return messages


def _ollama_tools(tools: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    encoded = []
    for tool in tools:
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
            raise UnsupportedConversationFeature(
                "tools require a name and input_schema mapping"
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


class OllamaTransport:
    """Stream structured conversations through Ollama's native chat API."""

    def __init__(
        self,
        *,
        base_url: str,
        default_max_tokens: int,
        temperature: float,
        think: bool = False,
        timeout_seconds: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be non-empty text")
        self._url = base_url.rstrip("/") + "/chat"
        self._default_max_tokens = int(default_max_tokens)
        self._temperature = float(temperature)
        self._think = bool(think)
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = client

    def _payload(self, request: ProviderCall) -> dict[str, Any]:
        conversation = request.conversation
        parameters = conversation.parameters
        max_tokens = parameters.get("max_tokens", self._default_max_tokens)
        temperature = parameters.get("temperature", self._temperature)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _ollama_messages(conversation),
            "stream": True,
            "think": _ollama_think(parameters.get("thinking"), self._think),
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        stop = parameters.get("stop")
        if stop is not None:
            payload["options"]["stop"] = thaw_value(stop)
        tools = _ollama_tools(conversation.tools)
        if tools:
            payload["tools"] = tools
        return payload

    async def stream(
        self,
        request: ProviderCall,
    ) -> AsyncIterator[ConversationEvent]:
        if not isinstance(request, ProviderCall):
            raise TypeError("request must be a ProviderCall")
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        message_started = False
        open_block: tuple[str, int] | None = None
        next_index = 0
        try:
            async with client.stream(
                "POST",
                self._url,
                json=self._payload(request),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError("Ollama stream chunks must be objects")
                    if not message_started:
                        message_started = True
                        yield ConversationEvent("message_start", {
                            "model": value.get("model") or request.model,
                        })
                    message = value.get("message")
                    if not isinstance(message, Mapping):
                        message = {}
                    for block_kind, field_name in (
                        ("thinking", "thinking"),
                        ("text", "content"),
                    ):
                        delta = message.get(field_name)
                        if not isinstance(delta, str) or not delta:
                            continue
                        if open_block is None or open_block[0] != block_kind:
                            if open_block is not None:
                                yield ConversationEvent(
                                    "content_stop",
                                    {"index": open_block[1]},
                                )
                            open_block = (block_kind, next_index)
                            next_index += 1
                            yield ConversationEvent("content_start", {
                                "index": open_block[1],
                                "block": {"type": block_kind},
                            })
                        yield ConversationEvent("content_delta", {
                            "index": open_block[1],
                            "delta": {"type": block_kind, "text": delta},
                        })
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list):
                        if open_block is not None:
                            yield ConversationEvent(
                                "content_stop",
                                {"index": open_block[1]},
                            )
                            open_block = None
                        for tool_call in tool_calls:
                            if not isinstance(tool_call, Mapping):
                                continue
                            function = tool_call.get("function")
                            if not isinstance(function, Mapping):
                                continue
                            index = next_index
                            next_index += 1
                            yield ConversationEvent("content_start", {
                                "index": index,
                                "block": {
                                    "type": "tool_call",
                                    "id": tool_call.get("id") or f"call_{index}",
                                    "name": function.get("name"),
                                },
                            })
                            yield ConversationEvent("content_delta", {
                                "index": index,
                                "delta": {
                                    "type": "tool_arguments",
                                    "arguments": function.get("arguments", {}),
                                },
                            })
                            yield ConversationEvent("content_stop", {"index": index})
                    if value.get("done") is True:
                        if open_block is not None:
                            yield ConversationEvent(
                                "content_stop",
                                {"index": open_block[1]},
                            )
                            open_block = None
                        prompt_tokens = _non_negative_integer(
                            value.get("prompt_eval_count"),
                            "prompt_eval_count",
                        )
                        output_tokens = _non_negative_integer(
                            value.get("eval_count"),
                            "eval_count",
                        )
                        usage = {
                            "input_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": (
                                prompt_tokens + output_tokens
                                if prompt_tokens is not None
                                and output_tokens is not None
                                else None
                            ),
                        }
                        yield ConversationEvent("usage", usage)
                        yield ConversationEvent("message_delta", {
                            "stop_reason": value.get("done_reason") or "stop",
                        })
                        yield ConversationEvent("message_stop")
                        return
                raise ValueError("Ollama stream ended before a completion event")
        finally:
            if owns_client:
                await client.aclose()

    async def count_tokens(
        self,
        request: ProviderCall,
    ) -> InputTokenCount | None:
        if not isinstance(request, ProviderCall):
            raise TypeError("request must be a ProviderCall")
        payload = self._payload(request)
        serialized = json.dumps(
            payload["messages"],
            sort_keys=True,
            separators=(",", ":"),
        )
        return InputTokenCount(
            max(1, (len(serialized) + 3) // 4),
            "estimate",
        )
