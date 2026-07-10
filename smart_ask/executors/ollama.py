"""Native local Ollama executors for text tasks and structured conversations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
import json
from numbers import Integral
from types import MappingProxyType
from typing import Any
from urllib.request import Request, urlopen

import httpx

from ..conversation import (
    ConversationEvent,
    ConversationExecutionRequest,
    ConversationMessage,
    ConversationRequest,
    thaw_value,
)
from ..domain import ExecutionRequest, ModelResult


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


def _ollama_messages(request: ConversationRequest) -> list[dict[str, Any]]:
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


class OllamaExecutor:
    """Execute existing text-task requests through Ollama's native chat API."""

    captures_output = True

    def __init__(
        self,
        *,
        base_url: str,
        default_max_tokens: int,
        temperature: float,
        system_prompts: Mapping[str, str] | None = None,
        max_tokens: Mapping[str, int] | None = None,
        temperatures: Mapping[str, float] | None = None,
        think: bool = False,
        timeout_seconds: float = 300.0,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be non-empty text")
        self._url = base_url.rstrip("/") + "/chat"
        self._default_max_tokens = int(default_max_tokens)
        self._temperature = float(temperature)
        self._system_prompts = MappingProxyType(dict(system_prompts or {}))
        self._max_tokens = MappingProxyType(dict(max_tokens or {}))
        self._temperatures = MappingProxyType(dict(temperatures or {}))
        self._think = bool(think)
        self._timeout_seconds = float(timeout_seconds)

    def execute(self, request: ExecutionRequest) -> ModelResult:
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")
        max_tokens = (
            request.max_tokens
            or self._max_tokens.get(request.model)
            or self._default_max_tokens
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else self._temperatures.get(request.model, self._temperature)
        )
        messages = []
        system_prompt = self._system_prompts.get(request.model)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload = json.dumps({
            "model": request.model,
            "messages": messages,
            "stream": False,
            "think": self._think,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode("utf-8")
        http_request = Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(http_request, timeout=self._timeout_seconds) as response:
            result = json.load(response)
        message = result.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Ollama response is missing message")
        text = message.get("content", "")
        if not isinstance(text, str):
            raise ValueError("Ollama message content must be text")
        prompt_tokens = _non_negative_integer(
            result.get("prompt_eval_count"),
            "prompt_eval_count",
        )
        completion_tokens = _non_negative_integer(
            result.get("eval_count"),
            "eval_count",
        )
        usage = None
        if prompt_tokens is not None and completion_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        native_finish = result.get("done_reason")
        finish_reason = {
            "stop": "stop",
            "length": "length",
        }.get(native_finish, "unknown")
        return ModelResult(
            model=result.get("model") if isinstance(result.get("model"), str) else None,
            text=text,
            raw_text=text,
            usage=usage,
            finish_reason=finish_reason,
            native_finish_reason=(
                native_finish if isinstance(native_finish, str) and native_finish else None
            ),
            applied_max_tokens=max_tokens,
            visible_output_tokens=completion_tokens if text.strip() else 0,
            reasoning_tokens=0 if not self._think else None,
        )


class OllamaConversationExecutor:
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

    def _payload(self, request: ConversationExecutionRequest) -> dict[str, Any]:
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
        request: ConversationExecutionRequest,
    ) -> AsyncIterator[ConversationEvent]:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
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
        request: ConversationExecutionRequest,
    ) -> int | None:
        if not isinstance(request, ConversationExecutionRequest):
            raise TypeError("request must be a ConversationExecutionRequest")
        payload = self._payload(request)
        serialized = json.dumps(
            payload["messages"],
            sort_keys=True,
            separators=(",", ":"),
        )
        return max(1, (len(serialized) + 3) // 4)
