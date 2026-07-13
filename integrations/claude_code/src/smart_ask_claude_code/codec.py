"""Anthropic wire translation at the external adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
from uuid import uuid4

from smart_ask.conversation import (
    ConversationEvent,
    ConversationMessage,
    SessionContext,
    thaw_value,
)
from smart_ask.conversation.model import Conversation


_ROOT_FIELDS = frozenset({
    "model",
    "system",
    "messages",
    "tools",
    "max_tokens",
    "temperature",
    "stop_sequences",
    "thinking",
    "tool_choice",
    "stream",
})


def _decode_block(block: Mapping[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    if block_type == "text":
        return dict(block)
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, Mapping):
            return {
                "type": "image",
                "data": source.get("data"),
                "media_type": source.get("media_type"),
                "source_type": source.get("type"),
                "extensions": {
                    key: value
                    for key, value in block.items()
                    if key not in ("type", "source")
                },
            }
    if block_type == "thinking":
        return {
            "type": "thinking",
            "thinking": block.get("thinking", ""),
            "signature": block.get("signature"),
            "extensions": {
                key: value
                for key, value in block.items()
                if key not in ("type", "thinking", "signature")
            },
        }
    if block_type == "tool_use":
        return {
            "type": "tool_call",
            "id": block.get("id"),
            "name": block.get("name"),
            "arguments": block.get("input", {}),
            "extensions": {
                key: value
                for key, value in block.items()
                if key not in ("type", "id", "name", "input")
            },
        }
    if block_type == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = [_decode_block(item) for item in content]
        return {
            "type": "tool_result",
            "id": block.get("tool_use_id"),
            "content": content,
            "is_error": block.get("is_error", False),
            "extensions": {
                key: value
                for key, value in block.items()
                if key not in ("type", "tool_use_id", "content", "is_error")
            },
        }
    return dict(block)


def _decode_content(content: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(content, str):
        return ({"type": "text", "text": content},)
    if not isinstance(content, list):
        raise ValueError("message content must be text or an array")
    if any(not isinstance(block, Mapping) for block in content):
        raise ValueError("message content blocks must be objects")
    return tuple(_decode_block(block) for block in content)


def decode_request(
    body: Mapping[str, Any],
    headers: Mapping[str, str],
) -> tuple[Conversation, SessionContext]:
    """Decode without retaining the adapter model alias in SmartAsk's domain."""

    raw_system = body.get("system", [])
    if isinstance(raw_system, str):
        system = ({"type": "text", "text": raw_system},)
    elif isinstance(raw_system, list) and all(
        isinstance(block, Mapping) for block in raw_system
    ):
        system = tuple(_decode_block(block) for block in raw_system)
    else:
        raise ValueError("system must be text or an array of blocks")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")
    messages = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise ValueError("messages must contain objects")
        role = raw.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("every message requires a role")
        messages.append(ConversationMessage(
            role=role,
            content=_decode_content(raw.get("content")),
            extensions={
                key: value
                for key, value in raw.items()
                if key not in ("role", "content")
            },
        ))
    raw_tools = body.get("tools", [])
    if not isinstance(raw_tools, list) or any(
        not isinstance(tool, Mapping) for tool in raw_tools
    ):
        raise ValueError("tools must be an array of objects")
    tools = tuple({
        "name": tool.get("name"),
        "description": tool.get("description"),
        "input_schema": tool.get("input_schema", {}),
        "extensions": {
            key: value
            for key, value in tool.items()
            if key not in ("name", "description", "input_schema")
        },
    } for tool in raw_tools)
    parameters = {}
    for source, target in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("stop_sequences", "stop"),
        ("thinking", "thinking"),
        ("tool_choice", "tool_choice"),
    ):
        if source in body:
            parameters[target] = body[source]
    conversation = Conversation(
        system=system,
        messages=tuple(messages),
        tools=tools,
        parameters=parameters,
        extensions={
            key: value for key, value in body.items() if key not in _ROOT_FIELDS
        },
    )
    session = SessionContext(
        session_id=headers.get("x-claude-code-session-id"),
        agent_id=headers.get("x-claude-code-agent-id"),
        parent_agent_id=headers.get("x-claude-code-parent-agent-id"),
    )
    return conversation, session


def _stop_reason(reason: Any) -> str | None:
    return {
        "stop": "end_turn",
        "end_turn": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_call": "tool_use",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "refusal": "refusal",
    }.get(reason, reason if isinstance(reason, str) else None)


class AnthropicEventEncoder:
    """Stateful normalized-event to Anthropic SSE encoder."""

    def __init__(self, requested_model: str, *, input_tokens: int = 0):
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
        ):
            raise ValueError("input_tokens must be a non-negative integer")
        self.requested_model = requested_model
        self.message_id = "msg_" + uuid4().hex
        self.usage: dict[str, int] = {"input_tokens": input_tokens}

    @staticmethod
    def _sse(event: str, value: Mapping[str, Any]) -> bytes:
        data = json.dumps(value, separators=(",", ":"))
        return f"event: {event}\ndata: {data}\n\n".encode()

    def encode(self, event: ConversationEvent) -> bytes | None:
        data = thaw_value(event.data)
        if event.kind == "heartbeat":
            return self._sse("ping", {"type": "ping"})
        if event.kind == "usage":
            self.usage.update({
                key: value
                for key, value in data.items()
                if isinstance(value, int) and not isinstance(value, bool)
            })
            return None
        if event.kind == "message_start":
            return self._sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.requested_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": data.get(
                            "input_tokens",
                            self.usage["input_tokens"],
                        ),
                    },
                },
            })
        if event.kind == "content_start":
            block = data.get("block", {})
            block_type = block.get("type")
            if block_type == "text":
                content_block = {"type": "text", "text": ""}
            elif block_type == "thinking":
                content_block = {"type": "thinking", "thinking": "", "signature": ""}
            elif block_type == "tool_call":
                content_block = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
            else:
                content_block = block
            return self._sse("content_block_start", {
                "type": "content_block_start",
                "index": data.get("index"),
                "content_block": content_block,
            })
        if event.kind == "content_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")
            if delta_type == "text":
                encoded_delta = {"type": "text_delta", "text": delta.get("text", "")}
            elif delta_type == "thinking":
                encoded_delta = {
                    "type": "thinking_delta",
                    "thinking": delta.get("text", ""),
                }
            elif delta_type == "tool_arguments":
                encoded_delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        delta.get("arguments", {}),
                        separators=(",", ":"),
                    ),
                }
            elif delta_type == "tool_arguments_json":
                encoded_delta = {
                    "type": "input_json_delta",
                    "partial_json": delta.get("json", ""),
                }
            else:
                encoded_delta = delta
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": data.get("index"),
                "delta": encoded_delta,
            })
        if event.kind == "content_stop":
            return self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": data.get("index"),
            })
        if event.kind == "message_delta":
            return self._sse("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _stop_reason(data.get("stop_reason")),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": self.usage.get("output_tokens", 0),
                },
            })
        if event.kind == "message_stop":
            return self._sse("message_stop", {"type": "message_stop"})
        if event.kind == "error":
            return self.error(
                data.get("type", "api_error"),
                data.get("message", "SmartAsk runtime error"),
            )
        raise ValueError(f"cannot encode event kind {event.kind!r}")

    def error(self, error_type: str, message: str) -> bytes:
        return self._sse("error", {
            "type": "error",
            "error": {"type": error_type, "message": message},
        })


class AnthropicMessageAssembler:
    """Build a non-streaming Anthropic message from normalized events."""

    def __init__(self, requested_model: str):
        self.requested_model = requested_model
        self.message_id = "msg_" + uuid4().hex
        self.blocks: dict[int, dict[str, Any]] = {}
        self.tool_argument_json: dict[int, str] = {}
        self.usage: dict[str, int] = {}
        self.stop_reason: str | None = None

    def observe(self, event: ConversationEvent) -> None:
        data = thaw_value(event.data)
        if event.kind == "content_start":
            block = data.get("block", {})
            block_type = block.get("type")
            if block_type == "text":
                value = {"type": "text", "text": ""}
            elif block_type == "thinking":
                value = {"type": "thinking", "thinking": "", "signature": ""}
            elif block_type == "tool_call":
                value = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
            else:
                value = block
            self.blocks[int(data["index"])] = value
        elif event.kind == "content_delta":
            block = self.blocks[int(data["index"])]
            delta = data.get("delta", {})
            if delta.get("type") == "text":
                block["text"] += delta.get("text", "")
            elif delta.get("type") == "thinking":
                block["thinking"] += delta.get("text", "")
            elif delta.get("type") == "tool_arguments":
                block["input"] = delta.get("arguments", {})
            elif delta.get("type") == "tool_arguments_json":
                index = int(data["index"])
                self.tool_argument_json[index] = (
                    self.tool_argument_json.get(index, "") + delta.get("json", "")
                )
        elif event.kind == "usage":
            self.usage.update(data)
        elif event.kind == "message_delta":
            self.stop_reason = _stop_reason(data.get("stop_reason"))
        elif event.kind == "error":
            raise RuntimeError(data.get("message", "SmartAsk runtime error"))

    def message(self) -> dict[str, Any]:
        for index, value in self.tool_argument_json.items():
            self.blocks[index]["input"] = json.loads(value or "{}")
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.requested_model,
            "content": [self.blocks[index] for index in sorted(self.blocks)],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": self.usage.get("input_tokens", 0),
                "output_tokens": self.usage.get("output_tokens", 0),
            },
        }
