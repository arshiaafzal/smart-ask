"""Harness-neutral immutable conversation values and streaming events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Literal


EventKind = Literal[
    "message_start",
    "content_start",
    "content_delta",
    "content_stop",
    "message_delta",
    "message_stop",
    "usage",
    "heartbeat",
    "error",
]

EVENT_KINDS = frozenset({
    "message_start",
    "content_start",
    "content_delta",
    "content_stop",
    "message_delta",
    "message_stop",
    "usage",
    "heartbeat",
    "error",
})


def freeze_value(value: Any) -> Any:
    """Recursively copy JSON-like data into immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): freeze_value(item) for key, item in value.items()
        })
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"conversation values must be JSON-compatible, got {type(value).__name__}")


def thaw_value(value: Any) -> Any:
    """Return a mutable JSON-compatible copy of an immutable value."""

    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


def _mapping_tuple(
    values: Sequence[Mapping[str, Any]],
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of mappings")
    frozen = tuple(freeze_value(value) for value in values)
    if any(not isinstance(value, Mapping) for value in frozen):
        raise TypeError(f"{name} must contain mappings")
    return frozen


@dataclass(frozen=True)
class ConversationMessage:
    """One normalized message with open-schema structured content blocks."""

    role: str
    content: tuple[Mapping[str, Any], ...]
    extensions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("message role must be non-empty text")
        object.__setattr__(self, "content", _mapping_tuple(self.content, "content"))
        extensions = freeze_value(self.extensions)
        if not isinstance(extensions, Mapping):
            raise TypeError("message extensions must be a mapping")
        object.__setattr__(self, "extensions", extensions)


@dataclass(frozen=True)
class SessionContext:
    """Caller-provided correlation metadata with no harness-specific semantics."""

    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    extensions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for name in ("session_id", "agent_id", "parent_agent_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"{name} must be non-empty trimmed text or None")
        extensions = freeze_value(self.extensions)
        if not isinstance(extensions, Mapping):
            raise TypeError("session extensions must be a mapping")
        object.__setattr__(self, "extensions", extensions)


@dataclass(frozen=True)
class ConversationRequest:
    """Complete normalized conversation plus open inference extensions."""

    system: tuple[Mapping[str, Any], ...]
    messages: tuple[ConversationMessage, ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    extensions: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", _mapping_tuple(self.system, "system"))
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages or any(
            not isinstance(message, ConversationMessage) for message in self.messages
        ):
            raise ValueError("messages must contain ConversationMessage values")
        object.__setattr__(self, "tools", _mapping_tuple(self.tools, "tools"))
        for name in ("parameters", "extensions"):
            frozen = freeze_value(getattr(self, name))
            if not isinstance(frozen, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, frozen)

    def latest_human_instruction(self) -> tuple[str, str] | None:
        """Return routing text and a stable turn fingerprint."""

        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if message.role != "user":
                continue
            texts = [
                block.get("text")
                for block in message.content
                if block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block.get("text").strip()
            ]
            if not texts:
                continue
            text = "\n\n".join(texts)
            fingerprint_source = json.dumps(
                {"human_turn_index": index, "text": text},
                sort_keys=True,
                separators=(",", ":"),
            )
            return text, sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return None

    def with_latest_human_text(self, text: str, *, before: bool) -> "ConversationRequest":
        """Add policy text without flattening other blocks or messages."""

        if not isinstance(text, str) or not text:
            raise ValueError("control text must be non-empty")
        messages = list(self.messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role != "user":
                continue
            blocks = list(message.content)
            control = freeze_value({"type": "text", "text": text})
            if before:
                blocks.insert(0, control)
            else:
                blocks.append(control)
            messages[index] = ConversationMessage(
                role=message.role,
                content=tuple(blocks),
                extensions=message.extensions,
            )
            return ConversationRequest(
                system=self.system,
                messages=tuple(messages),
                tools=self.tools,
                parameters=self.parameters,
                extensions=self.extensions,
            )
        raise ValueError("conversation has no user message for control text")

    def with_system_text(self, text: str) -> "ConversationRequest":
        """Append strategy-owned system guidance without replacing caller blocks."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("system text must be non-empty")
        return ConversationRequest(
            system=self.system + (freeze_value({"type": "text", "text": text}),),
            messages=self.messages,
            tools=self.tools,
            parameters=self.parameters,
            extensions=self.extensions,
        )

    def with_parameters(self, updates: Mapping[str, Any]) -> "ConversationRequest":
        """Return a copy with validated strategy-level inference overrides."""

        if not isinstance(updates, Mapping):
            raise TypeError("parameter updates must be a mapping")
        parameters = dict(thaw_value(self.parameters))
        parameters.update(thaw_value(freeze_value(updates)))
        return ConversationRequest(
            system=self.system,
            messages=self.messages,
            tools=self.tools,
            parameters=parameters,
            extensions=self.extensions,
        )


@dataclass(frozen=True)
class ConversationExecutionRequest:
    """One physical model attempt selected by SmartAsk."""

    model: str
    role: str
    conversation: ConversationRequest

    def __post_init__(self) -> None:
        for name in ("model", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not isinstance(self.conversation, ConversationRequest):
            raise TypeError("conversation must be a ConversationRequest")


@dataclass(frozen=True)
class ConversationEvent:
    """Provider-neutral event emitted by a conversation executor."""

    kind: EventKind
    data: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown conversation event kind: {self.kind!r}")
        data = freeze_value(self.data)
        if not isinstance(data, Mapping):
            raise TypeError("event data must be a mapping")
        object.__setattr__(self, "data", data)
