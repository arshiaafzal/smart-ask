"""Harness-neutral immutable conversation values and streaming events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
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
        if any(not isinstance(key, str) for key in value):
            raise TypeError("conversation mapping keys must be strings")
        return MappingProxyType({
            key: freeze_value(item) for key, item in value.items()
        })
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("conversation numbers must be finite")
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
class ConversationEvent:
    """Provider-neutral event emitted by a model-call transport."""

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
