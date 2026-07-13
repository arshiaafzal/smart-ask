"""Optional cross-invocation route affinity for conversation methods."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
import time
from typing import Protocol, runtime_checkable

from .._numeric import is_finite_real
from ..conversation.domain import thaw_value
from ..conversation.model import Conversation, RunMetadata


@dataclass(frozen=True)
class RouteAffinity:
    """The compiled profile selected for one stable human turn."""

    profile_id: str
    target_id: str
    locked: bool = False

    def __post_init__(self) -> None:
        for name in ("profile_id", "target_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not isinstance(self.locked, bool):
            raise TypeError("locked must be a boolean")


@runtime_checkable
class RouteMemory(Protocol):
    """Store profile affinity without embedding session policy in the engine."""

    async def get(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> RouteAffinity | None: ...

    async def put(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
        affinity: RouteAffinity,
    ) -> None: ...


@dataclass(frozen=True)
class _StoredAffinity:
    affinity: RouteAffinity
    stored_at: float


class InMemoryRouteMemory:
    """Concurrency-safe bounded LRU affinity storage with monotonic expiry."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, Real)
            or not is_finite_real(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be finite and positive")
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, Integral)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._clock = clock
        self._entries: OrderedDict[str, _StoredAffinity] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> RouteAffinity | None:
        key = route_memory_key(conversation, metadata)
        if key is None:
            return None
        async with self._lock:
            self._prune(self._clock())
            stored = self._entries.get(key)
            if stored is None:
                return None
            self._entries.move_to_end(key)
            return stored.affinity

    async def put(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
        affinity: RouteAffinity,
    ) -> None:
        if not isinstance(affinity, RouteAffinity):
            raise TypeError("affinity must be a RouteAffinity")
        key = route_memory_key(conversation, metadata)
        if key is None:
            return
        async with self._lock:
            now = self._clock()
            self._prune(now)
            current = self._entries.get(key)
            if current is not None and current.affinity.locked and not affinity.locked:
                return
            self._entries[key] = _StoredAffinity(affinity, now)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, value in self._entries.items()
            if now - value.stored_at > self._ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)


def route_memory_key(
    conversation: Conversation,
    metadata: RunMetadata,
) -> str | None:
    """Fingerprint stable human-turn context, excluding later tool traffic."""

    if not isinstance(conversation, Conversation):
        raise TypeError("conversation must be a Conversation")
    if not isinstance(metadata, RunMetadata):
        raise TypeError("metadata must be RunMetadata")
    if metadata.session_id is None:
        return None

    instruction_index = _latest_instruction_index(conversation)
    if instruction_index is None:
        return None
    principal = metadata.extensions.get("principal_id")
    if principal is not None and not isinstance(principal, str):
        raise TypeError("metadata principal_id must be text when present")
    agent = metadata.agent_id or metadata.parent_agent_id or "root"
    payload = {
        "strategy_digest": metadata.strategy_digest,
        "principal_id": principal or "",
        "session_id": metadata.session_id,
        "agent_id": agent,
        "system": thaw_value(conversation.system),
        "messages": [
            {
                "role": message.role,
                "content": thaw_value(message.content),
                "extensions": thaw_value(message.extensions),
            }
            for message in conversation.messages[: instruction_index + 1]
        ],
        "tools": thaw_value(conversation.tools),
        "parameters": thaw_value(conversation.parameters),
        "extensions": thaw_value(conversation.extensions),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _latest_instruction_index(conversation: Conversation) -> int | None:
    for index in range(len(conversation.messages) - 1, -1, -1):
        message = conversation.messages[index]
        if message.role != "user":
            continue
        if any(
            block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block.get("text").strip()
            for block in message.content
        ):
            return index
    return None
