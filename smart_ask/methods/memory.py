"""Optional cross-invocation route affinity for conversation methods."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
import re
import time
from typing import Protocol, runtime_checkable

from .._numeric import is_finite_real
from ..conversation.domain import thaw_value
from ..conversation.model import Conversation, RunMetadata


_SYSTEM_REMINDER = re.compile(
    r"<system-reminder\b[^>]*>.*?</system-reminder>",
    flags=re.IGNORECASE | re.DOTALL,
)


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


@dataclass(frozen=True)
class CompactRouteState:
    """Persistent compact context for one routed human instruction."""

    profile_id: str
    target_id: str
    conversation: Conversation
    source_message_count: int

    def __post_init__(self) -> None:
        for name in ("profile_id", "target_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if (
            isinstance(self.source_message_count, bool)
            or not isinstance(self.source_message_count, int)
            or self.source_message_count < 1
        ):
            raise ValueError("source_message_count must be a positive integer")

    def apply(self, current: Conversation) -> Conversation:
        """Append post-handoff tool traffic to the compact conversation."""

        if not isinstance(current, Conversation):
            raise TypeError("current must be a Conversation")
        if len(current.messages) < self.source_message_count:
            raise ValueError("current conversation predates compact handoff")
        tail = current.messages[self.source_message_count :]
        return Conversation(
            system=self.conversation.system,
            messages=self.conversation.messages + tail,
            tools=self.conversation.tools,
            parameters=current.parameters,
            extensions=current.extensions,
        )


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


@dataclass(frozen=True)
class _StoredCompactState:
    state: CompactRouteState
    stored_at: float


class InMemoryRouteMemory:
    """Concurrency-safe bounded LRU affinity storage with monotonic expiry."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        session_ttl_seconds: float = 300.0,
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
        if (
            isinstance(session_ttl_seconds, bool)
            or not isinstance(session_ttl_seconds, Real)
            or not is_finite_real(session_ttl_seconds)
            or session_ttl_seconds <= 0
        ):
            raise ValueError("session_ttl_seconds must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._ttl_seconds = float(ttl_seconds)
        self._session_ttl_seconds = float(session_ttl_seconds)
        self._max_entries = int(max_entries)
        self._clock = clock
        self._entries: OrderedDict[str, _StoredAffinity] = OrderedDict()
        self._session_entries: OrderedDict[str, _StoredAffinity] = OrderedDict()
        self._compact_entries: OrderedDict[str, _StoredCompactState] = OrderedDict()
        self._session_compact_entries: OrderedDict[
            str, _StoredCompactState
        ] = OrderedDict()
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
            if current is not None and (
                current.affinity.profile_id != affinity.profile_id
                or current.affinity.target_id != affinity.target_id
            ):
                self._compact_entries.pop(key, None)
            self._entries[key] = _StoredAffinity(affinity, now)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                evicted, _ = self._entries.popitem(last=False)
                self._compact_entries.pop(evicted, None)

    async def get_recent_session_affinity(
        self,
        metadata: RunMetadata,
    ) -> RouteAffinity | None:
        """Return the last successful route only for switch detection."""

        key = _session_route_key(metadata)
        if key is None:
            return None
        async with self._lock:
            self._prune(self._clock())
            stored = self._session_entries.get(key)
            if stored is None:
                return None
            self._session_entries.move_to_end(key)
            return stored.affinity

    async def put_recent_session_affinity(
        self,
        metadata: RunMetadata,
        affinity: RouteAffinity,
    ) -> None:
        """Remember a successful route without forcing future routing."""

        if not isinstance(affinity, RouteAffinity):
            raise TypeError("affinity must be a RouteAffinity")
        key = _session_route_key(metadata)
        if key is None:
            return
        async with self._lock:
            now = self._clock()
            self._prune(now)
            self._session_entries[key] = _StoredAffinity(affinity, now)
            self._session_entries.move_to_end(key)
            while len(self._session_entries) > self._max_entries:
                self._session_entries.popitem(last=False)

    async def get_compact_state(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> CompactRouteState | None:
        key = route_memory_key(conversation, metadata)
        if key is None:
            return None
        async with self._lock:
            self._prune(self._clock())
            stored = self._compact_entries.get(key)
            if stored is None:
                return None
            self._compact_entries.move_to_end(key)
            return stored.state

    async def put_compact_state(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
        state: CompactRouteState,
    ) -> None:
        if not isinstance(state, CompactRouteState):
            raise TypeError("state must be a CompactRouteState")
        key = route_memory_key(conversation, metadata)
        if key is None:
            return
        async with self._lock:
            now = self._clock()
            self._prune(now)
            self._compact_entries[key] = _StoredCompactState(state, now)
            self._compact_entries.move_to_end(key)
            while len(self._compact_entries) > self._max_entries:
                self._compact_entries.popitem(last=False)
            session_key = _session_route_key(metadata)
            if session_key is not None:
                self._session_compact_entries[session_key] = _StoredCompactState(
                    state,
                    now,
                )
                self._session_compact_entries.move_to_end(session_key)
                while len(self._session_compact_entries) > self._max_entries:
                    self._session_compact_entries.popitem(last=False)

    async def get_recent_compact_state(
        self,
        metadata: RunMetadata,
    ) -> CompactRouteState | None:
        key = _session_route_key(metadata)
        if key is None:
            return None
        async with self._lock:
            self._prune(self._clock())
            stored = self._session_compact_entries.get(key)
            if stored is None:
                return None
            self._session_compact_entries.move_to_end(key)
            return stored.state

    async def clear_compact_state(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> None:
        route_key = route_memory_key(conversation, metadata)
        session_key = _session_route_key(metadata)
        async with self._lock:
            if route_key is not None:
                self._compact_entries.pop(route_key, None)
            if session_key is not None:
                self._session_compact_entries.pop(session_key, None)

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, value in self._entries.items()
            if now - value.stored_at > self._ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)
            self._compact_entries.pop(key, None)
        expired_sessions = [
            key
            for key, value in self._session_entries.items()
            if now - value.stored_at > self._session_ttl_seconds
        ]
        for key in expired_sessions:
            self._session_entries.pop(key, None)
            self._session_compact_entries.pop(key, None)


def _session_route_key(metadata: RunMetadata) -> str | None:
    if not isinstance(metadata, RunMetadata):
        raise TypeError("metadata must be RunMetadata")
    if metadata.session_id is None:
        return None
    payload = {
        "strategy_digest": metadata.strategy_digest,
        "principal_id": metadata.extensions.get("principal_id") or "",
        "session_id": metadata.session_id,
        "agent_id": metadata.agent_id or metadata.parent_agent_id or "root",
    }
    return sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


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
        "system": _semantic_value(conversation.system),
        "messages": [
            _route_message_payload(message)
            for message in conversation.messages[: instruction_index + 1]
        ],
        "tools": _semantic_value(conversation.tools),
        "parameters": _semantic_value(conversation.parameters),
        "extensions": _semantic_value(conversation.extensions),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _route_message_payload(message) -> dict[str, object]:
    """Canonicalize generated reminder text out of an affinity fingerprint."""

    content = []
    for frozen_block in message.content:
        block = _semantic_value(frozen_block)
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            text = _SYSTEM_REMINDER.sub("", block["text"]).strip()
            if not text:
                continue
            block["text"] = text
        content.append(block)
    return {
        "role": message.role,
        "content": content,
        "extensions": _semantic_value(message.extensions),
    }


def _semantic_value(value):
    """Remove transport-only cache annotations from an affinity key."""

    thawed = thaw_value(value)
    if isinstance(thawed, dict):
        return {
            key: _semantic_value(item)
            for key, item in thawed.items()
            if key != "cache_control"
        }
    if isinstance(thawed, (list, tuple)):
        return [_semantic_value(item) for item in thawed]
    return thawed


def _latest_instruction_index(conversation: Conversation) -> int | None:
    """Find the latest real human instruction, ignoring harness reminders.

    A coding harness may attach ``<system-reminder>`` text to tool-result
    messages. Treating that generated text as a new instruction changes the
    affinity key after every tool call and defeats cache-preserving route
    pinning.
    """

    for index in range(len(conversation.messages) - 1, -1, -1):
        message = conversation.messages[index]
        if message.role != "user":
            continue
        if any(
            block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and _SYSTEM_REMINDER.sub("", block.get("text")).strip()
            for block in message.content
        ):
            return index
    return None
