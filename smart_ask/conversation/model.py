"""Immutable values for strategy-owned conversation execution.

This module deliberately contains no provider clients or strategy policy.  It
defines the structured input a strategy sees, the calls and decisions made
during one invocation, and the evidence returned by the execution engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, NewType

from .domain import (
    ConversationEvent,
    ConversationMessage,
    freeze_value,
    thaw_value,
)


DecisionId = NewType("DecisionId", str)
CallStatus = Literal["planned", "running", "completed", "error", "cancelled"]
ProviderRequestStatus = Literal["running", "completed", "error", "cancelled"]
RunStatus = Literal["running", "completed", "error", "cancelled"]
OutputStatus = Literal["usable", "empty", "truncated", "refused"]
InputTokenCountProvenance = Literal["exact", "estimate"]
TokenCountProvenance = Literal[
    "exact",
    "estimate",
    "upper_bound",
    "estimated_max",
]


def _trimmed(value: object, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or value != value.strip():
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be non-empty trimmed text{suffix}")


def _optional_count(value: object, name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None")


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
class Conversation:
    """The complete immutable conversation snapshot for one method invocation."""

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
        messages = tuple(self.messages)
        if not messages or any(
            not isinstance(message, ConversationMessage) for message in messages
        ):
            raise ValueError("messages must contain ConversationMessage values")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", _mapping_tuple(self.tools, "tools"))
        for name in ("parameters", "extensions"):
            frozen = freeze_value(getattr(self, name))
            if not isinstance(frozen, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, frozen)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        system: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "Conversation":
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if system is not None and (
            not isinstance(system, str) or not system.strip()
        ):
            raise ValueError("system must be non-empty text or None")
        system_blocks = (
            ()
            if system is None
            else ({"type": "text", "text": system},)
        )
        return cls(
            system=system_blocks,
            messages=(ConversationMessage(
                role="user",
                content=({"type": "text", "text": text},),
            ),),
            parameters=parameters or {},
        )

    def latest_user_message(self) -> ConversationMessage | None:
        """Return the literal latest user-role message, including tool results."""

        return next(
            (message for message in reversed(self.messages) if message.role == "user"),
            None,
        )

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

    def with_latest_human_text(self, text: str, *, before: bool) -> "Conversation":
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
            return Conversation(
                system=self.system,
                messages=tuple(messages),
                tools=self.tools,
                parameters=self.parameters,
                extensions=self.extensions,
            )
        raise ValueError("conversation has no user message for control text")

    def with_system_text(self, text: str) -> "Conversation":
        """Append strategy-owned system guidance without replacing caller blocks."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("system text must be non-empty")
        return Conversation(
            system=self.system + (freeze_value({"type": "text", "text": text}),),
            messages=self.messages,
            tools=self.tools,
            parameters=self.parameters,
            extensions=self.extensions,
        )

    def with_parameters(self, updates: Mapping[str, Any]) -> "Conversation":
        """Return a copy with validated strategy-level inference overrides."""

        if not isinstance(updates, Mapping):
            raise TypeError("parameter updates must be a mapping")
        parameters = dict(thaw_value(self.parameters))
        parameters.update(thaw_value(freeze_value(updates)))
        return Conversation(
            system=self.system,
            messages=self.messages,
            tools=self.tools,
            parameters=parameters,
            extensions=self.extensions,
        )


@dataclass(frozen=True)
class RunMetadata:
    """Correlation and strategy identity for one independent invocation."""

    strategy_name: str
    strategy_digest: str
    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    request_id: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _trimmed(self.strategy_name, "strategy_name")
        _trimmed(self.strategy_digest, "strategy_digest")
        for name in ("session_id", "agent_id", "parent_agent_id", "request_id"):
            _trimmed(getattr(self, name), name, optional=True)
        frozen = freeze_value(self.extensions)
        if not isinstance(frozen, Mapping):
            raise TypeError("extensions must be a mapping")
        object.__setattr__(self, "extensions", frozen)


@dataclass(frozen=True)
class ModelCallSpec:
    """One logical model call selected by a strategy method."""

    profile_id: str
    target_id: str
    role: str
    conversation: Conversation
    phase: str | None = None

    def __post_init__(self) -> None:
        for name in ("profile_id", "target_id", "role"):
            _trimmed(getattr(self, name), name)
        _trimmed(self.phase, "phase", optional=True)
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")


@dataclass(frozen=True)
class DecisionDraft:
    """A strategy decision before the engine assigns its stable identifier."""

    gate: str
    outcome: str
    reason_code: str | None = None
    selected_profile_id: str | None = None
    evidence_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _trimmed(self.gate, "gate")
        _trimmed(self.outcome, "outcome")
        _trimmed(self.reason_code, "reason_code", optional=True)
        _trimmed(
            self.selected_profile_id,
            "selected_profile_id",
            optional=True,
        )
        ids = tuple(self.evidence_call_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_call_ids must not contain duplicates")
        for call_id in ids:
            _trimmed(call_id, "evidence call id")
        object.__setattr__(self, "evidence_call_ids", ids)


@dataclass(frozen=True)
class DecisionRecord:
    """One immutable, auditable strategy decision."""

    decision_id: DecisionId
    gate: str
    outcome: str
    reason_code: str | None
    selected_profile_id: str | None
    evidence_call_ids: tuple[str, ...]
    sequence: int


@dataclass(frozen=True)
class ModelCallResult:
    """Successful normalized evidence from one buffered logical model call."""

    call_id: str
    events: tuple[ConversationEvent, ...]
    selected_model: str | None
    actual_model: str | None
    text: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    provider_cost_usd: float | None
    tool_call_count: int
    stream_complete: bool
    output_status: OutputStatus
    duration_ms: float

    def __post_init__(self) -> None:
        _trimmed(self.call_id, "call_id")
        _trimmed(self.selected_model, "selected_model", optional=True)
        _trimmed(self.actual_model, "actual_model", optional=True)
        _trimmed(self.stop_reason, "stop_reason", optional=True)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        events = tuple(self.events)
        if any(not isinstance(event, ConversationEvent) for event in events):
            raise TypeError("events must contain ConversationEvent values")
        object.__setattr__(self, "events", events)
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            _optional_count(getattr(self, name), name)
        _optional_count(self.tool_call_count, "tool_call_count")
        if not isinstance(self.stream_complete, bool):
            raise TypeError("stream_complete must be a boolean")
        if self.output_status not in ("usable", "empty", "truncated", "refused"):
            raise ValueError("output_status is invalid")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, Real)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be non-negative")
        if self.provider_cost_usd is not None and (
            isinstance(self.provider_cost_usd, bool)
            or not isinstance(self.provider_cost_usd, Real)
            or self.provider_cost_usd < 0
        ):
            raise ValueError("provider_cost_usd must be non-negative or None")

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call_count > 0


@dataclass(frozen=True)
class ProviderRequestRecord:
    """Evidence for one physical request at the executor/provider boundary."""

    provider_request_id: str
    call_id: str
    sequence: int
    status: ProviderRequestStatus
    target_id: str
    requested_max_output_tokens: int | None
    selected_model: str | None
    actual_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    visible_output_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    provider_cost_usd: float | None
    stop_reason: str | None
    stream_complete: bool
    tool_call_count: int
    visible_text_chars: int
    output_status: OutputStatus
    time_to_first_output_ms: float | None
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ModelCallRecord:
    """Policy-level identity and causality for one logical model call."""

    call_id: str
    sequence: int
    profile_id: str
    target_id: str
    selected_model: str | None
    role: str
    phase: str | None
    caused_by_decision_id: DecisionId | None
    provider_request_ids: tuple[str, ...]
    status: CallStatus
    error: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """Content-free source of truth for one method invocation."""

    run_id: str
    metadata: RunMetadata
    status: RunStatus
    started_at: float
    duration_ms: float
    decisions: tuple[DecisionRecord, ...]
    model_calls: tuple[ModelCallRecord, ...]
    provider_requests: tuple[ProviderRequestRecord, ...]
    final_call_id: str | None
    final_decision_id: DecisionId | None
    error: str | None = None


@dataclass(frozen=True)
class CompletedRun:
    """Collected user-visible events and their authoritative run record."""

    events: tuple[ConversationEvent, ...]
    record: RunRecord


@dataclass(frozen=True)
class InputTokenCount:
    """One target transport's count, with honest tokenizer provenance."""

    value: int
    provenance: InputTokenCountProvenance

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Integral)
            or self.value < 0
        ):
            raise ValueError("value must be a non-negative integer")
        if self.provenance not in ("exact", "estimate"):
            raise ValueError("provenance must be exact or estimate")


@dataclass(frozen=True)
class TokenCount:
    """Read-only token count across every currently reachable final request."""

    value: int
    provenance: TokenCountProvenance
    candidate_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Integral)
            or self.value < 0
        ):
            raise ValueError("value must be a non-negative integer")
        if self.provenance not in (
            "exact",
            "estimate",
            "upper_bound",
            "estimated_max",
        ):
            raise ValueError("invalid token-count provenance")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, Integral)
            or self.candidate_count < 1
        ):
            raise ValueError("candidate_count must be a positive integer")


@dataclass(frozen=True)
class _PreparedPayload:
    kind: Literal["live", "replay"]
    call_id: str
    selected_by: DecisionId
    result: ModelCallResult | None = None


_PREPARED_KEY = object()


class PreparedResponse:
    """Opaque, scope-bound, single-use plan for the one visible response.

    Strategy methods obtain instances only through ``RunScope.plan_live`` or
    ``RunScope.plan_replay``.  The execution engine is the only consumer.
    """

    __slots__ = ("__owner", "__payload", "__used")

    def __init__(
        self,
        owner: object,
        payload: _PreparedPayload,
        *,
        _key: object,
    ) -> None:
        if _key is not _PREPARED_KEY:
            raise TypeError("PreparedResponse values are created by RunScope")
        self.__owner = owner
        self.__payload = payload
        self.__used = False

    @classmethod
    def _create(
        cls,
        owner: object,
        payload: _PreparedPayload,
    ) -> "PreparedResponse":
        return cls(owner, payload, _key=_PREPARED_KEY)

    def _consume(self, owner: object) -> _PreparedPayload:
        if owner is not self.__owner:
            raise RuntimeError("prepared response belongs to a different run")
        if self.__used:
            raise RuntimeError("prepared response has already been consumed")
        self.__used = True
        return self.__payload
