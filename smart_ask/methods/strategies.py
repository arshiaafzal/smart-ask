"""Conversation-native strategy methods.

These methods own semantic orchestration while :class:`RunScope` owns every
effectful model call.  They deliberately do not know about providers, clients,
metrics stores, trace writers, or harness protocols.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from numbers import Integral, Real
import re
from types import MappingProxyType
from typing import Any, Literal, cast

from ..conversation.domain import ConversationMessage, freeze_value, thaw_value
from ..conversation.engine import RunScope
from ..conversation.model import (
    Conversation,
    DecisionDraft,
    DecisionId,
    ModelCallResult,
    ModelCallSpec,
    PreparedResponse,
)
from .memory import CompactRouteState, RouteAffinity, RouteMemory


Difficulty = Literal["easy", "hard"]
RouteLabel = Literal["sonnet", "opus", "uncertain"]
ClassifierFallback = Literal[
    "easy", "hard", "sonnet", "opus", "uncertain", "raise"
]
RoutingProjection = Literal["latest_user_text", "full_conversation"]
ClassifierPrefilter = Literal["none", "exact_replies"]
ContinuationPolicy = Literal[
    "raise",
    "route_easy",
    "route_hard",
    "classify_latest_human_instruction",
    "classify_full_conversation",
    "classify_tool_result",
]
ToolCallPolicy = Literal["accept_and_pin", "escalate", "raise"]
_SYSTEM_REMINDER = re.compile(
    r"<system-reminder\b[^>]*>.*?</system-reminder>",
    flags=re.IGNORECASE | re.DOTALL,
)
_EXACT_REPLY = re.compile(
    r"^\s*reply\s+with\s+exactly\b[\s\S]*?\bdo\s+not\s+use\s+tools?\.?\s*$",
    re.IGNORECASE,
)


def _without_system_reminders(text: str) -> str:
    """Remove injected reminder spans while retaining adjacent human text."""

    return _SYSTEM_REMINDER.sub("", text).strip()


class RoutingInputError(ValueError):
    """The configured projection cannot obtain input for this invocation."""


class CandidateToolCallError(RuntimeError):
    """A buffered candidate emitted a tool call forbidden by method policy."""


@dataclass(frozen=True)
class RequestTransform:
    """An explicit, structured transformation applied before one model call."""

    system_suffix: tuple[str, ...] = ()
    latest_user_prefix: str = ""
    latest_user_suffix: str = ""
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    keep_last_messages: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.system_suffix, (str, bytes, bytearray)):
            raise TypeError("system_suffix must be a sequence of text values")
        if not isinstance(self.system_suffix, tuple):
            object.__setattr__(self, "system_suffix", tuple(self.system_suffix))
        if any(
            not isinstance(text, str) or not text.strip()
            for text in self.system_suffix
        ):
            raise ValueError("system_suffix entries must be non-empty text")
        for name in ("latest_user_prefix", "latest_user_suffix"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be text")
        frozen = freeze_value(self.parameters)
        if not isinstance(frozen, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "parameters", frozen)
        klm = self.keep_last_messages
        if klm is not None and (
            isinstance(klm, bool)
            or not isinstance(klm, int)
            or klm < 1
        ):
            raise ValueError("keep_last_messages must be a positive integer or None")

    def apply(self, conversation: Conversation) -> Conversation:
        """Apply transforms without flattening unrelated structured content."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        transformed = conversation
        # Context truncation: keep first message (original task) + last N-1.
        # Applied before text transforms so prefixes/suffixes attach correctly.
        if self.keep_last_messages is not None:
            msgs = list(transformed.messages)
            n = self.keep_last_messages
            if len(msgs) > n:
                head = msgs[:1]
                tail_start = len(msgs) - (n - 1) if n > 1 else len(msgs)
                # A tool_result is valid only when its tool_call remains in the
                # preceding assistant message. Expand the boundary across that
                # exchange (and any harness-only messages between them).
                if tail_start < len(msgs) and any(
                    block.get("type") == "tool_result"
                    for block in msgs[tail_start].content
                ):
                    while tail_start > 1:
                        tail_start -= 1
                        if msgs[tail_start].role == "assistant" and any(
                            block.get("type") == "tool_call"
                            for block in msgs[tail_start].content
                        ):
                            break
                tail = msgs[tail_start:]
                # avoid duplicating the anchor if it is already inside the tail
                if tail and msgs[0] is tail[0]:
                    msgs = tail
                else:
                    msgs = head + tail
                transformed = Conversation(
                    system=transformed.system,
                    messages=tuple(msgs),
                    tools=transformed.tools,
                    parameters=transformed.parameters,
                    extensions=transformed.extensions,
                )
        for text in self.system_suffix:
            transformed = transformed.with_system_text(text)
        if self.latest_user_prefix:
            transformed = transformed.with_latest_human_text(
                self.latest_user_prefix,
                before=True,
            )
        if self.latest_user_suffix:
            transformed = transformed.with_latest_human_text(
                self.latest_user_suffix,
                before=False,
            )
        if self.parameters:
            transformed = transformed.with_parameters(self.parameters)
        return transformed

    def then(self, other: "RequestTransform") -> "RequestTransform":
        """Compose two declarative transforms in their application order."""

        if not isinstance(other, RequestTransform):
            raise TypeError("other must be a RequestTransform")
        parameters = dict(thaw_value(self.parameters))
        parameters.update(thaw_value(other.parameters))
        # keep_last_messages: other overrides self when both are set
        keep = (
            other.keep_last_messages
            if other.keep_last_messages is not None
            else self.keep_last_messages
        )
        return RequestTransform(
            system_suffix=self.system_suffix + other.system_suffix,
            latest_user_prefix=(
                other.latest_user_prefix + self.latest_user_prefix
            ),
            latest_user_suffix=(
                self.latest_user_suffix + other.latest_user_suffix
            ),
            parameters=parameters,
            keep_last_messages=keep,
        )


@dataclass(frozen=True)
class ModelProfile:
    """A compiled, provider-independent generation profile."""

    profile_id: str
    target_id: str
    transform: RequestTransform = field(default_factory=RequestTransform)

    def __post_init__(self) -> None:
        for name in ("profile_id", "target_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be non-empty trimmed text")
        if not isinstance(self.transform, RequestTransform):
            raise TypeError("transform must be a RequestTransform")

    def call(
        self,
        conversation: Conversation,
        *,
        role: str,
        phase: str,
        method_transform: RequestTransform | None = None,
    ) -> ModelCallSpec:
        """Build a call from the immutable input and explicit transforms."""

        if not isinstance(role, str) or not role or role != role.strip():
            raise ValueError("role must be non-empty trimmed text")
        if not isinstance(phase, str) or not phase or phase != phase.strip():
            raise ValueError("phase must be non-empty trimmed text")
        transform = self.transform
        if method_transform is not None:
            transform = transform.then(method_transform)
        return ModelCallSpec(
            profile_id=self.profile_id,
            target_id=self.target_id,
            role=role,
            conversation=transform.apply(conversation),
            phase=phase,
        )


@dataclass(frozen=True)
class DifficultyAssessment:
    """One normalized classification and the call that supports it."""

    difficulty: Difficulty
    reason_code: str
    evidence_call_ids: tuple[str, ...] = ()
    route: RouteLabel | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.difficulty not in ("easy", "hard"):
            raise ValueError("difficulty must be easy or hard")
        route = self.route
        if route is None:
            route = "sonnet" if self.difficulty == "easy" else "opus"
            object.__setattr__(self, "route", route)
        if route not in ("sonnet", "opus", "uncertain"):
            raise ValueError("route must be sonnet, opus, or uncertain")
        expected = "easy" if route == "sonnet" else "hard"
        if self.difficulty != expected:
            raise ValueError("difficulty must agree with route")
        confidence = self.confidence
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1 or None")
        if confidence is not None:
            object.__setattr__(self, "confidence", float(confidence))
        if (
            not isinstance(self.reason_code, str)
            or not self.reason_code
            or self.reason_code != self.reason_code.strip()
        ):
            raise ValueError("reason_code must be non-empty trimmed text")
        if not isinstance(self.evidence_call_ids, tuple):
            object.__setattr__(
                self,
                "evidence_call_ids",
                tuple(self.evidence_call_ids),
            )


class StructuredDifficultyClassifier:
    """Classify a structured conversation through the shared run scope."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        prompt: str,
        projection: RoutingProjection,
        continuation: ContinuationPolicy,
        fallback: ClassifierFallback,
        max_prompt_chars: int | None,
        prefilter: ClassifierPrefilter = "none",
        sonnet_min_confidence: float = 0.75,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(profile, ModelProfile):
            raise TypeError("profile must be a ModelProfile")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if projection not in ("latest_user_text", "full_conversation"):
            raise ValueError("unknown routing projection")
        if continuation not in (
            "raise",
            "route_easy",
            "route_hard",
            "classify_latest_human_instruction",
            "classify_full_conversation",
            "classify_tool_result",
        ):
            raise ValueError("unknown continuation policy")
        if fallback not in (
            "easy", "hard", "sonnet", "opus", "uncertain", "raise"
        ):
            raise ValueError("unknown classifier fallback")
        if prefilter not in ("none", "exact_replies"):
            raise ValueError("unknown classifier prefilter")
        if (
            isinstance(sonnet_min_confidence, bool)
            or not isinstance(sonnet_min_confidence, Real)
            or not 0 <= sonnet_min_confidence <= 1
        ):
            raise ValueError("sonnet_min_confidence must be between 0 and 1")
        if max_prompt_chars is not None and (
            isinstance(max_prompt_chars, bool)
            or not isinstance(max_prompt_chars, Integral)
            or max_prompt_chars < 1
        ):
            raise ValueError("max_prompt_chars must be positive or None")
        if projection == "full_conversation" and max_prompt_chars is not None:
            raise ValueError(
                "full_conversation projection cannot use a character cap"
            )
        self._profile = profile
        self._prompt = prompt
        self._projection = projection
        self._continuation = continuation
        self._fallback = fallback
        self._prefilter = prefilter
        self._sonnet_min_confidence = float(sonnet_min_confidence)
        self._max_prompt_chars = (
            None if max_prompt_chars is None else int(max_prompt_chars)
        )
        frozen_parameters = freeze_value(parameters or {})
        if not isinstance(frozen_parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        self._parameters = frozen_parameters

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    async def classify(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> DifficultyAssessment:
        """Classify once, applying explicit continuation and failure policies."""

        direct_human_text = True
        try:
            projected = self._project(conversation, self._projection)
        except RoutingInputError:
            direct_human_text = False
            continuation = self._continuation
            if continuation == "raise":
                raise
            if continuation in ("route_easy", "route_hard"):
                difficulty = cast(Difficulty, continuation.removeprefix("route_"))
                route: RouteLabel = "sonnet" if difficulty == "easy" else "opus"
                return DifficultyAssessment(
                    difficulty,
                    f"continuation_{route}",
                    route=route,
                    confidence=1.0,
                )
            if continuation == "classify_latest_human_instruction":
                projected = self._project_latest_human_instruction(conversation)
            elif continuation == "classify_tool_result":
                projected = self._project_tool_result(conversation)
            else:
                projected = self._project(conversation, "full_conversation")

        if direct_human_text and self._prefilter == "exact_replies":
            deterministic = self._prefilter_direct_human_text(projected)
            if deterministic is not None:
                return deterministic

        transform = RequestTransform(
            system_suffix=(self._prompt,),
            parameters=self._parameters,
        )
        spec = self._profile.call(
            projected,
            role="classifier",
            phase="classification",
            method_transform=transform,
        )
        try:
            result = await run.call_buffered(spec)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._fallback == "raise":
                raise
            return self._fallback_assessment("classifier_execution_fallback")

        try:
            if result.output_status != "usable":
                raise ValueError(
                    f"classifier output was {result.output_status}"
                )
            route, confidence = self._parse(result.text)
        except ValueError:
            if self._fallback == "raise":
                raise
            return self._fallback_assessment(
                "classifier_invalid_fallback",
                (result.call_id,),
            )
        if route == "sonnet" and confidence is not None and (
            confidence < self._sonnet_min_confidence
        ):
            return DifficultyAssessment(
                "hard",
                "classifier_sonnet_below_confidence_threshold",
                (result.call_id,),
                route="uncertain",
                confidence=confidence,
            )
        difficulty: Difficulty = "easy" if route == "sonnet" else "hard"
        return DifficultyAssessment(
            difficulty,
            f"classifier_{route}",
            (result.call_id,),
            route=route,
            confidence=confidence,
        )

    def _fallback_assessment(
        self,
        reason_prefix: str,
        evidence_call_ids: tuple[str, ...] = (),
    ) -> DifficultyAssessment:
        fallback = self._fallback
        if fallback == "raise":
            raise RuntimeError("raise fallback cannot be converted to a route")
        route: RouteLabel = {
            "easy": "sonnet",
            "hard": "opus",
            "sonnet": "sonnet",
            "opus": "opus",
            "uncertain": "uncertain",
        }[fallback]
        difficulty: Difficulty = "easy" if route == "sonnet" else "hard"
        return DifficultyAssessment(
            difficulty,
            f"{reason_prefix}_{route}",
            evidence_call_ids,
            route=route,
            confidence=0.0,
        )

    @staticmethod
    def _prefilter_direct_human_text(
        projected: Conversation,
    ) -> DifficultyAssessment | None:
        latest = projected.latest_user_message()
        if latest is None:
            return None
        text = "\n".join(
            block["text"]
            for block in latest.content
            if block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if _EXACT_REPLY.match(text):
            return DifficultyAssessment(
                "easy",
                "deterministic_exact_reply_sonnet",
                route="sonnet",
                confidence=1.0,
            )
        return None

    def _project(
        self,
        conversation: Conversation,
        projection: RoutingProjection,
    ) -> Conversation:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if projection == "full_conversation":
            return conversation
        latest = conversation.latest_user_message()
        if latest is None:
            raise RoutingInputError(
                "conversation has no user message; continuation policy is required"
            )
        latest_index = next(
            index
            for index in range(len(conversation.messages) - 1, -1, -1)
            if conversation.messages[index] is latest
        )
        trailing = conversation.messages[latest_index + 1 :]
        if any(
            message.content and message.role not in ("system", "developer")
            for message in trailing
        ):
            raise RoutingInputError(
                "conversation does not end in a user message; continuation policy is required"
            )
        texts = [
            cleaned
            for block in latest.content
            if block.get("type") == "text"
            and isinstance(block.get("text"), str)
            # A harness can place reminders and the human prompt in one block.
            # Strip only the metadata span, never the entire content block.
            for cleaned in (_without_system_reminders(block["text"]),)
            if cleaned
        ]
        if not texts:
            raise RoutingInputError(
                "latest user message has no text; continuation policy is required"
            )
        text = "\n\n".join(texts)
        if self._max_prompt_chars is not None:
            text = text[: self._max_prompt_chars]
        return _text_conversation(text)

    def _project_latest_human_instruction(
        self,
        conversation: Conversation,
    ) -> Conversation:
        projected = conversation.latest_human_instruction()
        if projected is None:
            raise RoutingInputError("conversation has no human text to classify")
        text = _without_system_reminders(projected[0])
        if not text:
            raise RoutingInputError("human text only contains injected context")
        if self._max_prompt_chars is not None:
            text = text[: self._max_prompt_chars]
        return _text_conversation(text)

    def _project_tool_result(
        self,
        conversation: Conversation,
    ) -> Conversation:
        """Extract tool_result text from the last two user messages.

        Looking at two messages instead of one preserves failure context when
        the most recent turn was a diagnostic read (pure source code) that
        follows a turn with test failures.  The tail is taken so that pytest
        summaries and error lines — which appear at the end — are always
        included within the character budget.
        """
        user_messages = [
            m for m in conversation.messages if m.role == "user"
        ]
        recent = user_messages[-2:] if len(user_messages) >= 2 else user_messages
        if not recent:
            raise RoutingInputError("no user message to classify")
        has_any_tool_result = any(
            b.get("type") == "tool_result"
            for m in recent
            for b in m.content
        )
        if not has_any_tool_result:
            raise RoutingInputError("no tool_result blocks in recent messages")
        texts = []
        for msg in recent:
            for block in msg.content:
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, str):
                    if content.strip():
                        texts.append(content)
                elif isinstance(content, (list, tuple)):
                    for item in content:
                        if isinstance(item, Mapping) and item.get("type") == "text":
                            t = item.get("text", "")
                            if isinstance(t, str) and t.strip():
                                texts.append(t)
        tool_text = "\n\n---\n\n".join(texts)
        task_text = _latest_real_human_text(conversation)
        prefix = f"ORIGINAL TASK:\n{task_text}\n\nLATEST TOOL OUTPUT:\n"
        if self._max_prompt_chars is None:
            text = prefix + tool_text
        else:
            # Keep enough of the task to understand the goal and preserve both
            # ends of long output. File identity and declarations tend to be at
            # the head, while test summaries and errors tend to be at the tail.
            task_budget = min(700, self._max_prompt_chars // 2)
            prefix = (
                f"ORIGINAL TASK:\n{task_text[:task_budget]}\n\n"
                "LATEST TOOL OUTPUT:\n"
            )
            remaining = max(0, self._max_prompt_chars - len(prefix))
            if len(tool_text) <= remaining:
                bounded_tool_text = tool_text
            else:
                separator = "\n\n...[middle omitted]...\n\n"
                content_budget = max(0, remaining - len(separator))
                head_budget = content_budget // 2
                tail_budget = content_budget - head_budget
                bounded_tool_text = (
                    tool_text[:head_budget]
                    + separator
                    + (tool_text[-tail_budget:] if tail_budget else "")
                )
            text = (prefix + bounded_tool_text)[: self._max_prompt_chars]
        return _text_conversation(text)

    @staticmethod
    def _parse(text: str) -> tuple[RouteLabel, float | None]:
        def reject_duplicates(pairs):
            payload = {}
            for key, value in pairs:
                if key in payload:
                    raise ValueError(f"duplicate classifier key {key!r}")
                payload[key] = value
            return payload

        try:
            payload = json.loads(text, object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("classifier returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("classifier output must be an object")
        if set(payload) == {"d"}:
            legacy = payload["d"]
            if legacy not in ("easy", "hard"):
                raise ValueError("legacy classifier label must be easy or hard")
            route: RouteLabel = "sonnet" if legacy == "easy" else "opus"
            return route, None
        if set(payload) != {"route", "confidence"}:
            raise ValueError(
                "classifier object must contain exactly route and confidence"
            )
        route = payload["route"]
        confidence = payload["confidence"]
        if route not in ("sonnet", "opus", "uncertain"):
            raise ValueError("classifier route must be sonnet, opus, or uncertain")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("classifier confidence must be between 0 and 1")
        return cast(RouteLabel, route), float(confidence)


class MarkerCandidatePolicy:
    """Apply explicit request transforms and assess a buffered candidate."""

    def __init__(
        self,
        *,
        marker: str,
        self_check_suffix: str,
        escalation_prefix: str,
        tool_calls: ToolCallPolicy,
    ):
        if (
            not isinstance(marker, str)
            or not marker
            or marker != marker.strip()
            or "\n" in marker
            or "\r" in marker
        ):
            raise ValueError("marker must be non-empty, trimmed, single-line text")
        for name, value in (
            ("self_check_suffix", self_check_suffix),
            ("escalation_prefix", escalation_prefix),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if tool_calls not in ("accept_and_pin", "escalate", "raise"):
            raise ValueError(
                "tool_calls must be accept_and_pin, escalate, or raise"
            )
        self._marker = marker
        self._self_check_suffix = self_check_suffix
        self._escalation_prefix = escalation_prefix
        self._tool_calls = tool_calls

    @property
    def candidate_transform(self) -> RequestTransform:
        return RequestTransform(latest_user_suffix=self._self_check_suffix)

    @property
    def escalation_transform(self) -> RequestTransform:
        return RequestTransform(latest_user_prefix=self._escalation_prefix)

    def escalation_conversation(
        self,
        conversation: Conversation,
        candidate: ModelCallResult,
    ) -> Conversation:
        """Append the actual cheap attempt and an explicit correction turn."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(candidate, ModelCallResult):
            raise TypeError("candidate must be a ModelCallResult")
        messages = list(conversation.messages)
        candidate_message = _assistant_candidate_message(candidate)
        if candidate_message.content:
            messages.append(candidate_message)
        messages.append(ConversationMessage(
            role="user",
            content=(freeze_value({
                "type": "text",
                "text": self._escalation_prefix,
            }),),
        ))
        return Conversation(
            system=conversation.system,
            messages=tuple(messages),
            tools=conversation.tools,
            parameters=conversation.parameters,
            extensions=conversation.extensions,
        )

    def assess(self, result: ModelCallResult) -> tuple[Literal["accept", "escalate"], str]:
        if not isinstance(result, ModelCallResult):
            raise TypeError("result must be a ModelCallResult")
        if result.has_tool_call:
            if self._tool_calls == "raise":
                raise CandidateToolCallError(
                    "candidate emitted a tool call forbidden by method policy"
                )
            if self._tool_calls == "accept_and_pin":
                return "accept", "candidate_tool_call_accept_and_pin"
            return "escalate", "candidate_tool_call_escalate"
        if result.output_status != "usable":
            return "escalate", f"candidate_{result.output_status}"
        escalated = bool(re.search(
            rf"^\s*{re.escape(self._marker)}\s*$",
            result.text,
            re.MULTILINE,
        ))
        if escalated:
            return "escalate", "candidate_marker_escalate"
        return "accept", "candidate_marker_absent"


class TerminalHandoffPolicy:
    """Build a compact finalization request only from strong completion evidence."""

    _EDIT_TOOLS = frozenset({"edit", "write", "multiedit", "notebookedit"})
    _TASK_TOOLS = frozenset({"taskcreate"})
    _TEST_COMMAND = re.compile(r"\b(?:pytest|unittest)\b", re.IGNORECASE)
    _PYTEST_PASS = re.compile(r"\b(\d+)\s+passed\b", re.IGNORECASE)
    _UNITTEST_PASS = re.compile(
        r"\bRan\s+(\d+)\s+tests?\b[\s\S]*?^OK\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    _FAILURE = re.compile(
        r"(?:\bFAILED\b|\bERRORS?\b|Traceback\s*\(|AssertionError|"
        r"\b[1-9]\d*\s+failed\b|\b[1-9]\d*\s+errors?\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        prompt: str,
        marker: str,
        min_passed_tests: int,
        max_prompt_chars: int,
        max_tokens: int,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if (
            not isinstance(marker, str)
            or not marker
            or marker != marker.strip()
            or "\n" in marker
            or "\r" in marker
        ):
            raise ValueError("marker must be non-empty, trimmed, single-line text")
        for name, value in (
            ("min_passed_tests", min_passed_tests),
            ("max_prompt_chars", max_prompt_chars),
            ("max_tokens", max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._prompt = prompt
        self._marker = marker
        self._min_passed_tests = min_passed_tests
        self._max_prompt_chars = max_prompt_chars
        self._max_tokens = max_tokens

    def compact_conversation(
        self,
        conversation: Conversation,
    ) -> Conversation | None:
        """Return bounded evidence for finalization, or None when uncertain."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        latest = conversation.latest_user_message()
        if latest is None:
            return None

        calls: dict[str, tuple[str, Mapping[str, Any]]] = {}
        edits: list[str] = []
        has_task_bookkeeping = False
        for message in conversation.messages:
            if message.role != "assistant":
                continue
            for block in message.content:
                if block.get("type") != "tool_call":
                    continue
                name = block.get("name")
                call_id = block.get("id")
                arguments = block.get("arguments", {})
                if not isinstance(name, str):
                    continue
                normalized_name = name.lower()
                if normalized_name in self._TASK_TOOLS:
                    has_task_bookkeeping = True
                if not isinstance(arguments, Mapping):
                    arguments = MappingProxyType({})
                if isinstance(call_id, str) and call_id:
                    calls[call_id] = (normalized_name, arguments)
                if normalized_name in self._EDIT_TOOLS:
                    edits.append(self._edit_summary(name, arguments))

        # A compact final answer must not abandon harness task state.
        if has_task_bookkeeping or not edits:
            return None

        successful_test: tuple[str, str] | None = None
        for block in latest.content:
            if block.get("type") != "tool_result" or block.get("is_error") is True:
                continue
            call_id = block.get("id") or block.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in calls:
                continue
            name, arguments = calls[call_id]
            if name != "bash":
                continue
            command = arguments.get("command")
            if not isinstance(command, str) or not self._TEST_COMMAND.search(command):
                continue
            output = self._tool_result_text(block)
            if self._is_strong_test_success(output):
                successful_test = (command, output)

        if successful_test is None:
            return None
        task = ""
        for message in reversed(conversation.messages):
            if message.role != "user":
                continue
            texts = [
                block.get("text")
                for block in message.content
                if block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            task = _without_system_reminders("\n\n".join(texts))
            if task:
                break
        if not task:
            return None
        command, output = successful_test
        evidence = (
            "ORIGINAL TASK:\n"
            f"{task[:1200]}\n\n"
            "IMPLEMENTATION LEDGER:\n"
            + "\n".join(f"- {item}" for item in edits[-8:])
            + "\n\nVERIFICATION COMMAND:\n"
            + command[:600]
            + "\n\nVERIFICATION OUTPUT:\n"
            + output[-1600:]
        )
        evidence = evidence[: self._max_prompt_chars]
        return Conversation(
            system=({"type": "text", "text": self._prompt},),
            messages=(ConversationMessage(
                role="user",
                content=({"type": "text", "text": evidence},),
            ),),
            tools=(),
            parameters={},
        )

    @property
    def transform(self) -> RequestTransform:
        return RequestTransform(parameters={
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        })

    def assess(self, result: ModelCallResult) -> tuple[bool, str]:
        if not isinstance(result, ModelCallResult):
            raise TypeError("result must be a ModelCallResult")
        if result.has_tool_call:
            return False, "terminal_handoff_tool_call"
        if result.output_status != "usable":
            return False, f"terminal_handoff_{result.output_status}"
        if re.search(
            rf"^\s*{re.escape(self._marker)}\s*$",
            result.text,
            re.MULTILINE,
        ):
            return False, "terminal_handoff_requested_opus"
        if not result.text.strip():
            return False, "terminal_handoff_empty"
        return True, "terminal_handoff_accepted"

    def _is_strong_test_success(self, output: str) -> bool:
        if not output or self._FAILURE.search(output):
            return False
        matches = [int(value) for value in self._PYTEST_PASS.findall(output)]
        matches.extend(int(value) for value in self._UNITTEST_PASS.findall(output))
        return bool(matches) and max(matches) >= self._min_passed_tests

    @staticmethod
    def _tool_result_text(block: Mapping[str, Any]) -> str:
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, (tuple, list)):
            return ""
        texts = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if item.get("type") == "text" and isinstance(text, str):
                texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def _edit_summary(name: str, arguments: Mapping[str, Any]) -> str:
        path = next((
            arguments.get(key)
            for key in ("file_path", "path", "notebook_path")
            if isinstance(arguments.get(key), str)
        ), "unknown file")
        detail = ""
        for key in ("new_string", "content", "new_source"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                detail = " — " + re.sub(r"\s+", " ", value.strip())[:300]
                break
        return f"{name}: {path}{detail}"


class CompactHandoffPolicy:
    """Summarize warm-model state before a cold cross-model handoff."""

    _COMPACT_SYSTEM = (
        "You are continuing an existing coding-agent session from a compact "
        "cross-model handoff. Treat the original request, constraints, and "
        "verified state below as authoritative. Use the available tools to "
        "inspect anything omitted, continue from the recorded state, and do "
        "not repeat completed work. Never claim an action or test not shown "
        "in the handoff or observed through tools."
    )

    def __init__(
        self,
        *,
        prompt: str,
        marker: str,
        max_summary_chars: int,
        max_tool_result_chars: int,
        max_tokens: int,
        tool_names: tuple[str, ...],
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        if (
            not isinstance(marker, str)
            or not marker
            or marker != marker.strip()
            or "\n" in marker
            or "\r" in marker
        ):
            raise ValueError("marker must be nonempty, trimmed, and single-line")
        for name, value in (
            ("max_summary_chars", max_summary_chars),
            ("max_tool_result_chars", max_tool_result_chars),
            ("max_tokens", max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(tool_names, (str, bytes, bytearray))
            or not isinstance(tool_names, tuple)
            or not tool_names
            or any(not isinstance(name, str) or not name.strip() for name in tool_names)
        ):
            raise ValueError("tool_names must be a non-empty tuple of names")
        self._prompt = prompt
        self._marker = marker
        self._max_summary_chars = max_summary_chars
        self._max_tool_result_chars = max_tool_result_chars
        self._max_tokens = max_tokens
        self._tool_names = frozenset(name.lower() for name in tool_names)

    @property
    def summarizer_transform(self) -> RequestTransform:
        return RequestTransform(
            latest_user_suffix="\n\n" + self._prompt,
            parameters={"max_tokens": self._max_tokens, "temperature": 0.0},
        )

    def summary_conversation(self, conversation: Conversation) -> Conversation:
        """Remove live-agent controls that are invalid or wasteful for summary."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        parameters = dict(thaw_value(conversation.parameters))
        # Coding harnesses commonly enable adaptive thinking. Anthropic rejects
        # temperature=0 with that mode, and a bounded factual handoff does not
        # need thinking or tool choice. The summary prompt supplies its own
        # output limit and temperature through ``summarizer_transform``.
        for name in ("thinking", "temperature", "tool_choice"):
            parameters.pop(name, None)
        return Conversation(
            system=conversation.system,
            messages=conversation.messages,
            tools=(),
            parameters=parameters,
            extensions=conversation.extensions,
        )

    def assess(self, result: ModelCallResult) -> tuple[bool, str]:
        if not isinstance(result, ModelCallResult):
            raise TypeError("result must be a ModelCallResult")
        if result.has_tool_call:
            return False, "compact_handoff_summarizer_tool_call"
        if result.output_status != "usable":
            return False, f"compact_handoff_summarizer_{result.output_status}"
        text = result.text.strip()
        if not text:
            return False, "compact_handoff_summarizer_empty"
        if re.search(
            rf"^\s*{re.escape(self._marker)}\s*$",
            text,
            re.MULTILINE,
        ):
            return False, "compact_handoff_summarizer_unsafe"
        return True, "compact_handoff_summary_accepted"

    def compact_state(
        self,
        conversation: Conversation,
        summary: str,
        profile: ModelProfile,
    ) -> CompactRouteState:
        """Build bounded state while retaining deterministic task evidence."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be non-empty text")
        if not isinstance(profile, ModelProfile):
            raise TypeError("profile must be a ModelProfile")
        task = _latest_real_human_text(conversation)
        if not task:
            raise ValueError("conversation has no real human instruction")
        latest_tool_output = _latest_tool_output(conversation)
        handoff = (
            "ORIGINAL REQUEST:\n"
            + task[:3000]
            + "\n\nWARM-MODEL STATE SUMMARY:\n"
            + summary.strip()[: self._max_summary_chars]
        )
        if latest_tool_output:
            handoff += (
                "\n\nLATEST TOOL OUTPUT (verbatim tail):\n"
                + latest_tool_output[-self._max_tool_result_chars :]
            )
        tools = tuple(
            tool
            for tool in conversation.tools
            if isinstance(tool.get("name"), str)
            and tool["name"].lower() in self._tool_names
        )
        compact = Conversation(
            system=({"type": "text", "text": self._COMPACT_SYSTEM},),
            messages=(ConversationMessage(
                role="user",
                content=({"type": "text", "text": handoff},),
            ),),
            tools=tools,
            parameters=conversation.parameters,
            extensions=conversation.extensions,
        )
        return CompactRouteState(
            profile_id=profile.profile_id,
            target_id=profile.target_id,
            conversation=compact,
            source_message_count=len(conversation.messages),
        )


def _latest_real_human_text(conversation: Conversation) -> str:
    for message in reversed(conversation.messages):
        if message.role != "user":
            continue
        text = "\n\n".join(
            block["text"]
            for block in message.content
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        cleaned = _without_system_reminders(text)
        if cleaned:
            return cleaned
    return ""


def _latest_tool_output(conversation: Conversation) -> str:
    latest = conversation.latest_user_message()
    if latest is None:
        return ""
    texts: list[str] = []
    for block in latest.content:
        if block.get("type") != "tool_result":
            continue
        content = block.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, (tuple, list)):
            texts.extend(
                item["text"]
                for item in content
                if isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
    return "\n".join(texts)


class FixedStrategyMethod:
    """Prepare exactly one live call through a fixed compiled profile."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        role: str,
        transform: RequestTransform,
    ):
        if not isinstance(profile, ModelProfile):
            raise TypeError("profile must be a ModelProfile")
        if not isinstance(role, str) or not role or role != role.strip():
            raise ValueError("role must be non-empty trimmed text")
        if not isinstance(transform, RequestTransform):
            raise TypeError("transform must be a RequestTransform")
        self._profile = profile
        self._role = role
        self._transform = transform

    def token_count_candidates(
        self,
        conversation: Conversation,
    ) -> tuple[ModelCallSpec, ...]:
        """Return the sole reachable generation request without executing it."""

        return (self._profile.call(
            conversation,
            role=self._role,
            phase="fixed",
            method_transform=self._transform,
        ),)

    async def respond(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> PreparedResponse:
        decision = run.record_decision(DecisionDraft(
            gate="initial",
            outcome="fixed",
            reason_code="configured_fixed_route",
            selected_profile_id=self._profile.profile_id,
        ))
        return run.plan_live(
            self._profile.call(
                conversation,
                role=self._role,
                phase="fixed",
                method_transform=self._transform,
            ),
            caused_by=decision,
        )


class DifficultyStrategyMethod:
    """Confidence-route a turn and adapt after cheap-model tool evidence."""

    def __init__(
        self,
        *,
        classifier: StructuredDifficultyClassifier,
        easy: ModelProfile,
        hard: ModelProfile,
        route_memory: RouteMemory | None,
        terminal_handoff: TerminalHandoffPolicy | None = None,
        compact_handoff: CompactHandoffPolicy | None = None,
    ):
        if not isinstance(classifier, StructuredDifficultyClassifier):
            raise TypeError("classifier must be a StructuredDifficultyClassifier")
        if not isinstance(easy, ModelProfile) or not isinstance(hard, ModelProfile):
            raise TypeError("easy and hard must be ModelProfile values")
        if route_memory is not None and not isinstance(route_memory, RouteMemory):
            raise TypeError("route_memory must implement RouteMemory or be None")
        if terminal_handoff is not None and not isinstance(
            terminal_handoff,
            TerminalHandoffPolicy,
        ):
            raise TypeError(
                "terminal_handoff must be TerminalHandoffPolicy or None"
            )
        if compact_handoff is not None and not isinstance(
            compact_handoff,
            CompactHandoffPolicy,
        ):
            raise TypeError("compact_handoff must be CompactHandoffPolicy or None")
        self._classifier = classifier
        self._easy = easy
        self._hard = hard
        self._route_memory = route_memory
        self._terminal_handoff = terminal_handoff
        self._compact_handoff = compact_handoff

    def token_count_candidates(
        self,
        conversation: Conversation,
    ) -> tuple[ModelCallSpec, ...]:
        """Return both possible generation requests without classifying."""

        return (
            self._easy.call(
                conversation,
                role="generator",
                phase="initial-easy",
            ),
            self._hard.call(
                conversation,
                role="writer",
                phase="initial-hard",
            ),
        )

    async def respond(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> PreparedResponse:
        remembered = await _remembered_profile(
            self._route_memory,
            conversation,
            run,
            (self._easy, self._hard),
        )
        if remembered is not None:
            profile, affinity = remembered
            hard = profile.profile_id == self._hard.profile_id
            if hard and self._terminal_handoff is not None:
                prepared = await self._try_terminal_handoff(conversation, run)
                if prepared is not None:
                    return prepared
            if not hard and _latest_message_has_tool_result(conversation):
                assessment = await self._classifier.classify(conversation, run)
                if assessment.route != "sonnet":
                    decision = self._assessment_decision(
                        run,
                        assessment,
                        self._hard,
                        gate="adaptive_escalation",
                    )
                    return await self._plan_profile(
                        conversation,
                        run,
                        self._hard,
                        decision,
                        phase="adaptive-escalation",
                        source_profile=self._easy,
                    )
                decision = self._assessment_decision(
                    run,
                    assessment,
                    self._easy,
                    gate="continuation_check",
                )
                return await self._plan_profile(
                    conversation,
                    run,
                    self._easy,
                    decision,
                    phase="continued-easy",
                    use_stored_compact=True,
                )
            decision = _memory_decision(run, affinity)
            return await self._plan_profile(
                conversation,
                run,
                profile,
                decision,
                phase="remembered-hard" if hard else "remembered-easy",
                use_stored_compact=True,
            )

        assessment = await self._classifier.classify(conversation, run)
        hard = assessment.difficulty == "hard"
        profile = self._hard if hard else self._easy
        decision = self._assessment_decision(run, assessment, profile)
        previous = await _recent_session_profile(
            self._route_memory,
            run,
            (self._easy, self._hard),
        )
        source = (
            previous
            if previous is not None and previous.target_id != profile.target_id
            else None
        )
        return await self._plan_profile(
            conversation,
            run,
            profile,
            decision,
            phase="initial-hard" if hard else "initial-easy",
            source_profile=source,
            context_independent=(
                assessment.reason_code == "deterministic_exact_reply_sonnet"
            ),
        )

    @staticmethod
    def _assessment_decision(
        run: RunScope,
        assessment: DifficultyAssessment,
        profile: ModelProfile,
        *,
        gate: str = "difficulty",
    ) -> DecisionId:
        return run.record_decision(DecisionDraft(
            gate=gate,
            outcome=assessment.route or assessment.difficulty,
            reason_code=assessment.reason_code,
            selected_profile_id=profile.profile_id,
            evidence_call_ids=assessment.evidence_call_ids,
            confidence=assessment.confidence,
        ))

    async def _plan_profile(
        self,
        conversation: Conversation,
        run: RunScope,
        profile: ModelProfile,
        decision: DecisionId,
        *,
        phase: str,
        source_profile: ModelProfile | None = None,
        use_stored_compact: bool = False,
        context_independent: bool = False,
    ) -> PreparedResponse:
        selected_conversation = conversation
        final_decision = decision
        state: CompactRouteState | None = None
        if use_stored_compact:
            state = await _remembered_compact_state(
                self._route_memory,
                conversation,
                run,
                profile,
            )
            if state is not None:
                selected_conversation = state.apply(conversation)
        if state is None and source_profile is not None and self._compact_handoff:
            if context_independent:
                state = self._context_independent_state(conversation, profile)
                selected_conversation = state.conversation
                final_decision = run.record_decision(DecisionDraft(
                    gate="compact_handoff",
                    outcome="accept",
                    reason_code="context_independent_switch",
                    selected_profile_id=profile.profile_id,
                ))
            else:
                compacted = await self._try_compact_handoff(
                    conversation,
                    run,
                    source_profile,
                    profile,
                )
                state, final_decision = compacted
                if state is not None:
                    selected_conversation = state.conversation
        _defer_affinity(
            self._route_memory,
            conversation,
            run,
            profile,
            locked=False,
        )
        _defer_recent_session_affinity(self._route_memory, run, profile)
        if state is not None:
            _defer_compact_state(
                self._route_memory,
                conversation,
                run,
                state,
            )
        else:
            _defer_clear_compact_state(
                self._route_memory,
                conversation,
                run,
            )
        hard = profile.profile_id == self._hard.profile_id
        return run.plan_live(
            profile.call(
                selected_conversation,
                role="writer" if hard else "generator",
                phase=phase,
            ),
            caused_by=final_decision,
        )

    async def _try_compact_handoff(
        self,
        conversation: Conversation,
        run: RunScope,
        source: ModelProfile,
        target: ModelProfile,
    ) -> tuple[CompactRouteState | None, DecisionId]:
        policy = self._compact_handoff
        if policy is None:
            return None
        attempt = run.record_decision(DecisionDraft(
            gate="compact_handoff",
            outcome="attempt",
            reason_code="cross_model_switch",
            selected_profile_id=target.profile_id,
        ))
        summarizer_conversation = conversation
        source_state = await _remembered_compact_state(
            self._route_memory,
            conversation,
            run,
            source,
        )
        if source_state is None:
            source_state = await _recent_compact_state(
                self._route_memory,
                run,
                source,
            )
        if source_state is not None:
            try:
                summarizer_conversation = source_state.apply(conversation)
            except ValueError:
                summarizer_conversation = conversation
        summarizer_conversation = policy.summary_conversation(
            summarizer_conversation
        )
        try:
            summary = await run.call_buffered(
                source.call(
                    summarizer_conversation,
                    role="summarizer",
                    phase="compact-handoff-summary",
                    method_transform=policy.summarizer_transform,
                ),
                caused_by=attempt,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            decision = run.record_decision(DecisionDraft(
                gate="compact_handoff",
                outcome="fallback_full",
                reason_code="compact_handoff_summarizer_error",
                selected_profile_id=target.profile_id,
            ))
            return None, decision
        accepted, reason = policy.assess(summary)
        if not accepted:
            decision = run.record_decision(DecisionDraft(
                gate="compact_handoff",
                outcome="fallback_full",
                reason_code=reason,
                selected_profile_id=target.profile_id,
                evidence_call_ids=(summary.call_id,),
            ))
            return None, decision
        try:
            state = policy.compact_state(conversation, summary.text, target)
        except (TypeError, ValueError):
            decision = run.record_decision(DecisionDraft(
                gate="compact_handoff",
                outcome="fallback_full",
                reason_code="compact_handoff_state_invalid",
                selected_profile_id=target.profile_id,
                evidence_call_ids=(summary.call_id,),
            ))
            return None, decision
        decision = run.record_decision(DecisionDraft(
            gate="compact_handoff",
            outcome="accept",
            reason_code=reason,
            selected_profile_id=target.profile_id,
            evidence_call_ids=(summary.call_id,),
        ))
        return state, decision

    def _context_independent_state(
        self,
        conversation: Conversation,
        profile: ModelProfile,
    ) -> CompactRouteState:
        text = _latest_real_human_text(conversation)
        compact = Conversation.from_text(
            text,
            system="Follow the user's exact response instruction. Do not use tools.",
            parameters=thaw_value(conversation.parameters),
        )
        return CompactRouteState(
            profile.profile_id,
            profile.target_id,
            compact,
            len(conversation.messages),
        )

    async def _try_terminal_handoff(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> PreparedResponse | None:
        policy = self._terminal_handoff
        if policy is None:
            return None
        compact = policy.compact_conversation(conversation)
        if compact is None:
            return None
        attempt = run.record_decision(DecisionDraft(
            gate="terminal_handoff",
            outcome="attempt",
            reason_code="verified_test_success_after_edit",
            selected_profile_id=self._easy.profile_id,
        ))
        try:
            candidate = await run.call_buffered(
                self._easy.call(
                    compact,
                    role="finalizer",
                    phase="terminal-handoff",
                    method_transform=policy.transform,
                ),
                caused_by=attempt,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            decision = run.record_decision(DecisionDraft(
                gate="terminal_handoff",
                outcome="fallback_hard",
                reason_code="terminal_handoff_error",
                selected_profile_id=self._hard.profile_id,
            ))
            return await self._plan_profile(
                conversation,
                run,
                self._hard,
                decision,
                phase="terminal-handoff-fallback",
                use_stored_compact=True,
            )
        accepted, reason = policy.assess(candidate)
        if accepted:
            decision = run.record_decision(DecisionDraft(
                gate="terminal_handoff",
                outcome="accept_easy",
                reason_code=reason,
                selected_profile_id=self._easy.profile_id,
                evidence_call_ids=(candidate.call_id,),
            ))
            _defer_recent_session_affinity(
                self._route_memory,
                run,
                self._easy,
            )
            return run.plan_replay(candidate, accepted_by=decision)
        decision = run.record_decision(DecisionDraft(
            gate="terminal_handoff",
            outcome="fallback_hard",
            reason_code=reason,
            selected_profile_id=self._hard.profile_id,
            evidence_call_ids=(candidate.call_id,),
        ))
        return await self._plan_profile(
            conversation,
            run,
            self._hard,
            decision,
            phase="terminal-handoff-fallback",
            use_stored_compact=True,
        )


class CascadeStrategyMethod:
    """Classify, buffer an easy candidate, and explicitly accept or escalate."""

    def __init__(
        self,
        *,
        classifier: StructuredDifficultyClassifier,
        candidate_policy: MarkerCandidatePolicy,
        easy: ModelProfile,
        hard: ModelProfile,
        route_memory: RouteMemory | None,
    ):
        if not isinstance(classifier, StructuredDifficultyClassifier):
            raise TypeError("classifier must be a StructuredDifficultyClassifier")
        if not isinstance(candidate_policy, MarkerCandidatePolicy):
            raise TypeError("candidate_policy must be a MarkerCandidatePolicy")
        if not isinstance(easy, ModelProfile) or not isinstance(hard, ModelProfile):
            raise TypeError("easy and hard must be ModelProfile values")
        if route_memory is not None and not isinstance(route_memory, RouteMemory):
            raise TypeError("route_memory must implement RouteMemory or be None")
        self._classifier = classifier
        self._candidate_policy = candidate_policy
        self._easy = easy
        self._hard = hard
        self._route_memory = route_memory

    def token_count_candidates(
        self,
        conversation: Conversation,
    ) -> tuple[ModelCallSpec, ...]:
        """Return every reachable generation shape without routing effects."""

        return (
            self._easy.call(
                conversation,
                role="generator",
                phase="initial-easy",
                method_transform=self._candidate_policy.candidate_transform,
            ),
            self._hard.call(
                conversation,
                role="writer",
                phase="initial-hard",
            ),
            self._hard.call(
                conversation,
                role="fixer",
                phase="escalation",
                method_transform=self._candidate_policy.escalation_transform,
            ),
        )

    async def respond(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> PreparedResponse:
        remembered = await _remembered_profile(
            self._route_memory,
            conversation,
            run,
            (self._easy, self._hard),
        )
        if remembered is not None:
            profile, affinity = remembered
            initial = _memory_decision(run, affinity)
            if profile.profile_id == self._hard.profile_id or affinity.locked:
                return run.plan_live(
                    profile.call(
                        conversation,
                        role=(
                            "writer"
                            if profile.profile_id == self._hard.profile_id
                            else "generator"
                        ),
                        phase=(
                            "remembered-hard"
                            if profile.profile_id == self._hard.profile_id
                            else "pinned-continuation"
                        ),
                    ),
                    caused_by=initial,
                )
            return await self._easy_candidate(conversation, run, initial)

        assessment = await self._classifier.classify(conversation, run)
        if assessment.difficulty == "hard":
            decision = self._difficulty_decision(
                run,
                assessment,
                self._hard.profile_id,
            )
            _defer_affinity(
                self._route_memory,
                conversation,
                run,
                self._hard,
                locked=False,
            )
            return run.plan_live(
                self._hard.call(
                    conversation,
                    role="writer",
                    phase="initial-hard",
                ),
                caused_by=decision,
            )

        initial = self._difficulty_decision(
            run,
            assessment,
            self._easy.profile_id,
        )
        return await self._easy_candidate(conversation, run, initial)

    async def _easy_candidate(
        self,
        conversation: Conversation,
        run: RunScope,
        initial: DecisionId,
    ) -> PreparedResponse:
        candidate = await run.call_buffered(
            self._easy.call(
                conversation,
                role="generator",
                phase="initial-easy",
                method_transform=self._candidate_policy.candidate_transform,
            ),
            caused_by=initial,
        )
        outcome, reason_code = self._candidate_policy.assess(candidate)
        selected = self._hard if outcome == "escalate" else self._easy
        candidate_decision = run.record_decision(DecisionDraft(
            gate="candidate",
            outcome=outcome,
            reason_code=reason_code,
            selected_profile_id=selected.profile_id,
            evidence_call_ids=(candidate.call_id,),
        ))
        if outcome == "accept":
            await _store_affinity(
                self._route_memory,
                conversation,
                run,
                self._easy,
                locked=candidate.has_tool_call,
            )
            return run.plan_replay(candidate, accepted_by=candidate_decision)
        _defer_affinity(
            self._route_memory,
            conversation,
            run,
            self._hard,
            locked=False,
        )
        return run.plan_live(
            self._hard.call(
                self._candidate_policy.escalation_conversation(
                    conversation,
                    candidate,
                ),
                role="fixer",
                phase="escalation",
            ),
            caused_by=candidate_decision,
        )

    @staticmethod
    def _difficulty_decision(
        run: RunScope,
        assessment: DifficultyAssessment,
        profile_id: str,
    ) -> DecisionId:
        return run.record_decision(DecisionDraft(
            gate="difficulty",
            outcome=assessment.route or assessment.difficulty,
            reason_code=assessment.reason_code,
            selected_profile_id=profile_id,
            evidence_call_ids=assessment.evidence_call_ids,
            confidence=assessment.confidence,
        ))


def _assistant_candidate_message(result: ModelCallResult) -> ConversationMessage:
    """Materialize provider-neutral output events as one assistant message."""

    blocks: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for event in result.events:
        index = event.data.get("index")
        if event.kind == "content_start" and isinstance(index, int):
            block = thaw_value(event.data.get("block", {}))
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                block["text"] = ""
            elif block_type == "thinking":
                block["thinking"] = ""
            elif block_type == "tool_call":
                block["arguments"] = {}
                block["_arguments_json"] = ""
            blocks[index] = block
            order.append(index)
            continue
        if event.kind != "content_delta" or not isinstance(index, int):
            continue
        block = blocks.get(index)
        delta = thaw_value(event.data.get("delta", {}))
        if block is None or not isinstance(delta, dict):
            continue
        text = delta.get("text")
        if isinstance(text, str):
            field = "thinking" if block.get("type") == "thinking" else "text"
            block[field] = str(block.get(field, "")) + text
        json_delta = delta.get("json")
        if isinstance(json_delta, str):
            block["_arguments_json"] = (
                str(block.get("_arguments_json", "")) + json_delta
            )
        if "arguments" in delta:
            block["arguments"] = thaw_value(delta["arguments"])

    materialized = []
    for index in order:
        block = blocks[index]
        raw_arguments = block.pop("_arguments_json", "")
        if raw_arguments:
            try:
                block["arguments"] = json.loads(raw_arguments)
            except json.JSONDecodeError:
                block["arguments"] = {"_raw": raw_arguments}
        materialized.append(freeze_value(block))
    if not materialized and result.text:
        materialized.append(freeze_value({"type": "text", "text": result.text}))
    return ConversationMessage(role="assistant", content=tuple(materialized))


def _text_conversation(text: str) -> Conversation:
    return Conversation(
        system=(),
        messages=(ConversationMessage(
            role="user",
            content=(freeze_value({"type": "text", "text": text}),),
        ),),
        tools=(),
        parameters={},
        extensions={},
    )


async def _remembered_profile(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
    profiles: tuple[ModelProfile, ...],
) -> tuple[ModelProfile, RouteAffinity] | None:
    if memory is None:
        return None
    affinity = await memory.get(conversation, run.metadata)
    if affinity is None:
        return None
    matches = [
        profile
        for profile in profiles
        if profile.profile_id == affinity.profile_id
        and profile.target_id == affinity.target_id
    ]
    if len(matches) != 1:
        raise RuntimeError("route memory returned an unknown compiled profile")
    return matches[0], affinity


async def _recent_session_profile(
    memory: RouteMemory | None,
    run: RunScope,
    profiles: tuple[ModelProfile, ...],
) -> ModelProfile | None:
    if memory is None:
        return None
    getter = getattr(memory, "get_recent_session_affinity", None)
    if not callable(getter):
        return None
    affinity = await getter(run.metadata)
    if affinity is None:
        return None
    matches = [
        profile
        for profile in profiles
        if profile.profile_id == affinity.profile_id
        and profile.target_id == affinity.target_id
    ]
    if len(matches) != 1:
        raise RuntimeError("session memory returned an unknown compiled profile")
    return matches[0]


async def _remembered_compact_state(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
    profile: ModelProfile,
) -> CompactRouteState | None:
    if memory is None:
        return None
    getter = getattr(memory, "get_compact_state", None)
    if not callable(getter):
        return None
    state = await getter(conversation, run.metadata)
    if state is None:
        return None
    if (
        state.profile_id != profile.profile_id
        or state.target_id != profile.target_id
    ):
        raise RuntimeError("compact state does not match remembered profile")
    return state


async def _recent_compact_state(
    memory: RouteMemory | None,
    run: RunScope,
    profile: ModelProfile,
) -> CompactRouteState | None:
    if memory is None:
        return None
    getter = getattr(memory, "get_recent_compact_state", None)
    if not callable(getter):
        return None
    state = await getter(run.metadata)
    if state is None:
        return None
    if (
        state.profile_id != profile.profile_id
        or state.target_id != profile.target_id
    ):
        return None
    return state


async def _store_affinity(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
    profile: ModelProfile,
    *,
    locked: bool,
) -> None:
    if memory is None:
        return
    await memory.put(
        conversation,
        run.metadata,
        RouteAffinity(profile.profile_id, profile.target_id, locked=locked),
    )


def _defer_affinity(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
    profile: ModelProfile,
    *,
    locked: bool,
) -> None:
    if memory is None:
        return

    async def commit() -> None:
        await _store_affinity(
            memory,
            conversation,
            run,
            profile,
            locked=locked,
        )

    run.defer_success(commit)


def _defer_recent_session_affinity(
    memory: RouteMemory | None,
    run: RunScope,
    profile: ModelProfile,
) -> None:
    if memory is None:
        return
    putter = getattr(memory, "put_recent_session_affinity", None)
    if not callable(putter):
        return

    async def commit() -> None:
        await putter(
            run.metadata,
            RouteAffinity(profile.profile_id, profile.target_id),
        )

    run.defer_success(commit)


def _defer_compact_state(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
    state: CompactRouteState,
) -> None:
    if memory is None:
        return
    putter = getattr(memory, "put_compact_state", None)
    if not callable(putter):
        return

    async def commit() -> None:
        await putter(conversation, run.metadata, state)

    run.defer_success(commit)


def _defer_clear_compact_state(
    memory: RouteMemory | None,
    conversation: Conversation,
    run: RunScope,
) -> None:
    if memory is None:
        return
    clearer = getattr(memory, "clear_compact_state", None)
    if not callable(clearer):
        return

    async def commit() -> None:
        await clearer(conversation, run.metadata)

    run.defer_success(commit)


def _latest_message_has_tool_result(conversation: Conversation) -> bool:
    latest = conversation.latest_user_message()
    return latest is not None and any(
        block.get("type") == "tool_result" for block in latest.content
    )


def _memory_decision(run: RunScope, affinity: RouteAffinity) -> DecisionId:
    return run.record_decision(DecisionDraft(
        gate="route-memory",
        outcome="locked" if affinity.locked else "hit",
        reason_code=(
            "route_memory_locked" if affinity.locked else "route_memory_hit"
        ),
        selected_profile_id=affinity.profile_id,
    ))
