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
from numbers import Integral
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
from .memory import RouteAffinity, RouteMemory


Difficulty = Literal["easy", "hard"]
ClassifierFallback = Literal["easy", "hard", "raise"]
RoutingProjection = Literal["latest_user_text", "full_conversation"]
ContinuationPolicy = Literal[
    "raise",
    "route_easy",
    "route_hard",
    "classify_latest_human_instruction",
    "classify_full_conversation",
]
ToolCallPolicy = Literal["accept_and_pin", "escalate", "raise"]


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

    def apply(self, conversation: Conversation) -> Conversation:
        """Apply transforms without flattening unrelated structured content."""

        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        transformed = conversation
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
        return RequestTransform(
            system_suffix=self.system_suffix + other.system_suffix,
            latest_user_prefix=(
                other.latest_user_prefix + self.latest_user_prefix
            ),
            latest_user_suffix=(
                self.latest_user_suffix + other.latest_user_suffix
            ),
            parameters=parameters,
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

    def __post_init__(self) -> None:
        if self.difficulty not in ("easy", "hard"):
            raise ValueError("difficulty must be easy or hard")
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
        parameters: Mapping[str, Any] | None = None,
    ):
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
        ):
            raise ValueError("unknown continuation policy")
        if fallback not in ("easy", "hard", "raise"):
            raise ValueError("fallback must be easy, hard, or raise")
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

        try:
            projected = self._project(conversation, self._projection)
        except RoutingInputError:
            continuation = self._continuation
            if continuation == "raise":
                raise
            if continuation in ("route_easy", "route_hard"):
                difficulty = cast(Difficulty, continuation.removeprefix("route_"))
                return DifficultyAssessment(
                    difficulty,
                    f"continuation_{difficulty}",
                )
            if continuation == "classify_latest_human_instruction":
                projected = self._project_latest_human_instruction(conversation)
            else:
                projected = self._project(conversation, "full_conversation")

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
            fallback = cast(Difficulty, self._fallback)
            return DifficultyAssessment(
                fallback,
                f"classifier_execution_fallback_{fallback}",
            )

        try:
            if result.output_status != "usable":
                raise ValueError(
                    f"classifier output was {result.output_status}"
                )
            difficulty = self._parse(result.text)
        except ValueError:
            if self._fallback == "raise":
                raise
            fallback = cast(Difficulty, self._fallback)
            return DifficultyAssessment(
                fallback,
                f"classifier_invalid_fallback_{fallback}",
                (result.call_id,),
            )
        return DifficultyAssessment(
            difficulty,
            f"classifier_{difficulty}",
            (result.call_id,),
        )

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
        if latest is None or latest is not conversation.messages[-1]:
            raise RoutingInputError(
                "conversation does not end in a user message; continuation policy is required"
            )
        texts = [
            block.get("text")
            for block in latest.content
            if block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block.get("text").strip()
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
        text = projected[0]
        if self._max_prompt_chars is not None:
            text = text[: self._max_prompt_chars]
        return _text_conversation(text)

    @staticmethod
    def _parse(text: str) -> Difficulty:
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
        if not isinstance(payload, dict) or set(payload) != {"d"}:
            raise ValueError("classifier object must contain exactly 'd'")
        difficulty = payload["d"]
        if difficulty not in ("easy", "hard"):
            raise ValueError("classifier label must be easy or hard")
        return cast(Difficulty, difficulty)


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
    """Classify a conversation, then prepare one easy or hard live call."""

    def __init__(
        self,
        *,
        classifier: StructuredDifficultyClassifier,
        easy: ModelProfile,
        hard: ModelProfile,
        route_memory: RouteMemory | None,
    ):
        if not isinstance(classifier, StructuredDifficultyClassifier):
            raise TypeError("classifier must be a StructuredDifficultyClassifier")
        if not isinstance(easy, ModelProfile) or not isinstance(hard, ModelProfile):
            raise TypeError("easy and hard must be ModelProfile values")
        if route_memory is not None and not isinstance(route_memory, RouteMemory):
            raise TypeError("route_memory must implement RouteMemory or be None")
        self._classifier = classifier
        self._easy = easy
        self._hard = hard
        self._route_memory = route_memory

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
            decision = _memory_decision(run, affinity)
            return run.plan_live(
                profile.call(
                    conversation,
                    role="writer" if hard else "generator",
                    phase="remembered-hard" if hard else "remembered-easy",
                ),
                caused_by=decision,
            )

        assessment = await self._classifier.classify(conversation, run)
        hard = assessment.difficulty == "hard"
        profile = self._hard if hard else self._easy
        decision = run.record_decision(DecisionDraft(
            gate="difficulty",
            outcome=assessment.difficulty,
            reason_code=assessment.reason_code,
            selected_profile_id=profile.profile_id,
            evidence_call_ids=assessment.evidence_call_ids,
        ))
        _defer_affinity(
            self._route_memory,
            conversation,
            run,
            profile,
            locked=False,
        )
        return run.plan_live(
            profile.call(
                conversation,
                role="writer" if hard else "generator",
                phase="initial-hard" if hard else "initial-easy",
            ),
            caused_by=decision,
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
            outcome=assessment.difficulty,
            reason_code=assessment.reason_code,
            selected_profile_id=profile_id,
            evidence_call_ids=assessment.evidence_call_ids,
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


def _memory_decision(run: RunScope, affinity: RouteAffinity) -> DecisionId:
    return run.record_decision(DecisionDraft(
        gate="route-memory",
        outcome="locked" if affinity.locked else "hit",
        reason_code=(
            "route_memory_locked" if affinity.locked else "route_memory_hit"
        ),
        selected_profile_id=affinity.profile_id,
    ))
