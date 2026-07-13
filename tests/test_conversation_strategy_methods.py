import asyncio
import unittest

from smart_ask.conversation.domain import ConversationEvent, ConversationMessage
from smart_ask.conversation.model import Conversation, ModelCallResult, RunMetadata
from smart_ask.methods.memory import (
    InMemoryRouteMemory,
    RouteAffinity,
    route_memory_key,
)
from smart_ask.methods.strategies import (
    CandidateToolCallError,
    CascadeStrategyMethod,
    DifficultyStrategyMethod,
    FixedStrategyMethod,
    MarkerCandidatePolicy,
    ModelProfile,
    RequestTransform,
    RoutingInputError,
    StructuredDifficultyClassifier,
)


def conversation(*, latest_text="write code", tool_continuation=False):
    messages = [
        ConversationMessage(
            role="user",
            content=({"type": "text", "text": "earlier question"},),
        ),
        ConversationMessage(
            role="assistant",
            content=({
                "type": "tool_call",
                "id": "call-1",
                "name": "read",
                "arguments": {"path": "a.py"},
            },),
        ),
    ]
    if tool_continuation:
        messages.append(ConversationMessage(
            role="user",
            content=({
                "type": "tool_result",
                "tool_call_id": "call-1",
                "content": "contents",
            },),
        ))
    else:
        messages.append(ConversationMessage(
            role="user",
            content=(
                {"type": "text", "text": latest_text},
                {"type": "image", "source": {"id": "image-1"}},
            ),
        ))
    return Conversation(
        system=({"type": "text", "text": "caller system"},),
        messages=tuple(messages),
        tools=({"name": "read", "input_schema": {"type": "object"}},),
        parameters={"temperature": 0.7},
        extensions={"provider": {"feature": True}},
    )


def result(
    call_id,
    text,
    *,
    tool_call_count=0,
    stop_reason="stop",
    output_status="usable",
):
    return ModelCallResult(
        call_id=call_id,
        events=(ConversationEvent("message_stop", {"stop_reason": stop_reason}),),
        selected_model="selected/model",
        actual_model="actual/model",
        text=text,
        stop_reason=stop_reason,
        input_tokens=10,
        output_tokens=3,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        provider_cost_usd=None,
        tool_call_count=tool_call_count,
        stream_complete=True,
        output_status=output_status,
        duration_ms=1.0,
    )


class FakeRunScope:
    def __init__(self, buffered=(), *, metadata=None):
        self.buffered = list(buffered)
        self.metadata = metadata or RunMetadata(
            "strategy",
            "strategy-digest",
            session_id="session-1",
        )
        self.calls = []
        self.decisions = []
        self.live = []
        self.replays = []
        self.success_operations = []
        self.live_response = object()
        self.replay_response = object()

    async def call_buffered(self, spec, *, caused_by=None):
        self.calls.append((spec, caused_by))
        value = self.buffered.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def defer_success(self, operation):
        self.success_operations.append(operation)

    async def commit_success(self):
        for operation in self.success_operations:
            await operation()

    def record_decision(self, draft):
        self.decisions.append(draft)
        return f"decision-{len(self.decisions)}"

    def plan_live(self, spec, *, caused_by):
        self.live.append((spec, caused_by))
        return self.live_response

    def plan_replay(self, value, *, accepted_by):
        self.replays.append((value, accepted_by))
        return self.replay_response


def classifier(
    *,
    projection="latest_user_text",
    continuation="raise",
    fallback="raise",
    max_prompt_chars=1200,
):
    return StructuredDifficultyClassifier(
        profile=ModelProfile(
            "classifier",
            "vendor-classifier",
            RequestTransform(parameters={"temperature": 0}),
        ),
        prompt="Return exactly {\"d\": \"easy\"} or {\"d\": \"hard\"}.",
        projection=projection,
        continuation=continuation,
        fallback=fallback,
        max_prompt_chars=max_prompt_chars,
        parameters={"max_tokens": 20},
    )


EASY = ModelProfile(
    "easy",
    "vendor-easy",
    RequestTransform(
        system_suffix=("easy system",),
        parameters={"max_tokens": 100},
    ),
)
HARD = ModelProfile(
    "hard",
    "vendor-hard",
    RequestTransform(
        system_suffix=("hard system",),
        parameters={"max_tokens": 200},
    ),
)


class RequestTransformTests(unittest.TestCase):
    def test_transform_preserves_structured_history_tools_and_original(self):
        original = conversation()
        transformed = RequestTransform(
            system_suffix=("strategy system",),
            latest_user_prefix="BEFORE\n",
            latest_user_suffix="\nAFTER",
            parameters={"temperature": 0, "max_tokens": 123},
        ).apply(original)

        self.assertEqual(original.system[-1]["text"], "caller system")
        self.assertEqual(transformed.system[-1]["text"], "strategy system")
        self.assertEqual(transformed.tools, original.tools)
        self.assertEqual(
            transformed.messages[-1].content[2],
            original.messages[-1].content[1],
        )
        self.assertEqual(transformed.messages[-1].content[0]["text"], "BEFORE\n")
        self.assertEqual(transformed.messages[-1].content[-1]["text"], "\nAFTER")
        self.assertEqual(dict(transformed.parameters), {
            "temperature": 0,
            "max_tokens": 123,
        })

    def test_profile_and_method_transforms_compose_explicitly(self):
        spec = ModelProfile(
            "profile",
            "vendor-model",
            RequestTransform(
                latest_user_prefix="PROFILE-BEFORE ",
                latest_user_suffix=" PROFILE-AFTER",
                parameters={"max_tokens": 10, "temperature": 0.5},
            ),
        ).call(
            conversation(latest_text="task"),
            role="generator",
            phase="test",
            method_transform=RequestTransform(
                latest_user_prefix="METHOD-BEFORE ",
                latest_user_suffix=" METHOD-AFTER",
                parameters={"max_tokens": 20},
            ),
        )

        texts = [
            block["text"]
            for block in spec.conversation.messages[-1].content
            if block["type"] == "text"
        ]
        self.assertEqual(texts, [
            "METHOD-BEFORE PROFILE-BEFORE ",
            "task",
            " PROFILE-AFTER METHOD-AFTER",
        ])
        self.assertEqual(spec.conversation.parameters["max_tokens"], 20)
        self.assertEqual(spec.conversation.parameters["temperature"], 0.5)


class FixedStrategyMethodTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_prepares_one_live_call_with_full_conversation(self):
        original = conversation()
        run = FakeRunScope()
        method = FixedStrategyMethod(
            profile=EASY,
            role="generator",
            transform=RequestTransform(latest_user_suffix=" self-check"),
        )

        prepared = await method.respond(original, run)

        self.assertIs(prepared, run.live_response)
        self.assertEqual(run.calls, [])
        self.assertEqual(len(run.live), 1)
        spec, decision_id = run.live[0]
        self.assertEqual(spec.profile_id, "easy")
        self.assertEqual(spec.target_id, "vendor-easy")
        self.assertEqual(spec.conversation.messages[:-1], original.messages[:-1])
        self.assertEqual(spec.conversation.tools, original.tools)
        self.assertEqual(spec.conversation.messages[-1].content[-1]["text"], " self-check")
        self.assertEqual(decision_id, "decision-1")
        self.assertEqual(run.decisions[0].outcome, "fixed")

    def test_token_count_candidate_uses_the_fixed_transform(self):
        method = FixedStrategyMethod(
            profile=EASY,
            role="generator",
            transform=RequestTransform(latest_user_suffix=" fixed suffix"),
        )

        candidates = method.token_count_candidates(conversation())

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].phase, "fixed")
        self.assertEqual(
            candidates[0].conversation.messages[-1].content[-1]["text"],
            " fixed suffix",
        )


class DifficultyStrategyMethodTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_call_and_generation_share_the_run_scope(self):
        original = conversation(latest_text="abcdefgh")
        run = FakeRunScope([result("classification", '{"d":"hard"}')])
        method = DifficultyStrategyMethod(
            classifier=classifier(max_prompt_chars=5),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )

        prepared = await method.respond(original, run)

        self.assertIs(prepared, run.live_response)
        classifier_spec, caused_by = run.calls[0]
        self.assertIsNone(caused_by)
        self.assertEqual(classifier_spec.role, "classifier")
        self.assertEqual(classifier_spec.conversation.tools, ())
        self.assertEqual(
            classifier_spec.conversation.messages[0].content[0]["text"],
            "abcde",
        )
        generation_spec, decision_id = run.live[0]
        self.assertEqual(generation_spec.profile_id, "hard")
        self.assertEqual(generation_spec.conversation.messages, original.messages)
        self.assertEqual(generation_spec.conversation.tools, original.tools)
        self.assertEqual(decision_id, "decision-1")
        self.assertEqual(
            run.decisions[0].evidence_call_ids,
            ("classification",),
        )

    async def test_tool_only_continuation_has_no_magic_prompt(self):
        original = conversation(tool_continuation=True)
        method = DifficultyStrategyMethod(
            classifier=classifier(continuation="raise"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope()

        with self.assertRaisesRegex(RoutingInputError, "continuation policy"):
            await method.respond(original, run)

        self.assertEqual(run.calls, [])
        self.assertEqual(run.decisions, [])

    async def test_continuation_can_explicitly_route_without_classification(self):
        original = conversation(tool_continuation=True)
        method = DifficultyStrategyMethod(
            classifier=classifier(continuation="route_hard"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope()

        await method.respond(original, run)

        self.assertEqual(run.calls, [])
        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.decisions[0].reason_code, "continuation_hard")

    async def test_full_conversation_projection_is_not_silently_truncated(self):
        original = conversation(tool_continuation=True)
        method = DifficultyStrategyMethod(
            classifier=classifier(
                projection="full_conversation",
                continuation="raise",
                max_prompt_chars=None,
            ),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result("classification", '{"d":"easy"}')])

        await method.respond(original, run)

        classifier_request = run.calls[0][0].conversation
        self.assertEqual(classifier_request.messages, original.messages)
        self.assertEqual(classifier_request.tools, original.tools)

    async def test_classifier_failure_policy_is_explicit_and_cancellation_escapes(self):
        fallback_method = DifficultyStrategyMethod(
            classifier=classifier(fallback="hard"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        fallback_run = FakeRunScope([RuntimeError("offline")])

        await fallback_method.respond(conversation(), fallback_run)

        self.assertEqual(fallback_run.live[0][0].profile_id, "hard")
        self.assertEqual(
            fallback_run.decisions[0].reason_code,
            "classifier_execution_fallback_hard",
        )

        cancelled_run = FakeRunScope([asyncio.CancelledError()])
        with self.assertRaises(asyncio.CancelledError):
            await fallback_method.respond(conversation(), cancelled_run)

    async def test_unusable_classifier_output_uses_the_declared_fallback(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(fallback="hard"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"d":"easy"}',
            stop_reason="length",
            output_status="truncated",
        )])

        await method.respond(conversation(), run)

        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(
            run.decisions[0].reason_code,
            "classifier_invalid_fallback_hard",
        )

    def test_token_count_candidates_do_not_classify_or_read_memory(self):
        class FailingMemory:
            async def get(self, _conversation, _metadata):
                raise AssertionError("token counting read route memory")

            async def put(self, _conversation, _metadata, _affinity):
                raise AssertionError("token counting wrote route memory")

        method = DifficultyStrategyMethod(
            classifier=classifier(),
            easy=EASY,
            hard=HARD,
            route_memory=FailingMemory(),
        )

        candidates = method.token_count_candidates(conversation())

        self.assertEqual(
            [(value.profile_id, value.phase) for value in candidates],
            [("easy", "initial-easy"), ("hard", "initial-hard")],
        )


class CascadeStrategyMethodTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def method(*, tool_calls="accept_and_pin", route_memory=None):
        return CascadeStrategyMethod(
            classifier=classifier(),
            candidate_policy=MarkerCandidatePolicy(
                marker="ESCALATE_NOW",
                self_check_suffix="\nCheck; emit ESCALATE_NOW on failure.",
                escalation_prefix="Repair this request:\n",
                tool_calls=tool_calls,
            ),
            easy=EASY,
            hard=HARD,
            route_memory=route_memory,
        )

    async def test_accepted_candidate_is_replayed_without_another_call(self):
        candidate = result("candidate", "looks good")
        run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            candidate,
        ])

        prepared = await self.method().respond(conversation(), run)

        self.assertIs(prepared, run.replay_response)
        self.assertEqual(len(run.calls), 2)
        self.assertEqual(run.live, [])
        self.assertEqual(run.replays, [(candidate, "decision-2")])
        candidate_spec, caused_by = run.calls[1]
        self.assertEqual(candidate_spec.profile_id, "easy")
        self.assertEqual(caused_by, "decision-1")
        self.assertEqual(
            candidate_spec.conversation.messages[-1].content[-1]["text"],
            "\nCheck; emit ESCALATE_NOW on failure.",
        )
        self.assertEqual(run.decisions[1].outcome, "accept")
        self.assertEqual(run.decisions[1].evidence_call_ids, ("candidate",))

    async def test_marker_escalates_to_a_live_hard_call(self):
        run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            result("candidate", "draft\nESCALATE_NOW\n"),
        ])

        prepared = await self.method().respond(conversation(), run)

        self.assertIs(prepared, run.live_response)
        hard_spec, caused_by = run.live[0]
        self.assertEqual(hard_spec.profile_id, "hard")
        self.assertEqual(hard_spec.phase, "escalation")
        self.assertEqual(caused_by, "decision-2")
        self.assertEqual(
            hard_spec.conversation.messages[-2].role,
            "assistant",
        )
        self.assertEqual(
            hard_spec.conversation.messages[-2].content[0]["text"],
            "draft\nESCALATE_NOW\n",
        )
        self.assertEqual(
            hard_spec.conversation.messages[-1].content[0]["text"],
            "Repair this request:\n",
        )
        self.assertEqual(run.decisions[1].outcome, "escalate")

    async def test_hard_classification_skips_the_easy_candidate(self):
        run = FakeRunScope([result("classification", '{"d":"hard"}')])

        await self.method().respond(conversation(), run)

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.live[0][0].phase, "initial-hard")

    async def test_empty_candidate_escalates_instead_of_being_accepted(self):
        run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            result("candidate", "", output_status="empty"),
        ])

        await self.method().respond(conversation(), run)

        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.decisions[1].reason_code, "candidate_empty")

    async def test_candidate_tool_call_policy_is_never_implicit(self):
        candidate = result(
            "candidate",
            "",
            tool_call_count=1,
            stop_reason="tool_call",
        )
        accept_run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            candidate,
        ])
        await self.method(tool_calls="accept_and_pin").respond(
            conversation(),
            accept_run,
        )
        self.assertEqual(
            accept_run.decisions[1].reason_code,
            "candidate_tool_call_accept_and_pin",
        )
        self.assertEqual(len(accept_run.replays), 1)

        escalate_run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            candidate,
        ])
        await self.method(tool_calls="escalate").respond(
            conversation(),
            escalate_run,
        )
        self.assertEqual(escalate_run.decisions[1].reason_code, "candidate_tool_call_escalate")
        self.assertEqual(escalate_run.live[0][0].profile_id, "hard")

        raise_run = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            candidate,
        ])
        with self.assertRaises(CandidateToolCallError):
            await self.method(tool_calls="raise").respond(
                conversation(),
                raise_run,
            )
        self.assertEqual(len(raise_run.calls), 2)
        self.assertEqual(raise_run.live, [])
        self.assertEqual(raise_run.replays, [])

    def test_token_count_candidates_cover_all_transformed_generation_shapes(self):
        class FailingMemory:
            async def get(self, _conversation, _metadata):
                raise AssertionError("token counting read route memory")

            async def put(self, _conversation, _metadata, _affinity):
                raise AssertionError("token counting wrote route memory")

        original = conversation(latest_text="task")
        method = self.method(route_memory=FailingMemory())

        candidates = method.token_count_candidates(original)

        self.assertEqual(
            [(value.profile_id, value.phase) for value in candidates],
            [
                ("easy", "initial-easy"),
                ("hard", "initial-hard"),
                ("hard", "escalation"),
            ],
        )
        self.assertEqual(
            candidates[0].conversation.messages[-1].content[-1]["text"],
            "\nCheck; emit ESCALATE_NOW on failure.",
        )
        self.assertEqual(
            candidates[1].conversation.messages,
            original.messages,
        )
        self.assertEqual(
            candidates[2].conversation.messages[-1].content[0]["text"],
            "Repair this request:\n",
        )


class RouteMemoryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def tool_turn():
        base = Conversation(
            system=({"type": "text", "text": "system"},),
            messages=(ConversationMessage(
                role="user",
                content=({"type": "text", "text": "read a.py"},),
            ),),
            tools=({"name": "read", "input_schema": {"type": "object"}},),
            parameters={"thinking": {"type": "enabled", "budget_tokens": 32}},
        )
        continuation = Conversation(
            system=base.system,
            messages=base.messages + (
                ConversationMessage(
                    role="assistant",
                    content=({
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "read",
                        "arguments": {"path": "a.py"},
                    },),
                ),
                ConversationMessage(
                    role="user",
                    content=({
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "content": "file contents",
                    },),
                ),
            ),
            tools=base.tools,
            parameters=base.parameters,
        )
        return base, continuation

    async def test_key_is_stable_through_tool_traffic_but_scoped(self):
        base, continuation = self.tool_turn()
        metadata = RunMetadata(
            "strategy",
            "digest-a",
            session_id="session-1",
            agent_id="agent-1",
            extensions={"principal_id": "user-1"},
        )

        self.assertEqual(
            route_memory_key(base, metadata),
            route_memory_key(continuation, metadata),
        )
        self.assertNotEqual(
            route_memory_key(base, metadata),
            route_memory_key(base, RunMetadata(
                "strategy",
                "digest-b",
                session_id="session-1",
                agent_id="agent-1",
                extensions={"principal_id": "user-1"},
            )),
        )
        self.assertIsNone(route_memory_key(
            base,
            RunMetadata("strategy", "digest-a"),
        ))

    async def test_memory_expires_evicts_and_does_not_unlock_a_pin(self):
        now = [0.0]
        memory = InMemoryRouteMemory(
            ttl_seconds=10,
            max_entries=1,
            clock=lambda: now[0],
        )
        first, _ = self.tool_turn()
        second = Conversation.from_text("another turn")
        metadata = RunMetadata("strategy", "digest", session_id="session")
        pinned = RouteAffinity("easy", "vendor-easy", locked=True)
        await memory.put(first, metadata, pinned)
        await memory.put(
            first,
            metadata,
            RouteAffinity("hard", "vendor-hard", locked=False),
        )
        self.assertEqual(await memory.get(first, metadata), pinned)

        await memory.put(
            second,
            metadata,
            RouteAffinity("hard", "vendor-hard"),
        )
        self.assertIsNone(await memory.get(first, metadata))
        now[0] = 11.0
        self.assertIsNone(await memory.get(second, metadata))

    async def test_difficulty_reuses_the_remembered_profile(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = DifficultyStrategyMethod(
            classifier=classifier(),
            easy=EASY,
            hard=HARD,
            route_memory=memory,
        )
        original = Conversation.from_text("task")
        first = FakeRunScope([result("classification", '{"d":"easy"}')])
        await method.respond(original, first)
        await first.commit_success()

        second = FakeRunScope()
        await method.respond(original, second)

        self.assertEqual(second.calls, [])
        self.assertEqual(second.live[0][0].profile_id, "easy")
        self.assertEqual(second.decisions[0].gate, "route-memory")
        self.assertEqual(second.decisions[0].outcome, "hit")

    async def test_accepted_tool_call_pins_direct_continuation(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = CascadeStrategyMethodTests.method(route_memory=memory)
        base, continuation = self.tool_turn()
        first = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            result(
                "candidate",
                "",
                tool_call_count=1,
                stop_reason="tool_call",
            ),
        ])
        await method.respond(base, first)

        second = FakeRunScope()
        await method.respond(continuation, second)

        self.assertEqual(second.calls, [])
        self.assertEqual(second.live[0][0].profile_id, "easy")
        self.assertEqual(second.live[0][0].phase, "pinned-continuation")
        self.assertEqual(second.decisions[0].outcome, "locked")
        self.assertEqual(
            second.live[0][0].conversation.messages,
            continuation.messages,
        )

    async def test_escalation_updates_the_remembered_profile(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = CascadeStrategyMethodTests.method(route_memory=memory)
        original = Conversation.from_text("task")
        first = FakeRunScope([
            result("classification", '{"d":"easy"}'),
            result("candidate", "ESCALATE_NOW"),
        ])
        await method.respond(original, first)
        await first.commit_success()

        second = FakeRunScope()
        await method.respond(original, second)

        self.assertEqual(second.calls, [])
        self.assertEqual(second.live[0][0].profile_id, "hard")
        self.assertEqual(second.live[0][0].phase, "remembered-hard")

    async def test_failed_live_route_does_not_poison_memory(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = DifficultyStrategyMethod(
            classifier=classifier(),
            easy=EASY,
            hard=HARD,
            route_memory=memory,
        )
        original = Conversation.from_text("task")
        failed = FakeRunScope([result("classification", '{"d":"hard"}')])

        await method.respond(original, failed)

        retry = FakeRunScope([result("classification-2", '{"d":"easy"}')])
        await method.respond(original, retry)
        self.assertEqual(retry.calls[0][0].profile_id, "classifier")


if __name__ == "__main__":
    unittest.main()
