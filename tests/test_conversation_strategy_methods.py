import asyncio
import unittest

from smart_ask.conversation.domain import ConversationEvent, ConversationMessage
from smart_ask.conversation.model import Conversation, ModelCallResult, RunMetadata
from smart_ask.methods.memory import (
    CompactRouteState,
    InMemoryRouteMemory,
    RouteAffinity,
    route_memory_key,
)
from smart_ask.methods.strategies import (
    CandidateToolCallError,
    CascadeStrategyMethod,
    CompactHandoffPolicy,
    DifficultyStrategyMethod,
    FixedStrategyMethod,
    MarkerCandidatePolicy,
    ModelProfile,
    RequestTransform,
    RoutingInputError,
    StructuredDifficultyClassifier,
    TerminalHandoffPolicy,
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
    prefilter="none",
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
        prefilter=prefilter,
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


def completed_coding_turn(
    output="5 passed in 0.04s",
    *,
    command=".venv/bin/python -m pytest tests/test_impl.py -q",
    include_edit=True,
    include_task=False,
):
    messages = [ConversationMessage(
        role="user",
        content=({"type": "text", "text": "Fix the cache bug and run tests."},),
    )]
    if include_task:
        messages.extend((
            ConversationMessage("assistant", ({
                "type": "tool_call", "id": "task-1", "name": "TaskCreate",
                "arguments": {"subject": "Fix cache"},
            },)),
            ConversationMessage("user", ({
                "type": "tool_result", "id": "task-1", "content": "Created",
            },)),
        ))
    if include_edit:
        messages.extend((
            ConversationMessage("assistant", ({
                "type": "tool_call", "id": "edit-1", "name": "Edit",
                "arguments": {
                    "file_path": "cache.py",
                    "old_string": "return None",
                    "new_string": "return self.value",
                },
            },)),
            ConversationMessage("user", ({
                "type": "tool_result", "id": "edit-1",
                "content": "The file has been updated successfully.",
            },)),
        ))
    messages.extend((
        ConversationMessage("assistant", ({
            "type": "tool_call", "id": "test-1", "name": "Bash",
            "arguments": {"command": command},
        },)),
        ConversationMessage("user", (
            {"type": "tool_result", "id": "test-1", "content": output},
            {"type": "text", "text": "<system-reminder>Be concise.</system-reminder>"},
        )),
    ))
    return Conversation(
        system=({"type": "text", "text": "caller system"},),
        messages=tuple(messages),
        tools=({"name": "Bash", "input_schema": {"type": "object"}},),
        parameters={},
    )


def terminal_policy():
    return TerminalHandoffPolicy(
        prompt="Summarize or output NEEDS_OPUS.",
        marker="NEEDS_OPUS",
        min_passed_tests=2,
        max_prompt_chars=6000,
        max_tokens=512,
    )


def compact_policy():
    return CompactHandoffPolicy(
        prompt="Summarize state or output HANDOFF_UNSAFE.",
        marker="HANDOFF_UNSAFE",
        max_summary_chars=2000,
        max_tool_result_chars=1000,
        max_tokens=256,
        tool_names=("read", "bash", "edit", "write"),
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

    def test_context_truncation_keeps_tool_call_for_leading_result(self):
        messages = (
            ConversationMessage("user", ({"type": "text", "text": "task"},)),
            ConversationMessage("assistant", ({"type": "text", "text": "read"},)),
            ConversationMessage("user", ({"type": "text", "text": "continue"},)),
            ConversationMessage("assistant", ({
                "type": "tool_call", "id": "call-1", "name": "read",
                "arguments": {},
            },)),
            ConversationMessage("system", ({
                "type": "text", "text": "harness context",
            },)),
            ConversationMessage("user", ({
                "type": "tool_result", "id": "call-1", "content": "file",
            },)),
            ConversationMessage("assistant", ({
                "type": "tool_call", "id": "call-2", "name": "grep",
                "arguments": {},
            },)),
            ConversationMessage("user", ({
                "type": "tool_result", "id": "call-2", "content": "match",
            },)),
        )
        original = Conversation(system=(), messages=messages)

        transformed = RequestTransform(keep_last_messages=4).apply(original)

        self.assertEqual(
            [message.role for message in transformed.messages],
            ["user", "assistant", "system", "user", "assistant", "user"],
        )
        self.assertEqual(
            transformed.messages[1].content[0]["id"],
            transformed.messages[3].content[0]["id"],
        )


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
    def test_classifier_parser_rejects_malformed_confidence_and_duplicate_keys(self):
        invalid = (
            '{"route":"sonnet"}',
            '{"route":"sonnet","confidence":true}',
            '{"route":"sonnet","confidence":1.1}',
            '{"route":"sonnet","confidence":0.9,"confidence":0.8}',
            '{"route":"easy","confidence":0.9}',
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    StructuredDifficultyClassifier._parse(payload)

    async def test_substantive_code_action_uses_confidence_classifier(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.93}',
        )])

        await method.respond(
            conversation(latest_text="Fix this issue in the repository."),
            run,
        )

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "easy")
        self.assertEqual(run.decisions[0].reason_code, "classifier_sonnet")
        self.assertEqual(run.decisions[0].confidence, 0.93)

    async def test_exact_reply_is_the_only_deterministic_route(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope()

        await method.respond(
            conversation(
                latest_text="Reply with exactly OK. Do not use tools.",
            ),
            run,
        )

        self.assertEqual(run.calls, [])
        self.assertEqual(run.live[0][0].profile_id, "easy")
        self.assertEqual(
            run.decisions[0].reason_code,
            "deterministic_exact_reply_sonnet",
        )

    async def test_simple_repo_query_uses_confidence_classifier(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.98}',
        )])

        await method.respond(
            conversation(latest_text="How many files are in this repo?"),
            run,
        )

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "easy")
        self.assertEqual(run.decisions[0].reason_code, "classifier_sonnet")

    async def test_summary_uses_confidence_classifier(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )

        for prompt in (
            "Summarize the completed change and verification in two bullets. "
            "Do not run tools or modify files.",
            "Briefly recap what changed.",
            "Provide me a concise summary of the test result.",
        ):
            with self.subTest(prompt=prompt):
                run = FakeRunScope([result(
                    "classification",
                    '{"route":"sonnet","confidence":0.96}',
                )])
                await method.respond(conversation(latest_text=prompt), run)
                self.assertEqual(len(run.calls), 1)
                self.assertEqual(run.live[0][0].profile_id, "easy")
                self.assertEqual(run.decisions[0].reason_code, "classifier_sonnet")

    async def test_classifier_routes_summary_then_edit_to_opus(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"route":"opus","confidence":0.91}',
        )])

        await method.respond(
            conversation(latest_text="Summarize the bug, then fix the code."),
            run,
        )

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(
            run.decisions[0].reason_code,
            "classifier_opus",
        )

    async def test_classifier_routes_inventory_then_edit_to_opus(self):
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"route":"opus","confidence":0.87}',
        )])

        await method.respond(
            conversation(
                latest_text="List the files, then edit config.py to enable debug mode.",
            ),
            run,
        )

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(
            run.decisions[0].reason_code,
            "classifier_opus",
        )

    async def test_agentic_prefilter_does_not_override_tool_result_policy(self):
        original = conversation(tool_continuation=True)
        method = DifficultyStrategyMethod(
            classifier=classifier(
                continuation="classify_tool_result",
                prefilter="exact_replies",
            ),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result("classification", '{"d":"easy"}')])

        await method.respond(original, run)

        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.live[0][0].profile_id, "easy")

    async def test_tool_projection_preserves_output_head_and_tail(self):
        base = Conversation.from_text("Report the schema version.")
        original = Conversation(
            system=base.system,
            messages=base.messages + (
                ConversationMessage("assistant", ({
                    "type": "tool_call",
                    "id": "read-1",
                    "name": "Read",
                    "arguments": {"file_path": "strategy.yaml"},
                },)),
                ConversationMessage("user", ({
                    "type": "tool_result",
                    "id": "read-1",
                    "content": "schema_version: 3\n" + "x" * 1000 + "\nEND_MARKER",
                },)),
            ),
        )
        method = DifficultyStrategyMethod(
            classifier=classifier(
                continuation="classify_tool_result",
                max_prompt_chars=400,
            ),
            easy=EASY,
            hard=HARD,
            route_memory=None,
        )
        run = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.98}',
        )])

        await method.respond(original, run)

        projected = run.calls[0][0].conversation.messages[0].content[0]["text"]
        self.assertLessEqual(len(projected), 400)
        self.assertIn("Report the schema version.", projected)
        self.assertIn("schema_version: 3", projected)
        self.assertIn("[middle omitted]", projected)
        self.assertIn("END_MARKER", projected)

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
        self.assertEqual(run.decisions[0].reason_code, "continuation_opus")

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
            "classifier_execution_fallback_opus",
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
            "classifier_invalid_fallback_opus",
        )

    def test_token_count_candidates_do_not_classify_or_read_memory(self):
        class FailingMemory:
            async def get(self, _conversation, _metadata):
                raise AssertionError("token counting read route memory")

            async def put(self, _conversation, _metadata, _affinity):
                raise AssertionError("token counting wrote route memory")

        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=FailingMemory(),
        )

        candidates = method.token_count_candidates(conversation())

        self.assertEqual(
            [(value.profile_id, value.phase) for value in candidates],
            [("easy", "initial-easy"), ("hard", "initial-hard")],
        )


class AdaptiveRoutingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def base_and_continuation(output="complex parser failure"):
        base = Conversation(
            system=({"type": "text", "text": "large caller system"},),
            messages=(ConversationMessage(
                "user",
                ({"type": "text", "text": "Fix the small parser issue."},),
            ),),
            tools=(
                {"name": "Read", "input_schema": {"type": "object"}},
                {"name": "Bash", "input_schema": {"type": "object"}},
                {"name": "Agent", "input_schema": {"type": "object"}},
            ),
            parameters={"max_tokens": 8192},
        )
        continuation = Conversation(
            system=base.system,
            messages=base.messages + (
                ConversationMessage("assistant", ({
                    "type": "tool_call",
                    "id": "read-1",
                    "name": "Read",
                    "arguments": {"file_path": "parser.py"},
                },)),
                ConversationMessage("user", ({
                    "type": "tool_result",
                    "id": "read-1",
                    "content": output,
                },)),
            ),
            tools=base.tools,
            parameters=base.parameters,
        )
        return base, continuation

    @staticmethod
    def method(memory):
        return DifficultyStrategyMethod(
            classifier=classifier(
                continuation="classify_tool_result",
                prefilter="exact_replies",
            ),
            easy=EASY,
            hard=HARD,
            route_memory=memory,
            compact_handoff=compact_policy(),
        )

    async def test_low_confidence_sonnet_routes_as_uncertain_to_opus(self):
        method = self.method(None)
        run = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.51}',
        )])

        await method.respond(Conversation.from_text("Fix a typo."), run)

        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.decisions[0].outcome, "uncertain")
        self.assertEqual(
            run.decisions[0].reason_code,
            "classifier_sonnet_below_confidence_threshold",
        )
        self.assertEqual(run.decisions[0].confidence, 0.51)

    async def test_explicit_uncertain_routes_to_opus_even_at_high_confidence(self):
        method = self.method(None)
        run = FakeRunScope([result(
            "classification",
            '{"route":"uncertain","confidence":0.96}',
        )])

        await method.respond(Conversation.from_text("Fix the unclear failure."), run)

        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.decisions[0].outcome, "uncertain")
        self.assertEqual(run.decisions[0].reason_code, "classifier_uncertain")
        self.assertEqual(run.decisions[0].confidence, 0.96)

    async def test_sonnet_tool_evidence_can_continue_without_handoff(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = self.method(memory)
        base, continuation = self.base_and_continuation(
            "The typo is localized to one constant in parser.py."
        )
        first = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.94}',
        )])
        await method.respond(base, first)
        await first.commit_success()
        second = FakeRunScope([
            result(
                "continuation-classifier",
                '{"route":"sonnet","confidence":0.92}',
            ),
        ], metadata=first.metadata)

        await method.respond(continuation, second)

        self.assertEqual([call[0].role for call in second.calls], ["classifier"])
        self.assertEqual(second.live[0][0].profile_id, "easy")
        self.assertEqual(second.live[0][0].phase, "continued-easy")
        self.assertEqual(second.live[0][0].conversation.messages, continuation.messages)

    async def test_sonnet_tool_evidence_escalates_through_compact_handoff(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = self.method(memory)
        base, continuation = self.base_and_continuation()
        first = FakeRunScope([result(
            "classification",
            '{"route":"sonnet","confidence":0.94}',
        )])
        await method.respond(base, first)
        await first.commit_success()

        second = FakeRunScope([
            result("continuation-classifier", '{"route":"opus","confidence":0.89}'),
            result(
                "handoff-summary",
                "Read parser.py. The failure reveals non-local AST behavior; inspect Parser.run.",
            ),
        ])
        await method.respond(continuation, second)

        self.assertEqual(
            [call[0].profile_id for call in second.calls],
            ["classifier", "easy"],
        )
        summary_spec = second.calls[1][0]
        self.assertEqual(summary_spec.conversation.tools, ())
        self.assertNotIn("thinking", summary_spec.conversation.parameters)
        selected = second.live[0][0]
        self.assertEqual(selected.profile_id, "hard")
        self.assertEqual(selected.phase, "adaptive-escalation")
        self.assertEqual(
            [tool["name"] for tool in selected.conversation.tools],
            ["Read", "Bash"],
        )
        compact_text = selected.conversation.messages[0].content[0]["text"]
        self.assertIn("ORIGINAL REQUEST", compact_text)
        self.assertIn("Parser.run", compact_text)
        self.assertIn("complex parser failure", compact_text)
        self.assertNotEqual(selected.conversation.system, continuation.system)

        await second.commit_success()
        third_conversation = Conversation(
            system=continuation.system,
            messages=continuation.messages + (
                ConversationMessage("assistant", ({
                    "type": "tool_call",
                    "id": "bash-1",
                    "name": "Bash",
                    "arguments": {"command": "pytest -q"},
                },)),
                ConversationMessage("user", ({
                    "type": "tool_result",
                    "id": "bash-1",
                    "content": "2 failed in 0.2s",
                },)),
            ),
            tools=continuation.tools,
            parameters=continuation.parameters,
        )
        third = FakeRunScope(metadata=second.metadata)
        await method.respond(third_conversation, third)

        self.assertEqual(third.calls, [])
        persisted = third.live[0][0].conversation
        self.assertEqual(persisted.system, selected.conversation.system)
        self.assertEqual(persisted.messages[0], selected.conversation.messages[0])
        self.assertEqual(
            persisted.messages[-1].content[0]["content"],
            "2 failed in 0.2s",
        )

    async def test_cross_message_model_switch_is_summarized_by_warm_model(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = self.method(memory)
        base = Conversation(
            system=({"type": "text", "text": "large system " * 1000},),
            messages=(ConversationMessage(
                "user",
                ({"type": "text", "text": "Design a concurrent scheduler."},),
            ),),
        )
        first = FakeRunScope([result(
            "classification",
            '{"route":"opus","confidence":0.97}',
        )])
        await method.respond(base, first)
        await first.commit_success()
        summary_request = Conversation(
            system=base.system,
            messages=base.messages + (
                ConversationMessage("assistant", ({
                    "type": "text", "text": "Implemented scheduler.py",
                },)),
                ConversationMessage("user", ({
                    "type": "text", "text": "Summarize the completed work.",
                },)),
            ),
        )
        second = FakeRunScope([
            result("classification-2", '{"route":"sonnet","confidence":0.99}'),
            result("handoff-summary", "scheduler.py now uses a bounded worker queue."),
        ], metadata=first.metadata)

        await method.respond(summary_request, second)

        self.assertEqual(
            [call[0].profile_id for call in second.calls],
            ["classifier", "hard"],
        )
        selected = second.live[0][0]
        self.assertEqual(selected.profile_id, "easy")
        self.assertLess(
            len(str(selected.conversation)),
            len(str(summary_request)),
        )
        self.assertEqual(second.decisions[-1].reason_code, "compact_handoff_summary_accepted")

        await second.commit_success()
        next_request = Conversation(
            system=summary_request.system,
            messages=summary_request.messages + (
                ConversationMessage("assistant", ({
                    "type": "text", "text": "Two-bullet summary",
                },)),
                ConversationMessage("user", ({
                    "type": "text", "text": "Now redesign the scheduler architecture.",
                },)),
            ),
        )
        third = FakeRunScope([
            result("classification-3", '{"route":"opus","confidence":0.96}'),
            result("handoff-summary-2", "A redesign is requested; no new edits yet."),
        ], metadata=second.metadata)

        await method.respond(next_request, third)

        warm_summary_request = third.calls[1][0].conversation
        self.assertEqual(warm_summary_request.system, selected.conversation.system)
        self.assertLess(
            len(str(warm_summary_request)),
            len(str(next_request)),
        )

    async def test_unsafe_summary_falls_back_to_full_context(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = self.method(memory)
        base = Conversation.from_text("Design a parser.")
        first = FakeRunScope([result(
            "classification",
            '{"route":"opus","confidence":0.98}',
        )])
        await method.respond(base, first)
        await first.commit_success()
        followup = Conversation(
            system=base.system,
            messages=base.messages + (
                ConversationMessage("assistant", ({"type": "text", "text": "done"},)),
                ConversationMessage("user", ({"type": "text", "text": "Explain it."},)),
            ),
        )
        second = FakeRunScope([
            result("classification-2", '{"route":"sonnet","confidence":0.97}'),
            result("handoff-summary", "HANDOFF_UNSAFE"),
        ], metadata=first.metadata)

        await method.respond(followup, second)

        self.assertEqual(second.live[0][0].conversation.messages, followup.messages)
        self.assertEqual(second.live[0][1], "decision-3")
        self.assertEqual(
            second.decisions[-1].reason_code,
            "compact_handoff_summarizer_unsafe",
        )


class TerminalHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def _method_and_memory(self, original):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        run = FakeRunScope()
        await memory.put(
            original,
            run.metadata,
            RouteAffinity("hard", "vendor-hard"),
        )
        method = DifficultyStrategyMethod(
            classifier=classifier(),
            easy=EASY,
            hard=HARD,
            route_memory=memory,
            terminal_handoff=terminal_policy(),
        )
        return method, run

    async def test_verified_edit_uses_compact_easy_finalizer(self):
        original = completed_coding_turn()
        method, run = await self._method_and_memory(original)
        run.buffered.append(result("finalizer", "Fixed cache.py; 5 tests pass."))

        prepared = await method.respond(original, run)

        self.assertIs(prepared, run.replay_response)
        self.assertEqual(len(run.calls), 1)
        candidate = run.calls[0][0]
        self.assertEqual(candidate.profile_id, "easy")
        self.assertEqual(candidate.phase, "terminal-handoff")
        self.assertEqual(candidate.role, "finalizer")
        self.assertEqual(candidate.conversation.tools, ())
        self.assertEqual(candidate.conversation.parameters["max_tokens"], 512)
        compact_text = candidate.conversation.messages[0].content[0]["text"]
        self.assertIn("Fix the cache bug", compact_text)
        self.assertIn("cache.py", compact_text)
        self.assertIn("5 passed", compact_text)
        self.assertNotIn("system-reminder", compact_text)
        self.assertEqual(run.live, [])
        self.assertEqual(run.decisions[-1].outcome, "accept_easy")

    async def test_uncertain_finalizer_falls_back_to_full_context_opus(self):
        original = completed_coding_turn()
        method, run = await self._method_and_memory(original)
        run.buffered.append(result("finalizer", "NEEDS_OPUS"))

        prepared = await method.respond(original, run)

        self.assertIs(prepared, run.live_response)
        fallback = run.live[0][0]
        self.assertEqual(fallback.profile_id, "hard")
        self.assertEqual(fallback.phase, "terminal-handoff-fallback")
        self.assertEqual(fallback.conversation.messages, original.messages)
        self.assertEqual(run.decisions[-1].reason_code, "terminal_handoff_requested_opus")

    async def test_ambiguous_evidence_stays_on_remembered_opus(self):
        cases = (
            completed_coding_turn("1 passed in 0.01s"),
            completed_coding_turn("4 passed, 1 failed in 0.04s"),
            completed_coding_turn("5 passed in 0.04s", include_edit=False),
            completed_coding_turn("5 passed in 0.04s", include_task=True),
            completed_coding_turn(
                "5 passed in 0.04s",
                command="python script.py",
            ),
        )
        for original in cases:
            with self.subTest(messages=len(original.messages)):
                method, run = await self._method_and_memory(original)
                await method.respond(original, run)
                self.assertEqual(run.calls, [])
                self.assertEqual(run.live[0][0].phase, "remembered-hard")

    async def test_finalizer_error_falls_back_and_cancellation_escapes(self):
        original = completed_coding_turn()
        method, run = await self._method_and_memory(original)
        run.buffered.append(RuntimeError("offline"))

        await method.respond(original, run)

        self.assertEqual(run.live[0][0].profile_id, "hard")
        self.assertEqual(run.decisions[-1].reason_code, "terminal_handoff_error")

        method, cancelled = await self._method_and_memory(original)
        cancelled.buffered.append(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await method.respond(original, cancelled)


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

    async def test_system_reminder_on_tool_result_is_not_a_new_instruction(self):
        base, continuation = self.tool_turn()
        reminder = ConversationMessage(
            role="user",
            content=(
                {
                    "type": "tool_result",
                    "tool_call_id": "call-1",
                    "content": "file contents",
                },
                {
                    "type": "text",
                    "text": (
                        "<system-reminder>Use the requested tool and run "
                        "tests.</system-reminder>"
                    ),
                },
            ),
        )
        with_reminder = Conversation(
            system=continuation.system,
            messages=continuation.messages[:-1] + (reminder,),
            tools=continuation.tools,
            parameters=continuation.parameters,
        )
        metadata = RunMetadata(
            "strategy",
            "digest",
            session_id="session",
        )

        self.assertEqual(
            route_memory_key(base, metadata),
            route_memory_key(with_reminder, metadata),
        )

    async def test_changing_reminder_on_original_prompt_does_not_change_key(self):
        metadata = RunMetadata(
            "strategy",
            "digest",
            session_id="session",
        )

        def prompted(reminder):
            return Conversation(
                system=(),
                messages=(ConversationMessage(
                    role="user",
                    content=(
                        {
                            "type": "text",
                            "text": f"<system-reminder>{reminder}</system-reminder>",
                        },
                        {"type": "text", "text": "fix the bug"},
                    ),
                ),),
            )

        self.assertEqual(
            route_memory_key(prompted("first metadata"), metadata),
            route_memory_key(prompted("updated metadata"), metadata),
        )

    async def test_cache_control_movement_does_not_change_key(self):
        metadata = RunMetadata(
            "strategy",
            "digest",
            session_id="session",
        )

        def prompted(cached):
            block = {"type": "text", "text": "fix the bug"}
            if cached:
                block["cache_control"] = {"type": "ephemeral"}
            return Conversation(
                system=(),
                messages=(ConversationMessage(
                    role="user",
                    content=(block,),
                ),),
            )

        self.assertEqual(
            route_memory_key(prompted(True), metadata),
            route_memory_key(prompted(False), metadata),
        )

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

    async def test_compact_state_expires_by_scope_and_can_be_cleared(self):
        now = [0.0]
        memory = InMemoryRouteMemory(
            ttl_seconds=60,
            session_ttl_seconds=10,
            max_entries=10,
            clock=lambda: now[0],
        )
        original = Conversation.from_text("Fix a typo.")
        metadata = RunMetadata("strategy", "digest", session_id="session")
        affinity = RouteAffinity("easy", "vendor-easy")
        state = CompactRouteState(
            "easy",
            "vendor-easy",
            Conversation.from_text("compact handoff"),
            1,
        )
        await memory.put(original, metadata, affinity)
        await memory.put_recent_session_affinity(metadata, affinity)
        await memory.put_compact_state(original, metadata, state)

        self.assertEqual(await memory.get_compact_state(original, metadata), state)
        self.assertEqual(await memory.get_recent_compact_state(metadata), state)
        now[0] = 11.0
        self.assertIsNone(await memory.get_recent_compact_state(metadata))
        self.assertEqual(await memory.get_compact_state(original, metadata), state)

        await memory.clear_compact_state(original, metadata)
        self.assertIsNone(await memory.get_compact_state(original, metadata))
        self.assertIsNone(await memory.get_recent_compact_state(metadata))

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

    async def test_new_human_message_is_reclassified_after_hard_turn(self):
        memory = InMemoryRouteMemory(ttl_seconds=60, max_entries=10)
        method = DifficultyStrategyMethod(
            classifier=classifier(prefilter="exact_replies"),
            easy=EASY,
            hard=HARD,
            route_memory=memory,
        )
        first_conversation = Conversation.from_text("Fix this code bug.")
        first = FakeRunScope([result("classification", '{"d":"hard"}')])
        await method.respond(first_conversation, first)
        await first.commit_success()
        second_conversation = Conversation(
            system=first_conversation.system,
            messages=first_conversation.messages + (
                ConversationMessage("assistant", ({
                    "type": "text", "text": "The fix is complete.",
                },)),
                ConversationMessage("user", ({
                    "type": "text", "text": "Summarize that briefly.",
                },)),
            ),
        )
        second = FakeRunScope(
            [result("classification-2", '{"route":"sonnet","confidence":0.97}')],
            metadata=first.metadata,
        )

        await method.respond(second_conversation, second)

        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.live[0][0].profile_id, "easy")
        self.assertEqual(second.live[0][0].phase, "initial-easy")
        self.assertEqual(second.decisions[0].gate, "difficulty")
        self.assertEqual(
            second.decisions[0].reason_code,
            "classifier_sonnet",
        )

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
