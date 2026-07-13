import asyncio
import unittest

from smart_ask.conversation.domain import ConversationEvent
from smart_ask.conversation.engine import (
    BufferedResponseLimitExceeded,
    RunDeadlineExceeded,
    StrategyEngine,
    TokenCountUnavailable,
)
from smart_ask.conversation.model import (
    Conversation,
    DecisionDraft,
    InputTokenCount,
    ModelCallSpec,
    RunMetadata,
)


def text_events(text, model="actual-model", selected_model="selected-model"):
    return (
        ConversationEvent("message_start", {
            "model": model,
            "selected_model": selected_model,
        }),
        ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "text"},
        }),
        ConversationEvent("content_delta", {
            "index": 0,
            "delta": {"type": "text", "text": text},
        }),
        ConversationEvent("content_stop", {"index": 0}),
        ConversationEvent("usage", {
            "input_tokens": 10,
            "output_tokens": 2,
            "reasoning_tokens": 1,
        }),
        ConversationEvent("message_delta", {"stop_reason": "stop"}),
        ConversationEvent("message_stop"),
    )


class FakeExecutor:
    def __init__(self, responses, *, delay=0, token_counts=None):
        self.responses = list(responses)
        self.requests = []
        self.token_requests = []
        self.delay = delay
        self.token_counts = [
            None if value is None else (
                value if isinstance(value, InputTokenCount)
                else InputTokenCount(value, "exact")
            )
            for value in (token_counts or [])
        ]

    async def stream(self, request):
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        for event in self.responses.pop(0):
            yield event

    async def count_tokens(self, request):
        self.token_requests.append(request)
        return self.token_counts.pop(0) if self.token_counts else None


def spec(conversation, profile, target, phase):
    return ModelCallSpec(
        profile_id=profile,
        target_id=target,
        role="generation",
        phase=phase,
        conversation=conversation,
    )


def metadata():
    return RunMetadata("test-strategy", "digest", session_id="session-1")


class FixedMethod:
    async def respond(self, conversation, run):
        decision = run.record_decision(DecisionDraft(
            gate="start",
            outcome="fixed",
            selected_profile_id="writer",
        ))
        return run.plan_live(
            spec(conversation, "writer", "selected-model", "fixed"),
            caused_by=decision,
        )

    def token_count_candidates(self, conversation):
        return (spec(conversation, "writer", "selected-model", "fixed"),)


class CascadeMethod:
    async def respond(self, conversation, run):
        start = run.record_decision(DecisionDraft(
            gate="difficulty",
            outcome="easy",
            selected_profile_id="cheap",
        ))
        cheap = await run.call_buffered(
            spec(conversation, "cheap", "cheap-model", "initial-easy"),
            caused_by=start,
        )
        if "ESCALATE" not in cheap.text:
            accepted = run.record_decision(DecisionDraft(
                gate="candidate",
                outcome="accept",
                selected_profile_id="cheap",
                evidence_call_ids=(cheap.call_id,),
            ))
            return run.plan_replay(cheap, accepted_by=accepted)
        escalated = run.record_decision(DecisionDraft(
            gate="candidate",
            outcome="escalate",
            selected_profile_id="hard",
            evidence_call_ids=(cheap.call_id,),
        ))
        return run.plan_live(
            spec(conversation, "hard", "hard-model", "escalation"),
            caused_by=escalated,
        )

    def token_count_candidates(self, conversation):
        return (
            spec(conversation, "cheap", "cheap-model", "initial-easy"),
            spec(conversation, "hard", "hard-model", "initial-hard"),
            spec(conversation, "hard", "hard-model", "escalation"),
        )


class StrategyEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_record_does_not_retain_raw_provider_error(self):
        class FailingExecutor(FakeExecutor):
            async def stream(self, _request):
                raise RuntimeError("secret prompt fragment")
                yield

        engine = StrategyEngine(FixedMethod(), FailingExecutor([]))
        handle = engine.start(Conversation.from_text("hello"), metadata())
        with self.assertRaises(Exception):
            _events = [event async for event in handle.events()]

        record = await handle.result()
        self.assertNotIn("secret prompt fragment", repr(record))
        self.assertEqual(record.provider_requests[0].error, "RuntimeError")

    async def test_total_run_deadline_covers_live_generation(self):
        executor = FakeExecutor([text_events("late")], delay=0.1)
        engine = StrategyEngine(
            FixedMethod(),
            executor,
            heartbeat_seconds=0.005,
            deadline_seconds=0.02,
        )
        handle = engine.start(Conversation.from_text("hello"), metadata())

        with self.assertRaises(RunDeadlineExceeded):
            _events = [event async for event in handle.events()]
        record = await handle.result()
        self.assertEqual(record.status, "error")
        self.assertEqual(record.error, "RunDeadlineExceeded")

    async def test_fixed_live_response_records_causality_and_usage(self):
        executor = FakeExecutor([text_events("answer")])
        completed = await StrategyEngine(FixedMethod(), executor).complete(
            Conversation.from_text("hello"),
            metadata(),
        )

        self.assertEqual(completed.events[-1].kind, "message_stop")
        self.assertEqual(completed.record.status, "completed")
        self.assertEqual(completed.record.final_call_id, "call-1")
        self.assertEqual(len(completed.record.decisions), 1)
        call = completed.record.model_calls[0]
        self.assertEqual(call.caused_by_decision_id, "decision-1")
        provider = completed.record.provider_requests[0]
        self.assertEqual(provider.input_tokens, 10)
        self.assertEqual(provider.output_tokens, 2)
        self.assertEqual(provider.reasoning_tokens, 1)
        self.assertEqual(provider.output_status, "usable")
        self.assertIsNotNone(provider.time_to_first_output_ms)

    async def test_accepted_buffer_is_replayed_without_second_charge(self):
        executor = FakeExecutor([text_events("good draft", "cheap-actual")])
        completed = await StrategyEngine(CascadeMethod(), executor).complete(
            Conversation.from_text("write code"),
            metadata(),
        )

        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(len(completed.record.model_calls), 1)
        self.assertEqual(len(completed.record.provider_requests), 1)
        self.assertEqual(completed.record.final_call_id, "call-1")
        self.assertEqual(completed.record.final_decision_id, "decision-2")
        visible = "".join(
            event.data["delta"]["text"]
            for event in completed.events
            if event.kind == "content_delta"
        )
        self.assertEqual(visible, "good draft")

    async def test_rejected_buffer_never_leaks_and_hard_response_is_live(self):
        executor = FakeExecutor([
            text_events("bad ESCALATE", "cheap-actual"),
            text_events("correct", "hard-actual"),
        ])
        completed = await StrategyEngine(CascadeMethod(), executor).complete(
            Conversation.from_text("write code"),
            metadata(),
        )

        visible = "".join(
            event.data["delta"]["text"]
            for event in completed.events
            if event.kind == "content_delta"
        )
        self.assertEqual(visible, "correct")
        self.assertEqual(completed.record.final_call_id, "call-2")
        self.assertEqual(
            [call.target_id for call in completed.record.model_calls],
            ["cheap-model", "hard-model"],
        )
        self.assertEqual(
            [request.target_id for request in completed.record.provider_requests],
            ["cheap-model", "hard-model"],
        )

    async def test_heartbeats_cover_hidden_preparation(self):
        executor = FakeExecutor([text_events("accepted")], delay=0.03)
        completed = await StrategyEngine(
            CascadeMethod(),
            executor,
            heartbeat_seconds=0.005,
        ).complete(Conversation.from_text("hello"), metadata())

        self.assertTrue(any(event.kind == "heartbeat" for event in completed.events))
        self.assertEqual(completed.record.status, "completed")

    async def test_reasoning_without_visible_answer_is_recorded_as_empty(self):
        reasoning_only = (
            ConversationEvent("message_start", {
                "model": "actual-model",
                "selected_model": "selected-model",
            }),
            ConversationEvent("content_start", {
                "index": 0,
                "block": {"type": "thinking"},
            }),
            ConversationEvent("content_delta", {
                "index": 0,
                "delta": {"type": "thinking", "text": "hidden"},
            }),
            ConversationEvent("content_stop", {"index": 0}),
            ConversationEvent("usage", {
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 5,
            }),
            ConversationEvent("message_delta", {"stop_reason": "stop"}),
            ConversationEvent("message_stop"),
        )
        completed = await StrategyEngine(
            FixedMethod(),
            FakeExecutor([reasoning_only]),
        ).complete(Conversation.from_text("hello"), metadata())

        provider = completed.record.provider_requests[0]
        self.assertEqual(provider.output_status, "empty")
        self.assertEqual(provider.visible_text_chars, 0)
        self.assertEqual(provider.reasoning_tokens, 5)

    async def test_token_count_is_exact_for_one_pure_candidate(self):
        executor = FakeExecutor([], token_counts=[321])
        engine = StrategyEngine(FixedMethod(), executor)

        count = await engine.count_tokens(Conversation.from_text("hello"))

        self.assertEqual(count.value, 321)
        self.assertEqual(count.provenance, "exact")
        self.assertEqual(count.candidate_count, 1)
        self.assertEqual(len(executor.token_requests), 1)
        self.assertEqual(executor.requests, [])

    async def test_token_count_returns_maximum_as_upper_bound(self):
        executor = FakeExecutor([], token_counts=[100, 250, 180])
        engine = StrategyEngine(CascadeMethod(), executor)

        count = await engine.count_tokens(Conversation.from_text("hello"))

        self.assertEqual(count.value, 250)
        self.assertEqual(count.provenance, "upper_bound")
        self.assertEqual(count.candidate_count, 3)
        self.assertEqual(executor.requests, [])

    async def test_token_count_has_no_character_fallback(self):
        engine = StrategyEngine(
            FixedMethod(),
            FakeExecutor([], token_counts=[None]),
        )

        with self.assertRaises(TokenCountUnavailable):
            await engine.count_tokens(Conversation.from_text("hello"))

    async def test_cancellation_reaches_buffered_provider_call(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingExecutor(FakeExecutor):
            async def stream(self, request):
                self.requests.append(request)
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                if False:
                    yield ConversationEvent("message_stop")

        engine = StrategyEngine(CascadeMethod(), BlockingExecutor([]))
        handle = engine.start(Conversation.from_text("hello"), metadata())

        async def consume():
            return [event async for event in handle.events()]

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(cancelled.is_set())
        record = await handle.result()
        self.assertEqual(record.status, "cancelled")
        self.assertEqual(record.model_calls[0].status, "cancelled")
        self.assertEqual(record.provider_requests[0].status, "cancelled")

    async def test_buffer_limit_fails_run_without_exposing_candidate(self):
        executor = FakeExecutor([text_events("too large")])
        engine = StrategyEngine(
            CascadeMethod(),
            executor,
            max_buffer_bytes=1,
        )
        handle = engine.start(Conversation.from_text("hello"), metadata())

        with self.assertRaises(BufferedResponseLimitExceeded):
            _events = [event async for event in handle.events()]

        record = await handle.result()
        self.assertEqual(record.status, "error")
        self.assertEqual(record.model_calls[0].status, "error")

    async def test_prepared_response_is_scope_bound_and_single_use(self):
        captured = []

        class CapturingMethod(FixedMethod):
            async def respond(self, conversation, run):
                response = await super().respond(conversation, run)
                captured.append(response)
                return response

        engine = StrategyEngine(CapturingMethod(), FakeExecutor([text_events("ok")]))
        _completed = await engine.complete(Conversation.from_text("hello"), metadata())

        with self.assertRaises(RuntimeError):
            captured[0]._consume(object())


if __name__ == "__main__":
    unittest.main()
