import unittest

from smart_ask import Conversation, InputTokenCount, RunMetadata, RunMetricsStore
from smart_ask.metrics import DEFAULT_PRICE_CATALOG, aggregate_resources
from smart_ask.conversation.domain import ConversationEvent
from smart_ask.conversation.engine import StrategyEngine
from smart_ask.methods import FixedStrategyMethod, ModelProfile, RequestTransform


class UsageExecutor:
    async def stream(self, spec):
        yield ConversationEvent("message_start", {
            "selected_model": "google/gemini-2.5-flash-lite",
            "model": "google/gemini-2.5-flash-lite",
        })
        yield ConversationEvent("content_start", {
            "index": 0,
            "block": {"type": "text"},
        })
        yield ConversationEvent("content_delta", {
            "index": 0,
            "delta": {"type": "text", "text": "answer"},
        })
        yield ConversationEvent("content_stop", {"index": 0})
        yield ConversationEvent("usage", {
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "cache_read_tokens": 10,
            "provider_cost_usd": 0.001,
        })
        yield ConversationEvent("message_delta", {"stop_reason": "stop"})
        yield ConversationEvent("message_stop")

    async def count_tokens(self, _spec):
        return InputTokenCount(100, "exact")


async def completed_run(session="session-1"):
    method = FixedStrategyMethod(
        profile=ModelProfile("writer", "gemini-target"),
        role="writer",
        transform=RequestTransform(),
    )
    return await StrategyEngine(method, UsageExecutor()).complete(
        Conversation.from_text("hello"),
        RunMetadata("strategy", "digest", session_id=session),
    )


class RunMetricsTests(unittest.IsolatedAsyncioTestCase):
    def test_requested_model_fallback_remains_priceable(self):
        resources = aggregate_resources(
            [{
                "call_id": "call-1",
                "status": "completed",
                "target_id": "target",
                "selected_model": "google/gemini-2.5-flash-lite",
                "actual_model": None,
                "input_tokens": 100,
                "output_tokens": 20,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "provider_cost_usd": None,
                "output_status": "usable",
                "duration_ms": 10.0,
                "time_to_first_output_ms": 2.0,
                "tool_call_count": 0,
            }],
            [{"call_id": "call-1", "profile_id": "easy", "role": "writer"}],
            price_catalog=DEFAULT_PRICE_CATALOG,
        )

        self.assertGreater(resources["overall"]["known_cost_usd"], 0)
        self.assertIn(
            "requested:google/gemini-2.5-flash-lite",
            resources["by_model"],
        )

    async def test_records_calls_tokens_cost_and_dimensions_once(self):
        completed = await completed_run()
        store = RunMetricsStore()

        envelope = store.record(completed.record)

        session = envelope["session"]
        self.assertEqual(session["runs"], 1)
        resources = session["resources"]
        self.assertEqual(resources["overall"]["requests"], 1)
        self.assertEqual(resources["overall"]["successful_requests"], 1)
        self.assertEqual(resources["overall"]["known_total_tokens"], 120)
        self.assertEqual(resources["overall"]["known_input_tokens"], 100)
        self.assertEqual(resources["overall"]["known_output_tokens"], 20)
        self.assertEqual(resources["overall"]["known_reasoning_tokens"], 5)
        self.assertEqual(resources["overall"]["known_cache_read_tokens"], 10)
        self.assertEqual(
            resources["by_target"]["gemini-target"]["known_total_tokens"],
            120,
        )
        self.assertEqual(
            resources["by_model"]["google/gemini-2.5-flash-lite"][
                "requests"
            ],
            1,
        )
        self.assertEqual(resources["by_profile"]["writer"]["requests"], 1)
        self.assertEqual(resources["by_role"]["writer"]["requests"], 1)
        self.assertGreater(resources["overall"]["known_cost_usd"], 0)
        self.assertEqual(
            resources["overall"]["cost_sources"],
            {"provider": 1},
        )

    async def test_aggregates_multiple_invocations_into_one_session(self):
        store = RunMetricsStore()
        store.record((await completed_run()).record)
        store.record((await completed_run()).record)

        session = store.sessions["session-1"]
        self.assertEqual(session["runs"], 2)
        self.assertEqual(
            session["resources"]["overall"]["known_total_tokens"],
            240,
        )

    async def test_retention_is_bounded_and_sink_only_mode_keeps_no_runs(self):
        store = RunMetricsStore(max_records=1, max_sessions=1)
        store.record((await completed_run("one")).record)
        store.record((await completed_run("two")).record)
        self.assertEqual(len(store.records), 1)
        self.assertEqual(list(store.sessions), ["two"])

        sink_only = RunMetricsStore(max_records=0, max_sessions=0)
        sink_only.record((await completed_run("three")).record)
        self.assertEqual(sink_only.records, ())
        self.assertEqual(sink_only.sessions, {})

    async def test_default_record_is_content_free(self):
        completed = await completed_run()
        record = RunMetricsStore().record(completed.record)["run"]

        rendered = repr(record)
        self.assertNotIn("hello", rendered)
        self.assertNotIn("answer", rendered)
        self.assertIn("reasoning_tokens", rendered)

    async def test_sink_failure_does_not_change_run_evidence(self):
        def fail(_value):
            raise RuntimeError("disk full")

        store = RunMetricsStore(sink=fail)
        completed = await completed_run()
        envelope = store.record(completed.record)

        self.assertEqual(envelope["run"]["status"], "completed")
        self.assertEqual(store.sink_errors, ("RuntimeError: disk full",))


if __name__ == "__main__":
    unittest.main()
