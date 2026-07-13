import unittest

from smart_ask.benchmarks.runner import _run_one
from smart_ask.benchmarks.suite import BenchmarkCase, Evaluation
from smart_ask.conversation.domain import ConversationEvent
from smart_ask.conversation.engine import StrategyEngine
from smart_ask.conversation.model import DecisionDraft, ModelCallSpec


class _Suite:
    name = "engine-suite"

    def evaluate(self, case, output):
        return Evaluation(output == "answer", 1.0, {"task": case.task_id})


class _Strategy:
    digest = "a" * 64


class _Method:
    async def respond(self, conversation, run):
        decision = run.record_decision(DecisionDraft(
            gate="start",
            outcome="fixed",
            selected_profile_id="writer",
        ))
        return run.plan_live(ModelCallSpec(
            profile_id="writer",
            target_id="test-target",
            role="generation",
            phase="fixed",
            conversation=conversation,
        ), caused_by=decision)


class _Executor:
    async def stream(self, spec):
        assert spec.conversation.latest_human_instruction()[0] == "question"
        for event in (
            ConversationEvent("message_start", {
                "selected_model": "selected",
                "model": "actual",
            }),
            ConversationEvent("content_start", {
                "index": 0,
                "block": {"type": "text"},
            }),
            ConversationEvent("content_delta", {
                "index": 0,
                "delta": {"type": "text", "text": "answer"},
            }),
            ConversationEvent("content_stop", {"index": 0}),
            ConversationEvent("usage", {
                "input_tokens": 4,
                "output_tokens": 1,
                "reasoning_tokens": 0,
            }),
            ConversationEvent("message_delta", {"stop_reason": "stop"}),
            ConversationEvent("message_stop"),
        ):
            yield event


class BenchmarkStrategyEngineRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_serializes_engine_evidence_and_evaluates_visible_text(self):
        record = await _run_one(
            _Suite(),
            _Strategy(),
            "fixed",
            StrategyEngine(_Method(), _Executor()),
            BenchmarkCase("task-1", "question", {"group": "smoke"}),
        )

        self.assertEqual(record["output"]["text"], "answer")
        self.assertEqual(record["evaluation"]["passed"], True)
        self.assertEqual(record["final_call"], "call-1")
        self.assertEqual(record["decisions"][0]["gate"], "start")
        self.assertEqual(record["model_calls"][0]["target_id"], "test-target")
        self.assertEqual(record["provider_requests"][0]["input_tokens"], 4)
        self.assertEqual(record["run"]["metadata"]["request_id"], "task-1")


if __name__ == "__main__":
    unittest.main()
