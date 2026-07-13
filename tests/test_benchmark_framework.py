import tempfile
from pathlib import Path
import unittest

from smart_ask.benchmarks import BenchmarkCase, Evaluation, run_matrix
from smart_ask.benchmarks.artifacts import JsonlResultSink, MemoryResultSink, load_run
from smart_ask.conversation.domain import ConversationEvent
from smart_ask.conversation.engine import StrategyEngine
from smart_ask.conversation.model import DecisionDraft, ModelCallSpec
from smart_ask.strategy import load_strategy


class Suite:
    name = "tiny"
    dataset_identity = {"name": "tiny", "version": "1"}
    evaluator_identity = {"name": "exact", "version": "1"}

    def load_cases(self, limit=None):
        values = [BenchmarkCase("one", "one"), BenchmarkCase("two", "two")]
        return values if limit is None else values[:limit]

    def evaluate(self, case, output):
        return Evaluation(output == case.prompt, float(output == case.prompt))


class FixedMethod:
    async def respond(self, conversation, run):
        decision = run.record_decision(DecisionDraft(
            gate="start",
            outcome="fixed",
            selected_profile_id="model",
        ))
        return run.plan_live(ModelCallSpec(
            profile_id="model",
            target_id="test-target",
            role="writer",
            conversation=conversation,
        ), caused_by=decision)

    def token_count_candidates(self, conversation):
        return (ModelCallSpec("model", "test-target", "writer", conversation),)


class EchoExecutor:
    async def stream(self, spec):
        text = spec.conversation.latest_human_instruction()[0]
        for event in (
            ConversationEvent("message_start", {
                "selected_model": "test/model",
                "model": "test/model",
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
                "input_tokens": 2,
                "output_tokens": 1,
                "provider_cost_usd": 0.001,
            }),
            ConversationEvent("message_delta", {"stop_reason": "stop"}),
            ConversationEvent("message_stop"),
        ):
            yield event


class FailingExecutor:
    async def stream(self, _spec):
        yield ConversationEvent("message_start", {
            "selected_model": "test/model",
            "model": "test/model",
        })
        raise RuntimeError("provider exploded")


class ClosableExecutor(EchoExecutor):
    def __init__(self, closed):
        self.closed = closed

    async def aclose(self):
        self.closed.append(self)


class BenchmarkFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.strategy = load_strategy("builtin:local-qwen")

    @staticmethod
    def engine_factory(_strategy):
        return StrategyEngine(FixedMethod(), EchoExecutor())

    def test_matrix_uses_canonical_ledger_and_derives_summary(self):
        sink = MemoryResultSink()
        result = run_matrix(
            Suite(),
            (self.strategy,),
            engine_factory=self.engine_factory,
            sink=sink,
            workers=2,
        )

        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0]["schema"], "smart-ask.benchmark-result/v2")
        self.assertEqual(result.records[0]["final_call"], "call-1")
        summary = result.summaries[self.strategy.config.name]
        self.assertEqual(summary["outcomes"]["passed"], 2)
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual(summary["resources"]["overall"]["requests"], 2)
        self.assertEqual(
            summary["resources"]["overall"]["known_total_tokens"],
            6,
        )
        self.assertEqual(
            result.manifest["strategies"][0]["deployment"]["status"],
            "unresolved",
        )

    def test_manifest_records_resolved_target_fingerprints(self):
        result = run_matrix(
            Suite(),
            (self.strategy,),
            engine_factory=self.engine_factory,
            sink=MemoryResultSink(),
            limit=1,
            deployment_manifest_factory=lambda _strategy: {
                "status": "resolved",
                "digest": "f" * 64,
                "targets": [{
                    "target_id": "test-target",
                    "configuration_digest": "e" * 64,
                }],
            },
        )

        deployment = result.manifest["strategies"][0]["deployment"]
        self.assertEqual(deployment["status"], "resolved")
        self.assertEqual(deployment["digest"], "f" * 64)
        self.assertEqual(
            deployment["targets"][0]["configuration_digest"],
            "e" * 64,
        )

    def test_execution_failure_preserves_run_ledger(self):
        result = run_matrix(
            Suite(),
            (self.strategy,),
            engine_factory=lambda _strategy: StrategyEngine(
                FixedMethod(),
                FailingExecutor(),
            ),
            sink=MemoryResultSink(),
            limit=1,
        )

        record = result.records[0]
        self.assertEqual(record["error"]["stage"], "execution")
        self.assertEqual(record["run"]["status"], "error")
        self.assertEqual(record["model_calls"][0]["status"], "error")
        self.assertEqual(record["provider_requests"][0]["status"], "error")

    def test_matrix_closes_every_case_engine(self):
        closed = []
        result = run_matrix(
            Suite(),
            (self.strategy,),
            engine_factory=lambda _strategy: StrategyEngine(
                FixedMethod(),
                ClosableExecutor(closed),
            ),
            sink=MemoryResultSink(),
        )

        self.assertEqual(len(result.records), 2)
        self.assertEqual(len(closed), 2)

    def test_jsonl_sink_persists_and_loads_v2_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary, "run")
            run_matrix(
                Suite(),
                (self.strategy,),
                engine_factory=self.engine_factory,
                sink=JsonlResultSink(directory),
                limit=1,
            )

            loaded = load_run(directory)
            self.assertEqual(loaded["manifest"]["schema"], "smart-ask.benchmark-run/v2")
            self.assertEqual(len(loaded["records"]), 1)
            self.assertEqual(
                loaded["summary"]["schema"],
                "smart-ask.benchmark-summary/v2",
            )

    def test_resume_skips_completed_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary, "run")
            run_matrix(
                Suite(),
                (self.strategy,),
                engine_factory=self.engine_factory,
                sink=JsonlResultSink(directory),
                limit=1,
            )
            result = run_matrix(
                Suite(),
                (self.strategy,),
                engine_factory=self.engine_factory,
                sink=JsonlResultSink(directory, resume=True),
                limit=1,
            )
            self.assertEqual(len(result.records), 1)


if __name__ == "__main__":
    unittest.main()
