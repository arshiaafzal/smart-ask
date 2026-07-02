from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.artifacts import JsonlResultSink, MemoryResultSink, load_run
from benchmarks.cli import run_suite_cli
from benchmarks.runner import TracedExecutor, _code_identity, run_matrix
from benchmarks.suite import BenchmarkCase, Evaluation
from smart_ask import (
    Attempt,
    ExecutionRequest,
    ModelResult,
    RouteResult,
    RoutingEvent,
    RunResult,
    StrategyBuilder,
    load_strategy,
)


def _usage(prompt=10, completion=2):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


@dataclass(frozen=True)
class _Config:
    name: str


class _Loaded:
    def __init__(self, name):
        self.config = _Config(name)
        self.path = Path(f"{name}.yaml")
        self.digest = f"digest-{name}"

    def manifest(self):
        return {
            "name": self.config.name,
            "path": str(self.path),
            "digest": self.digest,
        }


class _Suite:
    name = "fake-suite"
    dataset_identity = {"dataset": "fake", "revision": "1"}

    def load_cases(self, limit=None):
        cases = [
            BenchmarkCase("task-1", "first"),
            BenchmarkCase("task-2", "second"),
        ]
        return cases if limit is None else cases[:limit]

    def evaluate(self, case, output):
        passed = output.startswith("pass")
        return Evaluation(passed, 1.0 if passed else 0.0, {"case": case.task_id})


class _Executor:
    captures_output = True

    def __init__(self, text):
        self.text = text

    def execute(self, request):
        return ModelResult(
            request.model,
            self.text,
            usage=_usage(),
            raw_text=self.text,
        )


class _Application:
    def __init__(self, classifier, generation):
        self.classifier = classifier
        self.generation = generation
        self.executor = generation

    def run_detailed(self, task, on_route=None):
        classification = self.classifier.execute(
            ExecutionRequest("classifier-model", task.prompt, max_tokens=20)
        )
        event = RoutingEvent(
            "difficulty-classifier",
            "easy",
            "fake classification",
            model=classification.model,
            role="classifier",
            usage=classification.usage,
        )
        route = RouteResult(
            action="execute",
            model="generation-model",
            prompt=task.prompt,
            role="generator",
            phase="initial-easy",
            label="easy primary",
            routing_events=(event,),
        )
        if on_route:
            on_route(route, 1)
        result = self.generation.execute(
            ExecutionRequest(route.model, route.prompt)
        )
        return RunResult(task, (Attempt(route, result),), (event,))


class _SequenceExecutor:
    captures_output = True

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def execute(self, request):
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResult(
            request.model,
            outcome,
            usage=_usage(),
            raw_text=outcome,
        )


class _PartialFailureApplication:
    def __init__(self, generation):
        self.generation = generation
        self.executor = generation

    def run_detailed(self, task, on_route=None):
        initial = RouteResult(
            action="execute",
            model="generation-model",
            prompt=task.prompt,
            phase="initial-easy",
            label="easy primary",
        )
        if on_route:
            on_route(initial, 1)
        self.generation.execute(ExecutionRequest(initial.model, initial.prompt))

        escalation = RouteResult(
            action="execute",
            model="generation-model",
            prompt=f"retry: {task.prompt}",
            phase="escalation",
            label="hard retry",
        )
        if on_route:
            on_route(escalation, 2)
        self.generation.execute(
            ExecutionRequest(escalation.model, escalation.prompt)
        )
        raise AssertionError("the second executor call should fail")


def _application_factory(loaded, recorder):
    text = "pass answer" if loaded.config.name == "passing" else "wrong answer"
    classifier = TracedExecutor(_Executor('{"d":"easy"}'), recorder, "classifier")
    generation = TracedExecutor(_Executor(text), recorder, "generation")
    return _Application(classifier, generation)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_matrix_retains_calls_attempts_outputs_cost_latency_and_comparison(self):
        sink = MemoryResultSink()
        result = run_matrix(
            _Suite(),
            [_Loaded("passing"), _Loaded("failing")],
            application_factory=_application_factory,
            sink=sink,
            workers=4,
            price_catalog={
                "classifier-model": {"input": 0.1, "output": 0.2},
                "generation-model": {"input": 0.3, "output": 0.4},
            },
        )

        self.assertEqual(len(result.records), 4)
        record = next(
            item for item in result.records
            if item["strategy_id"] == "passing" and item["task_id"] == "task-1"
        )
        self.assertEqual(record["route"], "initial-easy")
        self.assertEqual(record["classifier_decision"], "easy")
        self.assertEqual([call["channel"] for call in record["calls"]], [
            "classifier", "generation",
        ])
        self.assertEqual(record["attempts"][0]["output"]["text"], "pass answer")
        self.assertEqual(record["final_output"]["raw_text"], "pass answer")
        self.assertEqual(record["usage"]["total_tokens"], 24)
        self.assertAlmostEqual(record["cost_usd"], 5.2)
        self.assertIsNotNone(record["total_latency_ms"])
        self.assertIsNotNone(record["calls"][0]["latency_ms"])
        self.assertEqual(len(result.manifest["cases"]), 2)
        self.assertTrue(result.manifest["case_digest"])
        self.assertEqual(
            set(result.manifest["runtime"]["platform"]),
            {"system", "release", "machine", "implementation"},
        )
        self.assertEqual(
            set(result.manifest["runtime"]["dependencies"]),
            {"datasets", "openai", "pydantic", "PyYAML"},
        )
        self.assertIn("dirty_hash", result.manifest["runtime"]["code"])

        pair = result.comparison["pairs"][0]
        self.assertEqual(pair["only_reference_passes"], 2)
        self.assertEqual(pair["only_candidate_passes"], 0)
        self.assertEqual(pair["missing_tasks"], 0)

    def test_real_strategy_models_require_prices_before_execution(self):
        loaded = load_strategy(
            "strategies/python-function-completion-difficulty-v1.yaml"
        )

        with self.assertRaisesRegex(ValueError, "Missing benchmark prices"):
            run_matrix(
                _Suite(),
                [loaded],
                application_factory=lambda *_args: self.fail("must not build"),
                sink=MemoryResultSink(),
                price_catalog={},
            )

    def test_jsonl_sink_resumes_and_legacy_reader_marks_missing_fields(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            manifest = {
                "benchmark": "fake",
                "dataset": {"revision": "1"},
                "strategies": [{"name": "one"}],
                "case_ids": ["task"],
            }
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)
            sink.append({"strategy_id": "one", "task_id": "task"})
            sink.finalize({}, {})

            resumed = JsonlResultSink(run_dir, resume=True)
            resumed.start(manifest)
            self.assertEqual(resumed.completed_keys, {("one", "task")})
            self.assertEqual(len(load_run(run_dir)["records"]), 1)

            legacy_path = Path(directory) / "results_product.json"
            legacy_path.write_text(json.dumps({
                "results": [{
                    "question_id": "1234567890abcdef-full-id",
                    "difficulty": "medium",
                    "gate1": "hard",
                    "model": "opus-G1",
                    "passed": 2,
                    "total": 2,
                    "pass_all": True,
                }],
                "token_log": {"calls": [{
                    "task_id": "1234567890abcdef",
                    "model": "generation-model",
                    "role": "writer",
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "cost_usd": 0.5,
                }]},
            }))
            legacy = load_run(legacy_path)["records"][0]
            self.assertEqual(legacy["route"], "initial-hard")
            self.assertEqual(legacy["classifier_decision"], "hard")
            self.assertEqual(legacy["cost_usd"], 0.5)
            self.assertIn("latency", legacy["legacy_missing"])

    def test_resume_rejects_worker_or_runtime_changes(self):
        with TemporaryDirectory() as directory:
            manifest = {
                "benchmark": "fake",
                "dataset": {"revision": "1"},
                "strategies": [{"name": "one"}],
                "case_ids": ["task"],
                "case_digest": "cases",
                "pricing": {"model": {"input": 1, "output": 2}},
                "workers": 2,
                "runtime": {
                    "python": "3.test",
                    "platform": {"system": "test"},
                    "dependencies": {"package": "1"},
                    "code": {"git_commit": "abc", "dirty": False},
                },
            }
            run_dir = Path(directory) / "run"
            JsonlResultSink(run_dir).start(manifest)

            with self.assertRaisesRegex(ValueError, "'workers'"):
                JsonlResultSink(run_dir, resume=True).start({
                    **manifest,
                    "workers": 3,
                })
            with self.assertRaisesRegex(ValueError, "'runtime'"):
                JsonlResultSink(run_dir, resume=True).start({
                    **manifest,
                    "runtime": {**manifest["runtime"], "python": "3.changed"},
                })

    def test_resume_repairs_only_a_torn_final_jsonl_record(self):
        with TemporaryDirectory() as directory:
            manifest = {
                "benchmark": "fake",
                "dataset": {"revision": "1"},
                "strategies": [{"name": "one"}],
                "case_ids": ["task-1", "task-2"],
            }
            run_dir = Path(directory) / "recoverable"
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)
            sink.append({"strategy_id": "one", "task_id": "task-1"})
            records_path = run_dir / "records.jsonl"
            complete_bytes = records_path.read_bytes()
            with records_path.open("ab") as handle:
                handle.write(b'{"strategy_id":"one","task_id":"task-2"')

            resumed = JsonlResultSink(run_dir, resume=True)
            resumed.start(manifest)
            self.assertEqual(records_path.read_bytes(), complete_bytes)
            self.assertEqual(resumed.completed_keys, {("one", "task-1")})
            resumed.append({"strategy_id": "one", "task_id": "task-2"})
            self.assertEqual(len(load_run(run_dir)["records"]), 2)

            no_newline_dir = Path(directory) / "complete-without-newline"
            no_newline = JsonlResultSink(no_newline_dir)
            no_newline.start(manifest)
            no_newline.append({"strategy_id": "one", "task_id": "task-1"})
            no_newline_records = no_newline_dir / "records.jsonl"
            no_newline_records.write_bytes(
                no_newline_records.read_bytes().removesuffix(b"\n")
            )
            resumed_complete = JsonlResultSink(no_newline_dir, resume=True)
            resumed_complete.start(manifest)
            resumed_complete.append({"strategy_id": "one", "task_id": "task-2"})
            self.assertEqual(len(load_run(no_newline_dir)["records"]), 2)

            terminated_dir = Path(directory) / "malformed-terminated-final"
            terminated = JsonlResultSink(terminated_dir)
            terminated.start(manifest)
            terminated.append({"strategy_id": "one", "task_id": "task-1"})
            with (terminated_dir / "records.jsonl").open("ab") as handle:
                handle.write(b'{"broken":\n')
            with self.assertRaises(json.JSONDecodeError):
                JsonlResultSink(terminated_dir, resume=True).start(manifest)

            corrupt_dir = Path(directory) / "corrupt"
            corrupt = JsonlResultSink(corrupt_dir)
            corrupt.start(manifest)
            corrupt.append({"strategy_id": "one", "task_id": "task-1"})
            corrupt_records = corrupt_dir / "records.jsonl"
            with corrupt_records.open("ab") as handle:
                handle.write(b'{"broken":\n')
                handle.write(b'{"strategy_id":"one","task_id":"task-2"}\n')

            with self.assertRaises(json.JSONDecodeError):
                JsonlResultSink(corrupt_dir, resume=True).start(manifest)

            corrupt_final_dir = Path(directory) / "corrupt-final"
            corrupt_final = JsonlResultSink(corrupt_final_dir)
            corrupt_final.start(manifest)
            corrupt_final.append({"strategy_id": "one", "task_id": "task-1"})
            with (corrupt_final_dir / "records.jsonl").open("ab") as handle:
                handle.write(b'{"broken":\n')

            with self.assertRaises(json.JSONDecodeError):
                JsonlResultSink(corrupt_final_dir, resume=True).start(manifest)

    def test_code_identity_hashes_tracked_and_untracked_worktree_content(self):
        with TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            for command in (
                ["git", "init", "--quiet"],
                ["git", "config", "user.email", "benchmark@example.com"],
                ["git", "config", "user.name", "Benchmark Test"],
            ):
                subprocess.run(command, cwd=repository, check=True)

            tracked = repository / "tracked.txt"
            tracked.write_text("original\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "--quiet", "-m", "initial"],
                cwd=repository,
                check=True,
            )

            clean = _code_identity(repository)
            self.assertFalse(clean["dirty"])
            self.assertIsNone(clean["dirty_hash"])

            tracked.write_text("changed\n")
            tracked_only = _code_identity(repository)
            self.assertTrue(tracked_only["dirty"])
            self.assertIsNotNone(tracked_only["dirty_hash"])

            untracked = repository / "untracked.txt"
            untracked.write_text("first\n")
            combined = _code_identity(repository)
            self.assertNotEqual(combined["dirty_hash"], tracked_only["dirty_hash"])
            self.assertEqual(combined, _code_identity(repository))

            run_output = repository / "arbitrary-run-directory"
            run_output.mkdir()
            (run_output / "records.jsonl").write_text('{"task_id":"one"}\n')
            (run_output / "notes.txt").write_text("generated output\n")
            self.assertEqual(combined, _code_identity(repository))

            untracked.write_text("second\n")
            changed = _code_identity(repository)
            self.assertNotEqual(changed["dirty_hash"], combined["dirty_hash"])

    def test_failed_later_call_keeps_partial_attempt_and_unknown_total_cost(self):
        def application_factory(_loaded, recorder):
            generation = TracedExecutor(
                _SequenceExecutor(["pass partial", RuntimeError("provider down")]),
                recorder,
                "generation",
            )
            return _PartialFailureApplication(generation)

        result = run_matrix(
            _Suite(),
            [_Loaded("partial-failure")],
            application_factory=application_factory,
            sink=MemoryResultSink(),
            limit=1,
            price_catalog={
                "generation-model": {"input": 0.3, "output": 0.4},
            },
        )

        record = result.records[0]
        self.assertEqual(record["route"], "escalation")
        self.assertEqual(len(record["calls"]), 2)
        self.assertIsNotNone(record["calls"][0]["cost_usd"])
        self.assertIsNone(record["calls"][1]["usage"])
        self.assertIsNone(record["calls"][1]["cost_usd"])
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["attempts"][0]["route"]["phase"], "initial-easy")
        self.assertEqual(record["attempts"][0]["output"]["text"], "pass partial")
        self.assertTrue(record["attempts"][0]["reconstructed"])
        self.assertIsNone(record["cost_usd"])
        self.assertEqual(result.summaries["partial-failure"]["missing_cost_tasks"], 1)

    def test_repeatable_strategy_cli_is_dependency_injected(self):
        created_sink = MemoryResultSink()

        class Builder:
            def __init__(self, *, executor_wrapper, **_kwargs):
                self.executor_wrapper = executor_wrapper

            def build(self, loaded):
                classifier = self.executor_wrapper(
                    _Executor('{"d":"easy"}'), "classifier"
                )
                text = "pass" if loaded.config.name == "passing" else "wrong"
                generation = self.executor_wrapper(_Executor(text), "generation")
                return _Application(classifier, generation)

        with redirect_stdout(StringIO()):
            result = run_suite_cli(
                _Suite(),
                [
                    "--strategy", "passing.yaml",
                    "--strategy", "failing.yaml",
                    "--limit", "1",
                ],
                strategy_loader=lambda path: _Loaded(Path(path).stem),
                builder_factory=Builder,
                sink_factory=lambda *_args, **_kwargs: created_sink,
            )
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.comparison["pairs"][0]["paired_tasks"], 1)

    def test_north_star_cli_uses_real_strategy_loader_and_builder_offline(self):
        sink = MemoryResultSink()

        class Completions:
            def create(self, **kwargs):
                content = '{"d":"easy"}' if kwargs["max_tokens"] == 20 else "pass"
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=content),
                    )],
                    usage=_usage(),
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        )

        def builder_factory(**kwargs):
            return StrategyBuilder(
                env={"OPENROUTER_API_KEY": "test"},
                openrouter_client_factory=lambda _url, _key: client,
                executor_wrapper=kwargs["executor_wrapper"],
            )

        with redirect_stdout(StringIO()):
            result = run_suite_cli(
                _Suite(),
                [
                    "--strategy", "strategies/python-function-completion-difficulty-v1.yaml",
                    "--strategy", "strategies/python-function-completion-difficulty-v2.yaml",
                    "--limit", "1",
                ],
                strategy_loader=load_strategy,
                builder_factory=builder_factory,
                sink_factory=lambda *_args, **_kwargs: sink,
            )
        self.assertEqual(len(result.records), 2)
        self.assertTrue(all(record["evaluation"]["passed"] for record in result.records))
        self.assertEqual(
            {record["classifier_decision"] for record in result.records},
            {"easy"},
        )


if __name__ == "__main__":
    unittest.main()
