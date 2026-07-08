from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import MappingProxyType, SimpleNamespace
import unittest

from smart_ask.benchmarks.artifacts import (
    JsonlResultSink,
    MemoryResultSink,
    load_run,
)
from smart_ask.benchmarks.artifact_schema import SCHEMA_VERSION, validate_record
from smart_ask.benchmarks.cli import _load_price_catalog, run_suite_cli
from smart_ask.benchmarks.compare import compare, summarize
from smart_ask.benchmarks.humaneval.suite import HumanEvalSuite
from smart_ask.benchmarks.livebench.suite import (
    LiveBenchPublicTestsSuite,
    run_public_tests as run_livebench_public_tests,
)
from smart_ask.benchmarks.run_manifest import _code_identity, _package_source_hash
from smart_ask.benchmarks.runner import run_matrix
from smart_ask.benchmarks.suite import BenchmarkCase, Evaluation
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
from smart_ask.metrics import (
    METRICS_WIRE_SCHEMA,
    PriceCatalog,
    RunStats,
    StatsCollector,
)
from smart_ask.strategy.loader import compute_strategy_digest
from smart_ask.strategy.schema import StrategyConfig


def _usage(prompt=10, completion=2):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _price_catalog(*models: str) -> PriceCatalog:
    prices = {
        model: {
            "input": 0.1 if model == "classifier-model" else 0.3,
            "output": 0.2 if model == "classifier-model" else 0.4,
        }
        for model in models
    }
    return PriceCatalog(
        catalog_id="test-catalog",
        effective_date="2026-07-01",
        source="unit-test",
        prices=prices,
    )


def _fake_strategy_config(name: str) -> StrategyConfig:
    return StrategyConfig.model_validate({
        "schema_version": 2,
        "name": name,
        "method": {
            "type": "fixed",
            "role": "generator",
            "model": {"model": "model"},
        },
        "generation": {"type": "openrouter"},
    })


def _difficulty_strategy_config(name: str, *, cascade: bool = False) -> StrategyConfig:
    method = {
        "type": "cascade" if cascade else "difficulty",
        "classifier": {
            "type": "llm",
            "model": "classifier-model",
            "executor": {"type": "openrouter"},
            "prompt": {"type": "inline", "text": "classify"},
            "fallback": "easy",
        },
        "easy": {"model": "generation-model"},
        "hard": {"model": "generation-model"},
    }
    if cascade:
        method["escalation"] = {
            "type": "marker",
            "marker": "ESCALATE",
            "self_check_suffix": {"type": "inline", "text": "ESCALATE"},
            "escalation_prefix": {"type": "inline", "text": "retry"},
        }
    return StrategyConfig.model_validate({
        "schema_version": 2,
        "name": name,
        "method": method,
        "generation": {"type": "openrouter"},
    })


@dataclass(frozen=True)
class _Config:
    name: str


class _Loaded:
    def __init__(self, name):
        self.config = _difficulty_strategy_config(
            name,
            cascade=name == "partial-failure",
        )
        self.path = Path(f"{name}.yaml")
        self.digest = compute_strategy_digest(self.config, {})

    def manifest(self):
        return {
            "name": self.config.name,
            "digest": self.digest,
            "config": self.config.model_dump(mode="json"),
            "prompts": [],
        }


class _Suite:
    name = "fake-suite"
    dataset_identity = {"dataset": "fake", "revision": "1"}
    evaluator_identity = {"name": "prefix-pass", "version": 1}

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
        self.call_count = 0

    def execute(self, request):
        self.call_count += 1
        return ModelResult(
            request.model,
            self.text,
            usage=_usage(),
            raw_text=self.text,
        )


class _Application:
    def __init__(self, classifier, generation, stats_collector, strategy_id):
        self.classifier = classifier
        self.generation = generation
        self.executor = generation
        self.stats_collector = stats_collector
        self.strategy_id = strategy_id
        self.metrics_executors = MappingProxyType({
            "classifier": (classifier,),
            "generation": (generation,),
        })

    def run_detailed(self, task, on_route=None, on_result=None):
        classification = self.classifier.execute(ExecutionRequest(
            "classifier-model",
            "classify\n" + task.prompt,
            "classifier",
            max_tokens=20,
            temperature=0.0,
        ))
        event = RoutingEvent(
            "difficulty-classifier",
            "easy",
            "fake classification",
            model=classification.model,
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
        result = self.generation.execute(ExecutionRequest(
            route.model or "",
            route.prompt or "",
            route.role,
        ))
        if on_result:
            on_result(result, 1)
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
    def __init__(self, classifier, generation, stats_collector, strategy_id):
        self.classifier = classifier
        self.generation = generation
        self.executor = generation
        self.stats_collector = stats_collector
        self.strategy_id = strategy_id
        self.metrics_executors = MappingProxyType({
            "classifier": (classifier,),
            "generation": (generation,),
        })

    def run_detailed(self, task, on_route=None, on_result=None):
        classification = self.classifier.execute(ExecutionRequest(
            "classifier-model",
            "classify\n" + task.prompt,
            "classifier",
            max_tokens=20,
            temperature=0.0,
        ))
        classifier_event = RoutingEvent(
            "difficulty-classifier",
            "easy",
            "fake classification",
            model=classification.model,
        )
        initial = RouteResult(
            action="execute",
            model="generation-model",
            prompt=task.prompt + "ESCALATE",
            role="generator",
            phase="initial-easy",
            label="easy primary",
            routing_events=(classifier_event,),
        )
        if on_route:
            on_route(initial, 1)
        first = self.generation.execute(ExecutionRequest(
            initial.model or "",
            initial.prompt or "",
            initial.role,
        ))
        if on_result:
            on_result(first, 1)

        escalation = RouteResult(
            action="execute",
            model="generation-model",
            prompt=f"retry{task.prompt}",
            role="fixer",
            phase="escalation",
            label="hard retry",
            routing_events=(RoutingEvent(
                "response-escalation",
                "escalate",
                "retry required",
            ),),
        )
        if on_route:
            on_route(escalation, 2)
        self.generation.execute(ExecutionRequest(
            escalation.model or "",
            escalation.prompt or "",
            escalation.role,
        ))
        raise AssertionError("the second executor call should fail")


def _application_factory(loaded, recorder):
    text = "pass answer" if loaded.config.name == "passing" else "wrong answer"
    classifier = recorder.wrap(_Executor('{"d":"easy"}'), "classifier")
    generation = recorder.wrap(_Executor(text), "generation")
    return _Application(classifier, generation, recorder, loaded.config.name)


def _manifest(*task_ids: str, created_at: str = "2026-07-01T00:00:00Z"):
    prompt_digest = hashlib.sha256(b"prompt").hexdigest()
    payload_digest = hashlib.sha256(b"{}").hexdigest()
    cases = [{
        "task_id": task_id,
        "prompt_sha256": prompt_digest,
        "payload_sha256": payload_digest,
    } for task_id in task_ids]
    case_digest = hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    strategy_config = _fake_strategy_config("one")
    strategy_digest = compute_strategy_digest(strategy_config, {})
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "fake-suite",
        "dataset": {"dataset": "fake", "revision": "1"},
        "evaluator": {"name": "prefix-pass", "version": 1},
        "strategies": [{
            "name": "one",
            "digest": strategy_digest,
            "config": strategy_config.model_dump(mode="json"),
            "prompts": [],
        }],
        "case_ids": list(task_ids),
        "cases": cases,
        "case_digest": case_digest,
        "pricing": {
            "catalog_id": "test-catalog",
            "effective_date": "2026-07-01",
            "source": "unit-test",
            "prices": {"model": {"input": 0.1, "output": 0.2}},
            "currency": "USD",
        },
        "metrics": {
            "schema": METRICS_WIRE_SCHEMA,
            "scope": "run",
            "record_unit": "strategy-task",
            "interaction_unit": "model-executor-call",
        },
        "workers": 1,
        "runtime": {
            "python": "3.11.test",
            "platform": {
                "system": "test",
                "release": "test",
                "machine": "test",
                "implementation": "CPython",
            },
            "dependencies": {
                "datasets": None,
                "openai": "2.test",
                "pydantic": "2.test",
                "PyYAML": "6.test",
            },
            "code": {
                "package_version": "0.2.0",
                "package_hash": "b" * 64,
                "git_commit": "a" * 40,
                "dirty": False,
                "dirty_hash": None,
            },
        },
        "created_at": created_at,
    }


def _record(
    task_id: str,
    *,
    strategy_id: str = "one",
    passed: bool = True,
    complete_cost: bool = True,
    failed_interactions: int = 0,
):
    if failed_interactions not in (0, 1):
        raise ValueError("the fixture contains exactly one interaction")
    failed = bool(failed_interactions)
    usage_known = {
        "prompt_tokens": 0 if failed else 2,
        "completion_tokens": 0 if failed else 1,
        "total_tokens": 0 if failed else 3,
        "visible_output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }
    cost_usd = 0.4 if complete_cost and not failed else None
    catalog = {
        "catalog_id": "test-catalog",
        "effective_date": "2026-07-01",
        "source": "unit-test",
        "prices": {"model": {"input": 0.1, "output": 0.2}},
    }
    call = {
        "run_id": f"run-{strategy_id}-{task_id}",
        "call_id": "call-1",
        "ordinal": 1,
        "channel": "generation",
        "role": "generator",
        "status": "error" if failed else "ok",
        "telemetry_status": (
            "partial" if failed or not complete_cost else "complete"
        ),
        "models": {
            "requested": "model",
            "actual": None if failed else "model",
            "priced": None if failed else "model",
        },
        "timing": {"latency_ms": 1.0, "started_offset_ms": 0.0},
        "usage": {
            "prompt_tokens": None if failed else 2,
            "completion_tokens": None if failed else 1,
            "total_tokens": None if failed else 3,
            "visible_output_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
            "cache_write_input_tokens": None,
            "completeness": {
                "total": not failed,
                "breakdown": not failed,
            },
            "status": "unavailable" if failed else "complete",
            "diagnostic": "call failed" if failed else None,
        },
        "cost": {
            "usd": cost_usd,
            "provider_reported_usd": None,
            "status": (
                "unavailable" if failed or not complete_cost else "priced"
            ),
            "source": None if failed else "test-catalog",
            "catalog_id": None if failed else "test-catalog",
            "diagnostic": (
                "call failed" if failed
                else "pricing unavailable" if not complete_cost
                else None
            ),
        },
        "response": {
            "finish_reason": "error" if failed else "unknown",
            "native_finish_reason": None,
            "output_status": None if failed else "usable",
            "output_empty": None if failed else False,
            "refusal": None,
            "requested_max_tokens": None,
            "applied_max_tokens": None,
            "max_tokens_reached": False if failed else None,
        },
        "error": (
            {
                "category": "unknown",
                "type": "RuntimeError",
                "message": "provider down",
            }
            if failed else None
        ),
        "request": {
            "model": "model",
            "role": "generator",
            "prompt": "prompt",
            "max_tokens": None,
            "temperature": None,
        },
        "output": (
            None if failed
            else {"model": "model", "text": "answer", "raw_text": "answer"}
        ),
    }
    error = (
        {"stage": "execution", "type": "RuntimeError", "message": "provider down"}
        if failed else None
    )
    missing_usage = 1 if failed else 0
    missing_cost = 1 if cost_usd is None else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "strategy_digest": compute_strategy_digest(
            _fake_strategy_config(strategy_id),
            {},
        ),
        "task_id": task_id,
        "input": {"prompt": "prompt"},
        "route": "fixed",
        "classifier_decision": None,
        "routing_events": [{
            "source": "fixed-method",
            "outcome": "fixed",
            "reason": "configured route",
            "model": None,
            "call_ids": [],
        }],
        "attempts": [{
            "index": 1,
            "route": {
                "action": "execute",
                "phase": "fixed",
                "label": "fixed",
                "model": "model",
                "role": "generator",
                "prompt": "prompt",
            },
            "call_id": "call-1",
            "status": "error" if failed else "ok",
            **({"reconstructed": True} if failed else {}),
        }],
        "calls": [call],
        "final_output": None if failed else call["output"],
        "evaluation": {
            "passed": False if failed else passed,
            "score": 1.0 if passed and not failed else 0.0,
            "details": {},
        },
        "metrics": {
            "schema": METRICS_WIRE_SCHEMA,
            "scope": "run",
            "runs": 1,
            "timing": {
                "run_duration_ms": 2.0,
                "cumulative_run_duration_ms": 2.0,
            },
            "identity": {
                "run_id": f"run-{strategy_id}-{task_id}",
                "task_id": task_id,
                "strategy_id": strategy_id,
            },
            "interactions": {
                "total": 1,
                "failed": failed_interactions,
                "by_channel": {"generation": 1},
                "by_role": {"generator": 1},
                "by_requested_model": {"model": 1},
                "by_actual_model": {} if failed else {"model": 1},
                "by_priced_model": {} if failed else {"model": 1},
                "errors_by_category": {"unknown": 1} if failed else {},
            },
            "usage": {
                "known": usage_known,
                "total_tokens": None if failed else 3,
                "completeness": {
                    "total": not failed,
                    "breakdown": not failed,
                    "missing_total_calls": missing_usage,
                    "missing_breakdown_calls": missing_usage,
                    "error_calls": 0,
                    "details": {
                        field: {"complete": False, "missing_calls": 1}
                        for field in (
                            "visible_output_tokens",
                            "reasoning_tokens",
                            "cached_input_tokens",
                            "cache_write_input_tokens",
                        )
                    },
                },
            },
            "cost": {
                "known_usd": cost_usd or 0.0,
                "total_usd": cost_usd,
                "completeness": {
                    "complete": cost_usd is not None,
                    "missing_calls": missing_cost,
                    "error_calls": 0,
                },
                "priced_calls_by_source": (
                    {"test-catalog": 1} if cost_usd is not None else {}
                ),
                "catalogs": [] if failed else [catalog],
                "provider_reported": {
                    "known_usd": 0.0,
                    "total_usd": None,
                    "complete": False,
                    "missing_calls": 1,
                },
            },
            "routing": {
                "generation_attempts": 1,
                "events": 1,
            },
            "responses": {
                "finish_reasons": {"error" if failed else "unknown": 1},
                "output_statuses": {
                    "unavailable" if failed else "usable": 1
                },
                "output_emptiness": {
                    "unknown" if failed else "nonempty": 1
                },
                "max_tokens_reached_calls": 0,
            },
            "outcomes": {
                "passed": int(not failed and passed),
                "incorrect": int(not failed and not passed),
                "routing_error": 0,
                "execution_error": int(failed),
                "evaluation_error": 0,
                "unrated": 0,
            },
        },
        "evaluation_latency_ms": None if failed else 0.5,
        "error": error,
        "started_at": "2026-07-01T00:00:00Z",
        "finished_at": "2026-07-01T00:00:01Z",
    }


def _finalize_sink(sink, strategy_order=("one",)):
    records = sink.existing_records
    sink.finalize(
        summarize(records, manifest=sink.manifest),
        compare(
            records,
            strategy_order=strategy_order,
            manifest=sink.manifest,
        ),
    )


class BenchmarkRunnerTests(unittest.TestCase):
    def test_benchmark_values_are_deeply_immutable_strict_json(self):
        payload = {"nested": {"items": [1, 2]}}
        case = BenchmarkCase("task", "prompt", payload)
        payload["nested"]["items"].append(3)
        self.assertEqual(case.payload["nested"]["items"], (1, 2))
        with self.assertRaises(TypeError):
            case.payload["nested"]["new"] = True
        with self.assertRaises(ValueError):
            BenchmarkCase("task", "   ")
        with self.assertRaises(TypeError):
            BenchmarkCase("task", "prompt", {"bad": {1, 2}})

        details = {"nested": [{"score": 1.0}]}
        evaluation = Evaluation(True, 1.0, details)
        details["nested"][0]["score"] = 2.0
        self.assertEqual(evaluation.details["nested"][0]["score"], 1.0)
        with self.assertRaises(ValueError):
            Evaluation(True, 1.0, {"bad": float("nan")})

    def test_suite_identity_and_dataset_loader_are_read_only(self):
        for suite_type in (HumanEvalSuite, LiveBenchPublicTestsSuite):
            with self.subTest(suite=suite_type.__name__):
                with self.assertRaises(TypeError):
                    suite_type(dataset_loader="not callable")
                suite = suite_type(dataset_loader=lambda: [], timeout=3)
                with self.assertRaises(TypeError):
                    suite.dataset_identity["revision"] = "changed"
                with self.assertRaises(TypeError):
                    suite.evaluator_identity["type"] = "changed"
        self.assertEqual(
            LiveBenchPublicTestsSuite(dataset_loader=lambda: [], timeout=3)
            .evaluator_identity["timeout_seconds_per_test"],
            3,
        )
        self.assertEqual(
            LiveBenchPublicTestsSuite.name,
            "livebench-coding-public-tests",
        )
        with self.assertRaises(KeyError):
            run_livebench_public_tests(
                "print('x')",
                "",
                [{"testtype": "stdin", "input": "x"}],
                timeout=1,
            )

    def test_livebench_public_test_smoke_executes_stdin_and_structured_cases(self):
        starter = "class Solution:\n    def solve(self, values):\n        pass\n"
        self.assertEqual(
            run_livebench_public_tests(
                "print(input()[::-1])",
                "",
                [{"testtype": "stdin", "input": "abc\n", "output": "cba\n"}],
                timeout=1,
            ),
            (1, 1),
        )
        structural_code = (
            "class Solution:\n"
            "    def solve(self, selector):\n"
            "        if selector == 0:\n"
            "            return {'b': 2, 'a': 1}\n"
            "        return \"line 1\\n'quoted'\"\n"
        )
        self.assertEqual(
            run_livebench_public_tests(
                structural_code,
                starter,
                [
                    {
                        "testtype": "functional",
                        "input": "0",
                        "output": "{'a': 1, 'b': 2}",
                    },
                    {
                        "testtype": "functional",
                        "input": "1",
                        "output": "line 1\n'quoted'",
                    },
                ],
                timeout=1,
            ),
            (2, 2),
        )
        code = (
            "class Solution:\n"
            "    def solve(self, values):\n"
            "        return {'items': sorted(values)}\n"
        )
        self.assertEqual(
            run_livebench_public_tests(
                code,
                starter,
                [{
                    "testtype": "functional",
                    "input": "[3, 1, 2]",
                    "output": "{'items': [1, 2, 3]}",
                }],
                timeout=1,
            ),
            (1, 1),
        )

    def test_memory_sink_lifecycle_matches_persistent_sink(self):
        sink = MemoryResultSink()
        sink.start(_manifest("task-1"))
        sink.append(_record("task-1"))
        _finalize_sink(sink)
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            sink.append(_record("task-1"))
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            _finalize_sink(sink)
        sink.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            sink.start(_manifest("task-1"))

        with TemporaryDirectory() as directory:
            persistent = JsonlResultSink(Path(directory) / "run")
            persistent.start(_manifest("task-1"))
            persistent.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                persistent.append(_record("task-1"))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                _finalize_sink(persistent)

    def test_record_state_strategy_and_nested_details_are_strict(self):
        sink = MemoryResultSink()
        sink.start(_manifest("task-1"))

        wrong_phase = deepcopy(_record("task-1"))
        wrong_phase["attempts"][0]["route"]["phase"] = "initial-easy"
        wrong_phase["route"] = "initial-easy"
        with self.assertRaisesRegex(ValueError, "phases.*strategy"):
            sink.append(wrong_phase)

        wrong_prompt = deepcopy(_record("task-1"))
        wrong_prompt["attempts"][0]["route"]["prompt"] = "tampered"
        wrong_prompt["calls"][0]["request"]["prompt"] = "tampered"
        with self.assertRaisesRegex(ValueError, "prompt contradicts strategy"):
            sink.append(wrong_prompt)

        wrong_tuning = deepcopy(_record("task-1"))
        wrong_tuning["calls"][0]["request"]["max_tokens"] = 99
        wrong_tuning["calls"][0]["response"]["requested_max_tokens"] = 99
        with self.assertRaisesRegex(ValueError, "tuning contradicts"):
            sink.append(wrong_tuning)

        before_manifest = deepcopy(_record("task-1"))
        before_manifest["started_at"] = "2026-06-30T23:59:59Z"
        with self.assertRaisesRegex(ValueError, "manifest.created_at"):
            sink.append(before_manifest)

        bad_state = deepcopy(_record("task-1"))
        bad_state["final_output"] = None
        with self.assertRaisesRegex(ValueError, "requires output"):
            sink.append(bad_state)

        bad_details = deepcopy(_record("task-1"))
        bad_details["evaluation"]["details"] = {"nested": {"bad": float("nan")}}
        with self.assertRaisesRegex(ValueError, "finite"):
            sink.append(bad_details)

        bad_channel = deepcopy(_record("task-1"))
        bad_channel["calls"][0]["channel"] = "other"
        bad_channel["metrics"]["interactions"]["by_channel"] = {"other": 1}
        with self.assertRaisesRegex(ValueError, "channel unsupported"):
            sink.append(bad_channel)

    def test_manifest_code_identity_is_canonical(self):
        uppercase = _manifest("task-1")
        uppercase["runtime"]["code"]["package_hash"] = "B" * 64
        with self.assertRaisesRegex(ValueError, "lowercase"):
            MemoryResultSink().start(uppercase)

        unknown_dirty = _manifest("task-1")
        unknown_dirty["runtime"]["code"].update({
            "git_commit": None,
            "dirty": False,
        })
        with self.assertRaisesRegex(ValueError, "unknown Git commit"):
            MemoryResultSink().start(unknown_dirty)

    def test_record_sets_reject_duplicate_metrics_run_ids(self):
        first = _record("task-1")
        second = _record("task-2")
        second["metrics"]["identity"]["run_id"] = (
            first["metrics"]["identity"]["run_id"]
        )
        with self.assertRaisesRegex(ValueError, "run_id"):
            summarize([first, second])

    def test_json_artifacts_and_price_catalogs_reject_duplicate_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                '{"schema_version":5,"schema_version":5}\n'
            )
            (run / "records.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_run(run)

            catalog = root / "catalog.json"
            catalog.write_text(
                '{"catalog_id":"one","catalog_id":"two",'
                '"effective_date":"2026-07-01","source":"test","prices":{}}'
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                _load_price_catalog(catalog)

    def test_nested_evaluation_details_serialize_canonically(self):
        class NestedSuite(_Suite):
            def evaluate(self, case, output):
                return Evaluation(True, 1.0, {"rows": [{"task": case.task_id}]})

        sink = MemoryResultSink()
        result = run_matrix(
            NestedSuite(),
            [_Loaded("passing")],
            application_factory=_application_factory,
            sink=sink,
            limit=1,
            price_catalog=_price_catalog("classifier-model", "generation-model"),
        )
        expected = {"rows": [{"task": "task-1"}]}
        self.assertEqual(result.records[0]["evaluation"]["details"], expected)
        self.assertEqual(sink.records[0]["evaluation"]["details"], expected)

    def test_invalid_evaluator_result_is_an_explicit_evaluation_error(self):
        class InvalidSuite(_Suite):
            def evaluate(self, case, output):
                return None

        result = run_matrix(
            InvalidSuite(),
            [_Loaded("passing")],
            application_factory=_application_factory,
            sink=MemoryResultSink(),
            limit=1,
            price_catalog=_price_catalog(
                "classifier-model",
                "generation-model",
            ),
        )
        record = result.records[0]
        self.assertEqual(record["error"]["stage"], "evaluation")
        self.assertEqual(record["metrics"]["outcomes"]["evaluation_error"], 1)
        self.assertIsNotNone(record["final_output"])
        self.assertEqual(
            record["evaluation"],
            {"passed": False, "score": 0.0, "details": {}},
        )

    def test_quality_summary_excludes_task_errors_from_rated_results(self):
        passed = _record("passed", passed=True)
        incorrect = _record("incorrect", passed=False)
        evaluation_error = _record("evaluation-error", passed=False)
        evaluation_error["error"] = {
            "stage": "evaluation",
            "type": "RuntimeError",
            "message": "evaluator failed",
        }
        evaluation_error["metrics"]["outcomes"]["incorrect"] = 0
        evaluation_error["metrics"]["outcomes"]["evaluation_error"] = 1
        execution_error = _record(
            "execution-error",
            passed=False,
            failed_interactions=1,
        )
        routing_error = _record("routing-error", passed=False)
        routing_error.update({
            "route": None,
            "routing_events": [],
            "attempts": [],
            "calls": [],
            "final_output": None,
            "evaluation_latency_ms": None,
            "error": {
                "stage": "routing",
                "type": "RuntimeError",
                "message": "routing failed",
            },
        })
        routing_error["metrics"] = RunStats(
            run_id="run-one-routing-error",
            task_id="routing-error",
            strategy_id="one",
            duration_ms=2.0,
            calls=(),
            outcome="routing_error",
        ).to_dict(include_calls=False)

        evaluation = summarize([
            passed,
            incorrect,
            evaluation_error,
            execution_error,
            routing_error,
        ])["one"]["evaluation"]
        self.assertEqual(evaluation, {
            "rated_tasks": 2,
            "excluded_tasks": 3,
            "all_task_success_rate": 0.2,
            "pass_rate": 0.5,
            "mean_score": 0.5,
        })

        errors_only = summarize([
            evaluation_error,
            execution_error,
            routing_error,
        ])["one"]["evaluation"]
        self.assertEqual(errors_only, {
            "rated_tasks": 0,
            "excluded_tasks": 3,
            "all_task_success_rate": 0.0,
            "pass_rate": None,
            "mean_score": None,
        })

    def test_pair_quality_excludes_errors_but_keeps_resource_deltas(self):
        def evaluation_error(task_id, strategy_id):
            record = _record(
                task_id,
                strategy_id=strategy_id,
                passed=False,
            )
            record["error"] = {
                "stage": "evaluation",
                "type": "RuntimeError",
                "message": "evaluator failed",
            }
            record["metrics"]["outcomes"]["incorrect"] = 0
            record["metrics"]["outcomes"]["evaluation_error"] = 1
            return record

        records = [
            _record("a", strategy_id="reference", passed=True),
            _record("a", strategy_id="candidate", passed=False),
            _record("b", strategy_id="reference", passed=False),
            _record("b", strategy_id="candidate", passed=False),
            evaluation_error("c", "reference"),
            _record("c", strategy_id="candidate", passed=True),
            _record(
                "f",
                strategy_id="reference",
                passed=False,
                failed_interactions=1,
            ),
            _record("f", strategy_id="candidate", passed=True),
            _record("reference-only", strategy_id="reference", passed=True),
            _record("candidate-only", strategy_id="candidate", passed=True),
        ]

        pair = compare(
            records,
            strategy_order=("reference", "candidate"),
        )["pairs"][0]
        self.assertEqual(pair["tasks"], 6)
        self.assertEqual(pair["paired_tasks"], 4)
        self.assertEqual(pair["missing_tasks"], 2)
        self.assertEqual(pair["rated_pairs"], 2)
        self.assertEqual(pair["excluded_pairs"], 2)
        self.assertEqual(pair["both_pass"], 0)
        self.assertEqual(pair["only_reference_passes"], 1)
        self.assertEqual(pair["only_candidate_passes"], 0)
        self.assertEqual(pair["neither_passes"], 1)
        self.assertEqual(pair["mean_score_delta"], -0.5)
        self.assertEqual(pair["mean_duration_delta_ms"], 0.0)
        self.assertEqual(pair["missing_cost_pairs"], 1)

        by_task = {row["task_id"]: row for row in pair["per_task"]}
        evaluation_row = by_task["c"]
        self.assertEqual(evaluation_row["reference_outcome"], "evaluation_error")
        self.assertEqual(evaluation_row["candidate_outcome"], "passed")
        self.assertIsNone(evaluation_row["reference_passed"])
        self.assertTrue(evaluation_row["candidate_passed"])
        self.assertFalse(evaluation_row["quality_rated"])
        self.assertIsNone(evaluation_row["score_delta"])
        self.assertEqual(evaluation_row["cost_delta_usd"], 0.0)
        self.assertEqual(evaluation_row["duration_delta_ms"], 0.0)

        execution_row = by_task["f"]
        self.assertEqual(execution_row["reference_outcome"], "execution_error")
        self.assertFalse(execution_row["quality_rated"])
        self.assertIsNone(execution_row["score_delta"])
        self.assertIsNone(execution_row["cost_delta_usd"])
        self.assertEqual(execution_row["duration_delta_ms"], 0.0)

        missing_row = by_task["candidate-only"]
        self.assertTrue(missing_row["reference_missing"])
        self.assertEqual(missing_row["candidate_outcome"], "passed")
        self.assertTrue(missing_row["candidate_passed"])
        self.assertFalse(missing_row["quality_rated"])

    def test_score_comparisons_never_emit_non_finite_numbers(self):
        reference = _record("overflow", strategy_id="reference", passed=True)
        candidate = _record("overflow", strategy_id="candidate", passed=True)
        reference["evaluation"]["score"] = -1.7e308
        candidate["evaluation"]["score"] = 1.7e308

        pair = compare([reference, candidate])["pairs"][0]

        self.assertIsNone(pair["mean_score_delta"])
        self.assertIsNone(pair["per_task"][0]["score_delta"])

    def test_summary_is_independent_of_persisted_record_order(self):
        records = [
            _record("task-a"),
            _record("task-b"),
            _record("task-c"),
        ]
        for record, duration in zip(records, (1.0, 1e-16, 1e-16)):
            record["metrics"]["timing"]["run_duration_ms"] = duration
            record["metrics"]["timing"]["cumulative_run_duration_ms"] = duration
            record["calls"][0]["timing"] = {
                "latency_ms": 0.0,
                "started_offset_ms": 0.0,
            }

        self.assertEqual(summarize(records), summarize(reversed(records)))

    def test_matrix_emits_one_canonical_metrics_shape(self):
        sink = MemoryResultSink()
        catalog = _price_catalog("classifier-model", "generation-model")
        applications = []

        def application_factory(strategy, recorder):
            application = _application_factory(strategy, recorder)
            applications.append(application)
            return application

        result = run_matrix(
            _Suite(),
            [_Loaded("passing"), _Loaded("failing")],
            application_factory=application_factory,
            sink=sink,
            workers=4,
            price_catalog=catalog,
        )

        self.assertEqual(len(result.records), 4)
        self.assertEqual(len(applications), 4)
        self.assertEqual(len({id(application) for application in applications}), 4)
        record = next(
            item for item in result.records
            if item["strategy_id"] == "passing" and item["task_id"] == "task-1"
        )
        self.assertEqual(record["route"], "initial-easy")
        self.assertEqual(record["classifier_decision"], "easy")
        self.assertEqual(record["final_output"]["raw_text"], "pass answer")
        self.assertTrue({
            "usage",
            "cost_usd",
            "known_cost_usd",
            "statistics",
            "strategy_path",
            "total_latency_ms",
        }.isdisjoint(record))

        metrics = record["metrics"]
        self.assertEqual(metrics["schema"], METRICS_WIRE_SCHEMA)
        self.assertEqual(metrics["scope"], "run")
        self.assertEqual(metrics["identity"]["strategy_id"], "passing")
        self.assertEqual(metrics["interactions"]["total"], 2)
        self.assertEqual(metrics["interactions"]["failed"], 0)
        self.assertEqual(
            metrics["interactions"]["by_role"],
            {"classifier": 1, "generator": 1},
        )
        self.assertEqual(
            metrics["interactions"]["by_requested_model"],
            {"classifier-model": 1, "generation-model": 1},
        )
        self.assertEqual(
            metrics["interactions"]["by_actual_model"],
            {"classifier-model": 1, "generation-model": 1},
        )
        self.assertEqual(metrics["usage"]["known"]["total_tokens"], 24)
        self.assertEqual(metrics["usage"]["total_tokens"], 24)
        self.assertAlmostEqual(metrics["cost"]["total_usd"], 5.2)
        self.assertGreaterEqual(metrics["timing"]["run_duration_ms"], 0)

        self.assertEqual(
            [call["channel"] for call in record["calls"]],
            ["classifier", "generation"],
        )
        self.assertEqual(
            [call["role"] for call in record["calls"]],
            ["classifier", "generator"],
        )
        self.assertEqual(
            [call["call_id"] for call in record["calls"]],
            ["call-1", "call-2"],
        )
        self.assertIn("latency_ms", record["calls"][0]["timing"])
        self.assertIn("completeness", record["calls"][0]["usage"])
        self.assertIn("status", record["calls"][0]["cost"])
        self.assertEqual(record["attempts"][0]["call_id"], "call-2")
        self.assertTrue({"output", "usage", "cost_usd", "latency_ms"}.isdisjoint(
            record["attempts"][0]
        ))
        self.assertEqual(record["routing_events"][0]["call_ids"], ["call-1"])

        self.assertEqual(result.manifest["schema_version"], SCHEMA_VERSION)
        self.assertNotIn("path", result.manifest["strategies"][0])
        self.assertEqual(result.manifest["evaluator"], _Suite.evaluator_identity)
        self.assertEqual(result.manifest["pricing"]["catalog_id"], "test-catalog")
        self.assertEqual(
            set(result.manifest["pricing"]["prices"]),
            {"classifier-model", "generation-model"},
        )
        self.assertEqual(result.manifest["metrics"]["schema"], METRICS_WIRE_SCHEMA)
        summary_metrics = result.summaries["passing"]["metrics"]
        self.assertEqual(summary_metrics["scope"], "summary")
        self.assertEqual(summary_metrics["runs"], 2)
        self.assertEqual(summary_metrics["interactions"]["total"], 4)
        self.assertEqual(summary_metrics["outcomes"]["passed"], 2)
        self.assertEqual(summary_metrics["outcomes"]["incorrect"], 0)
        passing_summary = result.summaries["passing"]
        self.assertEqual(passing_summary["resources"]["total"]["calls"], 4)
        self.assertEqual(passing_summary["routing_flow"]["tasks"], 2)
        self.assertEqual(len(passing_summary["routing_flow"]["paths"]), 1)
        path = passing_summary["routing_flow"]["paths"][0]
        self.assertEqual(path["states"], ["start", "cheap", "accept"])
        self.assertEqual(path["task_count"], 2)
        self.assertEqual(path["attempted_calls"], 2)
        self.assertEqual(path["failed_attempted_calls"], 0)
        self.assertEqual(path["usage"], {
            "known": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
                "visible_output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "total_tokens": 24,
            "completeness": {
                "total": True,
                "breakdown": True,
                "missing_total_calls": 0,
                "missing_breakdown_calls": 0,
                "error_calls": 0,
                "details": {
                    field: {"complete": False, "missing_calls": 2}
                    for field in (
                        "visible_output_tokens",
                        "reasoning_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                    )
                },
            },
        })
        self.assertEqual(path["cost"], {
            "known_usd": 7.6,
            "total_usd": 7.6,
            "completeness": {
                "complete": True,
                "missing_calls": 0,
                "error_calls": 0,
            },
            "provider_reported": {
                "known_usd": 0.0,
                "total_usd": None,
                "complete": False,
                "missing_calls": 2,
            },
            "catalog_estimate_minus_provider": {
                "known_usd": 0.0,
                "total_usd": None,
                "comparable_calls": 0,
                "missing_calls": 2,
            },
        })
        self.assertGreaterEqual(
            passing_summary["timing"]["wall_clock_record_span_ms"],
            0,
        )
        self.assertIn("counterfactual_routing", result.comparison)

        pair = result.comparison["pairs"][0]
        self.assertEqual(pair["only_reference_passes"], 2)
        self.assertEqual(pair["only_candidate_passes"], 0)
        self.assertEqual(pair["missing_tasks"], 0)
        self.assertEqual(pair["cost_source"], "catalog_estimate")

    def test_price_catalog_is_typed_versioned_and_complete_before_execution(self):
        with self.assertRaisesRegex(TypeError, "PriceCatalog"):
            run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=lambda *_args: self.fail("must not build"),
                sink=MemoryResultSink(),
                price_catalog={},
            )

        loaded = load_strategy(
            "smart_ask/resources/strategies/"
            "python-function-completion-difficulty-v1.yaml"
        )
        with self.assertRaisesRegex(ValueError, "Missing benchmark prices"):
            run_matrix(
                _Suite(),
                [loaded],
                application_factory=lambda *_args: self.fail("must not build"),
                sink=MemoryResultSink(),
                price_catalog=_price_catalog(),
            )

    def test_uninstrumented_application_factory_is_rejected(self):
        classifier = _Executor('{"d":"easy"}')
        generation = _Executor("pass answer")

        def uninstrumented(loaded, recorder):
            return _Application(
                classifier,
                generation,
                recorder,
                loaded.config.name,
            )

        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "aborted"
            with self.assertRaisesRegex(ValueError, "not instrumented"):
                run_matrix(
                    _Suite(),
                    [_Loaded("passing")],
                    application_factory=uninstrumented,
                    sink=JsonlResultSink(run_dir),
                    limit=1,
                    price_catalog=_price_catalog(
                        "classifier-model",
                        "generation-model",
                    ),
                )

            self.assertEqual(classifier.call_count, 0)
            self.assertEqual(generation.call_count, 0)

            persisted = json.loads((run_dir / "manifest.json").read_text())
            resumed = JsonlResultSink(run_dir, resume=True)
            resumed.start(persisted)
            resumed.close()

    def test_fully_completed_resume_rebuilds_reports_without_applications(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "complete"
            first = run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=_application_factory,
                sink=JsonlResultSink(run_dir),
                limit=1,
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )
            self.assertEqual(len(first.records), 1)

            resumed = run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=lambda *_args: self.fail(
                    "completed resume must not build an application"
                ),
                sink=JsonlResultSink(run_dir, resume=True),
                limit=1,
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )
            self.assertEqual(resumed.records, first.records)
            self.assertTrue((run_dir / "summary.json").is_file())

    def test_raw_generation_executor_is_rejected_before_execution(self):
        generation = _Executor("pass answer")

        def partially_instrumented(loaded, recorder):
            classifier = recorder.wrap(
                _Executor('{"d":"easy"}'),
                "classifier",
            )
            return _Application(
                classifier,
                generation,
                recorder,
                loaded.config.name,
            )

        with self.assertRaisesRegex(ValueError, "generation executor.*not instrumented"):
            run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=partially_instrumented,
                sink=MemoryResultSink(),
                limit=1,
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )
        self.assertEqual(generation.call_count, 0)

    def test_application_strategy_identity_is_checked_before_execution(self):
        def wrong_strategy(loaded, recorder):
            application = _application_factory(loaded, recorder)
            application.strategy_id = "different-strategy"
            return application

        with self.assertRaisesRegex(ValueError, "strategy_id"):
            run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=wrong_strategy,
                sink=MemoryResultSink(),
                limit=1,
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )

    def test_application_factory_must_isolate_pending_pairs(self):
        cached_application = None

        def reused_application(loaded, recorder):
            nonlocal cached_application
            if cached_application is None:
                cached_application = _application_factory(loaded, recorder)
            return cached_application

        with self.assertRaisesRegex(ValueError, "distinct application"):
            run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=reused_application,
                sink=MemoryResultSink(),
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )

        shared_wrappers = None

        def reused_executors(loaded, recorder):
            nonlocal shared_wrappers
            if shared_wrappers is None:
                shared_wrappers = (
                    recorder.wrap(_Executor('{"d":"easy"}'), "classifier"),
                    recorder.wrap(_Executor("pass answer"), "generation"),
                )
            return _Application(
                *shared_wrappers,
                recorder,
                loaded.config.name,
            )

        with self.assertRaisesRegex(ValueError, "share instrumented executors"):
            run_matrix(
                _Suite(),
                [_Loaded("passing")],
                application_factory=reused_executors,
                sink=MemoryResultSink(),
                price_catalog=_price_catalog(
                    "classifier-model",
                    "generation-model",
                ),
            )

    def test_jsonl_sink_new_resume_and_lock_lifecycle_are_safe(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            manifest = _manifest("task-1")
            sink = JsonlResultSink(run_dir)
            self.assertEqual(sink.start(manifest), manifest)
            sink.append(_record("task-1"))

            with self.assertRaisesRegex(RuntimeError, "already in use"):
                JsonlResultSink(run_dir, resume=True).start(manifest)

            with self.assertRaisesRegex(ValueError, "summaries"):
                sink.finalize({}, {})
            _finalize_sink(sink)
            self.assertTrue((run_dir / "summary.json").exists())
            with self.assertRaises(FileExistsError):
                JsonlResultSink(run_dir).start(manifest)

            resumed = JsonlResultSink(run_dir, resume=True)
            persisted = resumed.start(_manifest(
                "task-1",
                created_at="2026-07-02T00:00:00Z",
            ))
            self.assertEqual(persisted["created_at"], manifest["created_at"])
            self.assertEqual(resumed.completed_keys, {("one", "task-1")})
            self.assertFalse((run_dir / "summary.json").exists())
            _finalize_sink(resumed)
            self.assertEqual(len(load_run(run_dir)["records"]), 1)

            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                JsonlResultSink(missing, resume=True).start(manifest)

            nonempty = Path(directory) / "nonempty"
            nonempty.mkdir()
            (nonempty / "unrelated.txt").write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                JsonlResultSink(nonempty).start(manifest)

    def test_artifacts_reject_wrong_versions_ids_duplicates_and_legacy_files(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            manifest = _manifest("task-1")
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)

            with self.assertRaisesRegex(ValueError, "schema_version"):
                sink.append({**_record("task-1"), "schema_version": 4})
            with self.assertRaisesRegex(ValueError, "unknown task_id"):
                sink.append(_record("unknown"))

            sink.append(_record("task-1"))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                sink.append(_record("task-1"))
            _finalize_sink(sink)

            with (run_dir / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(_record("task-1")) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_run(run_dir)

            legacy_file = Path(directory) / "legacy.json"
            legacy_file.write_text('{"results": []}')
            with self.assertRaises(NotADirectoryError):
                load_run(legacy_file)

            summary_run = Path(directory) / "bad-summary"
            summary_sink = JsonlResultSink(summary_run)
            summary_sink.start(manifest)
            summary_sink.append(_record("task-1"))
            _finalize_sink(summary_sink)
            summary = json.loads((summary_run / "summary.json").read_text())
            original_summary = deepcopy(summary)
            summary["summaries"]["one"]["tasks"] = 999
            (summary_run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "persisted records"):
                load_run(summary_run)

            summary = original_summary
            summary["schema_version"] = 4
            (summary_run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_run(summary_run)

    def test_one_record_validator_guards_append_load_and_comparison(self):
        with TemporaryDirectory() as directory:
            manifest = _manifest("task-1")
            run_dir = Path(directory) / "run"
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)

            missing_evaluation = deepcopy(_record("task-1"))
            missing_evaluation["evaluation"].pop("score")
            with self.assertRaisesRegex(ValueError, "evaluation.*score"):
                sink.append(missing_evaluation)

            oversized_score = deepcopy(_record("task-1"))
            oversized_score["evaluation"]["score"] = 10**400
            with self.assertRaisesRegex(TypeError, "finite number"):
                sink.append(oversized_score)

            empty_metrics = deepcopy(_record("task-1"))
            empty_metrics["metrics"] = {}
            with self.assertRaisesRegex(ValueError, "metrics payload"):
                sink.append(empty_metrics)

            wrong_identity = deepcopy(_record("task-1"))
            wrong_identity["metrics"]["identity"]["task_id"] = "different"
            with self.assertRaisesRegex(ValueError, "identity"):
                sink.append(wrong_identity)
            with self.assertRaisesRegex(ValueError, "identity"):
                compare([wrong_identity])

            wrong_call_run = deepcopy(_record("task-1"))
            wrong_call_run["calls"][0]["run_id"] = "different-run"
            with self.assertRaisesRegex(ValueError, "run_id"):
                sink.append(wrong_call_run)

            dangling_attempt = deepcopy(_record("task-1"))
            dangling_attempt["attempts"][0]["call_id"] = "missing-call"
            with self.assertRaisesRegex(ValueError, "dangling call reference"):
                sink.append(dangling_attempt)

            dangling_event = deepcopy(_record("task-1"))
            dangling_event["routing_events"][0]["source"] = (
                "difficulty-classifier"
            )
            dangling_event["routing_events"][0]["outcome"] = "easy"
            dangling_event["routing_events"][0]["model"] = "model"
            dangling_event["routing_events"][0]["call_ids"] = ["missing-call"]
            with self.assertRaisesRegex(ValueError, "dangling call reference"):
                sink.append(dangling_event)

            contradictory_usage = deepcopy(_record("task-1"))
            contradictory_usage["calls"][0]["usage"]["status"] = "partial"
            with self.assertRaisesRegex(ValueError, "usage.*status"):
                sink.append(contradictory_usage)

            overallocated_details = deepcopy(_record("task-1"))
            overallocated_details["calls"][0]["usage"].update({
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": 10,
                "visible_output_tokens": 10,
                "reasoning_tokens": None,
                "cached_input_tokens": 10,
                "cache_write_input_tokens": None,
                "completeness": {"total": True, "breakdown": False},
                "status": "total_only",
                "diagnostic": "provider returned partial token details",
            })
            with self.assertRaisesRegex(
                ValueError,
                "input and output token details",
            ):
                sink.append(overallocated_details)

            contradictory_cost = deepcopy(_record("task-1"))
            contradictory_cost["calls"][0]["cost"]["usd"] = 0.5
            with self.assertRaisesRegex(ValueError, "metrics cost"):
                sink.append(contradictory_cost)

            contradictory_telemetry = deepcopy(_record("task-1"))
            contradictory_telemetry["calls"][0]["telemetry_status"] = "partial"
            with self.assertRaisesRegex(ValueError, "telemetry_status"):
                sink.append(contradictory_telemetry)

            pricing_failure = deepcopy(_record("task-1"))
            pricing_failure["calls"][0]["cost"].update({
                "usd": None,
                "status": "error",
                "source": None,
                "catalog_id": None,
                "diagnostic": "pricing failed: RuntimeError: unavailable",
            })
            pricing_failure["calls"][0]["telemetry_status"] = "error"
            pricing_failure["metrics"]["cost"].update({
                "known_usd": 0.0,
                "total_usd": None,
                "completeness": {
                    "complete": False,
                    "missing_calls": 1,
                    "error_calls": 1,
                },
                "priced_calls_by_source": {},
                "catalogs": [],
            })
            self.assertEqual(
                validate_record(pricing_failure, manifest),
                ("one", "task-1"),
            )

            contradictory_response = deepcopy(_record("task-1"))
            contradictory_response["calls"][0]["response"][
                "output_status"
            ] = "empty"
            with self.assertRaisesRegex(ValueError, "output_status"):
                sink.append(contradictory_response)

            contradictory_visible_tokens = deepcopy(_record("task-1"))
            contradictory_visible_tokens["calls"][0]["usage"][
                "visible_output_tokens"
            ] = 0
            with self.assertRaisesRegex(
                ValueError,
                "visible_output_tokens.*emptiness",
            ):
                sink.append(contradictory_visible_tokens)

            unsupported_truncation = deepcopy(_record("task-1"))
            unsupported_truncation["calls"][0]["response"].update({
                "finish_reason": "stop",
                "output_status": "truncated",
                "max_tokens_reached": False,
            })
            unsupported_truncation["metrics"]["responses"].update({
                "finish_reasons": {"stop": 1},
                "output_statuses": {"truncated": 1},
            })
            with self.assertRaisesRegex(ValueError, "truncated.*length"):
                sink.append(unsupported_truncation)

            unsupported_refusal = deepcopy(_record("task-1"))
            unsupported_refusal["calls"][0]["response"][
                "output_status"
            ] = "refused"
            unsupported_refusal["metrics"]["responses"]["output_statuses"] = {
                "refused": 1,
            }
            with self.assertRaisesRegex(ValueError, "output_status"):
                sink.append(unsupported_refusal)

            applied_limit = deepcopy(_record("task-1"))
            applied_limit["calls"][0]["response"][
                "applied_max_tokens"
            ] = 128
            self.assertEqual(
                validate_record(applied_limit, manifest),
                ("one", "task-1"),
            )

            requested_limit_mismatch = deepcopy(_record("task-1"))
            requested_limit_mismatch["calls"][0]["response"][
                "requested_max_tokens"
            ] = 64
            with self.assertRaisesRegex(
                ValueError,
                "requested_max_tokens disagree",
            ):
                sink.append(requested_limit_mismatch)

            invalid_applied_limit = deepcopy(_record("task-1"))
            invalid_applied_limit["calls"][0]["response"][
                "applied_max_tokens"
            ] = 0
            with self.assertRaisesRegex(ValueError, "applied_max_tokens"):
                sink.append(invalid_applied_limit)

            unavailable_output = deepcopy(_record("task-1"))
            unavailable_call = unavailable_output["calls"][0]
            unavailable_call["output"] = {
                "model": "model",
                "text": "",
                "raw_text": None,
            }
            unavailable_call["response"]["output_status"] = "unavailable"
            unavailable_call["response"]["output_empty"] = None
            unavailable_call["response"]["finish_reason"] = "stop"
            unavailable_call["response"]["max_tokens_reached"] = False
            unavailable_output["final_output"] = unavailable_call["output"]
            unavailable_output["metrics"]["responses"]["finish_reasons"] = {
                "stop": 1,
            }
            unavailable_output["metrics"]["responses"]["output_statuses"] = {
                "unavailable": 1,
            }
            unavailable_output["metrics"]["responses"]["output_emptiness"] = {
                "unknown": 1,
            }
            self.assertEqual(
                validate_record(unavailable_output, manifest),
                ("one", "task-1"),
            )

            refusal_at_length = deepcopy(_record("task-1"))
            refusal_call = refusal_at_length["calls"][0]
            refusal_call["output"] = {
                "model": "model",
                "text": "",
                "raw_text": "",
            }
            refusal_call["response"].update({
                "finish_reason": "length",
                "output_status": "refused",
                "output_empty": True,
                "refusal": "Request refused.",
                "max_tokens_reached": True,
            })
            refusal_at_length["final_output"] = refusal_call["output"]
            refusal_at_length["metrics"]["responses"] = {
                "finish_reasons": {"length": 1},
                "output_statuses": {"refused": 1},
                "output_emptiness": {"empty": 1},
                "max_tokens_reached_calls": 1,
            }
            self.assertEqual(
                validate_record(refusal_at_length, manifest),
                ("one", "task-1"),
            )

            unavailable_with_text = deepcopy(_record("task-1"))
            unavailable_with_text["calls"][0]["response"][
                "output_status"
            ] = "unavailable"
            unavailable_with_text["metrics"]["responses"][
                "output_statuses"
            ] = {"unavailable": 1}
            unavailable_with_text["metrics"]["responses"][
                "output_emptiness"
            ] = {"unknown": 1}
            with self.assertRaisesRegex(ValueError, "output_status"):
                sink.append(unavailable_with_text)

            failed_requested_limit = _record(
                "task-1",
                failed_interactions=1,
            )
            failed_requested_limit["calls"][0]["request"]["max_tokens"] = 17
            failed_requested_limit["calls"][0]["response"][
                "requested_max_tokens"
            ] = 17
            self.assertEqual(
                validate_record(failed_requested_limit),
                ("one", "task-1"),
            )

            failed_applied_limit = deepcopy(failed_requested_limit)
            failed_applied_limit["calls"][0]["response"][
                "applied_max_tokens"
            ] = 17
            with self.assertRaisesRegex(ValueError, "failed-call evidence"):
                validate_record(failed_applied_limit)

            contradictory_outcome = deepcopy(_record("task-1"))
            contradictory_outcome["metrics"]["outcomes"]["passed"] = 0
            contradictory_outcome["metrics"]["outcomes"]["incorrect"] = 1
            with self.assertRaisesRegex(ValueError, "outcome contradicts"):
                sink.append(contradictory_outcome)

            naive_started_at = deepcopy(_record("task-1"))
            naive_started_at["started_at"] = "2026-07-01T00:00:00"
            with self.assertRaisesRegex(ValueError, "timezone"):
                compare([naive_started_at])

            reversed_timestamps = deepcopy(_record("task-1"))
            reversed_timestamps["finished_at"] = "2026-06-30T23:59:59Z"
            with self.assertRaisesRegex(ValueError, "must not precede"):
                compare([reversed_timestamps])

            contradictory_timing = deepcopy(_record("task-1"))
            contradictory_timing["finished_at"] = contradictory_timing["started_at"]
            with self.assertRaisesRegex(ValueError, "do not cover"):
                compare([contradictory_timing])

            short_duration = deepcopy(_record("task-1"))
            short_duration["metrics"]["timing"]["run_duration_ms"] = 0.5
            short_duration["metrics"]["timing"][
                "cumulative_run_duration_ms"
            ] = 0.5
            with self.assertRaisesRegex(ValueError, "call timings"):
                compare([short_duration])

            sink.append(_record("task-1"))
            _finalize_sink(sink)
            malformed = deepcopy(_record("task-1"))
            malformed["evaluation"] = {"passed": True}
            (run_dir / "records.jsonl").write_text(json.dumps(malformed) + "\n")
            with self.assertRaisesRegex(ValueError, "evaluation"):
                load_run(run_dir)

    def test_manifest_and_record_snapshots_are_cryptographically_bound(self):
        manifest_mutations = []

        unknown_root = deepcopy(_manifest("task-1"))
        unknown_root["legacy"] = True
        manifest_mutations.append((unknown_root, "unknown fields"))

        empty_dataset = deepcopy(_manifest("task-1"))
        empty_dataset["dataset"] = {}
        manifest_mutations.append((empty_dataset, "must not be empty"))

        wrong_currency = deepcopy(_manifest("task-1"))
        wrong_currency["pricing"]["currency"] = "EUR"
        manifest_mutations.append((wrong_currency, "currency"))

        naive_time = deepcopy(_manifest("task-1"))
        naive_time["created_at"] = "2026-07-01T00:00:00"
        manifest_mutations.append((naive_time, "timezone"))

        wrong_case_digest = deepcopy(_manifest("task-1"))
        wrong_case_digest["case_digest"] = "0" * 64
        manifest_mutations.append((wrong_case_digest, "case_digest"))

        wrong_strategy_digest = deepcopy(_manifest("task-1"))
        wrong_strategy_digest["strategies"][0]["digest"] = "f" * 64
        manifest_mutations.append((wrong_strategy_digest, "digest"))

        legacy_strategy_path = deepcopy(_manifest("task-1"))
        legacy_strategy_path["strategies"][0]["path"] = "/tmp/one.yaml"
        manifest_mutations.append((legacy_strategy_path, "unknown fields"))

        unknown_runtime = deepcopy(_manifest("task-1"))
        unknown_runtime["runtime"]["legacy"] = True
        manifest_mutations.append((unknown_runtime, "unknown fields"))

        for index, (malformed, message) in enumerate(manifest_mutations):
            with self.subTest(manifest=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    MemoryResultSink().start(malformed)

        loaded = load_strategy(
            "smart_ask/resources/strategies/"
            "python-function-completion-difficulty-v1.yaml"
        )
        wrong_prompt = deepcopy(_manifest("task-1"))
        wrong_prompt["strategies"] = [loaded.manifest()]
        wrong_prompt["strategies"][0]["prompts"][0]["text"] += "tampered"
        with self.assertRaisesRegex(ValueError, "does not match its text"):
            MemoryResultSink().start(wrong_prompt)

        with TemporaryDirectory() as directory:
            sink = JsonlResultSink(Path(directory) / "run")
            sink.start(_manifest("task-1"))

            wrong_digest = deepcopy(_record("task-1"))
            wrong_digest["strategy_digest"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "strategy_digest"):
                sink.append(wrong_digest)

            legacy_path = deepcopy(_record("task-1"))
            legacy_path["strategy_path"] = "different.yaml"
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                sink.append(legacy_path)

            wrong_prompt_record = deepcopy(_record("task-1"))
            wrong_prompt_record["input"]["prompt"] = "different"
            with self.assertRaisesRegex(ValueError, "input prompt"):
                sink.append(wrong_prompt_record)

            fake_cost = deepcopy(_record("task-1"))
            fake_cost["calls"][0]["cost"]["usd"] = 999.0
            fake_cost["metrics"]["cost"]["known_usd"] = 999.0
            fake_cost["metrics"]["cost"]["total_usd"] = 999.0
            with self.assertRaisesRegex(ValueError, "manifest catalog"):
                sink.append(fake_cost)
            with self.assertRaisesRegex(ValueError, "manifest catalog"):
                validate_record(fake_cost)

            short_duration = deepcopy(_record("task-1"))
            short_duration["metrics"]["timing"]["run_duration_ms"] = 0.5
            short_duration["metrics"]["timing"][
                "cumulative_run_duration_ms"
            ] = 0.5
            with self.assertRaisesRegex(ValueError, "call timings"):
                sink.append(short_duration)

            route_drift = deepcopy(_record("task-1"))
            route_drift["attempts"][0]["route"]["model"] = "other"
            with self.assertRaisesRegex(ValueError, "call request"):
                sink.append(route_drift)

            decision_drift = deepcopy(_record("task-1"))
            decision_drift["classifier_decision"] = "easy"
            with self.assertRaisesRegex(ValueError, "classifier_decision"):
                sink.append(decision_drift)
            sink.close()

    def test_final_summary_requires_complete_matrix_and_exact_derived_values(self):
        with TemporaryDirectory() as directory:
            manifest = _manifest("task-1", "task-2")
            run_dir = Path(directory) / "run"
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)
            sink.append(_record("task-1"))
            records = sink.existing_records
            with self.assertRaisesRegex(ValueError, "every strategy/case"):
                sink.finalize(
                    summarize(records),
                    compare(records, strategy_order=["one"]),
                )
            sink.append(_record("task-2"))
            _finalize_sink(sink)

            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["comparison"]["strategy_order"] = []
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "comparison"):
                load_run(run_dir)

    def test_empty_suite_fails_before_creating_artifact_directory(self):
        class EmptySuite(_Suite):
            def load_cases(self, limit=None):
                return []

        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "not-created"
            with self.assertRaisesRegex(ValueError, "any cases"):
                run_matrix(
                    EmptySuite(),
                    [_Loaded("passing")],
                    application_factory=_application_factory,
                    sink=JsonlResultSink(run_dir),
                    price_catalog=_price_catalog(
                        "classifier-model",
                        "generation-model",
                    ),
                )
            self.assertFalse(run_dir.exists())

    def test_resume_validates_every_semantic_manifest_identity(self):
        with TemporaryDirectory() as directory:
            manifest = _manifest("task-1")
            run_dir = Path(directory) / "run"
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)
            sink.close()

            changes = {
                "workers": 2,
                "evaluator": {"name": "different", "version": 1},
                "metrics": {"schema": "different"},
            }
            for field, changed in changes.items():
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        JsonlResultSink(run_dir, resume=True).start({
                            **manifest,
                            field: changed,
                        })

            with self.assertRaisesRegex(ValueError, "schema_version"):
                JsonlResultSink(run_dir, resume=True).start({
                    **manifest,
                    "schema_version": 4,
                })

    def test_resume_repairs_only_an_unterminated_torn_final_record(self):
        with TemporaryDirectory() as directory:
            manifest = _manifest("task-1", "task-2")
            run_dir = Path(directory) / "recoverable"
            sink = JsonlResultSink(run_dir)
            sink.start(manifest)
            sink.append(_record("task-1"))
            records_path = run_dir / "records.jsonl"
            complete_bytes = records_path.read_bytes()
            with records_path.open("ab") as handle:
                handle.write(b'{"schema_version":5')
            sink.close()

            resumed = JsonlResultSink(run_dir, resume=True)
            resumed.start(manifest)
            self.assertEqual(records_path.read_bytes(), complete_bytes)
            resumed.append(_record("task-2"))
            _finalize_sink(resumed)
            self.assertEqual(len(load_run(run_dir)["records"]), 2)

            corrupt_dir = Path(directory) / "terminated-corruption"
            corrupt = JsonlResultSink(corrupt_dir)
            corrupt.start(manifest)
            corrupt.append(_record("task-1"))
            with (corrupt_dir / "records.jsonl").open("ab") as handle:
                handle.write(b'{"broken":\n')
            corrupt.close()
            with self.assertRaises(json.JSONDecodeError):
                JsonlResultSink(corrupt_dir, resume=True).start(manifest)

    def test_compare_is_strict_and_never_fabricates_total_cost(self):
        complete = _record("task-1", strategy_id="a")
        incomplete = _record(
            "task-1",
            strategy_id="b",
            complete_cost=False,
            failed_interactions=1,
        )
        summaries = summarize([complete, incomplete])
        self.assertEqual(
            summaries["b"]["metrics"]["interactions"]["failed"],
            1,
        )
        self.assertIsNone(summaries["b"]["metrics"]["cost"]["total_usd"])
        pair = compare(
            [complete, incomplete],
            strategy_order=["a", "b"],
        )["pairs"][0]
        self.assertIsNone(pair["total_cost_delta_usd"])

        missing_pair = compare(
            [complete],
            strategy_order=["a", "b"],
        )["pairs"][0]
        self.assertEqual(missing_pair["missing_tasks"], 1)
        self.assertIsNone(missing_pair["total_cost_delta_usd"])

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([complete, complete])
        old_shape = dict(complete)
        old_shape.pop("metrics")
        old_shape["cost_usd"] = 0.25
        with self.assertRaisesRegex(ValueError, "metrics"):
            compare([old_shape])

    def test_failed_generation_attempt_remains_linked_and_totals_are_unknown(self):
        def application_factory(_loaded, recorder):
            classifier = recorder.wrap(_Executor('{"d":"easy"}'), "classifier")
            generation = recorder.wrap(
                _SequenceExecutor(["pass partial", RuntimeError("provider down")]),
                "generation",
            )
            return _PartialFailureApplication(
                classifier,
                generation,
                recorder,
                _loaded.config.name,
            )

        result = run_matrix(
            _Suite(),
            [_Loaded("partial-failure")],
            application_factory=application_factory,
            sink=MemoryResultSink(),
            limit=1,
            price_catalog=_price_catalog("classifier-model", "generation-model"),
        )

        record = result.records[0]
        self.assertEqual(record["route"], "escalation")
        self.assertEqual(
            [call["status"] for call in record["calls"]],
            ["ok", "ok", "error"],
        )
        self.assertIsNotNone(record["calls"][1]["cost"]["usd"])
        self.assertIsNone(record["calls"][2]["cost"]["usd"])
        self.assertEqual(record["calls"][2]["usage"]["status"], "unavailable")
        self.assertEqual(
            [attempt["status"] for attempt in record["attempts"]],
            ["ok", "error"],
        )
        self.assertTrue(all(
            attempt["reconstructed"] for attempt in record["attempts"]
        ))
        self.assertEqual(
            [attempt["call_id"] for attempt in record["attempts"]],
            ["call-2", "call-3"],
        )

        metrics = record["metrics"]
        self.assertEqual(metrics["interactions"]["total"], 3)
        self.assertEqual(metrics["interactions"]["failed"], 1)
        self.assertEqual(metrics["routing"]["generation_attempts"], 2)
        self.assertFalse(metrics["usage"]["completeness"]["total"])
        self.assertEqual(metrics["usage"]["known"]["total_tokens"], 24)
        self.assertIsNone(metrics["usage"]["total_tokens"])
        self.assertIsNone(metrics["cost"]["total_usd"])
        self.assertEqual(metrics["outcomes"]["execution_error"], 1)
        paths = result.summaries["partial-failure"]["routing_flow"]["paths"]
        self.assertEqual(len(paths), 1)
        path = paths[0]
        self.assertEqual(
            path["states"],
            ["start", "cheap", "expensive", "error"],
        )
        self.assertEqual(path["task_count"], 1)
        self.assertEqual(path["attempted_calls"], 2)
        self.assertEqual(path["failed_attempted_calls"], 1)
        self.assertEqual(path["usage"], {
            "known": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "visible_output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "total_tokens": None,
            "completeness": {
                "total": False,
                "breakdown": False,
                "missing_total_calls": 1,
                "missing_breakdown_calls": 1,
                "error_calls": 0,
                "details": {
                    field: {"complete": False, "missing_calls": 2}
                    for field in (
                        "visible_output_tokens",
                        "reasoning_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                    )
                },
            },
        })
        self.assertEqual(path["cost"], {
            "known_usd": 3.8,
            "total_usd": None,
            "completeness": {
                "complete": False,
                "missing_calls": 1,
                "error_calls": 0,
            },
            "provider_reported": {
                "known_usd": 0.0,
                "total_usd": None,
                "complete": False,
                "missing_calls": 2,
            },
            "catalog_estimate_minus_provider": {
                "known_usd": 0.0,
                "total_usd": None,
                "comparable_calls": 0,
                "missing_calls": 2,
            },
        })
        self.assertEqual(
            result.summaries["partial-failure"]["metrics"]["interactions"]["failed"],
            1,
        )

    def test_repeatable_strategy_cli_injects_stats_collector_explicitly(self):
        created_sink = MemoryResultSink()
        received_collectors = []

        class Builder:
            def __init__(self, *, stats_collector, **kwargs):
                self.collector = stats_collector
                received_collectors.append(stats_collector)
                self.assert_no_wrapper = "executor_wrapper" not in kwargs

            def build(self, loaded):
                if not self.assert_no_wrapper:
                    raise AssertionError("legacy executor_wrapper was injected")
                classifier = self.collector.wrap(
                    _Executor('{"d":"easy"}'),
                    "classifier",
                )
                text = "pass" if loaded.config.name == "passing" else "wrong"
                generation = self.collector.wrap(_Executor(text), "generation")
                return _Application(
                    classifier,
                    generation,
                    self.collector,
                    loaded.config.name,
                )

        with TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "prices.json"
            catalog_path.write_text(json.dumps(
                _price_catalog("classifier-model", "generation-model").to_dict()
            ))
            with redirect_stdout(StringIO()):
                result = run_suite_cli(
                    _Suite(),
                    [
                        "--strategy", "passing.yaml",
                        "--strategy", "failing.yaml",
                        "--limit", "1",
                        "--price-catalog", str(catalog_path),
                    ],
                    strategy_loader=lambda path: _Loaded(Path(path).stem),
                    builder_factory=Builder,
                    sink_factory=lambda *_args, **_kwargs: created_sink,
                )
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.comparison["pairs"][0]["paired_tasks"], 1)
        self.assertEqual(len(received_collectors), 1)
        self.assertTrue(all(
            isinstance(collector, StatsCollector)
            for collector in received_collectors
        ))

    def test_north_star_cli_uses_real_loader_builder_and_resources_offline(self):
        sink = MemoryResultSink()

        class Completions:
            def create(self, **kwargs):
                content = '{"d":"easy"}' if kwargs["max_tokens"] == 20 else "pass"
                return SimpleNamespace(
                    model=kwargs["model"],
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
                stats_collector=kwargs["stats_collector"],
            )

        strategy_root = Path("smart_ask/resources/strategies")
        with redirect_stdout(StringIO()):
            result = run_suite_cli(
                _Suite(),
                [
                    "--strategy",
                    str(strategy_root / "python-function-completion-difficulty-v1.yaml"),
                    "--strategy",
                    str(strategy_root / "python-function-completion-difficulty-v2.yaml"),
                    "--limit",
                    "1",
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

            run_output = repository / "benchmark-results" / "suite" / "run"
            run_output.mkdir(parents=True)
            (run_output / "manifest.json").write_text("{}\n")
            (run_output / "records.jsonl").write_text('{"task_id":"one"}\n')
            (run_output / "notes.txt").write_text("generated output\n")
            self.assertEqual(combined, _code_identity(repository))

            run_output = repository / "explicit-custom-run"
            run_output.mkdir()
            (run_output / "manifest.json").write_text("{}\n")
            (run_output / "records.jsonl").write_text('{"task_id":"one"}\n')
            (run_output / "notes.txt").write_text("generated output\n")
            self.assertEqual(combined, _code_identity(repository))

            genuine_source = repository / "results" / "source.py"
            genuine_source.parent.mkdir()
            genuine_source.write_text("VALUE = 1\n")
            with_source = _code_identity(repository)
            self.assertNotEqual(with_source["dirty_hash"], combined["dirty_hash"])

            untracked.write_text("second\n")
            changed = _code_identity(repository)
            self.assertNotEqual(changed["dirty_hash"], with_source["dirty_hash"])

    def test_installed_code_identity_never_claims_an_unrelated_repository(self):
        with TemporaryDirectory() as directory:
            repository = Path(directory) / "unrelated"
            package = repository / ".venv" / "site-packages" / "smart_ask"
            package.mkdir(parents=True)
            (repository / ".git").mkdir()
            source = package / "__init__.py"
            source.write_text("VALUE = 1\n")

            first_hash = _package_source_hash(package)
            identity = _code_identity(package_root=package)
            self.assertEqual(identity["package_hash"], first_hash)
            self.assertIsNone(identity["git_commit"])
            self.assertIsNone(identity["dirty"])

            source.write_text("VALUE = 2\n")
            self.assertNotEqual(first_hash, _package_source_hash(package))


if __name__ == "__main__":
    unittest.main()
