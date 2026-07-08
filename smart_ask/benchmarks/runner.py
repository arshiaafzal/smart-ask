"""Strategy-matrix execution with per-call tracing and rich task records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..domain import ModelResult, RunResult, Task
from ..executors.base import ModelExecutor
from ..metrics import (
    CallRecord,
    DEFAULT_PRICE_CATALOG,
    PriceCatalog,
    StatsCollector,
)
from ..strategy.schema import FixedMethodConfig, StrategyConfig

from .artifact_schema import SCHEMA_VERSION
from .artifacts import ResultSink
from .compare import compare, summarize
from .run_manifest import build_manifest
from .suite import (
    BenchmarkCase,
    BenchmarkStrategy,
    BenchmarkSuite,
    Evaluation,
    _thaw_json,
)


class BenchmarkApplication(Protocol):
    """Instrumented, isolated application required for one benchmark case."""

    @property
    def executor(self) -> ModelExecutor:
        ...

    @property
    def stats_collector(self) -> StatsCollector:
        ...

    @property
    def strategy_id(self) -> str:
        ...

    @property
    def metrics_executors(self) -> Mapping[str, tuple[ModelExecutor, ...]]:
        ...

    def run_detailed(
        self,
        task: Task,
        on_route: Callable[..., None] | None = None,
        on_result: Callable[..., None] | None = None,
    ) -> RunResult:
        ...


ApplicationFactory = Callable[
    [BenchmarkStrategy, StatsCollector],
    BenchmarkApplication,
]
ProgressCallback = Callable[[Mapping[str, Any], int, int], None]


@dataclass(frozen=True)
class BenchmarkRun:
    """Completed matrix records plus benchmark-owned aggregate reports."""

    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    summaries: Mapping[str, Any]
    comparison: Mapping[str, Any]


def run_matrix(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    *,
    application_factory: ApplicationFactory,
    sink: ResultSink,
    workers: int = 1,
    limit: int | None = None,
    price_catalog: PriceCatalog | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkRun:
    """Run every strategy/case in an isolated application, persist, and compare."""

    if not strategies:
        raise ValueError("At least one strategy is required")
    if not callable(application_factory):
        raise TypeError("application_factory must be callable")
    for strategy in strategies:
        _validate_benchmark_strategy(strategy)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")

    strategy_ids = [_strategy_id(strategy) for strategy in strategies]
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("Strategy names must be unique within a comparison run")

    prices = DEFAULT_PRICE_CATALOG if price_catalog is None else price_catalog
    if not isinstance(prices, PriceCatalog):
        raise TypeError("price_catalog must be a PriceCatalog")
    configured_models = {
        model
        for strategy in strategies
        for model in _configured_model_ids(strategy)
    }
    unknown_models = sorted(configured_models - set(prices.prices))
    if unknown_models:
        raise ValueError(
            "Missing benchmark prices for configured models: "
            + ", ".join(unknown_models)
        )
    cases = tuple(suite.load_cases(limit))
    if not cases:
        raise ValueError("Benchmark suite did not provide any cases")
    case_ids = [case.task_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Benchmark case task IDs must be unique")
    manifest = sink.start(build_manifest(suite, strategies, cases, workers, prices))
    try:
        return _run_started_matrix(
            suite,
            strategies,
            application_factory=application_factory,
            sink=sink,
            workers=workers,
            progress=progress,
            cases=cases,
            strategy_ids=strategy_ids,
            price_catalog=prices,
            manifest=manifest,
        )
    finally:
        sink.close()


def _run_started_matrix(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    *,
    application_factory: ApplicationFactory,
    sink: ResultSink,
    workers: int,
    progress: ProgressCallback | None,
    cases: Sequence[BenchmarkCase],
    strategy_ids: Sequence[str],
    price_catalog: PriceCatalog,
    manifest: Mapping[str, Any],
) -> BenchmarkRun:
    """Execute a validated matrix while ``run_matrix`` owns sink cleanup."""

    completed = sink.completed_keys
    pending = [
        (strategy, strategy_id, case)
        for strategy, strategy_id in zip(strategies, strategy_ids)
        for case in cases
        if (strategy_id, case.task_id) not in completed
    ]
    total = len(pending)
    records = list(sink.existing_records)
    recorder = StatsCollector(
        price_catalog=price_catalog,
        require_active_capture=True,
    )
    applications = {}
    application_ids: set[int] = set()
    executor_ids: set[int] = set()
    for strategy, strategy_id, case in pending:
        application = application_factory(strategy, recorder)
        _validate_benchmark_application(strategy, application, recorder)
        application_id = id(application)
        if application_id in application_ids:
            raise ValueError(
                "application_factory must return a distinct application for "
                "every pending strategy/task pair"
            )
        application_ids.add(application_id)
        for channel_executors in application.metrics_executors.values():
            for instrumented_executor in channel_executors:
                executor_id = id(instrumented_executor)
                if executor_id in executor_ids:
                    raise ValueError(
                        "application_factory must not share instrumented executors "
                        "between strategy/task pairs"
                    )
                executor_ids.add(executor_id)
        applications[(strategy_id, case.task_id)] = application

    def execute(item):
        strategy, strategy_id, case = item
        return _run_one(
            suite,
            strategy,
            strategy_id,
            applications[(strategy_id, case.task_id)],
            case,
            recorder,
        )

    if workers == 1:
        iterator: Iterable[Mapping[str, Any]] = map(execute, pending)
        for done, record in enumerate(iterator, start=1):
            sink.append(record)
            records.append(record)
            if progress:
                progress(record, done, total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, item): item for item in pending}
            done = 0
            for future in as_completed(futures):
                record = future.result()
                sink.append(record)
                records.append(record)
                done += 1
                current = done
                if progress:
                    progress(record, current, total)

    records.sort(key=lambda item: (str(item["strategy_id"]), str(item["task_id"])))
    summaries = summarize(records, manifest=manifest)
    comparison = compare(
        records,
        strategy_order=strategy_ids,
        manifest=manifest,
    )
    sink.finalize(summaries, comparison)
    return BenchmarkRun(manifest, tuple(records), summaries, comparison)


def _run_one(
    suite: BenchmarkSuite,
    loaded_strategy: BenchmarkStrategy,
    strategy_id: str,
    application: BenchmarkApplication,
    case: BenchmarkCase,
    recorder: StatsCollector,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    observed_routes: list[Any] = []
    run = None
    evaluation: Evaluation | None = None
    evaluation_latency_ms: float | None = None
    error = None
    stage = "routing"

    def observe_route(route_result: Any, _number: int) -> None:
        nonlocal stage
        observed_routes.append(route_result)
        stage = "execution"

    def observe_result(_result: ModelResult, _number: int) -> None:
        nonlocal stage
        stage = "routing"

    with recorder.capture(strategy_id, case.task_id) as capture:
        try:
            run = application.run_detailed(
                Task(case.prompt, task_id=case.task_id),
                on_route=observe_route,
                on_result=observe_result,
            )
        except Exception as exc:
            error = _error(exc, stage)

    if run is not None:
        evaluation_started = time.perf_counter_ns()
        try:
            stage = "evaluation"
            evaluation = suite.evaluate(case, run.final_result.text)
            if not isinstance(evaluation, Evaluation):
                raise TypeError("benchmark evaluator must return an Evaluation")
        except Exception as exc:
            evaluation = None
            error = _error(exc, stage)
        finally:
            evaluation_latency_ms = (
                time.perf_counter_ns() - evaluation_started
            ) / 1_000_000

    calls = [_serialize_call(call) for call in capture.calls]
    generation_calls = [call for call in calls if call["channel"] == "generation"]
    event_values = (
        list(run.routing_events)
        if run is not None
        else [event for route in observed_routes for event in route.routing_events]
    )
    _validate_generation_coverage(run, observed_routes, generation_calls)
    attempts: list[dict[str, Any]] = []
    if run is not None:
        for index, attempt in enumerate(run.attempts, start=1):
            attempts.append(_serialize_attempt(
                index,
                attempt.route,
                generation_calls[index - 1],
            ))
    else:
        for index, (route_result, call) in enumerate(zip(
            observed_routes,
            generation_calls,
        ), start=1):
            attempts.append(_serialize_attempt(
                index,
                route_result,
                call,
                reconstructed=True,
            ))

    route = None
    if run is not None:
        route = run.final_route.phase
    elif observed_routes:
        route = observed_routes[-1].phase

    final_output = None
    if run is not None:
        final_output = _serialize_output(run.final_result)

    attempted_routes = (
        [attempt.route for attempt in run.attempts]
        if run is not None
        else observed_routes
    )
    run_stats = capture.stats.with_routing_counts(
        generation_attempts=len(attempted_routes),
        routing_events=len(event_values),
    ).with_outcome(_task_outcome(evaluation, error))
    routing_events = _serialize_events(event_values, calls)

    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "strategy_digest": loaded_strategy.digest,
        "task_id": case.task_id,
        "input": {"prompt": case.prompt},
        "route": route,
        "classifier_decision": next(
            (
                event.outcome
                for event in event_values
                if event.source == "difficulty-classifier"
            ),
            None,
        ),
        "routing_events": routing_events,
        "attempts": attempts,
        "calls": calls,
        "final_output": final_output,
        "evaluation": _serialize_evaluation(evaluation),
        "metrics": run_stats.to_dict(include_calls=False),
        "evaluation_latency_ms": evaluation_latency_ms,
        "error": error,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_call(call: CallRecord) -> dict[str, Any]:
    payload = call.stats.to_dict()
    payload.update({
        "request": {
            "model": call.request.model,
            "role": call.request.role,
            "prompt": call.request.prompt,
            "max_tokens": call.request.max_tokens,
            "temperature": call.request.temperature,
        },
        "output": _serialize_output(call.result),
    })
    return payload


def _validate_generation_coverage(
    run: Any,
    observed_routes: Sequence[Any],
    generation_calls: Sequence[Mapping[str, Any]],
) -> None:
    """Require one observed generation call for every attempted route."""

    expected_generation = (
        len(run.attempts) if run is not None else len(observed_routes)
    )
    if len(generation_calls) != expected_generation:
        raise RuntimeError(
            "Benchmark instrumentation coverage mismatch: generation calls "
            f"{len(generation_calls)}/{expected_generation}"
        )


def _serialize_attempt(
    index: int,
    route: Any,
    call: Mapping[str, Any],
    *,
    reconstructed: bool = False,
) -> dict[str, Any]:
    payload = {
        "index": index,
        "route": _serialize_route(route),
        "call_id": call["call_id"],
        "status": call["status"],
    }
    if reconstructed:
        payload["reconstructed"] = True
    return payload


def _serialize_route(route: Any) -> dict[str, Any]:
    return {
        "action": route.action,
        "phase": route.phase,
        "label": route.label,
        "model": route.model,
        "role": route.role,
        "prompt": route.prompt,
    }


def _serialize_events(
    events: Sequence[Any],
    calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize semantic events and link classifier decisions to call evidence."""

    classifier_calls = [
        str(call["call_id"])
        for call in calls
        if call["channel"] == "classifier"
    ]
    classifier_event_count = sum(
        event.source == "difficulty-classifier" for event in events
    )
    serialized = []
    classifier_index = 0
    for event in events:
        call_ids: list[str] = []
        if event.source == "difficulty-classifier":
            if classifier_event_count == 1:
                call_ids = classifier_calls
            elif len(classifier_calls) == classifier_event_count:
                call_ids = [classifier_calls[classifier_index]]
            classifier_index += 1
        serialized.append({
            "source": event.source,
            "outcome": event.outcome,
            "reason": event.reason,
            "model": event.model,
            "call_ids": call_ids,
        })
    return serialized


def _serialize_output(result: ModelResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "model": result.model,
        "text": result.text,
        "raw_text": result.raw_text,
    }


def _serialize_evaluation(evaluation: Evaluation | None) -> dict[str, Any]:
    if evaluation is None:
        return {"passed": False, "score": 0.0, "details": {}}
    return {
        "passed": evaluation.passed,
        "score": evaluation.score,
        "details": _thaw_json(evaluation.details),
    }


def _error(exc: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _task_outcome(
    evaluation: Evaluation | None,
    error: Mapping[str, Any] | None,
) -> str:
    if error is not None:
        return f"{error['stage']}_error"
    if evaluation is None:
        raise RuntimeError("completed benchmark task is missing an evaluation")
    return "passed" if evaluation.passed else "incorrect"


def _strategy_id(loaded: BenchmarkStrategy) -> str:
    return loaded.config.name


def _configured_model_ids(loaded: BenchmarkStrategy) -> set[str]:
    method = loaded.config.method
    if isinstance(method, FixedMethodConfig):
        return {method.model.model}
    return {method.classifier.model, method.easy.model, method.hard.model}


def _validate_benchmark_strategy(strategy: BenchmarkStrategy) -> None:
    try:
        config = strategy.config
        digest = strategy.digest
        manifest = strategy.manifest
    except AttributeError as exc:
        raise TypeError("benchmark strategy does not implement its contract") from exc
    if not isinstance(config, StrategyConfig):
        raise TypeError("benchmark strategies require a validated StrategyConfig")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("benchmark strategies require a lowercase SHA-256 digest")
    if not callable(manifest):
        raise TypeError("benchmark strategies require a callable manifest")


def _validate_benchmark_application(
    strategy: BenchmarkStrategy,
    application: Any,
    recorder: StatsCollector,
) -> None:
    """Reject missing instrumentation before any benchmark case can execute."""

    strategy_id = _strategy_id(strategy)
    if getattr(application, "strategy_id", None) != strategy_id:
        raise ValueError(
            f"benchmark application strategy_id must equal {strategy_id!r}"
        )
    if getattr(application, "stats_collector", None) is not recorder:
        raise ValueError(
            f"benchmark strategy {strategy_id!r} must retain the supplied "
            "StatsCollector"
        )
    if not callable(getattr(application, "run_detailed", None)):
        raise TypeError(
            f"benchmark strategy {strategy_id!r} application must expose "
            "callable run_detailed"
        )
    executor = getattr(application, "executor", None)
    if getattr(executor, "captures_output", None) is not True:
        raise ValueError(
            f"benchmark strategy {strategy_id!r} must use a generation "
            "executor that captures output"
        )
    metrics_executors = getattr(application, "metrics_executors", None)
    if not isinstance(metrics_executors, Mapping):
        raise TypeError(
            f"benchmark strategy {strategy_id!r} application must expose a "
            "metrics_executors mapping"
        )
    expected_channels = {"generation"}
    if not isinstance(strategy.config.method, FixedMethodConfig):
        expected_channels.add("classifier")
    if set(metrics_executors) != expected_channels:
        raise ValueError(
            f"benchmark strategy {strategy_id!r} metrics_executors must contain "
            f"exactly {sorted(expected_channels)!r}"
        )
    for channel in sorted(expected_channels):
        channel_executors = metrics_executors[channel]
        if not isinstance(channel_executors, tuple) or not channel_executors:
            raise TypeError(
                f"benchmark strategy {strategy_id!r} metrics_executors[{channel!r}] "
                "must be a non-empty tuple"
            )
        if any(
            not recorder.is_instrumented(item, channel=channel)
            for item in channel_executors
        ):
            raise ValueError(
                f"benchmark strategy {strategy_id!r} {channel} executor is not "
                "instrumented by the supplied StatsCollector"
            )
    if executor not in metrics_executors["generation"]:
        raise ValueError(
            f"benchmark strategy {strategy_id!r} public executor is not its "
            "instrumented generation executor"
        )
