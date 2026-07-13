"""Asynchronous strategy-engine execution for benchmark case matrices."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..conversation.domain import ConversationEvent, thaw_value
from ..conversation.metrics import run_record_dict
from ..conversation.model import CompletedRun, Conversation, RunMetadata, RunRecord
from ..metrics import DEFAULT_PRICE_CATALOG, PriceCatalog
from ..strategy.schema import StrategyConfig

from .suite import (
    BenchmarkCase,
    BenchmarkStrategy,
    BenchmarkSuite,
    Evaluation,
    _thaw_json,
)
from .compare import compare, summarize


class ResultSink(Protocol):
    """Persistence boundary for canonical benchmark records."""

    @property
    def completed_keys(self) -> set[tuple[str, str]]: ...

    @property
    def existing_records(self) -> list[dict[str, Any]]: ...

    def start(self, manifest: Mapping[str, Any]) -> dict[str, Any]: ...

    def append(self, record: Mapping[str, Any]) -> None: ...

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None: ...

    def close(self) -> None: ...


class BenchmarkEngine(Protocol):
    """Minimal strategy-engine surface required by benchmark execution."""

    def start(
        self,
        conversation: Conversation,
        metadata: RunMetadata,
    ) -> Any: ...

    async def aclose(self) -> None: ...


EngineFactory = Callable[[BenchmarkStrategy], BenchmarkEngine]
DeploymentManifestFactory = Callable[[BenchmarkStrategy], Mapping[str, Any]]
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
    engine_factory: EngineFactory,
    sink: ResultSink,
    workers: int = 1,
    limit: int | None = None,
    price_catalog: PriceCatalog | None = None,
    deployment_manifest_factory: DeploymentManifestFactory | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkRun:
    """Synchronously run a matrix using asynchronous strategy engines.

    Async applications should call :func:`run_matrix_async` directly.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_matrix_async(
            suite,
            strategies,
            engine_factory=engine_factory,
            sink=sink,
            workers=workers,
            limit=limit,
            price_catalog=price_catalog,
            deployment_manifest_factory=deployment_manifest_factory,
            progress=progress,
        ))
    raise RuntimeError(
        "run_matrix cannot run inside an active event loop; "
        "await run_matrix_async instead"
    )


async def run_matrix_async(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    *,
    engine_factory: EngineFactory,
    sink: ResultSink,
    workers: int = 1,
    limit: int | None = None,
    price_catalog: PriceCatalog | None = None,
    deployment_manifest_factory: DeploymentManifestFactory | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkRun:
    """Run every strategy/case pair through an isolated strategy engine."""

    _validate_inputs(strategies, engine_factory, workers, limit)
    strategy_ids = [_strategy_id(strategy) for strategy in strategies]
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("Strategy names must be unique within a comparison run")

    prices = DEFAULT_PRICE_CATALOG if price_catalog is None else price_catalog
    if not isinstance(prices, PriceCatalog):
        raise TypeError("price_catalog must be a PriceCatalog")
    cases = tuple(suite.load_cases(limit))
    if not cases:
        raise ValueError("Benchmark suite did not provide any cases")
    case_ids = [case.task_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Benchmark case task IDs must be unique")

    manifest = sink.start(_build_manifest(
        suite,
        strategies,
        cases,
        workers,
        prices,
        deployment_manifest_factory,
    ))
    try:
        return await _run_started_matrix(
            suite,
            strategies,
            engine_factory=engine_factory,
            sink=sink,
            workers=workers,
            progress=progress,
            cases=cases,
            strategy_ids=strategy_ids,
            manifest=manifest,
            price_catalog=prices,
        )
    finally:
        sink.close()


async def _run_started_matrix(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    *,
    engine_factory: EngineFactory,
    sink: ResultSink,
    workers: int,
    progress: ProgressCallback | None,
    cases: Sequence[BenchmarkCase],
    strategy_ids: Sequence[str],
    manifest: Mapping[str, Any],
    price_catalog: PriceCatalog,
) -> BenchmarkRun:
    completed = sink.completed_keys
    pending = [
        (strategy, strategy_id, case)
        for strategy, strategy_id in zip(strategies, strategy_ids, strict=True)
        for case in cases
        if (strategy_id, case.task_id) not in completed
    ]
    records = list(sink.existing_records)
    engines: dict[tuple[str, str], BenchmarkEngine] = {}
    engine_ids: set[int] = set()
    for strategy, strategy_id, case in pending:
        engine = engine_factory(strategy)
        _validate_engine(strategy_id, engine)
        if id(engine) in engine_ids:
            raise ValueError(
                "engine_factory must return a distinct engine for every "
                "pending strategy/task pair"
            )
        engine_ids.add(id(engine))
        engines[(strategy_id, case.task_id)] = engine

    semaphore = asyncio.Semaphore(workers)

    async def execute(item: tuple[BenchmarkStrategy, str, BenchmarkCase]):
        strategy, strategy_id, case = item
        async with semaphore:
            return await _run_one(
                suite,
                strategy,
                strategy_id,
                engines[(strategy_id, case.task_id)],
                case,
            )

    tasks = [asyncio.create_task(execute(item)) for item in pending]
    total = len(tasks)
    try:
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            record = await task
            sink.append(record)
            records.append(record)
            if progress is not None:
                progress(record, done, total)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        close_results = await asyncio.gather(
            *(engine.aclose() for engine in engines.values()),
            return_exceptions=True,
        )
        close_errors = [
            result for result in close_results if isinstance(result, BaseException)
        ]
        if close_errors:
            raise RuntimeError(
                f"failed to close {len(close_errors)} benchmark engine(s): "
                f"{close_errors[0]}"
            ) from close_errors[0]

    records.sort(key=lambda item: (str(item["strategy_id"]), str(item["task_id"])))
    summaries = summarize(records, price_catalog=price_catalog)
    comparison = compare(summaries, strategy_order=strategy_ids)
    sink.finalize(summaries, comparison)
    return BenchmarkRun(manifest, tuple(records), summaries, comparison)


async def _run_one(
    suite: BenchmarkSuite,
    loaded_strategy: BenchmarkStrategy,
    strategy_id: str,
    engine: BenchmarkEngine,
    case: BenchmarkCase,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    completed: CompletedRun | None = None
    evaluation: Evaluation | None = None
    evaluation_latency_ms: float | None = None
    error: dict[str, str] | None = None

    try:
        handle = engine.start(
            Conversation.from_text(case.prompt),
            RunMetadata(
                strategy_name=strategy_id,
                strategy_digest=loaded_strategy.digest,
                request_id=case.task_id,
                extensions={"benchmark": suite.name},
            ),
        )
        events: list[ConversationEvent] = []
        try:
            async for event in handle.events():
                events.append(event)
        except Exception as exc:
            error = _error(exc, "execution")
        record = await handle.result()
        if not isinstance(record, RunRecord):
            raise TypeError("benchmark run handle must return a RunRecord")
        completed = CompletedRun(tuple(events), record)
        if record.status != "completed" and error is None:
            error = {
                "stage": "execution",
                "type": "RunError",
                "message": record.error or f"run ended as {record.status}",
            }
    except Exception as exc:
        if error is None:
            error = _error(exc, "execution")

    output = _serialize_output(completed.events if completed is not None else ())
    if completed is not None and completed.record.status == "completed":
        evaluation_started = time.perf_counter_ns()
        try:
            evaluation = await asyncio.to_thread(
                suite.evaluate,
                case,
                output["text"],
            )
            if not isinstance(evaluation, Evaluation):
                raise TypeError("benchmark evaluator must return an Evaluation")
        except Exception as exc:
            error = _error(exc, "evaluation")
            evaluation = None
        finally:
            evaluation_latency_ms = (
                time.perf_counter_ns() - evaluation_started
            ) / 1_000_000

    canonical = (
        run_record_dict(completed.record)
        if completed is not None
        else None
    )
    run = None
    decisions: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    provider_requests: list[dict[str, Any]] = []
    final_call = None
    if canonical is not None:
        decisions = canonical.pop("decisions")
        model_calls = canonical.pop("model_calls")
        provider_requests = canonical.pop("provider_requests")
        final_call = canonical.pop("final_call_id")
        run = canonical

    return {
        "schema": "smart-ask.benchmark-result/v2",
        "strategy_id": strategy_id,
        "strategy_digest": loaded_strategy.digest,
        "task_id": case.task_id,
        "input": {
            "prompt": case.prompt,
            "payload": _thaw_json(case.payload),
        },
        "run": run,
        "decisions": decisions,
        "model_calls": model_calls,
        "provider_requests": provider_requests,
        "final_call": final_call,
        "output": output,
        "evaluation": _serialize_evaluation(evaluation),
        "evaluation_latency_ms": evaluation_latency_ms,
        "error": error,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_output(events: Sequence[ConversationEvent]) -> dict[str, Any]:
    text_parts: list[str] = []
    serialized_events: list[dict[str, Any]] = []
    for event in events:
        serialized_events.append({
            "kind": event.kind,
            "data": thaw_value(event.data),
        })
        if event.kind != "content_delta":
            continue
        delta = event.data.get("delta")
        if not isinstance(delta, Mapping) or delta.get("type") != "text":
            continue
        value = delta.get("text")
        if isinstance(value, str):
            text_parts.append(value)
    return {"text": "".join(text_parts), "events": serialized_events}


def _serialize_evaluation(
    evaluation: Evaluation | None,
) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "passed": evaluation.passed,
        "score": evaluation.score,
        "details": _thaw_json(evaluation.details),
    }


def _error(exc: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _strategy_id(loaded: BenchmarkStrategy) -> str:
    return loaded.config.name


def _build_manifest(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    cases: Sequence[BenchmarkCase],
    workers: int,
    price_catalog: PriceCatalog,
    deployment_manifest_factory: DeploymentManifestFactory | None,
) -> dict[str, Any]:
    case_identity = [
        {
            "task_id": case.task_id,
            "prompt_sha256": hashlib.sha256(
                case.prompt.encode("utf-8")
            ).hexdigest(),
            "payload_sha256": hashlib.sha256(json.dumps(
                _thaw_json(case.payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest(),
        }
        for case in cases
    ]
    strategy_manifests = []
    for strategy in strategies:
        strategy_manifest = strategy.manifest()
        if deployment_manifest_factory is None:
            deployment = {
                "status": "unresolved",
                "reason": "no deployment manifest factory was supplied",
            }
        else:
            deployment = dict(deployment_manifest_factory(strategy))
            if deployment.get("status") != "resolved":
                raise ValueError(
                    "deployment manifests supplied by a factory must be resolved"
                )
            if not isinstance(deployment.get("digest"), str):
                raise TypeError("resolved deployment manifests require a digest")
            if not isinstance(deployment.get("targets"), (list, tuple)):
                raise TypeError("resolved deployment manifests require targets")
        strategy_manifest["deployment"] = deployment
        strategy_manifests.append(strategy_manifest)

    return {
        "schema": "smart-ask.benchmark-run/v2",
        "benchmark": suite.name,
        "dataset": dict(suite.dataset_identity),
        "evaluator": dict(suite.evaluator_identity),
        "workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": case_identity,
        "strategies": strategy_manifests,
        "pricing": price_catalog.to_dict(),
    }


def _validate_inputs(
    strategies: Sequence[BenchmarkStrategy],
    engine_factory: EngineFactory,
    workers: int,
    limit: int | None,
) -> None:
    if not strategies:
        raise ValueError("At least one strategy is required")
    if not callable(engine_factory):
        raise TypeError("engine_factory must be callable")
    for strategy in strategies:
        _validate_benchmark_strategy(strategy)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")


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


def _validate_engine(strategy_id: str, engine: Any) -> None:
    if not callable(getattr(engine, "start", None)):
        raise TypeError(
            f"benchmark strategy {strategy_id!r} engine must expose "
            "start"
        )
    if not callable(getattr(engine, "aclose", None)):
        raise TypeError(
            f"benchmark strategy {strategy_id!r} engine must expose "
            "async aclose"
        )
