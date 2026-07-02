"""Strategy-matrix execution with per-call tracing and rich task records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from cost.tracker import MODEL_PRICES
from smart_ask import ExecutionRequest, ModelResult, Task

from .artifacts import ResultSink, SCHEMA_VERSION
from .compare import compare, summarize
from .suite import BenchmarkCase, BenchmarkSuite, Evaluation


ApplicationFactory = Callable[[Any, "TraceRecorder"], Any]
ProgressCallback = Callable[[Mapping[str, Any], int, int], None]


@dataclass
class CallTrace:
    """One timed executor invocation captured inside a benchmark task."""

    call_id: str
    channel: str
    request: ExecutionRequest
    result: ModelResult | None
    latency_ms: float
    started_offset_ms: float
    error: dict[str, str] | None = None


@dataclass
class TraceCapture:
    """Executor calls and elapsed wall time for one strategy/task pair."""

    strategy_id: str
    task_id: str
    started_ns: int
    calls: list[CallTrace] = field(default_factory=list)
    total_latency_ms: float | None = None


class TraceRecorder:
    """ContextVar-scoped recorder safe for concurrent benchmark workers."""

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns):
        self.clock = clock
        self._active: ContextVar[TraceCapture | None] = ContextVar(
            f"benchmark_trace_{id(self)}",
            default=None,
        )

    @contextmanager
    def capture(self, strategy_id: str, task_id: str):
        if self._active.get() is not None:
            raise RuntimeError("Benchmark trace captures cannot be nested")
        capture = TraceCapture(strategy_id, task_id, self.clock())
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            capture.total_latency_ms = (self.clock() - capture.started_ns) / 1_000_000
            self._active.reset(token)

    def current(self) -> TraceCapture:
        capture = self._active.get()
        if capture is None:
            raise RuntimeError("Traced executor called outside a benchmark capture")
        return capture


class TracedExecutor:
    """Record requests, outputs, usage, failures, and latency around an executor."""

    def __init__(
        self,
        delegate,
        recorder: TraceRecorder,
        channel: str,
        *,
        clock: Callable[[], int] | None = None,
    ):
        self.delegate = delegate
        self.recorder = recorder
        self.channel = channel
        self.clock = clock or recorder.clock
        self.captures_output = getattr(delegate, "captures_output", False)

    def execute(self, request: ExecutionRequest) -> ModelResult:
        capture = self.recorder.current()
        call_id = f"call-{len(capture.calls) + 1}"
        started_ns = self.clock()
        offset_ms = (started_ns - capture.started_ns) / 1_000_000
        try:
            result = self.delegate.execute(request)
        except Exception as exc:
            capture.calls.append(CallTrace(
                call_id=call_id,
                channel=self.channel,
                request=request,
                result=None,
                latency_ms=(self.clock() - started_ns) / 1_000_000,
                started_offset_ms=offset_ms,
                error=_error(exc, "execution"),
            ))
            raise
        capture.calls.append(CallTrace(
            call_id=call_id,
            channel=self.channel,
            request=request,
            result=result,
            latency_ms=(self.clock() - started_ns) / 1_000_000,
            started_offset_ms=offset_ms,
        ))
        return result


@dataclass(frozen=True)
class BenchmarkRun:
    """Completed matrix records plus benchmark-owned aggregate reports."""

    manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    summaries: Mapping[str, Any]
    comparison: Mapping[str, Any]


def run_matrix(
    suite: BenchmarkSuite,
    strategies: Sequence[Any],
    *,
    application_factory: ApplicationFactory,
    sink: ResultSink,
    workers: int = 1,
    limit: int | None = None,
    price_catalog: Mapping[str, Mapping[str, float]] | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkRun:
    """Run every strategy on the same cases, persist traces, then compare them."""

    if not strategies:
        raise ValueError("At least one strategy is required")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    cases = tuple(suite.load_cases(limit))
    strategy_ids = [_strategy_id(strategy) for strategy in strategies]
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("Strategy names must be unique within a comparison run")

    prices = MODEL_PRICES if price_catalog is None else price_catalog
    configured_models = {
        model
        for strategy in strategies
        for model in _configured_model_ids(strategy)
    }
    unknown_models = sorted(configured_models - set(prices))
    if unknown_models:
        raise ValueError(
            "Missing benchmark prices for configured models: "
            + ", ".join(unknown_models)
        )
    manifest = _build_manifest(suite, strategies, cases, workers, prices)
    sink.start(manifest)
    recorder = TraceRecorder()
    applications = {
        strategy_id: application_factory(strategy, recorder)
        for strategy_id, strategy in zip(strategy_ids, strategies)
    }

    completed = sink.completed_keys
    pending = [
        (strategy, strategy_id, case)
        for strategy, strategy_id in zip(strategies, strategy_ids)
        for case in cases
        if (strategy_id, case.task_id) not in completed
    ]
    total = len(pending)
    records = list(sink.existing_records)
    records_lock = threading.Lock()

    def execute(item):
        strategy, strategy_id, case = item
        return _run_one(
            suite,
            strategy,
            strategy_id,
            applications[strategy_id],
            case,
            recorder,
            prices,
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
                with records_lock:
                    records.append(record)
                    done += 1
                    current = done
                if progress:
                    progress(record, current, total)

    records.sort(key=lambda item: (str(item["strategy_id"]), str(item["task_id"])))
    summaries = summarize(records)
    comparison = compare(records, strategy_order=strategy_ids)
    sink.finalize(summaries, comparison)
    return BenchmarkRun(manifest, tuple(records), summaries, comparison)


def _run_one(
    suite: BenchmarkSuite,
    loaded_strategy: Any,
    strategy_id: str,
    application: Any,
    case: BenchmarkCase,
    recorder: TraceRecorder,
    price_catalog: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    observed_routes: list[Any] = []
    run = None
    evaluation: Evaluation | None = None
    evaluation_latency_ms: float | None = None
    error = None
    stage = "routing"

    with recorder.capture(strategy_id, case.task_id) as capture:
        try:
            stage = "execution"
            run = application.run_detailed(
                Task(case.prompt, task_id=case.task_id),
                on_route=lambda route, _number: observed_routes.append(route),
            )
        except Exception as exc:
            error = _error(exc, stage)

    if run is not None:
        evaluation_started = time.perf_counter_ns()
        try:
            stage = "evaluation"
            evaluation = suite.evaluate(case, run.final_result.text)
        except Exception as exc:
            error = _error(exc, stage)
        finally:
            evaluation_latency_ms = (
                time.perf_counter_ns() - evaluation_started
            ) / 1_000_000

    calls = [_serialize_call(call, price_catalog) for call in capture.calls]
    generation_calls = [call for call in calls if call["channel"] == "generation"]
    routing_events = (
        list(run.routing_events)
        if run is not None
        else [event for route in observed_routes for event in route.routing_events]
    )
    attempts = []
    if run is not None:
        for index, attempt in enumerate(run.attempts, start=1):
            call = generation_calls[index - 1] if index <= len(generation_calls) else None
            attempts.append(_serialize_attempt(index, attempt, call))
    else:
        for index, (route_result, call) in enumerate(
            zip(observed_routes, generation_calls),
            start=1,
        ):
            if call.get("error") is None and call.get("output") is not None:
                attempts.append(
                    _serialize_reconstructed_attempt(index, route_result, call)
                )

    usage = _sum_usage(calls)
    costs = [call["cost_usd"] for call in calls if call["cost_usd"] is not None]
    has_unknown_call_cost = any(call.get("cost_usd") is None for call in calls)
    route = None
    if run is not None:
        route = run.final_route.phase
    elif observed_routes:
        route = observed_routes[-1].phase

    final_output = None
    if run is not None:
        final_output = _serialize_output(run.final_result)

    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "strategy_digest": getattr(loaded_strategy, "digest", None),
        "strategy_path": str(getattr(loaded_strategy, "path", "")),
        "task_id": case.task_id,
        "input": {"prompt": case.prompt},
        "route": route,
        "classifier_decision": next(
            (
                event.outcome
                for event in routing_events
                if event.source == "difficulty-classifier"
            ),
            None,
        ),
        "routing_events": [_serialize_event(event, price_catalog) for event in routing_events],
        "attempts": attempts,
        "calls": calls,
        "final_output": final_output,
        "evaluation": _serialize_evaluation(evaluation),
        "usage": usage,
        "cost_usd": None if has_unknown_call_cost else sum(costs),
        "total_latency_ms": capture.total_latency_ms,
        "evaluation_latency_ms": evaluation_latency_ms,
        "error": error,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_call(
    call: CallTrace,
    price_catalog: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    usage = _usage(call.result.usage if call.result is not None else None)
    return {
        "call_id": call.call_id,
        "channel": call.channel,
        "request": {
            "model": call.request.model,
            "prompt": call.request.prompt,
            "max_tokens": call.request.max_tokens,
            "temperature": call.request.temperature,
        },
        "output": _serialize_output(call.result),
        "usage": usage,
        "cost_usd": _cost(call.request.model, usage, price_catalog),
        "latency_ms": call.latency_ms,
        "started_offset_ms": call.started_offset_ms,
        "error": call.error,
    }


def _serialize_attempt(index: int, attempt: Any, call: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "index": index,
        "route": _serialize_route(attempt.route),
        "output": _serialize_output(attempt.result),
        "usage": _usage(attempt.result.usage),
        "cost_usd": call.get("cost_usd") if call else None,
        "latency_ms": call.get("latency_ms") if call else None,
        "call_id": call.get("call_id") if call else None,
    }


def _serialize_reconstructed_attempt(
    index: int,
    route: Any,
    call: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve a successful attempt when a later step aborted the run."""

    return {
        "index": index,
        "route": _serialize_route(route),
        "output": call["output"],
        "usage": call.get("usage"),
        "cost_usd": call.get("cost_usd"),
        "latency_ms": call.get("latency_ms"),
        "call_id": call.get("call_id"),
        "reconstructed": True,
    }


def _serialize_route(route: Any) -> dict[str, Any]:
    return {
        "action": route.action,
        "phase": route.phase,
        "label": route.label,
        "model": route.model,
        "role": route.role,
        "prompt": route.prompt,
    }


def _serialize_event(
    event: Any,
    price_catalog: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    usage = _usage(event.usage)
    return {
        "source": event.source,
        "outcome": event.outcome,
        "reason": event.reason,
        "model": event.model,
        "role": event.role,
        "usage": usage,
        "cost_usd": _cost(event.model, usage, price_catalog),
    }


def _serialize_output(result: ModelResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "model": result.model,
        "text": result.text,
        "raw_text": result.raw_text,
        "returncode": result.returncode,
    }


def _serialize_evaluation(evaluation: Evaluation | None) -> dict[str, Any]:
    if evaluation is None:
        return {"passed": False, "score": 0.0, "details": {}}
    return {
        "passed": evaluation.passed,
        "score": evaluation.score,
        "details": dict(evaluation.details),
    }


def _usage(raw: Any) -> dict[str, int] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0))
        completion = raw.get("completion_tokens", raw.get("output_tokens", 0))
    else:
        prompt = getattr(raw, "prompt_tokens", getattr(raw, "input_tokens", 0))
        completion = getattr(
            raw,
            "completion_tokens",
            getattr(raw, "output_tokens", 0),
        )
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _sum_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    prompt = sum(int((call.get("usage") or {}).get("prompt_tokens", 0)) for call in calls)
    completion = sum(
        int((call.get("usage") or {}).get("completion_tokens", 0)) for call in calls
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _cost(
    model: str | None,
    usage: Mapping[str, int] | None,
    price_catalog: Mapping[str, Mapping[str, float]],
) -> float | None:
    if model is None or usage is None:
        return None
    prices = price_catalog.get(model)
    if prices is None:
        return None
    return (
        usage["prompt_tokens"] * float(prices["input"])
        + usage["completion_tokens"] * float(prices["output"])
    )


def _error(exc: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _strategy_id(loaded: Any) -> str:
    return str(loaded.config.name)


def _configured_model_ids(loaded: Any) -> set[str]:
    """Return model IDs from a real StrategyConfig; tolerate lightweight test fakes."""

    config = getattr(loaded, "config", None)
    method = getattr(config, "method", None)
    if method is None:
        return set()
    models = set()
    classifier = getattr(method, "classifier", None)
    if classifier is not None:
        models.add(str(classifier.model))
    if getattr(method, "type", None) == "fixed":
        models.add(str(method.model.model))
    else:
        models.add(str(method.easy.model))
        models.add(str(method.hard.model))
    return models


def _build_manifest(
    suite: BenchmarkSuite,
    strategies: Sequence[Any],
    cases: Sequence[BenchmarkCase],
    workers: int,
    price_catalog: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    case_ids = [case.task_id for case in cases]
    case_identity = [
        {
            "task_id": case.task_id,
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "payload_sha256": hashlib.sha256(
                json.dumps(
                    dict(case.payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
        }
        for case in cases
    ]
    case_digest = hashlib.sha256(
        json.dumps(case_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": suite.name,
        "dataset": dict(suite.dataset_identity),
        "case_ids": case_ids,
        "cases": case_identity,
        "case_digest": case_digest,
        "workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategies": [strategy.manifest() for strategy in strategies],
        "pricing": {model: dict(prices) for model, prices in price_catalog.items()},
        "runtime": {
            "python": sys.version,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "implementation": platform.python_implementation(),
            },
            "dependencies": _dependency_versions(),
            "code": _code_identity(),
        },
    }


def _dependency_versions() -> dict[str, str | None]:
    distributions = ("datasets", "openai", "pydantic", "PyYAML")
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _code_identity(repository: Path | None = None) -> dict[str, Any]:
    repository = repository or Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tracked_diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-color", "--binary", "HEAD", "--"],
            cwd=repository,
            capture_output=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        untracked = _identity_untracked_paths(path for path in untracked if path)
        dirty = bool(tracked_diff or untracked)
        return {
            "git_commit": commit,
            "dirty": dirty,
            "dirty_hash": (
                _dirty_worktree_hash(repository, tracked_diff, untracked)
                if dirty
                else None
            ),
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "dirty": None, "dirty_hash": None}


def _dirty_worktree_hash(
    repository: Path,
    tracked_diff: bytes,
    untracked: Sequence[bytes],
) -> str:
    digest = hashlib.sha256()
    _update_hash_frame(digest, b"tracked-diff")
    _update_hash_frame(digest, tracked_diff)

    for relative_path in untracked:
        path = repository / os.fsdecode(relative_path)
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode):
            kind = b"symlink"
            content = os.fsencode(os.readlink(path))
        elif stat.S_ISREG(file_stat.st_mode):
            kind = b"file"
            content = path.read_bytes()
        else:
            kind = b"special"
            content = str(stat.S_IFMT(file_stat.st_mode)).encode("ascii")
        _update_hash_frame(digest, b"untracked")
        _update_hash_frame(digest, relative_path)
        _update_hash_frame(digest, kind)
        _update_hash_frame(digest, content)
    return digest.hexdigest()


def _identity_untracked_paths(paths: Iterable[bytes]) -> list[bytes]:
    """Keep reproducibility inputs while excluding generated benchmark output."""

    candidates = list(paths)
    artifact_roots = {
        Path(os.fsdecode(path)).parent
        for path in candidates
        if Path(os.fsdecode(path)).name == "records.jsonl"
    }
    generated_directories = {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "artifacts",
        "build",
        "cache",
        "dist",
        "outputs",
        "results",
    }
    source_suffixes = {
        ".cfg",
        ".ini",
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }

    selected = []
    for raw_path in candidates:
        path = Path(os.fsdecode(raw_path))
        if any(
            path == root or root in path.parents
            for root in artifact_roots
        ):
            continue
        if any(
            part.lower() in generated_directories
            or part.lower().endswith(".egg-info")
            for part in path.parts[:-1]
        ):
            continue
        if path.name == "smart-ask" or path.suffix.lower() in source_suffixes:
            selected.append(raw_path)
    return sorted(selected)


def _update_hash_frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
