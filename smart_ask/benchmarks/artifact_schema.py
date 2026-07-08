"""Strict schema and integrity validation for benchmark run artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from .._numeric import checked_fsum, is_finite_real
from ..metrics import (
    METRICS_WIRE_SCHEMA,
    PriceCatalog,
    TokenUsage,
    aggregate_metric_payloads,
    price_usage,
)
from ..strategy.loader import compute_strategy_digest
from ..strategy.schema import (
    CascadeMethodConfig,
    FilePromptConfig,
    FixedMethodConfig,
    InlinePromptConfig,
    StrategyConfig,
)


SCHEMA_VERSION = 5

_REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "benchmark",
    "dataset",
    "evaluator",
    "strategies",
    "case_ids",
    "cases",
    "case_digest",
    "pricing",
    "metrics",
    "workers",
    "runtime",
    "created_at",
)
_MANIFEST_FIELDS = set(_REQUIRED_MANIFEST_FIELDS)
_STRATEGY_FIELDS = {"name", "digest", "config", "prompts"}
_PROMPT_FIELDS = {"declared_path", "sha256", "text"}
_CASE_FIELDS = {"task_id", "prompt_sha256", "payload_sha256"}
_RUNTIME_FIELDS = {"python", "platform", "dependencies", "code"}
_PLATFORM_FIELDS = {"system", "release", "machine", "implementation"}
_DEPENDENCY_FIELDS = {"datasets", "openai", "pydantic", "PyYAML"}
_CODE_FIELDS = {
    "package_version",
    "package_hash",
    "git_commit",
    "dirty",
    "dirty_hash",
}
_PRICING_FIELDS = {
    "catalog_id",
    "effective_date",
    "source",
    "prices",
    "currency",
}
_METRICS_DESCRIPTOR_FIELDS = {
    "schema",
    "scope",
    "record_unit",
    "interaction_unit",
}

# ``created_at`` identifies the persisted run rather than its resumability. A
# caller builds a fresh proposed manifest before discovering the original one,
# so start() returns the persisted manifest on resume.
_RESUME_IDENTITY_FIELDS = tuple(
    field for field in _REQUIRED_MANIFEST_FIELDS if field != "created_at"
)

_RECORD_FIELDS = {
    "schema_version",
    "strategy_id",
    "strategy_digest",
    "task_id",
    "input",
    "route",
    "classifier_decision",
    "routing_events",
    "attempts",
    "calls",
    "final_output",
    "evaluation",
    "metrics",
    "evaluation_latency_ms",
    "error",
    "started_at",
    "finished_at",
}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    root = _record_object(manifest, "manifest")
    _record_keys(root, _MANIFEST_FIELDS, "manifest")
    _require_schema_version(manifest, "manifest")
    _record_identifier(manifest["benchmark"], "manifest.benchmark")
    _validate_nonempty_descriptor(manifest["dataset"], "manifest.dataset")
    _validate_nonempty_descriptor(manifest["evaluator"], "manifest.evaluator")
    _validate_manifest_strategies(manifest["strategies"])
    _validate_manifest_cases(manifest)
    _validate_manifest_pricing(manifest["pricing"])
    _validate_metrics_descriptor(manifest["metrics"])
    workers = manifest["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Manifest field 'workers' must be a positive integer")
    _validate_runtime(manifest["runtime"])
    _validate_aware_datetime(manifest["created_at"], "manifest.created_at")


def _validate_nonempty_descriptor(value: Any, path: str) -> None:
    descriptor = _record_object(value, path)
    if not descriptor:
        raise ValueError(f"{path} must not be empty")
    _validate_json_value(descriptor, path)


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError(f"{path} must contain finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _record_string(key, f"{path} key")
            _validate_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _validate_manifest_strategies(value: Any) -> None:
    strategies = _record_list(value, "manifest.strategies")
    if not strategies:
        raise ValueError("manifest.strategies must not be empty")
    names: set[str] = set()
    for index, raw_strategy in enumerate(strategies, start=1):
        path = f"manifest.strategies[{index}]"
        strategy = _record_object(raw_strategy, path)
        _record_keys(strategy, _STRATEGY_FIELDS, path)
        name = _record_identifier(strategy["name"], f"{path}.name")
        if name in names:
            raise ValueError(f"manifest contains duplicate strategy name {name!r}")
        names.add(name)
        declared_digest = _validate_sha256(strategy["digest"], f"{path}.digest")
        raw_config = _record_object(strategy["config"], f"{path}.config")
        if not raw_config:
            raise ValueError(f"{path}.config must not be empty")
        _validate_json_value(raw_config, f"{path}.config")
        config = StrategyConfig.model_validate(raw_config)
        canonical_config = config.model_dump(mode="json")
        if dict(raw_config) != canonical_config:
            raise ValueError(f"{path}.config must be a complete canonical snapshot")
        if config.name != name:
            raise ValueError(f"{path}.config.name does not match strategy name")
        prompts = _record_list(strategy["prompts"], f"{path}.prompts")
        prompt_paths: set[str] = set()
        declared_prompt_hashes: dict[str, str] = {}
        for prompt_index, raw_prompt in enumerate(prompts, start=1):
            prompt_path = f"{path}.prompts[{prompt_index}]"
            prompt = _record_object(raw_prompt, prompt_path)
            _record_keys(prompt, _PROMPT_FIELDS, prompt_path)
            declared_path = _record_identifier(
                prompt["declared_path"],
                f"{prompt_path}.declared_path",
            )
            if declared_path in prompt_paths:
                raise ValueError(f"{path} contains duplicate prompt paths")
            prompt_paths.add(declared_path)
            text = _record_string(prompt["text"], f"{prompt_path}.text")
            if not text.strip():
                raise ValueError(f"{prompt_path}.text must contain non-whitespace text")
            expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
            digest = _validate_sha256(prompt["sha256"], f"{prompt_path}.sha256")
            if digest != expected:
                raise ValueError(f"{prompt_path}.sha256 does not match its text")
            declared_prompt_hashes[declared_path] = digest

        expected_prompt_paths = {
            prompt.path for prompt in _iter_file_prompts(config)
        }
        if prompt_paths != expected_prompt_paths:
            raise ValueError(
                f"{path}.prompts must exactly match declared file prompts"
            )
        expected_strategy_digest = compute_strategy_digest(
            config,
            declared_prompt_hashes,
        )
        if declared_digest != expected_strategy_digest:
            raise ValueError(f"{path}.digest does not match config and prompts")


def _iter_file_prompts(value: Any) -> Iterable[FilePromptConfig]:
    if isinstance(value, FilePromptConfig):
        yield value
        return
    if isinstance(value, BaseModel):
        for field_name in value.__class__.model_fields:
            yield from _iter_file_prompts(getattr(value, field_name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_file_prompts(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_file_prompts(item)


def _validate_manifest_cases(manifest: Mapping[str, Any]) -> None:
    case_ids = _record_list(manifest["case_ids"], "manifest.case_ids")
    if not case_ids:
        raise ValueError("manifest.case_ids must not be empty")
    for index, case_id in enumerate(case_ids, start=1):
        _record_identifier(case_id, f"manifest.case_ids[{index}]")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("manifest.case_ids must be unique")

    cases = _record_list(manifest["cases"], "manifest.cases")
    normalized_cases: list[Mapping[str, Any]] = []
    for index, raw_case in enumerate(cases, start=1):
        path = f"manifest.cases[{index}]"
        case = _record_object(raw_case, path)
        _record_keys(case, _CASE_FIELDS, path)
        _record_identifier(case["task_id"], f"{path}.task_id")
        _validate_sha256(case["prompt_sha256"], f"{path}.prompt_sha256")
        _validate_sha256(case["payload_sha256"], f"{path}.payload_sha256")
        normalized_cases.append(case)
    if [case["task_id"] for case in normalized_cases] != case_ids:
        raise ValueError("manifest.cases must match case_ids in order")
    expected_digest = hashlib.sha256(
        json.dumps(
            normalized_cases,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    digest = _validate_sha256(manifest["case_digest"], "manifest.case_digest")
    if digest != expected_digest:
        raise ValueError("manifest.case_digest does not match manifest.cases")


def _validate_manifest_pricing(value: Any) -> PriceCatalog:
    pricing = _record_object(value, "manifest.pricing")
    _record_keys(pricing, _PRICING_FIELDS, "manifest.pricing")
    if pricing["currency"] != "USD":
        raise ValueError("manifest.pricing.currency must be 'USD'")
    effective_date = _record_identifier(
        pricing["effective_date"],
        "manifest.pricing.effective_date",
    )
    try:
        date.fromisoformat(effective_date)
    except ValueError as exc:
        raise ValueError("manifest.pricing.effective_date must be an ISO date") from exc
    return PriceCatalog(
        catalog_id=pricing["catalog_id"],
        effective_date=effective_date,
        source=pricing["source"],
        prices=_record_object(pricing["prices"], "manifest.pricing.prices"),
    )


def _validate_metrics_descriptor(value: Any) -> None:
    metrics = _record_object(value, "manifest.metrics")
    _record_keys(metrics, _METRICS_DESCRIPTOR_FIELDS, "manifest.metrics")
    expected = {
        "schema": METRICS_WIRE_SCHEMA,
        "scope": "run",
        "record_unit": "strategy-task",
        "interaction_unit": "model-executor-call",
    }
    if dict(metrics) != expected:
        raise ValueError("manifest.metrics descriptor is invalid")


def _validate_runtime(value: Any) -> None:
    runtime = _record_object(value, "manifest.runtime")
    _record_keys(runtime, _RUNTIME_FIELDS, "manifest.runtime")
    _record_identifier(runtime["python"], "manifest.runtime.python")

    platform = _record_object(runtime["platform"], "manifest.runtime.platform")
    _record_keys(platform, _PLATFORM_FIELDS, "manifest.runtime.platform")
    for field in _PLATFORM_FIELDS:
        _record_identifier(platform[field], f"manifest.runtime.platform.{field}")

    dependencies = _record_object(
        runtime["dependencies"],
        "manifest.runtime.dependencies",
    )
    _record_keys(dependencies, _DEPENDENCY_FIELDS, "manifest.runtime.dependencies")
    for field in _DEPENDENCY_FIELDS:
        if dependencies[field] is not None:
            _record_identifier(
                dependencies[field],
                f"manifest.runtime.dependencies.{field}",
            )

    code = _record_object(runtime["code"], "manifest.runtime.code")
    _record_keys(code, _CODE_FIELDS, "manifest.runtime.code")
    if code["package_version"] is not None:
        _record_identifier(
            code["package_version"],
            "manifest.runtime.code.package_version",
        )
    _validate_sha256(code["package_hash"], "manifest.runtime.code.package_hash")
    if code["git_commit"] is not None:
        commit = _record_identifier(
            code["git_commit"],
            "manifest.runtime.code.git_commit",
        )
        if len(commit) not in (40, 64):
            raise ValueError(
                "manifest.runtime.code.git_commit must be a SHA-1 or SHA-256 id"
            )
        _validate_hex(
            commit,
            len(commit),
            "manifest.runtime.code.git_commit",
        )
    dirty = code["dirty"]
    if dirty is not None and not isinstance(dirty, bool):
        raise TypeError("manifest.runtime.code.dirty must be boolean or null")
    if code["dirty_hash"] is not None:
        _validate_sha256(code["dirty_hash"], "manifest.runtime.code.dirty_hash")
    if dirty is True and code["dirty_hash"] is None:
        raise ValueError("dirty runtime code requires dirty_hash")
    if dirty is not True and code["dirty_hash"] is not None:
        raise ValueError("clean or unknown runtime code cannot have dirty_hash")
    if code["git_commit"] is None:
        if dirty is not None:
            raise ValueError("unknown Git commit requires unknown dirty state")
    elif dirty is None:
        raise ValueError("known Git commit requires a boolean dirty state")


def _validate_aware_datetime(value: Any, path: str) -> datetime:
    raw = _record_identifier(value, path)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path} must include a timezone offset")
    return parsed


def _validate_sha256(value: Any, path: str) -> str:
    return _validate_hex(value, 64, path)


def _validate_hex(value: Any, length: int, path: str) -> str:
    raw = _record_identifier(value, path)
    if len(raw) != length:
        raise ValueError(f"{path} must contain {length} hexadecimal characters")
    try:
        int(raw, 16)
    except ValueError as exc:
        raise ValueError(f"{path} must be hexadecimal") from exc
    if raw != raw.lower():
        raise ValueError(f"{path} must use lowercase hexadecimal characters")
    return raw


def verify_resume_manifest(
    existing: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> None:
    validate_manifest(requested)
    for key in _RESUME_IDENTITY_FIELDS:
        if existing[key] != requested[key]:
            raise ValueError(f"Cannot resume: manifest field {key!r} has changed")


def validate_derived_reports(
    records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    summaries: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> None:
    """Require summary artifacts to be exact derivatives of persisted records."""

    # Local import keeps record validation available to compare.py without an
    # import-time cycle; reports are only materialized at finalize/load time.
    from .compare import compare, summarize

    snapshots = list(records)
    strategy_order = [
        strategy["name"]
        for strategy in manifest["strategies"]
    ]
    expected_keys = {
        (strategy_id, task_id)
        for strategy_id in strategy_order
        for task_id in manifest["case_ids"]
    }
    actual_keys = {
        (record["strategy_id"], record["task_id"])
        for record in snapshots
    }
    if actual_keys != expected_keys:
        raise ValueError(
            "Cannot finalize summary before every strategy/case record exists"
        )
    expected_summaries = summarize(snapshots, manifest=manifest)
    expected_comparison = compare(
        snapshots,
        strategy_order=strategy_order,
        manifest=manifest,
    )
    if dict(summaries) != expected_summaries:
        raise ValueError("Summary summaries do not match the persisted records")
    if dict(comparison) != expected_comparison:
        raise ValueError("Summary comparison does not match the persisted records")


def validate_summary_artifact(
    summary: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Validate a persisted summary as an exact derived cache."""

    root = _record_object(summary, "summary")
    _record_keys(
        root,
        {"schema_version", "summaries", "comparison"},
        "summary",
    )
    _require_schema_version(root, "summary")
    summaries = _record_object(root["summaries"], "summary.summaries")
    comparison = _record_object(root["comparison"], "summary.comparison")
    validate_derived_reports(records, manifest, summaries, comparison)


def validate_records(
    records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Validate canonical v5 records and reject duplicate strategy/task keys."""

    validated: list[Mapping[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    run_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        try:
            key = validate_record(record, manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid benchmark record {index}: {exc}") from exc
        if key in completed:
            raise ValueError(
                "Duplicate benchmark record for "
                f"strategy {key[0]!r}, task {key[1]!r}"
            )
        completed.add(key)
        run_id = record["metrics"]["identity"]["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate metrics run_id {run_id!r}")
        run_ids.add(run_id)
        validated.append(record)
    return validated


def validate_record(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Validate one canonical v5 record and return its unique key."""

    root = _record_object(record, "record")
    _record_keys(root, _RECORD_FIELDS, "record")
    _require_schema_version(root, "record")

    strategy_id = _record_identifier(root["strategy_id"], "record.strategy_id")
    task_id = _record_identifier(root["task_id"], "record.task_id")
    key = (strategy_id, task_id)
    if manifest is not None:
        if strategy_id not in _manifest_strategy_ids(manifest):
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        if task_id not in _manifest_case_ids(manifest):
            raise ValueError(f"unknown task_id {task_id!r}")

    _validate_sha256(root["strategy_digest"], "record.strategy_digest")

    input_value = _record_object(root["input"], "record.input")
    _record_keys(input_value, {"prompt"}, "record.input")
    if not isinstance(input_value["prompt"], str):
        raise TypeError("record.input.prompt must be a string")
    for field in ("route", "classifier_decision"):
        value = root[field]
        if value is not None:
            _record_identifier(value, f"record.{field}")

    evaluation = _validate_evaluation(root["evaluation"])
    error = _validate_error(root["error"])
    metrics = _record_object(root["metrics"], "record.metrics")
    aggregate_metric_payloads((metrics,))
    if "calls" in metrics:
        raise ValueError("record.metrics must not duplicate the record call ledger")
    identity = _record_object(metrics["identity"], "record.metrics.identity")
    if identity["strategy_id"] != strategy_id or identity["task_id"] != task_id:
        raise ValueError("record.metrics identity does not match the record")

    calls = _validate_calls(
        root["calls"],
        expected_run_id=identity["run_id"],
    )
    _validate_metric_call_counts(metrics, calls)
    _validate_metric_usage_and_cost(metrics, calls)
    _validate_metric_response_counts(metrics, calls)
    _validate_duration_covers_calls(metrics, calls)
    attempts = _validate_attempts(root["attempts"], calls)
    events = _validate_routing_events(
        root["routing_events"],
        calls,
        allow_partial_classifier_evidence=(
            error is not None and error["stage"] == "routing"
        ),
    )
    _validate_routing_counts(metrics, attempts, events)
    expected_classifier_decision = next(
        (
            event["outcome"]
            for event in events
            if event["source"] == "difficulty-classifier"
        ),
        None,
    )
    if root["classifier_decision"] != expected_classifier_decision:
        raise ValueError(
            "record.classifier_decision contradicts its routing events"
        )

    route = root["route"]
    if attempts and route != attempts[-1]["route"]["phase"]:
        raise ValueError("record.route must match the final attempted route")
    if not attempts and route is not None:
        raise ValueError("record.route must be null when there are no attempts")

    final_output = _validate_output(root["final_output"], "record.final_output")
    if final_output is not None:
        if not attempts:
            raise ValueError("record.final_output requires an attempted call")
        final_call = calls[attempts[-1]["call_id"]]
        if final_call["output"] != final_output:
            raise ValueError("record.final_output does not match its call evidence")

    latency = root["evaluation_latency_ms"]
    if latency is not None:
        _record_number(latency, "record.evaluation_latency_ms")
    _validate_record_state(
        evaluation=evaluation,
        error=error,
        final_output=final_output,
        evaluation_latency_ms=latency,
        attempts=attempts,
    )
    _validate_task_outcome(metrics, evaluation, error)
    started_at = _validate_aware_datetime(root["started_at"], "record.started_at")
    finished_at = _validate_aware_datetime(root["finished_at"], "record.finished_at")
    if finished_at < started_at:
        raise ValueError("record.finished_at must not precede record.started_at")
    _validate_record_timing(
        started_at,
        finished_at,
        metrics,
        latency,
    )
    if manifest is not None:
        manifest_started_at = _validate_aware_datetime(
            manifest["created_at"],
            "manifest.created_at",
        )
        if started_at < manifest_started_at:
            raise ValueError("record.started_at must not precede manifest.created_at")
        _validate_record_against_manifest(
            root,
            metrics,
            calls,
            attempts,
            events,
            error,
            manifest,
        )
    else:
        _validate_embedded_record_pricing(metrics, calls)
    return key


def _validate_embedded_record_pricing(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> None:
    raw_catalogs = metrics["cost"]["catalogs"]
    if not raw_catalogs:
        if any(call["cost"]["status"] == "priced" for call in calls.values()):
            raise ValueError(
                "priced benchmark calls require an embedded price catalog"
            )
        return
    if len(raw_catalogs) != 1:
        raise ValueError("benchmark records may reference only one price catalog")
    raw = raw_catalogs[0]
    catalog = PriceCatalog(
        catalog_id=raw["catalog_id"],
        effective_date=raw["effective_date"],
        source=raw["source"],
        prices=raw["prices"],
    )
    _validate_record_pricing(metrics, calls, catalog)


def _validate_record_against_manifest(
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
    attempts: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    error: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
) -> None:
    strategy = next(
        item
        for item in manifest["strategies"]
        if item["name"] == record["strategy_id"]
    )
    if record["strategy_digest"] != strategy["digest"]:
        raise ValueError("record.strategy_digest does not match its manifest strategy")
    case = next(
        item
        for item in manifest["cases"]
        if item["task_id"] == record["task_id"]
    )
    prompt_digest = hashlib.sha256(
        record["input"]["prompt"].encode("utf-8")
    ).hexdigest()
    if prompt_digest != case["prompt_sha256"]:
        raise ValueError("record input prompt does not match its manifest case")

    config = StrategyConfig.model_validate(strategy["config"])
    prompt_texts = {
        prompt["declared_path"]: prompt["text"]
        for prompt in strategy["prompts"]
    }
    _validate_record_strategy(
        record,
        calls,
        attempts,
        events,
        error,
        config,
        prompt_texts,
    )

    catalog = _validate_manifest_pricing(manifest["pricing"])
    _validate_record_pricing(metrics, calls, catalog)


def _validate_record_strategy(
    record: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
    attempts: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    error: Mapping[str, Any] | None,
    config: StrategyConfig,
    prompt_texts: Mapping[str, str],
) -> None:
    method = config.method
    if any(
        call["channel"] not in {"classifier", "generation"}
        for call in calls.values()
    ):
        raise ValueError("record contains a call channel unsupported by strategies")
    classifier_calls = [
        call for call in calls.values() if call["channel"] == "classifier"
    ]
    generation_calls = [
        call for call in calls.values() if call["channel"] == "generation"
    ]
    if len(generation_calls) != len(attempts):
        raise ValueError("record generation calls contradict strategy attempts")

    if isinstance(method, FixedMethodConfig):
        if classifier_calls:
            raise ValueError("fixed strategy cannot contain classifier calls")
        expected_events = [("fixed-method", "fixed")]
        expected_phases = ["fixed"]
        phase_profiles = {
            "fixed": (method.model.model, method.role),
        }
    else:
        if len(classifier_calls) > 1:
            raise ValueError("strategy may contain only one classifier call")
        for call in classifier_calls:
            expected_classifier_prompt = (
                _configured_prompt_text(method.classifier.prompt, prompt_texts)
                .rstrip("\r\n")
                + "\n"
                + record["input"]["prompt"][:method.classifier.max_prompt_chars]
            )
            if (
                call["models"]["requested"] != method.classifier.model
                or call["role"] != "classifier"
                or call["request"]["prompt"] != expected_classifier_prompt
                or call["request"]["max_tokens"]
                != method.classifier.parameters.max_tokens
                or call["request"]["temperature"]
                != method.classifier.parameters.temperature
            ):
                raise ValueError("classifier call contradicts strategy config")
        decision = record["classifier_decision"]
        classifier_events = [
            event for event in events
            if event["source"] == "difficulty-classifier"
        ]
        if decision is None:
            expected_events = []
            expected_phases = []
        else:
            expected_events = [("difficulty-classifier", decision)]
            expected_phases = [
                "initial-hard" if decision == "hard" else "initial-easy"
            ]
        if decision is not None and len(classifier_calls) != 1:
            raise ValueError("classifier decision requires one classifier call")
        if classifier_events and len(classifier_events) != 1:
            raise ValueError("strategy may contain only one classifier event")
        phase_profiles = {
            "initial-easy": (method.easy.model, "generator"),
            "initial-hard": (method.hard.model, "writer"),
            "escalation": (method.hard.model, "fixer"),
        }
        if isinstance(method, CascadeMethodConfig) and decision == "easy":
            escalation_events = [
                event for event in events
                if event["source"] == "response-escalation"
            ]
            if len(escalation_events) > 1:
                raise ValueError("cascade may contain only one escalation event")
            if escalation_events:
                outcome = escalation_events[0]["outcome"]
                expected_events.append(("response-escalation", outcome))
                if outcome == "escalate":
                    expected_phases.append("escalation")
            elif error is None:
                raise ValueError("completed easy cascade requires escalation decision")
        elif any(
            event["source"] == "response-escalation" for event in events
        ):
            raise ValueError("only an easy cascade may contain escalation events")

    actual_events = [(event["source"], event["outcome"]) for event in events]
    if actual_events != expected_events:
        raise ValueError("record routing events contradict strategy config")
    actual_phases = [attempt["route"]["phase"] for attempt in attempts]
    if actual_phases != expected_phases:
        allow_empty_routing_prefix = (
            error is not None
            and error["stage"] == "routing"
            and not actual_phases
            and not expected_events
        )
        if not allow_empty_routing_prefix:
            raise ValueError("record attempt phases contradict strategy config")
    for attempt in attempts:
        phase = attempt["route"]["phase"]
        if phase not in phase_profiles:
            raise ValueError("record attempt phase is invalid for strategy")
        expected_model, expected_role = phase_profiles[phase]
        if (
            attempt["route"]["model"] != expected_model
            or attempt["route"]["role"] != expected_role
        ):
            raise ValueError("record attempt model/role contradicts strategy config")
        request = calls[attempt["call_id"]]["request"]
        if request["max_tokens"] is not None or request["temperature"] is not None:
            raise ValueError("generation request tuning contradicts strategy runtime")
        expected_prompt = record["input"]["prompt"]
        if isinstance(method, FixedMethodConfig):
            expected_prompt = (
                (
                    _configured_prompt_text(method.prompt_prefix, prompt_texts)
                    if method.prompt_prefix is not None
                    else ""
                )
                + expected_prompt
                + (
                    _configured_prompt_text(method.prompt_suffix, prompt_texts)
                    if method.prompt_suffix is not None
                    else ""
                )
            )
        if isinstance(method, CascadeMethodConfig):
            if phase == "initial-easy":
                expected_prompt += _configured_prompt_text(
                    method.escalation.self_check_suffix,
                    prompt_texts,
                )
            elif phase == "escalation":
                expected_prompt = (
                    _configured_prompt_text(
                        method.escalation.escalation_prefix,
                        prompt_texts,
                    )
                    + expected_prompt
                )
        if request["prompt"] != expected_prompt:
            raise ValueError("generation request prompt contradicts strategy config")


def _configured_prompt_text(
    prompt: InlinePromptConfig | FilePromptConfig,
    prompt_texts: Mapping[str, str],
) -> str:
    if isinstance(prompt, InlinePromptConfig):
        return prompt.text
    return prompt_texts[prompt.path]


def _validate_duration_covers_calls(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> None:
    duration_ms = float(metrics["timing"]["run_duration_ms"])
    minimum_duration = max(
        (
            float(call["timing"]["started_offset_ms"])
            + float(call["timing"]["latency_ms"])
            for call in calls.values()
        ),
        default=0.0,
    )
    if duration_ms + 1e-9 < minimum_duration:
        raise ValueError("record.metrics duration does not cover its call timings")


def _validate_record_timing(
    started_at: datetime,
    finished_at: datetime,
    metrics: Mapping[str, Any],
    evaluation_latency_ms: Any,
) -> None:
    """Reconcile wall-clock bounds with monotonic run/evaluation durations."""

    elapsed_ms = (finished_at - started_at).total_seconds() * 1_000
    minimum_ms = float(metrics["timing"]["run_duration_ms"])
    if evaluation_latency_ms is not None:
        minimum_ms += float(evaluation_latency_ms)
    # ISO timestamps have microsecond resolution and use a different clock from
    # the monotonic duration measurements, so allow one millisecond of skew.
    if elapsed_ms + 1.0 < minimum_ms:
        raise ValueError(
            "record timestamps do not cover run and evaluation durations"
        )


def _validate_record_pricing(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
    catalog: PriceCatalog,
) -> None:
    known_cost = 0.0
    missing_cost_calls = 0
    pricing_error_calls = 0
    priced_calls = 0
    catalog_observed = False
    for call_index, call in enumerate(calls.values(), start=1):
        path = f"record.calls[{call_index}].cost"
        cost = call["cost"]
        usage = call["usage"]
        if cost["status"] == "error":
            if cost["usd"] is not None:
                raise ValueError(f"{path} pricing error cannot claim a cost")
            if cost["catalog_id"] not in (None, catalog.catalog_id):
                raise ValueError(f"{path}.catalog_id contradicts the manifest catalog")
            if cost["source"] != cost["catalog_id"]:
                raise ValueError(f"{path}.source contradicts its catalog_id")
            missing_cost_calls += 1
            pricing_error_calls += 1
            catalog_observed = (
                catalog_observed or cost["catalog_id"] is not None
            )
            continue
        if call["status"] == "error" or usage["status"] == "error":
            expected_status = "unavailable"
            expected_cost = None
            expected_catalog_id = None
        else:
            quote = price_usage(
                call["models"]["priced"],
                TokenUsage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    visible_output_tokens=usage["visible_output_tokens"],
                    reasoning_tokens=usage["reasoning_tokens"],
                    cached_input_tokens=usage["cached_input_tokens"],
                    cache_write_input_tokens=usage[
                        "cache_write_input_tokens"
                    ],
                ),
                catalog,
            )
            expected_status = quote.status
            expected_cost = quote.cost_usd
            expected_catalog_id = catalog.catalog_id

        if cost["status"] != expected_status:
            raise ValueError(f"{path}.status contradicts the manifest catalog")
        if expected_cost is None:
            if cost["usd"] is not None:
                raise ValueError(f"{path}.usd must be null")
            missing_cost_calls += 1
        else:
            if cost["usd"] is None or not math.isclose(
                float(cost["usd"]),
                expected_cost,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{path}.usd contradicts the manifest catalog")
            known_cost += expected_cost
            priced_calls += 1
        if cost["source"] != expected_catalog_id:
            raise ValueError(f"{path}.source contradicts the manifest catalog")
        if cost["catalog_id"] != expected_catalog_id:
            raise ValueError(f"{path}.catalog_id contradicts the manifest catalog")
        catalog_observed = catalog_observed or cost["catalog_id"] is not None

    metric_cost = metrics["cost"]
    if not math.isclose(
        float(metric_cost["known_usd"]),
        known_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("record.metrics cost contradicts the call ledger")
    expected_total = known_cost if missing_cost_calls == 0 else None
    if expected_total is None:
        if metric_cost["total_usd"] is not None:
            raise ValueError("record.metrics total cost must be null")
    elif not math.isclose(
        float(metric_cost["total_usd"]),
        expected_total,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("record.metrics total cost contradicts the call ledger")
    expected_completeness = {
        "complete": missing_cost_calls == 0,
        "missing_calls": missing_cost_calls,
        "error_calls": pricing_error_calls,
    }
    if dict(metric_cost["completeness"]) != expected_completeness:
        raise ValueError("record.metrics cost completeness contradicts calls")
    expected_sources = {catalog.catalog_id: priced_calls} if priced_calls else {}
    if dict(metric_cost["priced_calls_by_source"]) != expected_sources:
        raise ValueError("record.metrics cost sources contradict calls")
    expected_catalogs = [catalog.to_dict()] if catalog_observed else []
    if metric_cost["catalogs"] != expected_catalogs:
        raise ValueError("record.metrics catalog snapshot contradicts the manifest")


def _validate_evaluation(value: Any) -> Mapping[str, Any]:
    evaluation = _record_object(value, "record.evaluation")
    _record_keys(evaluation, {"passed", "score", "details"}, "record.evaluation")
    if not isinstance(evaluation["passed"], bool):
        raise TypeError("record.evaluation.passed must be a boolean")
    _record_finite_number(evaluation["score"], "record.evaluation.score")
    details = _record_object(evaluation["details"], "record.evaluation.details")
    _validate_json_value(details, "record.evaluation.details")
    return evaluation


def _validate_calls(
    value: Any,
    *,
    expected_run_id: str,
) -> dict[str, Mapping[str, Any]]:
    raw_calls = _record_list(value, "record.calls")
    calls: dict[str, Mapping[str, Any]] = {}
    required = {
        "run_id",
        "call_id",
        "ordinal",
        "channel",
        "role",
        "status",
        "telemetry_status",
        "models",
        "timing",
        "usage",
        "cost",
        "response",
        "error",
        "request",
        "output",
    }
    for expected_ordinal, raw_call in enumerate(raw_calls, start=1):
        path = f"record.calls[{expected_ordinal}]"
        call = _record_object(raw_call, path)
        _record_keys(call, required, path)
        run_id = _record_identifier(call["run_id"], f"{path}.run_id")
        if run_id != expected_run_id:
            raise ValueError(f"{path}.run_id must match record.metrics identity")
        call_id = _record_identifier(call["call_id"], f"{path}.call_id")
        if call_id in calls:
            raise ValueError(f"record contains duplicate call_id {call_id!r}")
        ordinal = _record_integer(call["ordinal"], f"{path}.ordinal")
        if ordinal != expected_ordinal:
            raise ValueError("record call ordinals must be contiguous and ordered")
        _record_identifier(call["channel"], f"{path}.channel")
        if call["channel"] not in {"classifier", "generation"}:
            raise ValueError(f"{path}.channel unsupported by benchmark strategies")
        _record_identifier(call["role"], f"{path}.role")
        if call["status"] not in ("ok", "error"):
            raise ValueError(f"{path}.status must be 'ok' or 'error'")
        if call["telemetry_status"] not in ("complete", "partial", "error"):
            raise ValueError(f"{path}.telemetry_status is invalid")

        models = _record_object(call["models"], f"{path}.models")
        _record_keys(models, {"requested", "actual", "priced"}, f"{path}.models")
        _record_identifier(models["requested"], f"{path}.models.requested")
        for name in ("actual", "priced"):
            if models[name] is not None:
                _record_identifier(models[name], f"{path}.models.{name}")

        timing = _record_object(call["timing"], f"{path}.timing")
        _record_keys(timing, {"latency_ms", "started_offset_ms"}, f"{path}.timing")
        _record_number(timing["latency_ms"], f"{path}.timing.latency_ms")
        _record_number(
            timing["started_offset_ms"],
            f"{path}.timing.started_offset_ms",
        )
        usage = _validate_call_usage(call["usage"], path)
        cost = _validate_call_cost(call["cost"], path)
        response = _validate_call_response(call["response"], path)
        expected_telemetry_status = (
            "error"
            if usage["status"] == "error" or cost["status"] == "error"
            else "complete"
            if usage["status"] == "complete" and cost["status"] == "priced"
            else "partial"
        )
        if call["telemetry_status"] != expected_telemetry_status:
            raise ValueError(f"{path}.telemetry_status contradicts telemetry evidence")

        request = _record_object(call["request"], f"{path}.request")
        _record_keys(
            request,
            {"model", "role", "prompt", "max_tokens", "temperature"},
            f"{path}.request",
        )
        if request["model"] != models["requested"]:
            raise ValueError(f"{path} request and requested model disagree")
        if request["role"] != call["role"]:
            raise ValueError(f"{path} request and call role disagree")
        if not isinstance(request["prompt"], str) or not request["prompt"].strip():
            raise ValueError(f"{path}.request.prompt must be non-empty")
        if request["max_tokens"] is not None:
            if _record_integer(
                request["max_tokens"],
                f"{path}.request.max_tokens",
            ) < 1:
                raise ValueError(f"{path}.request.max_tokens must be positive")
        if request["max_tokens"] != response["requested_max_tokens"]:
            raise ValueError(
                f"{path} request and response requested_max_tokens disagree"
            )
        if request["temperature"] is not None:
            temperature = _record_finite_number(
                request["temperature"],
                f"{path}.request.temperature",
            )
            if not 0 <= temperature <= 2:
                raise ValueError(f"{path}.request.temperature must be between 0 and 2")

        output = _validate_output(call["output"], f"{path}.output")
        error = _validate_call_error(call["error"], path)
        if call["status"] == "ok":
            if output is None or error is not None or models["priced"] is None:
                raise ValueError(f"{path} successful-call evidence is incomplete")
            if output["model"] != models["actual"]:
                raise ValueError(f"{path} output and actual model disagree")
            expected_priced_model = models["actual"] or models["requested"]
            if models["priced"] != expected_priced_model:
                raise ValueError(f"{path} priced model contradicts model evidence")
            _validate_success_response(response, output, path)
            visible_output_tokens = usage["visible_output_tokens"]
            if (
                response["output_status"] != "unavailable"
                and visible_output_tokens is not None
                and (visible_output_tokens == 0)
                is not response["output_empty"]
            ):
                raise ValueError(
                    f"{path}.usage.visible_output_tokens contradicts "
                    "captured output emptiness"
                )
        elif (
            output is not None
            or error is None
            or models["actual"] is not None
            or models["priced"] is not None
            or usage["status"] != "unavailable"
            or cost["status"] != "unavailable"
            or cost["provider_reported_usd"] is not None
            or response != {
                "finish_reason": "error",
                "native_finish_reason": None,
                "output_status": None,
                "output_empty": None,
                "refusal": None,
                "requested_max_tokens": request["max_tokens"],
                "applied_max_tokens": None,
                "max_tokens_reached": False,
            }
        ):
            raise ValueError(f"{path} failed-call evidence is contradictory")
        calls[call_id] = call
    return calls


def _validate_call_usage(value: Any, call_path: str) -> Mapping[str, Any]:
    path = f"{call_path}.usage"
    usage = _record_object(value, path)
    _record_keys(
        usage,
        {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "visible_output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "completeness",
            "status",
            "diagnostic",
        },
        path,
    )
    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "visible_output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
    )
    for field in token_fields:
        if usage[field] is not None:
            _record_integer(usage[field], f"{path}.{field}")
    completeness = _record_object(usage["completeness"], f"{path}.completeness")
    _record_keys(completeness, {"total", "breakdown"}, f"{path}.completeness")
    for field in ("total", "breakdown"):
        if not isinstance(completeness[field], bool):
            raise TypeError(f"{path}.completeness.{field} must be a boolean")
    if usage["status"] not in (
        "complete",
        "total_only",
        "partial",
        "unavailable",
        "error",
    ):
        raise ValueError(f"{path}.status is invalid")
    if usage["diagnostic"] is not None:
        _record_identifier(usage["diagnostic"], f"{path}.diagnostic")
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    total = usage["total_tokens"]
    total_complete = total is not None
    breakdown_complete = prompt is not None and completion is not None
    if breakdown_complete and total != prompt + completion:
        raise ValueError(f"{path} token total contradicts its breakdown")
    if total is not None:
        for component in (prompt, completion):
            if component is not None and component > total:
                raise ValueError(f"{path} token component exceeds total")
    completion_capacity = (
        completion
        if completion is not None
        else total - prompt
        if total is not None and prompt is not None
        else total
    )
    if completion_capacity is not None:
        visible = usage["visible_output_tokens"]
        reasoning = usage["reasoning_tokens"]
        for field, detail in (
            ("visible_output_tokens", visible),
            ("reasoning_tokens", reasoning),
        ):
            if detail is not None and detail > completion_capacity:
                raise ValueError(
                    f"{path}.{field} exceeds available completion tokens"
                )
        if (
            visible is not None
            and reasoning is not None
            and visible + reasoning > completion_capacity
        ):
            raise ValueError(
                f"{path} visible and reasoning tokens exceed available "
                "completion tokens"
            )
    prompt_capacity = (
        prompt
        if prompt is not None
        else total - completion
        if total is not None and completion is not None
        else total
    )
    if prompt_capacity is not None:
        cached = usage["cached_input_tokens"]
        cache_write = usage["cache_write_input_tokens"]
        for field, detail in (
            ("cached_input_tokens", cached),
            ("cache_write_input_tokens", cache_write),
        ):
            if detail is not None and detail > prompt_capacity:
                raise ValueError(
                    f"{path}.{field} exceeds available prompt tokens"
                )
        if (
            cached is not None
            and cache_write is not None
            and cached + cache_write > prompt_capacity
        ):
            raise ValueError(
                f"{path} cached and cache-write tokens exceed available "
                "prompt tokens"
            )
    if total is not None:
        known_details = sum(
            int(usage[field] or 0)
            for field in (
                "visible_output_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
            )
        )
        if known_details > total:
            raise ValueError(
                f"{path} known input and output token details exceed total"
            )
    if completeness != {
        "total": total_complete,
        "breakdown": breakdown_complete,
    }:
        raise ValueError(f"{path}.completeness contradicts token evidence")
    any_known = any(usage[field] is not None for field in token_fields)
    expected_status = (
        "complete"
        if breakdown_complete
        else "total_only"
        if total_complete and prompt is None and completion is None
        else "partial"
        if any_known
        else usage["status"]
    )
    if any_known and usage["status"] != expected_status:
        raise ValueError(f"{path}.status contradicts token evidence")
    if not any_known and usage["status"] not in ("unavailable", "error"):
        raise ValueError(f"{path}.status requires token evidence")
    if usage["status"] != "complete" and usage["diagnostic"] is None:
        raise ValueError(f"{path} incomplete usage requires a diagnostic")
    if usage["status"] == "complete" and usage["diagnostic"] is not None:
        raise ValueError(f"{path} complete usage cannot contain a diagnostic")
    return usage


def _validate_call_cost(value: Any, call_path: str) -> Mapping[str, Any]:
    path = f"{call_path}.cost"
    cost = _record_object(value, path)
    _record_keys(
        cost,
        {
            "usd",
            "status",
            "source",
            "catalog_id",
            "diagnostic",
            "provider_reported_usd",
        },
        path,
    )
    if cost["usd"] is not None:
        _record_number(cost["usd"], f"{path}.usd")
    if cost["provider_reported_usd"] is not None:
        _record_number(
            cost["provider_reported_usd"],
            f"{path}.provider_reported_usd",
        )
    if cost["status"] not in ("priced", "unpriced", "unavailable", "error"):
        raise ValueError(f"{path}.status is invalid")
    for field in ("source", "catalog_id", "diagnostic"):
        if cost[field] is not None:
            _record_identifier(cost[field], f"{path}.{field}")
    if cost["catalog_id"] is not None and cost["source"] != cost["catalog_id"]:
        raise ValueError(f"{path}.source must match catalog_id")
    if cost["status"] == "priced":
        if cost["usd"] is None or cost["source"] is None:
            raise ValueError(f"{path} priced cost requires usd and source")
    else:
        if cost["usd"] is not None or cost["diagnostic"] is None:
            raise ValueError(f"{path} non-priced cost evidence is contradictory")
    return cost


def _validate_call_response(value: Any, call_path: str) -> Mapping[str, Any]:
    path = f"{call_path}.response"
    response = _record_object(value, path)
    _record_keys(
        response,
        {
            "finish_reason",
            "native_finish_reason",
            "output_status",
            "output_empty",
            "refusal",
            "requested_max_tokens",
            "applied_max_tokens",
            "max_tokens_reached",
        },
        path,
    )
    if response["finish_reason"] not in {
        "stop",
        "length",
        "refusal",
        "content_filter",
        "tool_call",
        "error",
        "unknown",
    }:
        raise ValueError(f"{path}.finish_reason is invalid")
    if response["output_status"] not in {
        None,
        "usable",
        "empty",
        "truncated",
        "refused",
        "unavailable",
    }:
        raise ValueError(f"{path}.output_status is invalid")
    if response["output_empty"] is not None and not isinstance(
        response["output_empty"],
        bool,
    ):
        raise TypeError(f"{path}.output_empty must be boolean or null")
    for field in ("native_finish_reason", "refusal"):
        if response[field] is not None and (
            not isinstance(response[field], str) or not response[field].strip()
        ):
            raise ValueError(f"{path}.{field} must be non-empty text or null")
    requested_max_tokens = response["requested_max_tokens"]
    if requested_max_tokens is not None and _record_integer(
        requested_max_tokens,
        f"{path}.requested_max_tokens",
    ) < 1:
        raise ValueError(f"{path}.requested_max_tokens must be positive")
    applied_max_tokens = response["applied_max_tokens"]
    if applied_max_tokens is not None and _record_integer(
        applied_max_tokens,
        f"{path}.applied_max_tokens",
    ) < 1:
        raise ValueError(f"{path}.applied_max_tokens must be positive")
    if response["max_tokens_reached"] is not None and not isinstance(
        response["max_tokens_reached"],
        bool,
    ):
        raise TypeError(f"{path}.max_tokens_reached must be boolean or null")
    return response


def _validate_success_response(
    response: Mapping[str, Any],
    output: Mapping[str, Any],
    call_path: str,
) -> None:
    finish_reason = response["finish_reason"]
    if response["output_status"] == "unavailable":
        if (
            output["text"]
            or output["raw_text"] is not None
            or response["refusal"] is not None
            or response["output_empty"] is not None
        ):
            raise ValueError(
                f"{call_path}.response.output_status contradicts response evidence"
            )
        expected_status = "unavailable"
    else:
        expected_empty = not bool(output["text"].strip())
        if response["output_empty"] is not expected_empty:
            raise ValueError(
                f"{call_path}.response.output_empty contradicts output text"
            )
        expected_status = (
            "refused"
            if response["refusal"] is not None
            or finish_reason in {"refusal", "content_filter"}
            else "truncated"
            if finish_reason == "length"
            else "usable"
            if output["text"].strip()
            else "empty"
        )
    if response["output_status"] != expected_status:
        raise ValueError(
            f"{call_path}.response.output_status contradicts response evidence"
        )
    expected_limit = (
        True
        if finish_reason == "length"
        else None
        if finish_reason == "unknown"
        else False
    )
    if response["max_tokens_reached"] is not expected_limit:
        raise ValueError(
            f"{call_path}.response.max_tokens_reached contradicts finish reason"
        )


def _validate_output(value: Any, path: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    output = _record_object(value, path)
    _record_keys(output, {"model", "text", "raw_text"}, path)
    if output["model"] is not None:
        _record_identifier(output["model"], f"{path}.model")
    if not isinstance(output["text"], str):
        raise TypeError(f"{path}.text must be a string")
    if output["raw_text"] is not None and not isinstance(output["raw_text"], str):
        raise TypeError(f"{path}.raw_text must be a string or null")
    return output


def _validate_call_error(value: Any, call_path: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    path = f"{call_path}.error"
    error = _record_object(value, path)
    _record_keys(error, {"category", "type", "message"}, path)
    if error["category"] not in {
        "timeout",
        "rate_limit",
        "authentication",
        "provider_5xx",
        "invalid_response",
        "unknown",
    }:
        raise ValueError(f"{path}.category is invalid")
    _record_identifier(error["type"], f"{path}.type")
    if not isinstance(error["message"], str):
        raise TypeError(f"{path}.message must be a string")
    return error


def _validate_metric_call_counts(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> None:
    interactions = metrics["interactions"]
    call_values = tuple(calls.values())
    expected = {
        "total": len(call_values),
        "failed": sum(call["status"] == "error" for call in call_values),
        "by_channel": _call_counter(call_values, "channel"),
        "by_role": _call_counter(call_values, "role"),
        "by_requested_model": _call_model_counter(call_values, "requested"),
        "by_actual_model": _call_model_counter(call_values, "actual"),
        "by_priced_model": _call_model_counter(call_values, "priced"),
        "errors_by_category": dict(sorted(Counter(
            call["error"]["category"]
            for call in call_values
            if call["error"] is not None
        ).items())),
    }
    if dict(interactions) != expected:
        raise ValueError("record.metrics interactions contradict the call ledger")


def _validate_metric_usage_and_cost(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> None:
    call_values = tuple(calls.values())
    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "visible_output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
    )
    known_tokens = {
        field: sum(int(call["usage"][field] or 0) for call in call_values)
        for field in token_fields
    }
    missing_total = sum(
        call["usage"]["total_tokens"] is None for call in call_values
    )
    missing_breakdown = sum(
        call["usage"]["prompt_tokens"] is None
        or call["usage"]["completion_tokens"] is None
        for call in call_values
    )
    usage_errors = sum(
        call["usage"]["status"] == "error" for call in call_values
    )
    expected_usage = {
        "known": {
            field: known_tokens[field] for field in token_fields
        },
        "total_tokens": (
            known_tokens["total_tokens"] if missing_total == 0 else None
        ),
        "completeness": {
            "total": missing_total == 0,
            "breakdown": missing_breakdown == 0,
            "missing_total_calls": missing_total,
            "missing_breakdown_calls": missing_breakdown,
            "error_calls": usage_errors,
            "details": {
                field: {
                    "complete": all(
                        call["usage"][field] is not None for call in call_values
                    ),
                    "missing_calls": sum(
                        call["usage"][field] is None for call in call_values
                    ),
                }
                for field in (
                    "visible_output_tokens",
                    "reasoning_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                )
            },
        },
    }
    if dict(metrics["usage"]) != expected_usage:
        raise ValueError("record.metrics usage contradicts the call ledger")

    known_cost = checked_fsum(
        (
            float(call["cost"]["usd"])
            for call in call_values
            if call["cost"]["usd"] is not None
        ),
        name="record aggregate catalog cost",
    )
    missing_cost = sum(call["cost"]["usd"] is None for call in call_values)
    known_provider_cost = checked_fsum(
        (
            float(call["cost"]["provider_reported_usd"])
            for call in call_values
            if call["cost"]["provider_reported_usd"] is not None
        ),
        name="record aggregate provider-reported cost",
    )
    missing_provider_cost = sum(
        call["cost"]["provider_reported_usd"] is None
        for call in call_values
    )
    cost_errors = sum(call["cost"]["status"] == "error" for call in call_values)
    priced_calls_by_source = dict(sorted(Counter(
        call["cost"]["source"]
        for call in call_values
        if call["cost"]["status"] == "priced"
    ).items()))
    catalogs = metrics["cost"]["catalogs"]
    catalog_ids = {catalog["catalog_id"] for catalog in catalogs}
    referenced_catalog_ids = {
        call["cost"]["catalog_id"]
        for call in call_values
        if call["cost"]["catalog_id"] is not None
    }
    if catalog_ids != referenced_catalog_ids:
        raise ValueError("record.metrics price catalogs contradict the call ledger")
    expected_cost = {
        "known_usd": known_cost,
        "total_usd": known_cost if missing_cost == 0 else None,
        "completeness": {
            "complete": missing_cost == 0,
            "missing_calls": missing_cost,
            "error_calls": cost_errors,
        },
        "priced_calls_by_source": priced_calls_by_source,
        "catalogs": catalogs,
        "provider_reported": {
            "known_usd": known_provider_cost,
            "total_usd": (
                known_provider_cost if missing_provider_cost == 0 else None
            ),
            "complete": missing_provider_cost == 0,
            "missing_calls": missing_provider_cost,
        },
    }
    if dict(metrics["cost"]) != expected_cost:
        raise ValueError("record.metrics cost contradicts the call ledger")


def _validate_metric_response_counts(
    metrics: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> None:
    call_values = tuple(calls.values())
    expected = {
        "finish_reasons": dict(sorted(Counter(
            call["response"]["finish_reason"] for call in call_values
        ).items())),
        "output_statuses": dict(sorted(Counter(
            call["response"]["output_status"] or "unavailable"
            for call in call_values
        ).items())),
        "output_emptiness": dict(sorted(Counter(
            "unknown"
            if call["response"]["output_empty"] is None
            else "empty"
            if call["response"]["output_empty"]
            else "nonempty"
            for call in call_values
        ).items())),
        "max_tokens_reached_calls": sum(
            call["response"]["max_tokens_reached"] is True
            for call in call_values
        ),
    }
    if dict(metrics["responses"]) != expected:
        raise ValueError("record.metrics responses contradict the call ledger")


def _validate_task_outcome(
    metrics: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    error: Mapping[str, Any] | None,
) -> None:
    expected = (
        f"{error['stage']}_error"
        if error is not None
        else "passed"
        if evaluation["passed"]
        else "incorrect"
    )
    observed = {
        outcome for outcome, count in metrics["outcomes"].items() if count == 1
    }
    if observed != {expected}:
        raise ValueError("record.metrics outcome contradicts task evidence")


def _call_counter(
    calls: Iterable[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(sorted(Counter(call[field] for call in calls).items()))


def _call_model_counter(
    calls: Iterable[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(sorted(Counter(
        call["models"][field]
        for call in calls
        if call["models"][field] is not None
    ).items()))


def _validate_attempts(
    value: Any,
    calls: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    attempts = _record_list(value, "record.attempts")
    validated: list[Mapping[str, Any]] = []
    references: set[str] = set()
    generation_calls = {
        call_id
        for call_id, call in calls.items()
        if call["channel"] == "generation"
    }
    for expected_index, raw_attempt in enumerate(attempts, start=1):
        path = f"record.attempts[{expected_index}]"
        attempt = _record_object(raw_attempt, path)
        _record_keys(
            attempt,
            {"index", "route", "call_id", "status"},
            path,
            optional={"reconstructed"},
        )
        if _record_integer(attempt["index"], f"{path}.index") != expected_index:
            raise ValueError("record attempt indices must be contiguous and ordered")
        call_id = _record_identifier(attempt["call_id"], f"{path}.call_id")
        if call_id not in calls:
            raise ValueError(f"{path} has dangling call reference {call_id!r}")
        if call_id not in generation_calls:
            raise ValueError(f"{path} must reference a generation call")
        if call_id in references:
            raise ValueError(f"record attempts repeat call reference {call_id!r}")
        references.add(call_id)
        if attempt["status"] != calls[call_id]["status"]:
            raise ValueError(f"{path} status contradicts its call")
        if "reconstructed" in attempt and attempt["reconstructed"] is not True:
            raise ValueError(f"{path}.reconstructed may only be present as true")
        route = _validate_route(attempt["route"], path)
        request = calls[call_id]["request"]
        if (
            route["model"] != request["model"]
            or route["prompt"] != request["prompt"]
            or route["role"] != request["role"]
        ):
            raise ValueError(f"{path}.route contradicts its call request")
        validated.append(attempt)
    if references != generation_calls:
        raise ValueError("record attempts must reference every generation call once")
    return validated


def _validate_route(value: Any, attempt_path: str) -> Mapping[str, Any]:
    path = f"{attempt_path}.route"
    route = _record_object(value, path)
    _record_keys(
        route,
        {"action", "phase", "label", "model", "role", "prompt"},
        path,
    )
    if route["action"] != "execute":
        raise ValueError(f"{path}.action must be 'execute'")
    if route["phase"] not in (
        "initial-easy",
        "initial-hard",
        "escalation",
        "fixed",
    ):
        raise ValueError(f"{path}.phase is invalid")
    for field in ("model", "role"):
        _record_identifier(route[field], f"{path}.{field}")
    if not isinstance(route["label"], str):
        raise TypeError(f"{path}.label must be a string")
    if not isinstance(route["prompt"], str) or not route["prompt"].strip():
        raise ValueError(f"{path}.prompt must be non-empty")
    return route


def _validate_routing_events(
    value: Any,
    calls: Mapping[str, Mapping[str, Any]],
    *,
    allow_partial_classifier_evidence: bool = False,
) -> list[Mapping[str, Any]]:
    events = _record_list(value, "record.routing_events")
    validated: list[Mapping[str, Any]] = []
    classifier_references: list[str] = []
    for event_index, raw_event in enumerate(events, start=1):
        path = f"record.routing_events[{event_index}]"
        event = _record_object(raw_event, path)
        _record_keys(
            event,
            {"source", "outcome", "reason", "model", "call_ids"},
            path,
        )
        source = _record_identifier(event["source"], f"{path}.source")
        outcome = _record_identifier(event["outcome"], f"{path}.outcome")
        _record_identifier(event["reason"], f"{path}.reason")
        if event["model"] is not None:
            _record_identifier(event["model"], f"{path}.model")
        call_ids = _record_list(event["call_ids"], f"{path}.call_ids")
        expected_outcomes = {
            "difficulty-classifier": {"easy", "hard"},
            "response-escalation": {"accept", "escalate"},
            "fixed-method": {"fixed"},
        }
        if source not in expected_outcomes or outcome not in expected_outcomes[source]:
            raise ValueError(f"{path} has an invalid source/outcome pair")
        if source == "difficulty-classifier":
            if event["model"] is None or not call_ids:
                raise ValueError(
                    f"{path} classifier event requires a model and call evidence"
                )
        elif event["model"] is not None or call_ids:
            raise ValueError(
                f"{path} non-classifier event cannot contain model/call evidence"
            )
        references: set[str] = set()
        for raw_call_id in call_ids:
            call_id = _record_identifier(raw_call_id, f"{path}.call_ids[]")
            if call_id not in calls:
                raise ValueError(f"{path} has dangling call reference {call_id!r}")
            if call_id in references:
                raise ValueError(f"{path} repeats call reference {call_id!r}")
            if source == "difficulty-classifier" and (
                calls[call_id]["channel"] != "classifier"
            ):
                raise ValueError(f"{path} must reference classifier calls")
            references.add(call_id)
            classifier_references.append(call_id)
        if source == "difficulty-classifier":
            evidence_models = {
                calls[call_id]["models"]["actual"]
                or calls[call_id]["models"]["requested"]
                for call_id in references
            }
            if event["model"] not in evidence_models:
                raise ValueError(f"{path}.model contradicts classifier call evidence")
        validated.append(event)
    classifier_calls = sorted(
        call_id
        for call_id, call in calls.items()
        if call["channel"] == "classifier"
    )
    referenced = sorted(classifier_references)
    if (
        referenced != classifier_calls
        and not (
            allow_partial_classifier_evidence
            and set(referenced).issubset(classifier_calls)
        )
    ):
        raise ValueError(
            "record routing events must reference every classifier call exactly once"
        )
    return validated


def _validate_routing_counts(
    metrics: Mapping[str, Any],
    attempts: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> None:
    routing = metrics["routing"]
    expected = {
        "generation_attempts": len(attempts),
        "events": len(events),
    }
    if dict(routing) != expected:
        raise ValueError("record.metrics routing contradicts record evidence")


def _validate_error(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    error = _record_object(value, "record.error")
    _record_keys(error, {"stage", "type", "message"}, "record.error")
    if error["stage"] not in ("routing", "execution", "evaluation"):
        raise ValueError("record.error.stage is invalid")
    _record_identifier(error["type"], "record.error.type")
    if not isinstance(error["message"], str):
        raise TypeError("record.error.message must be a string")
    return error


def _validate_record_state(
    *,
    evaluation: Mapping[str, Any],
    error: Mapping[str, Any] | None,
    final_output: Mapping[str, Any] | None,
    evaluation_latency_ms: Any,
    attempts: list[Mapping[str, Any]],
) -> None:
    placeholder = {"passed": False, "score": 0.0, "details": {}}
    stage = None if error is None else error["stage"]
    reconstructed = ["reconstructed" in attempt for attempt in attempts]
    if stage in ("routing", "execution"):
        if final_output is not None or evaluation_latency_ms is not None:
            raise ValueError(
                "routing/execution errors cannot contain final evaluation evidence"
            )
        if dict(evaluation) != placeholder:
            raise ValueError("pre-evaluation errors require placeholder evaluation")
        if not all(reconstructed):
            raise ValueError("partial execution attempts must be reconstructed")
        return
    if final_output is None or evaluation_latency_ms is None:
        raise ValueError("completed execution requires output and evaluation latency")
    if any(reconstructed):
        raise ValueError("completed execution attempts cannot be reconstructed")
    if stage == "evaluation" and dict(evaluation) != placeholder:
        raise ValueError("evaluation errors require placeholder evaluation")


def _record_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _record_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _record_keys(
    value: Mapping[str, Any],
    required: set[str],
    path: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(extra))}")


def _record_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a nonempty string")
    return value


def _record_identifier(value: Any, path: str) -> str:
    raw = _record_string(value, path)
    if raw != raw.strip():
        raise ValueError(f"{path} must be trimmed")
    return raw


def _record_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TypeError(f"{path} must be a nonnegative integer")
    return int(value)


def _record_number(value: Any, path: str) -> float:
    result = _record_finite_number(value, path)
    if result < 0:
        raise TypeError(f"{path} must be nonnegative")
    return result


def _record_finite_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not is_finite_real(value)
    ):
        raise TypeError(f"{path} must be a finite number")
    return float(value)


def _manifest_strategy_ids(manifest: Mapping[str, Any]) -> set[str]:
    strategies = manifest.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("Manifest field 'strategies' must be a nonempty array")
    identifiers: list[str] = []
    for strategy in strategies:
        if not isinstance(strategy, Mapping) or "name" not in strategy:
            raise ValueError("Every manifest strategy must contain a 'name'")
        name = strategy["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("Every manifest strategy name must be nonempty")
        identifiers.append(name)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Manifest strategy names must be unique")
    return set(identifiers)


def _manifest_case_ids(manifest: Mapping[str, Any]) -> set[str]:
    case_ids = manifest.get("case_ids")
    if not isinstance(case_ids, list):
        raise ValueError("Manifest field 'case_ids' must be an array")
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("Manifest case_ids must be nonempty strings")
    identifiers = list(case_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Manifest case_ids must be unique")
    return set(identifiers)


def _require_schema_version(payload: Mapping[str, Any], kind: str) -> None:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{kind.capitalize()} schema_version must be {SCHEMA_VERSION}, "
            f"got {version!r}"
        )
