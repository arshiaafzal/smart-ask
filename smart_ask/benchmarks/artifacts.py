"""Crash-safe persistence for canonical benchmark v2 artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


RECORD_SCHEMA = "smart-ask.benchmark-result/v2"
MANIFEST_SCHEMA = "smart-ask.benchmark-run/v2"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("benchmark artifacts cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("benchmark artifact keys must be strings")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"benchmark artifacts cannot contain {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"records line {number} must be a JSON object")
            values.append(value)
    return values


def _loads(value: str | bytes) -> Any:
    def reject_constant(raw: str) -> None:
        raise ValueError(f"invalid JSON numeric constant: {raw}")

    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported benchmark manifest schema")
    cases = value.get("cases")
    strategies = value.get("strategies")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark manifest requires cases")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("benchmark manifest requires strategies")
    case_ids = [case.get("task_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids):
        raise ValueError("benchmark manifest case IDs must be unique")


def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in ("schema", "benchmark", "dataset", "evaluator", "cases", "strategies", "pricing")
    }


def _allowed_keys(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    strategy_ids = {
        item["config"]["name"]
        for item in manifest["strategies"]
    }
    task_ids = {item["task_id"] for item in manifest["cases"]}
    return {(strategy, task) for strategy in strategy_ids for task in task_ids}


def _validate_record(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, str]:
    if value.get("schema") != RECORD_SCHEMA:
        raise ValueError("unsupported benchmark record schema")
    key = (value.get("strategy_id"), value.get("task_id"))
    if key not in _allowed_keys(manifest):
        raise ValueError("benchmark record is not declared by the manifest")
    required = (
        "run",
        "decisions",
        "model_calls",
        "provider_requests",
        "final_call",
        "output",
        "evaluation",
        "error",
    )
    if any(name not in value for name in required):
        raise ValueError("benchmark record is missing canonical fields")
    strategy = next(
        item
        for item in manifest["strategies"]
        if item["config"]["name"] == key[0]
    )
    if value.get("strategy_digest") != strategy.get("digest"):
        raise ValueError("benchmark record strategy digest does not match manifest")
    _validate_ledger(value)
    return key  # type: ignore[return-value]


def _validate_ledger(value: Mapping[str, Any]) -> None:
    decisions = value["decisions"]
    calls = value["model_calls"]
    requests = value["provider_requests"]
    if not all(isinstance(items, list) for items in (decisions, calls, requests)):
        raise ValueError("canonical ledgers must be lists")

    def indexed(items, field):
        result = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get(field), str):
                raise ValueError(f"ledger entries require {field}")
            identity = item[field]
            if identity in result:
                raise ValueError(f"duplicate ledger identity: {identity}")
            result[identity] = item
        return result

    decision_by_id = indexed(decisions, "decision_id")
    call_by_id = indexed(calls, "call_id")
    request_by_id = indexed(requests, "provider_request_id")
    for call_id, call in call_by_id.items():
        caused_by = call.get("caused_by_decision_id")
        if caused_by is not None and caused_by not in decision_by_id:
            raise ValueError(f"call {call_id} references an unknown decision")
        provider_ids = call.get("provider_request_ids")
        if not isinstance(provider_ids, list) or len(provider_ids) != len(set(provider_ids)):
            raise ValueError(f"call {call_id} has invalid provider request IDs")
        for request_id in provider_ids:
            request = request_by_id.get(request_id)
            if request is None or request.get("call_id") != call_id:
                raise ValueError(f"call {call_id} has a broken provider reference")
    for request_id, request in request_by_id.items():
        call = call_by_id.get(request.get("call_id"))
        if call is None or request_id not in call.get("provider_request_ids", ()):
            raise ValueError(f"provider request {request_id} has a broken call reference")
    for decision_id, decision in decision_by_id.items():
        evidence = decision.get("evidence_call_ids")
        if not isinstance(evidence, list) or len(evidence) != len(set(evidence)):
            raise ValueError(f"decision {decision_id} has invalid evidence")
        for call_id in evidence:
            call = call_by_id.get(call_id)
            if call is None or call.get("status") != "completed":
                raise ValueError(f"decision {decision_id} has invalid call evidence")

    run = value["run"]
    final_call = value["final_call"]
    if run is None:
        if calls or requests or decisions or value.get("error") is None:
            raise ValueError("a missing run requires an evidence-free execution error")
        return
    if not isinstance(run, Mapping):
        raise ValueError("run must be an object or null")
    if final_call is not None and final_call not in call_by_id:
        raise ValueError("final call does not exist in the call ledger")
    final_decision = run.get("final_decision_id")
    if final_decision is not None and final_decision not in decision_by_id:
        raise ValueError("final decision does not exist in the decision ledger")
    if run.get("status") == "completed" and (
        final_call is None or final_decision is None
    ):
        raise ValueError("completed runs require a final call and decision")


class MemoryResultSink:
    def __init__(self):
        self._manifest: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self.summaries: dict[str, Any] = {}
        self.comparison: dict[str, Any] = {}

    @property
    def completed_keys(self):
        return {(item["strategy_id"], item["task_id"]) for item in self._records}

    @property
    def existing_records(self):
        return _json_safe(self._records)

    def start(self, manifest):
        value = _json_safe(manifest)
        _validate_manifest(value)
        self._manifest = value
        return _json_safe(value)

    def append(self, record):
        if self._manifest is None:
            raise RuntimeError("sink must be started")
        value = _json_safe(record)
        key = _validate_record(value, self._manifest)
        if key in self.completed_keys:
            raise ValueError("duplicate benchmark record")
        self._records.append(value)

    def finalize(self, summaries, comparison):
        self.summaries = _json_safe(summaries)
        self.comparison = _json_safe(comparison)

    def close(self):
        pass


class JsonlResultSink:
    """Append records immediately and atomically write manifest/summary files."""

    def __init__(self, directory: str | Path, *, resume: bool = False):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.records_path = self.directory / "records.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.lock_path = self.directory / ".run.lock"
        self._resume = resume
        self._manifest: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._completed: set[tuple[str, str]] = set()
        self._lock_file = None
        self._thread_lock = threading.Lock()
        self._closed = False

    @property
    def completed_keys(self):
        return set(self._completed)

    @property
    def existing_records(self):
        return _json_safe(self._records)

    def start(self, manifest):
        requested = _json_safe(manifest)
        _validate_manifest(requested)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+b")
        if fcntl is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.close()
                raise RuntimeError("benchmark directory is already in use") from exc
        try:
            if self._resume:
                existing = _read_json(self.manifest_path)
                _validate_manifest(existing)
                if _manifest_identity(existing) != _manifest_identity(requested):
                    raise ValueError("resume manifest does not match requested run")
                self._repair_torn_line()
                self._records = _read_jsonl(self.records_path)
                for record in self._records:
                    key = _validate_record(record, existing)
                    if key in self._completed:
                        raise ValueError("duplicate benchmark record in artifact")
                    self._completed.add(key)
                self._manifest = existing
                if self.summary_path.exists():
                    self.summary_path.unlink()
            else:
                contents = [path for path in self.directory.iterdir() if path != self.lock_path]
                if contents:
                    raise FileExistsError("new benchmark directory must be empty")
                self.records_path.touch()
                _atomic_json(self.manifest_path, requested)
                self._manifest = requested
        except Exception:
            self.close()
            raise
        return _json_safe(self._manifest)

    def append(self, record):
        if self._manifest is None:
            raise RuntimeError("sink must be started")
        value = _json_safe(record)
        key = _validate_record(value, self._manifest)
        with self._thread_lock:
            if key in self._completed:
                raise ValueError("duplicate benchmark record")
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._completed.add(key)
            self._records.append(value)

    def finalize(self, summaries, comparison):
        if self._manifest is None:
            raise RuntimeError("sink must be started")
        _atomic_json(self.summary_path, {
            "schema": "smart-ask.benchmark-summary/v2",
            "summaries": summaries,
            "comparison": comparison,
        })
        self.close()

    def _repair_torn_line(self):
        data = self.records_path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        newline = data.rfind(b"\n")
        candidate = data[newline + 1:]
        try:
            _loads(candidate)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.records_path.write_bytes(data[: newline + 1] if newline >= 0 else b"")
        else:
            with self.records_path.open("ab") as handle:
                handle.write(b"\n")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._lock_file is not None:
            if fcntl is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None


def load_run(directory: str | Path):
    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    _validate_manifest(manifest)
    records = _read_jsonl(directory / "records.jsonl")
    completed = set()
    for record in records:
        key = _validate_record(record, manifest)
        if key in completed:
            raise ValueError("duplicate benchmark record in artifact")
        completed.add(key)
    summary = (
        _read_json(directory / "summary.json")
        if (directory / "summary.json").exists()
        else None
    )
    return {"manifest": manifest, "records": records, "summary": summary}
