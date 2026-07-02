"""Crash-safe benchmark artifacts and legacy result compatibility."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Protocol


SCHEMA_VERSION = 3

_LEGACY_ROUTES = {
    "gemini": "initial-easy",
    "opus-G1": "initial-hard",
    "opus-esc": "escalation",
    "opus": "fixed",
}


class ResultSink(Protocol):
    """Persist completed strategy/task records as they become available."""

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        """Return strategy/task pairs already present in this sink."""

        ...

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        """Return records already held by the sink, including resumed work."""

        ...

    def start(self, manifest: Mapping[str, Any]) -> None:
        """Initialize or validate the run manifest before tasks execute."""

        ...

    def append(self, record: Mapping[str, Any]) -> None:
        """Persist one complete task record."""

        ...

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        """Persist aggregate reports after all task records are complete."""

        ...


class JsonlResultSink:
    """Append task records to JSONL and keep metadata in small JSON files."""

    def __init__(
        self,
        directory: str | Path,
        *,
        resume: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.records_path = self.directory / "records.jsonl"
        self.summary_path = self.directory / "summary.json"
        self._lock = threading.Lock()
        self._completed: set[tuple[str, str]] = set()
        self._resume = resume
        self._started = False

    def start(self, manifest: Mapping[str, Any]) -> None:
        normalized_manifest = _json_safe(dict(manifest))
        normalized_manifest["schema_version"] = SCHEMA_VERSION

        if self._resume and self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
            _verify_resume_manifest(existing, normalized_manifest)
            _repair_torn_final_line(self.records_path)
            self._completed = _read_completed_keys(self.records_path)
        else:
            self.manifest_path.write_text(
                json.dumps(normalized_manifest, indent=2, sort_keys=True) + "\n"
            )
            self.records_path.write_text("")
        self._started = True

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return set(self._completed)

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        return list(_read_jsonl(self.records_path)) if self.records_path.exists() else []

    def append(self, record: Mapping[str, Any]) -> None:
        if not self._started:
            raise RuntimeError("Result sink must be started before appending records")
        normalized = _json_safe(dict(record))
        key = (str(normalized["strategy_id"]), str(normalized["task_id"]))
        line = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if key in self._completed:
                return
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._completed.add(key)

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "summaries": summaries,
            "comparison": comparison,
        }
        self.summary_path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
        )


class MemoryResultSink:
    """In-memory sink useful for embedding and network-free unit tests."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []
        self.summaries: dict[str, Any] = {}
        self.comparison: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}

    def start(self, manifest: Mapping[str, Any]) -> None:
        self.manifest = _json_safe(dict(manifest))

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return {
            (str(record["strategy_id"]), str(record["task_id"]))
            for record in self.records
        }

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        return list(self.records)

    def append(self, record: Mapping[str, Any]) -> None:
        key = (str(record["strategy_id"]), str(record["task_id"]))
        if key not in self.completed_keys:
            self.records.append(_json_safe(dict(record)))

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        self.summaries = _json_safe(dict(summaries))
        self.comparison = _json_safe(dict(comparison))


def load_run(path: str | Path) -> dict[str, Any]:
    """Load a schema-v3 run directory or normalize a legacy JSON result file."""

    artifact = Path(path)
    if artifact.is_dir():
        manifest = json.loads((artifact / "manifest.json").read_text())
        records = list(_read_jsonl(artifact / "records.jsonl"))
        summary_path = artifact / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return {"manifest": manifest, "records": records, "summary": summary}

    payload = json.loads(artifact.read_text())
    if payload.get("schema_version") == SCHEMA_VERSION and "records" in payload:
        return {
            "manifest": payload.get("manifest", {}),
            "records": payload["records"],
            "summary": payload.get("summary", {}),
        }
    return _load_legacy_result(artifact, payload)


def _load_legacy_result(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    calls_by_task: dict[str, list[dict[str, Any]]] = {}
    for call in payload.get("token_log", {}).get("calls", []):
        task_id = call.get("task_id")
        if task_id is not None:
            calls_by_task.setdefault(str(task_id), []).append(dict(call))

    strategy_id = f"legacy:{path.stem}"
    records = []
    for source in payload.get("results", []):
        task_id = str(source.get("task_id") or source.get("question_id") or "unknown")
        calls = _legacy_calls_for_task(task_id, calls_by_task)
        cost = (
            sum(float(call.get("cost_usd", 0.0)) for call in calls)
            if calls
            else None
        )
        prompt_tokens = sum(int(call.get("prompt_tokens", 0)) for call in calls)
        completion_tokens = sum(int(call.get("completion_tokens", 0)) for call in calls)
        passed = bool(source.get("pass_all", source.get("passed", False)))
        route = source.get("route") or _LEGACY_ROUTES.get(
            source.get("model"), source.get("model", "unknown")
        )
        classifier_decision = source.get("classifier_decision") or source.get("gate1")
        if classifier_decision is None and "task_id" in source:
            classifier_decision = source.get("difficulty")
        missing = ["routing_events", "attempts", "outputs", "latency"]
        if cost is None:
            missing.append("cost")
        records.append({
            "schema_version": int(payload.get("schema_version", 1)),
            "strategy_id": strategy_id,
            "strategy_digest": None,
            "task_id": task_id,
            "route": route,
            "classifier_decision": classifier_decision,
            "routing_events": [],
            "attempts": [],
            "calls": calls,
            "final_output": None,
            "evaluation": {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "details": dict(source),
            },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost_usd": cost,
            "total_latency_ms": None,
            "evaluation_latency_ms": None,
            "error": None,
            "legacy_missing": missing,
        })

    return {
        "manifest": {
            "schema_version": int(payload.get("schema_version", 1)),
            "benchmark": path.parent.name,
            "strategies": [{"name": strategy_id, "source": str(path)}],
            "legacy": True,
        },
        "records": records,
        "summary": {},
    }


def _legacy_calls_for_task(
    task_id: str,
    calls_by_task: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    exact = calls_by_task.get(task_id)
    if exact is not None:
        return exact
    # Older LiveBench runners logged the first 16 characters while result rows
    # retained the full question ID. Only accept a unique prefix match.
    matches = [calls for key, calls in calls_by_task.items() if task_id.startswith(key)]
    return matches[0] if len(matches) == 1 else []


def _verify_resume_manifest(
    existing: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> None:
    for key in (
        "benchmark",
        "dataset",
        "strategies",
        "case_ids",
        "case_digest",
        "pricing",
        "workers",
        "runtime",
    ):
        if existing.get(key) != requested.get(key):
            raise ValueError(f"Cannot resume: manifest field {key!r} has changed")


def _read_completed_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(record["strategy_id"]), str(record["task_id"]))
        for record in _read_jsonl(path)
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _repair_torn_final_line(path: Path) -> None:
    """Discard one incomplete final JSONL record while preserving strictness.

    A process can be interrupted between writing a record and its terminating
    newline. Only the final non-blank line is eligible for repair: malformed
    JSON anywhere earlier still indicates artifact corruption and is raised.
    """

    if not path.exists():
        return

    data = path.read_bytes()
    nonblank_lines: list[tuple[int, bytes]] = []
    offset = 0
    for line in data.splitlines(keepends=True):
        start = offset
        offset += len(line)
        payload = line.rstrip(b"\r\n")
        if payload.strip():
            nonblank_lines.append((start, payload))

    if not nonblank_lines:
        return

    for _, payload in nonblank_lines[:-1]:
        json.loads(payload.decode("utf-8"))

    final_start, final_payload = nonblank_lines[-1]
    if data.endswith((b"\n", b"\r")):
        # A terminated record is not torn. Keep malformed complete lines
        # visible as corruption instead of silently dropping evidence.
        json.loads(final_payload.decode("utf-8"))
        return

    try:
        json.loads(final_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        if data.endswith((b"\n", b"\r")):
            raise
        with path.open("r+b") as handle:
            handle.truncate(final_start)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        # A complete record can be persisted just before its newline. Finish
        # that separator so the next append does not concatenate two objects.
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
