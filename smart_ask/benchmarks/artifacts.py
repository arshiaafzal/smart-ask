"""Crash-safe persistence for strictly validated benchmark artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable, Mapping, Protocol

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # Windows
    if _fcntl is None:
        import msvcrt as _msvcrt
    else:
        _msvcrt = None
except ImportError:  # pragma: no cover - unsupported platform guard
    _msvcrt = None

from .artifact_schema import (
    SCHEMA_VERSION,
    validate_derived_reports,
    validate_manifest,
    validate_record,
    validate_records,
    validate_summary_artifact,
    verify_resume_manifest,
)


class ResultSink(Protocol):
    """Persist completed strategy/task records as they become available."""

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        ...

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        ...

    def start(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        ...

    def append(self, record: Mapping[str, Any]) -> None:
        ...

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        ...

    def close(self) -> None:
        ...


class JsonlResultSink:
    """Append records to JSONL while atomically persisting run metadata."""

    def __init__(self, directory: str | Path, *, resume: bool = False):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.records_path = self.directory / "records.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.lock_path = self.directory / ".run.lock"
        self._thread_lock = threading.Lock()
        self._lock_file: Any | None = None
        self._completed: set[tuple[str, str]] = set()
        self._records: list[Mapping[str, Any]] = []
        self._manifest: dict[str, Any] | None = None
        self._resume = resume
        self._started = False
        self._finalized = False
        self._closed = False

    def start(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if self._started:
            raise RuntimeError("Result sink has already been started")
        self._acquire_run_lock()
        try:
            return self._start_locked(manifest)
        except BaseException:
            self.close()
            raise

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            return {}
        return _json_safe(self._manifest)

    def _start_locked(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        requested = _normalize_manifest(manifest)
        if self._resume:
            if not self.manifest_path.is_file() or not self.records_path.is_file():
                raise FileNotFoundError(
                    "Cannot resume: both manifest.json and records.jsonl must exist"
                )
            existing = _read_json_object(self.manifest_path, "manifest")
            validate_manifest(existing)
            verify_resume_manifest(existing, requested)
            _repair_torn_final_line(self.records_path)
            records = validate_records(_read_jsonl(self.records_path), existing)
            completed = {
                (record["strategy_id"], record["task_id"])
                for record in records
            }
            if self.summary_path.exists():
                self.summary_path.unlink()
                _fsync_directory(self.directory)
            canonical = dict(existing)
        else:
            contents = [
                path for path in self.directory.iterdir()
                if path != self.lock_path
            ]
            if contents:
                raise FileExistsError(
                    f"Refusing to start a new run in nonempty directory: "
                    f"{self.directory}"
                )
            validate_manifest(requested)
            self.records_path.touch(exist_ok=False)
            _fsync_file(self.records_path)
            _atomic_write_json(self.manifest_path, requested)
            records = []
            completed = set()
            canonical = dict(requested)

        self._records = records
        self._completed = completed
        self._manifest = canonical
        self._started = True
        return _json_safe(canonical)

    def _acquire_run_lock(self) -> None:
        if self._resume:
            if not self.directory.is_dir():
                raise FileNotFoundError(
                    f"Cannot resume missing run directory: {self.directory}"
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+b")
        try:
            _acquire_file_lock(lock_file)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError(
                f"Benchmark run directory is already in use: {self.directory}"
            ) from exc
        self._lock_file = lock_file

    def close(self) -> None:
        if self._closed:
            return
        lock_file = getattr(self, "_lock_file", None)
        if lock_file is None:
            self._closed = True
            return
        try:
            _release_file_lock(lock_file)
        finally:
            lock_file.close()
            self._lock_file = None
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return set(self._completed)

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        return [_json_safe(record) for record in self._records]

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if not self._started or self._manifest is None:
            raise RuntimeError("Result sink must be started before appending records")
        if self._finalized:
            raise RuntimeError("Result sink has already been finalized")
        normalized = _json_safe(dict(record))
        key = validate_record(normalized, self._manifest)
        run_id = normalized["metrics"]["identity"]["run_id"]
        line = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._thread_lock:
            if key in self._completed:
                raise ValueError(
                    "Duplicate benchmark record for "
                    f"strategy {key[0]!r}, task {key[1]!r}"
                )
            if any(
                item["metrics"]["identity"]["run_id"] == run_id
                for item in self._records
            ):
                raise ValueError(f"Duplicate metrics run_id {run_id!r}")
            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._completed.add(key)
            self._records.append(normalized)

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if not self._started or self._manifest is None:
            raise RuntimeError("Result sink must be started before finalizing")
        if self._finalized:
            raise RuntimeError("Result sink has already been finalized")
        validate_derived_reports(
            self._records,
            self._manifest,
            summaries,
            comparison,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "summaries": summaries,
            "comparison": comparison,
        }
        self._finalized = True
        try:
            _atomic_write_json(self.summary_path, _json_safe(payload))
        finally:
            self.close()


class MemoryResultSink:
    """Validated in-memory sink for embedding and network-free tests."""

    def __init__(self):
        self._records: list[dict[str, Any]] = []
        self._summaries: dict[str, Any] = {}
        self._comparison: dict[str, Any] = {}
        self._manifest: dict[str, Any] = {}
        self._started = False
        self._finalized = False
        self._closed = False

    def start(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if self._started:
            raise RuntimeError("Result sink has already been started")
        normalized = _normalize_manifest(manifest)
        validate_manifest(normalized)
        self._manifest = normalized
        self._started = True
        return _json_safe(self._manifest)

    @property
    def manifest(self) -> dict[str, Any]:
        return _json_safe(self._manifest)

    @property
    def records(self) -> list[dict[str, Any]]:
        return [_json_safe(record) for record in self._records]

    @property
    def summaries(self) -> dict[str, Any]:
        return _json_safe(self._summaries)

    @property
    def comparison(self) -> dict[str, Any]:
        return _json_safe(self._comparison)

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return {
            (record["strategy_id"], record["task_id"])
            for record in self._records
        }

    @property
    def existing_records(self) -> list[dict[str, Any]]:
        return [_json_safe(record) for record in self._records]

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if not self._started:
            raise RuntimeError("Result sink must be started before appending records")
        if self._finalized:
            raise RuntimeError("Result sink has already been finalized")
        normalized = _json_safe(dict(record))
        key = validate_record(normalized, self._manifest)
        run_id = normalized["metrics"]["identity"]["run_id"]
        if key in self.completed_keys:
            raise ValueError(
                "Duplicate benchmark record for "
                f"strategy {key[0]!r}, task {key[1]!r}"
            )
        if any(
            item["metrics"]["identity"]["run_id"] == run_id
            for item in self._records
        ):
            raise ValueError(f"Duplicate metrics run_id {run_id!r}")
        self._records.append(normalized)

    def finalize(
        self,
        summaries: Mapping[str, Any],
        comparison: Mapping[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("Result sink is closed")
        if not self._started:
            raise RuntimeError("Result sink must be started before finalizing")
        if self._finalized:
            raise RuntimeError("Result sink has already been finalized")
        validate_derived_reports(
            self._records,
            self._manifest,
            summaries,
            comparison,
        )
        self._summaries = _json_safe(dict(summaries))
        self._comparison = _json_safe(dict(comparison))
        self._finalized = True

    def close(self) -> None:
        """Make the in-memory sink terminal, matching persistent sinks."""

        self._closed = True


def load_run(path: str | Path) -> dict[str, Any]:
    """Load and validate one schema-v5 benchmark run directory."""

    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Benchmark artifact must be a run directory: {directory}"
        )
    manifest_path = directory / "manifest.json"
    records_path = directory / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "Benchmark run requires manifest.json and records.jsonl"
        )

    manifest = _read_json_object(manifest_path, "manifest")
    validate_manifest(manifest)
    records = validate_records(_read_jsonl(records_path), manifest)
    summary_path = directory / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = _read_json_object(summary_path, "summary")
        validate_summary_artifact(summary, records, manifest)
    return {"manifest": manifest, "records": records, "summary": summary}


def _acquire_file_lock(lock_file: Any) -> None:
    """Acquire one nonblocking process lock on POSIX or Windows."""

    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return
    if _msvcrt is None:  # pragma: no cover - no supported lock API
        raise RuntimeError("benchmark persistence requires OS file locking")
    lock_file.seek(0)
    if not lock_file.read(1):
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_NBLCK, 1)
    except OSError as exc:  # pragma: no cover - exercised on Windows
        raise BlockingIOError from exc


def _release_file_lock(lock_file: Any) -> None:
    if _fcntl is not None:
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    if _msvcrt is None:  # pragma: no cover - no supported lock API
        raise RuntimeError("benchmark persistence requires OS file locking")
    lock_file.seek(0)
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)


def _normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_safe(dict(manifest))
    supplied_version = normalized.get("schema_version")
    if supplied_version is not None and supplied_version != SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema_version must be {SCHEMA_VERSION}, "
            f"got {supplied_version!r}"
        )
    normalized["schema_version"] = SCHEMA_VERSION
    return normalized


def _read_json_object(path: Path, kind: str) -> dict[str, Any]:
    payload = _loads_json(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{kind.capitalize()} must be a JSON object")
    return payload


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = _loads_json(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL record on line {line_number} must be an object"
                )
            yield payload


def _repair_torn_final_line(path: Path) -> None:
    """Repair only one incomplete final JSONL line."""

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
        _loads_json(payload.decode("utf-8"))
    final_start, final_payload = nonblank_lines[-1]
    if data.endswith((b"\n", b"\r")):
        _loads_json(final_payload.decode("utf-8"))
        return
    try:
        _loads_json(final_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(final_start)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    # Windows supports flushing the replaced file but not opening directories
    # through ``os.open`` for a POSIX-style directory fsync.
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("artifact object keys must be strings")
            normalized[key] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact numbers must be finite")
        return value
    raise TypeError(
        f"artifact values must be JSON-compatible, got {type(value).__name__}"
    )


def _loads_json(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value
