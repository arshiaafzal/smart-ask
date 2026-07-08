"""Build reproducible benchmark run manifests and code identities."""

from __future__ import annotations

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
from typing import Any, Iterable, Sequence

from ..metrics import METRICS_WIRE_SCHEMA, PriceCatalog
from .artifact_schema import SCHEMA_VERSION
from .suite import BenchmarkCase, BenchmarkStrategy, BenchmarkSuite


def build_manifest(
    suite: BenchmarkSuite,
    strategies: Sequence[BenchmarkStrategy],
    cases: Sequence[BenchmarkCase],
    workers: int,
    price_catalog: PriceCatalog,
) -> dict[str, Any]:
    """Snapshot every input needed to identify and reproduce one run."""

    case_ids = [case.task_id for case in cases]
    case_identity = [
        {
            "task_id": case.task_id,
            "prompt_sha256": hashlib.sha256(
                case.prompt.encode("utf-8")
            ).hexdigest(),
            "payload_sha256": hashlib.sha256(
                json.dumps(
                    dict(case.payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
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
        "evaluator": dict(suite.evaluator_identity),
        "case_ids": case_ids,
        "cases": case_identity,
        "case_digest": case_digest,
        "workers": workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategies": [strategy.manifest() for strategy in strategies],
        "pricing": {
            **price_catalog.to_dict(),
            "currency": "USD",
        },
        "metrics": {
            "schema": METRICS_WIRE_SCHEMA,
            "scope": "run",
            "record_unit": "strategy-task",
            "interaction_unit": "model-executor-call",
        },
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
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _code_identity(
    repository: Path | None = None,
    *,
    package_root: Path | None = None,
) -> dict[str, Any]:
    """Identify installed bytes, plus Git state only for a matching checkout."""

    package_root = (
        Path(__file__).resolve().parents[1]
        if package_root is None
        else Path(package_root).resolve()
    )
    identity = {
        "package_version": _installed_package_version(),
        "package_hash": _package_source_hash(package_root),
        "git_commit": None,
        "dirty": None,
        "dirty_hash": None,
    }
    repository = (
        _matching_repository(package_root)
        if repository is None
        else Path(repository).resolve()
    )
    if repository is None:
        return identity

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
        identity.update({
            "git_commit": commit,
            "dirty": dirty,
            "dirty_hash": (
                _dirty_worktree_hash(repository, tracked_diff, untracked)
                if dirty
                else None
            ),
        })
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def _installed_package_version() -> str | None:
    try:
        return metadata.version("smart-ask")
    except metadata.PackageNotFoundError:
        return None


def _package_source_hash(package_root: Path) -> str:
    """Hash Python modules and installed package data independent of mtimes."""

    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise ValueError(f"smart_ask package root does not exist: {package_root}")
    included = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and _is_shipped_source(path.relative_to(package_root))
    )
    if not included:
        raise ValueError(f"smart_ask package root contains no source files: {package_root}")
    digest = hashlib.sha256()
    for path in included:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        _update_hash_frame(digest, relative)
        _update_hash_frame(digest, path.read_bytes())
    return digest.hexdigest()


def _is_shipped_source(path: Path) -> bool:
    if path.suffix in {".py", ".pyi"}:
        return True
    if path.parts and path.parts[0] == "resources":
        return True
    return path == Path("benchmarks/humaneval/README.md")


def _matching_repository(package_root: Path) -> Path | None:
    """Find only a checkout whose source package is this exact package tree."""

    for candidate in package_root.parents:
        marker = candidate / ".git"
        source_package = candidate / "smart_ask"
        if not marker.exists() or not source_package.is_dir():
            continue
        try:
            if source_package.samefile(package_root):
                return candidate
        except OSError:
            continue
    return None


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
    """Exclude explicit benchmark artifact roots, never generic directory names."""

    candidates = list(paths)
    decoded = {Path(os.fsdecode(path)) for path in candidates}
    artifact_roots = {
        path.parent
        for path in decoded
        if path.name == "records.jsonl"
        and path.parent / "manifest.json" in decoded
    }
    return sorted(
        raw_path
        for raw_path in candidates
        if not _is_default_benchmark_output(Path(os.fsdecode(raw_path)))
        and not any(
            Path(os.fsdecode(raw_path)) == root
            or root in Path(os.fsdecode(raw_path)).parents
            for root in artifact_roots
        )
    )


def _is_default_benchmark_output(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] == "benchmark-results"


def _update_hash_frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
