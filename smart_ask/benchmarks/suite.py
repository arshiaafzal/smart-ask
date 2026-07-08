"""Dataset and evaluation contracts owned by the benchmark application."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Protocol, Sequence

from .._numeric import is_finite_real
from ..strategy.schema import StrategyConfig


@dataclass(frozen=True)
class BenchmarkCase:
    """One stable dataset item presented to every strategy being compared."""

    task_id: str
    prompt: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or self.task_id != self.task_id.strip()
        ):
            raise ValueError("task_id must be a non-empty trimmed string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        payload = _freeze_json(self.payload, "payload")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True)
class Evaluation:
    """Dataset-specific assessment of one strategy's final output."""

    passed: bool
    score: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, Real)
            or not is_finite_real(self.score)
        ):
            raise ValueError("score must be finite")
        details = _freeze_json(self.details, "details")
        if not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "details", details)


class _FrozenDict(dict):
    """A JSON-serializable dictionary that cannot drift after construction."""

    def _immutable(self, *_args, **_kwargs):
        raise TypeError("benchmark JSON snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        if not is_finite_real(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return _FrozenDict(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Return the canonical mutable JSON form of a frozen benchmark value."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class BenchmarkSuite(Protocol):
    """Load a fixed case set and evaluate final model outputs."""

    name: str
    dataset_identity: Mapping[str, str]
    evaluator_identity: Mapping[str, Any]

    def load_cases(self, limit: int | None = None) -> Sequence[BenchmarkCase]:
        """Load the ordered case set shared by all strategies in one run."""

        ...

    def evaluate(self, case: BenchmarkCase, output: str) -> Evaluation:
        """Evaluate one final output using suite-owned correctness logic."""

        ...


class BenchmarkStrategy(Protocol):
    """Validated strategy surface required by the benchmark runner."""

    config: StrategyConfig
    digest: str

    def manifest(self) -> dict[str, Any]:
        ...
