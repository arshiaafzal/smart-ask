"""Dataset and evaluation contracts owned by the benchmark application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkCase:
    """One stable dataset item presented to every strategy being compared."""

    task_id: str
    prompt: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    """Dataset-specific assessment of one strategy's final output."""

    passed: bool
    score: float
    details: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkSuite(Protocol):
    """Load a fixed case set and evaluate final model outputs."""

    name: str
    dataset_identity: Mapping[str, str]

    def load_cases(self, limit: int | None = None) -> Sequence[BenchmarkCase]:
        """Load the ordered case set shared by all strategies in one run."""

        ...

    def evaluate(self, case: BenchmarkCase, output: str) -> Evaluation:
        """Evaluate one final output using suite-owned correctness logic."""

        ...
