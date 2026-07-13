"""HumanEval dataset loading and correctness evaluation."""

from __future__ import annotations

from numbers import Integral
from types import MappingProxyType
from typing import Sequence

from ..suite import BenchmarkCase, Evaluation
from .harness import run_tests


DATASET_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"


class HumanEvalSuite:
    """Evaluate generated Python function completions on HumanEval tests."""

    name = "humaneval"
    executes_untrusted_code = True
    dataset_identity = MappingProxyType({
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "revision": DATASET_REVISION,
    })

    def __init__(self, dataset_loader=None, timeout: int = 10):
        if dataset_loader is not None and not callable(dataset_loader):
            raise TypeError("dataset_loader must be callable or None")
        if isinstance(timeout, bool) or not isinstance(timeout, Integral) or timeout < 1:
            raise ValueError("timeout must be a positive integer")
        self._dataset_loader = dataset_loader
        self._timeout = int(timeout)
        self._unsafe_execution_allowed = False
        self._evaluator_identity = MappingProxyType({
            "type": "humaneval-subprocess",
            "timeout_seconds": self._timeout,
        })

    @property
    def evaluator_identity(self):
        return self._evaluator_identity

    def allow_unsafe_code_execution(self) -> None:
        self._unsafe_execution_allowed = True

    def load_cases(self, limit: int | None = None) -> Sequence[BenchmarkCase]:
        loader = (
            _load_dataset
            if self._dataset_loader is None
            else self._dataset_loader
        )
        rows = list(loader())
        if limit is not None:
            rows = rows[:limit]
        return [
            BenchmarkCase(
                task_id=row["task_id"],
                prompt=f"Complete this Python function:\n\n{row['prompt']}",
                payload={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
            for row in rows
        ]

    def evaluate(self, case: BenchmarkCase, output: str) -> Evaluation:
        if not self._unsafe_execution_allowed:
            raise RuntimeError(
                "HumanEval executes model-generated code without an OS sandbox; "
                "explicit unsafe execution opt-in is required"
            )
        passed = run_tests(
            case.payload["prompt"],
            output,
            case.payload["test"],
            case.payload["entry_point"],
            timeout=self._timeout,
        )
        return Evaluation(
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"entry_point": case.payload["entry_point"]},
        )


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run HumanEval") from exc
    return load_dataset(
        "openai/openai_humaneval",
        split="test",
        revision=DATASET_REVISION,
    )
