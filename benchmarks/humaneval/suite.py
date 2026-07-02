"""HumanEval dataset loading and correctness evaluation."""

from __future__ import annotations

from typing import Sequence

from harness import run_tests

from benchmarks.suite import BenchmarkCase, Evaluation


DATASET_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"


class HumanEvalSuite:
    """Evaluate generated Python function completions on HumanEval tests."""

    name = "humaneval"
    dataset_identity = {
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "revision": DATASET_REVISION,
    }

    def __init__(self, dataset_loader=None):
        self.dataset_loader = dataset_loader

    def load_cases(self, limit: int | None = None) -> Sequence[BenchmarkCase]:
        loader = self.dataset_loader or _load_dataset
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
        passed = run_tests(
            case.payload["prompt"],
            output,
            case.payload["test"],
            case.payload["entry_point"],
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
