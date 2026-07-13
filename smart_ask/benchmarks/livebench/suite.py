"""LiveBench coding dataset loading and noncanonical public-test smoke checks."""

from __future__ import annotations

import ast
import json
from numbers import Integral
import os
import re
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..code_output import extract_code
from ..suite import BenchmarkCase, Evaluation


DATASET_REVISION = "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4"


_STDLIB = (
    "from typing import List, Tuple, Dict, Optional, Set, Any, Union, Counter\n"
    "import sys, math, re, collections, itertools, functools, heapq, bisect, ast\n"
    "from collections import defaultdict, Counter, deque\n"
)


class LiveBenchPublicTestsSuite:
    """Smoke-test answers against public cases, not canonical LiveBench scoring."""

    name = "livebench-coding-public-tests"
    executes_untrusted_code = True
    dataset_identity = MappingProxyType({
        "dataset": "livebench/coding",
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
            "type": "smart-ask-livebench-public-test-smoke",
            "implementation_version": 1,
            "timeout_seconds_per_test": self._timeout,
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
        cases = []
        for row in rows:
            original = row["original_json"]
            if isinstance(original, str):
                original = json.loads(original)
            test_cases = row["public_test_cases"]
            if isinstance(test_cases, str):
                test_cases = json.loads(test_cases)
            cases.append(BenchmarkCase(
                task_id=row["question_id"],
                prompt=row["turns"][0],
                payload={
                    "title": row["question_title"],
                    "task": row["task"],
                    "starter_code": original.get("starter_code", ""),
                    "partial": row.get("partial_solution", ""),
                    "test_cases": test_cases,
                    "difficulty": original.get("difficulty", "?"),
                },
            ))
        return cases

    def evaluate(self, case: BenchmarkCase, output: str) -> Evaluation:
        if not self._unsafe_execution_allowed:
            raise RuntimeError(
                "LiveBench executes model-generated code without an OS sandbox; "
                "explicit unsafe execution opt-in is required"
            )
        code = extract_code(output)
        if case.payload["task"] == "coding_completion" and case.payload["partial"]:
            code = (
                code
                if "class Solution" in code
                else case.payload["partial"] + "\n" + code
            )
        passed, total = run_public_tests(
            code,
            case.payload["starter_code"],
            case.payload["test_cases"],
            timeout=self._timeout,
        )
        pass_all = passed == total and total > 0
        return Evaluation(
            passed=pass_all,
            score=passed / total if total else 0.0,
            details={
                "passed_tests": passed,
                "total_tests": total,
                "title": case.payload["title"],
                "task": case.payload["task"],
                "difficulty": case.payload["difficulty"],
            },
        )


def run_public_tests(
    code: str,
    starter_code: str,
    test_cases: Sequence[Mapping[str, Any]],
    timeout: int,
) -> tuple[int, int]:
    """Execute the suite's approximate public functional or stdin checks."""

    if isinstance(timeout, bool) or not isinstance(timeout, Integral) or timeout < 1:
        raise ValueError("timeout must be a positive integer")
    timeout = int(timeout)
    passed = 0
    for test_case in test_cases:
        if test_case.get("testtype", "functional") == "functional":
            ok = _run_functional(code, starter_code, test_case, timeout)
        else:
            ok = _run_stdin(code, test_case, timeout)
        passed += int(ok)
    return passed, len(test_cases)


def _run_functional(
    code: str,
    starter_code: str,
    test_case: Mapping[str, Any],
    timeout: int,
) -> bool:
    match = re.search(r"def (\w+)\(self", starter_code)
    if not match:
        return False
    function_name = match.group(1)
    values = [
        _parse_value(line.strip())
        for line in test_case["input"].splitlines()
        if line.strip()
    ]
    if not values:
        return False
    call = (
        f"_s.{function_name}({values[0]!r})"
        if len(values) == 1
        else f"_s.{function_name}(*{tuple(values)!r})"
    )
    expected = _parse_value(test_case["output"].strip())
    source = _STDLIB + "\n" + code + f"\n\n_s = Solution()\nprint(repr({call}))\n"
    completed = _run_source(source, timeout=timeout)
    if completed.returncode != 0:
        return False
    try:
        actual = ast.literal_eval(completed.stdout.strip())
    except (SyntaxError, ValueError):
        return False
    return actual == expected


def _run_stdin(code: str, test_case: Mapping[str, Any], timeout: int) -> bool:
    completed = _run_source(
        _STDLIB + "\n" + code,
        stdin=test_case["input"],
        timeout=timeout,
    )
    return (
        completed.returncode == 0
        and completed.stdout.strip() == test_case["output"].strip()
    )


def _run_source(source: str, *, stdin: str | None = None, timeout: int):
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        delete=False,
    ) as handle:
        handle.write(source)
        filename = handle.name
    try:
        return subprocess.run(
            [sys.executable, filename],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 124, "", "timeout")
    finally:
        os.unlink(filename)


def _parse_value(value: str):
    try:
        return ast.literal_eval(value.strip())
    except (SyntaxError, ValueError):
        return value.strip()


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run LiveBench") from exc
    return load_dataset(
        "livebench/coding",
        split="test",
        revision=DATASET_REVISION,
    )
