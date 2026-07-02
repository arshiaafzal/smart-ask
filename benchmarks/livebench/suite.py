"""LiveBench coding dataset loading and public-test evaluation."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Sequence

from benchmarks.suite import BenchmarkCase, Evaluation


DATASET_REVISION = "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4"


_STDLIB = (
    "from typing import List, Tuple, Dict, Optional, Set, Any, Union, Counter\n"
    "import sys, math, re, collections, itertools, functools, heapq, bisect, ast\n"
    "from collections import defaultdict, Counter, deque\n"
)


class LiveBenchSuite:
    """Evaluate LiveBench coding answers against all public test cases."""

    name = "livebench-coding"
    dataset_identity = {
        "dataset": "livebench/coding",
        "split": "test",
        "revision": DATASET_REVISION,
    }

    def __init__(self, dataset_loader=None, timeout: int = 10):
        self.dataset_loader = dataset_loader
        self.timeout = timeout

    def load_cases(self, limit: int | None = None) -> Sequence[BenchmarkCase]:
        loader = self.dataset_loader or _load_dataset
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
        code = output
        if case.payload["task"] == "coding_completion" and case.payload["partial"]:
            code = (
                code
                if "class Solution" in code
                else case.payload["partial"] + "\n" + code
            )
        passed, total = run_tests(
            code,
            case.payload["starter_code"],
            case.payload["test_cases"],
            timeout=self.timeout,
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


def run_tests(
    code: str,
    starter_code: str,
    test_cases: list,
    timeout: int = 10,
) -> tuple[int, int]:
    """Execute all public functional or stdin-style test cases."""

    passed = 0
    for test_case in test_cases:
        try:
            if test_case.get("testtype", "functional") == "functional":
                ok = _run_functional(code, starter_code, test_case, timeout)
            else:
                ok = _run_stdin(code, test_case, timeout)
        except Exception:
            ok = False
        passed += int(ok)
    return passed, len(test_cases)


def _run_functional(
    code: str,
    starter_code: str,
    test_case: dict,
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
    actual = _parse_value(completed.stdout.strip().strip("'\""))
    return _normalize(actual) == _normalize(expected)


def _run_stdin(code: str, test_case: dict, timeout: int) -> bool:
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
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
    except Exception:
        return value.strip()


def _normalize(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_normalize(item) for item in value) + "]"
    return str(value)


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
