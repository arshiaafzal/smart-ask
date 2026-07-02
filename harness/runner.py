"""
Code execution harness for benchmark evaluation.

Runs generated Python code against HumanEval-style test suites in a
sandboxed subprocess. Each run gets a fresh Python interpreter.

Usage
-----
    from harness import strip_fences, run_tests

    passed = run_tests(prompt, model_output, test_code, "entry_point_name")
"""

import os, sys, subprocess, tempfile


def strip_fences(text: str) -> str:
    """Remove markdown code fences (``` or ```python) from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()


def run_tests(
    prompt: str,
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> bool:
    """
    Execute `code` against `test_code` in a fresh subprocess.

    Writes a temporary .py file containing:
      1. Common stdlib imports (typing, math, re, collections, …)
      2. The implementation (strips fences; prepends prompt if `def entry_point`
         is not found in `code` — handles partial completions)
      3. The HumanEval test suite
      4. `check(entry_point)` call

    Returns True if all tests pass (subprocess exits 0), False otherwise.
    Times out after `timeout` seconds.
    """
    code = strip_fences(code)
    impl = code if f"def {entry_point}" in code else prompt + code
    full = (
        "from typing import List, Tuple, Dict, Optional, Set, Any, Union\n"
        "import math, re, collections, itertools, functools, heapq, bisect\n\n"
        + impl + "\n\n"
        + test_code + "\n\n"
        + f"check({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname], capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)
