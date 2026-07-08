"""Execute generated Python against HumanEval tests."""

import os
from numbers import Integral
import subprocess
import sys
import tempfile

from ..code_output import extract_code


def run_tests(
    prompt: str,
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int,
) -> bool:
    """Run generated code and a HumanEval test suite in a fresh subprocess."""

    if isinstance(timeout, bool) or not isinstance(timeout, Integral) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    code = extract_code(code)
    implementation = code if f"def {entry_point}" in code else prompt + code
    source = (
        "from typing import List, Tuple, Dict, Optional, Set, Any, Union\n"
        "import math, re, collections, itertools, functools, heapq, bisect\n\n"
        + implementation
        + "\n\n"
        + test_code
        + "\n\n"
        + f"check({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        delete=False,
    ) as handle:
        handle.write(source)
        filename = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            timeout=int(timeout),
        )
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(filename)


__all__ = ["run_tests"]
