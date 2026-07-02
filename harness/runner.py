"""Execute generated Python against HumanEval-style tests."""

import os
import subprocess
import sys
import tempfile


def strip_fences(text: str) -> str:
    """Remove an outer Markdown fence while preserving leading indentation."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return text.rstrip()
    lines = stripped.splitlines()
    inner = lines[1:] if len(lines) > 1 else lines
    if inner and inner[-1].strip() == "```":
        inner = inner[:-1]
    return "\n".join(inner).rstrip()


def run_tests(
    prompt: str,
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> bool:
    """Run generated code and a HumanEval test suite in a fresh subprocess."""

    code = strip_fences(code)
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(source)
        filename = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            timeout=timeout,
        )
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(filename)
