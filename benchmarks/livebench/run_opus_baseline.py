#!/usr/bin/env python3
"""Compatibility wrapper for the configured fixed-Opus strategy."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.livebench.__main__ import main as benchmark_main


DEFAULT_STRATEGY = ROOT / "strategies" / "python-code-generation-fixed-opus.yaml"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--strategy" not in args:
        args[0:0] = ["--strategy", str(DEFAULT_STRATEGY)]
    return benchmark_main(args)


if __name__ == "__main__":
    main()
