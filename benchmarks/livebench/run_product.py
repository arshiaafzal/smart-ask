#!/usr/bin/env python3
"""Compatibility wrapper for the configurable LiveBench benchmark command."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.artifacts import load_run
from benchmarks.compare import compare, format_report, summarize
from benchmarks.livebench.__main__ import main as benchmark_main
from benchmarks.livebench.suite import LiveBenchSuite, run_tests


DEFAULT_STRATEGY = ROOT / "strategies" / "python-code-generation-cascade.yaml"
LEGACY_RESULTS = Path(__file__).with_name("results_product.json")


def load_livebench():
    """Return the pinned LiveBench rows for older callers."""

    suite = LiveBenchSuite()
    return list(suite.load_cases())


def main(argv=None):
    """Run the default strategy or render the checked-in legacy result."""

    args = list(sys.argv[1:] if argv is None else argv)
    if "--report" in args:
        loaded = load_run(LEGACY_RESULTS)
        summaries = summarize(loaded["records"])
        comparison = compare(loaded["records"])
        print(format_report(summaries, comparison))
        return
    if "--strategy" not in args:
        args[0:0] = ["--strategy", str(DEFAULT_STRATEGY)]
    return benchmark_main(args)


if __name__ == "__main__":
    main()
