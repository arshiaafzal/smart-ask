"""Shared module-entrypoint plumbing for benchmark suites."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import JsonlResultSink
from .compare import format_report
from .runner import TracedExecutor, run_matrix


def run_suite_cli(
    suite,
    argv: Sequence[str] | None = None,
    *,
    strategy_loader: Callable[[str | Path], Any] | None = None,
    builder_factory: Callable[..., Any] | None = None,
    sink_factory: Callable[..., Any] = JsonlResultSink,
):
    """Load repeated strategy YAMLs, run one suite, and print comparisons."""

    parser = argparse.ArgumentParser(
        description=f"Compare routing strategies on {suite.name}",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        metavar="YAML",
        help="Strategy YAML to evaluate; repeat to compare several strategies",
    )
    parser.add_argument("-n", "--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.resume and args.output is None:
        parser.error("--resume requires an explicit --output directory")

    if strategy_loader is None or builder_factory is None:
        from smart_ask.strategy import StrategyBuilder, load_strategy

        strategy_loader = strategy_loader or load_strategy
        builder_factory = builder_factory or StrategyBuilder

    loaded = [strategy_loader(path) for path in args.strategy]
    output = args.output or _default_output(suite.name)
    sink = sink_factory(output, resume=args.resume)

    def application_factory(strategy, recorder):
        builder = builder_factory(
            env=os.environ,
            executor_wrapper=lambda executor, channel: TracedExecutor(
                executor,
                recorder,
                channel,
            ),
        )
        application = builder.build(strategy)
        if not getattr(application.executor, "captures_output", False):
            raise ValueError(
                f"benchmark strategy {strategy.config.name!r} must use a "
                "generation executor that captures output"
            )
        return application

    def progress(record, done, total):
        evaluation = record["evaluation"]
        symbol = "✓" if evaluation["passed"] else "✗"
        print(
            f"[{done:>4}/{total}] {symbol} "
            f"{record['strategy_id']:<20} {record['task_id']}"
        )

    result = run_matrix(
        suite,
        loaded,
        application_factory=application_factory,
        sink=sink,
        workers=args.workers,
        limit=args.limit,
        progress=progress,
    )
    print()
    print(format_report(result.summaries, result.comparison))
    print(f"\nArtifacts: {output}")
    return result


def _default_output(suite_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("benchmarks") / "results" / suite_name / stamp
