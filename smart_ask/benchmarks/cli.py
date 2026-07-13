"""Shared module-entrypoint plumbing for benchmark suites."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from ..metrics import PriceCatalog
from .artifacts import JsonlResultSink
from .compare import format_report
from .runner import run_matrix


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
        metavar="YAML|builtin:NAME",
        help="Strategy YAML or bundled strategy name; repeat to compare",
    )
    parser.add_argument("-n", "--limit", type=_positive_int, default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unsafe-code-execution",
        action="store_true",
        help="Allow benchmark-generated code to run without an OS sandbox",
    )
    parser.add_argument(
        "--price-catalog",
        type=Path,
        default=None,
        metavar="JSON",
        help="Versioned price catalog for models absent from the bundled snapshot",
    )
    args = parser.parse_args(argv)

    if getattr(suite, "executes_untrusted_code", False):
        if not args.allow_unsafe_code_execution:
            parser.error(
                "this suite executes model-generated code without an OS sandbox; "
                "pass --allow-unsafe-code-execution only in an isolated environment"
            )
        enable = getattr(suite, "allow_unsafe_code_execution", None)
        if not callable(enable):
            raise TypeError("unsafe benchmark suite lacks an explicit opt-in hook")
        enable()

    if args.resume and args.output is None:
        parser.error("--resume requires an explicit --output directory")

    if strategy_loader is None or builder_factory is None:
        from smart_ask.strategy import StrategyBuilder, load_strategy

        strategy_loader = strategy_loader or load_strategy
        builder_factory = builder_factory or StrategyBuilder

    loaded = [strategy_loader(path) for path in args.strategy]
    output = args.output or _default_output(suite.name)
    price_catalog = _load_price_catalog(args.price_catalog)
    sink = sink_factory(output, resume=args.resume)
    builder = None

    def get_builder():
        nonlocal builder
        if builder is None:
            builder = builder_factory(env=os.environ)
        return builder

    def engine_factory(strategy):
        return get_builder().build_engine(strategy)

    def deployment_manifest_factory(strategy):
        return get_builder().deployment_manifest(strategy)

    def progress(record, done, total):
        evaluation = record["evaluation"]
        symbol = "✓" if evaluation["passed"] else "✗"
        print(
            f"[{done:>4}/{total}] {symbol} "
            f"{record['strategy_id']:<20} {record['task_id']}"
        )

    try:
        result = run_matrix(
            suite,
            loaded,
            engine_factory=engine_factory,
            sink=sink,
            workers=args.workers,
            limit=args.limit,
            price_catalog=price_catalog,
            deployment_manifest_factory=deployment_manifest_factory,
            progress=progress,
        )
    finally:
        if builder is not None:
            closer = getattr(builder, "aclose", None)
            if callable(closer):
                asyncio.run(closer())
    print()
    print(format_report(result.summaries, result.comparison))
    print(f"\nArtifacts: {output}")
    return result


def _default_output(suite_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("benchmark-results") / suite_name / stamp


def _load_price_catalog(path: Path | None) -> PriceCatalog | None:
    if path is None:
        return None
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("price catalog must be a JSON object")
    expected = {"catalog_id", "effective_date", "source", "prices"}
    supplied = set(payload)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError("invalid price catalog fields (" + "; ".join(details) + ")")
    return PriceCatalog(
        catalog_id=payload["catalog_id"],
        effective_date=payload["effective_date"],
        source=payload["source"],
        prices=payload["prices"],
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value
