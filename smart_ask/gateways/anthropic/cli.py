"""Command-line entry point for the Anthropic protocol gateway."""

from __future__ import annotations

import argparse
import os
import sys

from smart_ask.conversation import RunMetricsStore
from smart_ask.metrics import JsonlMetricsSink
from smart_ask.observability import InvocationLogSink

from .app import create_app
from .catalog import StrategyCatalog
from .config import GatewayConfigError, load_gateway_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="smart-ask gateway anthropic",
        description="Run SmartAsk's Anthropic-compatible protocol gateway",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve")
    serve.add_argument("--config", required=True, metavar="GATEWAY.yaml")
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.error("unknown command")
    metrics_sink = None
    trace_sink = None
    try:
        import uvicorn

        config = load_gateway_config(args.config)
        env = dict(os.environ)
        if config.metrics.jsonl_path is not None:
            metrics_sink = JsonlMetricsSink(config.metrics.jsonl_path)
        if config.metrics.trace_directory is not None:
            trace_sink = InvocationLogSink(config.metrics.trace_directory)
        metrics = RunMetricsStore(
            sink=None if metrics_sink is None else metrics_sink.write,
        )
        catalog = StrategyCatalog.from_config(
            config,
            env=env,
            metrics=metrics,
            trace_observer=trace_sink,
        )
        app = create_app(config, catalog, env=env)
    except ModuleNotFoundError as exc:
        if exc.name != "uvicorn":
            raise
        if metrics_sink is not None:
            metrics_sink.close()
        if trace_sink is not None:
            trace_sink.close()
        sys.exit(
            "error: Anthropic gateway dependencies are missing; "
            "install smart-ask[anthropic-gateway]"
        )
    except (GatewayConfigError, OSError, ValueError) as exc:
        if metrics_sink is not None:
            metrics_sink.close()
        if trace_sink is not None:
            trace_sink.close()
        sys.exit(f"error: {exc}")
    for entry in catalog:
        print(
            f"{entry.model_id} -> {entry.reference} "
            f"({entry.loaded.digest[:12]})"
        )
    try:
        uvicorn.run(
            app,
            host=config.listen.host,
            port=config.listen.port,
            access_log=False,
        )
    finally:
        if metrics_sink is not None:
            metrics_sink.close()
        if trace_sink is not None:
            trace_sink.close()
