"""Command-line entry point for the external Claude Code adapter."""

from __future__ import annotations

import argparse
import os
import sys

from smart_ask.conversation import ConversationMetricsStore

from .app import create_app
from .catalog import StrategyCatalog
from .config import AdapterConfigError, load_adapter_config
from .metrics import JsonlSink
from .trace import JsonlTraceSink


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="smart-ask-claude-code",
        description="Run the external Claude Code adapter for SmartAsk",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve")
    serve.add_argument("--config", required=True, metavar="ADAPTER.yaml")
    args = parser.parse_args(argv)
    if args.command != "serve":
        parser.error("unknown command")
    metrics_sink = None
    trace_sink = None
    try:
        import uvicorn

        config = load_adapter_config(args.config)
        env = dict(os.environ)
        if config.metrics.jsonl_path is not None:
            metrics_sink = JsonlSink(config.metrics.jsonl_path)
        if config.metrics.trace_jsonl_path is not None:
            trace_sink = JsonlTraceSink(config.metrics.trace_jsonl_path)
        metrics = ConversationMetricsStore(
            sink=None if metrics_sink is None else metrics_sink.write,
        )
        catalog = StrategyCatalog.from_config(
            config,
            env=env,
            metrics=metrics,
            trace_sink=None if trace_sink is None else trace_sink.write,
        )
        app = create_app(config, catalog, env=env)
    except (AdapterConfigError, OSError, ValueError) as exc:
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
