"""Command-line entry point for the external Claude Code adapter."""

from __future__ import annotations

import argparse
import os
import sys

from smart_ask.conversation import ConversationMetricsStore

from .app import create_app
from .catalog import StrategyCatalog
from .config import AdapterConfigError, load_adapter_config
from .metrics import JsonlMetricsSink


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
    sink = None
    try:
        import uvicorn

        config = load_adapter_config(args.config)
        env = dict(os.environ)
        if config.metrics.jsonl_path is not None:
            sink = JsonlMetricsSink(config.metrics.jsonl_path)
        metrics = ConversationMetricsStore(
            sink=None if sink is None else sink.write,
        )
        catalog = StrategyCatalog.from_config(config, env=env, metrics=metrics)
        app = create_app(config, catalog, env=env)
    except (AdapterConfigError, OSError, ValueError) as exc:
        if sink is not None:
            sink.close()
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
        if sink is not None:
            sink.close()
