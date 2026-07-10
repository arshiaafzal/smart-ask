"""
smart-ask: run a configured model-routing strategy.

The selected strategy defines its method, prompts, models, parameters, and
classifier/generation transports.
"""

import argparse
from pathlib import Path
import sys

from . import RunStats, Task, aggregate_stats
from . import _terminal
from .strategy import (
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfigError,
    load_strategy,
)
# ── File context ──────────────────────────────────────────────────────
def _build_file_context(files: list[str]) -> str:
    parts = []
    for path in files:
        try:
            content = Path(path).read_text(encoding="utf-8")
            parts.append(f"--- file: {path} ---\n{content}\n---\n")
        except (OSError, UnicodeError) as exc:
            _terminal.warn(f"could not read {path}: {exc}")
    return "\n".join(parts) + "\n" if parts else ""

# ── Main ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description="Run tasks through a configured model-routing strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument("prompt",           nargs="*")
    p.add_argument("-f", "--file",     action="append", dest="files", metavar="FILE")
    forced = p.add_mutually_exclusive_group()
    forced.add_argument("--force-easy", action="store_true")
    forced.add_argument("--force-hard", action="store_true")
    p.add_argument(
        "--strategy",
        default="builtin:product",
        metavar="FILE|builtin:NAME",
        help="Strategy YAML or bundled strategy name (default: builtin:product)",
    )
    p.add_argument("--validate-strategy", action="store_true")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("-h", "--help",     action="store_true")
    args = p.parse_args(argv)

    try:
        loaded_strategy = load_strategy(args.strategy)
    except StrategyConfigError as exc:
        sys.exit(_terminal.format_error(exc))

    if args.validate_strategy:
        print(
            f"{loaded_strategy.config.name}: valid "
            f"({loaded_strategy.digest[:12]})"
        )
        return

    # ── Welcome ────────────────────────────────────────────────────────
    if args.help or (not args.prompt and sys.stdin.isatty() and not args.files):
        _terminal.show_welcome(loaded_strategy.config)
        if args.help:
            sys.exit(0)
        try:
            task = input(
                _terminal.question_prompt("What do you want to build today?")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if not task:
            sys.exit(0)
        args.prompt = [task]

    # ── Build initial prompt ───────────────────────────────────────────
    file_context = _build_file_context(args.files or [])

    if args.prompt:
        initial_prompt = " ".join(args.prompt)
    elif not sys.stdin.isatty():
        initial_prompt = sys.stdin.read().strip()
    else:
        initial_prompt = ""

    if not initial_prompt and not file_context:
        sys.exit(0)

    force = "hard" if args.force_hard else "easy" if args.force_easy else None
    try:
        app = StrategyBuilder().build(loaded_strategy, force=force)
    except StrategyBuildError as exc:
        sys.exit(_terminal.format_error(exc))

    session_stats: list[RunStats] = []
    turn_n          = 0
    pending         = (file_context + initial_prompt).strip()

    # ── REPL loop ──────────────────────────────────────────────────────
    while True:
        if pending is not None:
            user_input = pending
            pending    = None
        else:
            try:
                print()
                user_input = input(_terminal.input_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not user_input:
                continue

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break

        turn_n      += 1
        task = Task(user_input)
        run = None
        with app.capture_stats(task_id=f"turn-{turn_n}") as stats_capture:
            classification_enabled = (
                force is None
                and loaded_strategy.config.method.type in ("difficulty", "cascade")
            )
            spinner = (
                _terminal.Spinner("Classifying task difficulty")
                if classification_enabled
                else None
            )
            if spinner is not None:
                spinner.start()

            def display_route(route, _attempt_number=1):
                if spinner is not None:
                    spinner.stop()
                classification = next(
                    (
                        event.outcome
                        for event in route.routing_events
                        if event.outcome in ("easy", "hard")
                    ),
                    None,
                )
                route_kind = (
                    classification
                    or (f"forced-{force}" if force is not None else None)
                    or route.phase
                    or "fixed"
                )
                tag = "(forced)" if force is not None else route.label
                if route.model is None:
                    raise RuntimeError("planned execute route has no model")
                _terminal.print_route(
                    route.model,
                    route_kind,
                    _terminal.transport_name(
                        loaded_strategy.config.generation.type
                    ),
                    tag,
                )

            try:
                if args.dry_run:
                    display_route(app.plan(task))
                else:
                    run = app.run_detailed(task, on_route=display_route)
            finally:
                if spinner is not None:
                    spinner.stop()

        turn_stats = stats_capture.stats
        if run is not None:
            turn_stats = turn_stats.with_run_result(run)
        session_stats.append(turn_stats)
        _terminal.print_turn_stats(
            turn_stats,
            turn_n,
            aggregate_stats(session_stats),
        )

        if run is None:
            continue
        if app.executor.captures_output and run.final_result.text:
            print(run.final_result.text)


if __name__ == "__main__":
    main()
