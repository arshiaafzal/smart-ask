"""Run a configured SmartAsk strategy as one continuous conversation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from . import _terminal
from .conversation import (
    Conversation,
    ConversationEvent,
    ConversationMessage,
    RunMetadata,
    RunMetricsStore,
    RunRecord,
)
from .metrics import DEFAULT_PRICE_CATALOG, TokenUsage, price_usage
from .strategy import (
    StrategyBuildError,
    StrategyBuilder,
    StrategyConfigError,
    load_strategy,
)


def _build_file_context(files: list[str]) -> str:
    parts = []
    for path in files:
        try:
            content = Path(path).read_text(encoding="utf-8")
            parts.append(f"--- file: {path} ---\n{content}\n---\n")
        except (OSError, UnicodeError) as exc:
            _terminal.warn(f"could not read {path}: {exc}")
    return "\n".join(parts) + "\n" if parts else ""


def _user_message(text: str) -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=({"type": "text", "text": text},),
    )


def _visible_text(event: ConversationEvent) -> str:
    if event.kind != "content_delta":
        return ""
    delta = event.data.get("delta")
    if not hasattr(delta, "get") or delta.get("type") != "text":
        return ""
    text = delta.get("text")
    return text if isinstance(text, str) else ""


def _assistant_message(events: list[ConversationEvent]) -> ConversationMessage | None:
    text = "".join(_visible_text(event) for event in events)
    if not text:
        return None
    return ConversationMessage(
        role="assistant",
        content=({"type": "text", "text": text},),
    )


def _request_total_tokens(request) -> int | None:
    if request.input_tokens is None or request.output_tokens is None:
        return None
    return request.input_tokens + request.output_tokens


def _request_cost(request) -> tuple[float | None, str]:
    if request.provider_cost_usd is not None:
        return request.provider_cost_usd, "billed"
    total = _request_total_tokens(request)
    usage = TokenUsage(
        prompt_tokens=request.input_tokens,
        completion_tokens=request.output_tokens,
        total_tokens=total,
        reasoning_tokens=request.reasoning_tokens,
        cached_input_tokens=request.cache_read_tokens,
        cache_write_input_tokens=request.cache_write_tokens,
    )
    model = request.actual_model or request.selected_model
    quote = price_usage(model, usage, DEFAULT_PRICE_CATALOG)
    return quote.cost_usd, "est."


def _quantity_label(known: int | float, missing: int, suffix: str) -> str:
    if isinstance(known, float):
        rendered = f"${known:.6f}"
    else:
        rendered = f"{known:,}"
    if missing:
        return f"{rendered}+? {suffix}" if known else f"unknown {suffix}"
    return f"{rendered} {suffix}"


def _print_run_record(
    record: RunRecord,
    *,
    turn_number: int,
    session: dict[str, Any],
) -> None:
    for decision in record.decisions:
        profile = (
            ""
            if decision.selected_profile_id is None
            else f" -> {decision.selected_profile_id}"
        )
        reason = "" if decision.reason_code is None else f" ({decision.reason_code})"
        print(
            f"  decision {decision.sequence}: "
            f"{decision.gate} = {decision.outcome}{profile}{reason}"
        )

    providers = {
        request.provider_request_id: request
        for request in record.provider_requests
    }
    print(f"  {'-' * 72}")
    for call in record.model_calls:
        request = next(
            (
                providers[request_id]
                for request_id in reversed(call.provider_request_ids)
                if request_id in providers
            ),
            None,
        )
        if request is None:
            print(
                f"  {call.call_id:<10} {call.role:<12} {call.profile_id:<14} "
                f"{call.status}"
            )
            continue
        model = request.actual_model or request.selected_model or call.target_id
        total_tokens = _request_total_tokens(request)
        tokens = "tokens unknown" if total_tokens is None else f"{total_tokens:,} tok"
        cost, cost_source = _request_cost(request)
        cost_text = (
            "cost unknown" if cost is None else f"${cost:.6f} {cost_source}"
        )
        print(
            f"  {call.call_id:<10} {call.role:<12} "
            f"{model.split('/')[-1]:<24} {tokens:<14} {cost_text}"
        )
    print(f"  {'-' * 72}")

    run_tokens = [_request_total_tokens(value) for value in record.provider_requests]
    run_costs = [_request_cost(value)[0] for value in record.provider_requests]
    known_run_tokens = sum(value for value in run_tokens if value is not None)
    known_run_cost = sum(value for value in run_costs if value is not None)
    run_token_label = _quantity_label(
        known_run_tokens,
        sum(value is None for value in run_tokens),
        "tok",
    )
    run_cost_label = _quantity_label(
        known_run_cost,
        sum(value is None for value in run_costs),
        "cost",
    )
    resources = session["resources"]["overall"]
    session_token_label = _quantity_label(
        resources["known_total_tokens"],
        resources["missing_total_token_requests"],
        "tok",
    )
    session_cost_label = _quantity_label(
        resources["known_cost_usd"],
        resources["missing_cost_requests"],
        "cost",
    )
    print(
        f"  Turn {turn_number}: {len(record.model_calls)} model calls, "
        f"{len(record.provider_requests)} provider requests; "
        f"{run_token_label}, {run_cost_label}"
    )
    print(
        f"  Session: {session['runs']} turns, {session['model_calls']} model calls; "
        f"{session_token_label}, {session_cost_label}"
    )
    print()


async def _run_session(
    *,
    engine,
    loaded_strategy,
    initial_input: str,
) -> None:
    session_id = f"cli-{uuid4().hex}"
    metrics = RunMetricsStore()
    history: list[ConversationMessage] = []
    pending: str | None = initial_input
    turn_number = 0

    try:
        while True:
            if pending is not None:
                user_input = pending
                pending = None
            else:
                try:
                    print()
                    user_input = input(_terminal.input_prompt()).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_input:
                    continue

            if user_input in ("/exit", "/quit"):
                break

            turn_number += 1
            turn_history = history + [_user_message(user_input)]
            conversation = Conversation(
                system=(),
                messages=tuple(turn_history),
            )
            metadata = RunMetadata(
                strategy_name=loaded_strategy.config.name,
                strategy_digest=loaded_strategy.digest,
                session_id=session_id,
                request_id=f"turn-{turn_number}",
            )
            spinner = (
                _terminal.Spinner("Routing and generating response")
                if sys.stdout.isatty()
                else None
            )
            if spinner is not None:
                spinner.start()

            handle = engine.start(conversation, metadata)
            events: list[ConversationEvent] = []
            output_started = False
            stream_error: Exception | None = None
            try:
                async for event in handle.events():
                    events.append(event)
                    text = _visible_text(event)
                    if not text:
                        continue
                    if spinner is not None:
                        spinner.stop()
                    if not output_started:
                        print()
                        output_started = True
                    print(text, end="", flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                stream_error = exc
            finally:
                if spinner is not None:
                    spinner.stop()
                if output_started:
                    print()

            record = await handle.result()
            session = metrics.record(record)["session"]
            if stream_error is not None:
                _terminal.warn(stream_error)
                _print_run_record(
                    record,
                    turn_number=turn_number,
                    session=session,
                )
                continue

            assistant = _assistant_message(events)
            if assistant is not None:
                history = turn_history + [assistant]
            else:
                _terminal.warn("model returned no visible text; turn was not added to history")
            _print_run_record(
                record,
                turn_number=turn_number,
                session=session,
            )
    finally:
        await engine.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a continuous conversation through a configured strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("-f", "--file", action="append", dest="files", metavar="FILE")
    forced = parser.add_mutually_exclusive_group()
    forced.add_argument("--force-easy", action="store_true")
    forced.add_argument("--force-hard", action="store_true")
    parser.add_argument(
        "--strategy",
        default="builtin:product",
        metavar="FILE|builtin:NAME",
        help="Strategy YAML or bundled strategy name (default: builtin:product)",
    )
    parser.add_argument("--validate-strategy", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)

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

    if args.help or (not args.prompt and sys.stdin.isatty() and not args.files):
        _terminal.show_welcome(loaded_strategy.config)
        if args.help:
            return
        try:
            initial_prompt = input(
                _terminal.question_prompt("What do you want to build today?")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    elif args.prompt:
        initial_prompt = " ".join(args.prompt)
    elif not sys.stdin.isatty():
        initial_prompt = sys.stdin.read().strip()
    else:
        initial_prompt = ""

    file_context = _build_file_context(args.files or [])
    initial_input = (file_context + initial_prompt).strip()
    if not initial_input:
        return

    force = "hard" if args.force_hard else "easy" if args.force_easy else None
    try:
        engine = StrategyBuilder().build_engine(loaded_strategy, force=force)
    except StrategyBuildError as exc:
        sys.exit(_terminal.format_error(exc))

    try:
        asyncio.run(_run_session(
            engine=engine,
            loaded_strategy=loaded_strategy,
            initial_input=initial_input,
        ))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
