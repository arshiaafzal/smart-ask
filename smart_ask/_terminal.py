"""Private terminal presentation for the product CLI."""

from __future__ import annotations

import random
import re
import sys
import threading
import time

from .metrics import RunStats, StatsSummary
from .strategy.schema import FixedMethodConfig, StrategyConfig


_RESET = "\033[0m"
_BOLD = "\033[1m"
_ORANGE = "\033[38;5;208m"
_GOLD = "\033[38;5;220m"
_BLUE = "\033[38;5;39m"
_CYAN = "\033[38;5;87m"
_YELLOW = "\033[38;5;226m"
_GREEN = "\033[38;5;82m"
_PURPLE = "\033[38;5;135m"
_WHITE = "\033[97m"
_GRAY = "\033[38;5;245m"
_RED = "\033[38;5;196m"

_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_COLORS = (
    _ORANGE,
    _GOLD,
    _YELLOW,
    _GREEN,
    _CYAN,
    _BLUE,
    _PURPLE,
    _WHITE,
    _ORANGE,
    _GOLD,
)
_LOGO = (
    "███████╗███╗   ███╗ █████╗ ██████╗ ████████╗      █████╗ ███████╗██╗  ██╗",
    "██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝     ██╔══██╗██╔════╝██║ ██╔╝",
    "███████╗██╔████╔██║███████║██████╔╝   ██║        ███████║███████╗█████╔╝ ",
    "╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║        ██╔══██║╚════██║██╔═██╗ ",
    "███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║        ██║  ██║███████║██║  ██╗",
    "╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝        ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝",
)


def format_error(message: object) -> str:
    return f"{_RED}ERROR:{_RESET} {message}"


def warn(message: object) -> None:
    print(f"  {_RED}Warning:{_RESET} {message}", file=sys.stderr)


def question_prompt(message: str) -> str:
    return f"  {_GREEN}{message}{_RESET}  "


def input_prompt() -> str:
    return f"  {_GREEN}>{_RESET} "


def transport_name(transport_type: str) -> str:
    return {
        "hermes": "Hermes",
        "openrouter": "OpenRouter",
    }.get(transport_type, transport_type)


def show_welcome(config: StrategyConfig) -> None:
    transport = transport_name(config.generation.type)
    profile_rows = [
        (
            f"  {_BLUE}>{_RESET} {_WHITE}{label:<7}:{_RESET} "
            f"{_CYAN}{model.split('/')[-1]}{_RESET}"
        )
        for label, model in _configured_profiles(config)
    ]
    force_rows = []
    if not isinstance(config.method, FixedMethodConfig):
        force_rows = [
            f'  {_GRAY}smart-ask --force-hard "..."{_RESET}  use configured hard profile',
            f'  {_GRAY}smart-ask --force-easy "..."{_RESET}  use configured easy profile',
        ]
    _teaser()
    _box([
        f"   {_WHITE}{_BOLD}SMART ASK  --  Model Router{_RESET}",
        f"   {_GRAY}{config.name}{_RESET}",
        None,
        f"  {_GREEN}>{_RESET} {_WHITE}Method:{_RESET}   {_CYAN}{config.method.type}{_RESET}",
        f"  {_GREEN}>{_RESET} {_WHITE}Transport:{_RESET} {_CYAN}{transport}{_RESET}",
        None,
        *profile_rows,
        None,
        f"  {_CYAN}{_BOLD}Usage{_RESET}",
        f'  {_GRAY}smart-ask "task"{_RESET}              route and execute',
        f'  {_GRAY}smart-ask -f file.py "..."{_RESET}   include file context',
        *force_rows,
        f'  {_GRAY}smart-ask --dry-run "..."{_RESET}     classify; skip generation',
        f"  {_GRAY}Ctrl-D or /exit{_RESET}                 end session",
    ], inner_width=68)
    print()


class Spinner:
    """One idempotently stoppable terminal activity indicator."""

    __slots__ = ("_label", "_stop", "_thread")

    def __init__(self, label: str):
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("spinner is already started")
        self._thread = threading.Thread(
            target=_animate_spinner,
            args=(self._stop, self._label),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join()
        self._thread = None


def print_route(
    model_id: str,
    route_kind: str,
    transport: str,
    tag: str = "",
) -> None:
    model_name = model_id.split("/")[-1]
    easy_route = route_kind in ("easy", "forced-easy", "fixed")
    accent = _CYAN if easy_route else _ORANGE
    route_color = _GRAY if easy_route else _YELLOW
    print(
        f"\n  {accent}{_BOLD}▸{_RESET}  {accent}{_BOLD}{model_name}{_RESET}  "
        f"{route_color}[{route_kind}]{_RESET}  {_GRAY}{tag}{_RESET}"
    )
    print(f"  {_PURPLE}↳  {transport}{_RESET}\n")


def print_turn_stats(
    stats: RunStats,
    turn_number: int,
    session_stats: StatsSummary,
) -> None:
    if not stats.calls:
        return
    width = 64
    print(f"  {_GRAY}{'─' * width}{_RESET}")
    for call in stats.calls:
        if call.provider_cost_usd is not None:
            cost = f"${call.provider_cost_usd:.6f} billed"
        elif call.price_quote.cost_usd is not None:
            cost = f"${call.price_quote.cost_usd:.6f} est."
        else:
            cost = "cost unknown"
        tokens = (
            f"{call.usage.total_tokens:,} tok"
            if call.usage.total_tokens is not None
            else "tokens unknown"
        )
        model = call.actual_model or call.requested_model
        print(
            f"  {_GRAY}{model.split('/')[-1]:<28}  {call.channel:<12}  "
            f"{_YELLOW}{cost}{_RESET}  {_GRAY}{tokens}{_RESET}"
        )
    print(f"  {_GRAY}{'─' * width}{_RESET}")
    print(
        f"  {_WHITE}{_BOLD}Turn {turn_number:<4}{_RESET}  "
        f"{_YELLOW}{_cost_label(stats)}{_RESET}, {_GRAY}{_token_label(stats)}{_RESET}"
        f"   {_GRAY}│{_RESET}  Session  "
        f"{_YELLOW}{_cost_label(session_stats)}{_RESET}, "
        f"{_GRAY}{_token_label(session_stats)}{_RESET}"
    )
    print()


def _strip_ansi(value: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", value)


def _configured_profiles(config: StrategyConfig) -> tuple[tuple[str, str], ...]:
    method = config.method
    if isinstance(method, FixedMethodConfig):
        return (("Model", method.model.model),)
    return (("Easy", method.easy.model), ("Hard", method.hard.model))


def _box(rows: list[str | None], inner_width: int) -> None:
    print(f"  {_WHITE}{_BOLD}╔{'═' * inner_width}╗{_RESET}")
    for row in rows:
        if row is None:
            print(f"  {_WHITE}╠{'═' * inner_width}╣{_RESET}")
            continue
        padding = max(0, inner_width - len(_strip_ansi(row)))
        print(f"  {_WHITE}║{_RESET}{row}{' ' * padding}{_WHITE}║{_RESET}")
    print(f"  {_WHITE}╚{'═' * inner_width}╝{_RESET}")


def _teaser() -> None:
    logo_height = len(_LOGO)
    logo_width = max(len(line) for line in _LOGO)
    margin = 2
    width = logo_width + 2 * margin
    rain_above = 2
    height = rain_above + logo_height + 1
    rain_height = height + 2
    dark_colors = (
        "\033[38;5;22m",
        "\033[38;5;22m",
        "\033[38;5;28m",
        "\033[38;5;28m",
        "\033[38;5;34m",
        "\033[38;5;40m",
    )

    def rain_character() -> str:
        if random.random() < 0.28:
            return random.choice(dark_colors) + "$" + _RESET
        return " "

    logo_chars: list[list[str | None]] = [
        [None] * width for _ in range(height)
    ]
    for row, line in enumerate(_LOGO):
        for column, character in enumerate(line):
            if character != " ":
                logo_chars[rain_above + row][margin + column] = character

    rain = {
        column: [rain_character() for _ in range(rain_height)]
        for column in range(width)
    }

    def advance() -> None:
        for buffer in rain.values():
            buffer.pop()
            buffer.insert(0, rain_character())

    logo_color = "\033[1;38;5;82m"
    flash_color = "\033[1;38;5;118m"
    rain_frames = 8
    reveal_frames = logo_height
    total_frames = rain_frames + reveal_frames
    total_lines = height + 1
    animate = sys.stdout.isatty()
    if animate:
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
    try:
        for frame in range(total_frames if animate else 1):
            if animate and frame > 0:
                sys.stdout.write(f"\033[{total_lines}F")
            reveal_step = frame - rain_frames
            lines = []
            for row in range(height):
                row_chars = []
                logo_row = row - rain_above
                revealed = reveal_step >= 0 and 0 <= logo_row <= reveal_step
                flashing = logo_row == reveal_step
                for column in range(width):
                    logo_character = logo_chars[row][column]
                    if logo_character is not None and revealed:
                        color = flash_color if flashing else logo_color
                        row_chars.append(color + logo_character + _RESET)
                    else:
                        row_chars.append(rain[column][row])
                lines.append("\033[2K" + "".join(row_chars))
            lines.append("\033[2K")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            advance()
            if animate:
                time.sleep(0.05 if reveal_step < 0 else 0.10)
        if animate:
            time.sleep(0.8)
    finally:
        if animate:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


def _animate_spinner(stop: threading.Event, label: str) -> None:
    frame_number = 0
    while not stop.is_set():
        color = _COLORS[frame_number % len(_COLORS)]
        frame = _FRAMES[frame_number % len(_FRAMES)]
        dot_count = (frame_number // 5) % 4
        dots = "·" * dot_count + " " * (3 - dot_count)
        progress = (frame_number // 3) % 13
        bar = (
            f"{_GREEN}{'█' * progress}"
            f"{_GRAY}{'░' * (12 - progress)}{_RESET}"
        )
        sys.stdout.write(
            f"\r  {color}{_BOLD}{frame}{_RESET}  {_WHITE}{label}{_RESET}"
            f"{_GRAY}{dots}{_RESET}  {bar}  "
        )
        sys.stdout.flush()
        frame_number += 1
        time.sleep(0.07)
    sys.stdout.write("\r" + " " * 72 + "\r")
    sys.stdout.flush()


def _cost_label(stats: RunStats | StatsSummary) -> str:
    if stats.provider_cost_complete:
        return f"${stats.known_provider_cost_usd:.6f} billed"
    known = stats.known_cost_usd
    if stats.cost_complete:
        return f"${known:.6f} est."
    if stats.known_provider_cost_usd:
        return f"${stats.known_provider_cost_usd:.6f} billed + unknown"
    return "unknown" if known == 0 else f"${known:.6f} est. + unknown"


def _token_label(stats: RunStats | StatsSummary) -> str:
    if stats.total_usage_complete:
        return f"{stats.known_total_tokens:,} tok"
    if stats.known_total_tokens:
        return f"{stats.known_total_tokens:,}+? tok"
    return "tokens unknown"
