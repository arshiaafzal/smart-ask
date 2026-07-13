"""Private terminal presentation for the product CLI."""

from __future__ import annotations

import random
import re
import sys
import threading
import time

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


def show_welcome(config: StrategyConfig) -> None:
    profile_rows = [
        (
            f"  {_BLUE}>{_RESET} {_WHITE}{label:<7}:{_RESET} "
            f"{_CYAN}{target}{_RESET}"
        )
        for label, target in _configured_profiles(config)
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
        f"  {_GREEN}>{_RESET} {_WHITE}Targets:{_RESET}  {_CYAN}{len(config.target_ids)} configured{_RESET}",
        None,
        *profile_rows,
        None,
        f"  {_CYAN}{_BOLD}Usage{_RESET}",
        f'  {_GRAY}smart-ask "message"{_RESET}           start a conversation',
        f'  {_GRAY}smart-ask -f file.py "..."{_RESET}   include file context',
        *force_rows,
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


def _strip_ansi(value: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", value)


def _configured_profiles(config: StrategyConfig) -> tuple[tuple[str, str], ...]:
    method = config.method
    if isinstance(method, FixedMethodConfig):
        profile_names = (("Model", method.profile),)
    else:
        profile_names = (("Easy", method.easy), ("Hard", method.hard))
    return tuple(
        (label, f"{profile} -> {config.profiles[profile].target}")
        for label, profile in profile_names
    )


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
