"""
Real-time cost tracking for live smart-ask CLI sessions.

Tracks exact costs for Gate 1 pre-flight API calls (usage object available).
Gate 2 runs via `hermes -q` subprocess — token count is not exposed.
Interactive Hermes session runs through PTY — tokens are not tracked.

Usage (in the CLI)
------------------
    from cost.realtime import SessionCost

    session = SessionCost()
    difficulty, usage = gate1_classify(prompt, api_key)
    session.record_gate1(CLASSIFIER_MODEL, usage)

    if escalated:
        session.record_gate2_ran()

    session.model = chosen_model
    session.print_summary()
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .tracker import MODEL_PRICES


@dataclass
class SessionCost:
    """
    Accumulates and displays cost for a single smart-ask session.

    Tracks:
      Gate 1 — exact (OpenAI usage object available)
      Gate 2 — presence only (hermes subprocess, no usage object)
      Session — not tracked (PTY passthrough to Hermes)
    """

    gate1_cost:       float = 0.0
    gate1_in_tokens:  int   = 0
    gate1_out_tokens: int   = 0
    gate2_ran:        bool  = False
    model:            str   = ""
    tag:              str   = ""
    started_at:       float = field(default_factory=time.monotonic)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_gate1(self, model: str, usage) -> None:
        """Record exact Gate 1 cost from an OpenAI CompletionUsage object."""
        if usage is None:
            return
        prices = MODEL_PRICES.get(model, {})
        self.gate1_in_tokens  = usage.prompt_tokens
        self.gate1_out_tokens = usage.completion_tokens
        self.gate1_cost = (
            usage.prompt_tokens     * prices.get("input",  0) +
            usage.completion_tokens * prices.get("output", 0)
        )

    def record_gate2_ran(self) -> None:
        """Note that Gate 2 preflight ran (cost not available via hermes subprocess)."""
        self.gate2_ran = True

    # ── Display ───────────────────────────────────────────────────────────────

    def summary_lines(self) -> list:
        """Return formatted summary lines for display in the CLI."""
        lines = []
        if self.gate1_cost > 0:
            lines.append(
                f"  Gate 1  {self.gate1_in_tokens:,} in + {self.gate1_out_tokens:,} out"
                f"  →  ${self.gate1_cost:.6f}"
            )
        if self.gate2_ran:
            lines.append(
                "  Gate 2  hermes -q preflight ran  (tokens not tracked)"
            )
        if not lines:
            lines.append("  (no pre-flight cost data)")
        return lines

    def print_summary(self, color_on: bool = True) -> None:
        """
        Print a compact cost block to stdout.

        color_on=False strips ANSI for plain output.
        """
        GY = "\033[38;5;245m" if color_on else ""
        YL = "\033[38;5;226m" if color_on else ""
        R  = "\033[0m"        if color_on else ""
        DM = "\033[2m"        if color_on else ""

        print(f"\n  {DM}─── pre-flight cost ───{R}")
        for line in self.summary_lines():
            print(f"{GY}{line}{R}")
        if self.gate1_cost > 0:
            print(f"  {YL}total pre-flight  ${self.gate1_cost:.6f}{R}")
        print(f"  {DM}interactive session cost tracked by Hermes{R}\n")
