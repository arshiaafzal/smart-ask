"""
Exact token and cost tracker for smart-ask benchmarks.

Records every API call by reading prompt_tokens and completion_tokens
directly from the response.usage object — no estimation, no approximation.

Usage
-----
    tracker = TokenTracker()

    response = client.chat.completions.create(...)
    tracker.record(
        model   = "anthropic/claude-haiku-4.5",
        role    = "classifier",
        usage   = response.usage,
        task_id = "HumanEval/7",
    )

    tracker.report()          # print breakdown table
    tracker.total_cost()      # float: exact total $
    tracker.by_model()        # dict: per-model aggregates
    tracker.by_role()         # dict: per-role aggregates
    tracker.export_json()     # full call log as JSON string

Unknown models raise by default so benchmark cost reports cannot silently
under-count. Product callers that still want token evidence can construct
TokenTracker(allow_unpriced=True); those calls retain usage with a null cost.
"""

import threading
import json
from dataclasses import dataclass, asdict
from typing import Optional

# ── Prices per input/output token (fetched from OpenRouter 2026-07-01) ────────
MODEL_PRICES: dict[str, dict[str, float]] = {
    "anthropic/claude-haiku-4.5":   {"input": 0.0000008,  "output": 0.000001},
    "google/gemini-2.5-flash-lite": {"input": 0.0000001,  "output": 0.0000004},
    "anthropic/claude-opus-4.8":    {"input": 0.000005,   "output": 0.000025},
}


# ── Single call record ────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    model:             str            # full OpenRouter model ID
    role:              str            # "classifier" | "generator" | "confidence" | "fixer"
    prompt_tokens:     int            # exact value from response.usage
    completion_tokens: int            # exact value from response.usage
    cost_usd:          Optional[float] # None when tokens are known but price is not
    task_id:           Optional[str]  # e.g. "HumanEval/7"


# ── Tracker ───────────────────────────────────────────────────────────────────

class TokenTracker:
    """
    Thread-safe, per-call exact token tracker.

    Each call to .record() ingests the raw usage object from the OpenAI
    response and computes cost from exact token counts — never estimated.
    Unknown catalog entries remain strict errors unless allow_unpriced is set.
    """

    def __init__(self, allow_unpriced: bool = False):
        """Create a tracker; unknown models remain errors unless explicitly allowed."""

        self.allow_unpriced = allow_unpriced
        self._calls: list[CallRecord] = []
        self._lock  = threading.Lock()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        model:   str,
        role:    str,
        usage,                      # openai CompletionUsage object, or None on API error
        task_id: str = None,
    ) -> Optional[float]:
        """
        Record one API call from its response.usage.
        Returns the exact cost in USD for this single call, or None when the
        model is unpriced and this tracker was created with allow_unpriced=True.
        If usage is None (API call failed), returns 0.0 and records nothing.
        """
        if usage is None:
            return 0.0

        prices = MODEL_PRICES.get(model)
        if prices is None and not self.allow_unpriced:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Add it to cost/tracker.py MODEL_PRICES."
            )

        prompt_tok     = usage.prompt_tokens
        completion_tok = usage.completion_tokens
        cost = (
            prompt_tok * prices["input"] + completion_tok * prices["output"]
            if prices is not None
            else None
        )

        rec = CallRecord(
            model             = model,
            role              = role,
            prompt_tokens     = prompt_tok,
            completion_tokens = completion_tok,
            cost_usd          = cost,
            task_id           = task_id,
        )
        with self._lock:
            self._calls.append(rec)

        return cost

    # ── Aggregations ──────────────────────────────────────────────────────────

    def _snapshot(self) -> list[CallRecord]:
        with self._lock:
            return list(self._calls)

    def total_cost(self) -> float:
        """Return the known-cost subtotal across all recorded calls."""

        return sum(
            (c.cost_usd for c in self._snapshot() if c.cost_usd is not None),
            0.0,
        )

    def has_unpriced_calls(self) -> bool:
        """Return whether any recorded call has usage but no catalog price."""

        return any(c.cost_usd is None for c in self._snapshot())

    def by_model(self) -> dict[str, dict]:
        """Aggregate exact token usage grouped by model ID."""
        totals: dict[str, dict] = {}
        for c in self._snapshot():
            if c.model not in totals:
                totals[c.model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "unpriced_calls": 0,
                }
            t = totals[c.model]
            t["calls"]             += 1
            t["prompt_tokens"]     += c.prompt_tokens
            t["completion_tokens"] += c.completion_tokens
            t["total_tokens"]      += c.prompt_tokens + c.completion_tokens
            if c.cost_usd is None:
                t["cost_usd"] = None
                t["unpriced_calls"] += 1
            elif t["cost_usd"] is not None:
                t["cost_usd"] += c.cost_usd
        return totals

    def by_role(self) -> dict[str, dict]:
        """Aggregate exact token usage grouped by role."""
        totals: dict[str, dict] = {}
        for c in self._snapshot():
            if c.role not in totals:
                totals[c.role] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "unpriced_calls": 0,
                }
            t = totals[c.role]
            t["calls"]             += 1
            t["prompt_tokens"]     += c.prompt_tokens
            t["completion_tokens"] += c.completion_tokens
            if c.cost_usd is None:
                t["cost_usd"] = None
                t["unpriced_calls"] += 1
            elif t["cost_usd"] is not None:
                t["cost_usd"] += c.cost_usd
        return totals

    def n_calls(self) -> int:
        return len(self._snapshot())

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self, title: str = "Token Usage Report") -> None:
        """
        Print a formatted exact-token breakdown table.

        Columns: model | calls | prompt tokens | completion tokens | cost ($)
        """
        by_m  = self.by_model()
        by_r  = self.by_role()
        total = self.total_cost()
        W     = 76

        print(f"\n  {'─'*W}")
        print(f"  {title}")
        print(f"  {'─'*W}")

        # ── Per-model table ────────────────────────────────────────────────
        print(f"\n  BY MODEL")
        print(f"  {'model':<34}  {'calls':>5}  {'in tok':>9}  {'out tok':>9}  {'cost':>12}")
        print(f"  {'·'*W}")
        total_in = total_out = 0
        for model, t in sorted(by_m.items()):
            short = model.split("/")[-1]
            cost = (
                f"${t['cost_usd']:>11.6f}"
                if t["cost_usd"] is not None
                else f"{'unknown':>12}"
            )
            print(
                f"  {short:<34}  {t['calls']:>5,}  "
                f"{t['prompt_tokens']:>9,}  {t['completion_tokens']:>9,}  "
                f"{cost}"
            )
            total_in  += t["prompt_tokens"]
            total_out += t["completion_tokens"]
        print(f"  {'·'*W}")
        total_cost = f"${total:>11.6f}"
        if self.has_unpriced_calls():
            total_cost = f"${total:.6f}+?"
        print(
            f"  {'TOTAL':<34}  {self.n_calls():>5,}  "
            f"{total_in:>9,}  {total_out:>9,}  {total_cost:>12}"
        )

        # ── Per-role table ─────────────────────────────────────────────────
        print(f"\n  BY ROLE")
        print(f"  {'role':<18}  {'calls':>5}  {'in tok':>9}  {'out tok':>9}  {'cost':>12}")
        print(f"  {'·'*W}")
        for role, t in sorted(by_r.items()):
            cost = (
                f"${t['cost_usd']:>11.6f}"
                if t["cost_usd"] is not None
                else f"{'unknown':>12}"
            )
            print(
                f"  {role:<18}  {t['calls']:>5,}  "
                f"{t['prompt_tokens']:>9,}  {t['completion_tokens']:>9,}  "
                f"{cost}"
            )

        print(f"  {'─'*W}\n")

    # ── Export ────────────────────────────────────────────────────────────────

    def export_json(self) -> str:
        """Full call log as a JSON string for saving to disk."""
        return json.dumps(
            {
                "total_cost_usd": (
                    None if self.has_unpriced_calls() else self.total_cost()
                ),
                "known_cost_usd": self.total_cost(),
                "unpriced_calls": sum(
                    call.cost_usd is None for call in self._snapshot()
                ),
                "n_calls":        self.n_calls(),
                "by_model":       self.by_model(),
                "by_role":        self.by_role(),
                "calls":          [asdict(c) for c in self._snapshot()],
            },
            indent=2,
        )
