#!/usr/bin/env python3
"""
Analyze smart-ask metrics JSONL files to report cost and routing breakdown.

Usage:
    python3 analyze.py metrics.jsonl [metrics2.jsonl ...]
    python3 analyze.py results/baseline/   # all *.jsonl in directory
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


# Price per 1M tokens (USD) — OpenRouter pricing.
PRICES = {
    # model fragment → (input_per_1M, output_per_1M)
    "gemini-flash-lite":  (0.075, 0.30),
    "gemini-2.5-flash":   (0.075, 0.30),
    "claude-opus-4":      (15.0, 75.0),
    "claude-opus":        (15.0, 75.0),
}


def model_price(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a model string."""
    m = model.lower()
    for fragment, price in PRICES.items():
        if fragment in m:
            return price
    return (1.0, 3.0)  # conservative default


def compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    inp_p, out_p = model_price(model)
    return (input_tokens * inp_p + output_tokens * out_p) / 1_000_000


def analyze_file(path: Path) -> dict:
    """Parse one metrics JSONL file and return summary statistics."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Unwrap nested format: {"run": {...}, "session": {...}} → inner run dict
    unwrapped = []
    for r in records:
        if "run" in r and isinstance(r["run"], dict):
            unwrapped.append(r["run"])
        else:
            unwrapped.append(r)
    records = unwrapped

    runs = [r for r in records if r.get("schema", "").startswith("smart-ask.run")]
    if not runs:
        return {"file": str(path), "runs": 0, "error": "no run records found"}

    total_cost = 0.0
    model_costs: dict[str, float] = defaultdict(float)
    model_in_tokens: dict[str, int] = defaultdict(int)
    model_out_tokens: dict[str, int] = defaultdict(int)
    easy_turns = 0
    hard_turns = 0
    total_turns = 0

    for run in runs:
        provider_requests = run.get("provider_requests", [])
        decisions = run.get("decisions", [])

        # Count routing decisions.
        for decision in decisions:
            gate = decision.get("gate", "")
            profile = decision.get("selected_profile_id", "")
            if gate in ("difficulty", "route-memory"):
                total_turns += 1
                if profile == "easy":
                    easy_turns += 1
                elif profile == "hard":
                    hard_turns += 1

        for req in provider_requests:
            model = req.get("selected_model") or req.get("actual_model") or ""
            inp = req.get("input_tokens") or 0
            out = req.get("output_tokens") or 0
            provider_cost = req.get("provider_cost_usd")
            if provider_cost is not None:
                cost = float(provider_cost)
            else:
                cost = compute_cost(inp, out, model)
            total_cost += cost
            model_key = _model_key(model)
            model_costs[model_key] += cost
            model_in_tokens[model_key] += inp
            model_out_tokens[model_key] += out

    return {
        "file": path.name,
        "runs": len(runs),
        "total_cost_usd": total_cost,
        "model_breakdown": {
            k: {
                "cost_usd": model_costs[k],
                "input_tokens": model_in_tokens[k],
                "output_tokens": model_out_tokens[k],
            }
            for k in sorted(model_costs)
        },
        "routing": {
            "total_turns": total_turns,
            "easy_turns": easy_turns,
            "hard_turns": hard_turns,
            "easy_pct": (
                round(100 * easy_turns / total_turns, 1) if total_turns else 0
            ),
        },
    }


def _model_key(model: str) -> str:
    m = model.lower()
    if "gemini" in m:
        return "gemini (easy)"
    if "opus" in m or "claude" in m:
        return "claude-opus (hard)"
    if "classifier" in m:
        return "classifier"
    return model or "unknown"


def print_summary(summary: dict) -> None:
    f = summary["file"]
    runs = summary["runs"]
    if "error" in summary:
        print(f"  {f}: {summary['error']}")
        return

    cost = summary["total_cost_usd"]
    routing = summary["routing"]
    print(f"\n{'─'*60}")
    print(f"  File : {f}  ({runs} run{'s' if runs != 1 else ''})")
    print(f"  Cost : ${cost:.6f}")
    print(
        f"  Route: {routing['easy_turns']}✓ easy  "
        f"{routing['hard_turns']}✗ hard  "
        f"({routing['easy_pct']}% Gemini)"
    )
    print(f"  Model breakdown:")
    for model, stats in summary["model_breakdown"].items():
        print(
            f"    {model:<30}  ${stats['cost_usd']:.6f}  "
            f"({stats['input_tokens']:,} in / {stats['output_tokens']:,} out tokens)"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"warning: {arg} not found", file=sys.stderr)

    if not paths:
        print("No metrics files found.")
        sys.exit(1)

    summaries = [analyze_file(p) for p in paths]
    for s in summaries:
        print_summary(s)

    if len(summaries) > 1:
        total = sum(s.get("total_cost_usd", 0) for s in summaries)
        print(f"\n{'═'*60}")
        print(f"  Total across {len(summaries)} files: ${total:.6f}")


if __name__ == "__main__":
    main()
