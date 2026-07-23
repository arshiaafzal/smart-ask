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


# Direct API list prices per 1M tokens (USD), effective 2026-07-23.
PRICES = {
    # model fragment → (input_per_1M, output_per_1M)
    "claude-sonnet-4":     (3.0, 15.0),
    "claude-opus-4":       (5.0, 25.0),
    "gemini-flash-lite":  (0.075, 0.30),
    "gemini-2.5-flash":   (0.075, 0.30),
    "claude-opus":         (5.0, 25.0),
}


def model_price(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a model string."""
    m = model.lower()
    for fragment, price in PRICES.items():
        if fragment in m:
            return price
    return (1.0, 3.0)  # conservative default


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    inp_p, out_p = model_price(model)
    ordinary = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    return (
        ordinary * inp_p
        + cache_read_tokens * inp_p * 0.1
        + cache_write_tokens * inp_p * 1.25
        + output_tokens * out_p
    ) / 1_000_000


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
    model_cache_read: dict[str, int] = defaultdict(int)
    model_cache_write: dict[str, int] = defaultdict(int)
    model_requests: dict[str, int] = defaultdict(int)
    easy_turns = 0
    hard_turns = 0
    total_turns = 0

    for run in runs:
        provider_requests = run.get("provider_requests", [])
        decisions = run.get("decisions", [])
        calls = {
            call.get("call_id"): call
            for call in run.get("model_calls", [])
            if isinstance(call, dict)
        }

        # Count the profile that produced the visible response. This includes
        # replayed terminal handoffs and avoids double-counting their attempt
        # and acceptance decisions.
        final_call = calls.get(run.get("final_call_id"), {})
        final_profile = (
            final_call.get("profile_id")
            if isinstance(final_call, dict)
            else None
        )
        if final_profile in ("easy", "hard"):
            total_turns += 1
            if final_profile == "easy":
                easy_turns += 1
            else:
                hard_turns += 1
        else:
            # Compatibility with older records that lack final_call_id.
            for decision in reversed(decisions):
                gate = decision.get("gate", "")
                profile = decision.get("selected_profile_id", "")
                if gate in ("difficulty", "route-memory") and profile in (
                    "easy",
                    "hard",
                ):
                    total_turns += 1
                    if profile == "easy":
                        easy_turns += 1
                    else:
                        hard_turns += 1
                    break

        for req in provider_requests:
            model = req.get("selected_model") or req.get("actual_model") or ""
            inp = req.get("input_tokens") or 0
            out = req.get("output_tokens") or 0
            cache_read = req.get("cache_read_tokens") or 0
            cache_write = req.get("cache_write_tokens") or 0
            provider_cost = req.get("provider_cost_usd")
            if provider_cost is not None:
                cost = float(provider_cost)
            else:
                cost = compute_cost(
                    inp,
                    out,
                    model,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )
            total_cost += cost
            call = calls.get(req.get("call_id"), {})
            role = call.get("role") if isinstance(call, dict) else None
            model_key = _model_key(model, role)
            model_costs[model_key] += cost
            model_in_tokens[model_key] += inp
            model_out_tokens[model_key] += out
            model_cache_read[model_key] += cache_read
            model_cache_write[model_key] += cache_write
            model_requests[model_key] += 1

    return {
        "file": path.name,
        "runs": len(runs),
        "total_cost_usd": total_cost,
        "model_breakdown": {
            k: {
                "cost_usd": model_costs[k],
                "input_tokens": model_in_tokens[k],
                "output_tokens": model_out_tokens[k],
                "cache_read_tokens": model_cache_read[k],
                "cache_write_tokens": model_cache_write[k],
                "requests": model_requests[k],
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


def _model_key(model: str, role: str | None = None) -> str:
    m = model.lower()
    suffix = f" ({role})" if role else ""
    if "sonnet" in m:
        return "claude-sonnet" + suffix
    if "opus" in m:
        return "claude-opus" + suffix
    if "gemini" in m:
        return "gemini" + suffix
    if "claude" in m:
        return "claude" + suffix
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
        f"({routing['easy_pct']}% Sonnet generation)"
    )
    print(f"  Model breakdown:")
    for model, stats in summary["model_breakdown"].items():
        print(
            f"    {model:<30}  ${stats['cost_usd']:.6f}  "
            f"({stats['requests']} req; {stats['input_tokens']:,} in / "
            f"{stats['output_tokens']:,} out / "
            f"{stats['cache_read_tokens']:,} cache-read / "
            f"{stats['cache_write_tokens']:,} cache-write)"
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
