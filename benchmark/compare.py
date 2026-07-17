#!/usr/bin/env python3
"""
Compare two benchmark runs (before vs after an upgrade).

Usage:
    python3 compare.py results/baseline/ results/truncated/
    python3 compare.py results/baseline/task_lru.jsonl results/truncated/task_lru.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import analyze helpers.
sys.path.insert(0, str(Path(__file__).parent))
from analyze import analyze_file


def compare(before_paths: list[Path], after_paths: list[Path]) -> None:
    def total_cost(paths):
        return sum(analyze_file(p).get("total_cost_usd", 0) for p in paths)

    def total_turns(paths, kind):
        return sum(
            analyze_file(p).get("routing", {}).get(f"{kind}_turns", 0)
            for p in paths
        )

    before_cost = total_cost(before_paths)
    after_cost = total_cost(after_paths)
    before_easy = total_turns(before_paths, "easy")
    before_hard = total_turns(before_paths, "hard")
    after_easy = total_turns(after_paths, "easy")
    after_hard = total_turns(after_paths, "hard")

    savings = before_cost - after_cost
    savings_pct = 100 * savings / before_cost if before_cost else 0

    print(f"\n{'═'*65}")
    print(f"  Benchmark comparison: BEFORE vs AFTER")
    print(f"{'─'*65}")
    print(f"  {'Metric':<28} {'Before':>12} {'After':>12} {'Delta':>12}")
    print(f"{'─'*65}")
    print(f"  {'Total cost ($)':<28} {before_cost:>12.6f} {after_cost:>12.6f} {savings_pct:>+11.1f}%")
    print(f"  {'Easy (Gemini) turns':<28} {before_easy:>12} {after_easy:>12} {after_easy-before_easy:>+12}")
    print(f"  {'Hard (Opus) turns':<28} {before_hard:>12} {after_hard:>12} {after_hard-before_hard:>+12}")
    print(f"{'═'*65}")
    if savings > 0:
        print(f"  Saved ${savings:.6f}  ({savings_pct:.1f}% cheaper)")
    elif savings < 0:
        print(f"  Cost increased by ${-savings:.6f}  ({-savings_pct:.1f}% more expensive)")
    else:
        print("  No cost change.")
    print()


def resolve_paths(arg: str) -> list[Path]:
    p = Path(arg)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if p.is_file():
        return [p]
    return []


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    before = resolve_paths(sys.argv[1])
    after = resolve_paths(sys.argv[2])

    if not before or not after:
        print("error: could not find metrics files", file=sys.stderr)
        sys.exit(1)

    compare(before, after)


if __name__ == "__main__":
    main()
