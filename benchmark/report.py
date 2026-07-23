#!/usr/bin/env python3
"""
Per-task cost/routing/pass report comparing Opus-only vs Sonnet/Opus routing.

Usage:
    python3 report.py results/opus-baseline/ results/smart-routing/
    python3 report.py results/opus-baseline/ results/smart-routing/ --test-dir results/smart-routing/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import analyze_file


# Canonical task order (matches run_bench.sh).
TASK_ORDER = [
    "astropy_separability",
    "django_file_upload",
    "matplotlib_version_info",
    "seaborn_hue_order",
    "flask_blueprint_dots",
    "requests_redirect_copy",
    "pytest_assertion_rewrite",
    "sklearn_ridge_cv",
    "sphinx_inherited_members",
    "sympy_ccode_sinc",
    "lru",
]


def _load_results(run_dir: Path) -> dict[str, dict]:
    """Return {task_name: analyze_file_result} for all *.jsonl in a directory."""
    results: dict[str, dict] = {}
    paths = list(run_dir.glob("*.jsonl")) + list(run_dir.glob("*/metrics.jsonl"))
    for p in paths:
        task = p.parent.name if p.name == "metrics.jsonl" else p.stem
        results[task] = analyze_file(p)
    return results


def _pass_status(run_dir: Path, task: str) -> str:
    """Check the *.test.txt file in the run directory for pass/fail."""
    result_json = run_dir / task / "result.json"
    if result_json.exists():
        try:
            return "PASS" if json.loads(
                result_json.read_text(encoding="utf-8")
            ).get("passed") else "FAIL"
        except (json.JSONDecodeError, OSError):
            return "?"
    test_file = run_dir / f"{task}.test.txt"
    if not test_file.exists():
        return "?"
    content = test_file.read_text(errors="replace")
    # pytest: "N passed" with no "failed" → PASS
    # unittest: "OK" at the end → PASS
    lower = content.lower()
    if "failed" in lower or "error" in lower:
        return "FAIL"
    if "passed" in lower or lower.strip().endswith("ok"):
        return "PASS"
    return "?"


def report(opus_dir: Path, smart_dir: Path) -> None:
    opus_results = _load_results(opus_dir)
    smart_results = _load_results(smart_dir)

    col_task   = 30
    col_cost   = 9
    col_easy   = 6
    col_hard   = 6
    col_save   = 9
    col_pass   = 12  # "PASS / PASS"

    header = (
        f"{'Task':<{col_task}}  "
        f"{'Opus$':>{col_cost}}  "
        f"{'Smart$':>{col_cost}}  "
        f"{'Save%':>{col_save}}  "
        f"{'Easy':>{col_easy}}  "
        f"{'Hard':>{col_hard}}  "
        f"{'Opus/Smart':>{col_pass}}"
    )
    sep = "─" * len(header)

    print()
    print(f"  {'Benchmark Report':^{len(header)}}")
    print(f"  {'Opus dir: ' + str(opus_dir)}")
    print(f"  {'Smart dir: ' + str(smart_dir)}")
    print()
    print("  " + sep)
    print("  " + header)
    print("  " + sep)

    total_opus       = 0.0
    total_smart      = 0.0
    total_easy       = 0
    total_hard       = 0
    smart_pass       = 0
    smart_fail       = 0
    opus_pass        = 0
    opus_fail        = 0
    comparable_opus  = 0.0
    comparable_smart = 0.0
    comparable_count = 0

    tasks = [task for task in TASK_ORDER if task in opus_results or task in smart_results]
    tasks.extend(sorted((set(opus_results) | set(smart_results)) - set(tasks)))
    for task in tasks:
        opus_s  = opus_results.get(task, {})
        smart_s = smart_results.get(task, {})

        opus_cost  = opus_s.get("total_cost_usd", 0.0)
        smart_cost = smart_s.get("total_cost_usd", 0.0)
        easy = smart_s.get("routing", {}).get("easy_turns", 0)
        hard = smart_s.get("routing", {}).get("hard_turns", 0)

        opus_valid = opus_cost > 0  # False when credit exhausted
        if opus_valid and smart_cost > 0:
            save_pct = 100.0 * (opus_cost - smart_cost) / opus_cost
            save_str = f"{save_pct:>{col_save - 1}.1f}%"
            comparable_opus  += opus_cost
            comparable_smart += smart_cost
            comparable_count += 1
        elif not opus_valid:
            save_str = f"{'n/a (credits)':>{col_save}}"
        else:
            save_str = f"{'n/a':>{col_save}}"

        smart_status = _pass_status(smart_dir, task)
        opus_status  = _pass_status(opus_dir,  task) if opus_valid else "n/a"

        if smart_status == "PASS":
            smart_pass += 1
        elif smart_status == "FAIL":
            smart_fail += 1
        if opus_status == "PASS":
            opus_pass += 1
        elif opus_status == "FAIL":
            opus_fail += 1

        total_opus  += opus_cost
        total_smart += smart_cost
        total_easy  += easy
        total_hard  += hard

        pass_col = f"{opus_status} / {smart_status}"
        opus_cost_str = f"${opus_cost:>{col_cost - 1}.4f}" if opus_valid else f"{'(exhausted)':>{col_cost}}"
        row = (
            f"{task:<{col_task}}  "
            f"{opus_cost_str}  "
            f"${smart_cost:>{col_cost - 1}.4f}  "
            f"{save_str}  "
            f"{easy:>{col_easy}}  "
            f"{hard:>{col_hard}}  "
            f"{pass_col:>{col_pass}}"
        )
        print("  " + row)

    print("  " + sep)

    comp_save_pct = (
        100.0 * (comparable_opus - comparable_smart) / comparable_opus
        if comparable_opus else 0.0
    )
    total_row = (
        f"{'TOTAL (' + str(len(tasks)) + ' tasks)':<{col_task}}  "
        f"{'($' + f'{total_opus:.2f}' + ')':>{col_cost}}  "
        f"${total_smart:>{col_cost - 1}.4f}  "
        f"{'(' + f'{comp_save_pct:.1f}' + '%)':>{col_save}}  "
        f"{total_easy:>{col_easy}}  "
        f"{total_hard:>{col_hard}}  "
        f"{opus_pass}/{opus_pass + opus_fail}  /  {smart_pass}/{smart_pass + smart_fail}"
    )
    print("  " + total_row)
    print("  " + sep)
    if comparable_count < len(tasks):
        skipped = len(tasks) - comparable_count
        print(f"  Note: {skipped} task(s) skipped from Save% (credit exhaustion during opus run).")
        print(f"        Save% shown is for {comparable_count} tasks with valid opus data.")
    print()


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    opus_dir  = Path(sys.argv[1])
    smart_dir = Path(sys.argv[2])

    if not opus_dir.is_dir() or not smart_dir.is_dir():
        print("error: both arguments must be directories", file=sys.stderr)
        sys.exit(1)

    report(opus_dir, smart_dir)


if __name__ == "__main__":
    main()
