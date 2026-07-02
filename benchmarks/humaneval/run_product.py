#!/usr/bin/env python3
"""
smart-ask product benchmark — HumanEval.

Tests the EXACT cascade pipeline the smart-ask CLI uses:
  Gate 1  Gemini classifier  → easy / hard
  Gate 2  Gemini self-check  → ESCALATE_NOW triggers Opus
  Opus    hard (G1) or escalated (G2) tasks

This benchmark contains ONLY evaluation logic.
Gates are imported from methods/. Cost tracking from cost/.

Usage
-----
    python run_product.py          # all 164 problems
    python run_product.py -n 20   # first N (quick smoke test)
    python run_product.py --report
"""

import os, sys, json, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from harness import run_tests
from methods.cascade import OR_BASE, cascade_solve

OR_KEY       = os.environ.get("OPENROUTER_API_KEY", "")
RESULTS_FILE = Path(__file__).parent / "results_product.json"
WORKERS      = 8

# HumanEval-specific system prompts (function completion, not scratch generation)
EASY_SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Return ONLY the complete function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)
HARD_SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given Python function correctly and completely. "
    "Return ONLY the function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)


# ── Dataset ───────────────────────────────────────────────────────────

def load_humaneval():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")
    return list(load_dataset("openai/openai_humaneval", split="test"))


# ── Per-problem runner ────────────────────────────────────────────────

def run_problem(client, prob, tracker: TokenTracker) -> dict:
    prompt  = f"Complete this Python function:\n\n{prob['prompt']}"
    task_id = prob["task_id"]

    result = cascade_solve(prompt, client, easy_system=EASY_SYSTEM, hard_system=HARD_SYSTEM)
    for model_id, role, usage in result["usages"]:
        tracker.record(model_id, role, usage, task_id)

    passed = run_tests(prob["prompt"], result["code"], prob["test"], prob["entry_point"])
    return {
        "task_id":    task_id,
        "difficulty": result["gate1"],
        "escalated":  result["escalated"],
        "model":      result["model"],
        "passed":     passed,
    }


# ── Benchmark runner ──────────────────────────────────────────────────

def run(problems, n=None):
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")
    if n:
        problems = problems[:n]

    client  = OpenAI(base_url=OR_BASE, api_key=OR_KEY)
    tracker = TokenTracker()
    total   = len(problems)
    results = []
    lock    = threading.Lock()
    done_n  = [0]

    print(f"\n  Running {total} HumanEval problems through smart-ask...\n")
    print(f"  {'#':>5}  {'G1':>6}  {'model':<12}  {'esc':>4}  {'pass':>5}  task_id")
    print(f"  {'─'*68}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_problem, client, p, tracker): p for p in problems}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                p = futs[fut]
                r = {"task_id": p["task_id"], "difficulty": "?",
                     "escalated": False, "model": "error", "passed": False}
            with lock:
                results.append(r)
                done_n[0] += 1
                esc_sym  = "↗" if r.get("escalated") else " "
                pass_sym = "✓" if r["passed"] else "✗"
                print(
                    f"  {done_n[0]:>5}  {r['difficulty']:>6}  {r.get('model','?'):<12}  "
                    f"{esc_sym:>4}  {pass_sym:>5}  {r['task_id']}"
                )

    results.sort(key=lambda x: x["task_id"])
    RESULTS_FILE.write_text(json.dumps({
        "n":         total,
        "results":   results,
        "token_log": json.loads(tracker.export_json()),
    }, indent=2))
    return results, tracker


# ── Report ────────────────────────────────────────────────────────────

def print_report(results, tracker):
    total     = len(results)
    passed    = sum(1 for r in results if r["passed"])
    g1_hard   = sum(1 for r in results if r["difficulty"] == "hard")
    g1_easy   = total - g1_hard
    escalated = sum(1 for r in results if r.get("escalated"))
    gemini_n  = sum(1 for r in results if r.get("model") == "gemini")
    gemini_ok = sum(1 for r in results if r.get("model") == "gemini" and r["passed"])
    esc_ok    = sum(1 for r in results if r.get("model") == "opus-esc" and r["passed"])
    g1h_ok    = sum(1 for r in results if r.get("model") == "opus-G1"  and r["passed"])
    W         = 68
    cost      = tracker.total_cost() if tracker else None

    print("\n" + "=" * W)
    print("  smart-ask Product Benchmark — HumanEval")
    print("=" * W)
    print(f"\n  Routing breakdown  ({total} problems)")
    print(f"    G1 easy   → Gemini              {g1_easy:>4}  ({g1_easy/total*100:.0f}%)")
    print(f"    G1 hard   → Opus direct         {g1_hard:>4}  ({g1_hard/total*100:.0f}%)")
    print(f"    Escalated → Opus via ESCALATE   {escalated:>4}  ({escalated/total*100:.0f}%)")
    print(f"\n  {'─'*66}")
    print(f"  {'pass@1':38}  {passed}/{total} ({passed/total*100:.1f}%)")
    if cost is not None:
        print(f"  {'total cost':38}  ${cost:.6f}")
    print(f"\n  Per-path accuracy")
    if gemini_n:
        print(f"    Gemini (no escalation)         {gemini_ok}/{gemini_n}  ({gemini_ok/gemini_n*100:.1f}%)")
    if escalated:
        print(f"    Opus after ESCALATE            {esc_ok}/{escalated}  ({esc_ok/escalated*100:.1f}%)")
    if g1_hard:
        print(f"    Opus via G1-hard               {g1h_ok}/{g1_hard}  ({g1h_ok/g1_hard*100:.1f}%)")
    if tracker:
        tracker.report("Token & cost breakdown")
    print("=" * W + "\n")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="smart-ask product benchmark on HumanEval")
    p.add_argument("-n",       type=int, default=None, help="Limit to first N problems")
    p.add_argument("--report", action="store_true",    help="Print saved results")
    args = p.parse_args()

    if args.report:
        if not RESULTS_FILE.exists():
            sys.exit("No saved results. Run without --report first.")
        data    = json.loads(RESULTS_FILE.read_text())
        tracker = TokenTracker()
        for call in data["token_log"].get("calls", []):
            class _U:
                prompt_tokens     = call["prompt_tokens"]
                completion_tokens = call["completion_tokens"]
            tracker.record(call["model"], call["role"], _U(), call.get("task_id"))
        print_report(data["results"], tracker)
        return

    print("  Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"  Loaded {len(problems)} problems\n")
    results, tracker = run(problems, n=args.n)
    print_report(results, tracker)


if __name__ == "__main__":
    main()
