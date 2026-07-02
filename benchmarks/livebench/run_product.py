#!/usr/bin/env python3
"""
smart-ask product benchmark — LiveBench (coding category).

128 problems from real LeetCode/AtCoder contests (June 2024 release).
No contamination — questions postdate most model training cuts.

Two task types
--------------
  LCB_generation   (78)  Generate a complete Python Solution class.
  coding_completion (50)  Complete a partial solution snippet.

Two test harnesses
------------------
  functional  Solution().method(*args)  compared to expected return value
  stdin       code reads sys.stdin      stdout compared to expected

This benchmark contains ONLY evaluation logic.
All gate and model code is imported from methods/.
All cost tracking is imported from cost/.

Usage
-----
    python run_product.py          # all 128 problems
    python run_product.py -n 20   # first N
    python run_product.py --report
"""

import os, sys, json, ast, re, subprocess, tempfile, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from methods.cascade import (
    OR_BASE, CLASSIFIER_MODEL, EASY_MODEL, HARD_MODEL,
    gate1_classify, call_easy, call_hard,
)

OR_KEY       = os.environ.get("OPENROUTER_API_KEY", "")
RESULTS_FILE = Path(__file__).parent / "results_product.json"
WORKERS      = 6


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_livebench():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")
    ds = load_dataset("livebench/coding", split="test")
    problems = []
    for ex in ds:
        orig = ex["original_json"]
        if isinstance(orig, str):
            orig = json.loads(orig)
        tcs = ex["public_test_cases"]
        if isinstance(tcs, str):
            tcs = json.loads(tcs)
        problems.append({
            "question_id":  ex["question_id"],
            "task":         ex["task"],
            "title":        ex["question_title"],
            "prompt":       ex["turns"][0],
            "starter_code": orig.get("starter_code", ""),
            "partial":      ex.get("partial_solution", ""),
            "test_cases":   tcs,
            "difficulty":   orig.get("difficulty", "?"),
        })
    return problems


# ── Execution harnesses ────────────────────────────────────────────────────────

_STDLIB = (
    "from typing import List, Tuple, Dict, Optional, Set, Any, Union, Counter\n"
    "import sys, math, re, collections, itertools, functools, heapq, bisect, ast\n"
    "from collections import defaultdict, Counter, deque\n"
)

def _parse_val(s: str):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return s

def _normalize(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_normalize(x) for x in v) + "]"
    return str(v)


def run_functional(code: str, starter_code: str, tc: dict, timeout: int = 10) -> bool:
    """Execute a LeetCode-style test case via Solution().method(*args)."""
    m = re.search(r"def (\w+)\(self", starter_code)
    if not m:
        return False
    fn_name = m.group(1)
    raw_inp = tc["input"].strip()
    lines   = [l.strip() for l in raw_inp.splitlines() if l.strip()]
    args    = [_parse_val(l) for l in lines]
    call    = (f"_s.{fn_name}({repr(args[0])})" if len(args) == 1
               else f"_s.{fn_name}(*{repr(tuple(args))})")
    expected = _parse_val(tc["output"].strip())
    script = (
        _STDLIB + "\n" + code
        + f"\n\n_s = Solution()\n_r = {call}\nprint(repr(_r))\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script); fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname], capture_output=True, text=True, timeout=timeout
        )
        if res.returncode != 0:
            return False
        got = _parse_val(res.stdout.strip().strip("'\""))
        return _normalize(got) == _normalize(expected)
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def run_stdin(code: str, tc: dict, timeout: int = 10) -> bool:
    """Execute an AtCoder-style test case (stdin → stdout comparison)."""
    script = _STDLIB + "\n" + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script); fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname],
            input=tc["input"], capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            return False
        return res.stdout.strip() == tc["output"].strip()
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def run_tests(code: str, starter_code: str, test_cases: list, timeout: int = 10) -> tuple:
    """Run all public test cases. Returns (passed, total)."""
    passed = 0
    for tc in test_cases:
        ttype = tc.get("testtype", "functional")
        try:
            ok = (run_functional(code, starter_code, tc, timeout)
                  if ttype == "functional" else run_stdin(code, tc, timeout))
        except Exception:
            ok = False
        if ok:
            passed += 1
    return passed, len(test_cases)


# ── Per-problem runner ─────────────────────────────────────────────────────────

def run_problem(client, prob: dict, tracker: TokenTracker) -> dict:
    qid     = prob["question_id"][:16]
    prompt  = prob["prompt"]
    partial = prob["partial"]
    starter = prob["starter_code"]
    tcs     = prob["test_cases"]

    rec = {
        "question_id": prob["question_id"],
        "title":       prob["title"],
        "task":        prob["task"],
        "difficulty":  prob["difficulty"],
        "gate1":       None,
        "escalated":   False,
        "model":       None,
        "passed":      0,
        "total":       len(tcs),
        "pass_all":    False,
    }

    # Gate 1 — classify
    difficulty, g1_usage = gate1_classify(prompt, client)
    tracker.record(CLASSIFIER_MODEL, "classifier", g1_usage, qid)
    rec["gate1"] = difficulty

    if difficulty == "hard":
        code, usage = call_hard(prompt, client)
        tracker.record(HARD_MODEL, "writer", usage, qid)
        rec["model"] = "opus-G1"
    else:
        code, usage, escalated = call_easy(prompt, client)
        tracker.record(EASY_MODEL, "generator", usage, qid)
        rec["escalated"] = escalated
        if escalated:
            code, usage = call_hard(prompt, client, escalated=True)
            tracker.record(HARD_MODEL, "fixer", usage, qid)
            rec["model"] = "opus-esc"
        else:
            rec["model"] = "gemini"

    # For coding_completion: stitch partial + completion.
    # If model re-output the full class, use it directly.
    if prob["task"] == "coding_completion" and partial:
        full_code = code if "class Solution" in code else partial + "\n" + code
    else:
        full_code = code

    passed, total   = run_tests(full_code, starter, tcs)
    rec["passed"]   = passed
    rec["total"]    = total
    rec["pass_all"] = passed == total and total > 0
    return rec


# ── Benchmark runner ───────────────────────────────────────────────────────────

def run(problems: list, n: int = None):
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

    print(f"\n  Running {total} LiveBench coding problems through smart-ask...\n")
    print(f"  {'#':>5}  {'G1':>6}  {'model':<12}  {'esc':>4}  {'pass':>8}  title")
    print(f"  {'─'*80}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_problem, client, p, tracker): p for p in problems}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                r = fut.result()
            except Exception:
                r = {
                    "question_id": p["question_id"], "title": p["title"],
                    "task": p["task"], "difficulty": p["difficulty"],
                    "gate1": "?", "escalated": False,
                    "model": "error", "passed": 0,
                    "total": len(p["test_cases"]), "pass_all": False,
                }
            with lock:
                results.append(r)
                done_n[0] += 1
                esc_sym  = "↗" if r.get("escalated") else " "
                pass_sym = "✓" if r["pass_all"] else "✗"
                tc_str   = f"{r['passed']}/{r['total']}"
                print(
                    f"  {done_n[0]:>5}  {r['gate1']:>6}  {r['model']:<12}  "
                    f"{esc_sym:>4}  {pass_sym} {tc_str:<6}  {r['title'][:38]}"
                )

    results.sort(key=lambda x: x["title"])
    RESULTS_FILE.write_text(json.dumps({
        "n":         total,
        "results":   results,
        "token_log": json.loads(tracker.export_json()),
    }, indent=2))
    return results, tracker


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list, tracker):
    total     = len(results)
    passed    = sum(1 for r in results if r["pass_all"])
    g1_hard   = sum(1 for r in results if r.get("gate1") == "hard")
    g1_easy   = total - g1_hard
    escalated = sum(1 for r in results if r.get("escalated"))
    gemini_n  = sum(1 for r in results if r.get("model") == "gemini")
    gemini_ok = sum(1 for r in results if r.get("model") == "gemini" and r["pass_all"])
    esc_ok    = sum(1 for r in results if r.get("model") == "opus-esc" and r["pass_all"])
    g1h_ok    = sum(1 for r in results if r.get("model") == "opus-G1"  and r["pass_all"])
    cost      = tracker.total_cost() if tracker else None
    W         = 72

    print("\n" + "=" * W)
    print("  smart-ask Product Benchmark — LiveBench Coding")
    print("=" * W)
    print(f"\n  Routing breakdown  ({total} problems)")
    print(f"    G1 easy   → Gemini            {g1_easy:>4}  ({g1_easy/total*100:.0f}%)")
    print(f"    G1 hard   → Opus direct       {g1_hard:>4}  ({g1_hard/total*100:.0f}%)")
    print(f"    Escalated → Opus via ESCALATE {escalated:>4}  ({escalated/total*100:.0f}%)")
    print(f"\n  {'─'*70}")
    print(f"  {'pass@1 (all public tests)':38}  {passed}/{total} ({passed/total*100:.1f}%)")
    if cost is not None:
        print(f"  {'total cost':38}  ${cost:.6f}")
    print(f"\n  Per-path accuracy")
    if gemini_n:
        print(f"    Gemini (no escalation)       {gemini_ok}/{gemini_n}  ({gemini_ok/gemini_n*100:.1f}%)")
    if escalated:
        print(f"    Opus after ESCALATE          {esc_ok}/{escalated}  ({esc_ok/escalated*100:.1f}%)")
    if g1_hard:
        print(f"    Opus via G1-hard             {g1h_ok}/{g1_hard}  ({g1h_ok/g1_hard*100:.1f}%)")
    print(f"\n  By LeetCode difficulty")
    for diff in ("easy", "medium", "hard"):
        n  = sum(1 for r in results if r.get("difficulty") == diff)
        ok = sum(1 for r in results if r.get("difficulty") == diff and r["pass_all"])
        if n:
            print(f"    {diff:<8}  {ok}/{n}  ({ok/n*100:.0f}%)")
    if tracker:
        tracker.report("Token & cost breakdown")
    print("=" * W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="smart-ask product benchmark on LiveBench coding")
    p.add_argument("-n",       type=int,         default=None, help="Limit to first N problems")
    p.add_argument("--report", action="store_true",            help="Print saved results")
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

    print("  Loading LiveBench coding dataset...")
    problems = load_livebench()
    print(f"  Loaded {len(problems)} problems  "
          f"({sum(1 for p in problems if p['task']=='LCB_generation')} LCB_generation, "
          f"{sum(1 for p in problems if p['task']=='coding_completion')} coding_completion)\n")
    results, tracker = run(problems, n=args.n)
    print_report(results, tracker)


if __name__ == "__main__":
    main()
