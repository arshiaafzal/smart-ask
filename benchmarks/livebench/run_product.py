#!/usr/bin/env python3
"""
smart-ask product benchmark — LiveBench (coding category).

128 problems from real LeetCode/AtCoder contests (June 2024 release).
No contamination — questions postdate most model training cuts.

Two task types
--------------
  LCB_generation   (78)  Generate a complete Python Solution class.
                          LeetCode-style: method called with parsed args.
  coding_completion (50)  Complete a partial solution snippet.
                          Same execution harness; partial + completion stitched first.

Two test harnesses
------------------
  functional  Solution().method(*args)  compared to expected return value
  stdin       code reads sys.stdin      stdout compared to expected

Gate routing
------------
  Same logic as run_product.py / smart-ask CLI:
    Gate 1  Gemini classifier  → easy / hard
    Gate 2  Gemini self-check  → ESCALATE_NOW if answer is insufficient
    Opus    hard (G1) or escalated (G2) tasks

Usage
-----
    python run_product.py          # all 128 problems
    python run_product.py -n 20   # first N
    python run_product.py --report
"""

import os, sys, json, ast, re, subprocess, tempfile, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from methods.cascade import (
    OR_BASE, CLASSIFIER_MODEL, EASY_MODEL, HARD_MODEL,
    CLASSIFY_PROMPT, SELF_CHECK_SUFFIX, ESCALATE_MARKER,
)
from openai import OpenAI

OR_KEY       = os.environ.get("OPENROUTER_API_KEY", "")
RESULTS_FILE = Path(__file__).parent / "results_product.json"
WORKERS      = 6

# ── System prompts ─────────────────────────────────────────────────────────────

CODE_SYSTEM = (
    "You are an expert competitive programmer. "
    "Return ONLY the Python code — no explanation, no markdown fences, no extra text."
)
OPUS_SYSTEM = (
    "You are an expert competitive programmer. "
    "Write correct, complete Python code. "
    "Return ONLY the code — no explanation, no markdown fences, no extra text."
)


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
            "question_id":    ex["question_id"],
            "task":           ex["task"],
            "title":          ex["question_title"],
            "prompt":         ex["turns"][0],           # full prompt for model
            "starter_code":   orig.get("starter_code", ""),
            "partial":        ex.get("partial_solution", ""),
            "test_cases":     tcs,
            "difficulty":     orig.get("difficulty", "?"),
        })
    return problems


# ── Code extraction ────────────────────────────────────────────────────────────

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.rstrip()  # rstrip only — preserves leading indent for completion stitching


# ── Execution harnesses ────────────────────────────────────────────────────────

_STDLIB = (
    "from typing import List, Tuple, Dict, Optional, Set, Any, Union, Counter\n"
    "import sys, math, re, collections, itertools, functools, heapq, bisect, ast\n"
    "from collections import defaultdict, Counter, deque\n"
)

def _parse_val(s: str):
    """Parse a string value to Python object; return str on failure."""
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return s

def _normalize(v) -> str:
    """Normalize a value to a canonical string for comparison."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_normalize(x) for x in v) + "]"
    return str(v)


def run_functional(code: str, starter_code: str, tc: dict, timeout: int = 10) -> bool:
    """
    Execute a LeetCode-style test case.
    Instantiates Solution(), calls the method with parsed args, compares result.
    """
    # Extract method name from starter_code
    m = re.search(r"def (\w+)\(self", starter_code)
    if not m:
        return False
    fn_name = m.group(1)

    # Parse input — newline-separated args
    raw_inp = tc["input"].strip()
    lines   = [l.strip() for l in raw_inp.splitlines() if l.strip()]
    args    = [_parse_val(l) for l in lines]
    args_r  = repr(args[0]) if len(args) == 1 else repr(tuple(args))
    if len(args) == 1:
        call = f"_s.{fn_name}({repr(args[0])})"
    else:
        call = f"_s.{fn_name}(*{repr(tuple(args))})"

    expected_raw = tc["output"].strip()
    expected     = _parse_val(expected_raw)

    script = (
        _STDLIB
        + "\n"
        + code
        + f"\n\n_s = Solution()\n_r = {call}\nprint(repr(_r))\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
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
    """
    Execute an AtCoder-style test case (stdin → stdout comparison).
    """
    script = _STDLIB + "\n" + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname],
            input=tc["input"],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            return False
        return res.stdout.strip() == tc["output"].strip()
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def run_tests(code: str, starter_code: str, test_cases: list, timeout: int = 10) -> tuple:
    """
    Run all public test cases. Returns (passed, total).
    Picks functional or stdin harness based on testtype.
    """
    passed = 0
    for tc in test_cases:
        ttype = tc.get("testtype", "functional")
        try:
            if ttype == "functional":
                ok = run_functional(code, starter_code, tc, timeout)
            else:
                ok = run_stdin(code, tc, timeout)
        except Exception:
            ok = False
        if ok:
            passed += 1
    return passed, len(test_cases)


# ── Gate 1 ─────────────────────────────────────────────────────────────────────

def gate1_classify(client, prompt: str, tracker, task_id: str) -> str:
    r = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT + prompt[:1200]}],
        max_tokens=20, temperature=0,
    )
    tracker.record(CLASSIFIER_MODEL, "classifier", r.usage, task_id)
    raw = (r.choices[0].message.content or "").strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw).get("d", "easy")
    except Exception:
        return "easy"


# ── Gemini with self-check ─────────────────────────────────────────────────────

def gemini_generate(client, prompt: str, tracker, task_id: str):
    """
    Send prompt + SELF_CHECK_SUFFIX to Gemini.
    Returns (code: str, escalated: bool).
    """
    full = prompt + SELF_CHECK_SUFFIX
    r = client.chat.completions.create(
        model=EASY_MODEL,
        messages=[
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user",   "content": full},
        ],
        max_tokens=1024, temperature=0.0,
    )
    tracker.record(EASY_MODEL, "generator", r.usage, task_id)
    raw = r.choices[0].message.content or ""

    escalated = bool(re.search(
        rf'^\s*{re.escape(ESCALATE_MARKER)}\s*$', raw, re.MULTILINE
    ))
    code_part = raw.split(ESCALATE_MARKER)[0].strip() if escalated else raw
    return strip_fences(code_part), escalated


# ── Opus ───────────────────────────────────────────────────────────────────────

def run_opus(client, prompt: str, tracker, task_id: str, escalated: bool = False) -> str:
    if escalated:
        user_msg = (
            "A previous attempt at this task was flagged as insufficient. "
            "Please solve this correctly and completely:\n\n" + prompt
        )
        role = "fixer"
    else:
        user_msg = prompt
        role = "writer"
    r = client.chat.completions.create(
        model=HARD_MODEL,
        messages=[
            {"role": "system", "content": OPUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=1024, temperature=0.0,
    )
    tracker.record(HARD_MODEL, role, r.usage, task_id)
    return strip_fences(r.choices[0].message.content or "")


# ── Per-problem runner ─────────────────────────────────────────────────────────

def run_problem(client, prob: dict, tracker: TokenTracker) -> dict:
    qid      = prob["question_id"][:16]   # short ID for display
    prompt   = prob["prompt"]
    partial  = prob["partial"]
    starter  = prob["starter_code"]
    tcs      = prob["test_cases"]

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

    # Gate 1
    difficulty    = gate1_classify(client, prompt, tracker, qid)
    rec["gate1"]  = difficulty

    if difficulty == "hard":
        code          = run_opus(client, prompt, tracker, qid, escalated=False)
        rec["model"]  = "opus-G1"
    else:
        code, escalated = gemini_generate(client, prompt, tracker, qid)
        rec["escalated"] = escalated
        if escalated:
            code         = run_opus(client, prompt, tracker, qid, escalated=True)
            rec["model"] = "opus-esc"
        else:
            rec["model"] = "gemini"

    # For coding_completion: stitch partial + completion
    # If model re-output the full class (with "class Solution"), use it directly.
    if prob["task"] == "coding_completion" and partial:
        if "class Solution" in code:
            full_code = code
        else:
            full_code = partial + "\n" + code
    else:
        full_code = code

    passed, total     = run_tests(full_code, starter, tcs)
    rec["passed"]     = passed
    rec["total"]      = total
    rec["pass_all"]   = passed == total and total > 0
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
            except Exception as e:
                r = {
                    "question_id": p["question_id"],
                    "title":       p["title"],
                    "task":        p["task"],
                    "difficulty":  p["difficulty"],
                    "gate1":       "?", "escalated": False,
                    "model":       "error", "passed": 0,
                    "total":       len(p["test_cases"]), "pass_all": False,
                }
            with lock:
                results.append(r)
                done_n[0] += 1
                esc_sym  = "↗" if r.get("escalated") else " "
                pass_sym = "✓" if r["pass_all"] else "✗"
                tc_str   = f"{r['passed']}/{r['total']}"
                title    = r["title"][:38]
                print(
                    f"  {done_n[0]:>5}  {r['gate1']:>6}  {r['model']:<12}  "
                    f"{esc_sym:>4}  {pass_sym} {tc_str:<6}  {title}"
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
    g1h_ok    = sum(1 for r in results if r.get("model") == "opus-G1" and r["pass_all"])

    # By LeetCode difficulty
    for diff in ("easy", "medium", "hard"):
        n = sum(1 for r in results if r.get("difficulty") == diff)
        ok= sum(1 for r in results if r.get("difficulty") == diff and r["pass_all"])
        if n:
            print(f"    {diff:<8}  {ok}/{n}  ({ok/n*100:.0f}%)")

    cost = tracker.total_cost() if tracker else None
    W    = 72

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
        print(f"  {'total cost':38}  ${cost:.5f}")

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
