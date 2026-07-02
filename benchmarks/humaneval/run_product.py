#!/usr/bin/env python3
"""
smart-ask product benchmark — HumanEval.

Tests the EXACT logic the smart-ask CLI uses in real sessions:

  Gate 1   Gemini classifier  (pre-flight, same as CLI)
  Gemini   Generates answer + appended self-check instruction
  Escalate If model outputs ESCALATE → re-run with Opus + failure hint
  Opus     Hard tasks (G1) or escalated tasks (self-check triggered)

This is NOT a raw-API test. It mirrors the CLI word for word:
  • Same classifier prompt as _CLASSIFY_PROMPT in smart-ask
  • Same self-check suffix as _SELF_CHECK_SUFFIX in smart-ask
  • Same escalation restart logic

Usage
-----
    python run_product.py          # all 164 problems
    python run_product.py -n 20   # first N (quick smoke test)
    python run_product.py --report
"""

import os, sys, json, subprocess, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Repo root on sys.path so shared modules are importable ────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from harness import strip_fences, run_tests
from methods.cascade import (
    OR_BASE, CLASSIFIER_MODEL, EASY_MODEL, HARD_MODEL,
    CLASSIFY_PROMPT, SELF_CHECK_SUFFIX, ESCALATE_MARKER,
)

from openai import OpenAI

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

RESULTS_FILE = Path(__file__).parent / "results_product.json"
WORKERS      = 8

CODE_SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Return ONLY the complete function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

OPUS_SYSTEM = (
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
        sys.exit("pip install -r requirements.txt")
    return list(load_dataset("openai/openai_humaneval", split="test"))


# strip_fences and run_tests are imported from harness/

# ── Gate 1 ────────────────────────────────────────────────────────────

def gate1_classify(client, prompt, tracker, task_id) -> str:
    """Returns 'easy' or 'hard'."""
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


# ── Gemini with self-check ────────────────────────────────────────────

def gemini_selfcheck(client, prompt, tracker, task_id):
    """
    Mirrors CLI behaviour: sends prompt + SELF_CHECK_SUFFIX to Gemini.
    Returns (code: str, escalated: bool).
    """
    full_prompt = f"Complete this Python function:\n\n{prompt}{SELF_CHECK_SUFFIX}"
    r = client.chat.completions.create(
        model=EASY_MODEL,
        messages=[
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user",   "content": full_prompt},
        ],
        max_tokens=700, temperature=0.0,
    )
    tracker.record(EASY_MODEL, "generator", r.usage, task_id)
    raw = r.choices[0].message.content or ""

    escalated = bool(re.search(
        rf'^\s*{re.escape(ESCALATE_MARKER)}\s*$', raw, re.MULTILINE
    ))

    # Extract code portion (before the marker if present)
    if escalated:
        code_part = raw.split(ESCALATE_MARKER)[0].strip()
    else:
        code_part = raw

    return strip_fences(code_part), escalated


# ── Opus ──────────────────────────────────────────────────────────────

def run_opus(client, prompt, tracker, task_id, escalated=False):
    """
    Mirrors CLI: Opus gets original problem.
    If escalated, includes a brief failure hint (no Gemini code — same as v3 fresh-prompt).
    """
    if escalated:
        user_msg = (
            f"A previous attempt at this task was flagged as insufficient "
            f"(the model self-reported low confidence or a stub answer).\n\n"
            f"Please solve this correctly:\n\n"
            f"Complete this Python function:\n\n{prompt}"
        )
        role = "fixer"
    else:
        user_msg = f"Complete this Python function:\n\n{prompt}"
        role = "writer"

    r = client.chat.completions.create(
        model=HARD_MODEL,
        messages=[
            {"role": "system", "content": OPUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=512, temperature=0.0,
    )
    tracker.record(HARD_MODEL, role, r.usage, task_id)
    return strip_fences(r.choices[0].message.content or "")


# ── Per-problem runner ────────────────────────────────────────────────

def run_problem(client, prob, tracker) -> dict:
    prompt      = prob["prompt"]
    test_code   = prob["test"]
    entry       = prob["entry_point"]
    task_id     = prob["task_id"]

    rec = {
        "task_id":   task_id,
        "difficulty": None,
        "escalated":  False,
        "model":      None,
        "passed":     False,
    }

    # Gate 1
    difficulty    = gate1_classify(client, prompt, tracker, task_id)
    rec["difficulty"] = difficulty

    if difficulty == "hard":
        code         = run_opus(client, prompt, tracker, task_id, escalated=False)
        rec["model"] = "opus-G1"
    else:
        code, escalated = gemini_selfcheck(client, prompt, tracker, task_id)
        rec["escalated"] = escalated

        if escalated:
            code         = run_opus(client, prompt, tracker, task_id, escalated=True)
            rec["model"] = "opus-esc"
        else:
            rec["model"] = "gemini"

    rec["passed"] = run_tests(prompt, code, test_code, entry)
    return rec


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

    print(f"\n  Running {total} HumanEval problems through smart-ask product logic...\n")
    print(f"  {'#':>5}  {'G1':>6}  {'model':<12}  {'esc':>4}  {'pass':>5}  task_id")
    print(f"  {'─'*68}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_problem, client, p, tracker): p for p in problems}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                p = futs[fut]
                r = {"task_id": p["task_id"], "difficulty": "?",
                     "escalated": False, "model": "error", "passed": False}
            with lock:
                results.append(r)
                done_n[0] += 1
                esc_sym  = "↗" if r.get("escalated") else " "
                pass_sym = "✓" if r["passed"] else "✗"
                model_s  = r.get("model", "?")
                print(
                    f"  {done_n[0]:>5}  {r['difficulty']:>6}  {model_s:<12}  "
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
    opus_total= sum(1 for r in results if "opus" in (r.get("model") or ""))

    gemini_ok = sum(1 for r in results if r.get("model") == "gemini" and r["passed"])
    gemini_n  = sum(1 for r in results if r.get("model") == "gemini")
    esc_ok    = sum(1 for r in results if r.get("model") == "opus-esc" and r["passed"])
    g1h_ok    = sum(1 for r in results if r.get("model") == "opus-G1" and r["passed"])

    W = 68
    cost = tracker.total_cost() if tracker else None

    print("\n" + "=" * W)
    print("  smart-ask Product Benchmark — HumanEval")
    print("=" * W)

    print(f"\n  Routing breakdown  ({total} problems)")
    print(f"    G1 easy   → Gemini              {g1_easy:>4}  ({g1_easy/total*100:.0f}%)")
    print(f"    G1 hard   → Opus direct         {g1_hard:>4}  ({g1_hard/total*100:.0f}%)")
    print(f"    Escalated → Opus via ESCALATE   {escalated:>4}  ({escalated/total*100:.0f}%)")

    print(f"\n  {'─'*66}")
    print(f"  {'':38}  {'product':>12}  ")
    print(f"  {'─'*66}")
    print(f"  {'pass@1':38}  {passed}/{total} ({passed/total*100:.1f}%)")
    if cost is not None:
        print(f"  {'total cost':38}  ${cost:.5f}")

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
    p.add_argument("--report", action="store_true",    help="Print saved results without re-running")
    args = p.parse_args()

    if args.report:
        if not RESULTS_FILE.exists():
            sys.exit("No saved results. Run without --report first.")
        data    = json.loads(RESULTS_FILE.read_text())
        tracker = TokenTracker()
        # Re-hydrate tracker from saved log so costs display correctly
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
