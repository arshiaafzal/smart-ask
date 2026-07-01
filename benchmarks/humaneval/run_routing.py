#!/usr/bin/env python3
"""
smart-ask end-to-end routing benchmark.

Each HumanEval problem is run through the real smart-ask classifier
(claude-haiku-4.5), then routed to the appropriate model:
  easy → google/gemini-2.5-flash-lite
  hard → anthropic/claude-opus-4.8

Results are compared against:
  • smart-ask routing   — classifier + cheapest capable model
  • always-Opus         — every problem sent to Opus (expensive baseline)

Usage:
    python run_routing.py          # all 164 problems
    python run_routing.py -n 20   # first 20 problems (quick test)
    python run_routing.py --report
"""

import os, sys, json, subprocess, tempfile, argparse
from pathlib import Path
from openai import OpenAI

OR_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"

CLASSIFIER_MODEL = "anthropic/claude-haiku-4.5"
EASY_MODEL       = "google/gemini-2.5-flash-lite"
HARD_MODEL       = "anthropic/claude-opus-4.8"

# Per-token prices (fetched 2026-07-01)
PRICES = {
    CLASSIFIER_MODEL: {"input": 0.0000008,  "output": 0.000001},
    EASY_MODEL:       {"input": 0.0000001,  "output": 0.0000004},
    HARD_MODEL:       {"input": 0.000005,   "output": 0.000025},
}

RESULTS_FILE = Path(__file__).parent / "results_routing.json"

# ── Dataset ────────────────────────────────────────────────────────────────

def load_humaneval():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency — run: pip install -r requirements.txt")
    ds = load_dataset("openai/openai_humaneval", split="test")
    return list(ds)


# ── Classifier (mirrors smart-ask logic exactly) ───────────────────────────

def classify(client, task_text: str) -> tuple[str, int, int]:
    """
    Classify a task as 'easy' or 'hard' using claude-haiku-4.5.
    Returns (difficulty, input_tokens, output_tokens).
    """
    user_msg = (
        "Classify this AI task as easy or hard. "
        "easy=Q&A/simple-code/explain/debug/format/scripts. "
        "hard=complex-architecture/multi-system-design/advanced-research/novel-algorithms. "
        'Reply ONLY with JSON: {"d":"easy"} or {"d":"hard"}\n\nTask: ' + task_text[:800]
    )
    try:
        r = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=20,
            temperature=0,
        )
        raw = r.choices[0].message.content.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        difficulty = json.loads(raw).get("d", "easy")
        return difficulty, r.usage.prompt_tokens, r.usage.completion_tokens
    except Exception:
        return "easy", 0, 0   # default to easy on failure


# ── Code completion (same as run.py) ───────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Return ONLY the complete function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

def complete(client, model_id: str, prompt: str) -> tuple[str, int, int]:
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Complete this Python function:\n\n{prompt}"},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return (
        resp.choices[0].message.content or "",
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()

def check(prompt, completion, test_code, entry_point, timeout=10) -> bool:
    code = strip_fences(completion)
    impl = code if f"def {entry_point}" in code else prompt + code
    full = (
        "from typing import List, Tuple, Dict, Optional, Set, Any, Union\n"
        "import math, re, collections, itertools, functools, heapq, bisect\n\n"
        + impl + "\n\n"
        + test_code + "\n\n"
        + f"check({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname], capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


# ── Token cost helper ──────────────────────────────────────────────────────

def token_cost(model_id, in_tok, out_tok) -> float:
    p = PRICES[model_id]
    return in_tok * p["input"] + out_tok * p["output"]


# ── Benchmark runner ────────────────────────────────────────────────────────

def run(problems, n=None):
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set in environment")
    if n:
        problems = problems[:n]

    client = OpenAI(base_url=OR_BASE, api_key=OR_KEY)
    total  = len(problems)

    # Counters
    routing = {
        "passed": 0, "failed": 0, "errors": 0,
        "easy_count": 0, "hard_count": 0,
        "classifier_cost": 0.0,
        "model_cost": 0.0,
    }
    baseline = {
        "passed": 0, "failed": 0, "errors": 0,
        "cost": 0.0,
    }

    print(f"\n  Running {total} HumanEval problems through smart-ask routing...\n")
    print(f"  {'#':>5}  {'difficulty':<12}  {'model':<26}  {'routed':<6}  {'opus':<6}")
    print(f"  {'─'*60}")

    records = []

    for i, prob in enumerate(problems):
        prompt      = prob["prompt"]
        test_code   = prob["test"]
        entry_point = prob["entry_point"]
        task_id     = prob["task_id"]

        # ── 1. Classify ────────────────────────────────────────────────
        difficulty, clf_in, clf_out = classify(client, prompt)
        clf_cost = token_cost(CLASSIFIER_MODEL, clf_in, clf_out)
        routing["classifier_cost"] += clf_cost

        model_id = EASY_MODEL if difficulty == "easy" else HARD_MODEL
        routing[f"{difficulty}_count"] += 1

        # ── 2. smart-ask route ─────────────────────────────────────────
        try:
            completion, in_tok, out_tok = complete(client, model_id, prompt)
            passed_r = check(prompt, completion, test_code, entry_point)
            routing["passed" if passed_r else "failed"] += 1
            routing["model_cost"] += token_cost(model_id, in_tok, out_tok)
            routed_sym = "✓" if passed_r else "✗"
        except Exception as e:
            routing["errors"] += 1
            routed_sym = f"E"
            passed_r = False

        # ── 3. Always-Opus baseline ────────────────────────────────────
        try:
            completion_b, in_tok_b, out_tok_b = complete(client, HARD_MODEL, prompt)
            passed_b = check(prompt, completion_b, test_code, entry_point)
            baseline["passed" if passed_b else "failed"] += 1
            baseline["cost"] += token_cost(HARD_MODEL, in_tok_b, out_tok_b)
            opus_sym = "✓" if passed_b else "✗"
        except Exception:
            baseline["errors"] += 1
            opus_sym = "E"

        short_model = "gemini-lite" if model_id == EASY_MODEL else "opus-4.8   "
        print(f"  {i+1:>5}  {difficulty:<12}  {short_model:<26}  {routed_sym:<6}  {opus_sym:<6}  {task_id}")

        records.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "routed_model": model_id,
            "routed_passed": passed_r,
            "opus_passed": passed_b if "passed_b" in dir() else False,
        })

    RESULTS_FILE.write_text(json.dumps({
        "n": total,
        "routing": routing,
        "baseline": baseline,
        "records": records,
    }, indent=2))

    return routing, baseline, total


# ── Report ─────────────────────────────────────────────────────────────────

def print_report(routing, baseline, n):
    W = 66
    print("\n" + "=" * W)
    print("  smart-ask Routing Benchmark  —  HumanEval")
    print("=" * W)

    # Routing stats
    r_done   = routing["passed"] + routing["failed"]
    r_acc    = routing["passed"] / r_done * 100 if r_done else 0
    r_total  = routing["classifier_cost"] + routing["model_cost"]
    r_cps    = r_total / routing["passed"] if routing["passed"] else float("inf")

    # Opus baseline
    b_done   = baseline["passed"] + baseline["failed"]
    b_acc    = baseline["passed"] / b_done * 100 if b_done else 0
    b_cps    = baseline["cost"] / baseline["passed"] if baseline["passed"] else float("inf")

    pct_easy = routing["easy_count"] / n * 100
    pct_hard = routing["hard_count"] / n * 100

    print(f"\n  Classifier routing  ({n} problems)")
    print(f"    easy  →  gemini-2.5-flash-lite   {routing['easy_count']:>3}  ({pct_easy:.0f}%)")
    print(f"    hard  →  claude-opus-4.8          {routing['hard_count']:>3}  ({pct_hard:.0f}%)")

    print(f"\n  {'─'*64}")
    print(f"  {'':30}  {'smart-ask':>14}  {'always-Opus':>14}")
    print(f"  {'─'*64}")
    print(f"  {'pass@1':30}  {routing['passed']}/{r_done} ({r_acc:.1f}%)  {baseline['passed']}/{b_done} ({b_acc:.1f}%)")
    print(f"  {'total cost':30}  ${r_total:>13.5f}  ${baseline['cost']:>13.5f}")
    print(f"  {'cost / solved problem':30}  ${r_cps:>13.6f}  ${b_cps:>13.6f}")

    if baseline["cost"] > 0:
        savings_pct = (baseline["cost"] - r_total) / baseline["cost"] * 100
        multiplier  = baseline["cost"] / r_total if r_total > 0 else float("inf")
        acc_delta   = r_acc - b_acc
        print(f"  {'─'*64}")
        print(f"\n  Cost saving   {savings_pct:.1f}%  ({multiplier:.1f}x cheaper)")
        print(f"  Accuracy gap  {acc_delta:+.1f}pp vs always-Opus")
        print(f"  Cost saved    ${baseline['cost'] - r_total:.5f} per {n} problems")

    print("=" * W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="smart-ask end-to-end routing benchmark")
    p.add_argument("-n", type=int, default=None,
                   help="Number of problems to run (default: all 164)")
    p.add_argument("--report", action="store_true",
                   help="Print saved results without re-running")
    args = p.parse_args()

    if args.report:
        if not RESULTS_FILE.exists():
            sys.exit("No saved results. Run without --report first.")
        data = json.loads(RESULTS_FILE.read_text())
        print_report(data["routing"], data["baseline"], data["n"])
        return

    print("  Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"  Loaded {len(problems)} problems")

    routing, baseline, total = run(problems, n=args.n)
    print_report(routing, baseline, total)


if __name__ == "__main__":
    main()
