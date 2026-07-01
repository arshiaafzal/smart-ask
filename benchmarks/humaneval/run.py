#!/usr/bin/env python3
"""
HumanEval benchmark for smart-ask
Measures pass@1 accuracy and cost: Gemini 3.5 Flash vs Claude Opus 4.8

Usage:
    python run.py            # all 164 problems
    python run.py -n 20      # first 20 problems (quick test)
    python run.py --report   # print last saved results without re-running
"""

import os, sys, json, subprocess, tempfile, argparse
from pathlib import Path
from openai import OpenAI

OR_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"

# Prices from OpenRouter API (per token, fetched 2026-07-01)
MODELS = {
    "gemini-2.5-flash-lite": {
        "id":     "google/gemini-2.5-flash-lite",
        "input":  0.0000001,    # $ per input token
        "output": 0.0000004,    # $ per output token
        "label":  "Google Gemini 2.5 Flash Lite  [easy route]",
    },
    "claude-opus-4.8": {
        "id":     "anthropic/claude-opus-4.8",
        "input":  0.000005,     # $ per input token
        "output": 0.000025,     # $ per output token
        "label":  "Anthropic Claude Opus 4.8 [hard route]",
    },
}

RESULTS_FILE = Path(__file__).parent / "results.json"

# ── Dataset ────────────────────────────────────────────────────────────────

def load_humaneval():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency — run: pip install -r requirements.txt")
    ds = load_dataset("openai/openai_humaneval", split="test")
    return list(ds)


# ── Model calls ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Return ONLY the complete function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

def complete(client, model_id, prompt, max_tokens=512):
    """Call OpenRouter and return (completion, input_tokens, output_tokens)."""
    user_msg = f"Complete this Python function:\n\n{prompt}"
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text  = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


# ── Evaluation ─────────────────────────────────────────────────────────────

def strip_fences(text):
    """Remove ```python … ``` if the model wrapped its answer."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()

def check(prompt, completion, test_code, entry_point, timeout=10):
    """
    Run the model's completion against the HumanEval test harness.
    Returns True if all assertions pass.
    """
    code = strip_fences(completion)

    # If the completion contains the function definition, use it directly.
    # Otherwise treat it as a function body and prepend the prompt signature.
    if f"def {entry_point}" in code:
        impl = code
    else:
        impl = prompt + code

    # Standard HumanEval imports + the implementation + test harness
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
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


# ── Benchmark runner ────────────────────────────────────────────────────────

def run(problems, n=None):
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set in environment")

    if n:
        problems = problems[:n]

    client = OpenAI(base_url=OR_BASE, api_key=OR_KEY)

    stats = {
        name: {"passed": 0, "failed": 0, "errors": 0, "in_tok": 0, "out_tok": 0}
        for name in MODELS
    }

    total = len(problems)
    print(f"\n  Running {total} HumanEval problems against {len(MODELS)} models...\n")

    for i, prob in enumerate(problems):
        task_id     = prob["task_id"]
        prompt      = prob["prompt"]
        test_code   = prob["test"]
        entry_point = prob["entry_point"]

        for name, cfg in MODELS.items():
            label = name[:20].ljust(20)
            print(f"  [{i+1:>3}/{total}]  {label}  {task_id}", end="  ", flush=True)

            try:
                completion, in_tok, out_tok = complete(client, cfg["id"], prompt)
                passed = check(prompt, completion, test_code, entry_point)

                stats[name]["passed" if passed else "failed"] += 1
                stats[name]["in_tok"]  += in_tok
                stats[name]["out_tok"] += out_tok

                print("✓" if passed else "✗")
            except Exception as e:
                stats[name]["errors"] += 1
                print(f"ERR: {e}")

    # Save results for --report
    RESULTS_FILE.write_text(json.dumps({"n": total, "stats": stats}, indent=2))
    return stats, total


# ── Report ─────────────────────────────────────────────────────────────────

def print_report(stats, n_problems):
    W = 60
    print("\n" + "=" * W)
    print("  HumanEval Benchmark  —  smart-ask cost comparison")
    print("=" * W)

    costs = {}
    for name, cfg in MODELS.items():
        s    = stats[name]
        done = s["passed"] + s["failed"]
        acc  = s["passed"] / done * 100 if done else 0
        cost = s["in_tok"] * cfg["input"] + s["out_tok"] * cfg["output"]
        cps  = cost / s["passed"] if s["passed"] else float("inf")
        costs[name] = cost

        print(f"\n  {cfg['label']}")
        print(f"    pass@1      {s['passed']}/{done}  ({acc:.1f}%)")
        print(f"    total cost  ${cost:.5f}")
        print(f"    cost/solved ${cps:.6f}")
        print(f"    tokens      {s['in_tok']:,} in  /  {s['out_tok']:,} out")
        if s["errors"]:
            print(f"    errors      {s['errors']}")

    gem_cost = costs["gemini-2.5-flash-lite"]
    opu_cost = costs["claude-opus-4.8"]
    if opu_cost > 0 and gem_cost > 0:
        savings_pct = (opu_cost - gem_cost) / opu_cost * 100
        multiplier  = opu_cost / gem_cost
        print(f"\n  {'─'*40}")
        print(f"  Cost saving   {savings_pct:.1f}%  ({multiplier:.1f}x cheaper)")
        print(f"  Gemini cost   ${gem_cost:.5f}")
        print(f"  Opus cost     ${opu_cost:.5f}")
        print(f"  Difference    ${opu_cost - gem_cost:.5f} saved per {n_problems} problems")

    print("=" * W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="HumanEval cost benchmark for smart-ask")
    p.add_argument("-n", type=int, default=None,
                   help="Number of problems to run (default: all 164)")
    p.add_argument("--report", action="store_true",
                   help="Print saved results without re-running")
    args = p.parse_args()

    if args.report:
        if not RESULTS_FILE.exists():
            sys.exit("No saved results. Run without --report first.")
        data = json.loads(RESULTS_FILE.read_text())
        print_report(data["stats"], data["n"])
        return

    print("  Loading HumanEval dataset (downloads once, ~1 MB)...")
    problems = load_humaneval()
    print(f"  Loaded {len(problems)} problems")

    stats, n = run(problems, n=args.n)
    print_report(stats, n)


if __name__ == "__main__":
    main()
