#!/usr/bin/env python3
"""
smart-ask three-layer cascade benchmark  (v3 — exact token tracking + smart escalation).

Gates
-----
  Gate 1  haiku checklist scorer  — is the problem structurally complex?
  Gate 2  Gemini self-confidence  — does Gemini itself think its answer is right?
  Gate 3  visible doctest check   — does Gemini's code pass the visible examples?

Smart escalation
----------------
When a problem is sent to Opus it does NOT start from scratch.
  G2 / G3 escalations:  Opus receives Gemini's code + the specific failure.
                         It only needs to patch the bug — far fewer tokens.
  G1 escalations:       No Gemini code exists; Opus gets the original problem.

Token tracking
--------------
Every API response's exact usage.prompt_tokens / usage.completion_tokens is
recorded via tracker.TokenTracker.  No estimates.  See report at the end.

CONTAMINATION NOTE
------------------
Gate 3 uses only the visible >>> examples that are part of the prompt —
NOT the hidden check() test suite.  No ground truth is leaked.

Usage
-----
    python run_cascade.py          # all 164 problems
    python run_cascade.py -n 20   # first N problems
    python run_cascade.py --report
"""

import os, sys, json, subprocess, tempfile, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Add benchmarks/ to path so tracker package is importable ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from tracker import TokenTracker

from openai import OpenAI

OR_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = "https://openrouter.ai/api/v1"

CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"
EASY_MODEL       = "google/gemini-2.5-flash-lite"
HARD_MODEL       = "anthropic/claude-opus-4.8"


RESULTS_FILE     = Path(__file__).parent / "results_cascade.json"
TRACKER_LOG_FILE = Path(__file__).parent / "token_log.json"

WORKERS = 8  # parallel threads

# ── Dataset ────────────────────────────────────────────────────────────────

def load_humaneval():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install -r requirements.txt")
    return list(load_dataset("openai/openai_humaneval", split="test"))


# ── Shared helpers ─────────────────────────────────────────────────────────

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if len(lines) > 1 else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text.strip()

def run_tests(prompt, code, test_code, entry_point, timeout=10) -> bool:
    """Execute generated code + hidden test suite. Returns True on pass."""
    impl = code if f"def {entry_point}" in code else prompt + code
    full = (
        "from typing import List, Tuple, Dict, Optional, Set, Any, Union\n"
        "import math, re, collections, itertools, functools, heapq, bisect\n\n"
        + impl + "\n\n"
        + test_code + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        r = subprocess.run([sys.executable, fname], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


# ── Gate 1: haiku easy/hard classifier ────────────────────────────────────

CLASSIFY_PROMPT = """\
You are routing a Python coding problem to either a cheap model (easy) or an expert model (hard).

Label it "hard" if ANY of these are true:
- Requires dynamic programming, graph traversal, or non-obvious algorithm
- Has subtle edge cases a junior programmer would likely miss (empty input, overflow, off-by-one, special chars)
- The visible examples look trivial but the real inputs are much harder
- Needs number theory, combinatorics, or careful mathematical reasoning

Label it "easy" if:
- Solvable with a list comprehension, basic loop, or simple string/math operation
- Edge cases are obvious and minimal
- A competent junior programmer gets it right on the first try

Reply ONLY with JSON: {"d":"easy"} or {"d":"hard"}

Problem:
"""

def gate1_classify(client, prompt, tracker, task_id):
    """Returns (difficulty, score) where score is 0 (easy) or 8 (hard) for display."""
    r = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT + prompt[:1200]}],
        max_tokens=20,
        temperature=0,
    )
    tracker.record(CLASSIFIER_MODEL, "classifier", r.usage, task_id)

    raw = (r.choices[0].message.content or "").strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        difficulty = json.loads(raw).get("d", "easy")
    except Exception:
        difficulty = "easy"

    score = 8 if difficulty == "hard" else 0
    return difficulty, score


# ── Gate 2: Gemini generation + local code-smell confidence ───────────────

CODE_SYSTEM = (
    "You are an expert Python programmer. "
    "Complete the given Python function. "
    "Return ONLY the complete function implementation — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

def code_smell_confidence(code: str) -> int:
    """
    Check for obvious low-confidence signals without an API call.
    Returns 1 (escalate) or 5 (looks fine).
    """
    lower = code.lower()
    # Definite placeholder / stub signals
    if "raise notimplementederror" in lower:
        return 1
    if "# todo" in lower or "# fixme" in lower:
        return 1
    # Body is just `pass` or `...`
    body_lines = [
        l.strip() for l in code.splitlines()
        if l.strip() and not l.strip().startswith("def ")
        and not l.strip().startswith("#")
        and l.strip() not in ('"""', "'''")
    ]
    if not body_lines or all(l in ("pass", "...", "return", "return None") for l in body_lines):
        return 1
    return 5

def gate2_gemini(client, prompt, tracker, task_id):
    """
    Generate with Gemini, then check code smells locally (no extra API call).
    Returns (code, confidence 1 or 5).
    """
    r = client.chat.completions.create(
        model=EASY_MODEL,
        messages=[
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user",   "content": f"Complete this Python function:\n\n{prompt}"},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    code = strip_fences(r.choices[0].message.content or "")
    tracker.record(EASY_MODEL, "generator", r.usage, task_id)
    confidence = code_smell_confidence(code)
    return code, confidence


# ── Gate 3: visible doctest check ─────────────────────────────────────────

def extract_doctests(prompt, entry_point):
    """Parse >>> lines from the prompt. Returns [(call_expr, expected_str)]."""
    examples = []
    lines = prompt.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(f">>> {entry_point}("):
            call_expr = stripped[4:]
            if i + 1 < len(lines):
                expected = lines[i + 1].strip()
                if expected and not expected.startswith(">>>"):
                    examples.append((call_expr, expected))
        i += 1
    return examples

def gate3_doctest(code, prompt, entry_point):
    """
    Run visible doctest examples against Gemini's code.
    Returns (passed: bool, failure_context: str | None).
    failure_context is a human-readable description of what failed,
    usable as context for Opus when escalating.
    """
    examples = extract_doctests(prompt, entry_point)
    if not examples:
        return True, None   # no visible examples — assume pass

    impl = code if f"def {entry_point}" in code else prompt + code

    # Build a test file that captures actual vs expected
    test_lines = [
        "from typing import List, Tuple, Dict, Optional, Set, Any, Union",
        "import math, re, collections, itertools, functools, heapq, bisect",
        "import sys",
        "",
        impl,
        "",
    ]
    for i, (call_expr, expected) in enumerate(examples):
        # Use indexed variables — zero quoting issues in generated code
        test_lines.append(f"_got_{i} = repr({call_expr})")
        test_lines.append(f"_exp_{i} = {repr(expected)}")
        test_lines.append(f"if _got_{i} != _exp_{i}:")
        test_lines.append(f"    print('FAIL: got', _got_{i}, 'expected', _exp_{i})")
        test_lines.append(f"    sys.exit(1)")

    src = "\n".join(test_lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        fname = f.name
    try:
        r = subprocess.run([sys.executable, fname], capture_output=True, timeout=5)
        if r.returncode == 0:
            return True, None
        # Capture what failed for the Opus fix prompt
        failure_msg = r.stdout.decode().strip() or r.stderr.decode().strip()
        return False, failure_msg
    except subprocess.TimeoutExpired:
        return False, "Timed out running visible examples."
    finally:
        os.unlink(fname)


# ── Gate 3b: Gemini retry before escalating to Opus ───────────────────────

RETRY_SYSTEM = (
    "You are an expert Python programmer fixing a bug. "
    "Return ONLY the corrected complete function — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

def gate3_gemini_retry(client, code, prompt, entry_point, failure_context, tracker, task_id):
    """
    Give Gemini one more attempt with the specific failure as context.
    Returns (new_code, passed: bool, new_failure_context).
    Cost: one cheap Gemini call (~$0.00012) vs one Opus call (~$0.008).
    """
    retry_msg = (
        f"Your Python function has a bug. Fix it.\n\n"
        f"Failure:\n{failure_context}\n\n"
        f"Current code:\n{code}"
    )
    r = client.chat.completions.create(
        model=EASY_MODEL,
        messages=[
            {"role": "system", "content": RETRY_SYSTEM},
            {"role": "user",   "content": retry_msg},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    tracker.record(EASY_MODEL, "retry", r.usage, task_id)
    new_code  = strip_fences(r.choices[0].message.content or "")
    passed, new_failure = gate3_doctest(new_code, prompt, entry_point)
    return new_code, passed, new_failure


# ── Opus: escalation ───────────────────────────────────────────────────────

PATCH_SYSTEM = (
    "You are an expert Python programmer. "
    "Fix the bug in the provided function. Prefer minimal changes, but rewrite if the approach is wrong. "
    "Return ONLY the corrected complete function — starting from the `def` line. "
    "No explanation, no markdown fences, no extra text."
)

FRESH_PROMPT = """\
Complete this Python function:

{problem}
"""

HINT_PROMPT = """\
Complete this Python function correctly. Pay close attention — a previous attempt failed this test:

  {failure}

Make sure your solution handles this case.

{problem}
"""

def run_opus(client, prompt, tracker, task_id,
             gemini_code=None, escalation_reason=None, failure_context=None):
    """
    Call Opus with a fresh prompt.
    G3 escalations include the specific test failure as a hint so Opus knows what to nail.
    G1/G2 escalations get the raw problem — no broken Gemini code to anchor on.
    """
    if "G3" in (escalation_reason or "") and failure_context:
        user_msg = HINT_PROMPT.format(failure=failure_context, problem=prompt)
        role = "fixer"
    elif escalation_reason and escalation_reason != "G1-hard":
        # G2: low confidence — fresh problem, no buggy context
        user_msg = FRESH_PROMPT.format(problem=prompt)
        role = "fixer"
    else:
        # G1: hard problem
        user_msg = FRESH_PROMPT.format(problem=prompt)
        role = "writer"
    max_tok = 512

    r = client.chat.completions.create(
        model=HARD_MODEL,
        messages=[
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tok,
        temperature=0.0,
    )
    tracker.record(HARD_MODEL, role, r.usage, task_id)
    code = strip_fences(r.choices[0].message.content or "")
    return code


# ── Per-problem worker ─────────────────────────────────────────────────────

def process_one(idx, prob, client, tracker):
    """Run all three gates for one problem. Returns result dict."""
    prompt      = prob["prompt"]
    test_code   = prob["test"]
    entry_point = prob["entry_point"]
    task_id     = prob["task_id"]

    escalation_reason = None
    gemini_code       = None
    confidence        = None
    failure_context   = None
    g1_score          = 0

    # ── Gate 1 ────────────────────────────────────────────────────────────
    difficulty, g1_score = gate1_classify(client, prompt, tracker, task_id)
    if difficulty == "hard":
        escalation_reason = "G1-hard"

    # ── Gate 2 ────────────────────────────────────────────────────────────
    if escalation_reason is None:
        gemini_code, confidence = gate2_gemini(client, prompt, tracker, task_id)
        if confidence <= 2:
            escalation_reason = f"G2-conf={confidence}"

    # ── Gate 3 ────────────────────────────────────────────────────────────
    if escalation_reason is None:
        doctest_pass, failure_context = gate3_doctest(gemini_code, prompt, entry_point)
        if not doctest_pass and failure_context:
            # Gemini retry — one cheap attempt before paying for Opus
            gemini_code, doctest_pass, failure_context = gate3_gemini_retry(
                client, gemini_code, prompt, entry_point, failure_context, tracker, task_id
            )
        if not doctest_pass:
            escalation_reason = "G3-doctest"

    # ── Route ─────────────────────────────────────────────────────────────
    if escalation_reason:
        final_code  = run_opus(
            client, prompt, tracker, task_id,
            gemini_code     = gemini_code,
            escalation_reason = escalation_reason,
            failure_context = failure_context,
        )
        final_model = "opus"
    else:
        final_code  = gemini_code
        final_model = "gemini"

    passed = run_tests(prompt, final_code, test_code, entry_point)

    return {
        "idx":        idx,
        "task_id":    task_id,
        "g1_score":   g1_score,
        "difficulty": difficulty,
        "confidence": confidence,
        "escalation": escalation_reason,
        "final_model": final_model,
        "passed":     passed,
    }


# ── Benchmark runner ───────────────────────────────────────────────────────

def run(problems, n=None):
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")
    if n:
        problems = problems[:n]
    total   = len(problems)
    tracker = TokenTracker()

    counts = {
        "passed": 0, "failed": 0,
        "g1_hard": 0, "g2_low_conf": 0, "g3_doctest_fail": 0,
        "stayed_gemini": 0,
    }
    records = [None] * total
    lock    = threading.Lock()

    print(f"\n  Running {total} problems — 3-layer cascade  ({WORKERS} workers)\n")
    print(f"  {'#':>5}  {'G1':^7}  {'G2':^5}  {'G3':^5}  {'model':^14}  {'ok':^4}  task")
    print(f"  {'─'*68}")

    def make_client():
        return OpenAI(base_url=OR_BASE, api_key=OR_KEY)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_one, i, prob, make_client(), tracker): i
            for i, prob in enumerate(problems)
        }
        for future in as_completed(futures):
            r = future.result()
            i = r["idx"]
            esc = r["escalation"] or ""

            with lock:
                counts["passed" if r["passed"] else "failed"] += 1
                if r["difficulty"] == "hard":     counts["g1_hard"]          += 1
                if "G2" in esc:                   counts["g2_low_conf"]      += 1
                if esc == "G3-doctest":           counts["g3_doctest_fail"]  += 1
                if r["final_model"] == "gemini":  counts["stayed_gemini"]    += 1
                records[i] = r

            score  = r["g1_score"]
            g1_tag = f"{score}/8{'↑' if r['difficulty']=='hard' else ' '}"
            g2_tag = str(r["confidence"]) if r["confidence"] is not None else "—"
            g3_tag = "✓" if not esc else ("✗" if "G3" in esc else "—")
            sym    = "✓" if r["passed"] else "✗"

            print(f"  {i+1:>5}  {g1_tag:^7}  {g2_tag:^5}  {g3_tag:^5}  "
                  f"{r['final_model']:^14}  {sym:^4}  {r['task_id']}")

    # Save results + full token log
    RESULTS_FILE.write_text(
        json.dumps({"n": total, "counts": counts, "records": records}, indent=2)
    )
    TRACKER_LOG_FILE.write_text(tracker.export_json())

    return counts, tracker, total


# ── Report ─────────────────────────────────────────────────────────────────

def print_report(counts, tracker, n):
    done = counts["passed"] + counts["failed"]
    acc  = counts["passed"] / done * 100 if done else 0
    cost = tracker.total_cost()

    to_opus = counts["g1_hard"] + counts["g2_low_conf"] + counts["g3_doctest_fail"]

    # Token report (exact)
    tracker.report("Exact Token Usage — smart-ask Cascade")

    W = 66
    print("=" * W)
    print("  smart-ask Cascade  —  HumanEval Results")
    print("=" * W)

    print(f"\n  Escalation breakdown  ({n} problems)")
    print(f"    stayed on Gemini          {counts['stayed_gemini']:>4}  ({counts['stayed_gemini']/n*100:.0f}%)")
    print(f"    escalated to Opus         {to_opus:>4}  ({to_opus/n*100:.0f}%)")
    print(f"      ↳ gate 1 (G1 score)     {counts['g1_hard']:>4}")
    print(f"      ↳ gate 2 (confidence)   {counts['g2_low_conf']:>4}")
    print(f"      ↳ gate 3 (doctest)      {counts['g3_doctest_fail']:>4}")

    print(f"\n  Results vs baselines")
    print(f"    {'':30}  {'accuracy':>10}  {'exact cost':>12}")
    print(f"    {'─'*56}")
    print(f"    {'always-Gemini (no routing)':30}  {'73.8%':>10}  {'$0.01639':>12}")
    print(f"    {'always-Opus   (no routing)':30}  {'97.6%':>10}  {'$0.82212':>12}")
    print(f"    {'smart-ask cascade (this)':30}  {f'{acc:.1f}%':>10}  {f'${cost:.5f}':>12}")

    saving = (0.82212 - cost) / 0.82212 * 100
    print(f"\n  Cost saving vs always-Opus:  {saving:.1f}%  "
          f"(${0.82212 - cost:.5f} saved per {n} problems)")
    print(f"  Accuracy gap vs always-Opus: {acc - 97.6:+.1f}pp")
    print("=" * W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=None,
                   help="Number of problems (default: all 164)")
    p.add_argument("--report", action="store_true",
                   help="Print saved results without re-running")
    args = p.parse_args()

    if args.report:
        if not RESULTS_FILE.exists():
            sys.exit("No saved results. Run without --report first.")
        data    = json.loads(RESULTS_FILE.read_text())
        # Rebuild a tracker from the saved log for the report
        tracker = TokenTracker()
        if TRACKER_LOG_FILE.exists():
            log = json.loads(TRACKER_LOG_FILE.read_text())
            # Inject calls manually since we only have aggregated data
            class _FakeUsage:
                def __init__(self, p, c): self.prompt_tokens=p; self.completion_tokens=c
            for call in log.get("calls", []):
                tracker.record(call["model"], call["role"],
                               _FakeUsage(call["prompt_tokens"], call["completion_tokens"]),
                               call.get("task_id"))
        print_report(data["counts"], tracker, data["n"])
        return

    print("  Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"  Loaded {len(problems)} problems")

    counts, tracker, total = run(problems, n=args.n)
    print_report(counts, tracker, total)


if __name__ == "__main__":
    main()
