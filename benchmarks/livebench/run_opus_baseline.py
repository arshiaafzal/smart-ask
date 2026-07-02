#!/usr/bin/env python3
"""
Always-Opus baseline for LiveBench coding.

Re-uses Opus results already in results_product.json (G1-hard problems),
runs Opus on the G1-easy problems we only have Gemini results for.
Total: 128 Opus calls, ~59 new API calls.
"""

import os, sys, json, re, subprocess, tempfile, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from methods.cascade import OR_BASE, HARD_MODEL
from benchmarks.livebench.run_product import (
    load_livebench, run_tests, strip_fences,
)
from openai import OpenAI

OR_KEY       = os.environ.get("OPENROUTER_API_KEY", "")
RESULTS_FILE = Path(__file__).parent / "results_opus_baseline.json"
WORKERS      = 6

OPUS_SYSTEM = (
    "You are an expert competitive programmer. "
    "Write correct, complete Python code. "
    "Return ONLY the code — no explanation, no markdown fences, no extra text."
)


def run_opus(client, prompt: str, tracker, task_id: str) -> str:
    r = client.chat.completions.create(
        model=HARD_MODEL,
        messages=[
            {"role": "system", "content": OPUS_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024, temperature=0.0,
    )
    tracker.record(HARD_MODEL, "writer", r.usage, task_id)
    return strip_fences(r.choices[0].message.content or "")


def run_problem(client, prob: dict, tracker) -> dict:
    qid    = prob["question_id"][:16]
    prompt = prob["prompt"]
    code   = run_opus(client, prompt, tracker, qid)

    if prob["task"] == "coding_completion" and prob["partial"]:
        if "class Solution" in code:
            full_code = code
        else:
            full_code = prob["partial"] + "\n" + code
    else:
        full_code = code

    passed, total = run_tests(full_code, prob["starter_code"], prob["test_cases"])
    return {
        "question_id": prob["question_id"],
        "title":       prob["title"],
        "task":        prob["task"],
        "difficulty":  prob["difficulty"],
        "model":       "opus",
        "passed":      passed,
        "total":       total,
        "pass_all":    passed == total and total > 0,
    }


def main():
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    # Load saved product results — reuse Opus results, only run Gemini ones
    saved_path = Path(__file__).parent / "results_product.json"
    if not saved_path.exists():
        sys.exit("Run run_product.py first to generate results_product.json")

    saved = json.loads(saved_path.read_text())
    saved_by_id = {r["question_id"]: r for r in saved["results"]}

    print("  Loading LiveBench coding dataset...")
    problems = load_livebench()
    print(f"  Loaded {len(problems)} problems\n")

    # Problems already run with Opus in product benchmark
    opus_cached = {
        qid: r for qid, r in saved_by_id.items()
        if "opus" in r.get("model", "")
    }
    # Problems that need new Opus calls (were Gemini in product run)
    needs_opus = [p for p in problems if p["question_id"] not in opus_cached]

    print(f"  Reusing {len(opus_cached)} Opus results from product benchmark")
    print(f"  Running {len(needs_opus)} new Opus calls...\n")

    client  = OpenAI(base_url=OR_BASE, api_key=OR_KEY)
    tracker = TokenTracker()
    new_results = []
    lock   = threading.Lock()
    done_n = [0]

    print(f"  {'#':>5}  {'pass':>8}  title")
    print(f"  {'─'*60}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_problem, client, p, tracker): p for p in needs_opus}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                r = fut.result()
            except Exception:
                r = {
                    "question_id": p["question_id"], "title": p["title"],
                    "task": p["task"], "difficulty": p["difficulty"],
                    "model": "opus", "passed": 0,
                    "total": len(p["test_cases"]), "pass_all": False,
                }
            with lock:
                new_results.append(r)
                done_n[0] += 1
                sym = "✓" if r["pass_all"] else "✗"
                print(f"  {done_n[0]:>5}  {sym} {r['passed']}/{r['total']}     {r['title'][:45]}")

    # Merge: cached Opus + new Opus results
    all_results = list(opus_cached.values()) + new_results
    # Convert cached results to opus model label
    for r in all_results:
        r["model"] = "opus"

    # Report
    total  = len(all_results)
    passed = sum(1 for r in all_results if r["pass_all"])
    cost   = tracker.total_cost()

    print(f"\n{'='*64}")
    print(f"  Always-Opus Baseline — LiveBench Coding")
    print(f"{'='*64}")
    print(f"  pass@1   {passed}/{total}  ({passed/total*100:.1f}%)")
    print(f"  new cost ${cost:.5f}  (reused {len(opus_cached)} cached Opus calls)")

    print(f"\n  By difficulty")
    for diff in ("easy", "medium", "hard"):
        n  = sum(1 for r in all_results if r.get("difficulty") == diff)
        ok = sum(1 for r in all_results if r.get("difficulty") == diff and r["pass_all"])
        if n:
            print(f"    {diff:<8}  {ok}/{n}  ({ok/n*100:.0f}%)")

    if tracker.n_calls() > 0:
        tracker.report("New Opus calls — token breakdown")

    RESULTS_FILE.write_text(json.dumps({
        "n": total, "passed": passed,
        "results": all_results,
        "token_log": json.loads(tracker.export_json()),
    }, indent=2))
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
