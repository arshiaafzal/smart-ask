#!/usr/bin/env python3
"""
Always-Opus baseline for LiveBench coding.

Re-uses Opus results already in results_product.json (G1-hard problems),
runs Opus on the G1-easy problems we only have Gemini results for.
Total: 128 Opus calls, ~59 new API calls.

This benchmark contains ONLY evaluation logic.
All model code is imported from methods/.
All cost tracking is imported from cost/.
"""

import os, sys, json, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from cost import TokenTracker
from methods.cascade import OR_BASE, HARD_MODEL, call_hard
from benchmarks.livebench.run_product import load_livebench, run_tests

OR_KEY       = os.environ.get("OPENROUTER_API_KEY", "")
RESULTS_FILE = Path(__file__).parent / "results_opus_baseline.json"
WORKERS      = 6


# ── Per-problem runner ─────────────────────────────────────────────────────────

def run_problem(client, prob: dict, tracker: TokenTracker) -> dict:
    qid    = prob["question_id"][:16]
    prompt = prob["prompt"]

    code, usage = call_hard(prompt, client)
    tracker.record(HARD_MODEL, "writer", usage, qid)

    if prob["task"] == "coding_completion" and prob["partial"]:
        full_code = code if "class Solution" in code else prob["partial"] + "\n" + code
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not OR_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set")

    saved_path = Path(__file__).parent / "results_product.json"
    if not saved_path.exists():
        sys.exit("Run run_product.py first to generate results_product.json")

    saved       = json.loads(saved_path.read_text())
    saved_by_id = {r["question_id"]: r for r in saved["results"]}

    print("  Loading LiveBench coding dataset...")
    problems = load_livebench()
    print(f"  Loaded {len(problems)} problems\n")

    # Reuse problems already run with Opus; run fresh for the rest
    opus_cached = {qid: r for qid, r in saved_by_id.items()
                   if "opus" in r.get("model", "")}
    needs_opus  = [p for p in problems if p["question_id"] not in opus_cached]

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

    # Merge cached + new results; label all as opus
    all_results = list(opus_cached.values()) + new_results
    for r in all_results:
        r["model"] = "opus"

    total  = len(all_results)
    passed = sum(1 for r in all_results if r["pass_all"])

    print(f"\n{'='*64}")
    print(f"  Always-Opus Baseline — LiveBench Coding")
    print(f"{'='*64}")
    print(f"  pass@1   {passed}/{total}  ({passed/total*100:.1f}%)")
    print(f"  new API calls: {tracker.n_calls()}  "
          f"(reused {len(opus_cached)} cached Opus results)")

    print(f"\n  By difficulty")
    for diff in ("easy", "medium", "hard"):
        n  = sum(1 for r in all_results if r.get("difficulty") == diff)
        ok = sum(1 for r in all_results if r.get("difficulty") == diff and r["pass_all"])
        if n:
            print(f"    {diff:<8}  {ok}/{n}  ({ok/n*100:.0f}%)")

    if tracker.n_calls() > 0:
        tracker.report("New Opus calls — token & cost breakdown")

    RESULTS_FILE.write_text(json.dumps({
        "n": total, "passed": passed,
        "results":   all_results,
        "token_log": json.loads(tracker.export_json()),
    }, indent=2))
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
