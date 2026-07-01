# HumanEval Benchmark

Measures **pass@1 accuracy** and **cost** for the two smart-ask routes:

| Route | Model |
|-------|-------|
| Easy  | `google/gemini-2.5-flash-lite` |
| Hard  | `anthropic/claude-opus-4.8` |

HumanEval is 164 real Python programming problems from OpenAI.
Each problem has a function signature + docstring, and the model must complete the body.
Correctness is verified by running the model's code against the problem's test suite.

---

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
```

## Run

```bash
# Quick test — first 20 problems (~2 min)
python run.py -n 20

# Full benchmark — all 164 problems (~15 min)
python run.py

# Print last saved results without re-running
python run.py --report
```

## Results (full run — 164 problems)

```
  Running 164 HumanEval problems against 2 models...

  ════════════════════════════════════════════════════════════
  HumanEval Benchmark  —  smart-ask cost comparison
  ════════════════════════════════════════════════════════════

  Google Gemini 2.5 Flash Lite  [easy route]
    pass@1      121/164  (73.8%)
    total cost  $0.01639
    cost/solved $0.000135
    tokens      30,412 in  /  33,365 out

  Anthropic Claude Opus 4.8 [hard route]
    pass@1      160/164  (97.6%)
    total cost  $0.82212
    cost/solved $0.005138
    tokens      41,843 in  /  24,516 out

  ────────────────────────────────────────
  Cost saving   98.0%  (50.2x cheaper)
  Gemini cost   $0.01639
  Opus cost     $0.82212
  Difference    $0.80573 saved per 164 problems
  ════════════════════════════════════════════════════════════
```

## What it proves

- For **easy coding tasks**, Gemini 2.5 Flash Lite handles ~74% of them correctly at 1/50th the cost of Opus
- smart-ask routes easy tasks to Gemini → you only pay Opus prices for tasks that genuinely need it
- Even with Gemini's lower accuracy, the cost-per-correct-solution is **38x cheaper** ($0.000135 vs $0.005138)
- For tasks where quality matters most, use `--force-hard` to always route to Opus
