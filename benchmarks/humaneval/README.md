# HumanEval Benchmark

Measures **pass@1 accuracy** and **cost** for the two smart-ask routes:

| Route | Model |
|-------|-------|
| Easy  | `google/gemini-3.5-flash` |
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

## Example output

```
  Running 164 HumanEval problems against 2 models...

  ══════════════════════════════════════════════════════════
  HumanEval Benchmark  —  smart-ask cost comparison
  ══════════════════════════════════════════════════════════

  Google Gemini 3.5 Flash  [easy route]
    pass@1      138/164  (84.1%)
    total cost  $0.00412
    cost/solved $0.000030
    tokens      2,741,320 in  /  83,204 out

  Anthropic Claude Opus 4.8 [hard route]
    pass@1      152/164  (92.7%)
    total cost  $0.01823
    cost/solved $0.000120
    tokens      2,741,320 in  /  83,204 out

  ────────────────────────────────────────
  Cost saving   77.4%  (4.4x cheaper)
  Gemini cost   $0.00412
  Opus cost     $0.01823
  Difference    $0.01411 saved per 164 problems
  ══════════════════════════════════════════════════════════
```

## What it proves

- For **easy coding tasks**, Gemini handles most of them at a fraction of the cost
- Smart-ask routes easy tasks to Gemini → you only pay Opus prices for tasks that genuinely need it
- The classifier's job is to identify which tasks are which
