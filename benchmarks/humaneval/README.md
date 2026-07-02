# HumanEval Benchmark

The HumanEval benchmark applies one or more complete smart-ask strategy YAMLs
to the same pinned set of 164 Python function-completion tasks. The suite runs
each final response against the task's tests and records pass@1, score, routing,
model calls, tokens, priced cost, and latency.

The dataset identity is pinned in `benchmarks/humaneval/suite.py`, including its
revision, and is captured in every run manifest.

## Setup

From the repository root:

```bash
python -m pip install -e '.[bench]'
export OPENROUTER_API_KEY="sk-or-..."
```

Benchmark strategies must use a generation executor that captures response
text. The shipped Python function-completion configurations therefore use
direct OpenRouter generation rather than Hermes. Their names describe that
reusable task contract, not this benchmark suite.

The evaluator executes generated Python in a local subprocess. Its timeout is
not a security sandbox, so use an isolated environment for untrusted output.

## Run one strategy

```bash
python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-cascade.yaml \
  --limit 20 \
  --workers 4
```

Omit `--limit` for all 164 cases. Without `--output`, the runner creates a
timestamped directory under `benchmarks/results/humaneval/`.

## Compare strategies

`--strategy` is repeatable. Every strategy receives the same ordered case set,
and the final report includes paired outcomes rather than comparing unrelated
runs.

Compare two classifier prompts:

```bash
python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-difficulty-v1.yaml \
  --strategy strategies/python-function-completion-difficulty-v2.yaml \
  --workers 4 \
  --output benchmarks/results/humaneval/prompt-v1-v2
```

Compare the cascade with an always-Opus baseline:

```bash
python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-cascade.yaml \
  --strategy strategies/python-function-completion-fixed-opus.yaml \
  --workers 4 \
  --output benchmarks/results/humaneval/cascade-vs-opus
```

The runner requires unique strategy names within one matrix and known prices
for every configured model before it makes any API call.

## Resume an interrupted run

`--resume` requires an explicit output directory:

```bash
python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-difficulty-v1.yaml \
  --strategy strategies/python-function-completion-difficulty-v2.yaml \
  --workers 4 \
  --output benchmarks/results/humaneval/prompt-v1-v2 \
  --resume
```

Resume skips completed `(strategy_id, task_id)` pairs only when the saved
manifest still matches the benchmark, pinned dataset, strategies and prompt
digests, case set, pricing, worker count, dependency/runtime versions, and code
identity.

## Evidence and comparison output

Each run directory uses benchmark artifact schema version 3:

```text
<output>/
├── manifest.json
├── records.jsonl
└── summary.json
```

- `manifest.json` records the dataset revision and case hashes, full strategy
  snapshots and prompt hashes, pricing, Python runtime, and git identity.
- `records.jsonl` contains one append-safe record per strategy/task pair. A
  record includes the input, selected route, classifier decision, routing
  events, attempts, every classifier/generation call, requests, raw and cleaned
  outputs, usage, cost, latency, evaluation, errors, and timestamps.
- `summary.json` contains per-strategy aggregates and all paired comparisons.

Records are flushed and synced as they complete, which makes the JSONL file the
crash-safe evidence log. Comparisons report both-pass, only-reference,
only-candidate, neither-pass, and missing counts, plus per-task score, cost, and
latency deltas where available. Missing evidence remains explicit.

Strategy YAML itself uses `schema_version: 1`; that is separate from benchmark
artifact schema version 3.

There is no `ExperimentConfig`. The benchmark module owns suite, limit, worker,
output, and resume options, while each repeated `--strategy` names one root
`StrategyConfig` YAML.

## Compatibility wrapper

`python benchmarks/humaneval/run_product.py` remains as a compatibility command
that supplies `strategies/python-function-completion-cascade.yaml` when no
strategy is given. Its `--report` option reads the checked-in legacy JSON
through the compatibility loader. New comparisons should use
`python -m benchmarks.humaneval` and schema-v3 run directories.
