# HumanEval Benchmark

The HumanEval benchmark applies one or more complete smart-ask strategy YAMLs
to the same pinned set of 164 Python function-completion tasks. The suite runs
each final response against the task's tests and records pass@1, score, routing,
model calls, tokens, priced cost, and latency.

The dataset identity is pinned in `smart_ask/benchmarks/humaneval/suite.py`, including its
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
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --limit 20 \
  --workers 4
```

Omit `--limit` for all 164 cases. Without `--output`, the runner creates a
timestamped directory under `benchmark-results/humaneval/`.

## Compare strategies

`--strategy` is repeatable. Every strategy receives the same ordered case set,
and the final report includes paired outcomes rather than comparing unrelated
runs.

Compare two classifier prompts:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-difficulty-v1 \
  --strategy builtin:python-function-completion-difficulty-v2 \
  --workers 4 \
  --output benchmark-results/humaneval/prompt-v1-v2
```

Compare the cascade with exact cheap-only and expensive-only baselines:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --strategy builtin:python-function-completion-fixed-gemini-self-check \
  --strategy builtin:python-function-completion-fixed-opus \
  --workers 4 \
  --output benchmark-results/humaneval/cascade-counterfactuals
```

The cheap baseline repeats the cascade's self-check suffix exactly, so its
observed output is a like-for-like counterfactual candidate.

The runner requires unique strategy names within one matrix and known prices
for every configured model before it makes any API call.
Pass `--price-catalog catalog.json` to use another explicit, versioned catalog
snapshot.

## Resume an interrupted run

`--resume` requires an explicit output directory:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-difficulty-v1 \
  --strategy builtin:python-function-completion-difficulty-v2 \
  --workers 4 \
  --output benchmark-results/humaneval/prompt-v1-v2 \
  --resume
```

Resume skips completed `(strategy_id, task_id)` pairs only when the saved
manifest still matches the benchmark, pinned dataset, strategies and prompt
digests, evaluator, case set, pricing, metrics schema, worker count,
dependency/runtime versions, and code identity. A directory lock prevents two
runners from appending concurrently.

## Evidence and comparison output

Each run directory uses benchmark artifact schema version 5:

```text
<output>/
├── manifest.json
├── records.jsonl
└── summary.json
```

- `manifest.json` records the dataset and evaluator identities, case hashes,
  full strategy snapshots and prompt hashes, metrics/pricing provenance, Python
  runtime, installed package hash/version, and matching-checkout Git identity.
- `records.jsonl` contains one append-safe record per strategy/task pair. A
  record includes the input, selected route, classifier decision, routing
  events, attempts, every classifier/generation call, requests, provider
  outputs, one canonical per-run `metrics` envelope, evaluation, errors, and
  timestamps. Calls own usage, cost, and latency; attempts and events reference
  call IDs instead of duplicating those values.
- `summary.json` contains per-strategy outcome/resource aggregates, wall-clock
  record span and cumulative timing, routing transitions and complete paths,
  all paired comparisons, and counterfactual routing diagnostics when exact
  fixed easy/hard profile baselines are present. Exact matching includes the
  role, resolved prompts and user-prompt transforms, tuning, and executor.

Records are flushed and synced as they complete, which makes the JSONL file the
crash-safe evidence log. Comparisons report rated and excluded pairs,
both-pass, only-reference, only-candidate, neither-pass, and missing counts,
plus per-task score, cost, and latency deltas where available. Task errors are
excluded from quality rates but retain their resource evidence. Missing
evidence remains explicit.

Strategy YAML itself uses `schema_version: 2`; that is separate from benchmark
artifact schema version 5 and the metrics-v2 envelope.

There is no `ExperimentConfig`. The benchmark module owns suite, limit, worker,
output, and resume options, while each repeated `--strategy` names one root
`StrategyConfig` YAML.

Pre-current-schema checked-in results are isolated under `benchmark-history/`;
their README records provenance and evidence gaps. Runtime readers
intentionally do not normalize those formats.
