# HumanEval benchmark

HumanEval runs one or more complete SmartAsk strategies over the same pinned
164-task dataset and evaluates each final answer with the task's tests.

## Safety

The evaluator executes model-generated Python in a local subprocess. A timeout
is not a security sandbox. Run it only in an isolated environment and opt in
explicitly with `--allow-unsafe-code-execution`.

## Setup and run

```bash
python -m pip install -e '.[bench]'
export OPENROUTER_API_KEY="sk-or-..."

python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --allow-unsafe-code-execution \
  --limit 20 \
  --workers 4
```

Omit `--limit` for all cases. `--strategy` is repeatable, so routed, cheap-only,
and hard-only policies can be run over the exact same case set:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --strategy builtin:python-function-completion-fixed-gemini-self-check \
  --strategy builtin:python-function-completion-fixed-opus \
  --allow-unsafe-code-execution \
  --workers 4 \
  --output benchmark-results/humaneval/cascade-counterfactuals
```

Use `--price-catalog catalog.json` to replace the bundled versioned price
snapshot.

## Resume

`--resume` requires the same explicit output directory and an identical
manifest:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --allow-unsafe-code-execution \
  --output benchmark-results/humaneval/cascade-counterfactuals \
  --resume
```

The manifest binds the pinned dataset/evaluator, case hashes, schema-v3
strategy and prompt digests, resolved target fingerprints, pricing, and worker
count. A directory lock prevents concurrent writers.

## Artifacts

Benchmark artifact schema v2 writes:

```text
<output>/
├── manifest.json
├── records.jsonl
└── summary.json
```

- `manifest.json`: reproducibility identities and resolved target fingerprints.
- `records.jsonl`: one crash-safe canonical run ledger per strategy/task pair,
  including partial call evidence for failed executions.
- `summary.json`: mutually exclusive task outcomes, resource rollups by model,
  target, profile, and role, routing paths, and paired comparisons.

Task outcomes are `passed`, `incorrect`, `execution_error`,
`evaluation_error`, or `unrated`. Missing tokens or cost remain explicit.
