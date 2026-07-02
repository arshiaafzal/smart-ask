# smart-ask

`smart-ask` is a configurable terminal router for language-model tasks. A YAML
strategy chooses a routing method, models, prompts, generation settings, and
execution transports; the CLI loads that strategy and runs each task through
the resulting application.

The shipped product strategy classifies a task with Gemini 2.5 Flash Lite,
routes easy work to Gemini and hard work to Claude Opus, and hands generation
to Hermes:

```text
$ smart-ask "explain how a TCP handshake works"
  ▸  gemini-2.5-flash-lite  [easy]  classified easy
  ↳  Hermes

$ smart-ask "design a distributed event-sourcing architecture"
  ▸  claude-opus-4.8        [hard]  classified hard
  ↳  Hermes
```

These are defaults from [`strategies/product.yaml`](strategies/product.yaml),
not model choices embedded in the CLI.

## How it works

```text
strategies/product.yaml
  → load_strategy(): validate YAML, resolve prompt files, compute digest
  → StrategyBuilder: construct method, collaborators, and executors
  → SmartAsk(method, generation executor)
  → route and execute each Task
```

The default product flow is:

```text
Task
  → DifficultyRoutingMethod
      → LLMDifficultyClassifier
          → classifier OpenRouterExecutor → easy or hard
      → RouteResult with the selected model
  → SmartAsk
      → generation HermesExecutor → Hermes → selected model
```

Classification and generation use separate executor instances. Classification
needs a short captured response; the default product generation path delegates
the selected task to Hermes. A strategy may instead configure direct
OpenRouter generation, as the shipped direct-execution strategies do.

“Method” and “strategy” have distinct meanings in the codebase:

- A `RoutingMethod` is the runtime routing algorithm, such as difficulty,
  cascade, or fixed routing.
- A `StrategyConfig` is the complete reproducible YAML composition: method,
  classifier and escalation collaborators, model profiles, prompt sources,
  executors, parameters, and attempt limit.

## Requirements and installation

- Python 3.11+
- Hermes for strategies whose generation executor is `hermes`
- An OpenRouter API key for any OpenRouter classifier or generation executor

The project declares its runtime dependencies—OpenAI, Pydantic, and PyYAML—in
[`pyproject.toml`](pyproject.toml). From the repository root, install the exact
`smart-ask` executable plus the bundled strategies and prompts:

```bash
# Install the product runtime and command.
python3.11 -m pip install .

# Use an editable install while developing this checkout.
python3.11 -m pip install -e .

# Include benchmark dataset support when needed (editable form shown).
python3.11 -m pip install -e '.[bench]'
```

Configure the credential named by the strategy, which is
`OPENROUTER_API_KEY` in all shipped OpenRouter configurations:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

If that environment's `bin` directory is on `PATH`, verify the installed
command and bundled default strategy:

```bash
smart-ask --validate-strategy
```

An optional short alias is fine: `alias ask=smart-ask`.

## Product CLI

```text
smart-ask                                      prompt for independent tasks
smart-ask "task"                               route and execute one task
smart-ask -f FILE "task"                       prepend one or more files
smart-ask --strategy FILE "task"               use a strategy YAML
smart-ask --validate-strategy --strategy FILE   validate without credentials or calls
smart-ask --force-easy "task"                  use the configured easy profile
smart-ask --force-hard "task"                  use the configured hard profile
smart-ask --dry-run "task"                     plan and print the route only
smart-ask --help                                show the configured welcome/help screen
```

The default is the bundled `product.yaml`: the executable finds it in either a
source checkout or the installation data directory. Force flags replace the
configured method with a one-shot fixed method while retaining the selected
model profile and generation transport. In the shipped product strategy,
forcing a route also skips the classifier call.

The outer prompt remains open after a task, but each entry is an independent
query. Type `/exit` or `/quit`, or press Ctrl-D/Ctrl-C, to stop.

Classifier token usage is still shown when a configured model has no entry in
the local price catalog; its monetary cost is reported explicitly as unknown.

## Strategy YAML

A strategy file has schema version 1 and exactly one root `StrategyConfig`:

```yaml
schema_version: 1
name: product-difficulty-v1
method:
  type: difficulty
  classifier:
    type: llm
    model: google/gemini-2.5-flash-lite
    executor:
      type: openrouter
    prompt:
      type: file
      path: ../prompts/difficulty-v1.txt
    max_prompt_chars: 1200
    parameters:
      max_tokens: 20
      temperature: 0.0
  easy:
    model: google/gemini-2.5-flash-lite
  hard:
    model: anthropic/claude-opus-4.8
generation:
  type: hermes
  provider: openrouter
  command: hermes
max_attempts: 1
```

Supported method types are `difficulty`, `cascade`, and `fixed`. The supported
collaborators are an `llm` difficulty classifier and `marker` escalation
policy. Generation and classification can use `openrouter`; one-shot generation
can also use `hermes`. Model profiles can supply system prompts, maximum output
tokens, and temperature where the selected executor supports them.

Prompt-file paths are resolved relative to the strategy file. Loading rejects
duplicate keys, unknown fields, invalid compositions, missing or empty prompts,
and incompatible executor capabilities. The loaded strategy digest includes the
typed config and prompt contents, so prompt edits change its identity.

Shipped configurations include:

- `strategies/product.yaml` — product difficulty method with Hermes generation
- `strategies/python-function-completion-difficulty-v1.yaml` and
  `python-function-completion-difficulty-v2.yaml` for paired classifier-prompt
  comparison
- `strategies/python-function-completion-cascade.yaml` and
  `python-function-completion-fixed-opus.yaml`
- `strategies/python-code-generation-cascade.yaml` and
  `python-code-generation-fixed-opus.yaml`

Reusable strategy and prompt names describe their task/output contract rather
than the benchmark that happens to exercise them. The same strategy can be
passed to any compatible benchmark suite or application entrypoint.

## Benchmarking strategies

HumanEval and LiveBench are module-based benchmark applications. Repeat
`--strategy` to run the same case set against several configurations:

```bash
python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-difficulty-v1.yaml \
  --strategy strategies/python-function-completion-difficulty-v2.yaml \
  --limit 20 \
  --workers 4 \
  --output benchmarks/results/humaneval/prompt-comparison

python -m benchmarks.livebench \
  --strategy strategies/python-code-generation-cascade.yaml \
  --strategy strategies/python-code-generation-fixed-opus.yaml \
  --workers 4
```

Each run produces schema-v3 evidence:

```text
<output>/
├── manifest.json   dataset/code/pricing identity and strategy snapshots
├── records.jsonl   one append-safe record per strategy/task pair
└── summary.json    aggregate metrics and paired comparisons
```

Every task record retains routing events, attempts, classifier and generation
calls, provider-neutral requests, raw and normalized outputs, usage, cost,
latency, evaluation, errors, and timestamps. `--resume` continues an explicit
`--output` directory only when its suite, strategies, cases, pricing, workers,
code/runtime identity, and dependency versions still match. When `--output` is
omitted, a timestamped directory is created under
`benchmarks/results/<suite>/`.

The benchmark CLI owns run controls such as suite, case limit, workers, output,
and resume. There is no separate `ExperimentConfig`; a comparison is a fixed
suite plus one or more repeatable strategy YAMLs and those CLI controls.

The correctness harnesses execute model-generated Python in local subprocesses.
A timeout limits duration but is not a security sandbox; run benchmarks in an
isolated environment when model output is untrusted.

See [`benchmarks/humaneval/README.md`](benchmarks/humaneval/README.md) for the
HumanEval workflow and [`DESIGN.md`](DESIGN.md) for component boundaries.

## Project structure

```text
smart-ask/
├── smart-ask                     product CLI
├── smart_ask/
│   ├── application.py            SmartAsk coordinator
│   ├── domain.py                 immutable per-task values
│   ├── methods/                  runtime routing methods and collaborators
│   │   ├── base.py               RoutingMethod protocol
│   │   ├── difficulty.py
│   │   ├── cascade.py
│   │   ├── fixed.py
│   │   ├── classifiers/
│   │   └── escalation/
│   ├── executors/                Hermes and OpenRouter adapters
│   └── strategy/                 YAML schema, loader, and builder
├── strategies/                   shipped StrategyConfig YAML files
├── prompts/                      versioned prompt text
├── benchmarks/                   suites, matrix runner, artifacts, comparison
├── harness/                      generated-code execution
├── cost/                         model price catalog and accounting
├── tests/                        network-free tests
└── pyproject.toml                package and dependency metadata
```

## License

MIT
