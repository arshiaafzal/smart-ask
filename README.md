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

These are defaults from the bundled
[`product.yaml`](smart_ask/resources/strategies/product.yaml), not model choices
embedded in the CLI.

## How it works

```text
smart_ask/resources/strategies/product.yaml
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
  executors, and parameters. The builder derives the closed method's attempt
  bound.

## Requirements and installation

- Python 3.11+
- Hermes for strategies whose generation executor is `hermes`
- An OpenRouter API key for any OpenRouter classifier or generation executor
- An OpenAI API key for any first-party OpenAI classifier or generation executor
- A Groq API key for any Groq classifier or generation executor

The project declares its runtime dependencies—OpenAI, Pydantic, and PyYAML—in
[`pyproject.toml`](pyproject.toml). From the repository root, install the exact
`smart-ask` executable plus the bundled strategies and prompts:

```bash
# Install the product runtime and command.
python3.11 -m pip install .

# Use an editable install while developing this checkout.
python3.11 -m pip install -e .

# In a checkout, install the optional dataset dependency for benchmark tooling.
python3.11 -m pip install -e '.[bench]'

# Install the separately packaged Claude Code adapter when you need it.
python3.11 -m pip install -e ./integrations/claude_code
```

Configure the credential named by the strategy, which is
`OPENROUTER_API_KEY` in OpenRouter configurations or `OPENAI_API_KEY` in
first-party OpenAI configurations. Groq configurations use `GROQ_API_KEY`:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

If that environment's `bin` directory is on `PATH`, verify the installed
command and bundled default strategy:

```bash
smart-ask --validate-strategy
```

An optional short alias is fine: `alias ask=smart-ask`.

## Claude Code adapter

Claude Code support is not part of the `smart_ask` package. The separately
installable `smart-ask-claude-code` adapter consumes SmartAsk's public,
harness-neutral conversation runtime:

```text
Claude Code
  -> external protocol adapter
  -> SmartAsk conversation runtime
  -> strategy YAML
  -> configured generation executor
```

The dependency only points inward: `smart-ask-claude-code` depends on
`smart-ask`; nothing under `smart_ask/` imports or implements the external
protocol. The adapter translates requests and streaming events. SmartAsk loads
the YAML, routes the turn, executes the selected backend, handles escalation,
and records usage.

Each exposed model maps one-to-one to one strategy YAML. Its public alias comes
from the YAML filename, while the validated strategy name and digest remain in
SmartAsk metrics:

```text
claude-smart-ask-python-code-generation-cascade
  -> builtin:python-code-generation-cascade

claude-smart-ask-python-code-generation-fixed-opus
  -> builtin:python-code-generation-fixed-opus

claude-smart-ask-python-code-generation-codex-cascade
  -> builtin:python-code-generation-codex-cascade
```

Install both packages from a checkout, make sure Ollama is running, and copy or
edit [`claude-code-adapter.example.yaml`](claude-code-adapter.example.yaml):

```bash
python3.11 -m pip install -e .
python3.11 -m pip install -e ./integrations/claude_code

ollama pull qwen3:14b
ollama serve

export SMART_ASK_CLAUDE_CODE_TOKEN="local-secret"
smart-ask-claude-code serve --config claude-code-adapter.example.yaml
```

In another shell, point Claude Code at the adapter and select the strategy
alias:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY="$SMART_ASK_CLAUDE_CODE_TOKEN"

claude --model claude-smart-ask-local-qwen
```

The example exposes `builtin:local-qwen`, so this path needs no OpenRouter key.
An OpenRouter credential is required only when the selected YAML configures an
OpenRouter classifier or generation executor.

For the first-party Codex cascade, use
[`claude-code-openai-codex.example.yaml`](claude-code-openai-codex.example.yaml),
set `OPENAI_API_KEY`, and select
`claude-smart-ask-python-code-generation-codex-cascade`. The strategy uses
`gpt-5.1-codex-mini` for classification and easy work, then
`gpt-5.3-codex` for hard work and escalations. OpenAI currently marks the mini
model deprecated; this strategy intentionally retains it as the requested old,
small Codex tier.

The general launcher generates that adapter configuration automatically:

```bash
# Put OPENAI_API_KEY in the gitignored scripts/claude-smart-ask.local.env.
./scripts/claude-smart-ask \
  --strategy python-code-generation-codex-cascade
```

Any additional arguments are passed to Claude Code, for example
`-p "implement this function"`.

The adapter implements messages, streaming, token counting, model discovery,
authentication, request limits, and concurrency limits. It preserves tools,
images, thinking blocks, session correlation headers, and unknown fields in the
neutral request. Fixed, difficulty, and cascade behavior stays in SmartAsk.
Difficulty and cascade decisions remain sticky through one turn's tool loop;
the next human instruction is routed again. A cascade streams tool calls
immediately and only buffers a cheap attempt's final text while SmartAsk decides
whether to accept or escalate it.

Set `metrics.jsonl_path` in the adapter YAML to append one prompt-free SmartAsk
metrics envelope after every request. Each line contains the run and current
session aggregate: strategy identity, route path, selected and actual models,
tokens, estimated/provider cost, timing, tool-call counts, completion, errors,
and cancellation. Conversation text and tool arguments are not persisted.

For local development, the [launcher](scripts/claude-local-qwen) starts Ollama
and the adapter in the background, waits for both, and then invokes Claude Code.
See [scripts/README.md](scripts/README.md) for its encapsulation boundaries and
process-ownership model. Its first run automatically creates a private Python
environment and installs both checkout packages; manual adapter installation is
not required.

```bash
./scripts/claude-local-qwen                 # interactive
./scripts/claude-local-qwen -p "your task"  # one-shot
./scripts/claude-local-qwen status
./scripts/claude-local-qwen stop
```

It only stops processes that it started. Logs, PIDs, and its generated local
adapter token live under `${TMPDIR:-/tmp}/smart-ask-claude-local-qwen`.

## Product CLI

```text
smart-ask                                      prompt for independent tasks
smart-ask "task"                               route and execute one task
smart-ask -f FILE "task"                       prepend one or more files
smart-ask --strategy FILE "task"               use a strategy YAML
smart-ask --strategy builtin:NAME "task"       use an installed bundled strategy
smart-ask --validate-strategy --strategy FILE   validate without credentials or calls
smart-ask --force-easy "task"                  use the configured easy profile
smart-ask --force-hard "task"                  use the configured hard profile
smart-ask --dry-run "task"                     classify/plan; skip generation
smart-ask --help                                show the configured welcome/help screen
```

The default is the `product.yaml` package resource installed with `smart_ask`.
Force flags replace the configured method with a one-shot fixed method while
retaining the selected model profile and generation transport. In the shipped
product strategy, forcing a route also skips the classifier call.

The outer prompt remains open after a task, but each entry is an independent
query. Type `/exit` or `/quit`, or press Ctrl-D/Ctrl-C, to stop.

Classifier token usage is still shown when a configured model has no entry in
the local price catalog; its monetary cost is reported explicitly as unknown.

### Per-turn metrics

Every application built by `StrategyBuilder` can return an immutable metrics
snapshot for one task or conversation turn:

```python
from smart_ask import (
    StrategyBuilder,
    Task,
    aggregate_resources,
    aggregate_stats,
)

app = StrategyBuilder().build(loaded_strategy)
answer, turn_stats = app.run_with_stats(Task("hello", task_id="turn-1"))

print(turn_stats.interaction_count)
print(turn_stats.total_tokens)       # None when any call lacks usage
print(turn_stats.total_cost_usd)     # catalog estimate; None if incomplete
print(turn_stats.total_provider_cost_usd)  # provider charge, when reported
print(turn_stats.outcome)            # "unrated" for a normal conversation

session = aggregate_stats([turn_stats])
resources = aggregate_resources([turn_stats]).to_dict()
print(session.outcome_counts)
print(resources["by_model"])
```

The snapshot contains one `CallStats` entry per classifier or generation call,
including its run identity, run-local call ID, semantic role,
requested/actual/priced model, latency, token evidence, pricing provenance,
normalized finish reason and output status, orthogonal captured-output emptiness,
requested and adapter-applied token caps, and any
telemetry diagnostic. Successful transport and successful task completion are
separate: an HTTP-successful response can still be empty, truncated, refused,
or unusable. Empty captured text is counted even when the response also ended
by length or refusal. An executor that cannot capture its response reports
output emptiness as unknown and status as `unavailable`, while retaining any
independently observed finish reason.
An interaction means one logical `ModelExecutor.execute` invocation; retries
hidden inside a provider SDK are not separately visible. `StrategyBuilder`
wires one collector through both executor paths. Manual compositions pass that
same collector to `LLMDifficultyClassifier` and `SmartAsk`; mismatched
collectors are rejected instead of silently dropping classifier calls.
Callers own its lifetime: a benchmark stores it with its case record, while a
conversation can retain each turn and call `aggregate_stats(turns)` for session
totals. Objective benchmark turns are labeled `passed`, `incorrect`,
`routing_error`, `execution_error`, or `evaluation_error`; ordinary turns stay
`unrated` unless the caller explicitly applies feedback with
`turn_stats.with_outcome(...)`. `run()` and `run_detailed()` retain their
original return types.

Token totals and input/output breakdowns have separate completeness flags. A
provider can report an authoritative total without reporting the breakdown; in
that case token totals remain usable but catalog pricing is unknown. Missing
usage or price evidence is never silently treated as zero. Catalog-backed
quotes serialize the complete catalog snapshot, including rates, so custom
aggregation does not lose cost provenance.
The priced model is the provider-reported actual model when available and the
requested model otherwise; both identities remain visible.

`aggregate_resources(turns)` derives totals and per-model/channel/role/strategy
rollups from the canonical calls. Each rollup includes call errors, all known
token categories and missing counts, catalog-estimated cost completeness,
latency P50/P95, response outcomes, model-attribution fallbacks, and observed
visible-output throughput. Model rollups split verified `actual` identity from
`requested_fallback` identity, including their tokens, cost, and latency rather
than merging them under a bare model name. Aggregate run time is named
`cumulative_run_duration_ms`; benchmark reports separately expose
`wall_clock_record_span_ms`, the span from the earliest recorded start to the
latest recorded finish. That span can include gaps when a run is resumed.

OpenRouter's [provider-reported account charge](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
is retained separately from the versioned catalog estimate, with independent
completeness and their difference when both are known. Pairwise cost comparisons
use provider-reported values only when both sides are complete; otherwise both
sides use catalog estimates. Catalogs always contain input/output rates and may
also contain `input_cache_read`, `input_cache_write`, `internal_reasoning`, and
fixed `request` rates. When a differentiated rate is declared but its token
detail is missing, the estimate remains unavailable rather than assuming the
full-rate path. The bundled OpenRouter snapshot includes the advertised cache
rates for both bundled models.
Provider retries hidden inside the SDK, time to first token, provider queue
time, and model-only execution time are not reported because the current
non-streaming adapters cannot observe them. Context-window utilization is also
left unknown without authoritative tokenizer/window metadata. Repeated-run
variance and confidence intervals require a future trial dimension; the current
artifact identity intentionally permits one result per strategy/task pair.

`plan(task)` is a fresh-task dry-run inspection: it skips generation, but a
model-backed method still performs classification. Execution always performs
its own route selection, so a route planned for one task cannot be injected
into a different task. Use `run_detailed(..., on_route=...)` to observe routes
during normal execution without classifying twice. Dry-run classifier calls
still require credentials and appear in the per-turn metrics.

## Strategy YAML

A strategy file has schema version 2 and exactly one root `StrategyConfig`:

```yaml
schema_version: 2
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
    fallback: easy
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
```

Supported method types are `difficulty`, `cascade`, and `fixed`. The supported
collaborators are an `llm` difficulty classifier and `marker` escalation
policy. Generation and classification can use `openrouter` or the first-party
`openai` executor; one-shot generation can also use `hermes`. Model profiles
can supply system prompts, maximum output tokens, temperature, and reasoning
effort where the selected executor supports them. A `fixed`
method may also declare `prompt_prefix` and `prompt_suffix`; this lets a
counterfactual baseline reproduce the exact user-prompt transform used by a
routed call.

Prompt-file paths are resolved relative to the strategy file. Loading rejects
duplicate keys, unknown fields, invalid compositions, missing or empty prompts,
and incompatible executor capabilities. The loaded strategy digest includes the
typed config and prompt contents, so prompt edits change its identity.

Shipped configurations are addressable after installation as:

- `builtin:product` — product difficulty method with Hermes generation
- `builtin:python-function-completion-difficulty-v1` and
  `builtin:python-function-completion-difficulty-v2` for paired
  classifier-prompt comparison
- `builtin:python-function-completion-cascade` and
  `builtin:python-function-completion-fixed-gemini-self-check`
- `builtin:python-function-completion-fixed-gemini` and
  `builtin:python-function-completion-fixed-opus`
- `builtin:python-code-generation-cascade` and
  `builtin:python-code-generation-fixed-gemini-self-check` and
  `builtin:python-code-generation-fixed-opus`
- `builtin:python-code-generation-codex-cascade` — direct OpenAI Codex
  small-to-large cascade
- `builtin:python-code-generation-groq-cascade` — Groq GPT-OSS 20B-to-120B
  cascade

Reusable strategy and prompt names describe their task/output contract rather
than the benchmark that happens to exercise them. The same strategy can be
passed to any compatible benchmark suite or application entrypoint.

## Benchmarking strategies

The benchmark applications ship under the `smart_ask` package; the `bench`
extra installs their dataset dependency. HumanEval and the LiveBench coding
public-test smoke suite are module-based applications; repeat
`--strategy` to run the same case set against several configurations:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-difficulty-v1 \
  --strategy builtin:python-function-completion-difficulty-v2 \
  --limit 20 \
  --workers 4 \
  --output benchmark-results/humaneval/prompt-comparison

python -m smart_ask.benchmarks.livebench \
  --strategy builtin:python-code-generation-cascade \
  --strategy builtin:python-code-generation-fixed-gemini-self-check \
  --strategy builtin:python-code-generation-fixed-opus \
  --workers 4
```

The three-strategy matrix produces paired routed, cheap-only, and
expensive-only evidence for counterfactual routing diagnostics.

The LiveBench command runs this project's pinned, public-test approximation;
it does not reproduce the canonical LiveBench evaluator or produce an official
LiveBench score. Its manifest identifies the evaluator accordingly.

Each run produces strict schema-v5 evidence:

```text
<output>/
├── manifest.json   dataset/evaluator/code/pricing identity and strategy snapshots
├── records.jsonl   one append-safe record per strategy/task pair
└── summary.json    aggregate metrics and paired comparisons
```

Every task record has one canonical `metrics` envelope plus a call ledger.
Attempts and routing events reference call IDs instead of duplicating output,
usage, cost, or latency. The record also retains provider-neutral requests,
provider outputs, evaluation, errors, and timestamps. Missing token or
price evidence remains explicit rather than being counted as zero. `--resume`
continues an explicit `--output` directory only when its suite, dataset,
evaluator, strategies, cases, pricing, metrics schema, workers, code/runtime
identity, and dependency versions still match. A per-directory advisory lock
prevents concurrent writers.
Summary artifacts add validated resource rollups, explicit task-outcome counts,
complete routing transition/path ledgers, and exact-once downstream usage
attribution. Pass rates and score comparisons include only `passed` and
`incorrect` outcomes; routing, execution, and evaluation errors are excluded
from rated-quality denominators while retaining their resource evidence. The
report separately shows all-task success, so excluded errors cannot make the
headline result look perfect. When the
matrix includes matching fixed easy- and hard-profile
baselines, it also reports paired counterfactual routing diagnostics: cheap
opportunity capture, unnecessary expensive routing, unsafe cheap routing,
escalation precision, estimated cost regret, and quality regret. Missing or
ambiguous baselines disable only the diagnostics that depend on them; oracle
regret requires both, while cheap-only routing metrics can retain partial
evidence. Matching is
strict across model, role, resolved prompts and user-prompt transforms, tuning,
and executor configuration; a cascade cheap baseline therefore needs the same
`prompt_suffix` as its `self_check_suffix`. Ordinary single-path traces never
fabricate counterfactuals.
When `--output` is omitted, a timestamped directory is created under
`benchmark-results/<suite>/`.

The benchmark CLI owns run controls such as suite, case limit, workers, output,
and resume. There is no separate `ExperimentConfig`; a comparison is a fixed
suite plus one or more repeatable strategy YAMLs and those CLI controls.
Use `--price-catalog catalog.json` to supply an explicit catalog snapshot when
a configured model is absent from the bundled catalog.

The correctness harnesses execute model-generated Python in local subprocesses.
A timeout limits duration but is not a security sandbox; run benchmarks in an
isolated environment when model output is untrusted.

See the packaged
[`smart_ask/benchmarks/humaneval/README.md`](smart_ask/benchmarks/humaneval/README.md)
for the HumanEval workflow and [`DESIGN.md`](DESIGN.md) for component boundaries.

## Project structure

```text
smart-ask/
├── smart_ask/
│   ├── cli.py                    installed `smart-ask` entrypoint
│   ├── application.py            SmartAsk coordinator
│   ├── domain.py                 immutable per-task values
│   ├── conversation/             harness-neutral conversation runtime/metrics
│   ├── metrics/
│   │   ├── models.py             immutable call/run values and aggregation
│   │   ├── collector.py          scoped call capture and instrumentation
│   │   ├── rollups.py            resource dimensions and latency distributions
│   │   ├── wire.py               strict metrics/v2 serialization
│   │   └── cost.py               price catalog and cost calculation
│   ├── methods/                  runtime routing methods and collaborators
│   │   ├── base.py               RoutingMethod protocol
│   │   ├── difficulty.py
│   │   ├── cascade.py
│   │   ├── fixed.py
│   │   ├── classifiers/
│   │   └── escalation/
│   ├── executors/                Hermes, Ollama, and OpenRouter executors
│   ├── strategy/                 YAML schema, loader, and builder
│   ├── resources/                bundled strategies and prompts
│   └── benchmarks/               installed suites, runner, artifacts, comparison
├── benchmark-history/            immutable pre-current-schema evidence archive
├── benchmark-results/            generated run artifacts (ignored)
├── integrations/claude_code/     separately installable external adapter
├── tests/                        network-free tests
└── pyproject.toml                package and dependency metadata
```

## License

MIT
