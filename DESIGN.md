# Design

`smart-ask` separates runtime routing algorithms from reproducible composition.
A `RoutingMethod` decides the next action for one task. A root `StrategyConfig`
YAML declares which method and collaborators to use, how models are configured,
and which transports execute classifier and generation calls.

## Architectural vocabulary

| Term | Meaning |
|---|---|
| `RoutingMethod` | Runtime algorithm that returns the next `RouteResult` |
| classifier | Method collaborator that produces an easy/hard assessment |
| escalation policy | Cascade collaborator that accepts or escalates a response |
| `ModelExecutor` | Transport adapter that executes an `ExecutionRequest` |
| `StrategyConfig` | Complete typed YAML composition for one runnable method |
| `SmartAsk` | Coordinator that loops between a method and generation executor |
| benchmark suite | Dataset loader and correctness evaluator |

The term “strategy” refers to the complete YAML composition, not the runtime
algorithm class. Runtime algorithms therefore live under `smart_ask/methods/`
and are named `DifficultyRoutingMethod`, `CascadeRoutingMethod`, and
`FixedRoutingMethod`.

## Ownership map

Indentation means “belongs to,” not runtime call order.

```text
Shared application library (`smart_ask/`)
├── SmartAsk coordinator
├── methods
│   ├── RoutingMethod protocol
│   ├── DifficultyRoutingMethod
│   ├── CascadeRoutingMethod
│   ├── FixedRoutingMethod
│   ├── classifiers                    subordinate method collaborators
│   │   ├── DifficultyClassifier
│   │   └── LLMDifficultyClassifier
│   └── escalation policies            subordinate cascade collaborators
│       ├── EscalationPolicy
│       └── MarkerEscalationPolicy
├── executors
│   ├── ModelExecutor protocol
│   ├── HermesExecutor
│   └── OpenRouterExecutor
├── strategy configuration
│   ├── StrategyConfig schema
│   ├── load_strategy / LoadedStrategy
│   └── StrategyBuilder
└── immutable domain values
    ├── Task / Context
    ├── ExecutionRequest / ModelResult
    └── RouteResult / RoutingEvent / Attempt / RunResult

Application entrypoints
├── product CLI (`smart-ask`)
└── benchmark modules (`python -m benchmarks.<suite>`)
```

Classifiers and escalation policies are replaceable collaborators inside a
method. They are not complete methods and do not own the application loop.

## Strategy configuration

[`smart_ask/strategy/schema.py`](smart_ask/strategy/schema.py) defines one
strict Pydantic `StrategyConfig` with:

```text
StrategyConfig
├── schema_version: 1
├── name
├── method
│   ├── difficulty: classifier + easy/hard model profiles
│   ├── cascade: classifier + escalation policy + easy/hard profiles
│   └── fixed: one model profile + decision + optional role
├── generation: OpenRouter or Hermes executor config
└── max_attempts
```

An LLM classifier has its own executor, prompt source, model, prompt-length
limit, and request parameters. A model profile can include a system-prompt
source, maximum output tokens, and temperature. A marker policy owns its exact
marker, candidate self-check suffix, and escalation prefix.

Shipped strategy and prompt names describe reusable task/output contracts,
such as Python function completion or complete Python code generation. They do
not inherit the name of a benchmark suite that consumes them.

There is no `ExperimentConfig`. Benchmark run controls—suite, strategies,
limit, workers, output, and resume—belong to the benchmark command. This keeps a
strategy independently reusable by the product CLI, either benchmark suite, or
library callers.

### Loading and identity

`load_strategy(path)` performs configuration-only work:

1. Reads one UTF-8 YAML mapping through a duplicate-key-rejecting safe loader.
2. Validates the closed schema; unknown fields and incompatible compositions
   fail before any model call.
3. Resolves prompt files relative to the strategy YAML, preserving exact text.
4. Validates prompt existence/content and the marker producer/parser contract.
5. Computes a SHA-256 identity from the typed config and referenced prompt
   contents.

The resulting `LoadedStrategy` retains the source path, validated config,
digest, resolved prompt text, and a JSON-serializable manifest snapshot with
prompt text and hashes. Loading does not read credentials or create clients, so
the product can validate a strategy with `--validate-strategy` offline.

### Building

`StrategyBuilder.build(loaded)` performs environment-dependent composition:

```text
LoadedStrategy
  → build classifier executor and LLMDifficultyClassifier when required
  → build MarkerEscalationPolicy when required
  → build the selected RoutingMethod
  → build the separately configured generation executor
  → SmartAsk(method, generation executor, max_attempts)
```

OpenRouter clients may be shared by endpoint and credential name, but classifier
and generation executors remain distinct instances so prompts and model
parameters cannot leak between roles. Builder dependencies—environment, client
factory, Hermes runner, and executor wrapper—are injectable for offline tests
and benchmark tracing.

`StrategyBuilder.build(..., force="easy" | "hard")` replaces the configured
method with `FixedRoutingMethod` using the corresponding configured profile. It
does not need to build or call the classifier.

## Runtime methods

- `DifficultyRoutingMethod` classifies once, selects the configured easy or
  hard model, executes once, and then accepts the response.
- `CascadeRoutingMethod` classifies once. A hard task goes directly to the hard
  model. An easy task goes to the easy model and is assessed by the configured
  escalation policy; a marker decision can trigger one hard-model retry.
- `FixedRoutingMethod` selects one configured model without classification and
  is used by force overrides and baseline strategies.

The method classes depend on collaborator protocols, not concrete classifier or
policy implementations.

## Runtime composition

The shipped product strategy builds this object graph:

```text
SmartAsk
├── method: DifficultyRoutingMethod
│   └── classifier: LLMDifficultyClassifier
│       └── classifier executor: OpenRouterExecutor
└── generation executor: HermesExecutor
```

The shipped cascade strategies build:

```text
SmartAsk
├── method: CascadeRoutingMethod
│   ├── classifier: LLMDifficultyClassifier
│   │   └── classifier executor: OpenRouterExecutor
│   └── escalation policy: MarkerEscalationPolicy
└── generation executor: OpenRouterExecutor
```

The classifier contains classification behavior—prompt construction, response
parsing, validation, and fallback—but calls only the generic `ModelExecutor`
contract. OpenRouter is a configured transport, not part of the classifier's
policy.

## Contracts and capabilities

```python
class RoutingMethod(Protocol):
    requires_response_text: bool

    def route(self, task: Task, context: Context = Context()) -> RouteResult: ...


class DifficultyClassifier(Protocol):
    def classify(self, task: Task) -> DifficultyClassification: ...


class EscalationPolicy(Protocol):
    def prepare_candidate_prompt(self, task: Task) -> str: ...
    def assess(self, response: ModelResult) -> EscalationDecision: ...
    def prepare_escalation_prompt(self, task: Task) -> str: ...


class ModelExecutor(Protocol):
    captures_output: bool

    def execute(self, request: ExecutionRequest) -> ModelResult: ...
```

These are structural protocols; explicit inheritance is optional.
`CascadeRoutingMethod` requires captured generation text. The strategy schema
therefore rejects a cascade paired with Hermes generation. An LLM classifier
also requires a capturing executor. Hermes ignores per-request token and
temperature hints, so the schema rejects model tuning and system prompts when
Hermes is the generation transport.

## One-task lifecycle

```text
Task
  → method.route(Task, Context)
      → optional DifficultyClassification
      → RoutingEvent(s)
      → RouteResult
  → SmartAsk creates ExecutionRequest
  → generation executor returns ModelResult
  → Attempt is appended to immutable Context
  → method returns accept or another route
  → RunResult
```

`RouteResult.phase` is typed as `initial-easy`, `initial-hard`, `escalation`, or
`fixed` and drives method control flow. Its `label` is presentation metadata.
`SmartAsk.run(task)` returns the final `ModelResult`; `run_detailed(task)`
returns all attempts and passive routing events. A maximum-attempt guard
prevents a faulty method from routing forever.

## Benchmark application

The benchmark framework is separate from routing:

```text
python -m benchmarks.<suite> --strategy A.yaml --strategy B.yaml
  → load each StrategyConfig
  → load one pinned case set from the suite
  → build one traced SmartAsk application per strategy
  → run every strategy/task pair
  → suite evaluates each final output
  → append one schema-v3 evidence record
  → summarize each strategy and compute paired comparisons
```

`BenchmarkSuite` owns dataset loading and correctness evaluation.
`run_matrix` owns concurrency and applies every strategy to the same case set.
`TracedExecutor` wraps both classifier and generation transports, recording
their provider-neutral `ExecutionRequest` values, outputs, usage, failures,
channels, and latency without changing the routing library. Provider-specific
defaults and system prompts remain in the strategy snapshot.

### Evidence schema v3

Benchmark artifact schema version 3 is distinct from strategy YAML schema
version 1. A run directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Dataset revision and case hashes, strategy config/prompt snapshots and digests, pricing, Python and git identity |
| `records.jsonl` | One crash-safe line per strategy/task pair with input, route, classifier decision, events, attempts, calls, outputs, evaluation, usage, cost, latency, errors, and timestamps |
| `summary.json` | Per-strategy aggregates and all paired comparisons |

Each JSONL line is flushed and synced before completion is recorded. Resume
skips completed `(strategy_id, task_id)` pairs only after verifying the existing
manifest still matches the requested benchmark, dataset, strategies, cases,
pricing, worker count, runtime/dependency versions, and code identity.

The comparison layer reports accuracy/score, route counts, calls, attempts,
tokens, priced cost, errors, and latency summaries. Pairwise output retains
per-task outcomes, score/cost/latency deltas, missing tasks, and counts where
only one strategy passes. Missing cost or latency remains explicit rather than
being treated as zero. Legacy JSON result files can be normalized for reading,
with unavailable evidence marked as missing.

## Dependency direction

Arrows mean “imports or depends on”:

```text
product CLI / benchmark CLI
  → strategy loader and builder
      → concrete methods, collaborators, and executors
          → protocols and immutable domain values

benchmark runner
  → SmartAsk public behavior and BenchmarkSuite contract
  → tracing/artifact/comparison modules
```

Methods do not import concrete executors. Classifiers depend on the executor
protocol, not OpenRouter. Executors do not import methods. Benchmark concerns
do not enter the core application; tracing is injected through the builder's
executor wrapper.

## Main file responsibilities

| Path | Responsibility |
|---|---|
| `smart_ask/domain.py` | Immutable per-task route, request, response, and audit values |
| `smart_ask/application.py` | Method/executor coordination and attempt guard |
| `smart_ask/methods/` | Runtime methods and their classifier/escalation collaborators |
| `smart_ask/executors/` | Provider/process transport adapters |
| `smart_ask/strategy/schema.py` | Strict root `StrategyConfig` model |
| `smart_ask/strategy/loader.py` | Safe YAML loading, prompt resolution, digest and manifest |
| `smart_ask/strategy/builder.py` | Runtime object construction and dependency injection |
| `strategies/` | Shipped strategy YAML files |
| `prompts/` | Versioned prompt text referenced by YAML |
| `benchmarks/suite.py` | Benchmark case/evaluation contract |
| `benchmarks/runner.py` | Tracing, matrix execution, evidence serialization |
| `benchmarks/artifacts.py` | Schema-v3 JSONL persistence, resume, legacy reading |
| `benchmarks/compare.py` | Aggregates and paired comparisons |
| `benchmarks/humaneval/`, `livebench/` | Suite-specific datasets and evaluation |

## Public API

The root `smart_ask` package re-exports the coordinator and domain values;
`RoutingMethod` and all shipped method/collaborator implementations;
`ModelExecutor`, `HermesExecutor`, and `OpenRouterExecutor`; and
`StrategyConfig`, `LoadedStrategy`, `load_strategy`, and `StrategyBuilder` with
their configuration/build errors. Focused imports remain available from the
corresponding subpackages.
