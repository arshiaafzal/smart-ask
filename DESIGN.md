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
| `SmartRouter` | Routing-only coordinator shared by task and conversation runtimes |
| `ConversationRuntime` | Harness-neutral structured conversation coordinator |
| external protocol adapter | Separately packaged translator that consumes `ConversationRuntime` |
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
├── SmartRouter routing-only coordinator
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
│   ├── ConversationExecutor protocol
│   ├── HermesExecutor
│   ├── OllamaExecutor / OllamaConversationExecutor
│   └── OpenRouterExecutor / OpenRouterConversationExecutor
├── conversation runtime
│   ├── neutral messages, requests, session context, and events
│   ├── routing, cascade control, and turn state
│   └── per-attempt, run, session, and model metrics
├── strategy configuration
│   ├── StrategyConfig schema
│   ├── load_strategy / LoadedStrategy
│   └── StrategyBuilder
├── metrics
│   ├── StatsCollector / StatsCapture
│   ├── CallStats / RunStats / StatsSummary
│   └── PriceCatalog / PriceQuote
└── immutable domain values
    ├── Task / Context
    ├── ExecutionRequest / ModelResult
    └── RouteResult / RoutingEvent / Attempt / RunResult

Application entrypoints
├── product CLI (`smart_ask/cli.py`, installed as `smart-ask`)
└── benchmark modules (`python -m smart_ask.benchmarks.<suite>`)

External integrations
└── `integrations/claude_code/` → depends on `smart_ask`, never the reverse
```

Classifiers and escalation policies are replaceable collaborators inside a
method. They are not complete methods and do not own the application loop.

## Strategy configuration

[`smart_ask/strategy/schema.py`](smart_ask/strategy/schema.py) defines one
strict Pydantic `StrategyConfig` with:

```text
StrategyConfig
├── schema_version: 2
├── name
├── method
│   ├── difficulty: classifier + easy/hard model profiles
│   ├── cascade: classifier + escalation policy + easy/hard profiles
│   └── fixed: one model profile + required semantic role
└── generation: OpenAI, Groq, OpenRouter, Ollama, or Hermes executor config
```

An LLM classifier has its own executor, prompt source, model, prompt-length
limit, request parameters, and explicit `easy | hard | raise` fallback. A model
profile can include a system-prompt source, maximum output tokens, temperature,
and reasoning effort. A marker policy owns its exact marker, candidate self-check
suffix, and escalation prefix.

Shipped strategy and prompt names describe reusable task/output contracts,
such as Python function completion or complete Python code generation. They do
not inherit the name of a benchmark suite that consumes them.

There is no `ExperimentConfig`. Benchmark run controls—suite, strategies,
limit, workers, output, and resume—belong to the benchmark command. This keeps a
strategy independently reusable by the product CLI, either benchmark suite, or
library callers.

### Loading and identity

`load_strategy(path | "builtin:<name>")` performs configuration-only work:

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
  → SmartAsk(method, generation executor, derived attempt bound)
```

The builder derives the closed method bound: fixed and difficulty use one
attempt; cascade uses two. It is not a strategy knob.

`StrategyBuilder.build_router(loaded)` builds the same validated method and
classifier graph without constructing a generation executor. `SmartAsk` wraps
that router for direct tasks and benchmarks.

`StrategyBuilder.build_conversation_runtime(loaded)` composes the router with a
structured executor selected solely from the strategy. The resulting
`ConversationRuntime` accepts neutral messages, tools, images, parameters, and
session correlation values. It owns profile transforms, sticky per-turn route
decisions, physical attempts, cascade buffering/escalation, and metrics.

## External protocol adapters

Protocol-specific servers do not live in `smart_ask`. An integration is a
downstream package with this dependency direction:

```text
harness
  → external protocol adapter
      → ConversationRuntime
          → strategy-configured executor
```

The adapter owns wire decoding/encoding, authentication, discovery, and server
limits. It maps each public model alias to one strategy reference, then asks
`StrategyBuilder` for a complete runtime. It does not inspect generation types,
provider URLs, credentials, cheap/hard model fields, or routing policy.

`integrations/claude_code/` is one such package. It translates its external
messages and stream into neutral core values. The core never imports this
package and contains no ASGI routes, SSE framing, or harness-specific headers.

For a conversation request, SmartAsk projects only the latest human text for
routing while retaining the complete structured conversation for execution.
Session and agent correlation values keep difficulty/cascade decisions sticky
through a tool loop. Fixed and difficulty attempts stream directly. A cascade
streams tool calls immediately; it buffers only final cheap text until the core
accepts it or replaces it with an escalated attempt.

OpenRouter clients may be shared by endpoint and credential name, but classifier
and generation executors remain distinct so prompts and parameters cannot leak
between roles. Builder dependencies remain injectable for network-free tests.

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
  is used by force overrides and baseline strategies. Fixed baselines can
  explicitly reproduce a routed call's user-prompt prefix or suffix.

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
parsing, validation, and its configured fallback—but calls only the generic
`ModelExecutor` contract and instruments it with the shared `StatsCollector`.
OpenRouter is a configured transport, not part of the classifier's policy.

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
  → SmartAsk creates ExecutionRequest with the route's semantic role
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

`SmartAsk.run_with_stats(task)` and `run_detailed_with_stats(task)` add an
immutable `RunStats` snapshot without changing those existing APIs. The shared
`StatsCollector` observes executor calls through a `ContextVar`-scoped run:

```text
CallStats             one classifier or generation executor invocation
  ↓ aggregate
RunStats              one task, benchmark case, or conversation turn
  ↓ caller-owned aggregate_stats(...) / aggregate_resources(...)
StatsSummary          a conversation, benchmark, or other caller-defined group
```

Unknown usage and price evidence is represented by incomplete totals, never by
silently adding zero. A lower-level `capture_stats()` context supports custom
callback/manual workflows and lets benchmark failures retain partial call
evidence. Normal conversation turns are explicitly `unrated`; an evaluator or
caller can replace that state with one mutually exclusive objective outcome.

## Benchmark application

The benchmark framework is separate from routing:

```text
python -m smart_ask.benchmarks.<suite> --strategy A.yaml --strategy B.yaml
  → load each StrategyConfig
  → load one pinned case set from the suite
  → build one metrics-instrumented SmartAsk application per pending strategy/task
  → run every strategy/task pair
  → suite evaluates each final output
  → append one schema-v5 evidence record
  → summarize each strategy and compute paired comparisons
```

`BenchmarkSuite` owns dataset loading and correctness evaluation.
`run_matrix` owns concurrency, isolates each strategy/task pair in its own
application instance, and applies every strategy to the same case set.
The shared collector records both classifier and generation transports: their
provider-neutral `ExecutionRequest` values, outputs, usage, failures, semantic
roles, requested/actual/priced models, channels, latency, pricing provenance,
normalized termination/output state, and separate caller-requested and
adapter-applied generation caps. Benchmarks
retain the call ledger; regular callers receive the immutable `RunStats`
snapshot. Provider-specific defaults and system prompts remain in the strategy
snapshot.

Actual model identity is provider evidence and may be absent. Pricing uses that
actual model when present and otherwise the requested model; both identities,
the selected priced model, and the complete catalog snapshot remain explicit.
OpenRouter's reported account charge is recorded alongside—not merged with—the
catalog estimate, and each has independent completeness.

### Evidence schema v5

Benchmark artifact schema version 5 is distinct from strategy YAML schema
version 2. A run directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Dataset/evaluator identity, case hashes, strategy snapshots, metrics/pricing provenance, Python/dependency versions, package hash/version, and matching-checkout Git state |
| `records.jsonl` | One crash-safe line per strategy/task pair with input, route, classifier decision, events, attempts, call ledger, final output, evaluation, canonical metrics envelope, errors, and timestamps |
| `summary.json` | Per-strategy aggregates and all paired comparisons |

Each JSONL line is flushed and synced before completion is recorded. Resume
skips completed `(strategy_id, task_id)` pairs only after verifying the existing
manifest still matches the requested benchmark, dataset, strategies, cases,
evaluator, pricing, metrics schema, worker count, runtime/dependency versions,
and code identity. An advisory lock prevents concurrent writers to one run
directory.

Calls are the canonical source for request, output, usage, cost, and latency.
Attempts and semantic routing events reference call IDs. The record-level
`metrics` object and summary-level `metrics` object use the same
`smart-ask.metrics/v2` envelope at different scopes; there are no parallel flat
usage or cost totals.

The derived reporting layer has four explicit levels:

```text
Call evidence
  → resource rollups by model/channel/role/strategy
  → mutually exclusive task outcomes
  → routing transitions, complete paths, and benchmark-only counterfactuals
```

Only `passed` and `incorrect` task outcomes enter quality rates and paired score
comparisons. Routing, execution, and evaluation errors remain explicit excluded
tasks while their call, cost, token, and timing evidence remains reportable.

Generation usage belongs to the transition that launches its call, preventing
the same downstream cost from being counted at every gate. The transition
ledger retains gate/decision, route and model movement, task identity, and
resulting call ID. Paired fixed easy/hard baselines enable router regret and
opportunity/error rates. A match includes role, resolved system and user-prompt
transforms, tuning, and executor configuration. Missing evidence disables only
the dependent metric; full oracle regret requires both exact baselines.
Per-strategy summaries distinguish
`wall_clock_record_span_ms`
(which can include resumed-run gaps) from cumulative run time. Hidden SDK
retries, TTFT, provider queue time, model-only execution time, and authoritative
context-window utilization are absent because the current adapters cannot
observe them. Repeated-run variance needs a future trial dimension because a
run currently permits one result per strategy/task pair. The reader accepts
only schema-v5 run directories;
historical artifacts remain under `benchmark-history/` with provenance notes.

## Dependency direction

Arrows mean “imports or depends on”:

```text
product CLI / benchmark CLI
  → strategy loader and builder
      → concrete methods, collaborators, and executors
          → protocols and immutable domain values

benchmark runner
  → SmartAsk public behavior and BenchmarkSuite contract
  → core metrics plus artifact/comparison modules

external protocol adapter
  → SmartAsk conversation API and strategy builder

SmartAsk core
  ↛ external protocol adapter, ASGI framework, or wire protocol
```

Methods do not import concrete executors. Classifiers depend on the executor
protocol and core metrics, not OpenRouter. Executors do not import methods.
Benchmark concerns do not enter the core application. A shared collector is
injected into the builder and application, and benchmarks serialize its public
call and run evidence.

## Main file responsibilities

| Path | Responsibility |
|---|---|
| `smart_ask/domain.py` | Immutable per-task route, request, response, and audit values |
| `smart_ask/application.py` | Method/executor coordination and attempt guard |
| `smart_ask/conversation/` | Harness-neutral conversation domain, runtime, and metrics |
| `smart_ask/metrics/models.py` | Immutable token/call/run metrics and aggregation |
| `smart_ask/metrics/collector.py` | Context-scoped call capture and executor instrumentation |
| `smart_ask/metrics/cost.py` | Versioned prices and cost calculation |
| `smart_ask/metrics/rollups.py` | Resource aggregation by model, channel, role, and strategy |
| `smart_ask/metrics/wire.py` | Strict metrics-v2 serialization and reconciliation |
| `smart_ask/methods/` | Runtime methods and their classifier/escalation collaborators |
| `smart_ask/executors/` | Provider/process transport adapters |
| `smart_ask/strategy/schema.py` | Strict root `StrategyConfig` model |
| `smart_ask/strategy/loader.py` | Safe YAML loading, prompt resolution, digest and manifest |
| `smart_ask/strategy/builder.py` | Runtime object construction and dependency injection |
| `smart_ask/cli.py` | Installed product command |
| `smart_ask/resources/strategies/` | Shipped strategy YAML files |
| `smart_ask/resources/prompts/` | Versioned prompt text referenced by YAML |
| `smart_ask/benchmarks/suite.py` | Benchmark case/evaluation and strategy contracts |
| `smart_ask/benchmarks/runner.py` | Matrix execution and evidence serialization |
| `smart_ask/benchmarks/run_manifest.py` | Reproducible manifest and code identity |
| `smart_ask/benchmarks/artifacts.py` | Crash-safe JSONL persistence, locking, and resume |
| `smart_ask/benchmarks/artifact_schema.py` | Strict schema-v5 integrity validation |
| `smart_ask/benchmarks/compare.py` | Aggregates, resource/timing summaries, and paired comparisons |
| `smart_ask/benchmarks/routing_analysis.py` | Transition/path ledger with exact-once call attribution |
| `smart_ask/benchmarks/counterfactual.py` | Paired routing quality and regret diagnostics |
| `smart_ask/benchmarks/humaneval/`, `livebench/` | HumanEval and noncanonical LiveBench public-test evaluation |
| `benchmark-history/` | Read-only archive of pre-current-schema evidence and its caveats |
| `integrations/claude_code/` | Separately installed external protocol adapter |

## Public API

The root `smart_ask` package re-exports the coordinator, common metric/domain
values, shipped methods and collaborators, executors, and strategy loading and
building APIs. Focused metric imports use `smart_ask.metrics`; other focused
imports use their corresponding subpackages.
