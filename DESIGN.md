# Design

SmartAsk has one conversation-native execution model. Every caller presents an
immutable snapshot of the conversation for the current turn; one strategy
method performs all reasoning needed to select one visible response.

## Core model

```text
terminal / benchmark / protocol adapter
                 │
                 ▼
     Conversation + RunMetadata
                 │
                 ▼
          StrategyEngine
                 │
                 ▼
 StrategyMethod.respond(conversation, run)
       │                       │
       ├─ record decisions       ├─ hidden model calls
       ├─ assess candidates     └─ choose final call
       └─ return opaque PreparedResponse
                 │
                 ▼
      user-visible event stream
                 +
        canonical RunRecord
```

The unit of execution is one method invocation per incoming message/request.
It is not a whole session. A session is a caller-owned sequence of invocations
connected by a `session_id` and by the conversation history the caller passes.

The distinction matters:

- A `Conversation` describes the complete state visible to a method now.
- A `RunRecord` describes what one method invocation decided and spent.
- A session aggregates several independent records.

## Vocabulary and ownership

| Term | Owns |
|---|---|
| `Conversation` | Structured input snapshot: system blocks, ordered messages, tools, parameters, extensions |
| `StrategyMethod` | Routing, internal calls, candidate assessment, escalation, final selection |
| `RunScope` | Bounded effectful operations and canonical decision/call recording |
| `ModelCallSpec` | Provider-independent intent for one logical model call |
| `PreparedResponse` | Opaque, scope-bound selection of the visible response |
| `StrategyEngine` | Streaming, lifecycle, cancellation, heartbeats, and finalization |
| `RunRecord` | Immutable source of truth for one invocation's evidence |
| `TargetRegistry` | Deployment-approved mapping from logical target IDs to physical backends |
| `StrategyConfig` | Reproducible policy and request transforms, expressed in YAML |

No separate routing coordinator or one-shot input domain exists. A one-message
request is simply a `Conversation` with one user message. Benchmarks and chat
harnesses use the same path.

## Conversation representation

`Conversation` is an immutable aggregate, not a mutable session object. It
contains tuples of immutable structured blocks and ordered messages.

A caller normally keeps an append-only list of messages and creates a snapshot
for each turn:

```text
turn 1: [user-1]
turn 2: [user-1, assistant-1, user-2]
turn 3: [user-1, assistant-1, user-2, assistant-2, user-3]
```

A linked list would complicate serialization and random inspection without
changing provider behavior. Structural sharing can be added internally if
copying becomes measurable; it should not alter the public aggregate.

Most model APIs are stateless. Each physical request therefore re-encodes its
applicable system instructions, history, tools, and current message. This costs
input tokens repeatedly, but it gives every invocation a self-contained state
and allows providers to schedule requests independently. Provider-side prompt
caching may reduce compute or price without changing this interface.

### Projections and transforms

Every method receives the full conversation. It explicitly decides what each
subordinate call sees:

- A difficulty classifier may project only the latest user text or the full
  structured conversation.
- A model profile can append system instructions and override generation
  parameters.
- A fixed or cascade policy can prefix or suffix the latest human text.
- The final generation call normally retains the complete context.

Transforms return a new `Conversation`. They never flatten unrelated tool,
image, reasoning, or extension blocks into text.

## Method boundary

A method is a complete response algorithm:

```python
class StrategyMethod(Protocol):
    async def respond(
        self,
        conversation: Conversation,
        run: RunScope,
    ) -> PreparedResponse: ...
```

Intermediate attempts are local variables inside this call. They do not need
to be passed in from the caller because the same invocation creates, assesses,
and consumes them. The run scope records their observable evidence.

The bundled methods are:

- `FixedStrategyMethod`: records its selection and plans one live response.
- `DifficultyStrategyMethod`: obtains a structured assessment, records the
  route, and plans one easy or hard response.
- `CascadeStrategyMethod`: classifies, buffers a cheap candidate when
  appropriate, records acceptance or escalation, then replays that candidate
  or plans an expensive response.

Methods depend on `RunScope`, `ModelProfile`, and small policy collaborators.
They know no URLs, credentials, SDK clients, trace sinks, benchmark schemas, or
harness protocols.

### Why the prepared response is opaque

Only `RunScope` creates a `PreparedResponse`, and only its owning engine can
consume it. This enforces three invariants:

1. The selected call and decision belong to this invocation.
2. A prepared response can be consumed only once.
3. The method cannot fabricate a response that bypasses call evidence.

A live response begins its provider stream after the method commits it. A
replayed response emits events already buffered by the same run, as happens
when a cascade accepts a cheap candidate.

## Run scope and evidence

`RunScope` is the method's capability boundary. It can:

- record an ordered `DecisionRecord`;
- execute a bounded hidden call and return normalized evidence;
- plan a live final call;
- select a buffered result for replay.

Limits cap logical calls, buffered events, buffered bytes, and elapsed buffer
time. Decision evidence must refer to a completed call from the same run.

The record keeps three normalized ledgers:

```text
DecisionRecord
  decision id, gate, outcome, reason, selected profile, evidence call ids

ModelCallRecord
  call id, profile, target, role, phase, causing decision, status

ProviderRequestRecord
  request id, parent call, actual model, tokens, cost, timing, stop/output state
```

Logical calls and provider requests are separate because semantic method intent
and physical execution are different evidence. The request that a decision launches owns
its resources. Derived funnels must not copy one request's tokens or cost onto
all earlier gates.

`RunRecord` deliberately excludes prompts and generated content. It is safe for
normal operational accounting, subject to the metadata extensions supplied by
the caller. An opt-in content trace is observed live: it records the original
conversation, every transformed model-call conversation, chunked output,
decisions, and terminal state.

## Engine lifecycle

`StrategyEngine.start(conversation, metadata)` returns a single-consumer
`RunHandle`:

```text
start
  → method preparation
      → heartbeat events while hidden work is pending
  → commit PreparedResponse
  → visible response stream
  → bounded cleanup
  → immutable RunRecord
```

The caller consumes `handle.events()` and then awaits `handle.result()`. Calling
`complete(...)` performs both steps and returns all visible events with the
record.

Stream closure, task cancellation, provider failure, and method failure all
converge on the same finalizer. Active iterators are closed, hidden work is
cancelled, and the record receives a terminal `completed`, `error`, or
`cancelled` state. Cleanup is independently bounded so cancellation cannot
leave a result future unresolved indefinitely.

## Token counting

Token counting is read-only. A method declares every currently reachable final
request candidate after pure transforms. The executor counts those requests
without classifying, generating, mutating route memory, or emitting run
metrics.

- One reachable candidate produces an exact count.
- Several reachable candidates produce the maximum count as an upper bound.
- Missing authoritative tokenization fails explicitly.

This is suitable for external protocol preflight endpoints. It is not an
estimate of the hidden classification or escalation work a future invocation
will choose.

## Session-dependent policy

The conversation snapshot is sufficient for a stateless method decision.
Optional route memory is method policy, not engine or harness policy. It may
cache route affinity with a TTL and bounded LRU size, or pin a tool loop to a
profile after the model emits a tool call.

Any memory key must include robust session/agent identity and the relevant
human turn identity. Absence of usable identity must disable reuse rather than
allow accidental sharing between users. Reused decisions remain explicit in
the run evidence.

## Strategy configuration

The strict schema v3 root is:

```text
StrategyConfig
├── schema_version: 3
├── name
├── profiles: logical profile → trusted target + request transform
├── method
│   ├── fixed: profile + role + optional user transform
│   ├── difficulty: classifier + easy/hard profiles + optional memory
│   └── cascade: classifier + candidate policy + profiles + optional memory
└── limits: call, buffer, and deadline ceilings
```

The strategy owns reproducible semantics:

- profile names and target references;
- prompts and request transforms;
- classifier projection, fallback, and missing-input behavior;
- candidate marker and tool-output policy;
- optional route-memory bounds;
- limits tighter than deployment ceilings.

The strategy does not own deployment authority:

- network endpoints;
- environment-variable names;
- provider/client types;
- executable paths;
- timeout ceilings;
- the physical model behind a trusted target.

These values live in `TargetRegistry`. A `TargetDefinition` binds a stable
target ID to a structured transport, model, endpoint, credential handle,
capabilities, and hard limits. The public target snapshot exposes reproducible
model and capability metadata plus a configuration digest, but not endpoint or
credential values.

### Loading and building

`load_strategy(...)` performs pure validation:

1. Constrain the source to approved roots and file sizes.
2. Parse one UTF-8 mapping with duplicate-key rejection.
3. Validate the closed Pydantic schema.
4. Resolve referenced prompt files relative to the strategy source.
5. Compute a digest over typed configuration and exact prompt contents.

`StrategyBuilder` then performs environment-dependent composition:

```text
LoadedStrategy
  → resolve every profile/classifier target
  → reject missing targets or unsupported capabilities
  → compile explicit request transforms and method policy
  → create one target-backed model-call executor
  → create StrategyEngine with declared limits
```

Force-easy and force-hard are build-time substitutions that compile the chosen
profile into a fixed method. A strategy that already has only one fixed profile
does not accept those overrides.

## Transport boundary

The model-call executor resolves a trusted target and delegates to a structured
protocol transport. Transports translate neutral conversations and normalized
events; they do not route or evaluate candidates.

```text
StrategyMethod
  → ModelCallSpec(target_id, transformed Conversation)
  → TargetExecutorRegistry
  → OpenAI / OpenRouter / Groq / Ollama transport
  → normalized ConversationEvent stream
```

Clients can be pooled by trusted endpoint and credential. Strategy values can
never create a client or select an arbitrary command.

The complete logical `Conversation` is the invariant at this boundary; its wire
representation is transport-specific. For example, a stateless Anthropic
transport would send Opus the applicable history for every request. A stateful
local transport may instead send only the new suffix when it has a validated
context/KV-cache handle proving that the backend already holds the exact
conversation prefix. It must send the complete history after cache expiry,
server restart, conversation branching, or any other loss of continuity.

This optimization is invisible to strategy methods. A method never truncates
history to accommodate a cache and never assumes that backend state exists.
The current Ollama `/chat` transport sends the complete message history;
keeping the model loaded does not itself establish a persistent conversation
cache.

## External adapters

Harness-specific servers are downstream integrations:

```text
harness
  → protocol adapter
      → Conversation + RunMetadata
          → StrategyEngine
```

The adapter owns wire parsing, authentication, discovery, concurrency and
request-size limits, stream framing, and conversion of tools/images/reasoning.
It maps each advertised model alias to one loaded strategy. It does not inspect
cheap/hard profiles, start provider services, or implement routing.

The Claude Code adapter is separately packaged under
`integrations/claude_code/`. Core code never imports it and contains no ASGI,
Anthropic Messages, or Claude Code behavior. Claude Code's own agent system
instructions are part of the incoming conversation and are intentionally
preserved.

## Metrics and traces

`RunMetricsStore` accepts finalized records and derives caller-owned session
rollups:

- run, error, and cancellation counts;
- logical calls, provider requests, and retries;
- input/output/reasoning/cache tokens where available;
- reported or catalog-estimated cost with missing-evidence counts;
- breakdowns by actual/selected model and trusted target.

Raw ledgers remain canonical; ratios and funnels are derived. Unknown evidence
is never converted to zero.

The standard JSONL metrics sink stores the prompt-free record plus a current
session aggregate. An opt-in trace is one private directory per launcher
session: `session.json` indexes terminal status and one numbered `.log` file
holds each method invocation. Each event is one conventional log line with a
timestamp, level, component, message, and compact `key=value` evidence. `INFO`
shows calls, decisions, output, usage, and terminal state; `DEBUG` carries full
conversation evidence and thinking; `WARN` and `ERROR` mark failures.

The session index normalizes shared strategy/session context and content
digests. An exact repeated input points to its first ordinal without asserting
why it repeated. Within a log, calls identify reuse of the run input and print
small parameter changes inline. Custom contexts are logged where the call is
planned. Short content stays on one line; long content uses `begin`/`end`
markers with incremental indented continuations.

## Benchmark architecture

The benchmark framework is an engine caller, not a second runtime:

```text
load fixed case set
  → for each strategy/case pair
      → one-message Conversation
      → isolated StrategyEngine invocation
      → suite evaluates selected visible output
      → append canonical result
  → summarize quality, resources, timing, routes, counterfactuals
```

The suite owns dataset identity and evaluation. The matrix runner owns
concurrency and fair case ordering. The strategy owns model behavior.

A run directory contains a manifest, append-safe records, and a summary. The
manifest fingerprints the dataset, evaluator, cases, strategies, prompt
digests, trusted target resolutions, and price catalog. Resume is allowed only
when that identity still matches.

Counterfactual routing quality requires matched routed, cheap-only, and
expensive-only results for the same cases. It cannot be inferred from one
production path. The oracle is the cheapest profile meeting the quality
threshold; cost and quality regret compare the routed result to that oracle.

## Dependency direction

Arrows mean “imports or depends on”:

```text
terminal / benchmark / external adapter
  → strategy loader and builder
      → methods + target executor registry
          → conversation engine and immutable values

methods → conversation contracts
transports → conversation contracts
methods ⇛ concrete transports
core ⇛ external adapters
core ⇛ benchmark suites
```

## File responsibilities

| Path | Responsibility |
|---|---|
| `smart_ask/conversation/model.py` | Conversation input and canonical run evidence |
| `smart_ask/conversation/engine.py` | Run scope, streaming lifecycle, cleanup, token counting |
| `smart_ask/conversation/metrics.py` | Record serialization and session resource rollups |
| `smart_ask/methods/strategies.py` | Fixed, difficulty, cascade, projections, transforms |
| `smart_ask/methods/memory.py` | Optional bounded route affinity |
| `smart_ask/executors/` | Target resolution and structured transports |
| `smart_ask/strategy/schema.py` | Strict strategy schema v3 |
| `smart_ask/strategy/loader.py` | Safe loading, prompt resolution, digest |
| `smart_ask/strategy/targets.py` | Trusted deployment target definitions |
| `smart_ask/strategy/builder.py` | Policy compilation into one engine |
| `smart_ask/benchmarks/` | Case matrix, persistence, summaries, routing analysis |
| `integrations/claude_code/` | External Anthropic-compatible adapter |
