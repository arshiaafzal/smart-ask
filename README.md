# smart-ask

SmartAsk is a Claude Code cost router. Its default strategy uses Claude Sonnet
for straightforward agent turns and Claude Opus for difficult coding and
reasoning turns, through the native Anthropic Messages API.

With Claude Code already installed, the normal workflow is one command:

```bash
cp scripts/claude-smart-ask.local.env.example scripts/claude-smart-ask.local.env
# Put ANTHROPIC_API_KEY in the ignored local file, then:
./scripts/claude-smart-ask
```

All ordinary Claude Code arguments still work, for example
`./scripts/claude-smart-ask -p "fix the tests" --print`. Claude Code keeps its
normal UI and status information. Each response ends with a colored footer
showing the answering model, full turn cost (classifier plus generation), and
cumulative Claude Code session cost. SmartAsk removes this display-only footer
before the next model call, so it cannot influence routing or answers.
Prompt-free detailed metrics are also written to
`.smart-ask/claude-code/metrics.jsonl`. Costs are direct Anthropic list-price
estimates unless a provider reports billed cost.

Under the hood, `smart-ask` runs a configurable model strategy for each turn. A
strategy may select a fixed model, classify work by difficulty, or try a cheap
model and escalate. The same engine serves the terminal, benchmarks, and
external harness adapters.

```text
caller
  → immutable Conversation (complete history for this turn)
  → StrategyEngine
      → StrategyMethod
          → zero or more hidden classifier/candidate calls
          → one selected visible response
  → streamed events + canonical RunRecord
```

One engine invocation corresponds to one incoming message/request, not an
entire chat session. The caller owns the session history and passes a complete
snapshot on every invocation. The method can inspect that snapshot, make
several internal model calls, and choose one response. It does not expose its
intermediate candidates as separate user turns.

## Install

Python 3.11 or newer is required.

```bash
python3.11 -m pip install -e .

# Optional benchmark datasets.
python3.11 -m pip install -e '.[bench]'

# Optional Claude Code protocol adapter.
python3.11 -m pip install -e ./integrations/claude_code
```

Set only the credential required by the selected trusted target:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # default Claude Code strategy
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

The local Qwen target uses Ollama and needs no provider key:

```bash
ollama pull qwen3:14b
ollama serve
```

## Terminal conversation

```bash
smart-ask "explain this repository"
smart-ask --strategy builtin:local-qwen "hello"
smart-ask --strategy path/to/strategy.yaml "review this design"
smart-ask -f app.py -f tests/test_app.py "find the bug"
smart-ask --force-easy "use the easy profile"
smart-ask --force-hard "use the hard profile"
smart-ask --validate-strategy --strategy builtin:product
```

After the first response, the terminal remains open as one continuous
conversation. Every later turn contains all user and assistant messages that
preceded it. `/exit`, `/quit`, Ctrl-D, or Ctrl-C ends the session.

The terminal prints the method decisions, logical model calls, provider
requests, tokens, estimated or reported cost, and cumulative session totals.
Missing usage or pricing remains unknown rather than being counted as zero.

## Library API

Use `StrategyEngine` for both streaming and collected execution:

```python
import asyncio

from smart_ask import Conversation, RunMetadata, StrategyBuilder, load_strategy


async def main() -> None:
    loaded = load_strategy("builtin:local-qwen")
    engine = StrategyBuilder().build_engine(loaded)
    conversation = Conversation.from_text("Write a Python hello-world program.")
    metadata = RunMetadata(
        strategy_name=loaded.config.name,
        strategy_digest=loaded.digest,
        session_id="example-session",
        request_id="turn-1",
    )

    completed = await engine.complete(conversation, metadata)
    print(completed.events)
    print(completed.record.decisions)
    print(completed.record.provider_requests)
    await engine.aclose()


asyncio.run(main())
```

For live output, call `engine.start(...)`, consume `handle.events()` exactly
once, then await `handle.result()` for the final record. Closing or cancelling a
stream finalizes the run as cancelled and closes active provider work.

The important runtime values are:

| Value | Meaning |
|---|---|
| `Conversation` | Immutable, structured snapshot of system blocks, ordered messages, tools, parameters, and extensions |
| `StrategyMethod` | Complete response policy for one invocation |
| `RunScope` | Bounded capability through which a method records decisions and performs hidden or final model calls |
| `StrategyEngine` | Streams the method's selected response and finalizes evidence |
| `RunRecord` | Content-free source of truth for decisions, calls, provider requests, usage, timing, and terminal status |

The caller stores normal chat history as a sequence of messages. A new
`Conversation` snapshot can share immutable values safely; a linked-list API is
not needed. Provider APIs are generally stateless, so the full applicable
history is encoded again for each model request.

Methods always pass a complete logical `Conversation` to the model-call layer.
How that conversation is consumed is a transport concern: a stateless
Anthropic transport would send Opus the applicable history on every request,
while a stateful local transport could send only the new suffix when it holds a
validated context/KV-cache handle for that exact conversation prefix. If the
handle expired, the server restarted, or the conversation branched, the
transport must fall back to the complete history. Methods never manage or
depend on this optimization.

The current Ollama `/chat` transport sends the complete message history.
Keeping an Ollama model loaded is not, by itself, a persistent conversation
cache; suffix-only requests require an explicit stateful backend contract.

## Strategy YAML v3

A strategy describes policy using logical profile and target names:

```yaml
schema_version: 3
name: example-difficulty
profiles:
  easy:
    target: groq-oss-20b
    parameters: {max_tokens: 2048, reasoning_effort: low}
  hard:
    target: groq-oss-120b
    parameters: {max_tokens: 4096, reasoning_effort: low}
method:
  type: difficulty
  classifier:
    type: llm
    target: groq-oss-20b
    prompt: {type: file, path: ../prompts/difficulty-v1.txt}
    projection: latest-user-text
    fallback: hard
    missing_input: hard
    max_prompt_chars: 1200
    parameters: {max_tokens: 20, temperature: 0.0}
  easy: easy
  hard: hard
limits:
  max_model_calls: 4
  max_buffered_bytes: 8388608
  deadline_seconds: 600
```

Supported methods are:

- `fixed`: execute one profile.
- `difficulty`: classify, then execute the easy or hard profile.
- `cascade`: classify; easy work is tried cheaply and either accepted or
  escalated according to the configured candidate policy.

Profiles own provider-neutral request transforms such as system prompts,
maximum output tokens, temperature, and reasoning effort. Methods own routing,
candidate assessment, and escalation. Every method receives the complete
conversation and explicitly chooses any projection used by a classifier.

Strategy YAML cannot define URLs, credential variable names, provider types,
or executable commands. A profile references a target ID from a trusted
`TargetRegistry`. The deployment owns each target's transport, physical model,
endpoint, credential handle, capabilities, timeouts, and hard
token ceiling. This keeps an untrusted strategy from redirecting requests or
reading arbitrary environment variables.

`load_strategy(...)` safely parses one YAML mapping, rejects duplicate and
unknown fields, resolves prompt files relative to the strategy, enforces
allowed roots and file-size limits, and computes a digest over configuration
and prompt contents. It does not read credentials or create clients.

`StrategyBuilder.build_engine(...)` resolves the validated policy against the
trusted registry and builds the single asynchronous runtime. Operators can
provide a different `TargetRegistry` without changing strategy files.

Bundled strategies are stored in
[`smart_ask/resources/strategies`](smart_ask/resources/strategies) and are
addressed as `builtin:NAME` by Python and terminal APIs.

## Claude Code

The optional adapter lets Claude Code use a strategy as if it were a model:

```text
Claude Code
  → Anthropic Messages adapter
  → StrategyEngine
  → strategy-selected trusted targets
```

SmartAsk contains no Claude Code protocol code. The external adapter translates
messages, tools, images, streaming events, token-count requests, authentication,
and model discovery. It passes the resulting complete `Conversation` to the
same engine used everywhere else.

The launcher defaults to `agentic-coding-v1`, creates a private adapter on a
random loopback port, and opens Claude Code:

```bash
cp scripts/claude-smart-ask.local.env.example \
  scripts/claude-smart-ask.local.env
# Add ANTHROPIC_API_KEY.

./scripts/claude-smart-ask
./scripts/claude-smart-ask -p "fix the failing tests" --print
./scripts/claude-smart-ask --trace
```

The default sends every substantive user message—including prompts containing
words such as `fix`—to a small Sonnet classifier. It returns `sonnet`, `opus`,
or `uncertain` plus confidence. Only an explicit exact-reply instruction skips
classification. Clear local and mechanical work goes to Sonnet; architecture,
subtle debugging, concurrency, algorithms, and uncertainty go to Opus. A
Sonnet tool loop is reclassified after each tool result, so new failure or
complexity evidence can escalate the same user turn. Once Opus owns a difficult
tool loop, it remains in control to preserve reasoning continuity.

When a route changes models, the warm source model creates a bounded factual
handoff. The destination receives the original request, summary, latest tool
output, and core coding tools instead of blindly rereading the full harness
context. That compact state persists for later tool calls. Empty, unsafe,
truncated, tool-calling, or failed summaries fall back to full context. This
reduces cold payload; it does not pretend that Anthropic prompt caches can be
shared across different models. After an edit and an unambiguous multi-test
pass, the same conservative mechanism can let Sonnet produce the final summary.

The confidence threshold is the coding-agent version of calibrated threshold
routing used by RouteLLM-style routers; post-exploration escalation follows the
same cheap-first principle as FrugalGPT-style cascades. The important addition
for an interactive coding harness is persistent, evidence-bounded handoff state
across the destination model's subsequent tool calls.

The shared efficiency guidance asks both models to reuse facts already gathered
and avoid duplicate orchestration. It does not ban source exploration,
subagents, environment discovery, reproduction, or broader tests when those add
evidence needed for correctness.

The launcher also stops one instruction after 30 model
responses or $2.00 of known spend by default; the full interactive session has
no fixed cumulative cap.

To reproduce the canonical real-repository evaluation on macOS:

```bash
.venv/bin/python benchmark/real_swebench.py \
  --label my-routed-run --strategy agentic-coding-v1
.venv/bin/python benchmark/real_swebench.py \
  --label my-opus-control --strategy agentic-coding-fixed-opus
```

This creates fresh checkouts of `pytest-dev/pytest` at the SWE-bench base
commit, gives the issue to Claude Code, applies the official hidden test patch
afterward, and records patch quality, model routes, cache tokens, and cost. See
[benchmark/REAL_SWEBENCH.md](benchmark/REAL_SWEBENCH.md) for the exact task and
latest measured results.

Use `--strategy NAME` only for an optional non-default strategy such as
`local-qwen`.

Each advertised Claude Code model name maps to exactly one strategy YAML. Claude
Code still supplies its own agent instructions and full conversation context;
changing the SmartAsk profile's system prompt adds to that context rather than
removing harness-owned instructions.

For the local Qwen setup, `./scripts/claude-local-qwen` also starts and checks
Ollama. See [scripts/README.md](scripts/README.md) and the
[adapter README](integrations/claude_code/README.md).

## Runs, metrics, and traces

Every engine invocation produces one `RunRecord`. Its normalized ledgers are:

```text
decisions
model calls
provider requests
```

A logical call expresses method intent. A provider request records the
physical request that executed it. Usage includes input, output, reasoning,
cache-read, and cache-write tokens when reported, plus provider cost, output
status, stop reason, time to first output, duration, and errors.

`RunMetricsStore` stores bounded canonical records and derives session
resources by model, target, profile, and role. The normal metrics file contains
no prompt content. An opt-in incremental trace records transformed call
contexts, chunked output, decisions, and terminal state, and is
therefore sensitive: it can contain source code, system prompts, tool inputs,
tool results, and secrets. The launcher writes local operational data under
`.smart-ask/`, which is ignored by Git.

Each traced launcher session gets a directory containing a live `session.json`
index and one append-only `.log` file per method invocation. The log presents a
classic line-oriented stream: timestamp, level, component, message, and compact
`key=value` evidence. `INFO` exposes the execution path, `DEBUG` records input
and thinking, and `WARN`/`ERROR` expose operational problems. Long content uses
explicit `begin`/`end` lines with indented continuations. Concurrent invocations
cannot interleave. Exact repeated inputs are identified without assuming that
they were retries.

Routing summaries should be derived from the decision and call ledgers. The
provider request started by a decision owns its tokens and cost; the same usage
must not be attributed to every earlier gate.

## Benchmarks

Benchmark suites call the same engine once per strategy/case pair and evaluate
only the selected final output:

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --strategy builtin:python-function-completion-fixed-gemini-self-check \
  --strategy builtin:python-function-completion-fixed-opus \
  --limit 50 \
  --workers 4 \
  --output benchmark-results/humaneval/example
```

The output directory contains:

```text
manifest.json   dataset, evaluator, strategies, target fingerprints, pricing
records.jsonl   one canonical result per strategy/case pair
summary.json    quality, resources, timing, routing, and comparisons
```

Fixed cheap and expensive baselines make counterfactual routing metrics
possible: unnecessary expensive routes, unsafe cheap routes, opportunity
capture, cost regret, and quality regret. Production traces alone cannot prove
those counterfactuals.

The LiveBench integration is a pinned public-test approximation, not an
official LiveBench score.

## Repository layout

| Path | Responsibility |
|---|---|
| `smart_ask/conversation/` | Immutable conversation values, engine, run evidence, metrics |
| `smart_ask/methods/` | Provider-independent fixed, difficulty, and cascade policies |
| `smart_ask/executors/` | Structured transports selected through trusted targets |
| `smart_ask/strategy/` | Schema v3, safe loading, target registry, composition |
| `smart_ask/metrics/` | Token usage and price estimation |
| `smart_ask/benchmarks/` | Suites, matrix runner, artifacts, summaries, counterfactuals |
| `integrations/claude_code/` | Separately installed Anthropic-compatible adapter |
| `scripts/` | Local process launchers |

See [DESIGN.md](DESIGN.md) for ownership rules and detailed lifecycle.
