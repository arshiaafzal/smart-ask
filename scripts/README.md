# Development launchers

These scripts make local harness testing convenient. They compose public
interfaces; they do not contain routing, provider translation, or model policy.

## `claude-smart-ask`

The general launcher accepts a bundled strategy name and normal Claude Code
arguments. The default strategy needs no strategy flag:

```bash
cp scripts/claude-smart-ask.local.env.example \
  scripts/claude-smart-ask.local.env
# Add ANTHROPIC_API_KEY.

./scripts/claude-smart-ask

./scripts/claude-smart-ask -p "fix the failing tests" --print

./scripts/claude-smart-ask --trace
```

The default `agentic-coding-v1` strategy routes straightforward turns to
Claude Sonnet and difficult turns to Claude Opus through the native Anthropic
API. `--strategy NAME` optionally selects another bundled strategy under
`smart_ask/resources/strategies/NAME.yaml`. The launcher intentionally does not
accept provider configuration in place of a strategy.

Routing is reconsidered for each real user message. Once a message starts an
agentic read/edit/test loop, its selected model remains stable for that loop so
Anthropic prompt-cache reuse and reasoning continuity are preserved. Generated
system reminders and moving cache annotations do not create false new turns.
After a real edit and a clear multi-test pass, the default policy can hand only
the bounded final-summary evidence to Sonnet. Sonnet cannot use tools in that
handoff, and ambiguity or failure returns to Opus with the complete context.
Obvious code actions and exact-reply requests bypass the LLM classifier. A
five-minute hard-model lease also keeps a warmed Opus context across new human
messages, because one cached Opus request can be cheaper than rebuilding the
same large prefix for Sonnet. Recent user/tool boundaries use spare Anthropic
cache breakpoints to improve prefix reuse.

Both profiles also receive correctness-first efficiency guidance: reuse known
facts, avoid duplicate reads and bookkeeping, and follow an explicitly supplied
environment before probing. Exploration and environment discovery remain
available whenever they provide information needed for a correct result.

For each invocation it:

```text
strategy name
  → validate and load schema-v3 YAML
  → discover required credentials from trusted targets
  → generate a private one-strategy adapter configuration
  → start the external adapter on a free loopback port
  → discover its advertised Claude Code model alias
  → print the metrics path and optional trace directory
  → launch Claude Code with that alias
  → stop the owned adapter and remove transient state
```

The strategy owns profiles, prompts, routing, and target IDs. The trusted
target registry owns the physical model, transport, endpoint, credential
handle, and hard limits. The launcher only supplies required environment values
and connects processes.

### Local credentials

Provider keys are loaded automatically from the ignored file:

```text
scripts/claude-smart-ask.local.env
```

Use `SMART_ASK_SECRETS_FILE` to select another file. Exported environment
variables are also accepted. Provider credentials are passed to the adapter but
removed from the Claude Code child environment; Claude Code receives only the
generated loopback adapter token.

The script never writes real keys into a tracked file. The example file lists
supported names without secret values.

### Metrics and traces

The launcher prints its resolved metrics path before starting Claude Code. By
default it is:

```text
.smart-ask/claude-code/metrics.jsonl
```

Metrics include the selected and actual model, input/output/cache tokens,
routing decisions, timing, and cost. Anthropic costs are direct list-price
estimates unless the provider supplies billed cost. Every Claude Code response
also displays a colored footer with the answering model, this-turn cost, and
cumulative session cost. The adapter strips that footer from subsequent model
context.

This file is prompt-free. Each completed method invocation appends its
canonical record and current session aggregate.

The launcher also limits one human instruction to 30 model responses or $2.00
of known provider cost by default. A new user message gets a fresh budget. Set
`SMART_ASK_MAX_REQUESTS_PER_TURN` or
`SMART_ASK_MAX_COST_PER_TURN_USD` to choose different positive limits.

Add `--trace` immediately after the strategy name to create one unique trace
directory for this launcher session:

```bash
./scripts/claude-smart-ask --trace
```

Use `--trace-dir DIR` or `SMART_ASK_TRACE_DIR` for an explicit destination.
The launcher prints the directory before starting Claude Code. Its layout is:

```text
.smart-ask/claude-code/traces/<timestamp-id>/
├── session.json
├── 001-<run-id>.log
├── 002-<run-id>.log
└── ...
```

`session.json` is a live index of method invocations and their terminal
status. Each numbered file is an append-only, classic text log for one
invocation. Every event is one timestamped `LEVEL component message key=value`
line. `INFO` shows calls, decisions, output, usage, and terminal state; `DEBUG`
contains input and thinking; `WARN` and `ERROR` expose problems. Long content
uses `begin`/`end` lines with indented continuations.

The index normalizes repeated session/strategy context and identifies exact
input repetition with `same_input_as`. A call that reuses the input says
`context=run_input`; changed parameters are printed directly. Custom contexts
are logged only for calls that actually replace the invocation input.

Trace directories can contain source code, system instructions, tool inputs/results, and
secrets. They are opt-in and local. `.smart-ask/` is ignored by Git;
`benchmark-results/` is reserved for deliberate benchmark artifacts.

### What the launcher does not start

The general launcher does not start provider-specific services. Start Ollama
before using `local-qwen`, or use the specialized launcher below:

```bash
ollama serve
./scripts/claude-smart-ask --strategy local-qwen
```

## `claude-local-qwen`

This specialized launcher adds local service supervision:

```text
claude-local-qwen
  ├─ ensure Ollama is running
  ├─ ensure the Claude Code adapter is running
  ├─ wait for both health checks
  ├─ configure loopback authentication
  └─ launch Claude Code using the local-qwen strategy
```

The resulting request path is:

```text
Claude Code
  → external Anthropic Messages adapter
  → SmartAsk StrategyEngine
  → local-qwen profile
  → trusted local-qwen3-14b target
  → Ollama
```

Usage:

```bash
./scripts/claude-local-qwen                 # interactive harness
./scripts/claude-local-qwen -p "your task"  # one-shot harness request
./scripts/claude-local-qwen start           # services only
./scripts/claude-local-qwen status          # health and ownership
./scripts/claude-local-qwen logs            # follow service logs
./scripts/claude-local-qwen stop            # stop owned services
```

On first use it creates a private environment under its runtime state directory
and installs both checkout packages. Later runs reuse it. `stop` only signals
processes whose PID files were created by this launcher.

Default runtime state:

```text
${TMPDIR:-/tmp}/smart-ask-claude-local-qwen/
├── adapter.log
├── adapter.pid
├── ollama.log
├── ollama.pid
└── token
```

The directory and generated loopback token use restrictive permissions. The
token is not a provider credential.

Supported deployment overrides include:

```text
SMART_ASK_CLAUDE_CONFIG
SMART_ASK_CLAUDE_MODEL
SMART_ASK_CLAUDE_CODE_TOKEN
SMART_ASK_OLLAMA_BIN
SMART_ASK_ADAPTER_BIN
CLAUDE_BIN
SMART_ASK_PYTHON
SMART_ASK_AUTO_INSTALL
SMART_ASK_OLLAMA_URL
SMART_ASK_ADAPTER_URL
SMART_ASK_LAUNCHER_STATE_DIR
SMART_ASK_START_ATTEMPTS
```

## Encapsulation

| Layer | Owns | Does not own |
|---|---|---|
| SmartAsk core | Conversation execution, routing, hidden calls, run evidence | Claude Code HTTP/SSE behavior |
| Claude Code adapter | Protocol translation, authentication, discovery, server limits | Provider selection or routing policy |
| Target transport | Encoding a neutral model call for one approved backend | Harness semantics |
| Launcher | Process startup, readiness, environment wiring, logs, cleanup | Application decisions |

Starting Ollama belongs in a deployment launcher because it is specific to one
local target. The dependency direction remains:

```text
launcher → external adapter → SmartAsk → trusted target transport
```
