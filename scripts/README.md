# Development launchers

This directory contains deployment-specific convenience scripts. These scripts
compose already-separated components for local development; they are not part
of the `smart_ask` Python package or its external protocol adapters.

## `claude-smart-ask`

`claude-smart-ask` is the general one-command Claude Code launcher. Give it a
strategy name from `smart_ask/resources/strategies/`, followed by normal Claude
Code arguments:

```bash
cp scripts/claude-smart-ask.local.env.example \
  scripts/claude-smart-ask.local.env
# Edit the local file and set the key required by your strategy.

./scripts/claude-smart-ask \
  --strategy python-code-generation-codex-cascade

./scripts/claude-smart-ask --strategy local-qwen -p "hello"
./scripts/claude-smart-ask --strategy python-code-generation-groq-cascade
./scripts/claude-smart-ask --strategy claude-code-groq-difficulty
./scripts/claude-smart-ask --strategy claude-code-groq-difficulty --trace
```

For each invocation it:

```text
strategy reference
  → validates and loads the strategy
  → generates a private, one-strategy adapter configuration
  → starts the external adapter on an available loopback port
  → discovers the exact model alias advertised by the adapter
  → launches Claude Code with that alias
  → stops the adapter and removes transient state when Claude exits
```

The strategy remains the source of truth for the backend, models, credentials,
prompts, and routing. The launcher passes provider credentials to the adapter
but removes the strategy's provider-key variables from the Claude Code child
process. It automatically generates loopback authentication and writes metrics
to `.smart-ask/claude-code/metrics.jsonl` by default. The resolved metrics path
is printed immediately before Claude Code starts.

Pass `--trace` immediately after the strategy name to also write full
conversation traces to a unique file under
`.smart-ask/claude-code/traces/`. The launcher prints the exact path;
that file contains every user turn, retry, escalation, and hidden Claude Code
request handled during that launcher session. Unlike metrics, it includes
prompts, complete session context, tool inputs/results, model outputs, and
escalation evidence. It is opt-in because that content may include source code
or secrets. Use `--trace-path PATH` or `SMART_ASK_TRACE_PATH` when one explicit
destination is preferred.

`.smart-ask/` is ignored by Git and contains local operational data only.
`benchmark-results/` remains reserved for deliberate benchmark artifacts.

The trace is event-oriented JSONL rather than one large object per request.
Context blocks are written once, strategy changes are separate patch events,
and long text/output fields are split into chunks of at most 4,000 characters.

Provider keys are loaded automatically from
`scripts/claude-smart-ask.local.env`. That file is ignored by Git; the tracked
`.local.env.example` documents the supported entries. Set
`SMART_ASK_SECRETS_FILE` to use a different local file. Normal exported
environment variables continue to work when the local file does not override
them.

The launcher intentionally does not start provider-specific services. For
`local-qwen`, run `ollama serve` first or use the specialized
`claude-local-qwen` launcher below. For an OpenAI strategy, export the configured
OpenAI key before starting it.

## `claude-local-qwen`

`claude-local-qwen` turns the local Claude Code + SmartAsk + Qwen setup into one
command:

```text
scripts/claude-local-qwen
  ├── ensures the local Ollama service is running
  ├── ensures the external Claude Code adapter is running
  ├── waits for both health checks
  ├── supplies local adapter authentication to Claude Code
  └── launches Claude Code with claude-smart-ask-local-qwen
```

The resulting request path remains:

```text
Claude Code
  → external Claude Code protocol adapter
  → SmartAsk conversation runtime
  → builtin:local-qwen strategy
  → OllamaConversationExecutor
  → qwen3:14b
```

The launcher does not route requests, translate protocol messages, execute
models, or calculate metrics. It only starts processes and connects their
existing public interfaces.

## Why this encapsulation matters

Each layer has one responsibility:

| Layer | Owns | Must not own |
|---|---|---|
| SmartAsk core | Strategy loading, routing, execution, escalation, metrics | Claude Code, HTTP routes, ASGI, SSE |
| External Claude Code adapter | Protocol translation, authentication, server limits | Provider selection, Ollama startup, routing policy |
| Ollama executor | Encoding neutral conversations for Ollama | Claude Code protocol behavior |
| Local launcher | Development process startup, readiness, PIDs, logs | Application behavior from any layer above |

Starting Ollama is specific to the local-Qwen deployment. Putting that behavior
inside the generic adapter would make the adapter provider-aware. Putting it
inside SmartAsk would invert the intended dependency direction. Keeping it in a
development launcher preserves the invariant:

```text
external adapter → SmartAsk → strategy-configured backend
```

## Usage

On first use, the launcher creates a private environment under its runtime state
directory and installs both checkout packages there. Subsequent runs reuse it:

```bash
./scripts/claude-local-qwen
```

Manual installation remains supported but is not required:

```bash
python3.11 -m pip install -e .
python3.11 -m pip install -e ./integrations/claude_code
```

Then use:

```bash
./scripts/claude-local-qwen                 # interactive Claude Code
./scripts/claude-local-qwen -p "your task"  # one-shot harness run
./scripts/claude-local-qwen start           # services only
./scripts/claude-local-qwen status          # health and ownership
./scripts/claude-local-qwen logs            # follow service logs
./scripts/claude-local-qwen stop            # stop owned services
```

Background services remain available between harness runs. `stop` only signals
processes whose PIDs were created by this launcher; an Ollama or adapter process
started elsewhere is left alone.

## Runtime state and security

By default, runtime state is stored under:

```text
${TMPDIR:-/tmp}/smart-ask-claude-local-qwen/
├── adapter.log
├── adapter.pid
├── ollama.log
├── ollama.pid
└── token
```

The directory and generated token use the invoking user's restrictive umask.
The token authenticates only the loopback adapter; it is not a model-provider
credential. The launcher removes `OPENROUTER_API_KEY` from the local adapter and
Claude Code child environments because `builtin:local-qwen` does not need it.

SmartAsk's prompt-free run/session metrics continue to be written to the path
configured in `claude-code-adapter.example.yaml`.

## Overrides

The launcher supports these environment overrides for alternate installations
and isolated tests:

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

This launcher is intended for local development and harness testing. Production
process supervision should use the deployment environment's service manager.
