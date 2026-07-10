# Development launchers

This directory contains deployment-specific convenience scripts. These scripts
compose already-separated components for local development; they are not part
of the `smart_ask` Python package or its external protocol adapters.

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
