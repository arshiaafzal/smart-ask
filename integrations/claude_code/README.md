# SmartAsk Claude Code adapter

This separately installable adapter exposes SmartAsk strategies through the
Anthropic Messages protocol used by Claude Code:

```text
Claude Code
  → this adapter
  → SmartAsk StrategyEngine
  → strategy-selected trusted target
```

The adapter translates HTTP requests and streaming events. It owns
authentication, model discovery, request/concurrency limits, tools, images,
reasoning blocks, token-count requests, and Anthropic-compatible framing. It
does not choose providers, models, routes, or escalation behavior.

Each advertised Claude Code model name maps to exactly one strategy YAML. On
every Claude Code model request, the adapter builds one immutable `Conversation`
containing the complete supplied context and invokes that strategy once. The
strategy may make hidden classifier or candidate calls before selecting the one
response streamed back to Claude Code.

## One-command launch

From the repository root:

```bash
python3.11 -m pip install -e .
python3.11 -m pip install -e ./integrations/claude_code

cp scripts/claude-smart-ask.local.env.example \
  scripts/claude-smart-ask.local.env
# Add the key required by the strategy's trusted targets.

./scripts/claude-smart-ask \
  --strategy python-code-generation-codex-cascade

./scripts/claude-smart-ask \
  --strategy claude-code-groq-difficulty \
  --trace
```

The launcher searches the bundled strategies directory, generates a private
adapter configuration, starts the server on loopback, selects its advertised
alias, launches Claude Code, and cleans up the owned server afterward.

For local Qwen:

```bash
ollama serve
./scripts/claude-smart-ask --strategy local-qwen
```

Or use `../../scripts/claude-local-qwen`, which also starts and checks Ollama.

## Manual launch

Install both packages, configure the adapter token, and start the server:

```bash
python3.11 -m pip install -e .
python3.11 -m pip install -e ./integrations/claude_code

export SMART_ASK_CLAUDE_CODE_TOKEN="local-secret"
smart-ask-claude-code serve --config claude-code-adapter.example.yaml
```

In another shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY="$SMART_ASK_CLAUDE_CODE_TOKEN"
claude --model claude-smart-ask-local-qwen
```

The example adapter configuration uses schema version 1 and lists strategy
references:

```yaml
schema_version: 1
listen: {host: 127.0.0.1, port: 8787}
auth:
  token_env: SMART_ASK_CLAUDE_CODE_TOKEN
strategies:
  - builtin:local-qwen
limits:
  max_request_bytes: 33554432
  max_concurrent_requests: 32
metrics:
  jsonl_path: .smart-ask/claude-code/metrics.jsonl
security:
  allowed_strategy_roots: []
```

Adapter schema version 1 and SmartAsk strategy schema version 3 are independent
formats. A custom strategy file must be within an explicitly allowed absolute
root. Bundled strategies remain available through their installed resource
names.

The adapter alias is derived from the strategy filename, for example:

```text
builtin:local-qwen
  → claude-smart-ask-local-qwen

builtin:python-code-generation-codex-cascade
  → claude-smart-ask-python-code-generation-codex-cascade
```

The alias does not expose the cheap and expensive physical models. Claude Code
selects the strategy; the strategy selects trusted target profiles internally.

## Context and harness instructions

Claude Code sends its full current conversation, tools, and agent system
instructions on each request. The adapter preserves them. A strategy profile
may append a system instruction, but it does not delete harness-owned context.
This is why a non-Anthropic backend may still describe itself as Claude Code:
it is following the harness instructions it received.

Reasoning effort is also an incoming request parameter. The adapter preserves
it, and a strategy profile may apply its own explicit parameter transform. The
resolved transformed conversation is what the selected target receives.

## Metrics and traces

When `metrics.jsonl_path` is configured, every completed invocation appends:

- one canonical prompt-free run record;
- the current aggregate for its session.

The adapter does not recalculate routing facts. It persists the engine's
decision, logical-call, and provider-request ledgers. Tokens, cost,
timing, completion, and error summaries derive from those records.

Set `metrics.trace_directory` or launch with `--trace` for content-bearing
debugging. One private directory represents a launcher session:

```text
<trace-directory>/
├── session.json
├── 001-<run-id>.jsonl
├── 002-<run-id>.jsonl
└── ...
```

The live `session.json` index links every Claude Code model request to one
self-contained invocation file containing:

1. the complete immutable conversation once;
2. every transformed classifier/candidate/escalation call context;
3. chunked thinking, text, and tool output as it arrives;
4. ordered strategy decisions and usage;
5. terminal state.

The index stores shared session/strategy contexts and input digests once.
`same_input_as` identifies an earlier invocation with exactly the same logical
input; it deliberately does not claim that the later request was a retry.

Inside an invocation file, the first `conversation` event is the complete run
input. A transformed `model_call` either stores a complete conversation or
uses `conversation_ref: run_input` plus a `replace` mapping. Reconstruction is
exact: replace the named top-level conversation components with the recorded
values. Short completed output blocks use one `model_output` event; long blocks
retain incremental start/chunk/end events.

Invocation files are written during execution, so a slow call appears before
it finishes. The index reports `running` until the invocation reaches a
terminal state. Schema and long run IDs occur only at file boundaries.

This makes it possible to see which prompt was escalated, what context the
method saw, and which evidence caused the decision. Trace directories can contain
source code, system instructions, tool arguments/results, and secrets. Keep
them local and access-restricted.

## Encapsulation rules

- SmartAsk core never imports this package or implements Anthropic routes.
- The adapter never interprets difficulty, cheap/hard profiles, or candidate
  markers.
- Strategy YAML never supplies provider URLs or credential names.
- The launcher, not this adapter, starts deployment-specific services such as
  Ollama.
