# SmartAsk Claude Code adapter

This is a separately installable protocol adapter. It lets Claude Code select a
SmartAsk strategy as its model while keeping the dependency direction strict:

```text
Claude Code -> this adapter -> SmartAsk -> strategy-configured backend
```

The adapter translates HTTP requests and streams. It has no backend names,
provider credentials, or routing policy. SmartAsk loads each strategy, selects
the executor, performs routing and escalation, and records metrics.

From the repository root:

```bash
python3.11 -m pip install -e .
python3.11 -m pip install -e ./integrations/claude_code

ollama serve
export SMART_ASK_CLAUDE_CODE_TOKEN=local-secret
smart-ask-claude-code serve --config claude-code-adapter.example.yaml
```

Then start Claude Code in another shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY="$SMART_ASK_CLAUDE_CODE_TOKEN"
claude --model claude-smart-ask-local-qwen
```

The example uses `builtin:local-qwen`, so it makes no OpenRouter request and
requires no OpenRouter credential. Add another strategy reference to the
adapter YAML to expose another `claude-smart-ask-{yaml-stem}` alias.

To use the bundled first-party OpenAI Codex cascade instead:

```bash
# Set OPENAI_API_KEY in scripts/claude-smart-ask.local.env first.
./scripts/claude-smart-ask \
  --strategy python-code-generation-codex-cascade
```

The general launcher creates a private adapter configuration, selects its
advertised model alias, launches Claude Code, and cleans up the adapter. The
equivalent manual setup is:

```bash
export OPENAI_API_KEY="sk-..."
export SMART_ASK_CLAUDE_CODE_TOKEN="local-secret"
smart-ask-claude-code serve --config claude-code-openai-codex.example.yaml

export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY="$SMART_ASK_CLAUDE_CODE_TOKEN"
claude --model claude-smart-ask-python-code-generation-codex-cascade
```

The adapter still knows only the strategy alias. SmartAsk reads the YAML and
uses `OPENAI_API_KEY`; the Claude Code process never selects or calls the two
underlying Codex models itself.

When `metrics.jsonl_path` is configured, the adapter appends the prompt-free run
and session metrics envelopes produced by SmartAsk. It does not calculate or
reinterpret provider usage itself.

For routing debugging, `metrics.trace_jsonl_path` enables a separate opt-in
conversation trace. Each JSONL line is one semantic event: a context block,
strategy context change, route decision, attempt boundary, output chunk, or run
boundary. The original context is written once and attempt-specific changes are
recorded as patches, avoiding repeated copies of the whole conversation. A
single file header declares the schema, session, and strategy. Each run's full
ID appears only on its `run_start`; later events use a small local `run` number
so concurrent hidden requests can still be distinguished without repeating
UUIDs. Repeated context blocks, request metadata, and strategy changes are
defined once and referenced thereafter. Successful/default fields and values
inherited from the preceding route or attempt are omitted.
Traces can contain source code, tool results, system prompts, and secrets; keep
the file local and access-restricted.

For the bundled local-Qwen setup, `../../scripts/claude-local-qwen` provides a
single-command development launcher plus `start`, `status`, `logs`, and `stop`
commands. It is intentionally outside this generic adapter package because
starting Ollama is deployment-specific.
