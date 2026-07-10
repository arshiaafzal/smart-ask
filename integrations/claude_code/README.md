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
export OPENAI_API_KEY="sk-..."
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

For the bundled local-Qwen setup, `../../scripts/claude-local-qwen` provides a
single-command development launcher plus `start`, `status`, `logs`, and `stop`
commands. It is intentionally outside this generic adapter package because
starting Ollama is deployment-specific.
