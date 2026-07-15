---
name: smart-ask
description: Run, validate, inspect, and benchmark conversation-native SmartAsk strategies
---

# SmartAsk

Use this skill when the user wants to run a conversation through SmartAsk,
select or validate a strategy, inspect its run evidence, use it from Claude
Code, or compare strategies on a benchmark.

## Mental model

Every incoming message/request is one engine invocation:

```text
complete immutable Conversation
  → StrategyEngine
  → configured method performs hidden calls and routing
  → one visible response
  → canonical RunRecord
```

The caller owns conversation history and sends the complete applicable snapshot
on every turn. A session is a group of invocations, not one long-running method
call.

The YAML uses schema version 3. It describes logical profiles, request
transforms, and method policy. Profiles reference target IDs from the trusted
deployment registry. Do not put provider URLs, credential environment names,
or executable commands in strategy YAML.

## Prerequisites

```bash
python3.11 -m pip install -e .

# Optional benchmark support.
python3.11 -m pip install -e '.[bench]'

# Optional Anthropic gateway.
python3.11 -m pip install -e '.[anthropic-gateway]'
```

Set only the key required by the selected trusted target:

```bash
test -n "$OPENROUTER_API_KEY" && echo "OpenRouter key is set"
test -n "$OPENAI_API_KEY" && echo "OpenAI key is set"
test -n "$GROQ_API_KEY" && echo "Groq key is set"
```

Local Qwen uses Ollama instead of a provider key:

```bash
ollama serve
```

## Terminal commands

```bash
smart-ask "task description"
smart-ask --strategy builtin:local-qwen "hello"
smart-ask --strategy path/to/strategy.yaml "review this"
smart-ask --validate-strategy --strategy builtin:product
smart-ask --force-easy "use the configured easy profile"
smart-ask --force-hard "use the configured hard profile"
smart-ask -f FILE "use this file as context"
```

The terminal continues as one conversation after the first turn. Force flags
select profiles declared by the loaded strategy; they do not imply specific
models. A fixed strategy does not accept a force override.

## Claude Code harness

Prefer the general launcher:

```bash
cp scripts/claude-smart-ask.local.env.example \
  scripts/claude-smart-ask.local.env
# Put the required provider key in the ignored local file.

./scripts/claude-smart-ask \
  --strategy claude-code-groq-difficulty

./scripts/claude-smart-ask \
  --strategy python-code-generation-codex-cascade \
  --trace

./scripts/claude-smart-ask \
  --strategy local-qwen \
  -p "hello"
```

The argument after `--strategy` is a filename stem under
`smart_ask/resources/strategies/`. The launcher prints the metrics path and,
when enabled, the unique session trace directory before Claude Code starts.

The request path is:

```text
Claude Code → protocol gateway → StrategyEngine → trusted target
```

Every advertised Claude Code alias identifies one strategy. Claude Code still
sends its own system instructions and complete conversation; the gateway
preserves them. SmartAsk chooses the internal profile and backend.

For local service supervision:

```bash
./scripts/claude-local-qwen
./scripts/claude-local-qwen status
./scripts/claude-local-qwen logs
./scripts/claude-local-qwen stop
```

## Strategy YAML

Use schema version 3:

```yaml
schema_version: 3
name: example-fixed
profiles:
  main:
    target: local-qwen3-14b
    parameters: {max_tokens: 2048, reasoning_effort: low}
method:
  type: fixed
  role: generator
  profile: main
limits:
  max_model_calls: 2
  max_buffered_bytes: 8388608
  deadline_seconds: 600
```

Method types:

- `fixed`: one selected profile.
- `difficulty`: classifier followed by easy or hard profile.
- `cascade`: classifier, cheap candidate assessment, optional escalation.

Validate before a paid run:

```bash
smart-ask --validate-strategy --strategy path/to/strategy.yaml
```

Prompt-file paths are relative to the strategy source. Loading rejects
duplicate keys, unknown fields, disallowed roots, oversized files, missing
prompts, and invalid profile references. Building rejects targets absent from
the deployment registry or lacking required capabilities.

## Inspecting evidence

One completed invocation has:

- decisions: route gates, outcomes, reasons, selected profiles, evidence calls;
- logical model calls: role, phase, profile, target, causing decision;
- provider requests: physical attempts, actual model, tokens, cost, output
  status, timing, retries, and errors.

Normal metrics are prompt-free. The opt-in trace includes full conversation
content and can expose code, tool data, system prompts, or secrets.

Do not count unknown tokens or price as zero. Do not equate a successful
provider response with a correct benchmark result. Do not attribute a
multi-model task failure to the final model alone.

## Comparing strategies

```bash
python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-cascade \
  --strategy builtin:python-function-completion-fixed-gemini-self-check \
  --strategy builtin:python-function-completion-fixed-opus \
  --limit 50 \
  --workers 4 \
  --output benchmark-results/humaneval/comparison
```

Artifacts:

```text
manifest.json   reproducibility identity
records.jsonl   one canonical result per strategy/case pair
summary.json    quality, resource, route, and comparison summaries
```

Use matched cheap-only and expensive-only fixed strategies to measure routing
regret. A single routed production trace cannot reveal whether the unchosen
model would have succeeded more cheaply.

## Troubleshooting

| Symptom | Check |
|---|---|
| Invalid YAML or prompt | Validate the strategy; paths are relative to its file |
| Missing credential | Inspect the selected target and export its trusted credential variable |
| Unknown target | Add it to the deployment `TargetRegistry`, not the strategy YAML |
| Claude Code cannot connect | Check gateway readiness, loopback URL, token, and advertised alias |
| Local Qwen is slow | Check Ollama model load, context size, thinking mode, and hardware throughput |
| Empty visible output | Inspect stop reason, output status, reasoning tokens, and requested token cap |
| Benchmark resume mismatch | Use the original manifest identity or start a new output directory |
