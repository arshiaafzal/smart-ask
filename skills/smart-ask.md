---
name: smart-ask
description: Route AI tasks through a configurable smart-ask strategy
---

# smart-ask — Configurable Model Router

Use this skill when the user asks to run a task through smart-ask, validate or
select a smart-ask strategy, force one of its configured routes, or compare
several strategy YAMLs on a benchmark suite. Also use the separately installed
adapter to expose strategies as selectable Claude Code models.

## Default behavior

`smart-ask` loads the bundled `product.yaml` unless `--strategy` is supplied.
That configuration:

1. Classifies the task as easy or hard with Gemini 2.5 Flash Lite through a
   response-capturing OpenRouter executor.
2. Selects Gemini 2.5 Flash Lite for easy work or Claude Opus 4.8 for hard work.
3. Sends the selected task to Hermes for generation.

The model IDs, prompts, parameters, method, and transports come from YAML; they
are not fixed by the CLI.

## Prerequisites

```bash
# Validate the default strategy and Python dependencies without a model call.
smart-ask --validate-strategy

# Check the credential used by shipped OpenRouter configurations.
test -n "$OPENROUTER_API_KEY" && echo "OpenRouter key is set"

# Required when the selected strategy uses Hermes generation.
hermes --version
```

Install the runtime and command if needed:

```bash
python3.11 -m pip install .

# Install the external Claude Code adapter only when needed.
python3.11 -m pip install -e ./integrations/claude_code
```

## Product commands

```bash
smart-ask "task description"                     # default product strategy
smart-ask --strategy path/to/strategy.yaml "task" # another complete strategy
smart-ask --validate-strategy --strategy FILE     # schema/prompt validation only
smart-ask --force-hard "complex task"             # configured hard profile, no classifier
smart-ask --force-easy "simple task"              # configured easy profile, no classifier
smart-ask --dry-run "task"                        # classify/plan; skip generation
smart-ask -f FILE "task"                          # prepend file contents
smart-ask                                        # prompt for independent tasks
```

`ask` may be used as a local alias, but `smart-ask` is the installed command.

Force flags use profiles from the loaded YAML. Do not assume they always mean
Gemini or Opus when a custom strategy is selected. A fixed strategy has only its
declared profile and does not accept force overrides.

## Claude Code harness

```bash
ollama serve
export SMART_ASK_CLAUDE_CODE_TOKEN="local-secret"
smart-ask-claude-code serve --config claude-code-adapter.example.yaml

export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY="$SMART_ASK_CLAUDE_CODE_TOKEN"
claude --model claude-smart-ask-local-qwen
```

Every advertised alias identifies exactly one YAML. The external adapter only
translates the wire protocol; SmartAsk selects and invokes the backend from the
strategy. `builtin:local-qwen` therefore needs Ollama but no OpenRouter key.
OpenRouter credentials are needed only for strategies that configure it.

## Choosing a route

With the shipped product strategy:

- Easy: explanations, small scripts, local debugging, formatting, tests for
  existing code, and straightforward transformations.
- Hard: multi-system design, subtle algorithms, security analysis, large
  refactors, and advanced research synthesis.

Prefer normal classification unless the user explicitly requests a route or
the task clearly requires a forced profile.

## Comparing strategies

Install benchmark support, then repeat `--strategy` on a suite module:

```bash
python -m pip install -e '.[bench]'

python -m smart_ask.benchmarks.humaneval \
  --strategy builtin:python-function-completion-difficulty-v1 \
  --strategy builtin:python-function-completion-difficulty-v2 \
  --limit 20 \
  --workers 4 \
  --output benchmark-results/humaneval/prompt-comparison
```

The benchmark writes strict schema-v5 `manifest.json`, `records.jsonl`, and
`summary.json`. Each record has one canonical metrics-v2 envelope and a call
ledger; attempts and routing events reference calls rather than copying their
usage or cost. Summaries include explicit task outcomes, resource rollups,
routing transition/path ledgers, and counterfactual diagnostics when matching
fixed cheap/expensive baselines are present. A cascade cheap baseline must use
the same prompt suffix; the bundled `*-fixed-gemini-self-check` strategies do
so. Use `--resume` only with an explicit existing `--output` directory.
Benchmark controls stay on the command line;
there is no separate experiment configuration object. Supply
`--price-catalog JSON` for models not present in the bundled versioned catalog.

## Output interpretation

```text
▸ gemini-2.5-flash-lite  [easy]  classified easy
▸ claude-opus-4.8        [hard]  classified hard
```

Direct OpenRouter generation strategies print captured response text. Hermes
strategies let Hermes own terminal output. Every completed turn prints per-call
and turn/session token and cost accounting when those values are observable.

## Troubleshooting

| Error | Action |
|---|---|
| Strategy YAML or prompt error | Run `smart-ask --validate-strategy --strategy FILE`; paths are relative to the YAML |
| Required environment variable is not set | Export the `api_key_env` named in the strategy |
| `hermes: command not found` | Install Hermes or set `generation.command` to its executable |
| Classifier repeatedly falls back to easy | Check provider access and inspect the classifier prompt/model settings |
| Product routing cost is `unknown` | The configured classifier has no local price entry; token usage is still retained |
| Benchmark rejects a model price | Add that configured model to the benchmark price catalog before running |
| Resume manifest mismatch | Use the original suite/evaluator/strategies/cases/pricing/metrics schema or start a new output directory |
| Claude Code cannot reach SmartAsk | Verify the external adapter is running, its token matches `ANTHROPIC_API_KEY`, and the selected alias is configured |

## Installation for a new checkout

```bash
git clone https://github.com/arshiaafzal/smart-ask.git
cd smart-ask

python3.11 -m pip install .
smart-ask --validate-strategy

# Optional shorthand:
alias ask=smart-ask
export OPENROUTER_API_KEY="sk-or-your-key"
```
