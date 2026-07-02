---
name: smart-ask
description: Route AI tasks through a configurable smart-ask strategy
---

# smart-ask — Configurable Model Router

Use this skill when the user asks to run a task through smart-ask, validate or
select a smart-ask strategy, force one of its configured routes, or compare
several strategy YAMLs on a benchmark suite.

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
```

## Product commands

```bash
smart-ask "task description"                     # default product strategy
smart-ask --strategy path/to/strategy.yaml "task" # another complete strategy
smart-ask --validate-strategy --strategy FILE     # schema/prompt validation only
smart-ask --force-hard "complex task"             # configured hard profile, no classifier
smart-ask --force-easy "simple task"              # configured easy profile, no classifier
smart-ask --dry-run "task"                        # plan and print; skip generation
smart-ask -f FILE "task"                          # prepend file contents
smart-ask                                        # prompt for independent tasks
```

`ask` may be used as a local alias, but `smart-ask` is the installed command.

Force flags use profiles from the loaded YAML. Do not assume they always mean
Gemini or Opus when a custom strategy is selected. A fixed strategy has only its
declared profile and cannot be forced to the opposite decision.

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

python -m benchmarks.humaneval \
  --strategy strategies/python-function-completion-difficulty-v1.yaml \
  --strategy strategies/python-function-completion-difficulty-v2.yaml \
  --limit 20 \
  --workers 4 \
  --output benchmarks/results/humaneval/prompt-comparison
```

The benchmark writes schema-v3 `manifest.json`, `records.jsonl`, and
`summary.json`. Use `--resume` only with an explicit existing `--output`
directory. Benchmark controls stay on the command line; there is no separate
experiment configuration object.

## Output interpretation

```text
▸ gemini-2.5-flash-lite  [easy]  classified easy
▸ claude-opus-4.8        [hard]  classified hard
```

Direct OpenRouter generation strategies print captured response text. Hermes
strategies let Hermes own terminal output.

## Troubleshooting

| Error | Action |
|---|---|
| Strategy YAML or prompt error | Run `smart-ask --validate-strategy --strategy FILE`; paths are relative to the YAML |
| Required environment variable is not set | Export the `api_key_env` named in the strategy |
| `hermes: command not found` | Install Hermes or set `generation.command` to its executable |
| Classifier repeatedly falls back to easy | Check provider access and inspect the classifier prompt/model settings |
| Product routing cost is `unknown` | The configured classifier has no local price entry; token usage is still retained |
| Benchmark rejects a model price | Add that configured model to the benchmark price catalog before running |
| Resume manifest mismatch | Use the original suite/strategies/cases/pricing or start a new output directory |

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
