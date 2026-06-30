---
name: smart-ask
description: Route AI tasks through the smart-ask cost router (cheap vs powerful model)
---

# smart-ask — AI Cost Router Skill

Use this skill when the user wants to run a task through **smart-ask**, or when you want to delegate a sub-task to the most cost-effective model available.

## What smart-ask does

`ask` is a CLI that:
1. Sends the task to `claude-haiku-4.5` for classification (~$0.0001)
2. Routes **easy** tasks to `qwen/qwen3-coder` (480B, cheap)
3. Routes **hard** tasks to `claude-opus-4.8` (powerful)

## Prerequisites

Before using this skill, verify setup:

```bash
# Check the CLI is installed
which smart-ask || echo "NOT INSTALLED — see README"

# Check OpenRouter key is set
echo ${OPENROUTER_API_KEY:0:8}...

# Verify Hermes is reachable
hermes --version
```

## How to invoke

```bash
ask "task description"           # auto-classify and run (recommended)
ask -i "task"                    # interactive: watch agent, approve tools
ask --force-hard "complex task"  # skip classifier → Claude Opus directly
ask --force-easy "simple task"   # skip classifier → Qwen directly
ask --dry-run "task"             # classify only, print route, don't run
ask -v "task"                    # verbose: show raw tool calls
```

## When to use which flag

| Situation | Flag |
|-----------|------|
| Normal use — let the classifier decide | (none) |
| Multi-step refactor, system design, research | `--force-hard` |
| Quick scripts, formatting, simple Q&A | `--force-easy` |
| Just want to know which model would be used | `--dry-run` |
| Need to approve each tool call before it runs | `-i` |

## Routing heuristic (what the classifier uses)

**Easy** (→ Qwen): Q&A, explain concepts, simple scripts, debug a single function, reformat code, write tests for existing code, shell one-liners.

**Hard** (→ Claude Opus): Complex architecture design, multi-system integration, novel algorithms, security analysis, large-scale refactoring, advanced research synthesis.

## Example agent workflow

When a user asks you to do something that involves running a shell task:

```bash
# Instead of running directly, route through smart-ask:
ask "write a bash script that monitors disk usage and alerts if above 90%"

# For something clearly complex:
ask --force-hard "design the database schema for a multi-tenant SaaS application with row-level security"

# Dry-run first to show the user what will happen:
ask --dry-run "explain the CAP theorem"
```

## Output interpretation

```
  ▸  qwen/qwen3-coder  [easy]  cheap ✓     → routed to Qwen
  ▸  claude-opus-4.8   [hard]  powerful ✓  → routed to Claude Opus
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `OPENROUTER_API_KEY not set` | `export OPENROUTER_API_KEY="sk-or-..."` in your shell |
| `hermes: command not found` | Add `~/hermes-agent/.venv/bin` to PATH or reinstall Hermes |
| Classifier always returns "easy" | Check your OpenRouter key has credits; classifier falls back to "easy" on error |
| PTY mode hangs (`-i`) | Press `Ctrl-C` to abort; check Hermes config at `~/.hermes/config.yaml` |

## Installation (for new collaborators)

```bash
git clone https://github.com/arshiaafzal/smart-ask.git
cd smart-ask

# Update shebang if Hermes is not at ~/hermes-agent/
head -1 smart-ask   # check current path

cp smart-ask ~/.local/bin/smart-ask
chmod +x ~/.local/bin/smart-ask

# Add to ~/.zshrc or ~/.bashrc:
export OPENROUTER_API_KEY="sk-or-your-key"
alias ask="smart-ask"

source ~/.zshrc
```
