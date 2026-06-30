# smart-ask

A terminal CLI that routes AI tasks to the cheapest capable model — saving **80–95%** vs always using Claude Opus.

```
$ ask "explain how TCP handshake works"
  ▸  qwen/qwen3-coder  [easy]  cheap ✓

$ ask "design a distributed event-sourcing architecture"
  ▸  claude-opus-4.8   [hard]  powerful ✓
```

On startup (`ask` with no arguments) you get an animated dollar-rain teaser with the word **ask** crystallising from the storm.

---

## How it works

```
┌─────────────┐     ~$0.0001      ┌──────────────────┐
│  your task  │ ──→ haiku-4.5 ──→ │  easy / hard?    │
└─────────────┘   (classifier)    └──────────────────┘
                                         │
                    ┌────────────────────┴───────────────────┐
                    ▼                                         ▼
          qwen/qwen3-coder (480B)               claude-opus-4.8
             cheap & fast                         full power
```

1. You type `ask "your task"`
2. `claude-haiku-4.5` classifies it as **easy** or **hard** (one API call, ~$0.0001)
3. **Easy** → `qwen/qwen3-coder` via Hermes (cheap 480B MoE model)
4. **Hard** → `claude-opus-4.8` via OpenRouter (maximum capability)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | Comes with macOS / most Linux |
| [Hermes agent](https://github.com/NousResearch/hermes-agent) | Open-source agent harness |
| OpenRouter API key | [openrouter.ai](https://openrouter.ai) — get a free key |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/arshiaafzal/smart-ask.git
cd smart-ask
```

### 2. Install Hermes

Follow the [Hermes installation guide](https://github.com/NousResearch/hermes-agent). By default it lands at `~/hermes-agent/` with a venv at `~/hermes-agent/.venv/`.

If your Hermes lives somewhere else, update the shebang on line 1 of `smart-ask`:

```python
#!/path/to/your/hermes-venv/bin/python3
```

### 3. Install the openai package into the Hermes venv

```bash
~/hermes-agent/.venv/bin/pip install openai
```

### 4. Copy the script

```bash
cp smart-ask ~/.local/bin/smart-ask
chmod +x ~/.local/bin/smart-ask
```

### 5. Configure your shell (`~/.zshrc` or `~/.bashrc`)

```bash
export OPENROUTER_API_KEY="sk-or-..."   # your OpenRouter key
alias ask="smart-ask"
```

```bash
source ~/.zshrc   # reload
```

### 6. Verify

```bash
ask                        # shows animated teaser + help
ask --dry-run "hello"      # classify only, no model run
```

---

## Usage

```
ask "task"               auto-route + run
ask -i "task"            interactive — see agent work, approve tool calls
ask --force-hard "..."   always Claude Opus 4.8
ask --force-easy "..."   always Qwen
ask --dry-run "..."      classify only, skip run
ask -v "..."             verbose tool output
ask                      show help screen
```

### Examples

```bash
# Simple coding task → routed to Qwen (cheap)
ask "write a Python function to parse a CSV file"

# Architecture question → routed to Claude Opus
ask "design a fault-tolerant microservices system with CQRS"

# Interactive session — watch the agent, steer mid-task
ask -i "refactor my auth module to use JWT"

# Pipe a task from stdin
echo "explain async/await in JavaScript" | ask
```

---

## Models

| Role | Model | Provider | Cost |
|------|-------|----------|------|
| Classifier | `anthropic/claude-haiku-4.5` | OpenRouter | ~$0.0001 / call |
| Easy tasks | `qwen/qwen3-coder` | OpenRouter → Hermes | Very cheap |
| Hard tasks | `anthropic/claude-opus-4.8` | OpenRouter | Full price |

The **classifier threshold** is intentionally conservative — when in doubt it routes to Qwen and lets you override with `--force-hard` if needed.

---

## Interactive mode (`-i`)

`ask -i "task"` opens a full PTY session inside Hermes. Your prompt is auto-injected after Hermes boots (~2.5 s delay). You can:

- Watch the agent reason and use tools in real time
- Approve or reject tool calls
- Type follow-up messages mid-task
- Press `Ctrl-C` to abort

---

## Configuration reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key (`sk-or-...`) |

All three models (haiku classifier, Qwen, Opus) are accessed through OpenRouter — one key covers everything.

---

## Project structure

```
smart-ask/
├── smart-ask      # the CLI script (Python, no extra deps beyond openai)
├── skill.md       # Claude Code skill — lets your agent invoke smart-ask
└── README.md
```

---

## License

MIT
