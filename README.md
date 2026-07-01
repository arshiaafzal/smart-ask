# smart-ask

A terminal CLI that routes AI tasks to the cheapest capable model — saving **98%** vs always using Claude Opus.

```
$ ask "explain how TCP handshake works"
  ▸  gemini-2.5-flash-lite  [easy]  cheap ✓

$ ask "design a distributed event-sourcing architecture"
  ▸  claude-opus-4.8        [hard]  powerful ✓
```

On startup (`ask` with no arguments) you get an animated dollar-rain teaser, then a prompt asking what you want to build.

---

## How it works

```
┌─────────────┐     ~$0.0001      ┌──────────────────┐
│  your task  │ ──→ haiku-4.5 ──→ │  easy / hard?    │
└─────────────┘   (classifier)    └──────────────────┘
                                         │
                    ┌────────────────────┴───────────────────┐
                    ▼                                         ▼
       google/gemini-2.5-flash-lite            claude-opus-4.8
          cheap & fast (50x cheaper)              full power
```

1. You type `ask "your task"` (or just `ask` to be prompted)
2. `claude-haiku-4.5` classifies it as **easy** or **hard** (~$0.0001)
3. **Easy** → `google/gemini-2.5-flash-lite` via OpenRouter (50x cheaper than Opus)
4. **Hard** → `claude-opus-4.8` via OpenRouter (maximum capability)

All sessions run interactively inside Hermes — you can watch the agent work, steer mid-task, and approve/deny tool calls.

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
ask                        # shows animated teaser, then prompts for task
ask --dry-run "hello"      # classify only, no model run
```

---

## Usage

```
ask                      animated teaser → prompt for task → run
ask "task"               auto-route + run directly
ask --force-hard "..."   always Claude Opus 4.8
ask --force-easy "..."   always Gemini 3.5 Flash
ask --dry-run "..."      classify only, skip run
ask -v "..."             verbose tool output
ask --help               show help screen
```

### Examples

```bash
# Simple coding task → routed to Gemini (cheap)
ask "write a Python function to parse a CSV file"

# Architecture question → routed to Claude Opus
ask "design a fault-tolerant microservices system with CQRS"

# No argument — shows teaser then asks what you want to build
ask

# Pipe a task from stdin
echo "explain async/await in JavaScript" | ask
```

---

## Models

| Role | Model | Provider | Cost |
|------|-------|----------|------|
| Classifier | `anthropic/claude-haiku-4.5` | OpenRouter | ~$0.0001 / call |
| Easy tasks | `google/gemini-2.5-flash-lite` | OpenRouter | $0.0001 / 1M in, $0.0004 / 1M out |
| Hard tasks | `anthropic/claude-opus-4.8` | OpenRouter | $5 / 1M in, $25 / 1M out |

The easy model is **50x cheaper** than Opus per token. On our HumanEval benchmark (164 problems), Gemini cost $0.016 vs $0.82 for Opus — a **98% saving**.

The **classifier threshold** is intentionally conservative — when in doubt it routes to Gemini and lets you override with `--force-hard` if needed.

---

## Interactive sessions

Every session runs inside a full PTY session in Hermes. You can:

- Watch the agent reason and use tools in real time
- Approve or reject tool calls
- Type follow-up messages mid-task
- Press `Ctrl-C` to abort

---

## Configuration reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key (`sk-or-...`) |

All three models (haiku classifier, Gemini, Opus) are accessed through OpenRouter — one key covers everything.

---

## Project structure

```
smart-ask/
├── smart-ask                        # CLI executable — copy to ~/.local/bin/
├── benchmarks/
│   └── humaneval/
│       ├── run.py                   # HumanEval cost benchmark
│       ├── requirements.txt
│       └── README.md
├── skills/
│   └── smart-ask.md                 # Claude Code skill file
└── README.md
```

---

## License

MIT
