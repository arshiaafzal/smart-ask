#!/usr/bin/env bash
# Run all benchmark tasks through smart-code and evaluate accuracy.
#
# Usage:
#   ./benchmark/run_bench.sh [--strategy NAME] [--label LABEL] [--task TASK_NAME]
#
# LABEL is written under results/<label>/ so you can compare runs.
# Default strategy: agentic-coding-v1
# Default label:    run-$(date +%Y%m%d-%H%M%S)
#
# Requires smart-code to be on PATH (alias or script).
# Set PYBIN to the Python with pytest if needed (default: python3).

set -euo pipefail

BENCH_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASKS_DIR="$BENCH_DIR/tasks"
RESULTS_DIR="$BENCH_DIR/results"
STRATEGY="agentic-coding-v1"
LABEL=""
ONLY_TASK=""
RESET_TASKS=0
PYBIN="${PYBIN:-/Users/arshia.afzal/smart-ask/.venv/bin/python}"
SMART_CODE="${SMART_CODE:-$BENCH_DIR/../scripts/claude-smart-ask}"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)    STRATEGY="$2"; shift 2 ;;
        --label)       LABEL="$2"; shift 2 ;;
        --task)        ONLY_TASK="$2"; shift 2 ;;
        --reset-tasks) RESET_TASKS=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$LABEL" ]]; then
    LABEL="run-$(date +%Y%m%d-%H%M%S)"
fi

RUN_DIR="$RESULTS_DIR/$LABEL"
mkdir -p "$RUN_DIR"

# ── reset task files to bugged originals if requested ─────────────────────────
if [[ $RESET_TASKS -eq 1 ]]; then
    echo "  Resetting task impl.py files to originals..."
    for orig in "$TASKS_DIR"/*/impl.py.orig; do
        impl="${orig%.orig}"
        if [[ -f "$orig" ]]; then
            cp "$orig" "$impl"
            echo "    reset: $impl"
        fi
    done
    # LRU task (lives outside benchmark/tasks/).
    LRU_ORIG="$HOME/demo-lru/lru_cache_impl.py.orig"
    if [[ -f "$LRU_ORIG" ]]; then
        cp "$LRU_ORIG" "${LRU_ORIG%.orig}"
        echo "    reset: ${LRU_ORIG%.orig}"
    fi
    echo ""
fi

# ── task list ─────────────────────────────────────────────────────────────────
# Each entry: "task_dir_name|prompt"
TASKS=(
    "astropy_separability|Fix all bugs in impl.py so every test in test_impl.py passes."
    "django_file_upload|Fix all bugs in impl.py so every test in test_impl.py passes."
    "matplotlib_version_info|Fix all bugs in impl.py so every test in test_impl.py passes."
    "seaborn_hue_order|Fix all bugs in impl.py so every test in test_impl.py passes."
    "flask_blueprint_dots|Fix all bugs in impl.py so every test in test_impl.py passes."
    "requests_redirect_copy|Fix all bugs in impl.py so every test in test_impl.py passes."
    "pytest_assertion_rewrite|Fix all bugs in impl.py so every test in test_impl.py passes."
    "sklearn_ridge_cv|Fix all bugs in impl.py so every test in test_impl.py passes."
    "sphinx_inherited_members|Fix all bugs in impl.py so every test in test_impl.py passes."
    "sympy_ccode_sinc|Fix all bugs in impl.py so every test in test_impl.py passes."
)
# Add LRU task (lives outside benchmark/tasks/).
LRU_TASK="lru|Fix all bugs in lru_cache_impl.py so every test in test_lru_cache.py passes."
LRU_DIR="$HOME/demo-lru"

pass=0
fail=0
error=0
total=0

run_task() {
    local task_name="$1"
    local task_dir="$2"
    local prompt="$3"
    local metrics_file="$RUN_DIR/${task_name}.jsonl"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Task: $task_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -d "$task_dir" ]]; then
        echo "  ERROR: task directory not found: $task_dir" >&2
        ((error++)) || true
        return
    fi

    # Run smart-code from the task directory so Claude Code sees the right files.
    # Unset CLAUDECODE to allow running from inside an existing Claude Code session.
    local exit_code=0
    (
        cd "$task_dir"
        unset CLAUDECODE
        SMART_ASK_METRICS_PATH="$metrics_file" \
            "$SMART_CODE" --strategy "$STRATEGY" \
            -p "$prompt" --print --dangerously-skip-permissions
    ) > "$RUN_DIR/${task_name}.output.txt" 2>&1 || exit_code=$?

    # Evaluate accuracy by running tests.
    local test_file
    if [[ "$task_name" == "lru" ]]; then
        test_file="test_lru_cache.py"
    else
        test_file="test_impl.py"
    fi

    local test_result=0
    (cd "$task_dir" && "$PYBIN" -m pytest "$test_file" -q) \
        > "$RUN_DIR/${task_name}.test.txt" 2>&1 || test_result=$?

    if [[ $test_result -eq 0 ]]; then
        echo "  ✓ PASS (tests green)"
        ((pass++)) || true
    else
        echo "  ✗ FAIL (tests still failing)"
        tail -3 "$RUN_DIR/${task_name}.test.txt"
        ((fail++)) || true
    fi

    ((total++)) || true
}

# ── run tasks ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Benchmark run: $LABEL"
echo "  Strategy:      $STRATEGY"
echo "  Results dir:   $RUN_DIR"
echo "════════════════════════════════════════════════════"

for entry in "${TASKS[@]}"; do
    task_name="${entry%%|*}"
    prompt="${entry##*|}"
    if [[ -n "$ONLY_TASK" && "$task_name" != "$ONLY_TASK" ]]; then
        continue
    fi
    run_task "$task_name" "$TASKS_DIR/$task_name" "$prompt"
done

# LRU task
if [[ -z "$ONLY_TASK" || "$ONLY_TASK" == "lru" ]]; then
    lru_prompt="${LRU_TASK##*|}"
    run_task "lru" "$LRU_DIR" "$lru_prompt"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Results: $pass/$total passed  ($fail failed, $error errors)"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Analyze cost:"
echo "    python3 $BENCH_DIR/analyze.py $RUN_DIR/"
echo ""
echo "  Compare with another run:"
echo "    python3 $BENCH_DIR/compare.py $RUN_DIR/ <other-run-dir>/"
echo ""
