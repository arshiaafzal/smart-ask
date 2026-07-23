#!/usr/bin/env python3
"""Run one genuine SWE-bench task in its upstream repository.

The agent only sees the issue and the repository at ``base_commit``.  The
official SWE-bench test patch is applied after the agent exits, immediately
before evaluation, matching the benchmark's hidden-test contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS_FILE = ROOT / "swe_bench_tasks.json"
STATE_ROOT = ROOT / ".smart-ask" / "swebench"
RESULTS_ROOT = ROOT / "benchmark" / "results-real"
DEFAULT_TASK = "pytest-dev__pytest-11143"

# Reproducible local environments for the real tasks we actively support.
# The pins reflect the dependency generation of each upstream base commit.
SETUP_COMMANDS: dict[str, list[list[str]]] = {
    "pallets__flask-4045": [
        ["uv", "venv", "--python", "3.9", ".venv"],
        [
            "uv", "pip", "install", "--python", ".venv/bin/python",
            "-e", ".", "pytest<8", "Werkzeug<2.1", "Jinja2<3.1",
            "itsdangerous<2.1", "click<8.1",
        ],
    ],
    "pytest-dev__pytest-11143": [
        ["uv", "venv", "--python", "3.11", ".venv"],
        [
            "uv", "pip", "install", "--python", ".venv/bin/python",
            "-e", ".[test]",
        ],
    ],
}

# Some SWE-bench PASS_TO_PASS node IDs contain shortened parametrized IDs that
# pytest cannot select directly. For these Mac-local cases, run the complete
# affected file, which is a stricter superset of the official regression list.
LOCAL_REGRESSION_TARGETS: dict[str, list[str]] = {
    "pytest-dev__pytest-11143": ["testing/test_assertrewrite.py"],
}


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    output: Path | None = None,
    input_text: str | None = None,
    timeout: int | None = None,
    kill_process_group: bool = False,
) -> subprocess.CompletedProcess[str]:
    options: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "text": True,
        "input": input_text,
        "timeout": timeout,
    }
    if output is None:
        options["stdout"] = subprocess.PIPE
        options["stderr"] = subprocess.STDOUT
    else:
        handle = output.open("w", encoding="utf-8")
        options["stdout"] = handle
        options["stderr"] = subprocess.STDOUT
    try:
        if not kill_process_group:
            try:
                return subprocess.run(command, **options)
            except subprocess.TimeoutExpired as exc:
                return subprocess.CompletedProcess(
                    command,
                    124,
                    stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                )

        run_options = dict(options)
        run_options.pop("timeout", None)
        run_options.pop("input", None)
        process = subprocess.Popen(command, start_new_session=True, **run_options)
        try:
            stdout, _ = process.communicate(input=input_text, timeout=timeout)
            return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate()
            return subprocess.CompletedProcess(command, 124, stdout=stdout)
    finally:
        if output is not None:
            handle.close()


def _tasks() -> dict[str, dict[str, Any]]:
    rows = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return {row["instance_id"]: row for row in rows}


def _selectors(task: dict[str, Any], name: str) -> list[str]:
    value = task[name]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must contain test selector strings")
    return value


def _safe_label(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ValueError("label must use only letters, numbers, dot, underscore, dash")
    return value


def _cache_repo(task: dict[str, Any]) -> Path:
    cache = STATE_ROOT / "cache" / task["repo"].replace("/", "__")
    if cache.exists():
        completed = _run(
            ["git", "cat-file", "-e", f"{task['base_commit']}^{{commit}}"],
            cwd=cache,
        )
        if completed.returncode:
            fetched = _run(
                ["git", "fetch", "--quiet", "origin", task["base_commit"]],
                cwd=cache,
            )
            if fetched.returncode:
                raise RuntimeError(f"could not fetch base commit: {fetched.stdout}")
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        ["git", "clone", "--quiet", f"https://github.com/{task['repo']}.git", str(cache)],
        cwd=ROOT,
    )
    if completed.returncode:
        raise RuntimeError(f"could not clone {task['repo']}: {completed.stdout}")
    return cache


def _fresh_checkout(task: dict[str, Any], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark workspace: {destination}"
        )
    cache = _cache_repo(task)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        ["git", "clone", "--quiet", "--shared", str(cache), str(destination)],
        cwd=ROOT,
    )
    if completed.returncode:
        raise RuntimeError(f"could not create benchmark checkout: {completed.stdout}")
    checked_out = _run(
        ["git", "checkout", "--quiet", "--detach", task["base_commit"]],
        cwd=destination,
    )
    if checked_out.returncode:
        raise RuntimeError(f"could not check out base commit: {checked_out.stdout}")


def _setup(task_id: str, checkout: Path, log: Path) -> None:
    commands = SETUP_COMMANDS.get(task_id)
    if not commands:
        raise ValueError(
            f"no reproducible local environment is defined for {task_id}; "
            "add one to SETUP_COMMANDS"
        )
    with log.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=checkout,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(f"environment setup failed; see {log}")


def _agent_prompt(
    task: dict[str, Any],
) -> str:
    return (
        "Fix the following issue in this repository. Work directly in the "
        "repository, run relevant tests, and stop only when the fix is complete. "
        "Do not modify tests. Use .venv/bin/python -m pytest for test commands.\n\n"
        + task["problem_statement"].strip()
    )


def _cache_system_prompt(cache_namespace: str) -> str:
    return (
        "Evaluation cache namespace: "
        + cache_namespace
        + ". This identifier has no bearing on the task or response."
    )


def _evaluate(task: dict[str, Any], checkout: Path, result_dir: Path) -> dict[str, Any]:
    model_patch = _run(["git", "diff", "--binary"], cwd=checkout)
    (result_dir / "model.patch").write_text(model_patch.stdout, encoding="utf-8")
    patch_result = _run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=checkout,
        input_text=task["test_patch"],
    )
    if patch_result.returncode:
        return {
            "passed": False,
            "evaluation_error": "official test patch did not apply",
            "test_patch_output": patch_result.stdout,
        }

    python = checkout / ".venv" / "bin" / "python"
    fail_to_pass = _selectors(task, "FAIL_TO_PASS")
    pass_to_pass = _selectors(task, "PASS_TO_PASS")
    regression_targets = LOCAL_REGRESSION_TARGETS.get(
        task["instance_id"], pass_to_pass
    )
    f2p = _run(
        [str(python), "-m", "pytest", "-q", *fail_to_pass],
        cwd=checkout,
        output=result_dir / "fail-to-pass.txt",
        timeout=900,
    )
    p2p = _run(
        [str(python), "-m", "pytest", "-q", *regression_targets],
        cwd=checkout,
        output=result_dir / "pass-to-pass.txt",
        timeout=900,
    )
    return {
        "passed": f2p.returncode == 0 and p2p.returncode == 0,
        "fail_to_pass": {
            "passed": f2p.returncode == 0,
            "selectors": fail_to_pass,
            "exit_code": f2p.returncode,
        },
        "pass_to_pass": {
            "passed": p2p.returncode == 0,
            "selectors": pass_to_pass,
            "executed_targets": regression_targets,
            "scope": (
                "affected-file superset"
                if regression_targets != pass_to_pass
                else "official selectors"
            ),
            "exit_code": p2p.returncode,
        },
    }


def run_task(args: argparse.Namespace) -> int:
    tasks = _tasks()
    if args.task not in tasks:
        raise ValueError(f"unknown SWE-bench task: {args.task}")
    task = tasks[args.task]
    label = _safe_label(args.label)
    result_dir = RESULTS_ROOT / label / args.task
    if result_dir.exists():
        raise FileExistsError(f"result already exists: {result_dir}")
    result_dir.mkdir(parents=True)
    checkout = STATE_ROOT / "runs" / label / args.task
    _fresh_checkout(task, checkout)
    _setup(args.task, checkout, result_dir / "setup.txt")

    metrics = result_dir / "metrics.jsonl"
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    env["SMART_ASK_METRICS_PATH"] = str(metrics)
    started = datetime.now(timezone.utc)
    cache_namespace = sha256(
        f"{label}\0{started.isoformat()}".encode("utf-8")
    ).hexdigest()[:16]
    command = [
        str(ROOT / "scripts" / "claude-smart-ask"),
        "--strategy", args.strategy,
        "--trace-dir", str(result_dir / "trace"),
        "--append-system-prompt", _cache_system_prompt(cache_namespace),
        "-p", _agent_prompt(task),
        "--print", "--dangerously-skip-permissions",
    ]
    agent = _run(
        command,
        cwd=checkout,
        env=env,
        output=result_dir / "agent.txt",
        timeout=args.timeout,
        kill_process_group=True,
    )
    evaluation = _evaluate(task, checkout, result_dir)
    finished = datetime.now(timezone.utc)
    model_patch = result_dir / "model.patch"
    result = {
        "schema": "smart-ask.real-swebench/v1",
        "instance_id": args.task,
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "strategy": args.strategy,
        "label": label,
        "cache_namespace": cache_namespace,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "agent_exit_code": agent.returncode,
        "agent_timed_out": agent.returncode == 124,
        "model_patch_sha256": sha256(model_patch.read_bytes()).hexdigest(),
        "official_test_patch_sha256": sha256(
            task["test_patch"].encode("utf-8")
        ).hexdigest(),
        **evaluation,
    }
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"results: {result_dir}")
    return 0 if result.get("passed") else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--strategy", default="agentic-coding-v1")
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    try:
        raise SystemExit(run_task(args))
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
