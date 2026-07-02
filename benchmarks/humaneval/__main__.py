"""Run HumanEval for one or more strategy YAML files."""

from benchmarks.cli import run_suite_cli
from benchmarks.humaneval.suite import HumanEvalSuite


def main(argv=None, **dependencies):
    return run_suite_cli(HumanEvalSuite(), argv, **dependencies)


if __name__ == "__main__":
    main()
