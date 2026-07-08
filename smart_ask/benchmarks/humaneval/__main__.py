"""Run HumanEval for one or more strategy YAML files."""

from ..cli import run_suite_cli
from .suite import HumanEvalSuite


def main(argv=None, **dependencies):
    return run_suite_cli(HumanEvalSuite(), argv, **dependencies)


if __name__ == "__main__":
    main()
