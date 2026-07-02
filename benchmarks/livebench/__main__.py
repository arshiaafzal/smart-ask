"""Run LiveBench coding for one or more strategy YAML files."""

from benchmarks.cli import run_suite_cli
from benchmarks.livebench.suite import LiveBenchSuite


def main(argv=None, **dependencies):
    return run_suite_cli(LiveBenchSuite(), argv, **dependencies)


if __name__ == "__main__":
    main()
