"""Run the LiveBench public-test smoke suite for one or more strategies."""

from ..cli import run_suite_cli
from .suite import LiveBenchPublicTestsSuite


def main(argv=None, **dependencies):
    return run_suite_cli(LiveBenchPublicTestsSuite(), argv, **dependencies)


if __name__ == "__main__":
    main()
