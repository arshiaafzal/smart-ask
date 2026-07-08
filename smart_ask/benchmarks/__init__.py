"""Benchmark execution, artifacts, and comparison APIs."""

from .artifacts import JsonlResultSink, MemoryResultSink, load_run
from .compare import compare, format_report, summarize
from .counterfactual import evaluate_counterfactual_routing
from .routing_analysis import derive_routing_flow
from .runner import BenchmarkApplication, BenchmarkRun, run_matrix
from .suite import BenchmarkCase, BenchmarkStrategy, BenchmarkSuite, Evaluation

__all__ = [
    "BenchmarkApplication",
    "BenchmarkCase",
    "BenchmarkRun",
    "BenchmarkStrategy",
    "BenchmarkSuite",
    "Evaluation",
    "JsonlResultSink",
    "MemoryResultSink",
    "compare",
    "derive_routing_flow",
    "evaluate_counterfactual_routing",
    "format_report",
    "load_run",
    "run_matrix",
    "summarize",
]
