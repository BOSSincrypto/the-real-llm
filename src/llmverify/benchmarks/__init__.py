"""Benchmarks, their datasets, and the paraphrasing that makes them honest.

Declared explicitly rather than left as an implicit namespace package, for the
same reason as :mod:`llmverify.adapters`: the set of benchmarks a verdict rests
on should not be extensible by whatever else happens to be installed. Custom
benchmarks register through the ``llmverify.benchmarks`` entry-point group.
"""

from __future__ import annotations

from .base import (
    Benchmark,
    BenchmarkItem,
    GradedResult,
    Variant,
    all_benchmarks,
    get_benchmark,
    register_benchmark,
)
from .datasets import DatasetLoader, DatasetSpec
from .paraphrase import defeats_substring_match, paraphrase

__all__ = [
    "Benchmark",
    "BenchmarkItem",
    "DatasetLoader",
    "DatasetSpec",
    "GradedResult",
    "Variant",
    "all_benchmarks",
    "defeats_substring_match",
    "get_benchmark",
    "paraphrase",
    "register_benchmark",
]
