"""Reporting surfaces.

Two of them: a terminal report for the person who ran the check, and a
self-contained HTML document for the person they have to convince. Both read
only a :class:`~llmverify.results.RunResult` and neither decides anything.

The three module-level functions are the hooks
:mod:`llmverify.cli` looks for by name. They exist so that the command line can
work with this package absent -- it falls back to a plainer renderer of its own
-- and so that the richer output is used whenever it is present.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..results import RunResult
from .console import ConsoleReporter
from .html import render_comparison_html, render_html

__all__ = [
    "ConsoleReporter",
    "render_comparison",
    "render_comparison_html",
    "render_html",
    "render_result",
    "write_html",
]


def render_result(result: RunResult, *, console: Any = None, verbose: bool = False) -> None:
    """Print one run to the terminal."""
    ConsoleReporter(console, verbose=verbose).render(result)


def render_comparison(results: Sequence[RunResult], *, console: Any = None) -> None:
    """Print several runs side by side, best verdict first."""
    ConsoleReporter(console).render_comparison(list(results))


def write_html(results: Sequence[RunResult], path: Path) -> Path:
    """Write one self-contained HTML document covering ``results``.

    A single run gets the full report; several get the comparison, which is a
    different document rather than the same one repeated.
    """
    runs = list(results)
    if not runs:
        raise ValueError("write_html needs at least one run")
    if len(runs) == 1:
        return render_html(runs[0], path=path)
    return render_comparison_html(runs, path=path)
