"""Reporting surfaces.

Two of them: a terminal report for the person who ran the check, and a
self-contained HTML document for the person they have to convince. Both read
only a :class:`~llmverify.results.RunResult` and neither decides anything.
"""

from __future__ import annotations

from .console import ConsoleReporter
from .html import render_comparison_html, render_html

__all__ = ["ConsoleReporter", "render_comparison_html", "render_html"]
