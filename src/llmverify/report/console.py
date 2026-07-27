"""Terminal rendering of a run.

The console report has one job beyond legibility: make the *shape* of the
evidence visible. A verdict that rests on five probes from one family is a much
weaker thing than the same number of bans spread across five families, and the
arithmetic in :func:`~llmverify.evidence.aggregate` cannot say so on its own. So
the evidence table is grouped by layer, every contribution is shown signed and to
scale, and a per-family summary sits underneath the whole thing.

Nothing here decides anything. Every number displayed comes from the
:class:`~llmverify.results.RunResult`; this module only chooses what to show
first and how to make an adverse verdict auditable rather than merely alarming.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ..config import redact
from ..errors import ProviderError
from ..evidence import Evidence, EvidenceStatus, Verdict
from ..results import RunResult, compare_results

if TYPE_CHECKING:  # the runner imports this module, so its types load lazily
    from ..runner import ProgressCallback, ProgressEvent

__all__ = ["ConsoleReporter"]

_LN10 = math.log(10)

#: Colour per verdict. EVASION is deliberately off the green-to-red axis: it is
#: not a more severe MISMATCH, it is a statement that the measurements cannot be
#: trusted at all. The panel also changes shape for it, so the distinction
#: survives NO_COLOR and piped output.
_VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.MATCH: "bold green",
    Verdict.LIKELY_MATCH: "green",
    Verdict.INCONCLUSIVE: "yellow",
    Verdict.LIKELY_MISMATCH: "dark_orange",
    Verdict.MISMATCH: "bold red",
    Verdict.EVASION: "bold magenta",
}

_VERDICT_GLOSS: dict[Verdict, str] = {
    Verdict.MATCH: "The observations are what the claimed model should produce.",
    Verdict.LIKELY_MATCH: (
        "Consistent with the claimed model, on less evidence than a clean match."
    ),
    Verdict.INCONCLUSIVE: (
        "Not enough usable evidence to decide either way. This is a statement about the "
        "run, not about the provider."
    ),
    Verdict.LIKELY_MISMATCH: (
        "The endpoint behaves unlike the claimed model, on evidence short of the "
        "strongest band."
    ),
    Verdict.MISMATCH: "The endpoint behaves unlike the claimed model.",
    Verdict.EVASION: (
        "The endpoint's behaviour depends on whether an input is recognisable as a "
        "benchmark item, so every other measurement here was taken under conditions the "
        "provider chose rather than conditions you chose. This is not a point on the "
        "match/mismatch axis and cannot be read as one."
    ),
}

_STATUS_STYLE: dict[EvidenceStatus, str] = {
    EvidenceStatus.OK: "",
    EvidenceStatus.SKIPPED: "dim",
    EvidenceStatus.UNSUPPORTED: "cyan",
    EvidenceStatus.ERROR: "red",
    EvidenceStatus.TRUNCATED: "yellow",
}

_LAYER_TITLES: dict[int, str] = {
    0: "metadata, free",
    1: "cheap behavioural fingerprints",
    2: "capability probes",
    3: "statistical",
}

#: What a user could supply to make a skipped or truncated probe run, keyed by
#: substrings that appear in the probe's own explanation of why it did not.
_ENABLERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "baseline",
        ("baseline", "a/b", "side by side", "side-by-side"),
        "an API key for the model's official endpoint, passed as --baseline, so the run "
        "can measure a first-party endpoint side by side instead of relying on published "
        "numbers",
    ),
    (
        "dataset",
        ("hf_token", "huggingface", "gated", "licence", "license"),
        "a HuggingFace token in $HF_TOKEN, on an account that has accepted the dataset's "
        "licence",
    ),
    (
        "budget",
        ("budget", "ran out", "ceiling", "sample", "truncat", "afford"),
        "a larger budget: --max-samples, --max-cost or --max-wall",
    ),
)

_PROGRESS_DONE: dict[str, str] = {
    "ok": "ok",
    "done": "ok",
    "complete": "ok",
    "completed": "ok",
    "finished": "ok",
    "success": "ok",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "skipped": "skipped",
    "skip": "skipped",
    "unsupported": "unsupported",
    "truncated": "truncated",
}

_PROGRESS_RUNNING = frozenset({"start", "started", "starting", "running", "begin", "pending"})


class ConsoleReporter:
    """Renders progress, results and errors to a terminal.

    ``quiet`` keeps only the verdict and any warnings, which is what a CI job
    wants. ``verbose`` adds the untruncated detail strings and each probe's
    evidence data, which is what someone disputing a verdict wants.
    """

    def __init__(
        self, console: Console | None = None, *, quiet: bool = False, verbose: bool = False
    ) -> None:
        # Rich drops colour on its own for a non-tty; NO_COLOR is honoured
        # explicitly so the behaviour does not depend on the rich version.
        self.console = console or (
            Console(no_color=True) if os.environ.get("NO_COLOR") else Console()
        )
        self.quiet = quiet
        self.verbose = verbose
        self._progress: Progress | None = None
        self._tasks: dict[str, TaskID] = {}
        self._block = self._glyph("█", "#")
        self._axis = self._glyph("│", "|")

    # ------------------------------------------------------------------ progress

    def progress_callback(self) -> ProgressCallback:
        """A callback the runner can drive to show live per-probe progress.

        The returned callable is safe to hand to a runner that never emits a
        terminal state for a probe: the live display is torn down by the first
        ``render*`` call whatever happened before it.
        """
        if self.quiet:
            return lambda event: None
        return self._on_progress

    def _on_progress(self, event: ProgressEvent) -> None:
        state = (event.state or "").strip().lower()
        message = redact(event.message or "")

        if state in _PROGRESS_RUNNING:
            self._start_task(event, message)
            return

        outcome = _PROGRESS_DONE.get(state)
        if outcome is None:
            # An unrecognised state is treated as an update rather than dropped,
            # so a runner that grows a new state stays visible here.
            self._update_task(event, message)
            return

        self._finish_task(event, outcome, message)

    def _start_task(self, event: ProgressEvent, message: str) -> None:
        if not self.console.is_terminal:
            return
        progress = self._ensure_progress()
        description = f"L{event.layer} {event.probe}"
        if message:
            description = f"{description} - {message}"
        self._tasks[event.probe] = progress.add_task(description, total=None)

    def _update_task(self, event: ProgressEvent, message: str) -> None:
        task = self._tasks.get(event.probe)
        if task is None or self._progress is None:
            return
        description = f"L{event.layer} {event.probe}"
        if message:
            description = f"{description} - {message}"
        self._progress.update(task, description=description)

    def _finish_task(self, event: ProgressEvent, outcome: str, message: str) -> None:
        task = self._tasks.pop(event.probe, None)
        if task is not None and self._progress is not None:
            self._progress.remove_task(task)

        line = Text()
        line.append(f"{outcome:<11}", style=_OUTCOME_STYLE.get(outcome, ""))
        line.append(f"L{event.layer} ", style="dim")
        line.append(event.probe)
        line.append(f"  {event.elapsed:.1f}s", style="dim")
        if message:
            line.append(f"  {message}", style="dim")
        self._print(line)

        if not self._tasks:
            self._stop_progress()

    def _ensure_progress(self) -> Progress:
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                # markup=False: probe messages carry provider text, and a stray
                # square bracket in it must not be read as a style tag.
                TextColumn("{task.description}", style="progress.description", markup=False),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.start()
        return self._progress

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._tasks.clear()

    def _print(self, renderable: RenderableType) -> None:
        """Print above the live display when one is running."""
        target = self._progress.console if self._progress is not None else self.console
        target.print(renderable)

    # -------------------------------------------------------------------- render

    def render(self, result: RunResult) -> None:
        """Print the full report for one run."""
        self._stop_progress()

        if not self.quiet:
            self.console.print(self._header(result))
        self.console.print(self._verdict_panel(result))

        if not self.quiet:
            for renderable in self._evidence_tables(result):
                self.console.print(renderable)
            self.console.print(self._family_table(result))

        for renderable in self._notes(result):
            self.console.print(renderable)

        if result.verdict.verdict.is_adverse:
            strengthen = self._strengthen_panel(result)
            if strengthen is not None:
                self.console.print(strengthen)

        if self.verbose:
            for renderable in self._verbose_details(result):
                self.console.print(renderable)

    def render_comparison(self, results: list[RunResult]) -> None:
        """Print several providers side by side, best verdict first."""
        self._stop_progress()
        if not results:
            self.console.print(Text("no runs to compare", style="dim"))
            return

        summary = compare_results(results)
        table = Table(
            title="providers, ranked by verdict",
            box=box.SIMPLE_HEAVY,
            header_style="bold",
            title_justify="left",
        )
        table.add_column("provider")
        table.add_column("claimed model")
        table.add_column("verdict")
        table.add_column("P(claimed)", justify="right")
        table.add_column("evidence", justify="right")
        table.add_column("cost", justify="right")
        table.add_column("time", justify="right")

        for row in summary["providers"]:
            verdict = Verdict(row["verdict"])
            table.add_row(
                Text(redact(str(row["name"]))),
                Text(redact(str(row["claimed_model"]))),
                Text(verdict.value, style=_VERDICT_STYLE[verdict]),
                _format_probability(float(row["probability_genuine"])),
                f"{float(row['total_bans']):+.2f} ban",
                f"${float(row['estimated_cost_usd']):.4f}",
                f"{float(row['duration_s']):.1f}s",
            )
        self.console.print(table)

        warnings = [redact(str(w)) for w in summary["warnings"]]
        if warnings:
            self.console.print(
                Panel(
                    Text("\n".join(f"- {w}" for w in warnings)),
                    title="comparability",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )

    def render_error(self, exc: Exception) -> None:
        """Print an exception as a panel, with any provider response body."""
        self._stop_progress()

        body = Text(redact(str(exc)) or exc.__class__.__name__)
        if isinstance(exc, ProviderError):
            if exc.status is not None:
                body.append(f"\n\nHTTP status: {exc.status}", style="dim")
            if exc.body:
                excerpt = redact(exc.body)[:600]
                body.append("\n\nresponse body (truncated):\n", style="dim")
                body.append(excerpt)

        self.console.print(
            Panel(
                body,
                title=exc.__class__.__name__,
                border_style="red",
                box=box.ROUNDED,
            )
        )

    # -------------------------------------------------------------------- pieces

    def _header(self, result: RunResult) -> Panel:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right")
        grid.add_column()

        grid.add_row("provider", Text(redact(result.provider_name)))
        grid.add_row("endpoint", Text(_endpoint_host(result.base_url)))
        grid.add_row("requested model", Text(redact(result.requested_model)))
        if result.claimed_model != result.requested_model:
            grid.add_row("claimed model", Text(redact(result.claimed_model)))
        grid.add_row("api family", Text(result.api_family))

        if result.reference_as_of is None:
            reference = "none loaded"
        else:
            found = "model found" if result.reference_found else "model NOT in snapshot"
            reference = f"{result.reference_as_of.isoformat()} ({found})"
        grid.add_row("reference", Text(reference))
        grid.add_row(
            "run",
            Text(
                f"seed {result.seed} - layers "
                f"{', '.join(str(n) for n in result.layers) or 'none'}"
                + (" - baseline A/B" if result.baseline_used else "")
            ),
        )
        return Panel(
            grid,
            title=f"llmverify {result.tool_version}",
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )

    def _verdict_panel(self, result: RunResult) -> Panel:
        report = result.verdict
        verdict = report.verdict
        style = _VERDICT_STYLE[verdict]

        body = Text()
        body.append(verdict.value, style=style)
        body.append("\n\n")
        body.append("P(endpoint serves the claimed model)  ", style="dim")
        body.append(_format_probability(report.probability), style=style)
        body.append("\nweight of evidence                    ", style="dim")
        body.append(f"{report.total_bans:+.2f} ban")
        body.append(
            "   (positive supports the claim, negative refutes it)\n",
            style="dim",
        )
        body.append("cost, duration, samples               ", style="dim")
        body.append(
            f"${result.spent_usd:.4f} estimated - {result.duration_s:.1f}s - "
            f"{result.samples} samples"
        )
        body.append("\n\n")
        body.append(_VERDICT_GLOSS[verdict], style="dim")

        return Panel(
            body,
            title="verdict",
            title_align="left",
            border_style=style,
            # A different frame, not only a different colour: the distinction
            # between EVASION and MISMATCH has to survive a monochrome terminal.
            box=box.DOUBLE if verdict is Verdict.EVASION else box.HEAVY,
        )

    def _evidence_tables(self, result: RunResult) -> list[RenderableType]:
        layer_of = {timing.probe: timing.layer for timing in result.timings}
        grouped: dict[int, list[Evidence]] = {}
        for item in result.evidence:
            grouped.setdefault(layer_of.get(item.probe, -1), []).append(item)

        scale = max((abs(e.bans) for e in result.usable_evidence), default=0.0) or 1.0
        tables: list[RenderableType] = []
        for layer in sorted(grouped):
            title = _LAYER_TITLES.get(layer)
            heading = f"layer {layer} - {title}" if title else "probes"
            table = Table(
                title=heading,
                box=box.SIMPLE,
                header_style="bold",
                title_justify="left",
                title_style="bold",
                expand=True,
                pad_edge=False,
            )
            table.add_column("probe", no_wrap=True)
            table.add_column("check", no_wrap=True)
            table.add_column("status", no_wrap=True)
            table.add_column("ban", justify="right", no_wrap=True)
            table.add_column("contribution", no_wrap=True)
            table.add_column(
                "detail",
                ratio=1,
                no_wrap=not self.verbose,
                overflow="fold" if self.verbose else "ellipsis",
            )

            for item in sorted(grouped[layer], key=lambda e: (-abs(e.bans), e.probe, e.label)):
                usable = item.status is EvidenceStatus.OK
                table.add_row(
                    Text(item.probe),
                    Text(item.label),
                    Text(item.status.value, style=_STATUS_STYLE[item.status]),
                    f"{item.bans:+.2f}" if usable else "-",
                    self._bar(item.bans, scale) if usable else Text(""),
                    Text(redact(item.detail)),
                )
            tables.append(table)
        return tables

    def _bar(self, bans: float, scale: float, width: int = 7) -> Text:
        """A signed bar: refuting left of the axis, supporting right of it."""
        filled = min(width, round(abs(bans) / scale * width)) if scale > 0 else 0
        if filled == 0 and abs(bans) > 0:
            filled = 1
        left = self._block * filled if bans < 0 else ""
        right = self._block * filled if bans > 0 else ""

        bar = Text()
        bar.append(" " * (width - len(left)))
        bar.append(left, style="red")
        bar.append(self._axis, style="dim")
        bar.append(right, style="green")
        bar.append(" " * (width - len(right)))
        return bar

    def _family_table(self, result: RunResult) -> RenderableType:
        """Where the evidence came from, and how much damping removed."""
        raw: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in result.usable_evidence:
            raw[item.family] = raw.get(item.family, 0.0) + item.llr
            counts[item.family] = counts.get(item.family, 0) + 1

        totals = result.verdict.family_totals
        magnitude = sum(abs(v) for v in totals.values()) or 1.0

        table = Table(
            title="evidence by family",
            box=box.SIMPLE,
            header_style="bold",
            title_justify="left",
            title_style="bold",
        )
        table.add_column("family", no_wrap=True)
        table.add_column("probes", justify="right")
        table.add_column("raw", justify="right")
        table.add_column("after damping", justify="right")
        table.add_column("share", justify="right")
        table.add_column("", no_wrap=True)

        for family, damped in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
            raw_bans = raw.get(family, 0.0) / _LN10
            damped_bans = damped / _LN10
            bound = abs(raw_bans) - abs(damped_bans) > 0.05
            table.add_row(
                Text(family),
                str(counts.get(family, 0)),
                f"{raw_bans:+.2f}",
                f"{damped_bans:+.2f}",
                f"{abs(damped) / magnitude:.0%}",
                Text("damped", style="dim") if bound else "",
            )

        if not totals:
            return Text("no probe produced usable evidence.", style="dim")

        top_family, top_total = max(totals.items(), key=lambda kv: abs(kv[1]))
        if len(totals) > 1 and abs(top_total) / magnitude >= 0.6:
            note = Text(
                f"{abs(top_total) / magnitude:.0%} of the evidence weight comes from the "
                f"{top_family!r} family alone. Correlated probes within a family are damped "
                "toward its cap, but a verdict resting on one family is still a narrower "
                "result than the same number of bans drawn from several.",
                style="dim",
            )
            return Group(table, note)
        return table

    def _notes(self, result: RunResult) -> list[RenderableType]:
        out: list[RenderableType] = []
        notes = [redact(n) for n in result.verdict.notes]
        if notes:
            out.append(
                Panel(
                    Text("\n\n".join(notes)),
                    title="notes on this verdict",
                    title_align="left",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
        warnings = [redact(w) for w in result.warnings]
        if warnings:
            out.append(
                Panel(
                    Text("\n\n".join(f"- {w}" for w in warnings)),
                    title="warnings",
                    title_align="left",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
        return out

    def _strengthen_panel(self, result: RunResult) -> Panel | None:
        """Name what did not run, and what would let it.

        An adverse verdict is only as good as the evidence that was available to
        contradict it, so the probes that stayed silent are part of the finding.
        """
        buckets: dict[str, list[str]] = {}
        unavailable: list[str] = []

        for item in result.evidence:
            if item.status is EvidenceStatus.OK:
                continue
            haystack = f"{item.detail} {item.status.value}".lower()
            key = None
            for name, needles, _ in _ENABLERS:
                if any(needle in haystack for needle in needles):
                    key = name
                    break
            if key is None and item.status is EvidenceStatus.TRUNCATED:
                key = "budget"
            label = f"{item.probe}/{item.label}"
            if key is None:
                if item.status is EvidenceStatus.UNSUPPORTED:
                    unavailable.append(label)
                continue
            buckets.setdefault(key, []).append(label)

        if not buckets and not unavailable:
            return None

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column(overflow="fold")
        for name, _, remedy in _ENABLERS:
            probes = buckets.get(name)
            if not probes:
                continue
            entry = Text(remedy)
            entry.append(f"\nwould enable: {', '.join(sorted(set(probes)))}", style="dim")
            grid.add_row("-", entry)
        if unavailable:
            entry = Text(
                "these probes cannot run against this endpoint at all, whatever you supply; "
                "their silence is not evidence either way"
            )
            entry.append(f"\n{', '.join(sorted(set(unavailable)))}", style="dim")
            grid.add_row("-", entry)

        return Panel(
            grid,
            title="what would strengthen this",
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )

    def _verbose_details(self, result: RunResult) -> list[RenderableType]:
        out: list[RenderableType] = []
        for item in result.evidence:
            if not item.data and not item.detail:
                continue
            grid = Table.grid(padding=(0, 2))
            grid.add_column(style="dim", justify="right", no_wrap=True)
            grid.add_column(overflow="fold")
            if item.detail:
                grid.add_row("detail", Text(redact(item.detail)))
            for key, value in item.data.items():
                grid.add_row(Text(str(key)), Text(redact(_format_value(value))))
            out.append(
                Panel(
                    grid,
                    title=Text(f"{item.probe} / {item.label}"),
                    title_align="left",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )
        return out

    # ------------------------------------------------------------------- helpers

    def _glyph(self, preferred: str, fallback: str) -> str:
        """Fall back to ASCII when the output encoding cannot carry a glyph."""
        encoding = getattr(self.console.file, "encoding", None) or "utf-8"
        try:
            preferred.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return fallback
        return preferred


_OUTCOME_STYLE: dict[str, str] = {
    "ok": "green",
    "error": "red",
    "skipped": "dim",
    "unsupported": "cyan",
    "truncated": "yellow",
}


def _endpoint_host(base_url: str | None) -> str:
    """Host and port of an endpoint, never its credentials.

    ``urlsplit().hostname`` drops any ``user:password@`` prefix, which is the
    point: a key pasted into a base URL must not reach a report.
    """
    if not base_url:
        return "(adapter default)"
    parts = urlsplit(base_url)
    host = parts.hostname
    if not host:
        return "(unparsable base_url)"
    return f"{host}:{parts.port}" if parts.port else host


def _format_probability(probability: float) -> str:
    if probability >= 0.9999:
        return ">99.99%"
    if probability <= 0.0001:
        return "<0.01%"
    if probability >= 0.99 or probability <= 0.01:
        return f"{probability:.2%}"
    return f"{probability:.1%}"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return str(value)
