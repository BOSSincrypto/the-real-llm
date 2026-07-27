"""Command line interface.

Argparse rather than a CLI framework, because the dependency list is a feature
of this package: a tool that audits other people's supply chains has no business
adding three transitive packages to its own.

Rendering is looked up at run time. When :mod:`llmverify.report` is installed the
console and HTML output come from there; when it is not, the fallbacks in this
module keep every command working. The hooks are, all optional:

``render_result(result, *, console, verbose)``
    Show one run.
``render_comparison(results, *, console)``
    Show the ``compare`` table.
``write_html(results, path)``
    Write a standalone HTML report for one or more runs.

Every free-text value printed here -- error bodies, base URLs, probe details --
goes through :func:`llmverify.config.redact` first. Provider errors routinely
echo request headers back, and a report that leaks the key it authenticated with
is worse than no report.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import ProviderConfig, RunConfig, redact
from .errors import ConfigError, LLMVerifyError
from .evidence import EvidenceStatus, Verdict
from .results import RunResult, compare_results

__all__ = [
    "EXIT_INCONCLUSIVE",
    "EXIT_SUBSTITUTION",
    "EXIT_UNREACHABLE",
    "EXIT_USAGE",
    "EXIT_VERIFIED",
    "main",
]

EXIT_VERIFIED = 0
EXIT_SUBSTITUTION = 1
EXIT_INCONCLUSIVE = 2
EXIT_USAGE = 3
EXIT_UNREACHABLE = 4

#: Environment variable consulted for each adapter when neither the config nor
#: ``--api-key-env`` names one. Only used when the variable is actually set:
#: guessing a name that is not set would replace a clear "no key configured"
#: with a confusing complaint about a variable the user never mentioned.
_DEFAULT_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_WHAT_IT_PROVES = """\
What this can and cannot establish: llmverify measures what an endpoint does --
which parameters it honours, how it tokenizes, what its token accounting adds up
to, whether thinking blocks carry a signature only genuine inference can
produce, and how it scores against published numbers -- and combines those
observations into a log-odds posterior with every piece of evidence shown. It
cannot prove which weights ran. No software-side method can: a provider that
routes a fraction of traffic to the real model, or serves it only to inputs that
look like tests, defeats every statistical test in this package, and only
hardware attestation closes that gap. Read an adverse verdict as "this endpoint
does not behave like the model it claims", never as proof of fraud, and read a
favourable one as "nothing here contradicts the claim at the sample size paid
for".
"""

_EXAMPLE = """\
example
  llmverify check providers/acme.yaml --claimed-model claude-opus-5 --max-cost 0.25
  llmverify check --api openai --base-url https://api.example.com/v1 \\
      --model gpt-5.6-sol --api-key-env ACME_KEY --layers 0,1

exit codes
  0  verified          MATCH or LIKELY_MATCH
  1  substitution      LIKELY_MISMATCH, MISMATCH or EVASION
  2  inconclusive      not enough evidence to say either way
  3  usage error       bad configuration or command line
  4  unreachable       no probe got a usable response from the endpoint
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code and never raises for user error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = {
        "check": _cmd_check,
        "compare": _cmd_compare,
        "refresh": _cmd_refresh,
        "models": _cmd_models,
        "probes": _cmd_probes,
        "benchmarks": _cmd_benchmarks,
        "cache": _cmd_cache,
    }.get(args.command)

    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        return handler(args)
    except (LLMVerifyError, FileNotFoundError) as exc:
        _fail(str(exc))
        return EXIT_USAGE
    except KeyboardInterrupt:
        _console(stderr=True).print("\n[yellow]interrupted[/yellow]")
        return EXIT_INCONCLUSIVE
    except BrokenPipeError:
        # `llmverify probes | head` closed the pipe on us. The user got what
        # they asked for, so this is not a failure to report.
        return EXIT_VERIFIED


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="llmverify",
        description="Check whether a custom LLM provider serves the model it claims.",
        epilog=_EXAMPLE + "\n" + _WHAT_IT_PROVES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"llmverify {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    check = sub.add_parser(
        "check",
        help="verify one provider",
        description="Verify one provider against the model it claims to serve.",
        epilog=_EXAMPLE + "\n" + _WHAT_IT_PROVES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument(
        "provider",
        nargs="?",
        help="provider YAML. Omit it and describe the provider entirely with flags.",
    )
    _add_provider_flags(check)
    _add_run_flags(check)
    _add_output_flags(check)

    compare = sub.add_parser(
        "compare",
        help="verify several providers and tabulate them",
        description=(
            "Run the same checks against several providers and print the comparison. "
            "Providers are checked one at a time: running them together would make "
            "them contend for one uplink and flatter or slander their measured speed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare.add_argument("providers", nargs="+", help="two or more provider YAML files")
    _add_run_flags(compare)
    _add_output_flags(compare)

    refresh = sub.add_parser(
        "refresh",
        help="refresh the reference snapshot from live sources",
        description=(
            "Fetch published scores and pricing, diff them against the snapshot and "
            "print the result. A dry run by default: the numbers in the snapshot are "
            "what a provider gets accused of failing to reach, so a human should read "
            "the change before it takes effect."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    refresh.add_argument(
        "--source",
        default="epoch,openrouter",
        metavar="LIST",
        help="comma-separated sources to consult (default: epoch,openrouter)",
    )
    refresh.add_argument(
        "--write", action="store_true", help="apply the merge instead of only showing it"
    )
    refresh.add_argument(
        "--snapshot", metavar="PATH", help="snapshot to refresh (default: the bundled one)"
    )

    models = sub.add_parser(
        "models",
        help="list what the reference snapshot knows",
        description="List reference models, optionally filtered by a substring.",
    )
    models.add_argument("query", nargs="?", help="match against id, alias, vendor or name")
    models.add_argument("--snapshot", metavar="PATH", help="read a snapshot from PATH")
    models.add_argument("--json", metavar="PATH", help="also write the listing as JSON")

    probes = sub.add_parser(
        "probes", help="list available probes", description="List every registered probe."
    )
    probes.add_argument("--json", metavar="PATH", help="also write the listing as JSON")

    benchmarks = sub.add_parser(
        "benchmarks",
        help="list available benchmarks",
        description=(
            "List every registered benchmark. The score-spread column is the one to "
            "read: a benchmark on which every current model scores within a few points "
            "cannot separate them, whatever its reputation."
        ),
    )
    benchmarks.add_argument("--json", metavar="PATH", help="also write the listing as JSON")

    cache = sub.add_parser(
        "cache",
        help="show or clear the dataset cache",
        description="Report the dataset cache's size and location.",
    )
    cache.add_argument("--clear", action="store_true", help="delete every cached dataset page")
    cache.add_argument("--cache-dir", metavar="PATH", help="override the cache location")

    return parser


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("provider (override the YAML, or replace it entirely)")
    group.add_argument("--model", help="model identifier to send in requests")
    group.add_argument(
        "--claimed-model",
        help="canonical id the provider claims to serve, for reference lookup",
    )
    group.add_argument("--base-url", help="endpoint root, e.g. https://api.example.com/v1")
    group.add_argument(
        "--api",
        metavar="NAME",
        help="adapter to speak: openai, anthropic, gemini or a registered plugin",
    )
    group.add_argument(
        "--api-key-env",
        metavar="VAR",
        help="environment variable holding the API key. Never pass the key itself.",
    )
    group.add_argument("--effort", help="pin reasoning effort, e.g. high. Ask for one:")


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run")
    group.add_argument(
        "--layers",
        metavar="LIST",
        help="comma-separated layers to run (default 0,1,2,3; 0 is free, 3 costs money)",
    )
    group.add_argument(
        "--probe",
        action="append",
        metavar="NAME",
        help="run only this probe; repeatable. Disables early stopping.",
    )
    group.add_argument(
        "--exclude-probe", action="append", metavar="NAME", help="skip this probe; repeatable"
    )
    group.add_argument(
        "--benchmark", action="append", metavar="NAME", help="benchmark to use; repeatable"
    )
    group.add_argument(
        "--max-cost",
        type=float,
        metavar="USD",
        help="estimated spend ceiling; 0 or less means no ceiling",
    )
    group.add_argument(
        "--max-samples", type=int, metavar="N", help="request ceiling; 0 or less means none"
    )
    group.add_argument(
        "--max-wall", type=float, metavar="SECONDS", help="wall-clock ceiling; 0 means none"
    )
    group.add_argument("--seed", type=int, help="seed for every randomised choice")
    group.add_argument(
        "--no-anti-evasion",
        action="store_true",
        help="send benchmark items verbatim only, skipping the paraphrased arm",
    )
    group.add_argument(
        "--baseline",
        metavar="FILE",
        help="provider YAML for a first-party endpoint to measure against",
    )


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--html", metavar="PATH", help="write an HTML report to PATH")
    group.add_argument("--json", metavar="PATH", help="write the machine-readable result to PATH")
    group.add_argument(
        "--quiet", action="store_true", help="print the verdict line and nothing else"
    )
    group.add_argument("--verbose", action="store_true", help="show every piece of evidence")


# --------------------------------------------------------------------------- #
# check / compare
# --------------------------------------------------------------------------- #


def _cmd_check(args: argparse.Namespace) -> int:
    from .runner import provider_unreachable, verify_provider

    provider = _provider_from(args, args.provider)
    run = _run_from(args)
    console = _console()
    tracker = _ProgressTracker(console, quiet=args.quiet, verbose=args.verbose)

    try:
        result = asyncio.run(verify_provider(provider, run, progress=tracker))
    except KeyboardInterrupt:
        return _interrupted(console, tracker)

    _emit_result(args, console, result)
    if provider_unreachable(result):
        return EXIT_UNREACHABLE
    return _exit_code(result.verdict.verdict)


def _cmd_compare(args: argparse.Namespace) -> int:
    from .runner import verify_many

    providers = [
        _provider_from(args, path, allow_flag_overrides=False) for path in args.providers
    ]
    run = _run_from(args)
    console = _console()
    tracker = _ProgressTracker(console, quiet=args.quiet, verbose=args.verbose)

    try:
        results = asyncio.run(verify_many(providers, run, progress=tracker))
    except KeyboardInterrupt:
        return _interrupted(console, tracker)

    report = _load_report()
    if report is not None and hasattr(report, "render_comparison"):
        report.render_comparison(results, console=console)
    else:
        _render_comparison(console, results)

    _write_html(args, console, results)
    if args.json:
        summary = compare_results(results)
        summary["runs"] = [r.to_dict() for r in results]
        _write_json(console, args.json, summary)

    return _worst_exit_code(_result_exit_code(r) for r in results)


def _result_exit_code(result: RunResult) -> int:
    from .runner import provider_unreachable

    if provider_unreachable(result):
        return EXIT_UNREACHABLE
    return _exit_code(result.verdict.verdict)


#: How loudly each exit code should shout when several runs disagree. A
#: suspected substitution is the finding worth acting on, so it outranks an
#: endpoint that simply did not answer.
_EXIT_SEVERITY: dict[int, int] = {
    EXIT_VERIFIED: 0,
    EXIT_INCONCLUSIVE: 1,
    EXIT_UNREACHABLE: 2,
    EXIT_SUBSTITUTION: 3,
}


def _worst_exit_code(codes: Any) -> int:
    ranked = sorted(codes, key=lambda code: _EXIT_SEVERITY.get(code, 0))
    return ranked[-1] if ranked else EXIT_INCONCLUSIVE


def _emit_result(args: argparse.Namespace, console: Any, result: RunResult) -> None:
    report = _load_report()
    if args.quiet:
        console.print(_verdict_line(result))
    elif report is not None and hasattr(report, "render_result"):
        report.render_result(result, console=console, verbose=args.verbose)
    else:
        _render_result(console, result, verbose=args.verbose)

    _write_html(args, console, [result])
    if args.json:
        _write_json(console, args.json, result.to_dict())


def _exit_code(verdict: Verdict) -> int:
    if verdict in (Verdict.MATCH, Verdict.LIKELY_MATCH):
        return EXIT_VERIFIED
    if verdict.is_adverse:
        return EXIT_SUBSTITUTION
    return EXIT_INCONCLUSIVE


def _interrupted(console: Any, tracker: _ProgressTracker) -> int:
    """Report whatever finished before the user pressed Ctrl-C."""
    console.print("\n[yellow]interrupted before a verdict was reached[/yellow]")
    if tracker.completed:
        console.print("probes that ran to a conclusion first:")
        for name, outcome, elapsed, message in tracker.completed:
            console.print(f"  {name:<22} {outcome:<8} {elapsed:6.1f}s  {redact(message)}")
        console.print(
            "\nThese are observations, not a verdict: partial evidence has not been "
            "aggregated and no threshold was applied to it."
        )
    else:
        console.print("no probe finished, so there is nothing partial to show.")
    return EXIT_INCONCLUSIVE


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #


def _provider_from(
    args: argparse.Namespace, path: str | None, *, allow_flag_overrides: bool = True
) -> ProviderConfig:
    """Build a provider from a YAML file, from flags, or from both."""
    from .config import load_provider

    data: dict[str, Any] = {}
    if path:
        data = load_provider(path).model_dump()

    if allow_flag_overrides:
        for field, value in (
            ("api", getattr(args, "api", None)),
            ("model", getattr(args, "model", None)),
            ("claimed_model", getattr(args, "claimed_model", None)),
            ("base_url", getattr(args, "base_url", None)),
            ("api_key_env", getattr(args, "api_key_env", None)),
            ("reasoning_effort", getattr(args, "effort", None)),
        ):
            if value is not None:
                data[field] = value

    if not data.get("model"):
        raise ConfigError(
            "no model to check: pass a provider YAML, or --model together with --api "
            "and --base-url"
        )
    data.setdefault("api", "openai")
    data.setdefault("name", _derive_name(data))

    if not data.get("api_key_env"):
        import os

        guess = _DEFAULT_KEY_ENV.get(str(data["api"]))
        if guess and os.environ.get(guess):
            data["api_key_env"] = guess

    try:
        return ProviderConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"invalid provider configuration: {exc}") from exc


def _derive_name(data: dict[str, Any]) -> str:
    """Label a flag-only provider by its host, falling back to the model id."""
    base_url = data.get("base_url")
    if isinstance(base_url, str) and "//" in base_url:
        host = base_url.split("//", 1)[1].split("/", 1)[0]
        if host:
            return host
    return str(data.get("model", "provider"))


def _run_from(args: argparse.Namespace) -> RunConfig:
    from .config import BudgetConfig, load_provider

    defaults = RunConfig()
    budget = defaults.budget
    data: dict[str, Any] = {}

    if getattr(args, "layers", None):
        data["layers"] = _parse_layers(args.layers)
    if getattr(args, "probe", None):
        data["probes"] = tuple(args.probe)
    if getattr(args, "exclude_probe", None):
        data["exclude_probes"] = tuple(args.exclude_probe)
    if getattr(args, "benchmark", None):
        data["benchmarks"] = tuple(args.benchmark)
    if getattr(args, "seed", None) is not None:
        data["seed"] = args.seed
    if getattr(args, "no_anti_evasion", False):
        data["anti_evasion"] = False
    if getattr(args, "baseline", None):
        data["baseline"] = load_provider(args.baseline)

    ceilings: dict[str, Any] = {}
    for field, value in (
        ("max_cost_usd", getattr(args, "max_cost", None)),
        ("max_samples", getattr(args, "max_samples", None)),
        ("max_wall_s", getattr(args, "max_wall", None)),
    ):
        if value is not None:
            ceilings[field] = None if value <= 0 else value
    if ceilings:
        data["budget"] = BudgetConfig(**{**budget.model_dump(), **ceilings})

    try:
        return RunConfig(**data)
    except Exception as exc:
        raise ConfigError(f"invalid run configuration: {exc}") from exc


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers: list[int] = []
    for chunk in raw.replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            layer = int(chunk)
        except ValueError as exc:
            raise ConfigError(f"--layers takes integers, got {chunk!r}") from exc
        if layer not in (0, 1, 2, 3):
            raise ConfigError(f"--layers accepts 0, 1, 2 or 3, got {layer}")
        layers.append(layer)
    if not layers:
        raise ConfigError("--layers was empty")
    return tuple(sorted(set(layers)))


# --------------------------------------------------------------------------- #
# Listing commands
# --------------------------------------------------------------------------- #


def _cmd_probes(args: argparse.Namespace) -> int:
    from .probes import all_probes

    rows = [
        {
            "name": cls.name,
            "layer": cls.layer,
            "family": cls.family,
            "estimated_requests": cls.estimated_requests,
            "order": cls.order,
            "description": cls.description,
        }
        for cls in sorted(all_probes().values(), key=lambda c: (c.layer, c.order, c.name))
    ]

    console = _console()
    table = _table("probes", "probe", "layer", "family", "requests", "what it measures")
    for row in rows:
        table.add_row(
            row["name"],
            str(row["layer"]),
            row["family"],
            str(row["estimated_requests"]),
            row["description"],
        )
    console.print(table)
    console.print(
        "\nLayer 0 is free, layer 1 costs fractions of a cent, layer 2 seconds of "
        "generation, layer 3 real money. Probes run cheapest first."
    )
    if args.json:
        _write_json(console, args.json, {"probes": rows})
    return EXIT_VERIFIED


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    from .benchmarks.base import all_benchmarks

    rows = [
        {
            "name": cls.name,
            "reference_key": cls.reference_key,
            "dataset": cls.hf_dataset or "generated locally",
            "licence": cls.licence,
            "gated": cls.gated,
            "discriminative": cls.discriminative,
            "score_spread": cls.score_spread,
            "needs_sandbox": cls.needs_sandbox,
            "description": cls.description,
        }
        for cls in sorted(all_benchmarks().values(), key=lambda c: -c.score_spread)
    ]

    console = _console()
    table = _table(
        "benchmarks", "benchmark", "dataset", "licence", "gated", "separates", "spread"
    )
    for row in rows:
        table.add_row(
            row["name"],
            row["dataset"],
            row["licence"],
            "yes" if row["gated"] else "no",
            "[green]yes[/green]" if row["discriminative"] else "[yellow]no[/yellow]",
            f"{row['score_spread']:.1f} pp",
        )
    console.print(table)
    console.print(
        "\n'separates' is whether published scores across current models spread far "
        "enough to tell them apart. On a saturated benchmark every frontier model "
        "lands within a few points, so no affordable sample size distinguishes them; "
        "those stay available because a substituted model is often nowhere near "
        "frontier, but they are weighted down. 'spread' is the standard deviation of "
        "published scores in percentage points."
    )
    if args.json:
        _write_json(console, args.json, {"benchmarks": rows})
    return EXIT_VERIFIED


def _cmd_models(args: argparse.Namespace) -> int:
    from .reference import load_snapshot, snapshot_age_days

    snapshot = load_snapshot(Path(args.snapshot) if args.snapshot else None)
    query = (args.query or "").strip().lower()

    records = [
        record
        for record in snapshot.models
        if not query
        or query in record.id.lower()
        or query in record.vendor.lower()
        or query in (record.display_name or "").lower()
        or any(query in alias.lower() for alias in record.aliases)
    ]

    console = _console()
    table = _table(
        f"reference snapshot as of {snapshot.as_of}",
        "model id",
        "vendor",
        "context",
        "$/MTok in-out",
        "benchmarks",
    )
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (r.vendor, r.id)):
        benchmarks = sorted({score.benchmark for score in record.scores})
        rows.append(
            {
                "id": record.id,
                "vendor": record.vendor,
                "context_window": record.context_window,
                "input_per_mtok": record.pricing.input_per_mtok if record.pricing else None,
                "output_per_mtok": record.pricing.output_per_mtok if record.pricing else None,
                "benchmarks": benchmarks,
            }
        )
        table.add_row(
            record.id,
            record.vendor,
            _thousands(record.context_window),
            _price(record),
            f"{len(benchmarks)}: {', '.join(benchmarks[:4])}"
            + (" ..." if len(benchmarks) > 4 else "")
            if benchmarks
            else "[dim]none[/dim]",
        )

    console.print(table)
    if not rows:
        console.print(f"nothing in the snapshot matches {args.query!r}.")

    age = snapshot_age_days(snapshot)
    if age > 30:
        console.print(
            f"\n[yellow]this snapshot is {age} days old; run `llmverify refresh` "
            "and review the diff.[/yellow]"
        )
    if args.json:
        _write_json(console, args.json, {"as_of": snapshot.as_of.isoformat(), "models": rows})
    return EXIT_VERIFIED


def _cmd_cache(args: argparse.Namespace) -> int:
    from .benchmarks.datasets import cache_info, clear_cache

    cache_dir = Path(args.cache_dir) if args.cache_dir else RunConfig().cache_dir
    console = _console()

    if args.clear:
        freed = clear_cache(cache_dir)
        console.print(f"cleared {_bytes(freed)} from {cache_dir / 'datasets'}", soft_wrap=True)
        return EXIT_VERIFIED

    info = cache_info(cache_dir)
    console.print(f"location  {info['path']}", soft_wrap=True)
    console.print(f"entries   {info['entries']}")
    console.print(f"size      {_bytes(info['bytes'])}")
    console.print(f"oldest    {info['oldest'] or '-'}")
    console.print(f"newest    {info['newest'] or '-'}")
    if not info["entries"]:
        console.print("\nnothing cached yet; benchmark probes populate this on first use.")
    return EXIT_VERIFIED


def _cmd_refresh(args: argparse.Namespace) -> int:
    from .reference import default_snapshot_path
    from .reference.refresh import refresh

    sources = tuple(s for s in args.source.replace(" ", "").split(",") if s)
    path = Path(args.snapshot) if args.snapshot else default_snapshot_path()
    console = _console()

    try:
        report = asyncio.run(refresh(path, sources=sources, dry_run=not args.write))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted; nothing was written[/yellow]")
        return EXIT_INCONCLUSIVE

    console.print(redact(report.render_text()))
    if report.errors and not report.succeeded_sources:
        return EXIT_UNREACHABLE
    return EXIT_VERIFIED


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class _ProgressTracker:
    """Turns runner events into terminal lines, and remembers what finished.

    Progress goes to stderr so that ``--json`` on stdout stays pipeable, and it
    is line-oriented rather than a live display for the same reason: a run in CI
    should leave a readable log, not a transcript of cursor moves.
    """

    def __init__(self, console: Any, *, quiet: bool = False, verbose: bool = False) -> None:
        self._out = _console(stderr=True)
        self._quiet = quiet
        self._verbose = verbose
        #: ``(probe, outcome, elapsed, message)`` for every probe that ran to a
        #: conclusion, kept so that Ctrl-C can still show what was learned.
        self.completed: list[tuple[str, str, float, str]] = []

    def __call__(self, event: Any) -> None:
        from .runner import ProgressState

        if event.state in (ProgressState.FINISHED, ProgressState.FAILED):
            self.completed.append(
                (event.probe, event.state.value, event.elapsed_s, event.message)
            )
        if self._quiet:
            return

        label = event.probe if not event.provider else f"{event.provider}/{event.probe}"
        if event.state is ProgressState.STARTED:
            if self._verbose:
                self._out.print(f"[dim]  {label} ...[/dim]")
        elif event.state is ProgressState.FINISHED:
            self._out.print(
                f"  [green]done[/green] {label:<28} {event.elapsed_s:5.1f}s  "
                f"[dim]{redact(event.message)}[/dim]"
            )
        elif event.state is ProgressState.FAILED:
            self._out.print(
                f"  [red]fail[/red] {label:<28} {event.elapsed_s:5.1f}s  "
                f"[dim]{redact(event.message)}[/dim]"
            )
        elif self._verbose:
            self._out.print(f"  [dim]skip {label:<28}        {redact(event.message)}[/dim]")


# --------------------------------------------------------------------------- #
# Fallback rendering
# --------------------------------------------------------------------------- #

_VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.MATCH: "bold green",
    Verdict.LIKELY_MATCH: "green",
    Verdict.INCONCLUSIVE: "yellow",
    Verdict.LIKELY_MISMATCH: "red",
    Verdict.MISMATCH: "bold red",
    Verdict.EVASION: "bold magenta",
}


def _verdict_line(result: RunResult) -> str:
    report = result.verdict
    style = _VERDICT_STYLE.get(report.verdict, "white")
    return (
        f"[{style}]{report.verdict.value}[/{style}] {result.provider_name} "
        f"claiming {result.claimed_model} -- p(genuine)={report.probability:.4f}, "
        f"{report.total_bans:+.2f} bans, ${result.spent_usd:.4f}, {result.duration_s:.0f}s"
    )


def _render_result(console: Any, result: RunResult, *, verbose: bool) -> None:
    report = result.verdict
    console.print()
    console.print(_verdict_line(result))
    console.print(
        f"[dim]{result.api_family} at {redact(result.base_url or '-')}, model "
        f"{result.requested_model}, seed {result.seed}, reference "
        + (f"as of {result.reference_as_of}" if result.reference_as_of else "unavailable")
        + (", model found" if result.reference_found else ", model NOT in snapshot")
        + "[/dim]"
    )

    usable = [e for e in report.evidence if e.status is EvidenceStatus.OK]
    shown = usable if verbose else _headline_evidence(report)
    if shown:
        table = _table("evidence", "probe", "finding", "bans", "what it means")
        for item in shown:
            colour = "green" if item.llr > 0 else ("red" if item.llr < 0 else "dim")
            table.add_row(
                item.probe,
                item.label,
                f"[{colour}]{item.bans:+.2f}[/{colour}]",
                redact(item.detail),
            )
        console.print(table)

    if report.family_totals:
        nats_per_ban = math.log(10)
        totals = ", ".join(
            f"{family} {value / nats_per_ban:+.2f}"
            for family, value in sorted(
                report.family_totals.items(), key=lambda kv: -abs(kv[1])
            )
        )
        console.print(f"[dim]family totals (bans): {totals}[/dim]")

    counts = {
        status.value: len(report.by_status(status))
        for status in EvidenceStatus
        if report.by_status(status)
    }
    console.print(
        "[dim]"
        + ", ".join(f"{n} {name}" for name, n in counts.items())
        + f"; {result.samples} requests, {result.input_tokens}+{result.output_tokens} tokens"
        + "[/dim]"
    )

    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {redact(note)}")
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {redact(warning)}")


def _headline_evidence(report: Any, limit: int = 8) -> list[Any]:
    """The strongest evidence in each direction, refuting first.

    A reader who sees only the supporting half of an adverse verdict has been
    misled, so both directions are always represented.
    """
    refuting = report.top_refuting[: limit // 2]
    supporting = report.top_supporting[: limit - len(refuting)]
    return refuting + supporting


def _render_comparison(console: Any, results: Sequence[RunResult]) -> None:
    summary = compare_results(list(results))
    table = _table(
        "comparison", "provider", "claimed model", "verdict", "p(genuine)", "bans", "cost"
    )
    for row in summary["providers"]:
        verdict = Verdict(row["verdict"])
        style = _VERDICT_STYLE.get(verdict, "white")
        table.add_row(
            row["name"],
            row["claimed_model"],
            f"[{style}]{row['verdict']}[/{style}]",
            f"{row['probability_genuine']:.4f}",
            f"{row['total_bans']:+.2f}",
            f"${row['estimated_cost_usd']:.4f}",
        )
    console.print(table)
    for warning in summary["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {redact(warning)}")

    # Per-run warnings would otherwise vanish behind the table, and "this
    # endpoint never answered" changes how its row should be read.
    for result in results:
        for warning in result.warnings:
            console.print(f"[yellow]warning[/yellow] {result.provider_name}: {redact(warning)}")


# --------------------------------------------------------------------------- #
# Output plumbing
# --------------------------------------------------------------------------- #


def _load_report() -> Any | None:
    """The optional reporting package, or ``None`` when it is not installed."""
    try:
        from . import report
    except ImportError:
        return None
    return report


def _write_html(args: argparse.Namespace, console: Any, results: Sequence[RunResult]) -> None:
    path = getattr(args, "html", None)
    if not path:
        return
    report = _load_report()
    writer = getattr(report, "write_html", None) if report is not None else None
    if writer is None:
        _warn(
            console,
            "HTML reporting lives in llmverify.report, which is not available in this "
            f"build, so {path} was not written. --json carries the same data.",
        )
        return
    try:
        writer(list(results), Path(path))
    except Exception as exc:
        _warn(console, f"could not write {path}: {redact(str(exc))[:200]}")
    else:
        # soft_wrap keeps a long path on one line so it can be copied out.
        console.print(f"[dim]wrote {path}[/dim]", soft_wrap=True)


def _write_json(console: Any, path: str, payload: dict[str, Any]) -> None:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _warn(console, f"could not write {path}: {exc}")
    else:
        # soft_wrap keeps a long path on one line so it can be copied out.
        console.print(f"[dim]wrote {path}[/dim]", soft_wrap=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _console(*, stderr: bool = False) -> Any:
    from rich.console import Console

    return Console(stderr=stderr, highlight=False, soft_wrap=False)


def _table(title: str, *columns: str) -> Any:
    """A table whose last column absorbs the wrapping.

    Only the final column folds. Identifiers -- model ids, dataset repos, probe
    names -- become unreadable when broken mid-token, so they are allowed to
    overflow instead, and the prose column gives up the width.
    """
    from rich import box
    from rich.table import Table

    table = Table(title=title, box=box.SIMPLE_HEAD, title_justify="left", pad_edge=False)
    last = len(columns) - 1
    for index, column in enumerate(columns):
        table.add_column(column, overflow="fold" if index == last else "ellipsis")
    return table


def _warn(console: Any, message: str) -> None:
    console.print(f"[yellow]warning:[/yellow] {message}")


def _fail(message: str) -> None:
    with contextlib.suppress(Exception):
        _console(stderr=True).print(f"[red]error:[/red] {redact(message)}")


def _thousands(value: int | None) -> str:
    return f"{value:,}" if value else "-"


def _price(record: Any) -> str:
    pricing = record.pricing
    if pricing is None or pricing.input_per_mtok is None:
        return "-"
    out = f"{pricing.output_per_mtok:g}" if pricing.output_per_mtok is not None else "?"
    return f"{pricing.input_per_mtok:g} / {out}"


def _bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
