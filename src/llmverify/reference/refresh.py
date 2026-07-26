"""Refresh the reference snapshot from live sources, as a reviewable diff.

This module never silently rewrites thresholds. It fetches, computes a diff
against what is on disk, and by default stops there. Nothing about a published
benchmark score is urgent, and everything about it is load-bearing: the number
this file carries is the number a provider gets accused of failing to reach, so
a human should read the change in a pull request before it takes effect. Passing
``dry_run=False`` writes the merge, and even then the automated sources may only
touch the fields they actually cover.

Sources and their licensing position
------------------------------------

**Epoch AI Capabilities Index** (``epoch.ai/data/eci_benchmarks.csv``) is
published under CC-BY. It may be redistributed, including inside this package's
snapshot, provided Epoch AI is credited. That credit belongs in ``NOTICE``, not
only in a source comment.

**OpenRouter** (``openrouter.ai/api/v1/models``) requires no key. OpenRouter's
terms prohibit scraping the website; they say nothing about the documented
public API, which is what this module uses and nothing else. Requests are made
one at a time at refresh cadence, not in a loop.

**Artificial Analysis** index values are not fetched directly -- the Artificial
Analysis API returns 401 without a key -- but reach us embedded in OpenRouter's
model records. They remain Artificial Analysis's data and carry Artificial
Analysis's terms, which is why they are labelled with that provenance in the
snapshot rather than attributed to OpenRouter.

Attribution for all three belongs in ``NOTICE``.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..adapters._http import HttpClient
from ..errors import ReferenceDataError
from . import default_snapshot_path, find_model, load_snapshot
from .schema import BenchmarkScore, ModelRecord, Pricing, ReferenceSnapshot

__all__ = [
    "EPOCH_BENCHMARKS",
    "EPOCH_MODEL_IDS",
    "RefreshReport",
    "fetch_endpoint_quantization",
    "fetch_epoch",
    "fetch_openrouter",
    "refresh",
]

#: Live scores keyed by our model id.
EpochScores = dict[str, list[BenchmarkScore]]

EPOCH_CSV_URL = "https://epoch.ai/data/eci_benchmarks.csv"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"

KNOWN_SOURCES = ("epoch", "openrouter")

#: Epoch's ``benchmark`` column mapped onto this package's benchmark keys. Only
#: benchmarks listed here are imported; anything else Epoch publishes is ignored
#: rather than guessed at, because a benchmark whose variant we cannot name is a
#: benchmark whose scores we cannot compare.
EPOCH_BENCHMARKS: dict[str, str] = {
    "GPQA diamond": "gpqa_diamond",
    "ARC-AGI-2": "arc_agi_2",
    "SimpleQA Verified": "simpleqa_verified",
    "HLE": "hle",
    "SWE-Bench verified": "swe_bench_verified",
    "Terminal Bench": "terminal_bench",
    "MMLU": "mmlu",
    "Aider polyglot": "aider_polyglot",
    "OTIS Mock AIME 2024-2025": "otis_mock_aime",
    "MATH level 5": "math_level_5",
    "BBH": "bbh",
    "GSM8K": "gsm8k",
    "FrontierMath-Tiers-1-3-v2-Private": "frontiermath_t123",
    "FrontierMath-Tier-4-v2-Private": "frontiermath_t4",
    "ARC-AGI": "arc_agi_1",
    "CritPt": "critpt",
}

#: Epoch's ``model`` column mapped onto our model ids, written out by hand.
#:
#: This is a lookup table and not a similarity function on purpose. Epoch's
#: display names and our API ids agree often enough that string distance would
#: look like it worked, and then one day it would decide that "Muse Spark" is
#: "Muse Spark 1.1" -- two different models four months apart -- and attach the
#: wrong scores to the wrong record. An unlisted name is reported as unmatched
#: and a human adds a row here.
EPOCH_MODEL_IDS: dict[str, str] = {
    "Claude Opus 5": "claude-opus-5",
    "Claude Fable 5": "claude-fable-5",
    "Claude Sonnet 5": "claude-sonnet-5",
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
    "Claude Opus 4.8": "claude-opus-4-8",
    "Claude Opus 4.7": "claude-opus-4-7",
    "Claude Opus 4.6": "claude-opus-4-6",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "GPT-5.6 Sol": "gpt-5.6-sol",
    "GPT-5.6 Terra": "gpt-5.6-terra",
    "GPT-5.6 Luna": "gpt-5.6-luna",
    "GPT-5.5": "gpt-5.5",
    "GPT-5.5 Pro": "gpt-5.5-pro",
    "GPT-5.4": "gpt-5.4",
    "GPT-5.4 Mini": "gpt-5.4-mini",
    "GPT-5.4 Nano": "gpt-5.4-nano",
    "GPT-5.4 Pro": "gpt-5.4-pro",
    "GPT-5.3 Codex": "gpt-5.3-codex",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini 2.5 Pro (Jun 2025)": "gemini-2.5-pro",
    "Gemini 2.5 Flash (Jun 2025)": "gemini-2.5-flash",
    "DeepSeek-V4-Pro": "deepseek-v4-pro",
    "GLM-5.2": "glm-5.2",
    "GLM-5.1": "glm-5.1",
    "Kimi K3": "kimi-k3",
    "Kimi K2.7 Code": "kimi-k2.7-code",
    "Qwen3.7-Max": "qwen3.7-max",
    "Qwen 3.6 35B-A3B": "Qwen3.6-35B-A3B",
    "Grok 4.5": "grok-4.5",
    "Grok 4.3 Beta": "grok-4.3",
    "MiniMax-M3": "minimax-m3",
}

#: Epoch names that resemble a model we track but are not it. Listing them keeps
#: them out of ``unmatched_models``, where they would otherwise nag a reviewer
#: into adding the mapping that must never be added.
EPOCH_MODEL_NOT_OURS: dict[str, str] = {
    "Muse Spark": "Muse Spark 1.0 (2026-04-08), not meta/muse-spark-1.1",
    "Mistral Medium 3": "mistral-medium-2505, not mistral-medium-3-5",
    "Amazon Nova Pro": "amazon.nova-pro-v1, not Nova 2 Lite or Nova Premier",
}

#: ``model_version`` prefixes never mapped to any record. Epoch files
#: ``gpt-5.6-sol_promax`` and ``gpt-5.6-sol_prounknown`` under the display name
#: "GPT-5.6 Sol", and it cannot be settled from the data whether they are Sol Pro
#: runs or mislabelled Sol runs: the FrontierMath Tier-4 value on the promax row
#: is within a tenth of OpenAI's published figure for plain Sol. Attaching them
#: to either record would fabricate a reference threshold, so they are dropped
#: and surfaced as unmatched.
EPOCH_IGNORED_VERSIONS: tuple[str, ...] = ("gpt-5.6-sol_pro",)

#: Effort names Epoch appends to ``model_version`` after an underscore. Anything
#: else -- ``_unknown``, a thinking budget such as ``_32K`` -- leaves the effort
#: null, because "we do not know the effort" and "the effort was medium" produce
#: very different verdicts.
EPOCH_EFFORTS = frozenset({"max", "xhigh", "high", "medium", "low", "none", "minimal"})

_THINKING_BUDGET = re.compile(r"\d+k", re.IGNORECASE)

#: How far back an unmapped Epoch model must have been released to be worth
#: reporting. Epoch's index reaches back to Falcon-7B and GPT-3.5, and listing
#: every historical model this package has no opinion about would bury the one
#: new frontier release a reviewer actually needs to see.
UNMATCHED_HORIZON_DAYS = 180

_EPOCH_SOURCE = "Epoch AI Capabilities Index"
_AA_SOURCE = "Artificial Analysis, relayed by OpenRouter"

_CONFIDENCE_RANK = {"primary": 0, "independent": 1, "secondary": 2, "unverified": 3}

#: Fields the live sources are allowed to change. Everything else in a
#: ``ModelRecord`` -- token accounting, cutoffs, aliases, effort ladders, notes --
#: is hand-curated, often contradicts what a reseller catalogue reports, and is
#: preserved verbatim through a merge.
_AUTOMATED_FIELDS = ("context_window", "max_output_tokens", "pricing")


@dataclass(slots=True)
class RefreshReport:
    """What a refresh found, and what it would or did change."""

    fetched_at: dt.datetime
    #: Source name mapped to a one-line status: "ok, 2059 rows" or the failure.
    sources: dict[str, str] = field(default_factory=dict)
    #: Descriptions of records and scores present live but absent on disk.
    added: list[str] = field(default_factory=list)
    #: ``(model id, field path, old value, new value)``.
    changed: list[tuple[str, str, Any, Any]] = field(default_factory=list)
    #: Snapshot models whose declared OpenRouter slug is gone from the live
    #: catalogue. Reported for a human to act on; a refresh never deletes a
    #: record, because a catalogue hiccup must not erase a reference threshold.
    removed: list[str] = field(default_factory=list)
    #: Live model names that no mapping table resolves.
    unmatched_models: list[str] = field(default_factory=list)
    #: Changes the merge declined to apply, with the reason.
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = True
    written_to: Path | None = None

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    @property
    def succeeded_sources(self) -> list[str]:
        """Sources that actually returned data.

        An empty list means no diff was computed at all, which reads very
        differently from a diff that came back empty.
        """
        return sorted(n for n, s in self.sources.items() if not s.startswith("failed"))

    def render_text(self) -> str:
        """Human-readable summary, suitable for a terminal or a PR body."""
        lines = [
            f"reference refresh {self.fetched_at.isoformat(timespec='seconds')}",
            f"  mode: {'dry run' if self.dry_run else 'write'}",
        ]
        for name, status in sorted(self.sources.items()):
            lines.append(f"  source {name}: {status}")

        if self.errors:
            lines.append(f"\nerrors ({len(self.errors)})")
            lines.extend(f"  {e}" for e in self.errors)

        if not self.succeeded_sources:
            lines.append("\nno source was reached; the snapshot was not examined")
        elif not self.has_changes:
            lines.append("\nno changes: the snapshot matches every source consulted")
        if self.added:
            lines.append(f"\nadded ({len(self.added)})")
            lines.extend(f"  + {item}" for item in self.added)
        if self.changed:
            lines.append(f"\nchanged ({len(self.changed)})")
            lines.extend(
                f"  ~ {model} {path}: {old!r} -> {new!r}"
                for model, path, old, new in self.changed
            )
        if self.removed:
            lines.append(f"\nno longer listed upstream ({len(self.removed)}), not deleted")
            lines.extend(f"  ? {item}" for item in self.removed)
        if self.skipped:
            lines.append(f"\ndeclined ({len(self.skipped)})")
            lines.extend(f"  . {item}" for item in self.skipped)
        if self.unmatched_models:
            lines.append(f"\nunmatched upstream models ({len(self.unmatched_models)})")
            lines.extend(f"  ! {item}" for item in self.unmatched_models)
            lines.append(
                "  add a row to EPOCH_MODEL_IDS after confirming which model each one is"
            )
        if self.written_to is not None:
            lines.append(f"\nwrote {self.written_to}")
        elif self.has_changes:
            lines.append("\nnothing written; rerun with dry_run=False to apply")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Epoch AI
# --------------------------------------------------------------------------- #


def parse_epoch_csv(text: str, *, as_of: dt.date) -> tuple[EpochScores, list[str]]:
    """Turn the Epoch CSV into scores per model id, plus unmatched model names.

    The published column documentation describes ``performance`` as 0-100, but
    the file currently ships fractions (GPQA Diamond tops out at 0.928). Both
    are accepted: a value at or below 1.0 is read as a fraction. No benchmark in
    :data:`EPOCH_BENCHMARKS` has a real percentage below 1.0 that this could
    misread, and the alternative -- trusting the documentation -- would silently
    divide every threshold in the file by a hundred.
    """
    scores: EpochScores = {}
    unmatched: dict[str, str] = {}

    for row in csv.DictReader(io.StringIO(text)):
        benchmark = EPOCH_BENCHMARKS.get((row.get("benchmark") or "").strip())
        if benchmark is None:
            continue

        version = (row.get("model_version") or "").strip()
        if version.startswith(EPOCH_IGNORED_VERSIONS):
            unmatched.setdefault(
                version, f"{version}: deliberately unmapped, see EPOCH_IGNORED_VERSIONS"
            )
            continue

        name = (row.get("model") or "").strip()
        model_id = EPOCH_MODEL_IDS.get(name)
        if model_id is None:
            if (
                name
                and name not in EPOCH_MODEL_NOT_OURS
                and _is_recent(row.get("date"), as_of=as_of)
            ):
                unmatched.setdefault(name, f"{name} (model_version {version or '?'})")
            continue

        try:
            raw = float(row["performance"])
        except (KeyError, TypeError, ValueError):
            continue

        effort, budget = _split_epoch_version(version)
        note = f"Epoch model_version {version}" if version else "Epoch AI"
        if budget:
            note += f"; {budget} thinking budget, effort not stated"
        upstream = (row.get("source") or "").strip()
        if upstream and upstream != "Epoch evaluations":
            note += f"; Epoch relays this figure from {upstream}"

        scores.setdefault(model_id, []).append(
            BenchmarkScore(
                benchmark=benchmark,
                score=round(raw * 100 if raw <= 1.0 else raw, 2),
                effort=effort,
                optimized=(row.get("optimized") or "").strip().lower() == "true",
                source=_EPOCH_SOURCE,
                source_url=EPOCH_CSV_URL,
                as_of=as_of,
                confidence="independent",
                notes=note,
            )
        )
    return scores, sorted(unmatched.values())


def _is_recent(raw: Any, *, as_of: dt.date) -> bool:
    """Whether an Epoch release date falls inside the reporting horizon."""
    try:
        released = dt.date.fromisoformat(str(raw).strip())
    except (TypeError, ValueError):
        # An unparseable date is reported rather than hidden: a new row Epoch
        # has not dated yet is exactly the kind of thing worth a second look.
        return True
    return (as_of - released).days <= UNMATCHED_HORIZON_DAYS


def _split_epoch_version(version: str) -> tuple[str | None, str | None]:
    """Read the reasoning effort, or a thinking budget, off a model_version."""
    _stem, sep, tail = version.rpartition("_")
    if not sep:
        return None, None
    if tail.lower() in EPOCH_EFFORTS:
        return tail.lower(), None
    if _THINKING_BUDGET.fullmatch(tail):
        return None, tail
    return None, None


async def fetch_epoch(
    client: HttpClient, *, as_of: dt.date
) -> tuple[EpochScores, list[str], str]:
    """Fetch and parse the Epoch CSV. Returns (scores, unmatched, status)."""
    result = await client.request("GET", EPOCH_CSV_URL)
    if not result.ok:
        raise ReferenceDataError(f"Epoch AI returned HTTP {result.status}")
    scores, unmatched = parse_epoch_csv(result.text, as_of=as_of)
    total = sum(len(v) for v in scores.values())
    return scores, unmatched, f"ok, {total} scores across {len(scores)} known models"


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class OpenRouterFacts:
    """The subset of an OpenRouter model record this package trusts."""

    slug: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    pricing: Pricing | None = None
    scores: list[BenchmarkScore] = field(default_factory=list)


def parse_openrouter_models(
    payload: Any, snapshot: ReferenceSnapshot, *, as_of: dt.date
) -> tuple[dict[str, OpenRouterFacts], set[str]]:
    """Project OpenRouter's catalogue onto our records.

    Matching goes through :func:`~llmverify.reference.find_model`, so a live slug
    only lands on a record that already declares it as an id or alias. Slugs we
    do not track are ignored silently: OpenRouter lists hundreds of models this
    package has no opinion about, and reporting them all as unmatched would bury
    the handful that matter.
    """
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ReferenceDataError("OpenRouter response has no model list")

    facts: dict[str, OpenRouterFacts] = {}
    seen_slugs: set[str] = set()

    for entry in data:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("id") or "")
        if not slug:
            continue
        seen_slugs.add(slug)
        record = find_model(snapshot, slug)
        if record is None:
            continue
        # A first match wins: OpenRouter carries several routing variants per
        # model and the plain slug is the one records declare as an alias.
        if record.id in facts:
            continue
        top = entry.get("top_provider") or {}
        facts[record.id] = OpenRouterFacts(
            slug=slug,
            context_window=_as_int(entry.get("context_length")),
            max_output_tokens=_as_int(top.get("max_completion_tokens")),
            pricing=_openrouter_pricing(entry.get("pricing") or {}),
            scores=_artificial_analysis_scores(entry, as_of=as_of),
        )
    return facts, seen_slugs


def _openrouter_pricing(pricing: dict[str, Any]) -> Pricing | None:
    fields = {
        "input_per_mtok": "prompt",
        "output_per_mtok": "completion",
        "cache_read_per_mtok": "input_cache_read",
        "cache_write_per_mtok": "input_cache_write",
    }
    values = {name: _per_mtok(pricing.get(key)) for name, key in fields.items()}
    if all(v is None for v in values.values()):
        return None
    return Pricing(**values, source=OPENROUTER_MODELS_URL)


def _per_mtok(raw: Any) -> float | None:
    """OpenRouter quotes per-token USD as a string; we store per-million."""
    if raw in (None, ""):
        return None
    try:
        return round(float(raw) * 1e6, 6)
    except (TypeError, ValueError):
        return None


def _artificial_analysis_scores(entry: dict[str, Any], *, as_of: dt.date) -> list[BenchmarkScore]:
    block = ((entry.get("benchmarks") or {}).get("artificial_analysis")) or {}
    out = []
    for key, benchmark in (
        ("intelligence_index", "aa_intelligence_index"),
        ("coding_index", "aa_coding_index"),
        ("agentic_index", "aa_agentic_index"),
    ):
        value = block.get(key)
        if value is None:
            continue
        out.append(
            BenchmarkScore(
                benchmark=benchmark,
                score=float(value),
                unit="score",
                source=_AA_SOURCE,
                source_url=OPENROUTER_MODELS_URL,
                as_of=as_of,
                confidence="independent",
                notes="Composite index, not a single benchmark; conditions unpublished.",
            )
        )
    return out


async def fetch_openrouter(
    client: HttpClient, snapshot: ReferenceSnapshot, *, as_of: dt.date
) -> tuple[dict[str, OpenRouterFacts], set[str], str]:
    """Fetch OpenRouter's catalogue. Returns (facts, live slugs, status)."""
    result = await client.request("GET", OPENROUTER_MODELS_URL)
    if not result.ok:
        raise ReferenceDataError(f"OpenRouter returned HTTP {result.status}")
    facts, slugs = parse_openrouter_models(result.json, snapshot, as_of=as_of)
    return facts, slugs, f"ok, {len(slugs)} models listed, {len(facts)} matched to records"


async def fetch_endpoint_quantization(
    model_slug: str, *, timeout_s: float = 30.0
) -> list[dict[str, Any]]:
    """Per-provider endpoint details for one OpenRouter model.

    ``model_slug`` is the ``author/slug`` form, e.g. ``z-ai/glm-5.2``. Entries are
    returned verbatim; the fields worth reading are ``provider_name``,
    ``quantization`` (``fp4``/``fp8``/``int4``/``int8``/``fp16``/``bf16``/``fp32``
    or ``unknown``), ``context_length`` and ``supported_parameters``.

    A missing or ``unknown`` quantization is a result, not an error. These labels
    are self-reported and never audited, and roughly a third of endpoints decline
    to state one at all -- which is itself the most useful thing this call has to
    say about a provider.
    """
    slug = model_slug.strip().strip("/")
    if slug.count("/") != 1 or not all(slug.split("/")):
        raise ReferenceDataError(
            f"expected an OpenRouter 'author/slug' identifier, got {model_slug!r}"
        )
    async with HttpClient(
        base_url="https://openrouter.ai", timeout_s=timeout_s, max_concurrency=1
    ) as client:
        result = await client.request("GET", OPENROUTER_ENDPOINTS_URL.format(slug=slug))
        if not result.ok:
            raise ReferenceDataError(
                f"OpenRouter endpoints for {slug} returned HTTP {result.status}"
            )
        body = result.json if isinstance(result.json, dict) else {}
        data = body.get("data")
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        return [e for e in (endpoints or []) if isinstance(e, dict)]


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def _score_key(score: BenchmarkScore) -> tuple[str, str, str, str]:
    return (score.benchmark, score.effort or "", str(score.tools), score.source)


def _merge_scores(
    record: ModelRecord,
    incoming: Sequence[BenchmarkScore],
    report: RefreshReport,
) -> tuple[BenchmarkScore, ...]:
    """Upsert live scores into a record, never downgrading confidence."""
    existing = {_score_key(s): s for s in record.scores}
    merged = list(record.scores)

    for score in incoming:
        key = _score_key(score)
        current = existing.get(key)
        label = f"{record.id} {score.benchmark}" + (f"@{score.effort}" if score.effort else "")
        if current is None:
            merged.append(score)
            report.added.append(f"{label} = {score.score} ({score.source})")
            continue
        if _CONFIDENCE_RANK[score.confidence] > _CONFIDENCE_RANK[current.confidence]:
            report.skipped.append(
                f"{label}: kept {current.confidence} {current.score} rather than "
                f"overwrite it with {score.confidence} {score.score}"
            )
            continue
        if abs(current.score - score.score) > 1e-9:
            path = f"scores.{key[0]}@{key[1] or '-'}"
            report.changed.append((record.id, path, current.score, score.score))
        if current.model_dump() != score.model_dump():
            merged[merged.index(current)] = score
    return tuple(merged)


def _merge_record(
    record: ModelRecord,
    facts: OpenRouterFacts | None,
    epoch_scores: Sequence[BenchmarkScore],
    report: RefreshReport,
) -> ModelRecord:
    """Apply live facts to one record, preserving every hand-curated field."""
    updates: dict[str, Any] = {}

    if facts is not None:
        for name in _AUTOMATED_FIELDS:
            new = getattr(facts, name)
            if new is None:
                continue
            old = getattr(record, name)
            if name == "pricing":
                changes = _pricing_changes(old, new)
                if changes:
                    for sub, was, now in changes:
                        report.changed.append((record.id, f"pricing.{sub}", was, now))
                    updates["pricing"] = new
            elif old != new:
                report.changed.append((record.id, name, old, new))
                updates[name] = new

    incoming = [*epoch_scores, *(facts.scores if facts else ())]
    if incoming:
        merged = _merge_scores(record, incoming, report)
        if merged != record.scores:
            updates["scores"] = merged

    return record.model_copy(update=updates) if updates else record


def _pricing_changes(old: Pricing | None, new: Pricing) -> list[tuple[str, Any, Any]]:
    changes: list[tuple[str, Any, Any]] = []
    for name in (
        "input_per_mtok",
        "output_per_mtok",
        "cache_read_per_mtok",
        "cache_write_per_mtok",
    ):
        was = getattr(old, name) if old else None
        now = getattr(new, name)
        if now is None:
            # OpenRouter omitting a cache price is not evidence that the vendor
            # dropped it, so an existing hand-checked value stays.
            continue
        if was is None or abs(float(was) - float(now)) > 1e-9:
            changes.append((name, was, now))
    return changes


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _leading_comments(path: Path) -> str:
    """The comment block at the top of an existing snapshot.

    Carrying it across a rewrite is what makes ``dry_run=False`` safe to use more
    than once: the conventions documented up there survive, even though comments
    interleaved further down the file do not.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            kept.append(line)
            continue
        break
    return "\n".join(kept).rstrip() + "\n" if kept else ""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def refresh(
    snapshot_path: Path,
    *,
    sources: Sequence[str] = ("epoch", "openrouter"),
    dry_run: bool = True,
    timeout_s: float = 60.0,
) -> RefreshReport:
    """Fetch live data, diff it against ``snapshot_path``, optionally write it.

    A source that fails is recorded in the report and the others still run: a
    partial refresh that says which half succeeded is more useful than an
    exception, and no threshold moves either way while ``dry_run`` is set.
    """
    path = Path(snapshot_path or default_snapshot_path())
    today = dt.date.today()
    report = RefreshReport(fetched_at=dt.datetime.now(dt.timezone.utc), dry_run=dry_run)

    unknown = [s for s in sources if s not in KNOWN_SOURCES]
    if unknown:
        report.errors.append(
            f"unknown source(s) {', '.join(unknown)}; known: {', '.join(KNOWN_SOURCES)}"
        )
    wanted = [s for s in sources if s in KNOWN_SOURCES]

    snapshot = load_snapshot(path)

    epoch_scores: dict[str, list[BenchmarkScore]] = {}
    or_facts: dict[str, OpenRouterFacts] = {}
    live_slugs: set[str] = set()

    async with HttpClient(
        base_url="https://epoch.ai", timeout_s=timeout_s, max_concurrency=2
    ) as client:
        tasks = []
        if "epoch" in wanted:
            tasks.append(("epoch", fetch_epoch(client, as_of=today)))
        if "openrouter" in wanted:
            tasks.append(("openrouter", fetch_openrouter(client, snapshot, as_of=today)))

        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)

    for (name, _), outcome in zip(tasks, results, strict=True):
        if isinstance(outcome, BaseException):
            report.sources[name] = f"failed: {outcome}"
            report.errors.append(f"{name}: {outcome}")
            continue
        if name == "epoch":
            epoch_scores, unmatched, status = outcome
            report.unmatched_models.extend(unmatched)
        else:
            or_facts, live_slugs, status = outcome
        report.sources[name] = status

    if not report.succeeded_sources:
        return report

    merged_models = tuple(
        _merge_record(record, or_facts.get(record.id), epoch_scores.get(record.id, ()), report)
        for record in snapshot.models
    )

    if live_slugs:
        for record in snapshot.models:
            # Only records whose data actually came from OpenRouter can have
            # gone missing from it. A record that merely declares a plausible
            # OpenRouter-shaped alias -- claude-mythos-5 is invite-only and was
            # never listed -- must not be reported as disappearing every run.
            if not record.pricing or record.pricing.source != OPENROUTER_MODELS_URL:
                continue
            declared = [
                a for a in (record.id, *record.aliases) if "/" in a and not a.startswith("~")
            ]
            if declared and not any(slug in live_slugs for slug in declared):
                report.removed.append(f"{record.id} ({', '.join(declared)})")

    if dry_run or not report.has_changes:
        return report

    updated = snapshot.model_copy(
        update={
            "as_of": today,
            "generated_by": "refresh:" + "+".join(wanted),
            "models": merged_models,
        }
    )
    body = yaml.safe_dump(
        updated.to_yaml_dict(),
        sort_keys=False,
        allow_unicode=True,
        width=96,
        default_flow_style=False,
    )
    path.write_text(_leading_comments(path) + body, encoding="utf-8")
    report.written_to = path

    from . import clear_cache

    clear_cache()
    return report
