"""Loading and lookup for the reference snapshot.

The snapshot is a YAML file shipped inside the package rather than a service
fetched at runtime, so that a verdict is reproducible: rerunning last month's
command against last month's commit produces last month's answer. A user who
disagrees with a threshold can point ``load_snapshot`` at their own file.

Lookup is deliberately conservative. :func:`find_model` will normalise away
casing, separators, an OpenRouter vendor prefix and a trailing date stamp, but
it will never do approximate matching. Nothing in this package benefits from
guessing that ``gpt-5.6-luna`` "probably means" ``gpt-5.6-sol``: the two are
different models with a 12-point gap on ARC-AGI-2, and a wrong resolution here
turns into a confident, wrong accusation downstream. An identifier that does not
resolve exactly resolves to ``None``, and the caller reports that the model is
unknown to the snapshot.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..errors import ReferenceDataError
from .schema import (
    BenchmarkScore,
    FamilySignature,
    ModelRecord,
    Pricing,
    ReferenceSnapshot,
    TokenAccounting,
)

__all__ = [
    "BenchmarkScore",
    "FamilySignature",
    "ModelRecord",
    "Pricing",
    "ReferenceSnapshot",
    "TokenAccounting",
    "clear_cache",
    "default_snapshot_path",
    "find_model",
    "load_snapshot",
    "snapshot_age_days",
]

#: Parsed snapshots keyed by resolved path. Parsing and validating the bundled
#: file costs a few milliseconds, which is nothing once but noticeable when
#: every probe in a run asks for it.
_CACHE: dict[Path, ReferenceSnapshot] = {}
_CACHE_LOCK = threading.Lock()

#: Trailing snapshot stamps: ``-20260723`` and ``-2026-07-23``. Deliberately not
#: a bare four-digit group, because ``mistral-small-2603`` and ``qwen3.6-27b``
#: carry digits that are part of the identifier rather than a date.
_DATE_SUFFIX = re.compile(r"[-_](?:20\d{6}|20\d{2}-\d{2}-\d{2})$")

_SEPARATORS = re.compile(r"[-_. ]+")

#: OpenRouter variant tags that select a route rather than a different model.
#: Tags outside this set (``:thinking``, ``:online``, ...) change behaviour, so
#: an identifier carrying one is left alone and simply fails to resolve.
_ROUTING_TAGS = frozenset({"free", "nitro", "floor"})


def default_snapshot_path() -> Path:
    """Path of the snapshot bundled with the installed package."""
    return Path(__file__).resolve().parent / "data" / "reference.yaml"


def load_snapshot(path: Path | None = None) -> ReferenceSnapshot:
    """Load and validate a snapshot, caching by resolved path.

    Raises :class:`~llmverify.errors.ReferenceDataError` for a missing file,
    malformed YAML or a payload that fails schema validation. That is a hard
    error rather than a warning: running with a half-parsed reference would
    silently drop the thresholds a verdict depends on.
    """
    resolved = (path or default_snapshot_path()).expanduser().resolve()
    cached = _CACHE.get(resolved)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _CACHE.get(resolved)
        if cached is None:
            cached = _parse_snapshot(resolved)
            _CACHE[resolved] = cached
        return cached


def clear_cache() -> None:
    """Forget every cached snapshot.

    Needed after :mod:`llmverify.reference.refresh` rewrites a file on disk, and
    by tests that write snapshots to temporary paths.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


def _parse_snapshot(path: Path) -> ReferenceSnapshot:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReferenceDataError(f"reference snapshot not found: {path}") from exc
    except OSError as exc:
        raise ReferenceDataError(f"cannot read reference snapshot {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ReferenceDataError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ReferenceDataError(f"{path} must contain a mapping at the top level")

    try:
        return ReferenceSnapshot.model_validate(raw)
    except ValidationError as exc:
        raise ReferenceDataError(f"{path} does not match the reference schema: {exc}") from exc


def snapshot_age_days(snapshot: ReferenceSnapshot, *, today: dt.date | None = None) -> int:
    """Days between ``snapshot.as_of`` and today.

    Callers use this to warn that published numbers have gone stale. The result
    is negative when the snapshot is dated in the future, which means a hand
    edit or a skewed clock rather than a fresh snapshot, so it is reported
    rather than clamped.
    """
    return ((today or dt.date.today()) - snapshot.as_of).days


def find_model(snapshot: ReferenceSnapshot, identifier: str) -> ModelRecord | None:
    """Resolve a model identifier to its record, or ``None``.

    Three passes, each stricter than approximate matching:

    1. Exact match on ``id`` or on any declared alias, case-insensitively.
    2. The same, after dropping an OpenRouter routing tag such as ``:free``.
    3. Normalised match: casing, ``-``/``_``/``.``/space separators and a
       trailing date stamp are removed from both sides, and an OpenRouter-style
       ``vendor/`` prefix is stripped and then required to be one this record
       actually publishes under.

    The vendor guard is what stops ``openai/claude-opus-5`` from resolving.
    Pass 3 also refuses to answer when the normalised form matches more than one
    record, since picking one arbitrarily is how a verifier ends up comparing a
    provider against the wrong published scores.
    """
    ident = identifier.strip()
    if not ident:
        return None

    direct = snapshot.find(ident)
    if direct is not None:
        return direct

    untagged = _strip_routing_tag(ident)
    if untagged != ident:
        direct = snapshot.find(untagged)
        if direct is not None:
            return direct

    vendor, bare = _split_vendor(untagged)
    target = _normalise(bare)
    if not target:
        return None

    matches = []
    for record in snapshot.models:
        names, vendors = _record_keys(record)
        if vendor is not None and vendor not in vendors:
            continue
        if target in names:
            matches.append(record)
    return matches[0] if len(matches) == 1 else None


def _strip_routing_tag(identifier: str) -> str:
    head, sep, tag = identifier.partition(":")
    if sep and tag.strip().lower() in _ROUTING_TAGS:
        return head
    return identifier


def _split_vendor(identifier: str) -> tuple[str | None, str]:
    """Split ``anthropic/claude-opus-5`` into its vendor prefix and bare name.

    A leading ``~`` marks an OpenRouter alias entry such as
    ``~anthropic/claude-opus-latest`` and carries no meaning here.
    """
    stripped = identifier.lstrip("~")
    vendor, sep, rest = stripped.partition("/")
    if not sep:
        return None, stripped
    return vendor.strip().lower(), rest


def _normalise(name: str) -> str:
    return _SEPARATORS.sub("", _DATE_SUFFIX.sub("", name.strip().lower()))


def _record_keys(record: ModelRecord) -> tuple[set[str], set[str]]:
    """Normalised bare names and acceptable vendor prefixes for one record."""
    names: set[str] = set()
    vendors: set[str] = {record.vendor.strip().lower()}
    for raw in (record.id, *record.aliases):
        vendor, bare = _split_vendor(_strip_routing_tag(raw))
        if vendor:
            vendors.add(vendor)
        names.add(_normalise(bare))
    names.discard("")
    return names, vendors
