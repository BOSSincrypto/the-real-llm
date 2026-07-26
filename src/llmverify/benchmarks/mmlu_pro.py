"""MMLU-Pro: ten-option multiple choice across fourteen subject categories.

**Ten options, not four.** MMLU-Pro replaced MMLU's four choices with up to ten
(A-J) specifically to push random-guessing accuracy from 25% down to 10%, and
that changes what a low score means: an endpoint scoring 30% here is far worse
than an endpoint scoring 30% on a four-option benchmark. Option counts also vary
per row -- most items have ten, but some have as few as three -- so nothing in
this module assumes a fixed count.

**The rows are blocked by subject, which makes naive sampling misleading.**
Reading the first 600 rows of the split returns 600 business questions. That is
not a benchmark result, it is a report on how well an endpoint knows business,
and comparing it to a reference score measured over all fourteen categories is
meaningless. So rows are read from windows spread evenly across the split and
then interleaved by category, which costs a few more requests to the
datasets-server -- free, since dataset traffic does not touch the provider
budget -- and buys a sample that actually represents the benchmark. The
``categories`` argument narrows the sample deliberately, for a caller who wants
to know how an endpoint handles one subject.

**Why this benchmark is still here.** Anthropic, OpenAI and Google have all
dropped MMLU-Pro from flagship announcements, so almost every published score
belongs to an open-weight or open-API model: DeepSeek V4-Pro at 87.5 and
V4-Flash at 86.2, Amazon Nova 2 Pro at 81.6 and Lite at 80.9, Qwen3.7-Max
somewhere between 82 and 89.6 depending on which source you believe. Those are
exactly the models that get re-hosted, quantized and resold, which is the
failure mode this package exists to catch. A benchmark whose reference data
covers only open-weight models is useless for auditing a Claude endpoint and
close to the best available for auditing a DeepSeek one.

The published-score sample is small and one-sided, which is why
``discriminative`` is ``False``: five figures spanning 80.9 to 89.6 cannot
separate two competent models. What a knowledge benchmark does show clearly is
the tens-of-points collapse that follows an aggressive quantization or an
outright model swap.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import DatasetError
from .base import Benchmark, BenchmarkItem, register_benchmark
from .datasets import DatasetSpec

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = ["CATEGORIES", "MMLUPro"]

SPEC = DatasetSpec(
    dataset="TIGER-Lab/MMLU-Pro",
    config="default",
    split="test",
    gated=False,
    licence="mit",
)

#: Row count of the test split, verified against the datasets-server on
#: 2026-07-26. Used only to place the read windows; a split that has since grown
#: is still sampled correctly, and a split that has shrunk simply returns short
#: windows, which are skipped.
SPLIT_ROWS = 12032

#: Categories present in the split. Informational -- filtering matches whatever
#: the rows actually carry, so a new category does not need a code change.
CATEGORIES: tuple[str, ...] = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)

#: Rows per window. Small windows spread over many offsets beat few large ones:
#: the smallest category occupies about 380 consecutive rows, so windows must be
#: spaced closer than that or a whole subject can be missed.
WINDOW_ROWS = 25

#: Enough windows that the spacing (12032/32 = 376) stays under the smallest
#: category's block length.
MIN_WINDOWS = 32

#: Ceiling on windows, and so on requests. Reading the whole split would be 482
#: requests; this caps a careless ``limit`` at 120 requests and 3,000 rows,
#: which is far more than any run samples.
MAX_WINDOWS = 120


@register_benchmark
class MMLUPro(Benchmark):
    """Ten-option subject-matter multiple choice, sampled across all categories."""

    name: ClassVar[str] = "mmlu_pro"
    reference_key: ClassVar[str] = "mmlu_pro"
    hf_dataset: ClassVar[str | None] = SPEC.dataset
    hf_config: ClassVar[str | None] = SPEC.config
    hf_split: ClassVar[str] = SPEC.split
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = SPEC.licence
    #: Five published 2026 scores, all open-weight or open-API models, spanning
    #: 80.9 to 89.6. Too narrow a band, and too small a sample, to separate two
    #: competent models.
    discriminative: ClassVar[bool] = False
    #: Standard deviation of those five figures (87.5, 86.2, 81.6, 80.9, 89.6).
    #: Computed from the reference digest rather than published as such, and
    #: resting on a sample of five, so it is a weak ordering hint at best.
    score_spread: ClassVar[float] = 3.8
    description: ClassVar[str] = (
        "Subject-matter multiple choice with up to ten options (A-J), sampled across all "
        "fourteen categories rather than read off the front of the split, which is blocked by "
        "subject. Reference scores exist mainly for open-weight models -- Anthropic, OpenAI and "
        "Google have all dropped MMLU-Pro from flagship announcements -- which is precisely "
        "where re-hosting and quantization fraud happen. Not discriminative between competent "
        "models: it detects collapse, not degradation."
    )

    def __init__(
        self, *, categories: Sequence[str] | None = None, stratified: bool = True
    ) -> None:
        """Configure sampling.

        ``categories`` restricts the sample to the named subjects, matched
        case-insensitively against the row's own ``category`` field.
        ``stratified`` interleaves the sample by category so that truncating
        the item list to a budget keeps the subject mix roughly even; turning it
        off preserves dataset order, which is only useful for reproducing
        another harness.
        """
        self.categories = (
            frozenset(name.strip().casefold() for name in categories) if categories else None
        )
        self.stratified = stratified

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Read windows spread across the split and interleave them by category."""
        target = limit if limit is not None else SPLIT_ROWS
        windows = max(MIN_WINDOWS, math.ceil(max(target, 1) / WINDOW_ROWS))
        if self.categories is not None:
            # A filtered sample keeps only a fraction of what is read, so read
            # as widely as the ceiling allows rather than as narrowly as the
            # limit suggests.
            windows = MAX_WINDOWS
        windows = min(MAX_WINDOWS, windows)

        rows: list[dict[str, Any]] = []
        seen_offsets: set[int] = set()
        for offset in _offsets(windows):
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            rows.extend(await loader.rows(SPEC, limit=WINDOW_ROWS, offset=offset))

        items = [item for item in map(self._item, rows) if item is not None]
        deduplicated = list({item.id: item for item in items}.values())

        if not deduplicated:
            keys = ", ".join(sorted(rows[0])) if rows else "none"
            selected = ", ".join(sorted(self.categories)) if self.categories else "all"
            raise DatasetError(
                f"{SPEC} returned {len(rows)} row(s) but none produced a gradable item "
                f"(categories requested: {selected}). Columns present: {keys}."
            )

        ordered = _interleave_by_category(deduplicated) if self.stratified else deduplicated
        return ordered[:limit] if limit is not None else ordered

    def _item(self, row: dict[str, Any]) -> BenchmarkItem | None:
        """Turn one row into an item, or ``None`` if it cannot be trusted."""
        question = row.get("question")
        options = row.get("options")
        index = row.get("answer_index")
        letter = row.get("answer")
        category = str(row.get("category") or "").strip()

        if not isinstance(question, str) or not question.strip():
            return None
        if not isinstance(options, list) or len(options) < 2:
            return None
        if self.categories is not None and category.casefold() not in self.categories:
            return None

        texts = [str(option).strip() for option in options]
        if not all(texts) or len(set(texts)) != len(texts):
            return None

        position = _answer_position(index, letter, len(texts))
        if position is None:
            return None

        identifier = row.get("question_id")
        if identifier is None:
            # Never ``hash()``: it is salted per process, so ids built from it
            # would differ between runs and break both caching and the paired
            # comparisons the evasion probe makes.
            identifier = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        return BenchmarkItem(
            id=f"mmlu-pro-{identifier}",
            question=question.strip(),
            answer=texts[position],
            choices=tuple(texts),
            meta={
                "category": category,
                "source": str(row.get("src") or ""),
                "n_options": len(texts),
                "published_letter": letter if isinstance(letter, str) else None,
            },
        )


def _offsets(windows: int) -> list[int]:
    """Evenly spaced read offsets covering the split."""
    if windows <= 1:
        return [0]
    span = max(0, SPLIT_ROWS - WINDOW_ROWS)
    return [round(step * span / (windows - 1)) for step in range(windows)]


def _answer_position(index: Any, letter: Any, count: int) -> int | None:
    """Resolve the correct option's position from the two keys the row carries.

    The rows publish both ``answer_index`` and a letter, and they agreed on
    every row checked. Requiring them to agree here costs nothing and means a
    silently re-keyed dataset revision drops rows instead of scoring a correct
    endpoint wrong.
    """
    position: int | None = None
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < count:
        position = index

    if isinstance(letter, str) and len(letter.strip()) == 1:
        from_letter = ord(letter.strip().upper()) - ord("A")
        if not 0 <= from_letter < count:
            return None
        if position is not None and position != from_letter:
            return None
        position = from_letter

    return position


def _interleave_by_category(items: Iterable[BenchmarkItem]) -> list[BenchmarkItem]:
    """Round-robin items across categories, so a prefix of the list is balanced.

    The caller downstream samples randomly from whatever this returns, but it
    also truncates to a budget, and truncating a subject-ordered list is how a
    run ends up asking one endpoint fourteen chemistry questions and calling the
    result an MMLU-Pro score.
    """
    buckets: dict[str, list[BenchmarkItem]] = {}
    for item in items:
        buckets.setdefault(str(item.meta.get("category") or ""), []).append(item)

    ordered: list[BenchmarkItem] = []
    names = sorted(buckets)
    for position in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for name in names:
            bucket = buckets[name]
            if position < len(bucket):
                ordered.append(bucket[position])
    return ordered
