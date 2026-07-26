"""GPQA Diamond: graduate-level science multiple choice, and its hard limit.

**This benchmark cannot separate current frontier models, and the tool must not
pretend otherwise.** Every 2026 flagship scores between 87 and 93 on GPQA
Diamond; across the 35 published scores in the reference digest the spread is
21.5 points with a standard deviation of 5.0, and most of that spread comes from
the models nobody would confuse. Separating Claude Opus 5 (91.8) from GPT-5.6
Sol (91.3) at 95% confidence and 80% power needs roughly 48,500 items per arm.
GPQA Diamond contains 198. That is not an expensive experiment, it is an
impossible one, and no amount of budget changes it.

What remains is worth having, as long as the claim is stated narrowly. A
substitution that is *far* below frontier -- a small open-weight model, or a
badly quantized copy of the right one -- shows up here in a few dozen items,
because the gap being tested is then tens of points rather than half a point.
So the benchmark stays available, declares ``discriminative = False`` so the
runner ranks it below the wide-spread benchmarks, and supports exactly one
conclusion: consistency with the reference rules out a badly degraded endpoint
and says nothing at all about which frontier model is behind the endpoint.

**Access and republication.** The dataset is gated on HuggingFace with
auto-approval: accept the terms on the dataset page and export a read token as
``HF_TOKEN`` (or whatever :attr:`~llmverify.config.RunConfig.hf_token_env`
names). The gate asks users not to reveal examples from the dataset in plain
text online, to keep them out of future training corpora. That request is why
this package fetches items at runtime instead of vendoring a copy of them into
the repository, and why nothing here caches question text anywhere except the
user's own cache directory.

**Option assembly.** The rows store the correct answer and the three distractors
in separate columns, so the four options have to be assembled here. Doing that
in column order would put the correct answer first every time, which turns the
benchmark into a test of whether a model can prefer option A. The base ordering
is therefore permuted with a generator seeded from the row's own record id: the
same item always gets the same ordering, on every machine and every run, while
the correct answer's position is effectively random across items. The
:attr:`~llmverify.benchmarks.base.Variant.SHUFFLED` variant permutes again on
top of that, with the run's seed, which is what defeats answer-key memorisation.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import DatasetError
from .base import Benchmark, BenchmarkItem, register_benchmark
from .datasets import DatasetSpec

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = ["GPQADiamond"]

SPEC = DatasetSpec(
    dataset="Idavidrein/gpqa",
    config="gpqa_diamond",
    split="train",
    gated=True,
    licence="cc-by-4.0",
)

#: Column names as documented for the GPQA CSVs. They could not be confirmed
#: against the live datasets-server the way every other benchmark's were,
#: because the repository is gated and answers 401 without an accepted licence.
#: Resolution is therefore case- and separator-insensitive and lists what it
#: actually found when it fails, so a schema change produces a fixable error
#: message rather than an empty item list.
QUESTION_COLUMNS = ("question", "problem")
CORRECT_COLUMNS = ("correct answer", "correct_answer", "answer")
INCORRECT_COLUMNS = (
    ("incorrect answer 1", "incorrect_answer_1", "distractor 1"),
    ("incorrect answer 2", "incorrect_answer_2", "distractor 2"),
    ("incorrect answer 3", "incorrect_answer_3", "distractor 3"),
)
ID_COLUMNS = ("record id", "record_id", "id")


@register_benchmark
class GPQADiamond(Benchmark):
    """Four-option graduate science questions. Saturated at the frontier."""

    name: ClassVar[str] = "gpqa_diamond"
    reference_key: ClassVar[str] = "gpqa_diamond"
    hf_dataset: ClassVar[str | None] = SPEC.dataset
    hf_config: ClassVar[str | None] = SPEC.config
    hf_split: ClassVar[str] = SPEC.split
    gated: ClassVar[bool] = True
    licence: ClassVar[str] = SPEC.licence
    #: Not a judgement about the questions, which are hard. It is a statement
    #: about the published scores: 87-93% for every 2026 flagship leaves no room
    #: for 198 items to resolve anything.
    discriminative: ClassVar[bool] = False
    #: Standard deviation of the 35 published 2026 scores (71.3 to 92.8).
    score_spread: ClassVar[float] = 5.0
    description: ClassVar[str] = (
        "Graduate-level science multiple choice, 198 items, four options assembled and "
        "deterministically permuted per row. Saturated: all 2026 flagships score 87-93%, and "
        "separating two of them would need about 48,500 items against the 198 that exist. It "
        "can show that an endpoint is far below frontier, and that is the only thing it can "
        "show. Gated -- needs accepted dataset terms and an HF token."
    )

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Fetch rows and assemble four options per item.

        Rows whose options cannot be trusted are dropped rather than repaired:
        a missing distractor would leave a three-option question graded against
        a four-option reference score, and a duplicated option would make two
        letters correct at once. Both are rare and neither is worth guessing at.
        """
        rows = await loader.rows(SPEC, limit=limit)

        items: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            question = _text(row, *QUESTION_COLUMNS)
            # Options are collapsed to one line each because the renderer lists
            # them as "A. <option>"; the collapsed form is also what is stored
            # as the gold answer, so the two stay comparable after shuffling.
            correct = _one_line(_text(row, *CORRECT_COLUMNS))
            distractors = [_one_line(_text(row, *names)) for names in INCORRECT_COLUMNS]
            if not question or not correct or not all(distractors):
                continue

            options = [correct, *distractors]
            if len({option.casefold() for option in options}) != len(options):
                continue

            record_id = _text(row, *ID_COLUMNS) or f"row-{index}"
            # Seeded from the record id, not the run seed: the base ordering is
            # a property of the item and must not move between runs, or a
            # cached response from an earlier run would be graded against a
            # different lettering.
            random.Random(f"gpqa_diamond:{record_id}").shuffle(options)

            items.append(
                BenchmarkItem(
                    id=f"gpqa-{record_id}",
                    question=question,
                    # The base contract stores the correct option's *text*,
                    # never its letter, precisely so that shuffling is safe.
                    answer=correct,
                    choices=tuple(options),
                    meta={
                        "row_index": index,
                        "subdomain": _text(row, "subdomain", "high-level domain", "domain"),
                    },
                )
            )

        if not items:
            keys = ", ".join(sorted(rows[0])) if rows else "none"
            raise DatasetError(
                f"{SPEC} returned {len(rows)} row(s) but none carried a question, a correct "
                f"answer and three distractors. Columns present: {keys}."
            )
        return items


def _text(row: dict[str, Any], *names: str) -> str:
    """First non-empty value among ``names``, ignoring case and separators.

    GPQA's columns are human-written headers with spaces and apostrophes in
    them, and the exact spelling could not be verified live, so lookup folds
    case and treats spaces, underscores and hyphens as the same character.
    """
    lowered = {_key(str(key)): value for key, value in row.items()}
    for name in names:
        value = lowered.get(_key(name))
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _key(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", name).strip().casefold()
