"""AIME contest problems: one integer answer, graded exactly.

Every AIME answer is an integer between 0 and 999, which makes this the
cleanest grading signal in the package. There is no answer key to interpret, no
alias table, no judge and no partial credit: the response either contains the
right integer or it does not. Whatever a run measures here, it is not measuring
the grader.

Three years are provided, and they are not interchangeable.

**2026 is the useful one.** ``MathArena/aime_2026`` was published after the
training cutoff of every model in the reference snapshot, so a score on it is
much less likely to be reciting a memorised solution. It is also the only one of
the three where a published score does not sit near the ceiling.

**2024 and 2025 are near-saturated.** Published scores on the 2024/2025-era sets
cluster between 80 and 100 -- Epoch's OTIS Mock AIME series spans 80.0 to 100.0
across 35 models with a standard deviation of 5.3. That is barely more spread
than GPQA Diamond, so a frontier endpoint scoring 93 instead of 97 tells you
nothing. What these sets still do well is catch a substitution that is far below
frontier: a model that has to actually solve the problems rather than recall
them falls off a cliff, and the cliff is visible in a handful of items.

**Thirty items is thirty items.** Each contest year is exactly 30 problems.
A Wilson interval on 30 binary trials is roughly +/-18 points wide, so a single
year supports "this endpoint is nowhere near the claimed model" and nothing
finer. Running several years together is the only way to buy precision here,
and even then the total is 90.

Two rendering rules follow from the content. Paraphrasing pins
:data:`~llmverify.benchmarks.paraphrase.SAFE_MODE` on rather than inheriting the
module default, because a rewrite that reflows a quantity or a serial comma
inside a competition problem can change the answer, and a paraphrase that
changes the answer manufactures a false accusation. And option shuffling has
nothing to permute, so :attr:`~llmverify.benchmarks.base.Variant.SHUFFLED` is
rendered verbatim and says so in the render state rather than pretending a
permutation happened.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import DatasetError
from ..types import Message
from .base import Benchmark, BenchmarkItem, Variant, register_benchmark
from .datasets import DatasetSpec
from .paraphrase import paraphrase

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = ["AIME2024", "AIME2025", "AIME2026", "extract_integer"]

#: AIME answers are integers in this closed range, by the contest's own rules.
#: A parsed value outside it means the extractor misread something, so the row
#: or the response is discarded rather than graded.
ANSWER_MIN = 0
ANSWER_MAX = 999

_ANSWER_INSTRUCTION = (
    "The answer is an integer between 0 and 999. Put it on its own last line, formatted "
    'exactly as "ANSWER: <integer>", with no commas, units, LaTeX or explanation on that line.'
)

# Wider than the 0-999 the contest allows, deliberately. An endpoint that
# answers 1000000 has answered wrongly, not unintelligibly, and grading it as an
# extraction failure would quietly drop it from the accuracy test instead.
_INTEGER_RE = re.compile(r"^[+-]?\d{1,12}$")
_BOXED_RE = re.compile(r"\\(?:boxed|fbox|framebox)\s*\{")
_LATEX_NOISE_RE = re.compile(r"\\(?:text|mathrm|mbox|displaystyle|left|right|,|;|!|\s)")


def _last_boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}`` in ``text``, brace-matched.

    Brace matching rather than a regex because the argument legitimately
    contains braces of its own -- ``\\boxed{\\frac{1}{2}}`` is common even on a
    benchmark whose answers are integers, and a lazy regex would return
    ``\\frac{1`` and grade it as an extraction failure.
    """
    last: str | None = None
    for match in _BOXED_RE.finditer(text):
        depth = 1
        start = match.end()
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            last = text[start : index - 1]
    return last


def extract_integer(candidate: str | None) -> int | None:
    """Read an integer out of an answer fragment, or return ``None``.

    Tolerates the wrappers models put around a number when asked not to --
    dollar signs, thousands separators, ``\\text{}``, a trailing full stop --
    because none of them change what the number is. Anything else is treated as
    a failure to answer rather than guessed at.
    """
    if candidate is None:
        return None
    body = _LATEX_NOISE_RE.sub(" ", candidate)
    body = body.replace("$", " ").replace("{", " ").replace("}", " ")
    body = body.replace(",", "").replace("_", "").strip().strip(".")
    body = re.sub(r"\s+", "", body)
    if not _INTEGER_RE.match(body):
        return None
    return int(body)


class _AIMEBenchmark(Benchmark):
    """Shared loading, rendering and grading for one AIME contest year."""

    #: Set by each concrete year.
    spec: ClassVar[DatasetSpec]

    answer_instruction: ClassVar[str] = _ANSWER_INSTRUCTION

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Read one contest year. Each year is 30 problems, so ``limit`` rarely binds."""
        rows = await loader.rows(self.spec, limit=limit)

        items: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            question = _string(row, "problem", "question")
            answer = extract_integer(_string(row, "answer", "solution_answer", "gold"))
            if not question or answer is None or not ANSWER_MIN <= answer <= ANSWER_MAX:
                continue
            number = row.get("problem_idx") or row.get("id") or index + 1
            items.append(
                BenchmarkItem(
                    id=f"{self.name}-{_as_int(number, index + 1):03d}",
                    question=question,
                    answer=str(answer),
                    meta={"contest": self.name, "row_index": index},
                )
            )

        if not items:
            keys = ", ".join(sorted(rows[0])) if rows else "none"
            raise DatasetError(
                f"{self.spec} returned {len(rows)} row(s) but none carried a problem and an "
                f"integer answer in 0-999. Columns present: {keys}."
            )
        return items

    def render(
        self, item: BenchmarkItem, *, variant: Variant, rng: random.Random
    ) -> tuple[tuple[Message, ...], dict[str, Any]]:
        """Render one problem, pinning safe-mode paraphrase and skipping shuffling."""
        if variant is Variant.SHUFFLED:
            messages, state = super().render(item, variant=Variant.VERBATIM, rng=rng)
            state["shuffle_applicable"] = False
            return messages, state

        if variant is Variant.PARAPHRASED:
            rewritten = paraphrase(item.question, rng=rng, safe_mode=True)
            messages, state = super().render(
                replace(item, question=rewritten), variant=Variant.VERBATIM, rng=rng
            )
            state["variant"] = Variant.PARAPHRASED.value
            state["paraphrase_safe_mode"] = True
            return messages, state

        return super().render(item, variant=variant, rng=rng)

    def extract_answer(self, text: str) -> str | None:
        """Prefer the requested ``ANSWER:`` line, then a boxed answer, then a bare line.

        The order is deliberate. The ``ANSWER:`` line is what the prompt asked
        for, so honouring it first keeps the extractor from preferring a boxed
        intermediate result over a stated final one. ``\\boxed{}`` comes next
        because it is the convention every maths harness and most models use
        unprompted. The last fallback only fires when the final line of the
        response is nothing but an integer, which cannot be confused with a
        number quoted mid-sentence.
        """
        stated = extract_integer(super().extract_answer(text))
        if stated is not None:
            return str(stated)

        boxed = extract_integer(_last_boxed(text or ""))
        if boxed is not None:
            return str(boxed)

        for line in reversed((text or "").strip().splitlines()):
            stripped = line.strip().strip("*`$ ").strip(".")
            if stripped:
                value = extract_integer(stripped)
                return str(value) if value is not None else None
        return None

    def grade(
        self, item: BenchmarkItem, response_text: str, state: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Integer equality. Both sides are already canonical decimal strings."""
        extracted = self.extract_answer(response_text)
        if extracted is None:
            return None, None
        return extracted == item.answer, extracted


@register_benchmark
class AIME2026(_AIMEBenchmark):
    """AIME 2026, the freshest and least contaminated of the three years."""

    name: ClassVar[str] = "aime_2026"
    reference_key: ClassVar[str] = "aime_2026"
    hf_dataset: ClassVar[str | None] = "MathArena/aime_2026"
    hf_config: ClassVar[str | None] = "default"
    hf_split: ClassVar[str] = "train"
    gated: ClassVar[bool] = False
    #: The NC clause restricts commercial use of the items, which is why the
    #: licence is carried into the report instead of being assumed permissive.
    licence: ClassVar[str] = "cc-by-nc-sa-4.0"
    discriminative: ClassVar[bool] = True
    #: No cross-model spread has been measured for the 2026 set: the reference
    #: digest carries exactly one published score for it (GLM-5.2, 99.2). This
    #: value is an unverified estimate, placed above the 5.3 measured on the
    #: 2024/2025-era sets on the reasoning that a contest postdating every
    #: model's training cutoff is less saturated. Treat it as an ordering hint
    #: for the runner, not as a statistic.
    score_spread: ClassVar[float] = 8.0
    description: ClassVar[str] = (
        "AIME 2026: 30 competition problems with integer answers in 0-999, graded exactly. "
        "Published after every reference model's training cutoff, so a score here is less "
        "likely to be recall than on earlier years. Thirty items is enough to detect a badly "
        "degraded endpoint and not enough for a precise accuracy estimate. Its score_spread is "
        "an estimate: only one published 2026 score exists to compare against."
    )

    spec: ClassVar[DatasetSpec] = DatasetSpec(
        dataset="MathArena/aime_2026",
        config="default",
        split="train",
        gated=False,
        licence="cc-by-nc-sa-4.0",
    )


@register_benchmark
class AIME2025(_AIMEBenchmark):
    """AIME 2025. Near-saturated: useful against weak substitutions only."""

    name: ClassVar[str] = "aime_2025"
    reference_key: ClassVar[str] = "aime_2025"
    hf_dataset: ClassVar[str | None] = "MathArena/aime_2025"
    hf_config: ClassVar[str | None] = "default"
    hf_split: ClassVar[str] = "train"
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = "cc-by-nc-sa-4.0"
    discriminative: ClassVar[bool] = False
    #: Standard deviation of the 35 published scores on Epoch's OTIS Mock AIME
    #: 2024-2025 series, which span 80.0 to 100.0.
    score_spread: ClassVar[float] = 5.3
    description: ClassVar[str] = (
        "AIME 2025: 30 problems, integer answers, graded exactly. Near-saturated -- published "
        "scores on the 2024/2025-era sets cluster between 80 and 100 (sd 5.3) -- so a frontier "
        "result here cannot be told apart from any other frontier result. It still separates a "
        "frontier model from a substituted weaker one, which is the only claim it supports."
    )

    spec: ClassVar[DatasetSpec] = DatasetSpec(
        dataset="MathArena/aime_2025",
        config="default",
        split="train",
        gated=False,
        licence="cc-by-nc-sa-4.0",
    )


@register_benchmark
class AIME2024(_AIMEBenchmark):
    """AIME 2024. The most contaminated of the three, kept for continuity."""

    name: ClassVar[str] = "aime_2024"
    reference_key: ClassVar[str] = "aime_2024"
    hf_dataset: ClassVar[str | None] = "HuggingFaceH4/aime_2024"
    hf_config: ClassVar[str | None] = "default"
    hf_split: ClassVar[str] = "train"
    gated: ClassVar[bool] = False
    #: The dataset card carries no licence tag. Secondary listings describe the
    #: 2024 problem set as MIT-licensed, but that could not be confirmed from
    #: the repository itself, so it is recorded as unverified rather than
    #: asserted.
    licence: ClassVar[str] = "unverified"
    discriminative: ClassVar[bool] = False
    score_spread: ClassVar[float] = 5.3
    description: ClassVar[str] = (
        "AIME 2024: 30 problems, integer answers, graded exactly. Saturated and old enough to "
        "be well represented in training corpora, so a high score may be recall rather than "
        "reasoning. Kept because a low score is still strong evidence against the endpoint. "
        "Licence unverified: the dataset card carries no tag."
    )

    spec: ClassVar[DatasetSpec] = DatasetSpec(
        dataset="HuggingFaceH4/aime_2024",
        config="default",
        split="train",
        gated=False,
        licence="unverified",
    )


def _string(row: dict[str, Any], *names: str) -> str:
    """First non-empty value among ``names``, matched case-insensitively.

    Numbers are stringified: the 2026 and 2025 repositories type ``answer`` as
    ``int64`` while the 2024 one types it as a string.
    """
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
