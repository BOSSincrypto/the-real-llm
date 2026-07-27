"""Short factual questions with one unambiguous answer, graded by exact match.

This is the benchmark the runner should reach for first, and the reason is
entirely about spread rather than prestige. Across the 36 published 2026 scores
in the reference digest, SimpleQA Verified runs from 9.6 to 77.3 with a standard
deviation of 16.7 points. Separating GPT-5.6 Sol (71.6) from GPT-5.6 Terra
(43.1) at 95% confidence and 80% power takes about 47 items. The same
separation on GPQA Diamond -- where every 2026 flagship scores between 87 and 93
-- takes tens of thousands. Cost per item is also the lowest of any benchmark
here: the questions are one line long and the answers are a few words, so no
reasoning budget is spent producing something the grader then throws away.

**The reference key is not quite this dataset.** ``reference_key`` is
``simpleqa_verified`` because that is what the published numbers are for: a
curated subset with a re-checked answer key, not the original 4,326-item set
that :attr:`SimpleQAVerified.hf_dataset` points at. Running the original items
against a Verified score is comparing two related but distinct measurements.
The comparison is still worth making -- the questions are drawn from the same
pool and the score gap between models dwarfs the gap between subsets -- but it
is approximate, and the probe's tolerance allowance is doing real work here.

**This grader is stricter than the one the labs used.** The published SimpleQA
figures come from judge-based grading, which accepts any semantically correct
phrasing. Grading with a judge would put a second trusted model inside a tool
whose entire job is to check whether a model can be trusted, so this
implementation matches strings instead: normalisation, a small closed alias
table, and a date parser.
That systematically *under*-scores every endpoint, by an amount this package has
not measured. Two consequences follow, and both belong in the report rather than
in a footnote. Comparing this benchmark's absolute number against a lab's
published figure understates the endpoint -- which is the direction that
produces false accusations, not missed ones. And because the shortfall applies
equally to every model, comparisons *within* a run -- verbatim against
paraphrased, candidate against baseline -- stay valid.

Everything the alias table does is applied to the gold answer and to the
model's answer alike, so it can never turn a right answer into a wrong one. It
can only merge two spellings of the same answer, which is what it is for.
"""

from __future__ import annotations

import ast
import datetime as dt
import re
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import DatasetError
from .base import Benchmark, BenchmarkItem, normalise_text, register_benchmark
from .datasets import DatasetSpec

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = ["SimpleQAVerified", "answer_variants", "same_answer"]

#: Verified live on 2026-07-26 against the datasets-server: 4,326 rows, columns
#: ``metadata`` / ``problem`` / ``answer``, MIT licensed, topics interleaved
#: rather than blocked. That last property is why a contiguous read from offset
#: zero is an acceptable sample and no windowing is needed, unlike MMLU-Pro.
SPEC = DatasetSpec(
    dataset="basicv8vc/SimpleQA",
    config="default",
    split="test",
    gated=False,
    licence="mit",
)

#: Rows read when the caller sets no limit. The full split is 4,326 rows, which
#: is 44 requests to the datasets-server for a run that will sample at most a
#: few hundred items.
DEFAULT_LIMIT = 600


# --------------------------------------------------------------------------- #
# Answer canonicalisation
# --------------------------------------------------------------------------- #

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")

#: Whole-word phrase aliases, all written in post-:func:`normalise_text` form
#: (casefolded, apostrophes gone, commas turned to spaces). Deliberately short.
#: Every entry here is a pair of spellings that any human grader would accept as
#: the same answer; anything requiring world knowledge to equate is left out.
#: "America" is absent on purpose, because it would also rewrite "South America".
_PHRASE_ALIASES: dict[str, str] = {
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.a": "united states",
    "u.s.": "united states",
    "u.s": "united states",
    "us": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "u.k": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
    "uae": "united arab emirates",
    "ussr": "soviet union",
    "nyc": "new york city",
}

_CARDINAL_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}

_MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sept": 9,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Longest key first so "united states of america" wins over "us".
_PHRASE_RE = re.compile(
    r"(?<![\w.])(?:"
    + "|".join(re.escape(key) for key in sorted(_PHRASE_ALIASES, key=len, reverse=True))
    + r")(?![\w])"
)

_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")

# Only formats whose field order cannot be misread. "05/01/2020" is excluded
# on purpose: it is 5 January in Britain and 1 May in the United States, and a
# grader that guesses wrong marks a correct answer wrong.
_ISO_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_ISO_MONTH_RE = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_DAY_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\.?\s+(\d{4})$")
_MONTH_DAY_YEAR_RE = re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})$")
_MONTH_YEAR_RE = re.compile(r"^([a-z]+)\.?\s+(\d{4})$")


def _ordinal_form(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _token_alias(token: str) -> str:
    if token in _ORDINAL_WORDS:
        return _ordinal_form(_ORDINAL_WORDS[token])
    if token in _CARDINAL_WORDS:
        return str(_CARDINAL_WORDS[token])
    return token


def _canonical(text: str) -> str:
    """Normalise, drop a leading article, and apply the alias table."""
    base = normalise_text(text)
    base = _ARTICLE_RE.sub("", base)
    base = _PHRASE_RE.sub(lambda m: _PHRASE_ALIASES[m.group(0)], base)
    base = " ".join(_token_alias(token) for token in base.split())
    return base.strip(" .")


def _iso_date(text: str) -> str | None:
    """Return ``YYYY-MM-DD`` or ``YYYY-MM`` for a date this parser is sure of."""
    body = text.strip()

    match = _ISO_RE.match(body)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return _valid(year, month, day)

    match = _ISO_MONTH_RE.match(body)
    if match:
        year, month = (int(part) for part in match.groups())
        return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None

    match = _DAY_MONTH_YEAR_RE.match(body)
    if match and match.group(2) in _MONTHS:
        return _valid(int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))

    match = _MONTH_DAY_YEAR_RE.match(body)
    if match and match.group(1) in _MONTHS:
        return _valid(int(match.group(3)), _MONTHS[match.group(1)], int(match.group(2)))

    match = _MONTH_YEAR_RE.match(body)
    if match and match.group(1) in _MONTHS:
        return f"{int(match.group(2)):04d}-{_MONTHS[match.group(1)]:02d}"

    return None


def _valid(year: int, month: int, day: int) -> str | None:
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def answer_variants(answer: str) -> frozenset[str]:
    """Every spelling of ``answer`` this grader treats as the same answer.

    Two answers match when their variant sets intersect. Because the same
    function runs over the gold answer and over the model's, a transform here
    can only ever merge two spellings of one answer -- it cannot make a wrong
    answer match a right one unless the two were already written the same way.

    Beyond the alias table the variants cover three mechanical differences that
    show up constantly and mean nothing: internal full stops (``J.R.R. Tolkien``
    against ``JRR Tolkien``), hyphens and slashes used where a space would do,
    and an ordinal suffix on a bare number (``3rd`` against ``3``).
    """
    base = _canonical(answer)
    if not base:
        return frozenset()

    forms = {base}
    forms.add(base.replace(".", ""))
    forms.add(re.sub(r"[-/]", " ", base))
    forms.add(re.sub(r"[-/]", " ", base.replace(".", "")))

    iso = _iso_date(base)
    if iso is not None:
        forms.add(iso)

    forms |= {_ORDINAL_SUFFIX_RE.sub(r"\1", form) for form in tuple(forms)}
    return frozenset(re.sub(r"\s+", " ", form).strip() for form in forms if form.strip())


def same_answer(predicted: str, gold: str) -> bool:
    """Whether two answer strings are the same answer under this grader."""
    return bool(answer_variants(predicted) & answer_variants(gold))


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


@register_benchmark
class SimpleQAVerified(Benchmark):
    """SimpleQA short-answer accuracy, graded by normalised exact match."""

    name: ClassVar[str] = "simpleqa_verified"
    reference_key: ClassVar[str] = "simpleqa_verified"
    hf_dataset: ClassVar[str | None] = SPEC.dataset
    hf_config: ClassVar[str | None] = SPEC.config
    hf_split: ClassVar[str] = SPEC.split
    gated: ClassVar[bool] = SPEC.gated
    licence: ClassVar[str] = SPEC.licence
    discriminative: ClassVar[bool] = True
    #: Standard deviation of the 36 published 2026 SimpleQA Verified scores,
    #: which span 9.6 to 77.3. The widest spread of any benchmark here that
    #: costs one short answer per item.
    score_spread: ClassVar[float] = 16.7
    description: ClassVar[str] = (
        "Short factual questions with a single unambiguous answer. The widest cheap "
        "discriminator available: published 2026 scores span 9.6 to 77.3, so roughly 47 items "
        "separate two frontier models that GPQA Diamond could not separate with fifty thousand. "
        "Two caveats belong with any number it produces. The reference scores are for SimpleQA "
        "Verified, a curated subset, while the items run here come from the original 4,326-item "
        "set. And this grader matches strings where the published figures came from a model "
        "judge, so it under-scores every endpoint alike -- fine for comparing runs against each "
        "other, not directly comparable to a lab's headline figure."
    )
    answer_instruction: ClassVar[str] = (
        "Answer with the shortest phrase that fully answers the question -- a name, a number "
        "or a date -- and no explanation. Put it on its own last line, formatted exactly as "
        '"ANSWER: <answer>". Write any date as YYYY-MM-DD.'
    )

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Read questions from the front of the split, in file order.

        Taking a contiguous prefix is only defensible because the split's topics
        are interleaved rather than blocked, which was checked directly against
        the datasets-server. A benchmark whose rows are grouped by subject needs
        the windowed read that :mod:`llmverify.benchmarks.mmlu_pro` uses.
        """
        rows = await loader.rows(SPEC, limit=limit if limit is not None else DEFAULT_LIMIT)

        items: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            question = _text(row, "problem", "question", "prompt")
            answer = _text(row, "answer", "gold", "target")
            if not question or not answer:
                continue
            items.append(
                BenchmarkItem(
                    # Positional, because the split has no id column. Stable as
                    # long as the dataset revision is, which is what a rerun
                    # needs; a dataset update invalidates ids and cache alike.
                    id=f"simpleqa-{index:05d}",
                    question=question,
                    answer=answer,
                    meta={"row_index": index, **_metadata(row.get("metadata"))},
                )
            )

        if not items:
            keys = ", ".join(sorted(rows[0])) if rows else "none"
            raise DatasetError(
                f"{SPEC} returned {len(rows)} row(s) but none carried a question and an answer "
                f"in the expected columns. Columns present: {keys}."
            )
        return items

    def grade(
        self, item: BenchmarkItem, response_text: str, state: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Exact match against the gold answer, up to the alias table."""
        extracted = self.extract_answer(response_text)
        if extracted is None:
            return None, None
        return same_answer(extracted, item.answer), extracted


def _text(row: dict[str, Any], *names: str) -> str:
    """First non-empty value among ``names``, matched case-insensitively."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata(raw: Any) -> dict[str, Any]:
    """Parse the ``metadata`` cell, which holds a Python literal, not JSON.

    Failure is not an error: topic and answer type are reporting colour, and an
    item is perfectly gradable without them.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        key: parsed[key] for key in ("topic", "answer_type") if isinstance(parsed.get(key), str)
    }
