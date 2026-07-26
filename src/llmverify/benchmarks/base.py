"""Benchmark contract.

A benchmark supplies items, renders them into prompts, and grades responses
deterministically. Grading never calls another model: an LLM judge would add a
second trust assumption to a tool whose entire purpose is to check a trust
assumption.

Two things distinguish this from a normal eval harness.

**Variants.** Each item can be rendered verbatim, paraphrased, or with its
answer options permuted. A provider that routes recognisable benchmark strings
to the real model while serving everything else from a cheaper one will score
differently across variants, and that gap is measured directly by the evasion
probe.

**Discriminative power.** Most famous benchmarks cannot tell 2026 frontier
models apart. Every 2026 flagship scores 87-93% on GPQA Diamond, so separating
two of them at 95% confidence needs tens of thousands of items against the 198
that exist. Each benchmark declares its ``discriminative`` status honestly, and
the runner prefers high-variance benchmarks (SimpleQA Verified, ARC-AGI-2)
over prestigious saturated ones.
"""

from __future__ import annotations

import abc
import enum
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..types import Message, Role

if TYPE_CHECKING:
    from . import datasets as _datasets

__all__ = [
    "Benchmark",
    "BenchmarkItem",
    "GradedResult",
    "Variant",
    "all_benchmarks",
    "get_benchmark",
    "normalise_text",
    "register_benchmark",
]


class Variant(str, enum.Enum):
    """How an item is presented to the endpoint."""

    #: Exactly as published. Comparable to reference scores, but recognisable.
    VERBATIM = "verbatim"
    #: Semantically identical, textually different. Not string-matchable against
    #: a published dataset, so it defeats lookup-based routing.
    PARAPHRASED = "paraphrased"
    #: Verbatim wording, permuted answer options. Defeats answer-key memorisation
    #: without changing a single content word.
    SHUFFLED = "shuffled"


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One question, independent of how it will be rendered."""

    id: str
    question: str
    #: Gold answer. For multiple choice this is the *text* of the correct
    #: option, never its letter -- letters move under ``SHUFFLED``.
    answer: str
    choices: tuple[str, ...] = ()
    #: Anything the grader or renderer needs: constraint lists for IFEval,
    #: contest dates for date-filtered sets, and so on.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)


@dataclass(slots=True)
class GradedResult:
    item_id: str
    variant: Variant
    correct: bool | None
    raw_response: str
    extracted: str | None = None
    expected: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    error: str | None = None


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalise_text(text: str) -> str:
    """Casefold, strip punctuation and collapse whitespace, for exact matching."""
    text = text.strip().casefold()
    text = re.sub(r"[‘’“”]", "'", text)
    text = re.sub(r"[^\w\s.\-/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Benchmark(abc.ABC):
    """A source of gradable items."""

    name: ClassVar[str]
    #: Key used to look up published scores in the reference snapshot.
    reference_key: ClassVar[str]
    #: HuggingFace dataset spec, or ``None`` for procedurally generated sets.
    hf_dataset: ClassVar[str | None] = None
    hf_config: ClassVar[str | None] = None
    hf_split: ClassVar[str] = "test"
    #: Whether the dataset requires an accepted licence and an ``HF_TOKEN``.
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = "unknown"
    #: Honest assessment of whether this benchmark can separate current
    #: frontier models. Saturated benchmarks stay available -- a substituted
    #: model is often far below frontier -- but are weighted down and labelled.
    discriminative: ClassVar[bool] = True
    #: Standard deviation of published scores across 2026 models, in points.
    #: The runner uses this to order benchmarks by information per request.
    score_spread: ClassVar[float] = 10.0
    #: Whether responses need a code sandbox to grade. Unsupported for now;
    #: benchmarks declaring True are excluded rather than silently mis-graded.
    needs_sandbox: ClassVar[bool] = False
    description: ClassVar[str] = ""
    #: Instruction appended to every prompt so answers are machine-extractable.
    answer_instruction: ClassVar[str] = ""

    @abc.abstractmethod
    async def load(self, loader: _datasets.DatasetLoader, *, limit: int | None = None) -> list[BenchmarkItem]:
        """Fetch and parse items. Implementations must be deterministic."""

    # ------------------------------------------------------------------ render

    def render(
        self, item: BenchmarkItem, *, variant: Variant, rng: random.Random
    ) -> tuple[tuple[Message, ...], dict[str, Any]]:
        """Build the messages for one item, plus render state the grader needs.

        The returned dict carries anything the grader cannot recover from the
        item alone -- most importantly the option ordering under ``SHUFFLED``.
        """
        question = item.question
        if variant is Variant.PARAPHRASED:
            from .paraphrase import paraphrase

            question = paraphrase(question, rng=rng)

        state: dict[str, Any] = {"variant": variant.value}
        body = question

        if item.is_multiple_choice:
            options = list(item.choices)
            if variant is Variant.SHUFFLED:
                rng.shuffle(options)
            state["options"] = options
            lettered = "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
            body = f"{question}\n\n{lettered}"

        instruction = self.answer_instruction or self._default_instruction(item)
        if instruction:
            body = f"{body}\n\n{instruction}"

        return (Message(Role.USER, body),), state

    def _default_instruction(self, item: BenchmarkItem) -> str:
        if item.is_multiple_choice:
            return (
                "Answer with the single letter of the correct option on its own "
                'final line, formatted exactly as "ANSWER: X".'
            )
        return (
            "Put your final answer on its own last line, formatted exactly as "
            '"ANSWER: <answer>".'
        )

    # ------------------------------------------------------------------- grade

    def grade(
        self, item: BenchmarkItem, response_text: str, state: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Return ``(correct, extracted_answer)``.

        ``correct`` is ``None`` when no answer could be extracted at all, which
        is tracked separately from a wrong answer: a provider that mangles the
        output format is failing differently from one that reasons badly.
        """
        extracted = self.extract_answer(response_text)
        if extracted is None:
            return None, None

        if item.is_multiple_choice:
            options: list[str] = state.get("options") or list(item.choices)
            letter = extracted.strip().upper()[:1]
            if letter in _LETTERS[: len(options)]:
                return options[_LETTERS.index(letter)] == item.answer, letter
            # Some models answer with the option text instead of the letter.
            target = normalise_text(item.answer)
            return normalise_text(extracted) == target, extracted

        return normalise_text(extracted) == normalise_text(item.answer), extracted

    _ANSWER_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"ANSWER\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
    )

    def extract_answer(self, text: str) -> str | None:
        """Pull the final answer out of a free-form response."""
        matches = self._ANSWER_RE.findall(text or "")
        if matches:
            return matches[-1].strip().strip("*`\"'.")
        # Fall back to a bare final line that is just an option letter.
        for line in reversed((text or "").strip().splitlines()):
            stripped = line.strip().strip("*`\"'.() ")
            if len(stripped) == 1 and stripped.upper() in _LETTERS:
                return stripped.upper()
        return None


_BENCHMARKS: dict[str, type[Benchmark]] = {}


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    _BENCHMARKS[cls.name] = cls
    return cls


def all_benchmarks() -> dict[str, type[Benchmark]]:
    from . import (  # noqa: F401
        aime,
        gpqa,
        ifeval,
        mmlu_pro,
        simpleqa,
        synthetic,
    )

    from importlib.metadata import entry_points

    try:
        eps = entry_points(group="llmverify.benchmarks")
    except TypeError:  # pragma: no cover
        eps = entry_points().get("llmverify.benchmarks", [])  # type: ignore[assignment]
    for ep in eps:
        if ep.name in _BENCHMARKS:
            continue
        try:
            cls = ep.load()
        except Exception:
            continue
        if isinstance(cls, type) and issubclass(cls, Benchmark):
            _BENCHMARKS[cls.name] = cls
    return dict(_BENCHMARKS)


def get_benchmark(name: str) -> Benchmark:
    from ..errors import ConfigError

    registry = all_benchmarks()
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ConfigError(f"unknown benchmark {name!r}; available: {known}")
    return registry[name]()
