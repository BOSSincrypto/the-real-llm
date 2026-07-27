"""Procedurally generated items that no provider can have memorised.

Every other benchmark here has the same structural weakness: its items are
published, so a provider can recognise them. It can route them to the genuine
model while serving everything else from something cheaper, or it can have
absorbed the answer key during training and recite it. Paraphrasing raises the
cost of the first attack; it does nothing about the second.

These items are generated locally from a seed, so neither attack is available.
An arithmetic chain over six-digit operands drawn at run time has never appeared
in any corpus, cannot be looked up, and has an answer that is checkable to the
digit. The generator is deterministic, so the same seed produces the same items
on any machine, and no network is touched at all -- which also makes this the
benchmark that still works when a dataset is unavailable, gated, or has been
pulled from HuggingFace.

**There is no reference score, and that is not a detail.** Nobody publishes an
accuracy on questions that did not exist until this run generated them, so there
is nothing to compare an absolute number against. ``reference_key`` is therefore
set to :data:`NO_REFERENCE_KEY`, the empty string, which is the honest spelling
of "none" for a field the base class types as ``str``. The consequence is
mechanical and deliberate:
:meth:`~llmverify.reference.schema.ModelRecord.score_for` matches nothing, the
benchmark probe's plan reports no reference, and the probe either runs this
benchmark as a paired A/B against a configured baseline endpoint or skips it
with a message saying exactly that. An accuracy from this module compared
against anything other than a measured baseline is not evidence of anything.

**Difficulty has to be tuned, and this repository has not measured it.** An item
set that every model solves carries no information: a benchmark only discriminates
where models actually differ. The generators are therefore parameterised by
``difficulty`` (1-5, default 3), which controls chain length, operand magnitude,
modulus size and how many hops a substitution problem takes. The default is
demanding *by construction* -- multi-step exact integer arithmetic where a
single carry error fails the item -- rather than calibrated against a measured
frontier score, because no such measurement was made here. Anyone who wants a
calibrated setting should run difficulty 2 through 5 against a known-good
endpoint and pick the level that lands near 60-80%.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ..types import Message
from .base import Benchmark, BenchmarkItem, Variant, normalise_text, register_benchmark
from .paraphrase import paraphrase

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = ["NO_REFERENCE_KEY", "Synthetic", "generate"]

#: The reference key for a benchmark that has no published scores. An empty
#: string matches no ``BenchmarkScore.benchmark`` in any snapshot, so every
#: reference lookup returns ``None`` without any caller needing a special case.
NO_REFERENCE_KEY: Final[str] = ""

#: Items generated when the caller sets no limit.
DEFAULT_ITEMS = 200

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5

_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"


@dataclass(frozen=True, slots=True)
class _Generated:
    """One generated question and its exact answer."""

    question: str
    answer: str
    kind: str


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #


def _arithmetic_chain(rng: random.Random, difficulty: int) -> _Generated:
    """A mixed +, -, * expression evaluated under the usual precedence.

    Written in ordinary infix notation so that "the usual order of operations"
    is unambiguous, and stated in the prompt as well, because an item whose
    answer depends on a convention the reader has to guess is a broken item
    rather than a hard one.
    """
    terms = 2 + difficulty
    magnitude = 10 ** (1 + difficulty)
    numbers = [rng.randrange(magnitude // 10, magnitude) for _ in range(terms)]
    # At most one multiplication: two of them turn a hard arithmetic problem
    # into an unreadable one without making it more discriminating.
    operators = [rng.choice("+-") for _ in range(terms - 1)]
    if difficulty >= 2:
        operators[rng.randrange(len(operators))] = "*"

    expression = str(numbers[0])
    for operator, value in zip(operators, numbers[1:], strict=True):
        expression += f" {operator} {value}"

    total = 0
    term = numbers[0]
    sign = 1
    for operator, value in zip(operators, numbers[1:], strict=True):
        if operator == "*":
            term *= value
            continue
        total += sign * term
        sign = 1 if operator == "+" else -1
        term = value
    total += sign * term

    return _Generated(
        question=(
            "Evaluate this expression exactly, using the usual order of operations "
            f"(multiplication before addition and subtraction):\n\n{expression}"
        ),
        answer=str(total),
        kind="arithmetic",
    )


def _modular(rng: random.Random, difficulty: int) -> _Generated:
    """Modular exponentiation, which resists estimation entirely."""
    moduli = (97, 101, 211, 401, 1009, 3571, 10007, 32003)
    modulus = moduli[min(len(moduli) - 1, difficulty + 1)]
    base = rng.randrange(2, modulus - 1)
    exponent = rng.randrange(3 + difficulty, 8 + 4 * difficulty)
    return _Generated(
        question=(
            f"Compute {base}^{exponent} mod {modulus}. "
            f"Give the result as an integer between 0 and {modulus - 1}."
        ),
        answer=str(pow(base, exponent, modulus)),
        kind="modular",
    )


def _list_operation(rng: random.Random, difficulty: int) -> _Generated:
    """Sorting and set operations, stated in prose rather than as code.

    Values are kept to two digits so that a comma a model inserts as a thousands
    separator can never be confused with a list separator by the grader.
    """
    size = 5 + 2 * difficulty
    left = rng.sample(range(10, 100), size)
    kind = rng.choice(("nth_smallest", "intersection", "difference_sum"))

    if kind == "nth_smallest":
        position = rng.randrange(2, min(size, 3 + difficulty))
        values = ", ".join(str(value) for value in left)
        return _Generated(
            question=(
                f"Consider these numbers: {values}.\n"
                f"Sort them in increasing order and give the {_ordinal(position)} value."
            ),
            answer=str(sorted(left)[position - 1]),
            kind="list_sort",
        )

    right = rng.sample(range(10, 100), size)
    if kind == "intersection":
        shared = sorted(set(left) & set(right))
        answer = " ".join(str(value) for value in shared) if shared else "none"
        return _Generated(
            question=(
                f"List A: {', '.join(str(value) for value in left)}.\n"
                f"List B: {', '.join(str(value) for value in right)}.\n"
                "Which values appear in both lists? Give them in increasing order, separated "
                'by single spaces, or write "none" if there are no such values.'
            ),
            answer=answer,
            kind="list_intersection",
        )

    only_left = sum(value for value in left if value not in set(right))
    return _Generated(
        question=(
            f"List A: {', '.join(str(value) for value in left)}.\n"
            f"List B: {', '.join(str(value) for value in right)}.\n"
            "Add up every value that appears in list A but not in list B, and give the total."
        ),
        answer=str(only_left),
        kind="list_difference",
    )


def _substitution(rng: random.Random, difficulty: int) -> _Generated:
    """A chain of symbolic definitions, each depending on earlier ones.

    Multi-hop rather than multi-digit: the arithmetic at each step is easy and
    the difficulty is entirely in carrying values through the chain, which is a
    different capability from the one the arithmetic generator tests.
    """
    hops = 2 + difficulty
    names = list(string.ascii_lowercase[: hops + 1])
    values: dict[str, int] = {names[0]: rng.randrange(2, 20)}
    lines = [f"Let {names[0]} = {values[names[0]]}."]

    for index in range(1, len(names)):
        name = names[index]
        # Each step depends on the one before it, so the chain to the answer is
        # genuinely ``hops`` deep. A step that picked any earlier variable would
        # often produce a one-hop shortcut to the target and quietly make the
        # item easier than the difficulty setting claims.
        source = names[index - 1]
        operator = rng.choice(("+", "-", "*"))
        # Multiplication only by a literal: multiplying two chained values makes
        # the numbers explode without making the bookkeeping harder.
        if operator != "*" and difficulty >= 3 and index >= 2 and rng.random() < 0.5:
            operand_name = rng.choice(names[: index - 1])
            operand_value = values[operand_name]
            operand_text = operand_name
        else:
            operand_value = rng.randrange(2, 12)
            operand_text = str(operand_value)

        if operator == "+":
            values[name] = values[source] + operand_value
        elif operator == "-":
            values[name] = values[source] - operand_value
        else:
            values[name] = values[source] * operand_value
        lines.append(f"Let {name} = {source} {operator} {operand_text}.")

    target = names[-1]
    return _Generated(
        question="\n".join(lines) + f"\n\nWhat is the value of {target}?",
        answer=str(values[target]),
        kind="substitution",
    )


#: Pairwise-coprime moduli by difficulty. Coprimality is what makes the search
#: range constructible: over a window as wide as the moduli's product, the
#: remainder theorem guarantees exactly one solution.
_MODULI: tuple[tuple[int, ...], ...] = (
    (5, 7),
    (7, 9, 11),
    (9, 11, 13),
    (11, 13, 17),
    (13, 17, 19, 23),
)


def _constraint(rng: random.Random, difficulty: int) -> _Generated:
    """Find the one integer in a range satisfying two or three congruences.

    The range is built to be exactly as wide as the product of the moduli, so
    the answer is unique by construction rather than by lucky draw. Uniqueness
    is then confirmed by search anyway, because an item with two answers is
    ungradable and an item with none is unfair, and neither should ever escape
    into a run on the strength of an argument in a comment.
    """
    pool = _MODULI[min(len(_MODULI) - 1, difficulty - 1)]
    count = 3 if difficulty >= 4 and len(pool) >= 3 else 2
    moduli = rng.sample(pool, count)
    remainders = [rng.randrange(modulus) for modulus in moduli]

    product = 1
    for modulus in moduli:
        product *= modulus
    floor = rng.randrange(10, 40 * difficulty + 30)
    upper = floor + product - 1

    solutions = [
        value
        for value in range(floor, upper + 1)
        if all(
            value % modulus == remainder
            for modulus, remainder in zip(moduli, remainders, strict=True)
        )
    ]
    if len(solutions) != 1:
        # Unreachable while the pools stay pairwise coprime. Falling back rather
        # than raising keeps generation total: a benchmark that cannot produce
        # an item is worse than one that produces a different kind of item.
        return _modular(rng, difficulty)

    conditions = " and ".join(
        f"n leaves remainder {remainder} when divided by {modulus}"
        for modulus, remainder in zip(moduli, remainders, strict=True)
    )
    return _Generated(
        question=(
            f"Find the unique integer n with {floor} <= n <= {upper} such that {conditions}."
        ),
        answer=str(solutions[0]),
        kind="constraint",
    )


def _string_task(rng: random.Random, difficulty: int) -> _Generated:
    """Reverse, count or index into a nonsense word.

    Nonsense rather than real words, because a real word is memorable and a
    model can pattern-match its reversal. Presented without quotation marks so
    that no paraphrase transform has a quote to rewrite.
    """
    length = 6 + 2 * difficulty
    letters = "".join(
        rng.choice(_CONSONANTS if index % 2 == 0 else _VOWELS) for index in range(length)
    )
    kind = rng.choice(("reverse", "count", "nth"))

    if kind == "reverse":
        return _Generated(
            question=f"Write this string backwards, as a single word:\n\n{letters}",
            answer=letters[::-1],
            kind="string_reverse",
        )
    if kind == "count":
        letter = rng.choice(sorted(set(letters)))
        return _Generated(
            question=(
                f"How many times does the letter {letter} occur in this string?\n\n{letters}"
            ),
            answer=str(letters.count(letter)),
            kind="string_count",
        )
    position = rng.randrange(2, length)
    return _Generated(
        question=(
            f"What is the {_ordinal(position)} character of this string, counting from the "
            f"left starting at 1?\n\n{letters}"
        ),
        answer=letters[position - 1],
        kind="string_index",
    )


#: A generator turns a seeded RNG and a difficulty into one question.
_Generator = Callable[[random.Random, int], _Generated]

#: Applied round-robin so that a truncated item list keeps the mix even.
GENERATORS: tuple[tuple[str, _Generator], ...] = (
    ("arithmetic", _arithmetic_chain),
    ("substitution", _substitution),
    ("modular", _modular),
    ("list", _list_operation),
    ("constraint", _constraint),
    ("string", _string_task),
)


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def generate(
    count: int, *, difficulty: int = 3, seed: str = "llmverify.synthetic"
) -> list[BenchmarkItem]:
    """Build ``count`` items at ``difficulty``, deterministically from ``seed``.

    Each item draws from its own generator seeded with the item's index, so
    changing ``count`` extends the set instead of reshuffling it, and two runs
    that ask for different numbers of items still agree on the items they share.
    """
    level = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, difficulty))
    items: list[BenchmarkItem] = []
    for index in range(max(0, count)):
        family, generator = GENERATORS[index % len(GENERATORS)]
        rng = random.Random(f"{seed}:{level}:{family}:{index}")
        generated = generator(rng, level)
        items.append(
            BenchmarkItem(
                id=f"synthetic-d{level}-{index:04d}",
                question=generated.question,
                answer=generated.answer,
                meta={"kind": generated.kind, "family": family, "difficulty": level},
            )
        )
    return items


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


@register_benchmark
class Synthetic(Benchmark):
    """Locally generated items with exact answers and no published score."""

    name: ClassVar[str] = "synthetic"
    #: Deliberately empty. See :data:`NO_REFERENCE_KEY` and the module docstring:
    #: this benchmark is only interpretable against a measured baseline.
    reference_key: ClassVar[str] = NO_REFERENCE_KEY
    hf_dataset: ClassVar[str | None] = None
    hf_config: ClassVar[str | None] = None
    hf_split: ClassVar[str] = ""
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = "n/a (generated locally)"
    discriminative: ClassVar[bool] = True
    #: Zero because no cross-model spread has ever been published for items that
    #: are generated per run. Zero is also the right ordering signal: any
    #: benchmark with a real published spread should be run before this one.
    score_spread: ClassVar[float] = 0.0
    description: ClassVar[str] = (
        "Procedurally generated arithmetic, modular arithmetic, list and set operations, "
        "multi-hop substitution, constraint satisfaction and string manipulation. Seeded, "
        "offline, and impossible to memorise or string-match, so it is the one benchmark "
        "immune to both lookup routing and training contamination. It has no published "
        "reference score, so it is usable only in A/B mode against a baseline endpoint; "
        "outside A/B mode the benchmark probe skips it. Difficulty is tunable and has not "
        "been calibrated against a measured frontier score in this repository."
    )
    answer_instruction: ClassVar[str] = (
        "Give only the final answer on its own last line, formatted exactly as "
        '"ANSWER: <answer>", with no units, no thousands separators and no explanation.'
    )

    def __init__(self, *, difficulty: int = 3, seed: str = "llmverify.synthetic") -> None:
        """Configure generation.

        ``difficulty`` runs from 1 to 5 and is clamped. ``seed`` fixes the item
        set: two runs with the same seed and difficulty ask the same questions,
        which is what makes a candidate and a baseline endpoint comparable.
        """
        self.difficulty = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, difficulty))
        self.seed = seed

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Generate items. ``loader`` is accepted per the contract and unused.

        Taking a loader it will never call is the price of one uniform interface
        across benchmarks; the alternative is a special case in every caller.
        """
        return generate(
            limit if limit is not None else DEFAULT_ITEMS,
            difficulty=self.difficulty,
            seed=self.seed,
        )

    def render(
        self, item: BenchmarkItem, *, variant: Variant, rng: random.Random
    ) -> tuple[tuple[Message, ...], dict[str, Any]]:
        """Render an item, pinning safe-mode paraphrase and skipping shuffling.

        Paraphrasing these items is close to pointless -- an item generated
        seconds ago is not in anyone's lookup table -- but it stays available so
        that a run can use one benchmark across both arms of the evasion
        comparison. Safe mode is pinned for the same reason as on the maths
        benchmarks: a transform that reflows a digit string or rewrites
        punctuation inside a string-manipulation item changes the answer.
        """
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

    def grade(
        self, item: BenchmarkItem, response_text: str, state: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Exact match, after removing separators a model adds to long numbers."""
        extracted = self.extract_answer(response_text)
        if extracted is None:
            return None, None
        return _canonical(extracted) == _canonical(item.answer), extracted


_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _canonical(text: str) -> str:
    """Normalise an answer, treating a thousands separator as no separator.

    Only a comma between a digit and exactly three more digits is deleted, so
    ``12,345`` becomes one number while ``18, 42, 67`` stays three. That
    distinction is why every list this module generates holds two-digit values.
    """
    return normalise_text(_THOUSANDS_RE.sub("", text).replace(",", " "))
