"""IFEval: verifiable instruction following, graded programmatically.

This benchmark is unusual and valuable here for one reason: it has no answer
key and needs no judge. Every item carries machine-checkable constraints --
"at least three bullet points", "no commas", "wrap your title in double angular
brackets", "your entire response in Kannada" -- and grading is a function of the
response text alone. Nothing about the verdict depends on a second model's
opinion, which matters in a package whose whole subject is whether a model can
be trusted.

It also fails in a different way from a knowledge benchmark, which is what makes
it a complement rather than a duplicate. Instruction following degrades sharply
under aggressive quantization while factual recall degrades gently, so an
endpoint that still answers questions acceptably but has stopped counting its
own bullet points is showing a signature that SimpleQA would miss.

**The split is called ``train`` and is not one.** ``google/IFEval`` publishes
its 541 items under a single split named ``train``. It is the evaluation set --
there is no other -- and every published IFEval number is measured on it.

**Strict and loose.** The benchmark defines two accuracies. Strict applies the
verifiers to the response exactly as returned. Loose re-checks a small family of
cosmetic rewrites -- markdown asterisks removed, the first line dropped, the
last line dropped, and the combinations of those -- and counts the instruction
as followed if any of them passes. Loose exists because models prefix answers
with "Sure, here is..." and wrap things in bold, neither of which is a failure
to follow the instruction that was given. Both numbers are computed and both
are reported; the graded outcome is prompt-level *loose*, because a strict
grader biases this tool towards accusing an honest endpoint and that is the one
error it must not make. The per-item detail records all four figures.

**These verifiers are a reimplementation.** Google's reference implementation
depends on ``nltk`` for sentence and word tokenisation and on ``langdetect`` for
the language check, and this package will not take either as a dependency. The
families below are rebuilt from the published instruction ids and their kwargs,
in pure Python. Where the reference convention was not certain, the more lenient
reading was chosen, for the same reason as above. Two places are approximations
worth naming: sentence splitting is regex-based with an abbreviation guard, and
language identification uses Unicode script coverage where the script settles
the question and a function-word vote where it does not. Neither will agree with
``nltk`` and ``langdetect`` on every response.

**Instruction families this module will not grade, it skips.** An item whose
instruction id or kwargs it does not fully support never enters the pool, so no
response is ever graded against a constraint that was checked wrongly.
``combination:repeat_prompt`` is implemented but excluded by default, and the
reason is specific to this package rather than to the benchmark: the text the
model must repeat is the prompt itself, which the paraphrased arm of the
anti-evasion comparison deliberately changes. Keeping those items would produce
a systematic verbatim-versus-paraphrase gap that is an artefact of this tool and
would read as evidence of evasion. Pass ``include_repeat_prompt=True`` to
restore them for a run that does not paraphrase.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import DatasetError
from ..types import Message
from .base import Benchmark, BenchmarkItem, Variant, register_benchmark
from .datasets import DatasetSpec
from .paraphrase import paraphrase

if TYPE_CHECKING:
    from .datasets import DatasetLoader

__all__ = [
    "EXCLUDED_BY_DEFAULT",
    "SUPPORTED_INSTRUCTIONS",
    "IFEval",
    "IFEvalGrade",
    "detect_language",
    "grade_response",
    "split_sentences",
    "supports",
]

SPEC = DatasetSpec(
    dataset="google/IFEval",
    config="default",
    #: Named ``train``, but it is the evaluation set and the only split there is.
    split="train",
    gated=False,
    licence="apache-2.0",
)

#: Relations the benchmark uses for every counting constraint.
LESS_THAN = "less than"
AT_LEAST = "at least"

#: Answers accepted by ``detectable_format:constrained_response``.
CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _relation_holds(count: int, relation: Any, threshold: Any) -> bool:
    """Apply the benchmark's two counting relations."""
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        return False
    if relation == LESS_THAN:
        return count < threshold
    if relation == AT_LEAST:
        return count >= threshold
    return False


# --------------------------------------------------------------------------- #
# Text measurement
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\w+")

_ABBREVIATIONS = (
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "no",
    "vol",
    "fig",
    "approx",
    "e.g",
    "i.e",
)
#: Stand-in for a full stop that does not end a sentence. A private-use
#: character, so nothing in a real response can collide with it and be
#: turned back into a stray dot by the restore.
_DOT = "\uf8ff"


def _count_words(text: str) -> int:
    """Words, counted the way the reference harness's tokeniser counts them."""
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split into sentences, guarding the full stops that do not end one.

    An approximation of the reference implementation's ``nltk`` sentence
    tokeniser: abbreviations, decimal points and single-letter initials are
    masked, the remainder is split on terminal punctuation, and the mask is
    undone. It will disagree with ``nltk`` on unusual text, which is one reason
    the sentence-count family is a small part of the benchmark rather than all
    of it.
    """
    body = text.strip()
    if not body:
        return []

    for abbreviation in _ABBREVIATIONS:
        body = re.sub(
            rf"(?<![\w.]){re.escape(abbreviation)}\.",
            abbreviation + _DOT,
            body,
            flags=re.IGNORECASE,
        )
    body = re.sub(r"(?<=\d)\.(?=\d)", _DOT, body)
    body = re.sub(r"(?<![\w.])([A-Za-z])\.(?=\s)", rf"\1{_DOT}", body)
    body = body.replace("...", _DOT * 3)

    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+", body)
    return [part.replace(_DOT, ".").strip() for part in parts if part.strip()]


# --------------------------------------------------------------------------- #
# Language identification
# --------------------------------------------------------------------------- #

#: Unicode ranges per script. Where several supported languages share a script
#: -- Hindi, Marathi and Nepali all use Devanagari; Arabic, Persian and Urdu all
#: use the Arabic script -- the check verifies the script and stops there. That
#: is lenient by construction: it can credit a response a stricter grader would
#: fail, never the reverse, which is the safe direction for a tool that must not
#: manufacture accusations.
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)),
    "armenian": ((0x0530, 0x058F),),
    "bengali": ((0x0980, 0x09FF),),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "devanagari": ((0x0900, 0x097F),),
    "georgian": ((0x10A0, 0x10FF),),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "gujarati": ((0x0A80, 0x0AFF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "hangul": ((0x1100, 0x11FF), (0xAC00, 0xD7AF), (0x3130, 0x318F)),
    "hebrew": ((0x0590, 0x05FF),),
    "kana": ((0x3040, 0x309F), (0x30A0, 0x30FF)),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "sinhala": ((0x0D80, 0x0DFF),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "thai": ((0x0E00, 0x0E7F),),
}

_LANGUAGE_SCRIPT: dict[str, str] = {
    "ar": "arabic",
    "fa": "arabic",
    "ur": "arabic",
    "ps": "arabic",
    "bn": "bengali",
    "bg": "cyrillic",
    "ru": "cyrillic",
    "uk": "cyrillic",
    "sr": "cyrillic",
    "mk": "cyrillic",
    "be": "cyrillic",
    "hi": "devanagari",
    "mr": "devanagari",
    "ne": "devanagari",
    "sa": "devanagari",
    "el": "greek",
    "gu": "gujarati",
    "pa": "gurmukhi",
    "he": "hebrew",
    "hy": "armenian",
    "ka": "georgian",
    "kn": "kannada",
    "ml": "malayalam",
    "si": "sinhala",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "ko": "hangul",
    "ja": "kana",
    "zh": "han",
}

def _words(text: str) -> frozenset[str]:
    """Read a function-word list written as one space-separated string."""
    return frozenset(text.split())


#: Function words for the Latin-script languages IFEval asks for. Single-letter
#: words are left out on purpose: "a", "e", "o" and "y" appear in several of
#: these languages and in English, and including them makes the vote noise.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": _words("the and of to in is that for with you it as are this be on not have from"),
    "de": _words(
        "der die das und ist nicht ein eine den dem mit von sich auch fur auf zu sie es im "
        "werden oder aber"
    ),
    "es": _words(
        "el la los las de del que en un una por para con no se lo mas como pero muy este son"
    ),
    "fr": _words(
        "le la les de des du et un une est que qui pour dans avec vous ne pas sur ce sont "
        "plus mais"
    ),
    "it": _words(
        "il lo la gli le di del che un una per con non in sono si come piu anche questo "
        "della nel"
    ),
    "pt": _words(
        "os as de do da dos das que em um uma para com nao se por mais como mas ser esta sao"
    ),
    "nl": _words("de het een en van is dat niet op te zijn met voor aan ook er maar om deze"),
    "vi": _words("va cua la co khong duoc cac nhung nguoi cho trong voi mot de nay khi cung nhu"),
    "fi": _words("ja on ei etta se ovat tai kuin mutta myos han ne tama voi kun niin sen"),
    "sw": _words("na ya kwa ni wa katika kwamba hii hizo sana kama lakini pia yake hao wake zao"),
}

#: Share of a response's letters that must belong to the target script.
#: Responses in Hindi routinely carry a few English words, so this is well below
#: one; it is high enough that an English response with a stray Devanagari
#: character does not pass.
SCRIPT_COVERAGE = 0.30

#: Function-word hits the target language needs before the vote means anything.
MIN_FUNCTION_WORDS = 2

#: How close to the winning language's hit count the target may sit and still
#: pass. Romance languages share function words, so demanding an outright win
#: would fail genuinely Portuguese responses on Spanish overlap.
FUNCTION_WORD_MARGIN = 0.7


def _strip_marks(text: str) -> str:
    """Fold diacritics away so the function-word lists can be written in ASCII."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def detect_language(text: str, language: str) -> bool:
    """Whether ``text`` is predominantly written in ``language``.

    Script coverage decides it whenever the script can: a response in Thai, Han,
    Hangul or Devanagari is unmistakable. For Latin-script languages a
    function-word vote stands in, since no dependency-free script test can tell
    Portuguese from Italian.
    """
    script = _LANGUAGE_SCRIPT.get(language)
    if script is not None:
        return _script_coverage(text, script)
    return _function_word_vote(text, language)


def _script_coverage(text: str, script: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ranges = _SCRIPT_RANGES[script]
    hits = sum(1 for char in letters if any(lo <= ord(char) <= hi for lo, hi in ranges))
    if script == "han":
        # Japanese is written with Han characters too; kana is what separates
        # the two, so a response carrying kana is not Chinese.
        kana = _SCRIPT_RANGES["kana"]
        if any(any(lo <= ord(char) <= hi for lo, hi in kana) for char in letters):
            return False
    return hits / len(letters) >= SCRIPT_COVERAGE


def _function_word_vote(text: str, language: str) -> bool:
    target = _FUNCTION_WORDS.get(language)
    if target is None:
        return False
    tokens = _WORD_RE.findall(_strip_marks(text).casefold())
    if not tokens:
        return False
    counts = {
        name: sum(1 for token in tokens if token in words)
        for name, words in _FUNCTION_WORDS.items()
    }
    hits = counts[language]
    best = max(counts.values())
    return hits >= MIN_FUNCTION_WORDS and hits >= best * FUNCTION_WORD_MARGIN


# --------------------------------------------------------------------------- #
# Verifiers
# --------------------------------------------------------------------------- #


def _keywords_existence(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    lowered = response.casefold()
    return all(str(word).casefold() in lowered for word in kw["keywords"])


def _keywords_frequency(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    keyword = str(kw["keyword"]).casefold()
    if not keyword:
        return False
    return _relation_holds(response.casefold().count(keyword), kw.get("relation"), kw["frequency"])


def _forbidden_words(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    # Word boundaries rather than substrings: a response forbidden the word
    # "art" should not fail for containing "start".
    return not any(
        re.search(rf"\b{re.escape(str(word))}\b", response, flags=re.IGNORECASE)
        for word in kw["forbidden_words"]
        if str(word)
    )


def _letter_frequency(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    letter = str(kw["letter"]).casefold()[:1]
    if not letter:
        return False
    count = response.casefold().count(letter)
    return _relation_holds(count, kw.get("let_relation"), kw["let_frequency"])


def _response_language(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return detect_language(response, str(kw["language"]).strip().casefold())


def _number_sentences(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    count = len(split_sentences(response))
    return _relation_holds(count, kw.get("relation"), kw["num_sentences"])


def _number_words(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return _relation_holds(_count_words(response), kw.get("relation"), kw["num_words"])


def _number_paragraphs(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    """Paragraphs separated by the markdown divider ``***``.

    An empty leading or trailing segment is ignored -- a divider at the very top
    or bottom of the response is a formatting quirk, not a paragraph -- while an
    empty segment in the middle means two dividers in a row, which is a genuine
    failure to produce the paragraph that belonged between them.
    """
    segments = re.split(r"\s?\*\*\*\s?", response)
    count = len(segments)
    for index, segment in enumerate(segments):
        if segment.strip():
            continue
        if index in (0, len(segments) - 1):
            count -= 1
        else:
            return False
    return count == kw["num_paragraphs"]


def _nth_paragraph_first_word(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    paragraphs = [block for block in re.split(r"\n\n", response) if block.strip()]
    nth = kw["nth_paragraph"]
    if len(paragraphs) != kw["num_paragraphs"] or not 1 <= nth <= len(paragraphs):
        return False
    words = paragraphs[nth - 1].strip().split()
    if not words:
        return False
    first = words[0].strip().strip(".,?!'\"").casefold()
    return first == str(kw["first_word"]).strip().casefold()


def _number_placeholders(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return len(re.findall(r"\[.*?\]", response, flags=re.DOTALL)) >= kw["num_placeholders"]


def _postscript(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    marker = str(kw["postscript_marker"]).strip()
    # "P.S." is written with and without its full stops and with a space after
    # the P, so the marker is matched with optional separators rather than
    # literally.
    pattern = r"\s*".join(re.escape(char) for char in marker if not char.isspace())
    return bool(re.search(pattern, response, flags=re.IGNORECASE)) if pattern else False


def _number_bullets(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    starred = re.findall(r"^\s*\*[^*].*$", response, flags=re.MULTILINE)
    dashed = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(starred) + len(dashed) == kw["num_bullets"]


def _constrained_response(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    body = response.strip()
    return any(option in body for option in CONSTRAINED_RESPONSES)


def _highlighted_sections(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    single = [text for text in re.findall(r"\*[^\n*]*\*", response) if text.strip("*").strip()]
    double = [text for text in re.findall(r"\*\*[^\n*]*\*\*", response) if text.strip("*").strip()]
    return len(single) + len(double) >= kw["num_highlights"]


def _multiple_sections(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    spliter = str(kw["section_spliter"]).strip()
    if not spliter:
        return False
    pattern = rf"\s?{re.escape(spliter)}\s?\d+\s?"
    return len(re.split(pattern, response)) - 1 >= kw["num_sections"]


def _json_format(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    body = response.strip()
    for fence in ("```json", "```Json", "```JSON", "```"):
        body = body.removeprefix(fence)
    body = body.removesuffix("```").strip()
    try:
        json.loads(body)
    except (ValueError, TypeError):
        return False
    return True


def _title(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return any(match.strip("<>").strip() for match in re.findall(r"<<[^\n]+>>", response))


def _two_responses(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    parts = response.split("******")
    filled = []
    for index, part in enumerate(parts):
        if part.strip():
            filled.append(part.strip())
        elif index not in (0, len(parts) - 1):
            return False
    return len(filled) == 2 and filled[0] != filled[1]


def _repeat_prompt(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    """The response must begin by repeating the request.

    Both the dataset's ``prompt_to_repeat`` and the text actually sent are
    accepted, because the two differ whenever a variant rewrote the prompt. See
    the module docstring for why these items are excluded by default anyway.
    """
    body = response.strip().casefold()
    required = str(kw["prompt_to_repeat"]).strip().casefold()
    if required and body.startswith(required):
        return True
    rendered = prompt.strip().casefold()
    return bool(rendered) and body.startswith(rendered)


def _end_checker(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    phrase = str(kw["end_phrase"]).strip().casefold()
    return response.strip().strip('"').casefold().endswith(phrase)


def _capital_word_frequency(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    capitals = sum(1 for word in _WORD_RE.findall(response) if word.isupper())
    return _relation_holds(capitals, kw.get("capital_relation"), kw["capital_frequency"])


def _english_capital(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return response.isupper()


def _english_lowercase(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return response.islower()


def _no_comma(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return "," not in response


def _quotation(response: str, kw: Mapping[str, Any], prompt: str) -> bool:
    body = response.strip()
    return len(body) >= 2 and body.startswith('"') and body.endswith('"')


#: A verifier reads the response, the instruction's kwargs and the prompt that
#: was actually sent, and answers whether the instruction was followed.
_Verifier = Callable[[str, Mapping[str, Any], str], bool]

#: ``instruction id -> (verifier, required kwargs)``. An item is only offered to
#: an endpoint when every one of its instruction ids appears here *and* the
#: required kwargs are present, so nothing is ever graded against a constraint
#: this module only half understands.
_VERIFIERS: dict[str, tuple[_Verifier, tuple[str, ...]]] = {
    "keywords:existence": (_keywords_existence, ("keywords",)),
    "keywords:frequency": (_keywords_frequency, ("keyword", "frequency", "relation")),
    "keywords:forbidden_words": (_forbidden_words, ("forbidden_words",)),
    "keywords:letter_frequency": (
        _letter_frequency,
        ("letter", "let_frequency", "let_relation"),
    ),
    "language:response_language": (_response_language, ("language",)),
    "length_constraints:number_sentences": (_number_sentences, ("num_sentences", "relation")),
    "length_constraints:number_words": (_number_words, ("num_words", "relation")),
    "length_constraints:number_paragraphs": (_number_paragraphs, ("num_paragraphs",)),
    "length_constraints:nth_paragraph_first_word": (
        _nth_paragraph_first_word,
        ("num_paragraphs", "nth_paragraph", "first_word"),
    ),
    "detectable_content:number_placeholders": (_number_placeholders, ("num_placeholders",)),
    "detectable_content:postscript": (_postscript, ("postscript_marker",)),
    "detectable_format:number_bullet_lists": (_number_bullets, ("num_bullets",)),
    "detectable_format:constrained_response": (_constrained_response, ()),
    "detectable_format:number_highlighted_sections": (_highlighted_sections, ("num_highlights",)),
    "detectable_format:multiple_sections": (
        _multiple_sections,
        ("section_spliter", "num_sections"),
    ),
    "detectable_format:json_format": (_json_format, ()),
    "detectable_format:title": (_title, ()),
    "combination:two_responses": (_two_responses, ()),
    "combination:repeat_prompt": (_repeat_prompt, ("prompt_to_repeat",)),
    "startend:end_checker": (_end_checker, ("end_phrase",)),
    "startend:quotation": (_quotation, ()),
    "change_case:capital_word_frequency": (
        _capital_word_frequency,
        ("capital_frequency", "capital_relation"),
    ),
    "change_case:english_capital": (_english_capital, ()),
    "change_case:english_lowercase": (_english_lowercase, ()),
    "punctuation:no_comma": (_no_comma, ()),
}

#: Every instruction family this module can grade.
SUPPORTED_INSTRUCTIONS: frozenset[str] = frozenset(_VERIFIERS)

#: Implemented, but kept out of the pool unless the caller asks for it.
EXCLUDED_BY_DEFAULT: frozenset[str] = frozenset({"combination:repeat_prompt"})


def supports(instruction_id: str, kwargs: Mapping[str, Any]) -> bool:
    """Whether this module can grade one instruction with the kwargs given."""
    entry = _VERIFIERS.get(instruction_id)
    if entry is None:
        return False
    _verifier, required = entry
    if any(kwargs.get(name) is None for name in required):
        return False
    if instruction_id == "language:response_language":
        code = str(kwargs.get("language") or "").strip().casefold()
        return code in _LANGUAGE_SCRIPT or code in _FUNCTION_WORDS
    return True


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IFEvalGrade:
    """Both accuracies the benchmark defines, for one response."""

    prompt_strict: bool
    prompt_loose: bool
    instructions_strict: int
    instructions_loose: int
    total: int

    @property
    def summary(self) -> str:
        return (
            f"strict={int(self.prompt_strict)} loose={int(self.prompt_loose)} "
            f"instructions={self.instructions_loose}/{self.total}"
        )


def _loose_variants(response: str) -> tuple[str, ...]:
    """The response plus the cosmetic rewrites loose accuracy allows.

    Dropping the first or last line forgives "Sure, here is..." preambles and
    trailing offers of further help; removing asterisks forgives markdown
    emphasis. Nothing here can turn a response that ignored an instruction into
    one that followed it.
    """
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    base = (response, without_first, without_last, without_both)
    return base + tuple(variant.replace("*", "") for variant in base)


def grade_response(
    response: str, instructions: Sequence[Mapping[str, Any]], *, prompt: str = ""
) -> IFEvalGrade:
    """Check one response against its item's instructions.

    ``instructions`` is a sequence of ``{"id": ..., "kwargs": {...}}`` mappings,
    which is the shape :attr:`IFEval` stores in an item's ``meta``.
    """
    variants = _loose_variants(response)
    # One row per instruction, one column per rewrite. The first column is the
    # response exactly as returned, which is what strict accuracy reads.
    checks = [
        (_VERIFIERS[str(instruction["id"])][0], instruction.get("kwargs") or {})
        for instruction in instructions
    ]
    matrix = [
        [bool(verifier(text, kwargs, prompt)) for text in variants] for verifier, kwargs in checks
    ]
    if not matrix:
        return IFEvalGrade(False, False, 0, 0, 0)

    strict = [row[0] for row in matrix]
    return IFEvalGrade(
        prompt_strict=all(strict),
        prompt_loose=any(all(row[column] for row in matrix) for column in range(len(variants))),
        instructions_strict=sum(strict),
        instructions_loose=sum(any(row) for row in matrix),
        total=len(matrix),
    )


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


@register_benchmark
class IFEval(Benchmark):
    """Verifiable instruction following, graded without an answer key."""

    name: ClassVar[str] = "ifeval"
    reference_key: ClassVar[str] = "ifeval"
    hf_dataset: ClassVar[str | None] = SPEC.dataset
    hf_config: ClassVar[str | None] = SPEC.config
    hf_split: ClassVar[str] = SPEC.split
    gated: ClassVar[bool] = False
    licence: ClassVar[str] = SPEC.licence
    discriminative: ClassVar[bool] = True
    #: An estimate, not a measured statistic. The reference digest carries no
    #: set of comparable 2026 IFEval scores -- Anthropic, OpenAI and Google have
    #: all dropped it from flagship announcements, and the figures that remain
    #: are for IFBench or are single unreplicated numbers. Marked here so the
    #: runner can order benchmarks, and flagged as unverified wherever it is
    #: reported.
    score_spread: ClassVar[float] = 8.0
    description: ClassVar[str] = (
        "Verifiable instruction following, graded programmatically with no answer key and no "
        "judge: bullet counts, forbidden words, response language, casing, JSON-only output and "
        "so on. Complements knowledge benchmarks because instruction following degrades sharply "
        "under quantization while recall degrades gently. Reports prompt-level and "
        "instruction-level accuracy in both strict and loose form; the graded outcome is "
        "prompt-level loose. Verifiers are a dependency-free reimplementation, not Google's own "
        "code, and its score_spread is an estimate rather than a measured statistic."
    )

    def __init__(self, *, include_repeat_prompt: bool = False) -> None:
        """Configure which instruction families may enter the pool.

        ``include_repeat_prompt`` restores ``combination:repeat_prompt`` items,
        which are excluded by default because the text they require the model to
        echo is the prompt itself -- and the anti-evasion comparison
        deliberately rewrites the prompt.
        """
        self.include_repeat_prompt = include_repeat_prompt

    @property
    def _excluded(self) -> frozenset[str]:
        return frozenset() if self.include_repeat_prompt else EXCLUDED_BY_DEFAULT

    async def load(
        self, loader: DatasetLoader, *, limit: int | None = None
    ) -> list[BenchmarkItem]:
        """Read items whose every instruction this module can verify."""
        rows = await loader.rows(SPEC, limit=limit)

        items: list[BenchmarkItem] = []
        skipped: dict[str, int] = {}
        for index, row in enumerate(rows):
            prompt = row.get("prompt")
            ids = row.get("instruction_id_list")
            raw_kwargs = row.get("kwargs")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if not isinstance(ids, list) or not ids:
                continue
            if not isinstance(raw_kwargs, list) or len(raw_kwargs) != len(ids):
                # The two lists are positionally paired; a length mismatch means
                # the kwargs cannot be attributed to an instruction at all.
                continue

            instructions = []
            for identifier, kwargs in zip(ids, raw_kwargs, strict=True):
                name = str(identifier)
                # The published rows carry every kwarg key on every row, with
                # null in the ones that do not apply to that instruction.
                cleaned = {
                    key: value
                    for key, value in (kwargs if isinstance(kwargs, dict) else {}).items()
                    if value is not None
                }
                if name in self._excluded or not supports(name, cleaned):
                    skipped[name] = skipped.get(name, 0) + 1
                    instructions = []
                    break
                instructions.append({"id": name, "kwargs": cleaned})

            if not instructions:
                continue

            key = row.get("key")
            items.append(
                BenchmarkItem(
                    id=f"ifeval-{key if key is not None else index}",
                    question=prompt,
                    # There is no answer key: the constraints in ``meta`` are the
                    # whole of what "correct" means for this benchmark.
                    answer="",
                    meta={
                        "instructions": instructions,
                        "instruction_ids": [entry["id"] for entry in instructions],
                        "row_index": index,
                    },
                )
            )

        if not items:
            keys = ", ".join(sorted(rows[0])) if rows else "none"
            raise DatasetError(
                f"{SPEC} returned {len(rows)} row(s) but none consisted entirely of instruction "
                f"families this grader supports (skipped: {skipped or 'none'}). "
                f"Columns present: {keys}."
            )
        return items

    def render(
        self, item: BenchmarkItem, *, variant: Variant, rng: random.Random
    ) -> tuple[tuple[Message, ...], dict[str, Any]]:
        """Send the prompt as-is, with no answer-format instruction appended.

        Appending "put your answer on its own last line as ANSWER: ..." would
        break the item: half of these prompts constrain casing, punctuation or
        output format, and the appended sentence would violate the very
        constraint being measured.

        Paraphrasing pins safe mode. The unsafe transforms rewrite quotation
        marks and serial commas, and an IFEval prompt frequently quotes a string
        the model is required to reproduce exactly -- an end phrase, a title, a
        keyword -- so a rewritten quotation mark turns into a graded failure
        that has nothing to do with the endpoint.
        """
        if variant is Variant.SHUFFLED:
            messages, state = super().render(item, variant=Variant.VERBATIM, rng=rng)
            state["shuffle_applicable"] = False
        elif variant is Variant.PARAPHRASED:
            rewritten = paraphrase(item.question, rng=rng, safe_mode=True)
            messages, state = super().render(
                replace(item, question=rewritten), variant=Variant.VERBATIM, rng=rng
            )
            state["variant"] = Variant.PARAPHRASED.value
            state["paraphrase_safe_mode"] = True
        else:
            messages, state = super().render(item, variant=variant, rng=rng)

        # The repeat-prompt verifier needs the text that was actually sent,
        # which is not recoverable from the item once a variant has rewritten it.
        state["rendered_prompt"] = messages[0].text if messages else item.question
        return messages, state

    def _default_instruction(self, item: BenchmarkItem) -> str:
        return ""

    def grade(
        self, item: BenchmarkItem, response_text: str, state: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Run every verifier and return prompt-level loose accuracy.

        The returned extraction string carries all four figures the benchmark
        defines, so the per-item record in the report shows strict alongside
        loose rather than only the graded one.
        """
        instructions = item.meta.get("instructions") or []
        if not instructions or not (response_text or "").strip():
            return None, None
        try:
            grade = grade_response(response_text, instructions, prompt=_prompt_of(state, item))
        except (KeyError, TypeError, ValueError, re.error):
            # A verifier that cannot run has not observed a failure to follow
            # instructions, and recording one would be a fabrication.
            return None, None
        return grade.prompt_loose, grade.summary


def _prompt_of(state: Mapping[str, Any], item: BenchmarkItem) -> str:
    rendered = state.get("rendered_prompt")
    return rendered if isinstance(rendered, str) else item.question
