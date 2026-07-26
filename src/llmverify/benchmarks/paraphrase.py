"""Deterministic, meaning-preserving surface rewrites of benchmark questions.

A dishonest provider does not have to be dishonest all the time. The cheapest
evasion available to it is to string-match incoming requests against a table of
known benchmark items and route just those to the genuine model, serving
everything else from something smaller. That defeats any evaluation that sends
published items verbatim, and it is invisible to a run that only measures
absolute accuracy.

Paraphrasing breaks the match. The item still asks the same question and still
has the same gold answer, but it no longer equals -- or nearly equals -- any
string in a lookup table. Running the same items in both forms and comparing
accuracy therefore measures the routing itself: an honest endpoint scores the
same on both arms within sampling noise, while a lookup-routed one drops on the
paraphrased arm. That comparison is what
:class:`~llmverify.probes.evasion` is built on.

**The transforms may not change the answer.** This constraint dominates every
design decision below. A paraphrase that quietly alters a question turns into a
false accusation, which is the one failure mode this whole package exists to
avoid. So the transform set is small, closed, and hand-checked, and anything
that could carry meaning -- LaTeX, code, chemical formulae, quantities -- is
masked out before a single rewrite runs and restored byte-for-byte afterwards.

Digit-to-word substitution ("3" -> "three") is deliberately **not** implemented.
Deciding whether a bare integer is prose or part of an expression requires
parsing the surrounding maths, and every cheap approximation of that test fails
on realistic items. No paraphrase is better than a wrong one.

Guarantees
----------

*Deterministic.* Given the same ``rng`` state and the same input, the output is
identical. Every decision draws from ``rng`` in a fixed order, including the
draws for transforms that are skipped, so a seed selects the same choices
whether or not :data:`SAFE_MODE` is on.

*Almost always different.* ``paraphrase(x) != x`` is enforced: when no body
transform fired, a framing sentence is added unconditionally. Use
:func:`paraphrase_changed` to check. Idempotency is not claimed and not needed;
``paraphrase(paraphrase(x))`` is a legitimate second paraphrase.

*Not merely reframed.* Prepending a sentence changes the payload but leaves the
original question inside it, so a provider matching ``known_item in body``
still recognises the item. The rewrite is retried until the original is no
longer a substring; :func:`defeats_substring_match` reports whether that
succeeded, which it does not for an input with no interior whitespace to work
with.

*Not a semantic distance.* :func:`similarity` is token-level Jaccard, present
so tests and reports can quantify how far a rewrite moved, not to judge whether
meaning survived. Only the closed transform table does that.

A caveat worth stating: the framing prefixes and suffixes are neutral in
content but not in effect. Text such as "Please work through this carefully."
can shift accuracy on its own, independently of any routing, so a
verbatim-versus-paraphrase gap is only evidence when it is large relative to
that. The pools below are kept short and tonally flat for that reason.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

__all__ = [
    "SAFE_MODE",
    "defeats_substring_match",
    "paraphrase",
    "paraphrase_changed",
    "paraphrase_variants",
    "similarity",
]

#: When true, only transforms that provably cannot alter semantics are used.
#: Maths and code benchmarks should leave this on: rewriting a quotation mark
#: or a serial comma is harmless in a physics question and answer-changing in a
#: string-manipulation one. Individual calls can override it per benchmark
#: without mutating this global.
SAFE_MODE = True


# --------------------------------------------------------------------------- #
# Protected regions
# --------------------------------------------------------------------------- #

# Placeholders live in a private-use block so that no transform's regex can see
# them as words, digits or punctuation. Anything already in that block is
# masked first, so restoration cannot confuse the caller's characters with ours.
_PLACEHOLDER_FIRST = 0xF000
_PLACEHOLDER_LAST = 0xF8FF
_PLACEHOLDER_RE = re.compile("[\uf000-\uf8ff]")

_PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    _PLACEHOLDER_RE,
    re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\$[^$\n]+\$"),
    # Chemical formulae and similar subscripted tokens: H2O, C6H12O6, Fe2O3.
    re.compile(r"\b[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*\b"),
    # A quantity with a unit, including percentages and degrees.
    re.compile(
        r"\d+(?:[.,]\d+)*\s?(?:%|\u00b0[CFK]?|[A-Za-z\u00b5\u03bc\u03a9]+(?:/[A-Za-z]+)?)\b"
    ),
    # Any remaining bare number, so no transform can reorder or reshape one.
    re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w.])"),
)


class _MaskOverflow(Exception):
    """More protected regions than the placeholder block can represent."""


class _Masker:
    """Swaps protected spans for single unused characters, and back again."""

    __slots__ = ("_spans",)

    def __init__(self) -> None:
        self._spans: list[str] = []

    def mask(self, text: str) -> str:
        masked = text
        for pattern in _PROTECTED_PATTERNS:
            masked = pattern.sub(self._take, masked)
        return masked

    def _take(self, match: re.Match[str]) -> str:
        index = len(self._spans)
        if _PLACEHOLDER_FIRST + index > _PLACEHOLDER_LAST:
            raise _MaskOverflow
        self._spans.append(match.group(0))
        return chr(_PLACEHOLDER_FIRST + index)

    def restore(self, text: str) -> str:
        """Substitute every placeholder once, in a single pass.

        A single pass matters: a restored span may itself contain a character
        from the placeholder block, and re-scanning the output would replace it
        a second time.
        """

        def expand(match: re.Match[str]) -> str:
            index = ord(match.group(0)) - _PLACEHOLDER_FIRST
            if 0 <= index < len(self._spans):
                return self._spans[index]
            return match.group(0)

        return _PLACEHOLDER_RE.sub(expand, text)


# --------------------------------------------------------------------------- #
# Closed substitution table
# --------------------------------------------------------------------------- #

_FOLLOWING_NOUNS = (
    "question",
    "problem",
    "statement",
    "expression",
    "equation",
    "scenario",
    "passage",
    "reaction",
    "sentence",
    "argument",
    "diagram",
    "compound",
)

_PHRASE_TABLE: dict[str, tuple[str, ...]] = {
    "which of the following is": ("which one of these is", "which of these is"),
    "which of the following are": ("which of these are",),
    "which of the following": ("which of these",),
    "calculate": ("compute", "work out"),
    "determine": ("work out",),
    "compute": ("calculate",),
    **{f"the following {noun}": (f"this {noun}",) for noun in _FOLLOWING_NOUNS},
}

# Longest-first so that "which of the following is" wins over its own prefix.
_PHRASE_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(_PHRASE_TABLE, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Words after which "which" is either fixed by a preposition or starts a new
# relative clause that "that" cannot introduce.
_WHICH_BLOCKERS = frozenset(
    {
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "to",
        "under",
        "upon",
        "with",
        "within",
        "and",
        "or",
        "but",
        "that",
        "which",
    }
)
_WHICH_RE = re.compile(r"\b(\w+)(\s+)which\b")

# Subordinators safe to lowercase and move to the end of their own sentence.
_INTRO_WORDS = frozenset(
    {
        "after",
        "although",
        "assuming",
        "because",
        "before",
        "considering",
        "given",
        "if",
        "provided",
        "since",
        "suppose",
        "unless",
        "when",
        "whereas",
        "while",
    }
)

_PREFIXES = (
    "Consider the following question.",
    "Here is a problem:",
    "Please work through this carefully.",
    "Read the question below and answer it.",
    "Here is a question:",
    "Consider the problem below.",
)

_SUFFIXES = (
    "Please answer carefully.",
    "Answer to the best of your ability.",
    "Give the answer in the requested format.",
)

_TOKEN_RE = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


def _collapse_whitespace(text: str, rng: random.Random) -> str:
    out = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+\n", "\n", out)


def _sentence_spacing(text: str, rng: random.Random) -> str:
    """Toggle between one and two spaces after sentence-final punctuation."""
    if re.search(r"[.!?] {2,}\S", text):
        return re.sub(r"([.!?]) {2,}(?=\S)", r"\1 ", text)
    return re.sub(r"([.!?]) (?=[A-Z\"(\uf000-\uf8ff])", r"\1  ", text)


def _phrase_substitution(text: str, rng: random.Random) -> str:
    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        options = _PHRASE_TABLE[found.lower()]
        choice = options[rng.randrange(len(options))]
        if found[:1].isupper():
            choice = choice[:1].upper() + choice[1:]
        return choice

    return _PHRASE_RE.sub(replace, text)


def _which_to_that(text: str, rng: random.Random) -> str:
    """Rewrite restrictive "which" as "that".

    The pattern requires a word immediately before "which", which is what makes
    this safe: a non-restrictive clause is introduced by ", which", and a comma
    stops the match. Prepositional "in which"/"of which" are excluded by name.
    """

    def replace(match: re.Match[str]) -> str:
        if match.group(1).lower() in _WHICH_BLOCKERS:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}that"

    return _WHICH_RE.sub(replace, text)


def _clause_reorder(text: str, rng: random.Random) -> str:
    """Move a leading subordinate clause to the end of its sentence.

    Applied only to a text that is one sentence with exactly one comma and a
    recognised subordinator at the front. Those conditions are what make the
    move unambiguous; anything looser risks splitting a list or a clause whose
    scope depends on position.
    """
    body = text.strip()
    if body != text or "\n" in body or len(body) < 12:
        return text
    if body.count(",") != 1 or len(re.findall(r"[.!?]", body)) != 1 or body[-1] not in ".!?":
        return text

    head, _, tail = body.partition(",")
    intro = head.strip()
    main = tail.strip()[:-1].strip()
    terminator = body[-1]
    if intro.split()[:1] == [] or intro.split()[0].lower() not in _INTRO_WORDS:
        return text
    if len(intro.split()) < 3 or len(main.split()) < 2 or not main[:1].isalpha():
        return text

    lowered = intro[:1].lower() + intro[1:]
    return f"{main[:1].upper()}{main[1:]} {lowered}{terminator}"


def _oxford_comma(text: str, rng: random.Random) -> str:
    """Add or remove one serial comma, whichever direction the text allows."""
    if re.search(r",\s+and\s", text):
        return re.sub(r",(\s+and\s)", r"\1", text, count=1)
    # The leading comma is the evidence that this really is a list.
    return re.sub(r"(,\s[^,]{1,60}?)(\s+and\s)", r"\1,\2", text, count=1)


def _punctuation_style(text: str, rng: random.Random) -> str:
    """Swap ASCII punctuation for its typographic equivalents, or back."""
    out = re.sub(r"(?<=\w)'(?=\w)", "\u2019", text)
    if out.count('"') % 2 == 0 and '"' in out:
        state = {"open": True}

        def quote(_match: re.Match[str]) -> str:
            char = "\u201c" if state["open"] else "\u201d"
            state["open"] = not state["open"]
            return char

        out = re.sub(r'"', quote, out)
    if "\u2014" in out:
        return out.replace("\u2014", " -- ")
    return re.sub(r"\s+--\s+", "\u2014", out)


_Transform = Callable[[str, random.Random], str]

#: ``(name, function, probability, safe)``, applied in this order. Whitespace
#: collapse runs before sentence spacing so the two do not fight.
_TRANSFORMS: tuple[tuple[str, _Transform, float, bool], ...] = (
    ("whitespace", _collapse_whitespace, 0.5, False),
    ("phrases", _phrase_substitution, 0.9, True),
    ("which_that", _which_to_that, 0.7, False),
    ("clause_reorder", _clause_reorder, 0.6, False),
    ("oxford_comma", _oxford_comma, 0.4, False),
    ("punctuation", _punctuation_style, 0.5, False),
    ("sentence_spacing", _sentence_spacing, 0.5, True),
)


#: Whitespace runs inside the masked body, which are the only safe places to
#: reflow: the masker has already replaced every protected span, so no match
#: here can fall inside LaTeX, a code span or a chemical formula.
_INNER_SPACE_RE = re.compile(r"(?<=\S)[ \t]+(?=\S)")


def _breakable(text: str, index: int) -> bool:
    """Whether a line break at ``index`` cannot disturb a laid-out structure.

    Whitespace is meaningless in prose and load-bearing in a table or an
    aligned listing, where a break in the middle of a row would change what the
    row says. A pipe, a tab or a run of padding spaces is enough evidence of
    layout to leave the whole line alone. Two spaces are not: sentence spacing
    produces those on ordinary prose.
    """
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    line = text[line_start : line_end if line_end != -1 else len(text)]
    return "|" not in line and "\t" not in line and "   " not in line


def _reflow(text: str, rng: random.Random) -> str:
    """Turn one interior space into a newline.

    This exists for one reason: framing alone does not defeat a substring
    detector. Prepending "Consider the following question." leaves the original
    question intact inside the payload, so a provider checking
    ``known_item in request_body`` still recognises it and can route that one
    request to the genuine model. Breaking a single space inside the body
    changes the byte sequence without touching a word, which no model reads
    differently and no substring match survives.
    """
    positions = [m.span() for m in _INNER_SPACE_RE.finditer(text) if _breakable(text, m.start())]
    if not positions:
        return text
    # Bias towards the middle: a break in the first few characters is easier
    # for a detector to normalise away with a prefix-strip heuristic.
    start_index = len(positions) // 4
    candidates = positions[start_index:] or positions
    begin, end = candidates[rng.randrange(len(candidates))]
    return f"{text[:begin]}\n{text[end:]}"


def _frame(text: str, rng: random.Random, *, force: bool) -> str:
    """Add neutral framing. All four draws happen either way, for determinism."""
    want_prefix = rng.random() < 0.6
    want_suffix = rng.random() < 0.35
    prefix = _PREFIXES[rng.randrange(len(_PREFIXES))]
    suffix = _SUFFIXES[rng.randrange(len(_SUFFIXES))]
    if force and not want_prefix and not want_suffix:
        want_prefix = True

    out = text
    if want_prefix:
        out = f"{prefix}\n\n{out}"
    if want_suffix:
        out = f"{out}\n\n{suffix}"
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def paraphrase(text: str, *, rng: random.Random, safe_mode: bool | None = None) -> str:
    """Rewrite ``text`` without changing what it asks.

    ``safe_mode`` overrides :data:`SAFE_MODE` for this call, which is how a
    maths or code benchmark restricts itself to the provably neutral transforms
    without mutating module state that other benchmarks share.
    """
    if not text or not text.strip():
        return text

    safe = SAFE_MODE if safe_mode is None else safe_mode
    masker = _Masker()
    try:
        body = masker.mask(text)
    except _MaskOverflow:
        # Nothing can be rewritten safely, but framing the original still
        # defeats an exact-match lookup, so the item is not wasted.
        return _frame(text, rng, force=True)

    original = body
    for _name, transform, probability, transform_is_safe in _TRANSFORMS:
        selected = rng.random() < probability
        if selected and (transform_is_safe or not safe):
            body = transform(body, rng)

    body = _frame(body, rng, force=body == original)
    result = masker.restore(body)

    # Guarantee the property the caller actually needs. The lexical transforms
    # only fire when the input happens to contain a substitutable phrase, which
    # for a corpus of terse benchmark questions is most of the time not the
    # case, so without this the original survives verbatim inside the payload
    # and the whole verbatim-versus-paraphrased comparison measures nothing.
    for _ in range(4):
        if defeats_substring_match(text, result):
            break
        reflowed = _reflow(body, rng)
        if reflowed == body:
            break
        body = reflowed
        result = masker.restore(body)
    return result


def paraphrase_variants(
    text: str, *, rng: random.Random, n: int = 3, safe_mode: bool | None = None
) -> list[str]:
    """Produce up to ``n`` distinct paraphrases, none equal to ``text``.

    Fewer than ``n`` are returned when the input admits fewer distinct
    rewrites -- a three-word question has little surface to vary.
    """
    if n <= 0:
        return []
    seen: dict[str, None] = {}
    for _ in range(n * 12):
        if len(seen) >= n:
            break
        candidate = paraphrase(text, rng=rng, safe_mode=safe_mode)
        if candidate != text:
            seen.setdefault(candidate, None)
    return list(seen)[:n]


def paraphrase_changed(text: str, rng: random.Random) -> bool:
    """Whether one paraphrase of ``text`` actually differs from it."""
    return paraphrase(text, rng=rng) != text


def defeats_substring_match(original: str, candidate: str) -> bool:
    """Whether ``candidate`` would survive a naive corpus-matching detector.

    The check a dishonest provider can afford to run on every single request is
    substring containment against a table of known benchmark items. A rewrite
    that leaves the original sitting inside the payload buys nothing against
    that, however different it looks to a reader, so :func:`paraphrase`
    enforces this property rather than hoping for it.
    """
    stripped = original.strip()
    return bool(stripped) and stripped not in candidate


def similarity(a: str, b: str) -> float:
    """Token-level Jaccard overlap of two strings, in ``[0, 1]``.

    Case-insensitive and punctuation-blind, so it reports how much wording
    moved rather than how much formatting did. Two empty strings are identical
    by convention; an empty string and a non-empty one share nothing.
    """
    tokens_a = set(_TOKEN_RE.findall(a.lower()))
    tokens_b = set(_TOKEN_RE.findall(b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)
