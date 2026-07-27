"""Paraphrasing, and the one thing it must never do.

A paraphrase that changes what a question asks turns into a false accusation:
the endpoint answers the rewritten question correctly, the grader marks it wrong
against the original's answer key, and the evasion probe reads the gap as
routing. So the tests that matter here are the negative ones -- LaTeX, code
spans, chemical formulae, quantities and bare numbers come back byte-identical --
followed by determinism under a fixed seed and the substring-defeat property the
whole comparison rests on.
"""

from __future__ import annotations

import random
import re

import pytest

from llmverify.benchmarks.paraphrase import (
    defeats_substring_match,
    paraphrase,
    paraphrase_changed,
    paraphrase_variants,
    similarity,
)

#: Regions that carry meaning and must survive a rewrite untouched.
PROTECTED_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Calculate the value of $\\frac{3}{4} + \\frac{1}{8}$ for the following expression.",
        ("$\\frac{3}{4} + \\frac{1}{8}$",),
    ),
    (
        "Which of the following is the output of `sorted(xs, key=len)` on that list?",
        ("`sorted(xs, key=len)`",),
    ),
    (
        "Determine the mass of C6H12O6 needed to react fully with 3 mol of O2.",
        ("C6H12O6", "O2", "3 mol"),
    ),
    (
        "A sample is heated from 20.5 \u00b0C to 91.25 \u00b0C at 3.4 kPa. Compute the change.",
        ("20.5 \u00b0C", "91.25 \u00b0C", "3.4 kPa"),
    ),
    (
        "Compute 123456 + 654321 and report the remainder modulo 1000003.",
        ("123456", "654321", "1000003"),
    ),
    (
        "Evaluate \\[ \\int_0^1 x^2 \\, dx \\] and give the exact value.",
        ("\\[ \\int_0^1 x^2 \\, dx \\]",),
    ),
    (
        "Consider the block below.\n\n```python\ntotal = 0\nfor x in xs:\n    total += x\n```\n"
        "Which of the following describes it?",
        ("```python\ntotal = 0\nfor x in xs:\n    total += x\n```",),
    ),
    (
        "Given $E = mc^2$, determine E when m is 2.5 kg.",
        ("$E = mc^2$", "2.5 kg"),
    ),
)


def rng(seed: int = 7) -> random.Random:
    return random.Random(seed)


# --------------------------------------------------------------------------- #
# Protected regions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("text", "protected"), PROTECTED_CASES)
def test_protected_regions_survive_byte_identically(
    text: str, protected: tuple[str, ...]
) -> None:
    """Over many seeds, so a transform that only sometimes fires is still caught."""
    for seed in range(60):
        result = paraphrase(text, rng=rng(seed))
        for region in protected:
            assert region in result, f"seed {seed} damaged {region!r}: {result!r}"


@pytest.mark.parametrize(("text", "protected"), PROTECTED_CASES)
def test_protected_regions_survive_unsafe_mode_too(
    text: str, protected: tuple[str, ...]
) -> None:
    """Masking runs before any transform, so safe mode is not what protects them."""
    for seed in range(40):
        result = paraphrase(text, rng=rng(seed), safe_mode=False)
        for region in protected:
            assert region in result, f"seed {seed} damaged {region!r}: {result!r}"


def test_every_number_in_the_input_survives() -> None:
    text = "Add 4096 and 65536, then divide by 1024 and subtract 0.5."
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    for seed in range(50):
        result = paraphrase(text, rng=rng(seed), safe_mode=False)
        assert re.findall(r"\d+(?:\.\d+)?", result) == numbers


def test_a_placeholder_character_in_the_input_is_not_confused_with_ours() -> None:
    """The masker's placeholders live in a private-use block, and so might input."""
    text = "Consider the token \uf042 in the following expression: 2 + 2."
    result = paraphrase(text, rng=rng(3))
    assert "\uf042" in result
    assert "2 + 2" in result


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_same_seed_produces_the_same_rewrite() -> None:
    text = "Which of the following is the capital of Australia?"
    for seed in range(20):
        assert paraphrase(text, rng=rng(seed)) == paraphrase(text, rng=rng(seed))


def test_different_seeds_reach_different_rewrites() -> None:
    text = "Which of the following statements about the reaction is correct?"
    seen = {paraphrase(text, rng=rng(seed)) for seed in range(30)}
    assert len(seen) > 1


def test_a_seed_selects_the_same_choices_whether_or_not_safe_mode_is_on() -> None:
    """Every draw happens either way, including the ones a skipped transform makes."""
    text = "Determine which of the following is true."
    safe_first = paraphrase(text, rng=rng(11), safe_mode=True)
    safe_again = paraphrase(text, rng=rng(11), safe_mode=True)
    unsafe = paraphrase(text, rng=rng(11), safe_mode=False)
    assert safe_first == safe_again
    # The safe run must not have consumed a different number of draws from the
    # stream; if it had, the two would diverge in their framing choices as well
    # as in their body transforms.
    assert safe_first.endswith(unsafe[-24:]) or unsafe.endswith(safe_first[-24:])


def test_the_generator_stream_is_advanced_consistently() -> None:
    """Two paraphrases drawn in sequence from one stream are reproducible."""
    stream = rng(5)
    first = [paraphrase("Calculate the third prime.", rng=stream) for _ in range(4)]
    stream = rng(5)
    second = [paraphrase("Calculate the third prime.", rng=stream) for _ in range(4)]
    assert first == second


# --------------------------------------------------------------------------- #
# The property the evasion probe depends on
# --------------------------------------------------------------------------- #


PARAPHRASE_INPUTS: tuple[str, ...] = (
    "Which of the following is the capital of Australia?",
    "Compute 123456 + 654321.",
    "Given that the reaction proceeds to completion, determine the yield.",
    "What is the chemical symbol for tungsten?",
    "Evaluate $\\int_0^1 x^2 dx$ and state the result as a fraction.",
    "If the train leaves at 09:15, when does it arrive?",
)


@pytest.mark.parametrize("text", PARAPHRASE_INPUTS)
def test_a_paraphrase_is_never_the_original(text: str) -> None:
    for seed in range(30):
        assert paraphrase(text, rng=rng(seed)) != text


@pytest.mark.parametrize("text", PARAPHRASE_INPUTS)
def test_a_paraphrase_defeats_a_substring_lookup(text: str) -> None:
    """Framing alone is not enough: the original must not sit inside the payload."""
    for seed in range(30):
        result = paraphrase(text, rng=rng(seed))
        assert defeats_substring_match(text, result), f"seed {seed}: {result!r}"


def test_defeats_substring_match_is_honest_about_failure() -> None:
    assert defeats_substring_match("abc", "xyz") is True
    assert defeats_substring_match("abc", "the abc thing") is False
    assert defeats_substring_match("", "anything") is False


def test_paraphrase_changed_reports_the_common_case() -> None:
    assert paraphrase_changed("Which of the following is true?", rng(2))


def test_empty_and_blank_inputs_pass_through_untouched() -> None:
    assert paraphrase("", rng=rng()) == ""
    assert paraphrase("   \n ", rng=rng()) == "   \n "


# --------------------------------------------------------------------------- #
# Variants and similarity
# --------------------------------------------------------------------------- #


def test_variants_are_distinct_and_never_the_original() -> None:
    text = "Which of the following statements about the passage is correct?"
    variants = paraphrase_variants(text, rng=rng(9), n=3)
    assert 1 <= len(variants) <= 3
    assert len(set(variants)) == len(variants)
    assert text not in variants


def test_zero_variants_requested_returns_nothing() -> None:
    assert paraphrase_variants("anything", rng=rng(), n=0) == []


def test_similarity_is_a_token_overlap_not_a_meaning_measure() -> None:
    assert similarity("", "") == 1.0
    assert similarity("alpha beta", "alpha beta") == 1.0
    assert similarity("alpha beta", "gamma delta") == 0.0
    assert similarity("alpha beta", "ALPHA, BETA!") == 1.0
    assert 0.0 < similarity("alpha beta gamma", "alpha beta delta") < 1.0


#: The words that carry the question. The closed phrase table may rewrite
#: "which of the following" and "compute"; nothing may touch these.
DISTINCTIVE_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (PARAPHRASE_INPUTS[0], ("capital", "Australia")),
    (PARAPHRASE_INPUTS[1], ("123456", "654321")),
    (PARAPHRASE_INPUTS[2], ("reaction", "completion", "yield")),
    (PARAPHRASE_INPUTS[3], ("chemical", "symbol", "tungsten")),
    (PARAPHRASE_INPUTS[4], ("$\\int_0^1 x^2 dx$", "fraction")),
    (PARAPHRASE_INPUTS[5], ("train", "09:15", "arrive")),
)


@pytest.mark.parametrize(("text", "tokens"), DISTINCTIVE_TOKENS)
def test_a_paraphrase_keeps_every_word_that_carries_the_question(
    text: str, tokens: tuple[str, ...]
) -> None:
    """Framing lowers token overlap a lot; losing a content word would be fatal.

    :func:`similarity` is a Jaccard overlap and a neutral prefix can halve it
    without changing a word of the question, so the overlap is not the property
    worth asserting. What must hold is that no transform drops or rewrites the
    words the answer depends on.
    """
    for seed in range(30):
        result = paraphrase(text, rng=rng(seed))
        for token in tokens:
            assert token in result, f"seed {seed} lost {token!r}: {result!r}"
