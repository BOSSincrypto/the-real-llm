"""Rendering, answer extraction and grading, across the three variants.

None of these tests loads a dataset: ``load`` is the one method that touches the
network, and the behaviour worth pinning is what happens to an item once it
exists. Items are built by hand or generated locally, which also means the
grading tests can use answer formats taken from what models actually emit -- a
boxed fraction, a bolded letter, a preamble before the answer line -- rather than
from what the prompt asked for.

The shuffled path gets its own attention. Under ``SHUFFLED`` the letters move
but the gold answer is stored as option *text*, so a grader that compared
letters would silently mark a correct answer wrong on every permuted item.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from conftest import MockArithmetic, generate_items
from llmverify.benchmarks.aime import AIME2026, extract_integer
from llmverify.benchmarks.base import (
    Benchmark,
    BenchmarkItem,
    Variant,
    all_benchmarks,
    get_benchmark,
    normalise_text,
)
from llmverify.benchmarks.ifeval import (
    EXCLUDED_BY_DEFAULT,
    SUPPORTED_INSTRUCTIONS,
    IFEval,
    grade_response,
    split_sentences,
    supports,
)
from llmverify.benchmarks.mmlu_pro import MMLUPro
from llmverify.benchmarks.simpleqa import SimpleQAVerified, answer_variants, same_answer
from llmverify.benchmarks.synthetic import Synthetic, generate
from llmverify.errors import ConfigError

CHOICES = ("mercury", "venus", "earth", "mars", "jupiter")

MC_ITEM = BenchmarkItem(
    id="mc-1",
    question="Which planet has the shortest day?",
    answer="jupiter",
    choices=CHOICES,
)

FREE_ITEM = BenchmarkItem(
    id="free-1",
    question="What is the chemical symbol for tungsten?",
    answer="W",
)


def rng(seed: int = 3) -> random.Random:
    return random.Random(seed)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_the_registry_exposes_every_bundled_benchmark() -> None:
    registry = all_benchmarks()
    for name in ("mmlu_pro", "gpqa_diamond", "aime_2026", "simpleqa_verified", "ifeval",
                 "synthetic"):
        assert name in registry, name
    assert isinstance(get_benchmark("synthetic"), Synthetic)


def test_an_unknown_benchmark_lists_the_ones_that_exist() -> None:
    with pytest.raises(ConfigError, match="synthetic"):
        get_benchmark("not-a-benchmark")


def test_saturated_benchmarks_declare_themselves_as_such() -> None:
    """GPQA Diamond and MMLU-Pro cannot separate current frontier models."""
    registry = all_benchmarks()
    assert registry["gpqa_diamond"].discriminative is False
    assert registry["mmlu_pro"].discriminative is False
    assert registry["simpleqa_verified"].discriminative is True
    assert registry["gpqa_diamond"].score_spread < registry["simpleqa_verified"].score_spread


def test_gated_datasets_are_declared(mock_provider: Any = None) -> None:
    registry = all_benchmarks()
    assert registry["gpqa_diamond"].gated is True
    assert registry["mmlu_pro"].gated is False


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_verbatim_rendering_letters_the_options_in_order() -> None:
    benchmark = MMLUPro()
    messages, state = benchmark.render(MC_ITEM, variant=Variant.VERBATIM, rng=rng())
    body = messages[0].text
    assert MC_ITEM.question in body
    assert "A. mercury" in body and "E. jupiter" in body
    assert state["options"] == list(CHOICES)
    assert state["variant"] == "verbatim"
    assert "ANSWER: X" in body


def test_paraphrased_rendering_changes_the_wording_and_nothing_else() -> None:
    benchmark = MMLUPro()
    messages, state = benchmark.render(MC_ITEM, variant=Variant.PARAPHRASED, rng=rng(5))
    body = messages[0].text
    assert state["variant"] == "paraphrased"
    assert MC_ITEM.question not in body
    # The options are not paraphrased: they are the answer key.
    for option in CHOICES:
        assert f"{option}" in body
    assert state["options"] == list(CHOICES)


def test_shuffled_rendering_permutes_the_options_and_records_the_order() -> None:
    benchmark = MMLUPro()
    orders = set()
    for seed in range(20):
        _messages, state = benchmark.render(MC_ITEM, variant=Variant.SHUFFLED, rng=rng(seed))
        orders.add(tuple(state["options"]))
        assert sorted(state["options"]) == sorted(CHOICES)
    assert len(orders) > 1


def test_rendering_is_deterministic_for_a_given_seed() -> None:
    benchmark = MMLUPro()
    for variant in Variant:
        first = benchmark.render(MC_ITEM, variant=variant, rng=rng(11))
        second = benchmark.render(MC_ITEM, variant=variant, rng=rng(11))
        assert first[0][0].text == second[0][0].text
        assert first[1] == second[1]


def test_a_free_response_item_gets_the_free_response_instruction() -> None:
    benchmark = SimpleQAVerified()
    messages, state = benchmark.render(FREE_ITEM, variant=Variant.VERBATIM, rng=rng())
    assert "options" not in state
    assert "ANSWER:" in messages[0].text


def test_maths_benchmarks_pin_safe_mode_and_skip_shuffling() -> None:
    """A transform that reflows a digit string changes the answer."""
    item = BenchmarkItem(id="a", question="Find $n$ such that 3n = 42.", answer="14")
    benchmark = AIME2026()

    _messages, shuffled = benchmark.render(item, variant=Variant.SHUFFLED, rng=rng())
    assert shuffled["shuffle_applicable"] is False
    assert shuffled["variant"] == "verbatim"

    messages, paraphrased = benchmark.render(item, variant=Variant.PARAPHRASED, rng=rng(2))
    assert paraphrased["paraphrase_safe_mode"] is True
    assert paraphrased["variant"] == "paraphrased"
    assert "$n$" in messages[0].text
    assert "42" in messages[0].text


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is jupiter.\n\nANSWER: E", "E"),
        ("answer: e", "e"),
        ("ANSWER - E", "E"),
        ("**ANSWER: E**", "E"),
        ("ANSWER: `E`", "E"),
        ('ANSWER: "E".', "E"),
        ("Reasoning...\nANSWER: B\nANSWER: E", "E"),
        ("Long reasoning about planets.\n\nE", "E"),
        ("Long reasoning about planets.\n\n(E)", "E"),
        ("no answer at all", None),
    ],
)
def test_extraction_across_realistic_response_formats(text: str, expected: str | None) -> None:
    assert MMLUPro().extract_answer(text) == expected


def test_extraction_prefers_the_last_answer_line() -> None:
    """Models restate an answer after correcting themselves; the last one wins."""
    assert MMLUPro().extract_answer("ANSWER: A\nwait, no.\nANSWER: C") == "C"


def test_aime_extraction_order_prefers_the_requested_format() -> None:
    benchmark = AIME2026()
    assert benchmark.extract_answer("I get \\boxed{12} at first.\n\nANSWER: 204") == "204"
    assert benchmark.extract_answer("so the value is \\boxed{204}.") == "204"
    assert benchmark.extract_answer("...\n204") == "204"
    assert benchmark.extract_answer("...\n\\boxed{\\frac{1}{2}}") is None


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("204", 204),
        (" 204. ", 204),
        ("$204$", 204),
        ("1,204", 1204),
        ("\\text{204}", 204),
        ("\\boxed{204}", None),
        ("two hundred", None),
        (None, None),
    ],
)
def test_extract_integer_tolerates_wrappers_and_refuses_prose(
    fragment: str | None, expected: int | None
) -> None:
    assert extract_integer(fragment) == expected


def test_normalise_text_folds_case_punctuation_and_curly_quotes() -> None:
    # Apostrophes are punctuation and are folded away with the rest of it; the
    # curly-to-straight step still matters, because it runs first and makes the
    # two spellings identical before either is stripped.
    assert normalise_text("  The  Answer's Here!  ") == "the answer s here"
    curly = "The Answer’s Here"  # noqa: RUF001 - folding this is the point
    assert normalise_text(curly) == normalise_text("The Answer's Here")
    assert normalise_text("A/B-C") == "a/b-c"


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def test_grading_a_verbatim_multiple_choice_item() -> None:
    benchmark = MMLUPro()
    _messages, state = benchmark.render(MC_ITEM, variant=Variant.VERBATIM, rng=rng())
    assert benchmark.grade(MC_ITEM, "ANSWER: E", state) == (True, "E")
    assert benchmark.grade(MC_ITEM, "ANSWER: A", state) == (False, "A")


def test_grading_follows_the_shuffled_option_order() -> None:
    """The letter that is right depends on the permutation, so the state decides."""
    benchmark = MMLUPro()
    for seed in range(15):
        _messages, state = benchmark.render(MC_ITEM, variant=Variant.SHUFFLED, rng=rng(seed))
        options: list[str] = state["options"]
        correct_letter = "ABCDE"[options.index(MC_ITEM.answer)]
        assert benchmark.grade(MC_ITEM, f"ANSWER: {correct_letter}", state) == (
            True,
            correct_letter,
        )
        wrong_letter = "ABCDE"[(options.index(MC_ITEM.answer) + 1) % len(options)]
        correct, _extracted = benchmark.grade(MC_ITEM, f"ANSWER: {wrong_letter}", state)
        assert correct is False


def test_answering_with_the_option_text_instead_of_the_letter_is_accepted() -> None:
    benchmark = MMLUPro()
    _messages, state = benchmark.render(MC_ITEM, variant=Variant.VERBATIM, rng=rng())
    correct, extracted = benchmark.grade(MC_ITEM, "ANSWER: jupiter", state)
    assert correct is True
    assert extracted == "jupiter"


def test_a_letter_outside_the_option_range_is_not_silently_accepted() -> None:
    benchmark = MMLUPro()
    _messages, state = benchmark.render(MC_ITEM, variant=Variant.VERBATIM, rng=rng())
    correct, _extracted = benchmark.grade(MC_ITEM, "ANSWER: Z", state)
    assert correct is False


def test_an_unextractable_answer_is_not_a_wrong_answer() -> None:
    """The distinction the accuracy test depends on: excluded, never counted wrong."""
    benchmark = MMLUPro()
    _messages, state = benchmark.render(MC_ITEM, variant=Variant.VERBATIM, rng=rng())
    assert benchmark.grade(MC_ITEM, "I would rather not say.", state) == (None, None)


def test_simpleqa_grades_through_its_alias_table() -> None:
    item = BenchmarkItem(id="s", question="When?", answer="3 March 1999")
    benchmark = SimpleQAVerified()
    assert benchmark.grade(item, "ANSWER: 1999-03-03", {})[0] is True
    assert benchmark.grade(item, "ANSWER: March 3, 1999", {})[0] is True
    assert benchmark.grade(item, "ANSWER: 4 March 1999", {})[0] is False


def test_simpleqa_answer_variants_only_ever_merge_spellings() -> None:
    assert same_answer("J.R.R. Tolkien", "JRR Tolkien")
    assert same_answer("the Indian Ocean", "Indian Ocean")
    assert same_answer("3rd", "3")
    assert same_answer("three", "3")
    assert not same_answer("Tolkien", "Lewis")
    assert answer_variants("") == frozenset()


def test_synthetic_items_are_reproducible_and_exactly_gradable() -> None:
    first = generate(12, difficulty=3, seed="fixed")
    second = generate(12, difficulty=3, seed="fixed")
    assert [item.question for item in first] == [item.question for item in second]
    # Extending the set must not reshuffle the items already in it.
    assert generate(20, difficulty=3, seed="fixed")[:12] == first

    benchmark = Synthetic()
    for item in first:
        assert benchmark.grade(item, f"ANSWER: {item.answer}", {})[0] is True


def test_synthetic_grading_ignores_a_thousands_separator() -> None:
    item = BenchmarkItem(id="s", question="Add them.", answer="1234567")
    benchmark = Synthetic()
    assert benchmark.grade(item, "ANSWER: 1,234,567", {})[0] is True
    assert benchmark.grade(item, "ANSWER: 1234568", {})[0] is False


def test_synthetic_has_no_reference_key_by_design() -> None:
    """It is only interpretable against a measured baseline, never a published score."""
    assert Synthetic.reference_key == ""
    assert Synthetic.hf_dataset is None


def test_the_offline_fixture_benchmark_grades_what_the_mock_answers() -> None:
    benchmark = MockArithmetic()
    for item in generate_items(5):
        assert benchmark.grade(item, f"ANSWER: {item.answer}", {})[0] is True
        assert benchmark.grade(item, f"ANSWER: {int(item.answer) + 7}", {})[0] is False


# --------------------------------------------------------------------------- #
# IFEval verifiers
# --------------------------------------------------------------------------- #


def ifeval_item(*instructions: dict[str, Any], prompt: str = "Write something.") -> BenchmarkItem:
    return BenchmarkItem(
        id="if-1",
        question=prompt,
        answer="",
        meta={
            "instructions": list(instructions),
            "instruction_ids": [entry["id"] for entry in instructions],
        },
    )


@pytest.mark.parametrize(
    ("instruction", "passing", "failing"),
    [
        (
            {"id": "punctuation:no_comma", "kwargs": {}},
            "no commas at all here",
            "here, there are commas",
        ),
        (
            {"id": "change_case:english_lowercase", "kwargs": {}},
            "all lower case text",
            "Not All Lower Case",
        ),
        (
            {"id": "change_case:english_capital", "kwargs": {}},
            "ALL UPPER CASE TEXT",
            "not all upper case",
        ),
        (
            {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 2}},
            "* one\n* two",
            "* one\n* two\n* three",
        ),
        (
            {"id": "detectable_format:title", "kwargs": {}},
            "<<A Title>>\n\nand the body",
            "A Title\n\nand the body",
        ),
        (
            {"id": "detectable_format:json_format", "kwargs": {}},
            '```json\n{"a": 1}\n```',
            "this is not json",
        ),
        (
            {"id": "startend:end_checker", "kwargs": {"end_phrase": "Any other questions?"}},
            "Here it is. Any other questions?",
            "Here it is. Goodbye.",
        ),
        (
            {"id": "startend:quotation", "kwargs": {}},
            '"the whole answer is quoted"',
            "the whole answer is not quoted",
        ),
        (
            {"id": "keywords:existence", "kwargs": {"keywords": ["harbor", "lantern"]}},
            "the harbor lantern was lit",
            "the dock light was lit",
        ),
        (
            {"id": "keywords:forbidden_words", "kwargs": {"forbidden_words": ["kernel"]}},
            "nothing prohibited here",
            "the kernel panicked",
        ),
        (
            {
                "id": "length_constraints:number_words",
                "kwargs": {"num_words": 5, "relation": "at least"},
            },
            "one two three four five six",
            "one two three",
        ),
        (
            {"id": "detectable_content:postscript", "kwargs": {"postscript_marker": "P.S."}},
            "the answer\n\nP.S. one more thing",
            "the answer with no postscript",
        ),
        (
            {"id": "detectable_content:number_placeholders", "kwargs": {"num_placeholders": 2}},
            "send it to [address] on [date]",
            "send it to [address]",
        ),
        (
            {"id": "combination:two_responses", "kwargs": {}},
            "first answer\n******\nsecond answer",
            "only one answer",
        ),
        (
            {"id": "detectable_format:constrained_response", "kwargs": {}},
            "My answer is yes.",
            "Probably.",
        ),
    ],
)
def test_ifeval_verifiers_accept_and_reject(
    instruction: dict[str, Any], passing: str, failing: str
) -> None:
    assert grade_response(passing, [instruction]).prompt_strict is True
    assert grade_response(failing, [instruction]).prompt_strict is False


def test_ifeval_loose_forgives_a_preamble_and_markdown_emphasis() -> None:
    """"Sure, here is..." is not a failure to follow the instruction that was given."""
    instruction = {"id": "punctuation:no_comma", "kwargs": {}}
    response = "Sure, here it is:\nno commas below this line"
    grade = grade_response(response, [instruction])
    assert grade.prompt_strict is False
    assert grade.prompt_loose is True

    emphasised = {"id": "change_case:english_lowercase", "kwargs": {}}
    grade = grade_response("*all lower case*", [emphasised])
    assert grade.prompt_loose is True


def test_ifeval_requires_every_instruction_to_hold() -> None:
    instructions = [
        {"id": "punctuation:no_comma", "kwargs": {}},
        {"id": "change_case:english_lowercase", "kwargs": {}},
    ]
    assert grade_response("all lower and no comma", instructions).prompt_strict is True
    grade = grade_response("All Lower And No Comma", instructions)
    assert grade.prompt_strict is False
    assert grade.instructions_strict == 1
    assert grade.total == 2


def test_ifeval_grades_nothing_when_it_has_nothing_to_grade() -> None:
    grade = grade_response("anything", [])
    assert grade.total == 0
    assert grade.prompt_strict is False
    assert IFEval().grade(ifeval_item(), "", {}) == (None, None)


def test_ifeval_only_offers_items_it_can_fully_grade() -> None:
    assert supports("punctuation:no_comma", {}) is True
    assert supports("keywords:frequency", {"keyword": "a"}) is False
    assert supports("keywords:frequency", {"keyword": "a", "frequency": 2, "relation": "at least"})
    assert supports("an:instruction_family_nobody_implemented", {}) is False
    assert supports("language:response_language", {"language": "kn"}) is True
    assert supports("language:response_language", {"language": "zz"}) is False
    assert "combination:repeat_prompt" in SUPPORTED_INSTRUCTIONS
    assert "combination:repeat_prompt" in EXCLUDED_BY_DEFAULT


def test_ifeval_grade_returns_prompt_level_loose() -> None:
    item = ifeval_item({"id": "punctuation:no_comma", "kwargs": {}})
    correct, summary = IFEval().grade(item, "Sure, here it is:\nno commas here", {})
    assert correct is True
    assert "strict=0" in summary and "loose=1" in summary


def test_ifeval_appends_no_answer_instruction() -> None:
    """Half these prompts constrain the output format the instruction would break."""
    item = ifeval_item(
        {"id": "punctuation:no_comma", "kwargs": {}}, prompt="Write a haiku with no commas."
    )
    messages, state = IFEval().render(item, variant=Variant.VERBATIM, rng=rng())
    assert messages[0].text == "Write a haiku with no commas."
    assert state["rendered_prompt"] == messages[0].text


def test_ifeval_paraphrase_records_the_prompt_that_was_actually_sent() -> None:
    item = ifeval_item({"id": "punctuation:no_comma", "kwargs": {}})
    messages, state = IFEval().render(item, variant=Variant.PARAPHRASED, rng=rng(4))
    assert state["rendered_prompt"] == messages[0].text
    assert state["paraphrase_safe_mode"] is True


def test_split_sentences_guards_against_abbreviations() -> None:
    assert split_sentences("Dr. Smith arrived. He left.") == [
        "Dr. Smith arrived.",
        "He left.",
    ]
    assert split_sentences("") == []


def test_a_benchmark_declares_its_dataset_and_licence() -> None:
    for cls in all_benchmarks().values():
        assert isinstance(cls.reference_key, str)
        assert cls.licence
        assert issubclass(cls, Benchmark)
