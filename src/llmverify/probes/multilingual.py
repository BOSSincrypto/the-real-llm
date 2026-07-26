"""Non-Latin script handling, and the corruption that comes with cheap routing.

Two different things are measured here and reported separately, because they
fail for different reasons and mean different things.

**Unicode integrity is the important one.** The endpoint is asked to echo a
fixed string back verbatim -- Han characters, kana, hangul, Arabic and Hebrew
right-to-left runs, decomposed combining marks, an Indic consonant-vowel
cluster, a Thai tone mark, a zero-width-joiner emoji and a regional-indicator
pair -- and the reply is compared character for character. This is a task with
no reasoning in it at all: any model that receives the string intact can return
it intact. Replacement characters, dropped combining marks or mangled CJK in a
verbatim echo are therefore not a statement about the model's ability, they are
a statement about the pipeline the bytes travelled through, and CJK corruption
under low-precision routing is a failure that has been observed in the field
rather than a hypothetical.

Normalisation is treated as its own outcome and carries no weight. A reply that
differs from the original only by Unicode normalisation form has lost nothing;
some stacks normalise as a matter of course, and calling that corruption would
accuse an honest endpoint of something a linter did.

**Task accuracy is the weaker half.** A fixed short sentence is translated into
one Latin-script and three non-Latin-script languages and graded for required
content words, and one factual question with a number or proper-noun answer is
asked in Russian, Chinese and Arabic. Grading is by accepted-answer sets, never
by a judge model. The Latin-script translation is a control: an endpoint that
handles Spanish and fails Chinese, Russian and Arabic has a script problem,
while one that fails all four has a capability problem, and the two point at
different explanations.

Which script the answer comes back in is recorded and never weighed. Answering a
Chinese question in English is a post-training preference, not evidence about
which weights ran.

All non-ASCII text in this module is written as escapes. The strings are
compared byte for byte and a source file full of literal RTL runs, combining
marks and astral-plane codepoints is one editor normalisation away from
measuring something other than what it claims to.
"""

from __future__ import annotations

import difflib
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["ECHO_SEGMENTS", "ECHO_TEXT", "MultilingualProbe"]


@dataclass(frozen=True, slots=True)
class _Segment:
    """One script's worth of the echo string, tracked so failures can be located."""

    key: str
    label: str
    text: str


#: The echo string, segment by segment. Invisible control characters that are
#: meaningful only in isolation -- bidi overrides, standalone zero-width spaces
#: -- are deliberately excluded: a genuine model may drop them for reasons that
#: have nothing to do with the serving stack, and this probe must not
#: manufacture a finding out of that.
ECHO_SEGMENTS: tuple[_Segment, ...] = (
    _Segment("han", "Han characters", "\u6c49\u5b57\u4e0e\u6f22\u5b57"),
    _Segment("kana", "Japanese kana", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"),
    _Segment("hangul", "Korean hangul", "\ud55c\uad6d\uc5b4"),
    _Segment(
        "arabic",
        "Arabic, right-to-left",
        "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    ),
    _Segment(
        "hebrew",
        "Hebrew, right-to-left",
        "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
    ),
    _Segment(
        "combining",
        "Decomposed combining marks",
        "e\u0301a\u0308o\u0323\u0301",
    ),
    _Segment("devanagari", "Devanagari cluster", "\u0939\u093f\u0928\u094d\u0926\u0940"),
    _Segment("thai", "Thai tone mark", "\u0e01\u0e49"),
    _Segment("emoji", "ZWJ emoji and flag", "\U0001f469\u200d\U0001f4bb\U0001f1ef\U0001f1f5"),
)

#: The string sent for verbatim echo. Segments are separated by an ASCII pipe so
#: that a reply can be attributed to individual scripts without depending on how
#: the endpoint treats whitespace.
ECHO_TEXT: str = " | ".join(segment.text for segment in ECHO_SEGMENTS)

ECHO_PROMPT: str = (
    "Repeat the following text back exactly, character for character. Output the "
    "text and nothing else: no commentary, no quotation marks, no code fence.\n\n"
    + ECHO_TEXT
)

#: Unicode replacement character. Its presence in an echo is decoding damage.
REPLACEMENT_CHAR: str = "\ufffd"


@dataclass(frozen=True, slots=True)
class _Task:
    """One graded generation in or into a target language."""

    key: str
    kind: str
    language: str
    script: str
    prompt: str
    #: Groups of accepted strings. Every group must be matched by the reply, and
    #: any one member of a group satisfies it.
    required: tuple[tuple[str, ...], ...]


#: The sentence translated in every translation task. Short, concrete, and made
#: of words whose translations are not in dispute in any of these languages.
SOURCE_SENTENCE: str = "The cat drinks water in the house."

#: Translation targets. Spanish is the Latin-script control; the other three are
#: the scripts a broken tokenizer or a low-precision route damages first.
TRANSLATIONS: tuple[_Task, ...] = (
    _Task(
        "translate_es",
        "translation",
        "Spanish",
        "latin",
        f"Translate this sentence into Spanish. Reply with the translation alone.\n\n"
        f"{SOURCE_SENTENCE}",
        (("gato", "gata"), ("agua",), ("casa",)),
    ),
    _Task(
        "translate_ru",
        "translation",
        "Russian",
        "cyrillic",
        f"Translate this sentence into Russian. Reply with the translation alone.\n\n"
        f"{SOURCE_SENTENCE}",
        (
            ("\u043a\u043e\u0442", "\u043a\u043e\u0448"),
            ("\u0432\u043e\u0434",),
            ("\u0434\u043e\u043c",),
        ),
    ),
    _Task(
        "translate_zh",
        "translation",
        "Chinese",
        "han",
        f"Translate this sentence into Simplified Chinese. Reply with the translation "
        f"alone.\n\n{SOURCE_SENTENCE}",
        (("\u732b",), ("\u6c34",), ("\u623f", "\u5c4b", "\u5bb6")),
    ),
    _Task(
        "translate_ar",
        "translation",
        "Arabic",
        "arabic",
        f"Translate this sentence into Arabic. Reply with the translation alone.\n\n"
        f"{SOURCE_SENTENCE}",
        (
            ("\u0642\u0637",),
            ("\u0645\u0627\u0621",),
            ("\u0645\u0646\u0632\u0644", "\u0628\u064a\u062a"),
        ),
    ),
)

#: Factual questions asked in the target language, each with an answer that is a
#: number or a proper noun and so grades by exact accepted-answer match.
QUESTIONS: tuple[_Task, ...] = (
    _Task(
        "ask_ru",
        "question",
        "Russian",
        "cyrillic",
        "\u041a\u0430\u043a \u043d\u0430\u0437\u044b\u0432\u0430\u0435\u0442\u0441\u044f "
        "\u0441\u0442\u043e\u043b\u0438\u0446\u0430 \u0424\u0440\u0430\u043d\u0446"
        "\u0438\u0438? \u041e\u0442\u0432\u0435\u0442\u044c\u0442\u0435 \u043e\u0434"
        "\u043d\u0438\u043c \u0441\u043b\u043e\u0432\u043e\u043c.",
        (("\u043f\u0430\u0440\u0438\u0436", "paris"),),
    ),
    _Task(
        "ask_zh",
        "question",
        "Chinese",
        "han",
        "\u4e00\u5e74\u6709\u591a\u5c11\u4e2a\u6708\uff1f"
        "\u53ea\u7528\u4e00\u4e2a\u6570\u5b57\u56de\u7b54\u3002",
        (("12", "\u5341\u4e8c"),),
    ),
    _Task(
        "ask_ar",
        "question",
        "Arabic",
        "arabic",
        "\u0643\u0645 \u0639\u062f\u062f \u0623\u064a\u0627\u0645 "
        "\u0627\u0644\u0623\u0633\u0628\u0648\u0639\u061f \u0623\u062c\u0628 "
        "\u0628\u0631\u0642\u0645 \u0641\u0642\u0637.",
        (("7", "\u0667", "\u0633\u0628\u0639\u0629"),),
    ),
)

#: Codepoint ranges used to decide whether a reply came back in the script the
#: question was asked in. Reported only; never weighed.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x024F),),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
}

#: Accuracy at or above this counts as "the endpoint handles these languages".
_ACCURACY_FLOOR: float = 0.9

#: Accuracy this far below the floor earns the full weight the family allows.
_ACCURACY_SATURATION: float = 0.5


@dataclass(slots=True)
class _Result:
    """One task's outcome."""

    key: str
    kind: str
    language: str
    script: str
    correct: bool | None = None
    missing: list[str] = field(default_factory=list)
    script_fraction: float | None = None
    reply: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "language": self.language,
            "script": self.script,
            "correct": self.correct,
            "missing_required": self.missing,
            "reply_script_fraction": self.script_fraction,
            "reply": self.reply,
            "error": self.error,
        }


@register_probe
class MultilingualProbe(Probe):
    """Verbatim Unicode echo, plus translation and factual accuracy in four languages."""

    name: ClassVar[str] = "multilingual"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "multilingual"
    order: ClassVar[int] = 95
    estimated_requests: ClassVar[int] = 1 + len(TRANSLATIONS) + len(QUESTIONS)
    description: ClassVar[str] = (
        "Byte-for-byte Unicode echo across nine scripts, plus translation and factual "
        "questions in Spanish, Russian, Chinese and Arabic."
    )

    max_tokens: ClassVar[int] = 320

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        cost = 0.0
        tokens = 0

        # The echo goes first on purpose: it is the finding worth having, and a
        # run that is cut short for budget should be cut short somewhere else.
        echo_reply, echo_error, spent, used = await self._ask(ctx, ECHO_PROMPT)
        cost += spent
        tokens += used

        results: list[_Result] = []
        truncated = False
        for task in QUESTIONS + TRANSLATIONS:
            try:
                ctx.budget.check()
            except BudgetExhausted:
                truncated = True
                break
            reply, error, spent, used = await self._ask(ctx, task.prompt)
            cost += spent
            tokens += used
            results.append(_grade(task, reply, error))

        elapsed = time.perf_counter() - started
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        evidence = [self._integrity(echo_reply, echo_error, charged)]
        evidence.append(self._accuracy(ctx, results, truncated))
        script_item = self._script_fidelity(results)
        if script_item is not None:
            evidence.append(script_item)
        return evidence

    async def _ask(
        self, ctx: ProbeContext, prompt: str
    ) -> tuple[str | None, str | None, float, int]:
        """One short generation, returning text, error, cost and tokens."""
        response, error = await ctx.adapter.try_chat(
            ChatRequest(
                messages=(Message(Role.USER, prompt),),
                max_tokens=self.max_tokens,
            )
        )
        if response is None:
            return None, redact(str(error))[:220], ctx.budget.charge(None, None), 0
        cost = ctx.budget.charge(response.usage.input_tokens, response.usage.output_tokens)
        return response.text or "", None, cost, response.usage.total_tokens or 0

    # ------------------------------------------------------------- integrity

    def _integrity(
        self, reply: str | None, error: str | None, charged: dict[str, Any]
    ) -> Evidence:
        """Grade the verbatim echo and classify how it failed, if it did."""
        if reply is None:
            return self._ev(
                "unicode_integrity",
                0.0,
                status=EvidenceStatus.ERROR,
                detail=f"the verbatim echo request failed: {error}",
                **charged,
            )

        report = _echo_report(reply)
        data: dict[str, Any] = {
            **report,
            "segments": {segment.key: segment.label for segment in ECHO_SEGMENTS},
            "expected_chars": len(ECHO_TEXT),
        }

        if report["exact"]:
            return self._ev(
                "unicode_integrity",
                WEAK,
                detail=(
                    "the endpoint echoed a nine-script string back character for "
                    "character, including decomposed combining marks, two right-to-left "
                    "runs and a zero-width-joiner emoji. Nothing between the request and "
                    "the response is mangling text."
                ),
                data=data,
                **charged,
            )

        if report["normalisation_only"]:
            return self._ev(
                "unicode_integrity",
                0.0,
                detail=(
                    "the echo differed from the original only by Unicode normalisation "
                    f"form ({report['normal_form']}). No character was lost, so this is a "
                    "property of the serving stack's text handling and carries no weight "
                    "either way."
                ),
                data=data,
                **charged,
            )

        if report["replacement_chars"] or report["damaged_segments"]:
            damaged = ", ".join(report["damaged_segments"]) or "none identified"
            return self._ev(
                "unicode_integrity",
                -MODERATE,
                detail=(
                    "a verbatim echo came back corrupted: "
                    f"{report['replacement_chars']} replacement character(s), and these "
                    f"scripts did not survive intact: {damaged}. Echoing a string requires "
                    "no reasoning, so this is not a statement about the model's ability but "
                    "about the pipeline the bytes went through -- the shape that "
                    "low-precision routing and misconfigured serving stacks produce, and "
                    "that has been observed corrupting CJK in the field."
                ),
                data=data,
                **charged,
            )

        if report["contains_original"]:
            return self._ev(
                "unicode_integrity",
                0.0,
                detail=(
                    "the string came back intact but wrapped in commentary the prompt asked "
                    "the endpoint not to add. Every character survived, so nothing here is "
                    "about text handling; ignoring a formatting instruction is a different "
                    "question and is not weighed as corruption."
                ),
                data=data,
                **charged,
            )

        similarity = report["similarity"]
        return self._ev(
            "unicode_integrity",
            -MODERATE * min(1.0, max(0.0, (0.98 - similarity) / 0.3)),
            detail=(
                f"the echo was {similarity:.1%} similar to the original but not identical, "
                "with no replacement characters and no script lost outright. Some of that "
                "is ordinary formatting drift, so the weight scales with how much of the "
                "string failed to survive rather than treating any difference as damage."
            ),
            data=data,
            **charged,
        )

    # -------------------------------------------------------------- accuracy

    def _accuracy(
        self, ctx: ProbeContext, results: list[_Result], truncated: bool
    ) -> Evidence:
        """Translation and factual accuracy, with the Latin control called out."""
        graded = [r for r in results if r.correct is not None]
        if not graded:
            return self._ev(
                "multilingual_accuracy",
                0.0,
                status=EvidenceStatus.TRUNCATED if truncated else EvidenceStatus.ERROR,
                detail="no translation or factual task produced a gradable reply.",
                data={"results": {r.key: r.as_dict() for r in results}},
            )

        correct = sum(1 for r in graded if r.correct)
        fraction = correct / len(graded)
        non_latin = [r for r in graded if r.script != "latin"]
        latin = [r for r in graded if r.script == "latin"]
        non_latin_correct = sum(1 for r in non_latin if r.correct)
        latin_correct = sum(1 for r in latin if r.correct)

        data: dict[str, Any] = {
            "results": {r.key: r.as_dict() for r in results},
            "graded": len(graded),
            "correct": correct,
            "non_latin_graded": len(non_latin),
            "non_latin_correct": non_latin_correct,
            "latin_graded": len(latin),
            "latin_correct": latin_correct,
        }
        status = EvidenceStatus.TRUNCATED if truncated and len(graded) < 3 else EvidenceStatus.OK

        if fraction >= _ACCURACY_FLOOR:
            return self._ev(
                "multilingual_accuracy",
                0.5 * WEAK,
                status=status,
                detail=(
                    f"{correct} of {len(graded)} translation and factual tasks were correct "
                    "across Spanish, Russian, Chinese and Arabic. Expected of any current "
                    "model, so mildly supportive at most."
                ),
                data=data,
            )

        isolated = bool(latin) and latin_correct == len(latin) and non_latin_correct == 0
        detail = (
            f"only {correct} of {len(graded)} translation and factual tasks were correct."
        )
        if isolated:
            detail += (
                " The Latin-script control passed while every non-Latin task failed, which "
                "isolates the failure to script handling rather than to the underlying "
                "language ability -- the profile of a tokenizer or serving path that does "
                "not handle these scripts, not of a model that cannot translate."
            )
        else:
            detail += (
                " The Latin-script control failed too, so this is a general capability "
                "shortfall rather than a script-handling one."
            )
        fraction_short = min(
            1.0, (_ACCURACY_FLOOR - fraction) / _ACCURACY_SATURATION
        )
        return self._ev(
            "multilingual_accuracy",
            -MODERATE * fraction_short,
            status=status,
            detail=detail,
            data={**data, "script_isolated_failure": isolated},
        )

    def _script_fidelity(self, results: list[_Result]) -> Evidence | None:
        """Whether replies came back in the script they were asked in. Never weighed."""
        measured = [r for r in results if r.script_fraction is not None]
        if not measured:
            return None
        summary = ", ".join(
            f"{r.language} {r.script_fraction or 0.0:.0%}" for r in measured
        )
        return self._ev(
            "reply_script_fidelity",
            0.0,
            detail=(
                "share of each reply's letters written in the script the task used: "
                f"{summary}. Recorded only -- answering a Chinese question in English is a "
                "post-training preference, not a fact about which weights ran."
            ),
            data={r.key: r.script_fraction for r in measured},
        )

    # ----------------------------------------------------------------- helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        status: EvidenceStatus = EvidenceStatus.OK,
        detail: str = "",
        data: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        tokens: int = 0,
        duration_s: float = 0.0,
    ) -> Evidence:
        return Evidence(
            probe=self.name,
            label=label,
            llr=llr,
            cap=MODERATE,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def _echo_report(reply: str) -> dict[str, Any]:
    """Compare an echo against :data:`ECHO_TEXT` and describe the difference."""
    stripped = _unwrap(reply)
    exact = stripped == ECHO_TEXT
    normal_form = ""
    normalisation_only = False
    if not exact:
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            if unicodedata.normalize(form, stripped) == unicodedata.normalize(form, ECHO_TEXT):
                normalisation_only = True
                normal_form = form
                break

    damaged = [
        segment.key
        for segment in ECHO_SEGMENTS
        if segment.text not in stripped
        and unicodedata.normalize("NFC", segment.text)
        not in unicodedata.normalize("NFC", stripped)
    ]
    return {
        "exact": exact,
        # The whole string present inside a longer reply means nothing was lost;
        # the endpoint added commentary, which is a different failure entirely.
        "contains_original": not exact and ECHO_TEXT in stripped,
        "normalisation_only": normalisation_only and not exact,
        "normal_form": normal_form,
        "replacement_chars": stripped.count(REPLACEMENT_CHAR),
        "damaged_segments": damaged,
        "similarity": round(difflib.SequenceMatcher(None, stripped, ECHO_TEXT).ratio(), 4),
        "reply_chars": len(stripped),
        "reply": redact(stripped)[:400],
        "unwrapped": stripped != reply.strip(),
    }


def _unwrap(reply: str) -> str:
    """Strip a code fence or surrounding quotes a model may have added.

    A model that fences its echo has still echoed it, and grading the fence as
    corruption would confuse a formatting habit with a broken pipeline.
    """
    text = (reply or "").strip()
    if text.startswith("```"):
        body = text.split("\n", 1)[-1] if "\n" in text else ""
        text = body.rstrip()
        if text.endswith("```"):
            text = text[: -len("```")].rstrip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.strip()


def _grade(task: _Task, reply: str | None, error: str | None) -> _Result:
    """Grade one translation or factual answer by accepted-answer sets."""
    result = _Result(
        key=task.key, kind=task.kind, language=task.language, script=task.script, error=error
    )
    if reply is None:
        return result

    folded = reply.casefold()
    missing = [
        group[0]
        for group in task.required
        if not any(option.casefold() in folded for option in group)
    ]
    result.correct = not missing
    result.missing = missing
    result.script_fraction = _script_fraction(reply, task.script)
    result.reply = redact(reply)[:200]
    return result


def _script_fraction(text: str, script: str) -> float | None:
    """Share of a reply's letters that belong to ``script``."""
    ranges = SCRIPT_RANGES.get(script)
    if not ranges:
        return None
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    inside = sum(
        1 for ch in letters if any(low <= ord(ch) <= high for low, high in ranges)
    )
    return round(inside / len(letters), 4)
