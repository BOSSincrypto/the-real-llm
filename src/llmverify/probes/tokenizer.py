"""Measure the endpoint's tokenizer without ever seeing its vocabulary.

A tokenizer is a fingerprint of a model generation. Two endpoints running the
same weights agree on the token count of a fixed string to the token; two
endpoints running different generations of the same vendor's models do not, and
the disagreement is large enough to read off a single measurement. Nothing here
needs the vocabulary file, a local tokenizer library, or any cooperation from
the provider beyond the token counts it already reports.

The probe string is :data:`CANONICAL_TEXT`, assembled once from
:data:`SEGMENTS` and never to be changed. Changing it would silently invalidate
every ``canonical_text_tokens`` value in the reference snapshot, and a stale
reference in a tool that accuses people of fraud is worse than no reference. It
is written in the source as ASCII escapes for the same reason: a file full of
literal Cyrillic, Arabic and astral-plane codepoints is one editor
normalisation, one re-encoding or one well-meaning linter away from a different
string, and this particular string's whole value is that it does not move.

**Two measurement paths.**

*Token counting.* Anthropic and Gemini expose an endpoint that runs the real
tokenizer and returns a count. That path is free, has its own rate limit, and
removes generation from the measurement entirely, so the number is a pure
function of the prompt.

*Usage differencing.* An OpenAI-compatible endpoint exposes no such route, only
``usage.input_tokens`` on a completion. That number is not the prompt's token
count: it also contains whatever fixed envelope the stack wraps a request in --
chat template, a proxy's injected system prompt, role scaffolding. Differencing
is what makes this path viable at all. Two measurements whose prompts differ
only in the text of interest cancel the envelope exactly, provided the envelope
is fixed, and it is fixed for every stack that does not vary its system prompt
per request. :class:`TokenMeter` calibrates the envelope by measuring one unit
of text and then two copies of it: the second measurement minus the first is
the unit's own token count, and the first minus that is the envelope.

**Per-script ratios, not one total.** Tokenizer families differ most on
non-Latin scripts, where the choice of merges decides whether a Chinese
character costs one token or three and whether an emoji ZWJ sequence survives
as a unit or shatters into bytes. A single total blends all of that into one
number; the per-segment profile keeps the scripts apart and is far more
discriminative because of it. No reference profile exists to compare it
against, so it is reported at zero LLR -- an honest measurement a human can
read, not a guess dressed as evidence.

**The within-vendor generation check.** Claude 4.7 and later, including Fable
and Mythos, produce roughly 30% more tokens for identical text than 4.6 and
earlier. That is a property of the tokenizer, not of the sampling, so it shows
up on a single measurement of a fixed string. It gives this probe its one piece
of real weight: an endpoint claiming a post-4.7 model -- Opus 5, Sonnet 5,
Fable 5 -- whose canonical count sits near the claimed value divided by 1.30 is
running a pre-4.7 tokenizer, and no amount of prompt engineering changes that.
The check only fires within one vendor. Across vendors a 30% difference in
token count means nothing at all: tokenizers differ that much routinely, and
reading a cross-vendor ratio as a generation signal would be an accusation
built on a coincidence.

Where the reference snapshot records no ``canonical_text_tokens`` for the
claimed model, the measured profile is reported with zero LLR rather than
compared against a number nobody has verified.
"""

from __future__ import annotations

import inspect
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted, ProviderError, UnsupportedCapability
from ..evidence import MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..reference.schema import ModelRecord
from ..types import ChatRequest, Message, Role, ToolSpec
from . import Probe, ProbeContext, register_probe

__all__ = ["CANONICAL_TEXT", "SEGMENTS", "TokenMeter", "TokenizerProbe"]


@dataclass(frozen=True, slots=True)
class _Segment:
    """One script's worth of probe text, measured on its own."""

    key: str
    label: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


#: The segments of the canonical probe string, in fixed order. Every one of them
#: is here because tokenizer families disagree about it: Latin prose sets the
#: baseline, Cyrillic and Arabic separate byte-level fallback from real merges,
#: CJK separates one-token-per-character vocabularies from three-byte fallback,
#: the emoji run carries zero-width joiners, variation selectors, skin-tone
#: modifiers, a keycap and regional-indicator pairs, the code segment carries
#: indentation and punctuation runs, the digit runs separate per-digit
#: tokenizers from ones that merge two or three digits at a time, and the last
#: segment reaches into blocks -- Linear B, Gothic, Ogham, Deseret, Tifinagh,
#: cuneiform, hieroglyphs, Phoenician, Coptic, Canadian syllabics, runes, Old
#: Persian, Vai, Yi -- that almost no vocabulary covers, so they fall back to
#: bytes at a rate that differs sharply between families.
SEGMENTS: tuple[_Segment, ...] = (
    _Segment(
        "latin",
        "Latin prose",
        "The lighthouse keeper logged dense fog, a heavy swell and a freighter's horn"
        " at 04:12; the lamp burned for nineteen hours.",
    ),
    _Segment(
        "cyrillic",
        "Cyrillic",
        "\u0421\u043c\u043e\u0442\u0440\u0438\u0442\u0435\u043b\u044c \u043c\u0430"
        "\u044f\u043a\u0430 \u0437\u0430\u043f\u0438\u0441\u0430\u043b \u0433\u0443"
        "\u0441\u0442\u043e\u0439 \u0442\u0443\u043c\u0430\u043d, \u0441\u0438\u043b"
        "\u044c\u043d\u0443\u044e \u0437\u044b\u0431\u044c \u0438 \u0433\u0443\u0434"
        "\u043e\u043a \u0441\u0443\u0445\u043e\u0433\u0440\u0443\u0437\u0430; \u043b"
        "\u0430\u043c\u043f\u0430 \u0433\u043e\u0440\u0435\u043b\u0430 \u0434\u0435"
        "\u0432\u044f\u0442\u043d\u0430\u0434\u0446\u0430\u0442\u044c \u0447\u0430"
        "\u0441\u043e\u0432.",
    ),
    _Segment(
        "cjk",
        "Chinese",
        "\u706f\u5854\u770b\u5b88\u4eba\u8bb0\u5f55\u4e86\u6d53\u96fe\u3001\u5de8"
        "\u6d6a\u548c\u8d27\u8f6e\u7684\u6c7d\u7b1b\u58f0\uff0c\u90a3\u76cf\u706f"
        "\u8fde\u7eed\u71c3\u70e7\u4e86\u5341\u4e5d\u4e2a\u5c0f\u65f6\u3002",
    ),
    _Segment(
        "arabic",
        "Arabic",
        "\u0633\u062c\u0644 \u062d\u0627\u0631\u0633 \u0627\u0644\u0645\u0646\u0627"
        "\u0631\u0629 \u0627\u0644\u0636\u0628\u0627\u0628 \u0627\u0644\u0643\u062b"
        "\u064a\u0641 \u0648\u0627\u0644\u0645\u0648\u062c \u0627\u0644\u0639\u0627"
        "\u0644\u064a \u0648\u0628\u0648\u0642 \u0633\u0641\u064a\u0646\u0629 \u0627"
        "\u0644\u0634\u062d\u0646\u060c \u0648\u0638\u0644 \u0627\u0644\u0645\u0635"
        "\u0628\u0627\u062d \u0645\u0636\u0627\u0621 \u062a\u0633\u0639 \u0639\u0634"
        "\u0631\u0629 \u0633\u0627\u0639\u0629.",
    ),
    _Segment(
        "emoji",
        "Emoji with ZWJ sequences",
        "\U0001f469\u200d\U0001f4bb \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d"
        "\U0001f466 \U0001f9d1\u200d\U0001f680 \U0001f3f3\ufe0f\u200d\U0001f308 "
        "\U0001f44d\U0001f3fd \U0001f1ef\U0001f1f5 \U0001f1e6\U0001f1ea #\ufe0f\u20e3"
        " \U0001faf1\U0001f3fc\u200d\U0001faf2\U0001f3ff",
    ),
    _Segment(
        "code",
        "Source code",
        "def checksum(rows: list[int]) -> int:\n"
        "    total = 0\n"
        "    for i, x in enumerate(rows):\n"
        "        total = (total * 31 + x) % 1000003\n"
        "    return total",
    ),
    _Segment(
        "digits",
        "Digit runs",
        "3141592653589793238462643383279502884197169399375105820974944592 "
        "0000000000000000000000001 987654321098765432109876543210",
    ),
    _Segment(
        "rare",
        "Rare Unicode blocks",
        "\U00010000\U00010001\U00010002 \U00010330\U00010331\U00010332 \u169b\u1681"
        "\u1682\u1683\u169c \U00010400\U00010401\U00010402 \u2d40\u2d41\u2d42 "
        "\U00012000\U00012001\U00012002 \U00013000\U00013040\U00013080 \U00010900"
        "\U00010901\U00010902 \u2c80\u2c82\u2c84 \u1403\u14c4\u1483\u144e\u1450\u1466"
        " \u16a0\u16a2\u16a6\u16a8\u16b1\u16b2 \U000103a0\U000103a1\U000103a2 \ua540"
        "\ua541\ua542 \ua000\ua001\ua002",
    ),
)

#: The segments joined by a single newline. Frozen forever: the reference
#: snapshot's ``canonical_text_tokens`` values are counts of exactly this string,
#: and there is no version field that would let a changed string be detected.
CANONICAL_TEXT: str = "\n".join(segment.text for segment in SEGMENTS)

#: Ratio of post-4.7 to pre-4.7 Claude token counts for identical text, from
#: Anthropic's own statement of the change. Used only within the Anthropic
#: family; across vendors a ratio like this carries no meaning.
GENERATION_RATIO: float = 1.30

#: Relative half-width of the band around :data:`GENERATION_RATIO` that counts
#: as "this is the other generation's tokenizer". Wide enough that the ~30%
#: figure being approximate does not matter, narrow enough that the two bands
#: cannot overlap the "matches the claim" band.
_RATIO_BAND: float = 0.08

#: A measured total this far from the reference still counts as a match. The
#: two terms cover different errors: the constant absorbs the token or two that
#: envelope differencing can leave behind, and the relative term absorbs a
#: proxy that re-serialises the prompt on the way through.
_TOLERANCE_TOKENS: int = 4
_TOLERANCE_RELATIVE: float = 0.01

#: Tier names Anthropic has used in model identifiers. Fable and Mythos are
#: version 5 and therefore land on the post-4.7 side by version alone.
_CLAUDE_TIERS = "opus|sonnet|haiku|fable|mythos|instant"

#: ``claude-opus-4-5-20251101`` style. The minor group is bounded to two digits
#: and followed by a non-digit assertion so that a trailing snapshot date is not
#: swallowed as a minor version.
_CLAUDE_NEW_STYLE = re.compile(rf"claude-(?:{_CLAUDE_TIERS})-(\d{{1,2}})(?:-(\d{{1,2}}))?(?!\d)")

#: ``claude-3-5-sonnet-20241022`` style, from before the tier moved forward.
_CLAUDE_OLD_STYLE = re.compile(rf"claude-(\d{{1,2}})-(\d{{1,2}})-(?:{_CLAUDE_TIERS})")

#: The generation boundary Anthropic's tokenizer change falls on.
_TOKENIZER_BOUNDARY: tuple[int, int] = (4, 7)


class TokenMeter:
    """Measures what an endpoint charges, in tokens, for a given prompt.

    Two probes need this and neither should pay for it twice, so measurements
    and the calibrated envelope overhead are cached in ``ctx.shared``. The meter
    prefers a token-counting endpoint and silently falls back to differencing
    ``usage.input_tokens`` when the endpoint has no such route or refuses the
    request; :attr:`path` says which one produced a number.

    Every method may raise :class:`~llmverify.errors.ProviderError`,
    :class:`~llmverify.errors.UnsupportedCapability` or
    :class:`~llmverify.errors.BudgetExhausted`. Callers turn those into evidence
    rather than letting them escape, because "this endpoint will not tell me its
    token count" is a normal outcome, not a crash.
    """

    #: One unit of calibration text. It ends in a newline so that two copies
    #: tokenize as two independent units instead of merging at the seam, which
    #: is what makes ``2 * count(unit) - count(unit + unit)`` the envelope.
    calibration_unit: ClassVar[str] = "The quick brown fox jumps over the lazy dog.\n"

    #: An envelope larger than this is not a chat template. It is a system
    #: prompt somebody injected, and it is worth telling the user about.
    injected_prompt_threshold: ClassVar[int] = 40

    def __init__(self, ctx: ProbeContext, *, max_tokens: int = 1) -> None:
        self._ctx = ctx
        self._max_tokens = max_tokens
        self._cache: dict[tuple[Any, ...], int] = ctx.shared.setdefault("token_measurements", {})
        self._count_tokens = bool(ctx.adapter.capabilities.count_tokens_endpoint)
        # Only the Anthropic adapter takes a tool_choice, and it matters there:
        # the published tool-use overhead differs between auto and forced.
        self._count_tokens_takes_choice = (
            "tool_choice" in inspect.signature(ctx.adapter.count_tokens).parameters
        )
        self.generation_requests = 0
        self.count_tokens_calls = 0
        self.cost_usd = 0.0
        self.tokens = 0
        #: Why the token-counting route was abandoned, when it was.
        self.fallback_reason: str | None = None

    @property
    def path(self) -> str:
        return "count_tokens" if self._count_tokens else "usage_differencing"

    async def measure(
        self,
        text: str,
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: str | None = None,
    ) -> int:
        """Raw token count for one prompt, envelope included."""
        messages = (Message(Role.USER, text),)
        tool_tuple = tuple(tools)
        key = (text, tuple(t.name for t in tool_tuple), tool_choice, self.path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._ctx.budget.check()
        if self._count_tokens:
            try:
                value = await self._via_count_tokens(messages, tool_tuple, tool_choice)
            except (UnsupportedCapability, ProviderError) as exc:
                self._count_tokens = False
                self.fallback_reason = redact(str(exc))[:200]
            else:
                self.count_tokens_calls += 1
                self._cache[key] = value
                return value

        value = await self._via_generation(messages, tool_tuple, tool_choice)
        self._cache[(text, tuple(t.name for t in tool_tuple), tool_choice, self.path)] = value
        return value

    async def envelope_overhead(self) -> int:
        """Tokens the endpoint adds to every prompt, whatever the prompt says.

        Measured, not assumed: one unit of calibration text costs
        ``envelope + unit``, two copies cost ``envelope + 2 * unit``, so the
        difference is the unit and the envelope follows. A negative result means
        the envelope is not fixed -- the two measurements disagree about
        something other than the text -- and is clamped to zero with the raw
        numbers kept in :meth:`envelope_report`.
        """
        report = await self.envelope_report()
        return int(report["overhead"])

    async def envelope_report(self) -> dict[str, Any]:
        """The calibration in full, cached across probes for the run."""
        cache_key = f"token_envelope:{self.path}"
        cached = self._ctx.shared.get(cache_key)
        if isinstance(cached, dict):
            return cached

        unit = self.calibration_unit
        single = await self.measure(unit)
        double = await self.measure(unit + unit)
        unit_tokens = double - single
        raw = single - unit_tokens
        report: dict[str, Any] = {
            "path": self.path,
            "single": single,
            "double": double,
            "unit_tokens": unit_tokens,
            "overhead": max(0, raw),
            "raw_overhead": raw,
            "plausible": 0 <= raw < 4096 and unit_tokens > 0,
            "suggests_injected_prompt": raw >= self.injected_prompt_threshold,
        }
        self._ctx.shared[cache_key] = report
        return report

    async def tokens_for(
        self,
        text: str,
        *,
        tools: Sequence[ToolSpec] = (),
        tool_choice: str | None = None,
    ) -> int:
        """Token count for ``text`` alone, with the envelope differenced away."""
        overhead = await self.envelope_overhead()
        return await self.measure(text, tools=tools, tool_choice=tool_choice) - overhead

    # ------------------------------------------------------------------ paths

    async def _via_count_tokens(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        tool_choice: str | None,
    ) -> int:
        if self._count_tokens_takes_choice:
            return await self._ctx.adapter.count_tokens(  # type: ignore[call-arg]
                messages, tools, tool_choice=tool_choice
            )
        return await self._ctx.adapter.count_tokens(messages, tools)

    async def _via_generation(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolSpec, ...],
        tool_choice: str | None,
    ) -> int:
        request = ChatRequest(
            messages=messages,
            max_tokens=self._max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        response, error = await self._ctx.adapter.try_chat(request)
        if response is None:
            raise error if isinstance(error, Exception) else ProviderError("request failed")

        self.generation_requests += 1
        self.cost_usd += self._ctx.budget.charge(
            response.usage.input_tokens, response.usage.output_tokens
        )
        self.tokens += response.usage.total_tokens or 0

        value = response.usage.input_tokens
        if value is None:
            raise UnsupportedCapability(
                "the endpoint reports no usage.input_tokens, so its tokenizer cannot be "
                "measured without a token-counting route"
            )
        return value


@register_probe
class TokenizerProbe(Probe):
    """Token counts for a fixed multi-script string, and what they imply."""

    name: ClassVar[str] = "tokenizer"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "tokenizer"
    order: ClassVar[int] = 50
    #: Calibration, the whole string, and one per segment. All of them are free
    #: on the token-counting path.
    estimated_requests: ClassVar[int] = 2 + 1 + len(SEGMENTS)
    description: ClassVar[str] = (
        "Token count of a fixed Latin/Cyrillic/CJK/Arabic/emoji/code/digit probe "
        "string, plus per-script ratios, compared against the reference tokenizer."
    )

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        meter = TokenMeter(ctx)

        try:
            envelope = await meter.envelope_report()
            total = await meter.tokens_for(CANONICAL_TEXT)
            segments, truncated = await self._profile(ctx, meter)
        except BudgetExhausted as exc:
            return [
                self._ev(
                    "canonical_text_tokens",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=f"the tokenizer measurement ran out of budget: {exc}",
                    duration_s=time.perf_counter() - started,
                )
            ]
        except (ProviderError, UnsupportedCapability) as exc:
            return [
                self._ev(
                    "canonical_text_tokens",
                    0.0,
                    status=EvidenceStatus.UNSUPPORTED,
                    detail=(
                        "the endpoint would not produce a usable token count: "
                        f"{redact(str(exc))[:240]}"
                    ),
                    cost_usd=meter.cost_usd,
                    tokens=meter.tokens,
                    duration_s=time.perf_counter() - started,
                )
            ]

        elapsed = time.perf_counter() - started
        shared: dict[str, Any] = {
            "path": meter.path,
            "fallback_reason": meter.fallback_reason,
            "envelope": envelope,
            "canonical_chars": len(CANONICAL_TEXT),
            "generation_requests": meter.generation_requests,
            "count_tokens_calls": meter.count_tokens_calls,
        }
        ctx.shared["canonical_text_tokens"] = total

        return [
            self._total(ctx, total, shared, meter, elapsed),
            self._script_profile(segments, shared, truncated),
        ]

    # ------------------------------------------------------------- measurement

    async def _profile(
        self, ctx: ProbeContext, meter: TokenMeter
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        """Token counts for each script segment, measured separately.

        A segment that cannot be measured is recorded with its error rather than
        dropped, so a profile with a hole in it still reads as a profile.
        """
        profile: dict[str, dict[str, Any]] = {}
        truncated = False

        for segment in SEGMENTS:
            try:
                tokens = await meter.tokens_for(segment.text)
            except BudgetExhausted:
                truncated = True
                break
            except (ProviderError, UnsupportedCapability) as exc:
                profile[segment.key] = {
                    "label": segment.label,
                    "chars": segment.chars,
                    "error": redact(str(exc))[:160],
                }
                continue
            profile[segment.key] = {
                "label": segment.label,
                "chars": segment.chars,
                "tokens": tokens,
                "tokens_per_char": round(tokens / segment.chars, 4) if segment.chars else None,
                "chars_per_token": round(segment.chars / tokens, 4) if tokens > 0 else None,
            }
        return profile, truncated

    # ---------------------------------------------------------------- scoring

    def _total(
        self,
        ctx: ProbeContext,
        measured: int,
        shared: dict[str, Any],
        meter: TokenMeter,
        elapsed: float,
    ) -> Evidence:
        """Compare the canonical count against the reference, or report it plainly."""
        data: dict[str, Any] = {**shared, "measured_tokens": measured}
        expected = _reference_tokens(ctx.reference)
        charged = {
            "cost_usd": meter.cost_usd,
            "tokens": meter.tokens,
            "duration_s": elapsed,
        }

        if expected is None:
            near = _nearest_records(ctx, measured, exclude=ctx.reference)
            data["models_matching_measurement"] = [record.id for record in near]
            detail = (
                f"the canonical probe string measured {measured} tokens via "
                f"{meter.path}. The reference snapshot records no canonical token count "
                f"for {ctx.provider.target_model!r}, so there is nothing to compare it "
                "against and this is reported, not weighed."
            )
            if near:
                detail += (
                    " For information only, the same count is recorded for "
                    f"{', '.join(record.id for record in near)}; models of one vendor "
                    "share a tokenizer across a generation, so that is not a "
                    "substitution finding."
                )
            return self._ev("canonical_text_tokens", 0.0, detail=detail, data=data, **charged)

        tolerance = _tolerance(expected)
        ratio = measured / expected if expected else 0.0
        data.update(
            {
                "expected_tokens": expected,
                "tolerance_tokens": tolerance,
                "deviation": measured - expected,
                "ratio": round(ratio, 4),
                "reference_source": _reference_source(ctx.reference),
            }
        )

        if abs(measured - expected) <= tolerance:
            return self._ev(
                "canonical_text_tokens",
                MODERATE,
                cap=STRONG,
                detail=(
                    f"the canonical probe string measured {measured} tokens against a "
                    f"reference of {expected} for {ctx.provider.target_model!r} "
                    f"(tolerance +/-{tolerance}). This pins the tokenizer generation; it "
                    "does not separate models that share one."
                ),
                data=data,
                **charged,
            )

        impostor = _nearest_records(ctx, measured, exclude=ctx.reference)
        if impostor:
            names = ", ".join(record.id for record in impostor)
            data["models_matching_measurement"] = [record.id for record in impostor]
            return self._ev(
                "canonical_text_tokens",
                -STRONG,
                cap=STRONG,
                detail=(
                    f"the canonical probe string measured {measured} tokens, which is not "
                    f"the {expected} recorded for {ctx.provider.target_model!r} but is the "
                    f"count recorded for {names}. The endpoint is running that tokenizer."
                ),
                data=data,
                **charged,
            )

        generation = self._generation_mismatch(ctx, measured, expected, ratio, data, charged)
        if generation is not None:
            return generation

        # Everything else: a real disagreement that names no known alternative.
        relative = abs(ratio - 1.0)
        magnitude = WEAK if relative < 0.05 else MODERATE
        return self._ev(
            "canonical_text_tokens",
            -magnitude,
            cap=STRONG,
            detail=(
                f"the canonical probe string measured {measured} tokens against a "
                f"reference of {expected} for {ctx.provider.target_model!r}, a "
                f"{relative:.1%} disagreement that matches no other tokenizer in the "
                "snapshot. A re-serialising proxy can shift a count by a token or two; "
                "this is larger than that."
            ),
            data=data,
            **charged,
        )

    def _generation_mismatch(
        self,
        ctx: ProbeContext,
        measured: int,
        expected: int,
        ratio: float,
        data: dict[str, Any],
        charged: dict[str, Any],
    ) -> Evidence | None:
        """The Claude 4.7 tokenizer-generation check, or ``None`` when it cannot fire.

        It fires only inside the Anthropic family, and only when the claimed
        model's own generation is known. A 30% ratio between two vendors is
        ordinary tokenizer variation and means nothing.
        """
        claimed_version = _claude_version(ctx.provider.target_model) or _claude_version(
            ctx.reference.id if ctx.reference is not None else ""
        )
        if claimed_version is None:
            return None
        if ctx.reference is not None and ctx.reference.vendor.strip().lower() != "anthropic":
            return None

        post_boundary = claimed_version >= _TOKENIZER_BOUNDARY
        data["claimed_claude_version"] = f"{claimed_version[0]}.{claimed_version[1]}"
        data["claimed_tokenizer_generation"] = "post_4_7" if post_boundary else "pre_4_7"

        preamble = (
            f"the canonical probe string measured {measured} tokens where "
            f"{ctx.provider.target_model!r} should produce {expected} -- a ratio of "
            f"{ratio:.2f}, which is the "
        )
        low = 1.0 / GENERATION_RATIO
        if post_boundary and abs(ratio - low) <= _RATIO_BAND * low:
            return self._ev(
                "canonical_text_tokens",
                -STRONG,
                cap=STRONG,
                detail=(
                    preamble + "pre-4.7 Claude tokenizer profile. Claude 4.7 and later "
                    "produce roughly 30% more tokens for identical text than 4.6 and "
                    "earlier, so this endpoint is running an older Claude generation than "
                    "the one it claims."
                ),
                data=data,
                **charged,
            )
        if not post_boundary and abs(ratio - GENERATION_RATIO) <= _RATIO_BAND * GENERATION_RATIO:
            return self._ev(
                "canonical_text_tokens",
                -STRONG,
                cap=STRONG,
                detail=(
                    preamble + "post-4.7 Claude tokenizer profile. The endpoint is running "
                    "a newer Claude generation than the one it claims."
                ),
                data=data,
                **charged,
            )
        return None

    def _script_profile(
        self,
        segments: dict[str, dict[str, Any]],
        shared: dict[str, Any],
        truncated: bool,
    ) -> Evidence:
        """Report the per-script ratios. Deliberately unweighted.

        The profile is the most discriminative thing this probe produces and the
        one thing it has nothing to compare against: no published reference
        records per-script token ratios for any model. Assigning it an LLR would
        mean inventing the expectation it is scored against.
        """
        measured = {
            key: value for key, value in segments.items() if value.get("tokens") is not None
        }
        summary = ", ".join(
            f"{value['label']} {value['tokens_per_char']:.2f} tok/char"
            for value in measured.values()
            if value.get("tokens_per_char") is not None
        )
        return self._ev(
            "script_profile",
            0.0,
            status=EvidenceStatus.TRUNCATED if truncated and not measured else EvidenceStatus.OK,
            detail=(
                "per-script token-to-character ratios, recorded rather than weighed "
                "because no reference profile exists to compare them against: "
                + (summary or "no segment could be measured")
                + "."
            ),
            data={**shared, "segments": segments},
        )

    # ----------------------------------------------------------------- helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = MODERATE,
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
            cap=cap,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Reference lookup
# --------------------------------------------------------------------------- #


def _reference_tokens(record: ModelRecord | None) -> int | None:
    if record is None or record.token_accounting is None:
        return None
    return record.token_accounting.canonical_text_tokens


def _reference_source(record: ModelRecord | None) -> str | None:
    if record is None or record.token_accounting is None:
        return None
    return record.token_accounting.source


def _tolerance(expected: int) -> int:
    return max(_TOLERANCE_TOKENS, round(expected * _TOLERANCE_RELATIVE))


def _nearest_records(
    ctx: ProbeContext, measured: int, *, exclude: ModelRecord | None
) -> list[ModelRecord]:
    """Snapshot models whose recorded canonical count matches the measurement."""
    if ctx.snapshot is None:
        return []
    matches: list[ModelRecord] = []
    for record in ctx.snapshot.models:
        if exclude is not None and record.id == exclude.id:
            continue
        expected = _reference_tokens(record)
        if expected is not None and abs(measured - expected) <= _tolerance(expected):
            matches.append(record)
    return matches


# --------------------------------------------------------------------------- #
# Claude version parsing
# --------------------------------------------------------------------------- #


def _claude_version(model_id: str) -> tuple[int, int] | None:
    """Major and minor version of a Claude identifier, or ``None`` if it is not one.

    Both naming schemes Anthropic has shipped are accepted: the current
    ``claude-<tier>-<major>[-<minor>]`` and the older
    ``claude-<major>-<minor>-<tier>``. A missing minor reads as zero, so
    ``claude-opus-5`` is ``(5, 0)`` and sorts after ``claude-opus-4-8``.
    """
    ident = model_id.strip().lower().replace("_", "-").rsplit("/", 1)[-1]
    if "claude" not in ident:
        return None
    match = _CLAUDE_NEW_STYLE.search(ident) or _CLAUDE_OLD_STYLE.search(ident)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)
