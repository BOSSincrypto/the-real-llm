"""The tool-use system prompt, weighed to the token.

This is the strongest cheap identity check that exists for the Claude family,
and it costs a handful of free token-counting calls. Anthropic publishes the
exact token cost of the system prompt the API injects when a request carries
tools, and that number is different for every model generation: 286 for one,
354 for another, 675 for a third, 496 for a fourth. It is not a statistic and
not a behaviour. It is an integer the endpoint reports about itself, and it
either matches the claimed model or it matches a different one.

**The measurement is not the naive difference.** Sending one tool and
subtracting a no-tool baseline leaves a residual of *system prompt + the tool's
own JSON*, because the tool definition is itself part of the input. The tool's
contribution is removed with a second differencing: two tools identical except
for their names cost one system prompt and two definitions, so the second
measurement minus the first is one definition, and subtracting that from the
first residual leaves the system prompt alone.

That step rests on one assumption -- that ``probe_tool_one`` and
``probe_tool_two`` tokenize to the same number of tokens. They are the same
length and differ only in a common three-letter word, so they almost certainly
do; if they do not, both numbers shift together by the same small amount, which
is what the tolerance is for.

**Tolerance is +/-2 tokens.** The published figures are exact, so in principle
the comparison should be exact too. Two tokens of slack covers the harmless
ways a number moves without the model changing: a proxy that re-serialises the
tool schema with different whitespace, a JSON encoder that escapes a character
differently, the tool-name assumption above. It is far tighter than the gap
between any two generations in the table -- the closest pair differs by four --
so the slack costs no discriminating power at all.

**Why the family cap is DECISIVE.** Almost every other probe in this package
measures a behaviour, and behaviours drift, so their evidence is capped well
below certainty. This one measures an integer that a model generation either
produces or does not. A residual matching a *different* model in the snapshot
is the most useful sentence this tool can produce, because it does not merely
say "not what you claimed", it says which model it actually is.

Two honest caveats belong here rather than in a footnote. The residual is
computed from numbers the provider reports about its own usage, so a provider
willing to fabricate ``usage.input_tokens`` consistently across four different
requests can defeat it -- at which point every token-based measurement in the
tool is defeated too. And a proxy that injects its own system prompt inflates
every measurement equally; that cancels in the differencing, but it is measured
and reported anyway, because it is the most common innocent explanation for a
residual that matches nothing.

On an endpoint whose claimed model has no published tool-use overhead the probe
does not run at all. There is no comparison to make, and inventing one would be
worse than staying silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted, ProviderError, UnsupportedCapability
from ..evidence import DECISIVE, MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..reference.schema import ModelRecord
from ..types import ToolSpec
from . import Probe, ProbeContext, register_probe
from .tokenizer import TokenMeter

__all__ = ["TokenAccountingProbe"]

#: Slack in tokens on each published figure. See the module docstring for why
#: this is 2 and not 0.
TOLERANCE: int = 2

#: A short fixed prompt. Its own token cost cancels in every difference taken
#: here, so its content is irrelevant -- only its constancy matters.
_PROMPT = "Say ok."

#: A JSON Schema with no properties, so the definition stays as small as the
#: protocol allows and the two tools are byte-identical apart from the name.
_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_DESCRIPTION = "Return the current UTC time as an ISO 8601 string."

#: Identical in every token-bearing respect except the final word, which is the
#: same length in both. Their definitions therefore cost the same, which is what
#: makes the second differencing isolate the system prompt.
FIRST_TOOL = ToolSpec(name="probe_tool_one", description=_DESCRIPTION, parameters=_SCHEMA)
SECOND_TOOL = ToolSpec(name="probe_tool_two", description=_DESCRIPTION, parameters=_SCHEMA)


@dataclass(frozen=True, slots=True)
class Residuals:
    """The four raw measurements and the quantities derived from them."""

    base: int
    auto: int
    forced: int
    two_tools: int

    @property
    def tool_definition(self) -> int:
        """Token cost of one tool definition, from the one-tool/two-tool difference."""
        return self.two_tools - self.auto

    @property
    def system_auto(self) -> int:
        """Tool-use system prompt with ``tool_choice`` auto."""
        return self.auto - self.base - self.tool_definition

    @property
    def system_forced(self) -> int:
        """Tool-use system prompt with ``tool_choice`` forced."""
        return self.forced - self.base - self.tool_definition

    @property
    def forced_gap(self) -> int:
        """Forced minus auto.

        The most robust number here: the tool definition appears in both terms
        and cancels, so this survives any error in estimating it.
        """
        return self.forced - self.auto

    @property
    def coherent(self) -> bool:
        """Whether the derived quantities are physically possible."""
        return self.tool_definition > 0 and self.system_auto >= 0 and self.system_forced >= 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_base": self.base,
            "raw_tool_choice_auto": self.auto,
            "raw_tool_choice_forced": self.forced,
            "raw_two_tools": self.two_tools,
            "naive_residual_auto": self.auto - self.base,
            "naive_residual_forced": self.forced - self.base,
            "tool_definition_tokens": self.tool_definition,
            "system_prompt_auto": self.system_auto,
            "system_prompt_forced": self.system_forced,
            "forced_minus_auto": self.forced_gap,
        }


@register_probe
class TokenAccountingProbe(Probe):
    """Identify the model generation from its published tool-use overhead."""

    name: ClassVar[str] = "token_accounting"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "token_accounting"
    #: Early, and deliberately so: it is nearly free, and when it fires it
    #: settles the question before anything expensive runs.
    order: ClassVar[int] = 15
    estimated_requests: ClassVar[int] = 6
    description: ClassVar[str] = (
        "Tool-use system-prompt overhead measured by differencing, matched against "
        "the exact per-model figures Anthropic publishes."
    )

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only where a published overhead exists to compare against.

        This is the whole degradation story for non-Anthropic endpoints: no
        other vendor publishes these numbers, so no reference record carries
        them, so the probe skips itself rather than manufacturing a comparison.
        An OpenAI-compatible endpoint *claiming* a Claude model does run,
        because the residual is a property of whatever computed the tokens, not
        of the protocol it was packaged in.
        """
        if not ctx.adapter.capabilities.tools:
            return False
        accounting = ctx.reference.token_accounting if ctx.reference is not None else None
        if accounting is None:
            return False
        return (
            accounting.tool_overhead_auto is not None
            or accounting.tool_overhead_forced is not None
        )

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        meter = TokenMeter(ctx)

        try:
            envelope = await meter.envelope_report()
            residuals = Residuals(
                base=await meter.measure(_PROMPT),
                auto=await meter.measure(_PROMPT, tools=(FIRST_TOOL,), tool_choice="auto"),
                forced=await meter.measure(
                    _PROMPT, tools=(FIRST_TOOL,), tool_choice="required"
                ),
                two_tools=await meter.measure(
                    _PROMPT, tools=(FIRST_TOOL, SECOND_TOOL), tool_choice="auto"
                ),
            )
        except BudgetExhausted as exc:
            return [self._aborted(EvidenceStatus.TRUNCATED, str(exc), meter, started)]
        except (ProviderError, UnsupportedCapability) as exc:
            return [
                self._aborted(
                    EvidenceStatus.UNSUPPORTED,
                    "the endpoint would not produce the token counts this probe needs: "
                    f"{redact(str(exc))[:240]}",
                    meter,
                    started,
                )
            ]

        elapsed = time.perf_counter() - started
        data: dict[str, Any] = {
            "path": meter.path,
            "tolerance_tokens": TOLERANCE,
            "envelope": envelope,
            **residuals.as_dict(),
        }
        ctx.shared["tool_use_residuals"] = residuals

        evidence = [self._overhead(ctx, residuals, envelope, data, meter, elapsed)]
        gap = self._forced_gap(ctx, residuals, data)
        if gap is not None:
            evidence.append(gap)
        evidence.append(self._baseline(envelope, data))
        return evidence

    # ---------------------------------------------------------------- scoring

    def _overhead(
        self,
        ctx: ProbeContext,
        residuals: Residuals,
        envelope: dict[str, Any],
        data: dict[str, Any],
        meter: TokenMeter,
        elapsed: float,
    ) -> Evidence:
        """Match the measured system-prompt overhead against every known model."""
        charged = {
            "cost_usd": meter.cost_usd,
            "tokens": meter.tokens,
            "duration_s": elapsed,
        }
        measured = f"{residuals.system_auto} auto / {residuals.system_forced} forced"

        if not residuals.coherent:
            return self._ev(
                "tool_system_prompt",
                0.0,
                status=EvidenceStatus.ERROR,
                detail=(
                    "the four measurements do not decompose into a system prompt and a "
                    f"tool definition ({measured}, definition "
                    f"{residuals.tool_definition}). Something other than the tool payload "
                    "changed between requests, so no comparison is safe."
                ),
                data=data,
                **charged,
            )

        claimed = ctx.reference
        groups = _classify(ctx, residuals)
        data["models_matching_both"] = [record.id for record in groups.full]
        data["models_matching_single_published"] = [record.id for record in groups.single]
        data["models_matching_one_of_two"] = [record.id for record in groups.partial]
        data["expected_auto"] = _expected(claimed, "auto")
        data["expected_forced"] = _expected(claimed, "forced")

        if groups.holds(claimed, "full"):
            assert claimed is not None
            others = [record.id for record in groups.full if record.id != claimed.id]
            shared_note = (
                f" These figures are shared with {', '.join(others)}, so they identify "
                "the generation rather than the individual model."
                if others
                else ""
            )
            return self._ev(
                "tool_system_prompt",
                DECISIVE,
                detail=(
                    f"the tool-use system prompt measured {measured} tokens, matching both "
                    f"published figures for {claimed.id} exactly (+/-{TOLERANCE})."
                    + shared_note
                ),
                data=data,
                **charged,
            )

        if groups.holds(claimed, "single"):
            assert claimed is not None
            return self._ev(
                "tool_system_prompt",
                STRONG,
                detail=(
                    f"the tool-use system prompt measured {measured} tokens, matching the "
                    f"one figure the reference publishes for {claimed.id} "
                    f"({_expected_text(claimed)}). One integer rather than two, so this is "
                    "strong rather than conclusive."
                ),
                data=data,
                **charged,
            )

        if groups.full or groups.single:
            impostors = groups.full or groups.single
            names = ", ".join(record.id for record in impostors)
            return self._ev(
                "tool_system_prompt",
                -DECISIVE if groups.full else -STRONG,
                detail=(
                    f"the tool-use system prompt measured {measured} tokens. That is not "
                    f"{_expected_text(claimed)} for the claimed "
                    f"{ctx.provider.target_model!r} -- it is the published figure for "
                    f"{names}. This endpoint is serving that model generation."
                ),
                data=data,
                **charged,
            )

        if groups.holds(claimed, "partial"):
            assert claimed is not None
            return self._ev(
                "tool_system_prompt",
                -MODERATE,
                detail=(
                    f"the tool-use system prompt measured {measured} tokens: one of the two "
                    f"figures matches {claimed.id} and the other does not. The gap between "
                    "them is measured without any assumption about the tool definition, so "
                    "a half-match is a real inconsistency rather than an artefact."
                ),
                data=data,
                **charged,
            )

        if groups.partial:
            names = ", ".join(record.id for record in groups.partial)
            return self._ev(
                "tool_system_prompt",
                -MODERATE,
                detail=(
                    f"the tool-use system prompt measured {measured} tokens, matching "
                    f"neither figure published for {ctx.provider.target_model!r}; one of the "
                    f"two matches {names} instead."
                ),
                data=data,
                **charged,
            )

        injected = bool(envelope.get("suggests_injected_prompt"))
        return self._ev(
            "tool_system_prompt",
            -WEAK if injected else -MODERATE,
            detail=(
                f"the tool-use system prompt measured {measured} tokens, which matches no "
                "model in the reference snapshot"
                + (
                    f", and the endpoint already adds {envelope.get('overhead')} tokens to "
                    "every request before any tool is sent. A proxy that rewrites the "
                    "system prompt changes this number without changing the model, so this "
                    "is weighed lightly."
                    if injected
                    else ". The claimed model should produce "
                    f"{_expected_text(claimed)}."
                )
            ),
            data=data,
            **charged,
        )

    def _forced_gap(
        self, ctx: ProbeContext, residuals: Residuals, data: dict[str, Any]
    ) -> Evidence | None:
        """Weigh forced-minus-auto, the one number no assumption can shift.

        The tool definition contributes equally to both measurements, so it
        cancels here exactly. When the absolute figures are ambiguous this is
        still trustworthy.
        """
        record = ctx.reference
        auto = _expected(record, "auto")
        forced = _expected(record, "forced")
        if auto is None or forced is None:
            return None

        expected_gap = forced - auto
        measured_gap = residuals.forced_gap
        data["expected_forced_minus_auto"] = expected_gap

        matching = [
            other.id
            for other in _snapshot_models(ctx)
            if _gap(other) is not None and abs(_gap(other) - measured_gap) <= TOLERANCE
        ]
        data["models_matching_gap"] = matching

        if abs(measured_gap - expected_gap) <= TOLERANCE:
            return self._ev(
                "forced_choice_delta",
                MODERATE,
                cap=MODERATE,
                detail=(
                    f"forcing tool use added {measured_gap} tokens against a published "
                    f"{expected_gap} for {ctx.provider.target_model!r}. This difference is "
                    "free of any assumption about the tool definition's own token cost."
                ),
                data=data,
            )
        return self._ev(
            "forced_choice_delta",
            -MODERATE,
            cap=MODERATE,
            detail=(
                f"forcing tool use added {measured_gap} tokens where "
                f"{ctx.provider.target_model!r} should add {expected_gap}"
                + (f"; {', '.join(matching)} adds {measured_gap}." if matching else ".")
                + " The tool definition cancels in this difference, so it cannot be blamed "
                "on serialisation."
            ),
            data=data,
        )

    def _baseline(self, envelope: dict[str, Any], data: dict[str, Any]) -> Evidence:
        """Report the no-tool envelope. Never weighed, always useful.

        An endpoint that wraps every request in a large system prompt is the
        commonest innocent explanation for a residual that matches nothing, and
        the user deserves the number rather than the inference.
        """
        overhead = envelope.get("overhead")
        if not envelope.get("plausible"):
            return self._ev(
                "baseline_envelope",
                0.0,
                cap=WEAK,
                detail=(
                    "the request envelope could not be calibrated: two prompts differing "
                    "only by a repetition did not differ by a whole repetition's worth of "
                    "tokens, so the endpoint is not charging a fixed overhead."
                ),
                data=data,
            )
        return self._ev(
            "baseline_envelope",
            0.0,
            cap=WEAK,
            detail=(
                f"the endpoint charges {overhead} tokens of envelope before any prompt "
                "content, measured with no tools present. Recorded, not weighed: a chat "
                "template and an injected system prompt look identical from here, and "
                "both cancel out of every residual above."
            ),
            data=data,
        )

    # ----------------------------------------------------------------- helpers

    def _aborted(
        self,
        status: EvidenceStatus,
        detail: str,
        meter: TokenMeter,
        started: float,
    ) -> Evidence:
        return self._ev(
            "tool_system_prompt",
            0.0,
            status=status,
            detail=detail,
            data={"path": meter.path, "fallback_reason": meter.fallback_reason},
            cost_usd=meter.cost_usd,
            tokens=meter.tokens,
            duration_s=time.perf_counter() - started,
        )

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = DECISIVE,
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
# Reference matching
# --------------------------------------------------------------------------- #


def _expected(record: ModelRecord | None, which: str) -> int | None:
    if record is None or record.token_accounting is None:
        return None
    if which == "auto":
        return record.token_accounting.tool_overhead_auto
    return record.token_accounting.tool_overhead_forced


def _gap(record: ModelRecord | None) -> int | None:
    auto, forced = _expected(record, "auto"), _expected(record, "forced")
    if auto is None or forced is None:
        return None
    return forced - auto


def _expected_text(record: ModelRecord | None) -> str:
    auto, forced = _expected(record, "auto"), _expected(record, "forced")
    if auto is None and forced is None:
        return "any published figure"
    left = f"{auto} auto" if auto is not None else "an unrecorded auto figure"
    right = f"{forced} forced" if forced is not None else "an unrecorded forced figure"
    return f"{left} / {right}"


def _snapshot_models(ctx: ProbeContext) -> tuple[ModelRecord, ...]:
    return ctx.snapshot.models if ctx.snapshot is not None else ()


@dataclass(frozen=True, slots=True)
class _Classified:
    """Snapshot models grouped by how well they explain the measurement."""

    #: Both published figures present and both matched.
    full: tuple[ModelRecord, ...]
    #: Only one figure is published for this model, and it matched.
    single: tuple[ModelRecord, ...]
    #: One figure matched and the other is contradicted.
    partial: tuple[ModelRecord, ...]

    def holds(self, record: ModelRecord | None, bucket: str) -> bool:
        if record is None:
            return False
        return any(other.id == record.id for other in getattr(self, bucket))


def _classify(ctx: ProbeContext, residuals: Residuals) -> _Classified:
    """Group known models by how their published figures compare to the measurement.

    An absent figure is unknown, not contradicted. A model recorded with only
    one of the two numbers can therefore still match, but it lands in
    ``single`` rather than ``full``, because one integer is a weaker claim than
    two and the scoring treats it as such.
    """
    full: list[ModelRecord] = []
    single: list[ModelRecord] = []
    partial: list[ModelRecord] = []

    for record in _snapshot_models(ctx):
        verdicts = [
            _agrees(_expected(record, "auto"), residuals.system_auto),
            _agrees(_expected(record, "forced"), residuals.system_forced),
        ]
        matched = sum(1 for v in verdicts if v is True)
        contradicted = sum(1 for v in verdicts if v is False)
        if matched and contradicted:
            partial.append(record)
        elif matched == 2:
            full.append(record)
        elif matched == 1:
            single.append(record)
    return _Classified(tuple(full), tuple(single), tuple(partial))


def _agrees(expected: int | None, measured: int) -> bool | None:
    if expected is None:
        return None
    return abs(expected - measured) <= TOLERANCE
