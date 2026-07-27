"""Locate the endpoint's effective knowledge boundary by binary search.

A model cannot know what happened after it stopped training. That makes the
boundary a coarse but genuine fingerprint: a cheaper, older model substituted
for a frontier one usually has an older boundary too, and the gap is visible in
a handful of questions.

**The ladder.** :data:`LADDER` holds dated facts, each with an unambiguous
machine-checkable answer that could not be known before its date. Every one of
them is drawn from the verified research digest this package was built against
-- model releases, dated snapshot identifiers, one leaderboard retirement -- and
nothing else. Facts whose first-public date could not be pinned were left out
rather than dated by guesswork, because a rung placed at the wrong date biases
the estimate in whichever direction the error runs, and both directions produce
false evidence. The ladder is therefore sparse, with a gap between November 2025
and March 2026 that the estimate simply cannot resolve. Eleven well-dated rungs
and an honest interval beat thirty invented ones.

Each question states its own date and asks for something that cannot be guessed
from the date -- an exact API identifier, a snapshot suffix, a month. Answering
"what was released in June 2026" with a plausible-sounding name is not a pass.

**Self-knowledge is excluded.** Vendors post-train their models to know their
own names and release dates, sometimes for events after the training cutoff. A
rung about the claimed model itself would therefore be answered correctly by
the genuine article and read as knowledge from beyond its own boundary -- a
false accusation against exactly the endpoint we are trying to clear. Rungs
naming the claimed model are dropped before the search starts.

**The search.** Rungs are ordered by date and probed by binary search, so a
boundary is located in about four questions rather than eleven. The two rungs
bracketing the transition are then asked outright if the search did not already
ask them, because a single flaky answer at the transition is the one error that
would move the estimate. Each rung is one Bernoulli trial, graded by exact
match against a small accepted-answer set -- never by a judge model, which
would put a second unverified model inside a tool whose purpose is to verify
one. The result is reported as the interval between the latest rung answered
correctly and the earliest one answered wrongly, not as a point estimate, and
non-monotone results (a later fact known while an earlier one is not) are
reported rather than smoothed away.

**Weighting, and why it is capped at MODERATE.** A boundary *earlier* than the
claimed model's is real negative evidence: it is what serving a cheaper, older
model looks like. A boundary *later* is stronger still, because no model can
know what happened after its own training. Neither is ever more than moderate,
for three reasons that no amount of sampling fixes. Models are unreliable
narrators about events near their boundary, recalling some and not others
essentially at random. A retrieval-augmented proxy answers post-cutoff
questions legitimately and would look like a forgery here. And a model that
declines to answer is indistinguishable, from outside, from one that never
knew. The ``knowledge`` family caps at MODERATE for the same reasons.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, Evidence, EvidenceStatus
from ..types import ChatRequest, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["LADDER", "KnowledgeCutoffProbe"]


@dataclass(frozen=True, slots=True)
class _Event:
    """One dated fact and the answers that count as knowing it."""

    date: dt.date
    key: str
    question: str
    #: Normalised strings, any of which counts as correct if it appears in the
    #: normalised answer. Written lowercase and hyphen-separated to match
    #: :func:`_normalise`.
    answers: tuple[str, ...]
    #: Model identifiers the question is about. A rung is dropped when one of
    #: these is the model being verified.
    subjects: tuple[str, ...] = ()
    #: How the date was established. ``announced`` is a stated release date;
    #: ``identifier`` is read off the vendor's own dated naming convention.
    dating: str = "announced"


#: Appended to every question. The explicit UNKNOWN option matters: a model that
#: says so has told us it does not know, which is an observation, whereas a
#: model that rambles has told us nothing.
_INSTRUCTION = (
    'Reply with the answer alone on a final line formatted exactly as "ANSWER: <answer>". '
    'If you do not know, reply "ANSWER: UNKNOWN". Do not guess.'
)

#: Dated facts in date order. Every entry comes from the verified reference
#: research; see the module docstring on why the list is short.
LADDER: tuple[_Event, ...] = (
    _Event(
        dt.date(2025, 3, 13),
        "open_llm_leaderboard_retired",
        "HuggingFace retired its Open LLM Leaderboard and stopped updating it. In which "
        "month did that happen? Answer as YYYY-MM.",
        # The question asks for one format; the alternatives are here because a
        # model that knows the answer should not fail the rung on spelling.
        ("2025-03", "march-2025"),
    ),
    _Event(
        dt.date(2025, 8, 5),
        "claude_opus_4_1_snapshot",
        "Write out in full the dated snapshot API model identifier for Anthropic's "
        "Claude Opus 4.1, including the date suffix.",
        ("claude-opus-4-1-20250805",),
        subjects=("claude-opus-4-1-20250805",),
        dating="identifier",
    ),
    _Event(
        dt.date(2025, 9, 29),
        "claude_sonnet_4_5_snapshot",
        "Write out in full the dated snapshot API model identifier for Anthropic's "
        "Claude Sonnet 4.5, including the date suffix.",
        ("claude-sonnet-4-5-20250929",),
        subjects=("claude-sonnet-4-5-20250929",),
        dating="identifier",
    ),
    _Event(
        dt.date(2025, 11, 1),
        "claude_opus_4_5_snapshot",
        "Write out in full the dated snapshot API model identifier for Anthropic's "
        "Claude Opus 4.5, including the date suffix.",
        ("claude-opus-4-5-20251101",),
        subjects=("claude-opus-4-5-20251101",),
        dating="identifier",
    ),
    _Event(
        dt.date(2026, 3, 1),
        "mistral_small_2026",
        "Mistral's model identifiers end in a four-digit YYMM release stamp. Give the "
        "full identifier of the Mistral Small release from 2026.",
        ("mistral-small-2603",),
        subjects=("mistral-small-2603",),
        dating="identifier",
    ),
    _Event(
        dt.date(2026, 5, 1),
        "cohere_command_a_plus",
        "Cohere's Command A+ model identifier ends with the month and year it was "
        "released. Write the identifier out in full.",
        ("command-a-plus-05-2026",),
        subjects=("command-a-plus-05-2026",),
        dating="identifier",
    ),
    _Event(
        dt.date(2026, 6, 9),
        "claude_fable_5_release",
        "On 9 June 2026 Anthropic released a Claude model with a one-million-token "
        "context window. Give its exact API model identifier.",
        ("claude-fable-5",),
        subjects=("claude-fable-5",),
    ),
    _Event(
        dt.date(2026, 6, 30),
        "claude_sonnet_5_release",
        "On 30 June 2026 Anthropic released a mid-tier Claude model priced below its "
        "flagship. Give its exact API model identifier.",
        ("claude-sonnet-5",),
        subjects=("claude-sonnet-5",),
    ),
    _Event(
        dt.date(2026, 7, 9),
        "gpt_5_6_sol_pro_release",
        "On 9 July 2026 OpenAI released a 'pro' variant of one of its GPT-5.6 models. "
        "Give its exact model identifier.",
        ("gpt-5.6-sol-pro",),
        subjects=("gpt-5.6-sol-pro", "gpt-5.6-sol"),
    ),
    _Event(
        dt.date(2026, 7, 21),
        "gemini_3_6_flash_release",
        "On 21 July 2026 Google released a new Gemini Flash model. Give its exact "
        "model identifier.",
        ("gemini-3.6-flash",),
        subjects=("gemini-3.6-flash",),
    ),
    _Event(
        dt.date(2026, 7, 24),
        "claude_opus_5_release",
        "On 24 July 2026 Anthropic released a new flagship Claude model. Give its "
        "exact API model identifier.",
        ("claude-opus-5",),
        subjects=("claude-opus-5",),
    ),
)

#: How far the measured boundary may sit from the published one and still count
#: as agreement. Vendors state cutoffs to the month, models behave as if theirs
#: is earlier than stated, and the ladder has month-scale gaps, so anything
#: tighter than this would be measuring the ladder rather than the endpoint.
SLACK_DAYS: int = 60

_DAYS_PER_MONTH = 30.44

#: A disagreement this many months beyond the slack earns the full weight the
#: cap allows. Knowing the future is treated as reaching that point sooner,
#: because it has no innocent explanation short of retrieval.
_EARLY_SATURATION_MONTHS = 6.0
_LATE_SATURATION_MONTHS = 3.0

#: Below this the evidence is not worth reporting as evidence at all.
_MINIMUM_FRACTION = 0.35

_ANSWER_RE = re.compile(r"ANSWER\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

#: Trailing snapshot dates and version tags, stripped when deciding whether a
#: rung is about the model under test.
_VERSION_SUFFIX = re.compile(r"[-_](?:\d{8}|\d{6}|\d{4}-\d{2}-\d{2}|v\d+)$")


@dataclass(slots=True)
class _Outcome:
    """What one rung produced."""

    key: str
    date: dt.date
    answered: bool
    correct: bool
    extracted: str | None = None
    text: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "answered": self.answered,
            "correct": self.correct,
            "extracted": self.extracted,
            "error": self.error,
        }


@dataclass(slots=True)
class _Estimate:
    """The interval the answered rungs support."""

    lower: dt.date | None = None
    upper: dt.date | None = None
    monotonic: bool = True
    answered: int = 0
    correct: int = 0
    unanswered: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.lower is not None and self.upper is not None:
            return f"after {self.lower.isoformat()} and before {self.upper.isoformat()}"
        if self.lower is not None:
            return f"at or after {self.lower.isoformat()}"
        if self.upper is not None:
            return f"before {self.upper.isoformat()}"
        return "unconstrained"


@register_probe
class KnowledgeCutoffProbe(Probe):
    """Bracket the endpoint's knowledge boundary and compare it with the claim."""

    name: ClassVar[str] = "knowledge_cutoff"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "knowledge"
    order: ClassVar[int] = 60
    #: Binary search over eleven rungs plus the two bracketing confirmations.
    estimated_requests: ClassVar[int] = 7
    description: ClassVar[str] = (
        "Binary search over dated, exactly-graded facts to bracket the endpoint's "
        "knowledge boundary, compared with the claimed model's published cutoff."
    )

    #: Room for a reasoning model to think and still answer. A response with no
    #: extractable answer is recorded as unanswered, never as wrong.
    max_tokens: ClassVar[int] = 512

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only when there is a published cutoff to compare against.

        Without one the questions would still be askable, but the answers would
        cost real requests and buy nothing, since the whole probe is a
        comparison.
        """
        return ctx.reference is not None and ctx.reference.knowledge_cutoff is not None

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        events, dropped = _relevant_events(ctx)
        if len(events) < 2:
            return [
                self._ev(
                    "knowledge_boundary",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        "the dated-fact ladder has fewer than two rungs left once rungs "
                        "about the model under test are removed, which is not enough to "
                        "bracket anything."
                    ),
                    data={"dropped_self_referential": dropped},
                )
            ]

        outcomes, truncated, cost, tokens = await self._search(ctx, events)
        elapsed = time.perf_counter() - started

        estimate = _estimate(outcomes)
        claimed = ctx.reference.knowledge_cutoff if ctx.reference is not None else None
        data: dict[str, Any] = {
            "claimed_cutoff": claimed.isoformat() if claimed is not None else None,
            "slack_days": SLACK_DAYS,
            "ladder_size": len(events),
            "dropped_self_referential": dropped,
            "rungs_asked": len(outcomes),
            "estimated_lower": estimate.lower.isoformat() if estimate.lower else None,
            "estimated_upper": estimate.upper.isoformat() if estimate.upper else None,
            "monotonic": estimate.monotonic,
            "unanswered": estimate.unanswered,
            "outcomes": {outcome.key: outcome.as_dict() for outcome in outcomes},
        }
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        if estimate.answered == 0:
            return [
                self._ev(
                    "knowledge_boundary",
                    0.0,
                    status=EvidenceStatus.TRUNCATED if truncated else EvidenceStatus.ERROR,
                    detail=(
                        "no rung produced an extractable answer, so the endpoint's "
                        "knowledge boundary was not measured."
                    ),
                    data=data,
                    **charged,
                )
            ]

        return [
            self._compare(ctx, estimate, data, truncated, charged),
        ]

    # ------------------------------------------------------------------ search

    async def _search(
        self, ctx: ProbeContext, events: tuple[_Event, ...]
    ) -> tuple[list[_Outcome], bool, float, int]:
        """Binary search for the transition, then confirm the rungs around it.

        An unanswered rung is treated as "does not know" for the purpose of
        steering the search, because that is the more likely of the two
        explanations, but it contributes nothing to the interval afterwards.
        """
        asked: dict[int, _Outcome] = {}
        cost = 0.0
        tokens = 0
        truncated = False

        async def ask(index: int) -> _Outcome | None:
            nonlocal cost, tokens, truncated
            if index in asked:
                return asked[index]
            try:
                ctx.budget.check()
            except BudgetExhausted:
                truncated = True
                return None
            outcome, spent, used = await self._ask(ctx, events[index])
            cost += spent
            tokens += used
            asked[index] = outcome
            return outcome

        lo, hi = 0, len(events)
        while lo < hi:
            mid = (lo + hi) // 2
            outcome = await ask(mid)
            if outcome is None:
                break
            if outcome.correct:
                lo = mid + 1
            else:
                hi = mid

        for index in (lo - 1, lo):
            if 0 <= index < len(events):
                await ask(index)

        return [asked[index] for index in sorted(asked)], truncated, cost, tokens

    async def _ask(
        self, ctx: ProbeContext, event: _Event
    ) -> tuple[_Outcome, float, int]:
        """Put one rung to the endpoint and grade the reply."""
        request = ChatRequest(
            messages=(Message(Role.USER, f"{event.question}\n\n{_INSTRUCTION}"),),
            max_tokens=self.max_tokens,
        )
        response, error = await ctx.adapter.try_chat(request)
        if response is None:
            return (
                _Outcome(
                    key=event.key,
                    date=event.date,
                    answered=False,
                    correct=False,
                    error=redact(str(error))[:200],
                ),
                0.0,
                0,
            )

        cost = ctx.budget.charge(response.usage.input_tokens, response.usage.output_tokens)
        extracted = _extract(response.text)
        answered = extracted is not None
        correct = answered and _grade(extracted or "", event.answers)

        return (
            _Outcome(
                key=event.key,
                date=event.date,
                answered=answered,
                correct=correct,
                extracted=redact(extracted)[:120] if extracted else None,
                text=redact(response.text)[:400],
            ),
            cost,
            response.usage.total_tokens or 0,
        )

    # --------------------------------------------------------------- comparison

    def _compare(
        self,
        ctx: ProbeContext,
        estimate: _Estimate,
        data: dict[str, Any],
        truncated: bool,
        charged: dict[str, Any],
    ) -> Evidence:
        """Weigh the measured interval against the published cutoff."""
        claimed = ctx.reference.knowledge_cutoff if ctx.reference is not None else None
        assert claimed is not None  # guaranteed by applicable()
        slack = dt.timedelta(days=SLACK_DAYS)
        incomplete = truncated and estimate.answered < 2
        status = EvidenceStatus.TRUNCATED if incomplete else EvidenceStatus.OK
        summary = (
            f"the endpoint answered {estimate.correct} of {estimate.answered} dated "
            f"questions, placing its knowledge boundary {estimate.text}, against a "
            f"published cutoff of {claimed.isoformat()} for "
            f"{ctx.provider.target_model!r}."
        )
        # A non-monotone ladder means the endpoint knew a later fact but not an
        # earlier one, which is normal near a boundary and corrosive to the
        # bracketing argument, so it halves whatever conclusion follows.
        confidence = 1.0 if estimate.monotonic else 0.5
        note = (
            ""
            if estimate.monotonic
            else " The rungs were not monotone -- a later fact was known while an earlier "
            "one was not -- so this is weighed at half strength."
        )

        if estimate.lower is not None and estimate.lower > claimed + slack:
            months = (estimate.lower - claimed).days / _DAYS_PER_MONTH
            fraction = _fraction(months, _LATE_SATURATION_MONTHS)
            return self._ev(
                "knowledge_boundary",
                -MODERATE * fraction * confidence,
                status=status,
                detail=(
                    summary
                    + " The endpoint knows events from after the claimed model stopped "
                    "training, which that model cannot. A retrieval-augmented proxy would "
                    "look the same from outside, which is why this is capped at moderate."
                    + note
                ),
                data={**data, "direction": "later", "months_from_claimed": round(months, 2)},
                **charged,
            )

        if estimate.upper is not None and estimate.upper < claimed - slack:
            months = (claimed - estimate.upper).days / _DAYS_PER_MONTH
            fraction = _fraction(months, _EARLY_SATURATION_MONTHS)
            return self._ev(
                "knowledge_boundary",
                -MODERATE * fraction * confidence,
                status=status,
                detail=(
                    summary
                    + " The endpoint does not know events the claimed model was trained "
                    "through, which is what serving an older or cheaper model looks like. "
                    "Models are also unreliable narrators near their own boundary, so this "
                    "is capped at moderate."
                    + note
                ),
                data={**data, "direction": "earlier", "months_from_claimed": round(months, 2)},
                **charged,
            )

        return self._ev(
            "knowledge_boundary",
            0.5 * MODERATE * confidence,
            status=status,
            detail=(
                summary
                + " The measured interval contains the published cutoff. The ladder is "
                "sparse, so a wide interval agrees with the claim easily; this supports it "
                "without proving much."
                + note
            ),
            data={**data, "direction": "consistent"},
            **charged,
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
            # No observation of a model's own account of recent events may
            # exceed a 10:1 likelihood ratio, whichever way it points.
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
# Ladder selection
# --------------------------------------------------------------------------- #


def _relevant_events(ctx: ProbeContext) -> tuple[tuple[_Event, ...], list[str]]:
    """The ladder with rungs about the model under test removed."""
    identities = {ctx.provider.model, ctx.provider.target_model}
    if ctx.reference is not None:
        identities.add(ctx.reference.id)
        identities.update(ctx.reference.aliases)
    stems = {_canonical(name) for name in identities if name}

    kept: list[_Event] = []
    dropped: list[str] = []
    for event in LADDER:
        if any(_canonical(subject) in stems for subject in event.subjects):
            dropped.append(event.key)
        else:
            kept.append(event)
    return tuple(sorted(kept, key=lambda event: event.date)), dropped


def _canonical(name: str) -> str:
    """Reduce an identifier to the stem two namings of one model share.

    Strips an OpenRouter variant suffix and author prefix, then the dotted
    region and vendor segments a Bedrock or Vertex identifier carries
    (``us.anthropic.claude-...``), then any trailing snapshot date.
    """
    ident = name.strip().lower().split(":", 1)[0].rsplit("/", 1)[-1]
    for _ in range(3):
        stripped = re.sub(r"^[a-z0-9]+\.", "", ident)
        if stripped == ident:
            break
        ident = stripped
    while True:
        stripped = _VERSION_SUFFIX.sub("", ident)
        if stripped == ident:
            return ident
        ident = stripped


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def _extract(text: str) -> str | None:
    """Pull the final answer out of a reply, or ``None`` if there is not one."""
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        answer = matches[-1].strip().strip("*`\"'.")
        return answer or None
    for line in reversed((text or "").strip().splitlines()):
        stripped = line.strip().strip("*`\"'. ")
        if stripped:
            return stripped
    return None


def _normalise(text: str) -> str:
    """Fold an answer to the form the accepted-answer sets are written in."""
    folded = text.strip().casefold()
    folded = re.sub(r"[\s_/]+", "-", folded)
    folded = re.sub(r"[^a-z0-9.\-]", "", folded)
    return re.sub(r"-{2,}", "-", folded).strip("-")


def _grade(extracted: str, answers: tuple[str, ...]) -> bool:
    """Exact match against a small accepted set, after normalisation.

    Containment rather than equality, so that a model which answers with the
    identifier plus a snapshot suffix or a trailing full stop still passes. The
    accepted strings are specific enough -- full model identifiers, a year and
    month -- that containment cannot be satisfied by accident.
    """
    normalised = _normalise(extracted)
    if not normalised or normalised == "unknown":
        return False
    return any(answer in normalised for answer in answers)


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #


def _estimate(outcomes: list[_Outcome]) -> _Estimate:
    """Bracket the boundary from the rungs that produced an answer.

    Only answered rungs constrain anything. The lower bound is the latest date
    known, the upper bound the earliest date not known after it; either may be
    absent, which is the honest report when every rung fell on one side.
    """
    answered = [outcome for outcome in outcomes if outcome.answered]
    estimate = _Estimate(
        answered=len(answered),
        correct=sum(1 for outcome in answered if outcome.correct),
        unanswered=[outcome.key for outcome in outcomes if not outcome.answered],
    )
    if not answered:
        return estimate

    known = [outcome.date for outcome in answered if outcome.correct]
    unknown = [outcome.date for outcome in answered if not outcome.correct]

    estimate.lower = max(known) if known else None
    later_unknown = [date for date in unknown if estimate.lower is None or date > estimate.lower]
    estimate.upper = min(later_unknown) if later_unknown else None
    estimate.monotonic = not (known and unknown and min(unknown) < max(known))
    return estimate


def _fraction(months: float, saturation: float) -> float:
    """Scale a disagreement in months onto ``[_MINIMUM_FRACTION, 1]``."""
    return max(_MINIMUM_FRACTION, min(1.0, months / saturation))
