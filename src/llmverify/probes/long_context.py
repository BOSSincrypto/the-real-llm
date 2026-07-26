"""Measure the context window the endpoint actually serves, not the one it advertises.

A million-token context is the easiest specification in the industry to claim
and the most expensive to serve. An endpoint that truncates its input at 32k and
answers anyway looks perfectly healthy from every cheap probe in this package:
the model id echoes back, the usage object has the right shape, short prompts
come out fine. It only falls over when something is hidden past the point where
the truncation happens.

**The haystack.** Filler is numbered lines of words drawn at random from a fixed
vocabulary. That shape is deliberate. It is trivial to generate at megabyte
scale, it carries no meaning a model could reconstruct from priors, and it does
not compress into a summary the way natural prose does -- an endpoint that
silently summarises a long prompt before feeding it to a smaller model cannot
preserve a random line through that step. The needle is one line recording a
registry key for a numbered locker, and the key is random, so the only way to
answer is to have read that line.

**Three depths, and why the middle one matters.** The needle is planted at 10%,
50% and 90% of the way through the filler. A stack that silently keeps only the
last N tokens passes at 90% and fails at 10%, which is a signature no amount of
prompt engineering imitates. The middle is where lossy long-context attention
degrades first, so a model that is genuinely long-context capable but running at
a reduced KV budget fails there before it fails at either end.

**Three ways to fail, told apart.** The endpoint may refuse the request (an
explicit context-length error, which is honest and gives a hard ceiling), accept
it and quietly truncate (visible as ``usage.input_tokens`` far below what was
sent, or as a token count that stops rising when the prompt does), or accept it
and simply not retrieve. Each is reported separately because they implicate
different parts of the stack.

**Statistics are not needed here and are not used.** When an endpoint's window
is genuinely below its claim, retrieval collapses from near-certain to near-
impossible over one rung of the ladder. A few samples per cell settle that;
sequential testing would spend budget to sharpen a decision that is not close.

**Budget.** This probe can cost more than every other probe in the run combined:
one 1M-token rung at flagship input pricing is several dollars. It therefore
refuses to spend more than :data:`BUDGET_SHARE` of the run's cost ceiling,
degrades from a three-depth sweep to a single middle-depth probe when that is
all it can afford, refuses to send more than :data:`UNPRICED_TOKEN_CEILING`
tokens when no pricing is known to bound the spend at all, and reports
``TRUNCATED`` rather than overspending. A measured floor with an honest "we
could not afford to test higher" is worth more than an unaffordable certainty.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["DEPTHS", "LADDER_TOKENS", "LongContextProbe"]

#: Rungs of the ladder, in tokens. Rungs above the claimed context window are
#: never attempted: an endpoint refusing a prompt longer than the model accepts
#: is behaving correctly, and charging it for that would be a false accusation.
LADDER_TOKENS: tuple[int, ...] = (4_000, 16_000, 64_000, 128_000, 200_000, 400_000, 1_000_000)

#: Fractional positions of the needle within the filler.
DEPTHS: tuple[float, ...] = (0.10, 0.50, 0.90)

#: The depth used when the budget only allows one probe per rung. The middle is
#: where truncation and lossy attention show up first.
PRIMARY_DEPTH: float = 0.50

#: Fraction of the run's cost ceiling this probe may consume.
BUDGET_SHARE: float = 0.5

#: Fraction of the run's wall-clock ceiling this probe may consume. Long prompts
#: are slow as well as expensive, and a probe that eats the clock starves the
#: statistical layer that follows it.
WALL_SHARE: float = 0.5

#: Hard ceiling on total input tokens when the run has no pricing information,
#: since without prices the cost guard cannot bind at all.
UNPRICED_TOKEN_CEILING: int = 512_000

#: Characters per token assumed before the endpoint has told us its own ratio.
#: Numbered lines of short words sit near this on every tokenizer family, and it
#: is only a starting point: the first rung's ``usage.input_tokens`` replaces it
#: with the endpoint's measured ratio for every rung after.
CHARS_PER_TOKEN_PRIOR: float = 4.0

#: Reported input tokens below this fraction of the calibrated estimate means
#: the prompt did not arrive intact.
TRUNCATION_RATIO: float = 0.6

#: A shortfall of this many doublings between the claimed window and the
#: measured ceiling earns the full weight the cap allows.
_SHORTFALL_SATURATION: float = 3.0

#: Words per filler line. Enough that a line is a distinctive string, short
#: enough that line numbers stay a meaningful fraction of the text.
_WORDS_PER_LINE: int = 12

#: Fixed filler vocabulary. Common, short, and semantically unrelated, so that a
#: sequence drawn from it says nothing and predicts nothing.
_VOCABULARY: tuple[str, ...] = (
    "anchor", "basin", "cedar", "domain", "ember", "fabric", "gravel", "harbor",
    "ingot", "jasper", "kernel", "lantern", "marble", "nickel", "orchard", "pewter",
    "quartz", "ribbon", "saddle", "timber", "umber", "velvet", "walnut", "yarrow",
    "zephyr", "amber", "bramble", "copper", "dahlia", "elm", "flint", "granite",
    "hollow", "iris", "juniper", "kelp", "lichen", "meadow", "nectar", "opal",
    "pebble", "quill", "rattan", "slate", "thistle", "urchin", "vellum", "willow",
    "acorn", "birch", "cinder", "dune", "eddy", "fern", "gable", "heather",
    "inlet", "jetty", "knoll", "ledge", "mist", "nook", "oxide", "prairie",
)

#: Alphabet for the registry key. No characters that survive an OCR-style
#: confusion (0/O, 1/I/L), so a key that comes back wrong came back wrong.
_KEY_ALPHABET: str = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_KEY_LENGTH: int = 10

#: Substrings that mark a provider error as "this prompt is longer than I take"
#: rather than as an unrelated failure.
_CONTEXT_ERROR_MARKERS: tuple[str, ...] = (
    "context length",
    "context_length",
    "context window",
    "context_window",
    "maximum context",
    "max_tokens",
    "too long",
    "too many tokens",
    "prompt is too",
    "exceeds",
    "request too large",
    "payload too large",
)

_TIMEOUT_MARKERS: tuple[str, ...] = ("timeout", "timed out", "readtimeout", "connecttimeout")


@dataclass(slots=True)
class _Cell:
    """One (size, depth) measurement."""

    size: int
    depth: float
    found: bool | None
    estimated_input_tokens: int
    reported_input_tokens: int | None = None
    chars_sent: int = 0
    error: str | None = None
    error_kind: str = ""
    duration_s: float = 0.0

    @property
    def attempted(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "size_tokens": self.size,
            "depth": self.depth,
            "found": self.found,
            "estimated_input_tokens": self.estimated_input_tokens,
            "reported_input_tokens": self.reported_input_tokens,
            "error": self.error,
            "error_kind": self.error_kind,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass(slots=True)
class _Ladder:
    """Everything the climb produced."""

    cells: list[_Cell] = field(default_factory=list)
    #: Rungs that were skipped, and why.
    skipped: dict[int, str] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens: int = 0
    stopped_for_budget: bool = False

    def by_size(self, size: int) -> list[_Cell]:
        return [cell for cell in self.cells if cell.size == size]

    @property
    def verified(self) -> int | None:
        """Largest rung at which the needle was retrieved at least once."""
        passed = [cell.size for cell in self.cells if cell.found]
        return max(passed) if passed else None

    @property
    def failed_at(self) -> int | None:
        """Smallest rung at which every attempted depth failed to retrieve."""
        sizes = sorted({cell.size for cell in self.cells})
        for size in sizes:
            cells = self.by_size(size)
            if cells and all(cell.found is False or cell.error_kind == "context_limit"
                             for cell in cells):
                return size
        return None


@register_probe
class LongContextProbe(Probe):
    """Ladder search for the endpoint's real context ceiling, with needle retrieval."""

    name: ClassVar[str] = "long_context"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "long_context"
    order: ClassVar[int] = 70
    #: Three depths on each of the first few affordable rungs. The real number
    #: is decided by the budget at run time.
    estimated_requests: ClassVar[int] = 9
    description: ClassVar[str] = (
        "Needle-in-a-haystack retrieval at 10/50/90% depth over a ladder of prompt "
        "sizes, reporting the measured context ceiling and any silent truncation."
    )

    #: The answer is a ten-character key. Room for a reasoning model to think
    #: first, not room for an essay.
    max_tokens: ClassVar[int] = 256

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only when the claimed model has a published context window.

        The ladder is derived from that number, and without it there is nothing
        to compare a measured ceiling against -- which would make this the most
        expensive purely informational probe in the package.
        """
        return ctx.reference is not None and ctx.reference.context_window is not None

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        claimed = ctx.reference.context_window if ctx.reference is not None else None
        assert claimed is not None  # guaranteed by applicable()

        rungs = [size for size in LADDER_TOKENS if size <= claimed]
        if not rungs:
            return [
                self._ev(
                    "effective_context_window",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        f"the claimed context window of {claimed} tokens is below the "
                        f"smallest rung on the ladder ({LADDER_TOKENS[0]} tokens), so there "
                        "is nothing to measure."
                    ),
                )
            ]

        ladder = await self._climb(ctx, rungs, claimed)
        elapsed = time.perf_counter() - started
        ctx.shared["long_context_ladder"] = [cell.as_dict() for cell in ladder.cells]
        if ladder.verified is not None:
            ctx.shared["verified_context_tokens"] = ladder.verified

        data: dict[str, Any] = {
            "claimed_context_window": claimed,
            "rungs_planned": rungs,
            "rungs_skipped": {str(k): v for k, v in ladder.skipped.items()},
            "cells": [cell.as_dict() for cell in ladder.cells],
            "chars_per_token": round(_chars_per_token(ctx), 3),
            "verified_tokens": ladder.verified,
            "failed_at_tokens": ladder.failed_at,
        }
        charged = {
            "cost_usd": ladder.cost_usd,
            "tokens": ladder.tokens,
            "duration_s": elapsed,
        }

        if not ladder.cells:
            return [
                self._ev(
                    "effective_context_window",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=(
                        "no rung of the ladder was affordable within this probe's share of "
                        "the run budget, so the context window was not measured."
                    ),
                    data=data,
                    **charged,
                )
            ]

        evidence = [self._ceiling(ctx, ladder, claimed, data, charged)]
        depth_item = self._depth_profile(ladder, data)
        if depth_item is not None:
            evidence.append(depth_item)
        truncation_item = self._truncation(ctx, ladder, data)
        if truncation_item is not None:
            evidence.append(truncation_item)
        return evidence

    # ------------------------------------------------------------------- climb

    async def _climb(self, ctx: ProbeContext, rungs: list[int], claimed: int) -> _Ladder:
        """Work up the ladder, stopping at the first rung that fails outright.

        Ascending rather than bisecting: the cost of a rung grows with its size,
        so an ascending walk pays for the cheap information first and reaches a
        budget ceiling having already learned the most it could afford. A rung
        where every depth fails ends the climb, because everything above it
        costs more and can only fail too.
        """
        ladder = _Ladder()
        rng = ctx.rng("long_context")

        for size in rungs:
            depths = self._affordable_depths(ctx, size, ladder)
            if not depths:
                ladder.skipped[size] = "budget"
                ladder.stopped_for_budget = True
                break

            if len(depths) < len(DEPTHS):
                ladder.skipped[size] = "reduced to the middle depth by budget"

            rung_cells: list[_Cell] = []
            for depth in depths:
                try:
                    ctx.budget.check()
                except BudgetExhausted:
                    ladder.stopped_for_budget = True
                    break
                cell = await self._probe_cell(ctx, size, depth, rng, ladder)
                ladder.cells.append(cell)
                rung_cells.append(cell)
                if cell.error_kind == "context_limit":
                    # The endpoint has stated its own ceiling. Asking the other
                    # depths at this size would buy the same refusal twice.
                    break

            if ladder.stopped_for_budget:
                break
            if rung_cells and all(
                cell.found is False or cell.error_kind == "context_limit" for cell in rung_cells
            ):
                break
            if any(cell.error_kind == "timeout" for cell in rung_cells):
                # A stack that hangs on a long prompt will hang harder on a
                # longer one, and each hang costs the run its wall clock.
                ladder.skipped[size] = "the endpoint hung on this rung; climb abandoned"
                break

        return ladder

    async def _probe_cell(
        self,
        ctx: ProbeContext,
        size: int,
        depth: float,
        rng: random.Random,
        ladder: _Ladder,
    ) -> _Cell:
        """Build one haystack, ask for the needle, and grade the reply."""
        locker = f"{rng.randrange(1000, 10000)}"
        key = "".join(rng.choice(_KEY_ALPHABET) for _ in range(_KEY_LENGTH))
        prompt = _build_prompt(rng, tokens=size, chars_per_token=_chars_per_token(ctx),
                              locker=locker, key=key, depth=depth)
        estimated = int(len(prompt) / _chars_per_token(ctx))

        started = time.perf_counter()
        response, error = await ctx.adapter.try_chat(
            ChatRequest(
                messages=(Message(Role.USER, prompt),),
                max_tokens=self.max_tokens,
            )
        )
        duration = time.perf_counter() - started

        if response is None:
            kind, message = _classify_error(error)
            # An unanswered request still consumed the endpoint's attention and
            # possibly the caller's money; charging the sample keeps the run's
            # ceilings honest even though the token cost is unknown.
            ctx.budget.charge(None, None)
            return _Cell(
                size=size,
                depth=depth,
                found=None,
                estimated_input_tokens=estimated,
                chars_sent=len(prompt),
                error=message,
                error_kind=kind,
                duration_s=duration,
            )

        ladder.cost_usd += ctx.budget.charge(
            response.usage.input_tokens, response.usage.output_tokens
        )
        ladder.tokens += response.usage.total_tokens or 0
        _calibrate(ctx, prompt, response.usage.input_tokens, size)

        return _Cell(
            size=size,
            depth=depth,
            found=_retrieved(response.text, key),
            estimated_input_tokens=estimated,
            reported_input_tokens=response.usage.input_tokens,
            chars_sent=len(prompt),
            duration_s=duration,
        )

    # ------------------------------------------------------------------ budget

    def _affordable_depths(
        self, ctx: ProbeContext, size: int, ladder: _Ladder
    ) -> tuple[float, ...]:
        """Which depths this rung can be probed at, or ``()`` to stop climbing."""
        if ctx.budget.exhausted:
            return ()
        if not self._within_wall_clock(ctx):
            return ()

        remaining_samples = ctx.budget.remaining_samples
        if remaining_samples is not None and remaining_samples < 1:
            return ()

        sent_so_far = sum(cell.estimated_input_tokens for cell in ladder.cells)
        for depths in (DEPTHS, (PRIMARY_DEPTH,)):
            need = len(depths)
            if remaining_samples is not None and remaining_samples < need:
                continue
            if not self._within_spend(ctx, size * need, sent_so_far + size * need):
                continue
            return depths
        return ()

    def _within_spend(self, ctx: ProbeContext, input_tokens: int, cumulative: int) -> bool:
        """Whether sending ``input_tokens`` stays inside this probe's allowance."""
        budget = ctx.budget
        price = budget.price_in_per_mtok
        if price is None or budget.max_cost_usd is None:
            return cumulative <= UNPRICED_TOKEN_CEILING
        projected = input_tokens / 1_000_000 * price
        return budget.spent_usd + projected <= budget.max_cost_usd * BUDGET_SHARE

    def _within_wall_clock(self, ctx: ProbeContext) -> bool:
        if ctx.budget.max_wall_s is None:
            return True
        return ctx.budget.elapsed_s < ctx.budget.max_wall_s * WALL_SHARE

    # -------------------------------------------------------------- conclusions

    def _ceiling(
        self,
        ctx: ProbeContext,
        ladder: _Ladder,
        claimed: int,
        data: dict[str, Any],
        charged: dict[str, Any],
    ) -> Evidence:
        """The headline: what window the endpoint actually served."""
        verified = ladder.verified
        failed_at = ladder.failed_at
        top_rung = max((cell.size for cell in ladder.cells), default=0)

        if failed_at is not None:
            shortfall = math.log2(claimed / failed_at) if failed_at else _SHORTFALL_SATURATION
            fraction = max(0.25, min(1.0, shortfall / _SHORTFALL_SATURATION))
            floor = f"{verified} tokens" if verified is not None else "no rung at all"
            return self._ev(
                "effective_context_window",
                -STRONG * fraction,
                cap=STRONG,
                detail=(
                    f"retrieval succeeded up to {floor} and failed completely at "
                    f"{failed_at} tokens, against a claimed window of {claimed} tokens for "
                    f"{ctx.provider.target_model!r}. The effective window is roughly "
                    f"{2 ** shortfall:.0f}x smaller than advertised. Serving the claimed "
                    "model at its claimed window is the one thing this endpoint cannot be "
                    "doing."
                ),
                data={**data, "shortfall_doublings": round(shortfall, 2)},
                **charged,
            )

        if verified is not None and top_rung >= max(
            size for size in LADDER_TOKENS if size <= claimed
        ):
            return self._ev(
                "effective_context_window",
                MODERATE,
                cap=STRONG,
                detail=(
                    f"the needle was retrieved at {verified} tokens, the top of the ladder "
                    f"below the claimed {claimed}-token window. The endpoint really does "
                    "carry a prompt of that size and can find one random line inside it."
                ),
                data=data,
                **charged,
            )

        return self._ev(
            "effective_context_window",
            0.0,
            status=EvidenceStatus.TRUNCATED,
            detail=(
                f"retrieval was verified up to "
                f"{verified if verified is not None else 'no rung'} tokens and the climb "
                f"stopped there for budget, well below the claimed {claimed}-token window. "
                "That is a floor, not a ceiling: nothing here argues for or against the "
                "claim."
            ),
            data=data,
            **charged,
        )

    def _depth_profile(self, ladder: _Ladder, data: dict[str, Any]) -> Evidence | None:
        """Retrieval by depth, and the tail-truncation signature when it appears."""
        answered = [cell for cell in ladder.cells if cell.found is not None]
        if not answered:
            return None

        by_depth: dict[float, list[bool]] = {}
        for cell in answered:
            by_depth.setdefault(cell.depth, []).append(bool(cell.found))
        profile = {
            f"{depth:.0%}": {"attempts": len(results), "found": sum(results)}
            for depth, results in sorted(by_depth.items())
        }

        tail_only = [
            size
            for size in sorted({cell.size for cell in answered})
            if _found_at(answered, size, 0.90) is True and _found_at(answered, size, 0.10) is False
        ]
        middle_only_failures = [
            size
            for size in sorted({cell.size for cell in answered})
            if _found_at(answered, size, 0.50) is False
            and _found_at(answered, size, 0.90) is True
            and _found_at(answered, size, 0.10) is True
        ]
        data = {
            **data,
            "depth_profile": profile,
            "tail_only_sizes": tail_only,
            "middle_only_failure_sizes": middle_only_failures,
        }

        if tail_only:
            return self._ev(
                "needle_depth_profile",
                -MODERATE,
                cap=STRONG,
                detail=(
                    f"at {', '.join(str(s) for s in tail_only)} tokens the needle was found "
                    "at 90% depth but not at 10%. Retrieval that works only near the end of "
                    "the prompt is what keeping the last N tokens and discarding the rest "
                    "looks like from outside."
                ),
                data=data,
            )
        if middle_only_failures:
            return self._ev(
                "needle_depth_profile",
                -WEAK,
                detail=(
                    f"at {', '.join(str(s) for s in middle_only_failures)} tokens retrieval "
                    "succeeded at both ends of the prompt and failed in the middle. That is "
                    "the ordinary shape of lossy long-context attention rather than proof of "
                    "a different model, so it is weighed lightly."
                ),
                data=data,
            )
        return self._ev(
            "needle_depth_profile",
            0.0,
            detail=(
                "retrieval did not depend on where in the prompt the needle sat: "
                + ", ".join(
                    f"{depth} {value['found']}/{value['attempts']}"
                    for depth, value in profile.items()
                )
                + "."
            ),
            data=data,
        )

    def _truncation(
        self, ctx: ProbeContext, ladder: _Ladder, data: dict[str, Any]
    ) -> Evidence | None:
        """Whether the endpoint quietly shortened prompts it said it accepted."""
        measured = [
            cell
            for cell in ladder.cells
            if cell.reported_input_tokens is not None and cell.estimated_input_tokens > 0
        ]
        if not measured:
            return None

        short = [
            cell
            for cell in measured
            if (cell.reported_input_tokens or 0)
            < TRUNCATION_RATIO * cell.estimated_input_tokens
        ]
        # A count that stops rising while the prompt keeps growing is the same
        # finding seen from the other side, and it survives a bad chars-per-token
        # estimate, which the ratio test does not.
        by_size = sorted({(cell.size, cell.reported_input_tokens or 0) for cell in measured})
        plateau = [
            (small, large)
            for (small, small_tokens), (large, large_tokens) in pairwise(by_size)
            if large >= 2 * small and large_tokens < 1.2 * small_tokens
        ]

        if not short and not plateau:
            return None

        detail = (
            "the endpoint accepted long prompts but reports having read far fewer tokens "
            "than were sent"
        )
        if plateau:
            pairs = ", ".join(f"{a}->{b}" for a, b in plateau)
            detail += (
                f", and its reported input token count stops rising between rungs ({pairs}) "
                "even as the prompt doubles"
            )
        detail += (
            ". Silent truncation is a property of the serving configuration rather than of "
            "the weights, but an endpoint that cannot carry the claimed model's context is "
            "not serving the claimed model as advertised."
        )
        return self._ev(
            "silent_truncation",
            -MODERATE,
            cap=STRONG,
            detail=detail,
            data={
                **data,
                "short_cells": [cell.as_dict() for cell in short],
                "plateau_pairs": plateau,
                "truncation_ratio": TRUNCATION_RATIO,
            },
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
# Haystack construction
# --------------------------------------------------------------------------- #


def _build_prompt(
    rng: random.Random,
    *,
    tokens: int,
    chars_per_token: float,
    locker: str,
    key: str,
    depth: float,
) -> str:
    """Assemble preamble, filler with the needle planted, and the question.

    The question goes last because that is where a long-context API places a
    user's real question, and putting it first would let a truncating stack keep
    the instruction while dropping the evidence -- measuring our prompt layout
    rather than the endpoint.
    """
    preamble = (
        "Below is a machine-generated activity log. Read it, then answer the question "
        "that follows it.\n\n"
    )
    question = (
        f"\n\nThe log above contains exactly one line recording the registry key for "
        f"locker {locker}. Reply with that key alone, formatted exactly as "
        f'"ANSWER: <key>".'
    )
    target_chars = max(0, int(tokens * chars_per_token) - len(preamble) - len(question))

    lines: list[str] = []
    total = 0
    index = 1
    while total < target_chars:
        words = " ".join(rng.choice(_VOCABULARY) for _ in range(_WORDS_PER_LINE))
        line = f"{index:06d} {words}"
        lines.append(line)
        total += len(line) + 1
        index += 1

    needle = f"{len(lines) + 1:06d} MEMO the registry key for locker {locker} is {key}"
    position = min(len(lines), max(0, int(depth * len(lines))))
    lines.insert(position, needle)
    return preamble + "\n".join(lines) + question


def _chars_per_token(ctx: ProbeContext) -> float:
    value = ctx.shared.get("long_context_chars_per_token")
    return float(value) if isinstance(value, (int, float)) and value > 0 else CHARS_PER_TOKEN_PRIOR


def _calibrate(ctx: ProbeContext, prompt: str, input_tokens: int | None, size: int) -> None:
    """Replace the assumed chars-per-token ratio with the endpoint's own.

    Only the smallest rung calibrates. Above it a low token count is ambiguous
    between "this tokenizer is efficient" and "this prompt was truncated", and
    calibrating from a truncated measurement would hide the truncation from
    every rung that followed.
    """
    if input_tokens is None or input_tokens <= 0 or size != LADDER_TOKENS[0]:
        return
    if "long_context_chars_per_token" in ctx.shared:
        return
    ratio = len(prompt) / input_tokens
    if 1.0 <= ratio <= 12.0:
        ctx.shared["long_context_chars_per_token"] = ratio


# --------------------------------------------------------------------------- #
# Grading and error classification
# --------------------------------------------------------------------------- #


def _retrieved(text: str, key: str) -> bool:
    """Whether the reply contains the planted key.

    The whole reply is searched, not only the ``ANSWER:`` line: a model that
    quotes the needle line and then formats its answer badly did retrieve it,
    and grading that as a miss would blame the endpoint for a formatting habit.
    """
    folded = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return key.upper() in folded


def _classify_error(error: Exception | None) -> tuple[str, str]:
    """Label a failed request as a context-limit refusal, a hang, or neither."""
    message = redact(str(error))[:300] if error is not None else "request failed"
    haystack = message.lower()
    body = getattr(error, "body", None)
    if isinstance(body, str):
        haystack += " " + redact(body).lower()
    status = getattr(error, "status", None)

    if any(marker in haystack for marker in _TIMEOUT_MARKERS):
        return "timeout", message
    if any(marker in haystack for marker in _CONTEXT_ERROR_MARKERS):
        return "context_limit", message
    if status in (413, 422):
        return "context_limit", message
    return "other", message


def _found_at(cells: list[_Cell], size: int, depth: float) -> bool | None:
    for cell in cells:
        if cell.size == size and abs(cell.depth - depth) < 1e-9:
            return cell.found
    return None
