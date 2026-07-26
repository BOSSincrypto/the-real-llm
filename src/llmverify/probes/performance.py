"""Throughput, latency and price sanity -- suggestive, never probative.

This probe exists because the economics of a substitution are often visible
before the behaviour is. A reseller running a 30B model behind a flagship name
generates faster and charges less than the real thing, and both numbers are
published on the front page.

**It is capped at WEAK and it cannot move a verdict on its own, by
construction.** Every innocent explanation for these numbers is at least as
plausible as the guilty one. Throughput far above a model's first-party figure
is what better hardware looks like, what an under-subscribed cluster looks like,
what a newer serving stack looks like, and what a shorter output looks like. A
price far below the official one is what a loss leader looks like, what
committed-capacity pricing looks like, and what a reseller with a different cost
base looks like. First-party throughput figures also move without announcement
as vendors change their own infrastructure, which is a documented reason to
expect drift from an endpoint that is entirely genuine.

So the individual measurements are reported at zero LLR. Only the *joint*
observation -- throughput far above the claimed model's typical rate together
with a price far below its official one -- carries any weight at all, and the
``performance`` family caps at WEAK, which is roughly a 3:1 likelihood ratio.
Three such probes could not reach a verdict between them, which is the intended
property.

**Measurement.** At least :data:`SAMPLES` streamed generations of a fixed
prompt, sequentially -- concurrent samples contend for the same batch and
measure the endpoint's queueing rather than its speed. Time-to-first-token comes
from the transport layer at the first frame carrying content, so provider ping
and role frames do not flatter it. Output rate is tokens after the first token
divided by the time spent generating them. Median and interquartile range are
reported rather than a mean, because one slow sample from a cold route
otherwise dominates a five-sample average.

Where the endpoint streams no usage object, tokens per second cannot be computed
without a tokenizer and a characters-per-second figure is reported instead,
labelled as what it is.

**Cost per correct answer** is computed when benchmark results are present in
``ctx.shared`` under ``benchmark_summary`` (a mapping with ``correct``,
``graded`` and ``cost_usd``). It is an economics figure for a human reading the
report, not evidence, and it is reported at zero LLR.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted
from ..evidence import WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["PerformanceProbe"]

#: Fixed generation. Enumerating a range produces an output whose length barely
#: varies between models, which is what makes rates comparable at all; asking
#: for prose would measure verbosity as much as speed.
PROMPT: str = (
    "Write the integers from 1 to 120 in ascending order, separated by single "
    "spaces, on one line. Output nothing else."
)

#: Streamed samples. Five is the floor for a median and an interquartile range
#: to mean anything; more would buy precision this probe is capped too low to use.
SAMPLES: int = 5

MAX_TOKENS: int = 400

#: Throughput this many times the claimed model's typical first-party rate counts
#: as "far above". Set high deliberately: hardware and serving stacks differ by
#: less than this, and the point is to catch a different model, not a better GPU.
THROUGHPUT_FACTOR: float = 2.5

#: Advertised output price at or below this fraction of the official one counts
#: as "far below".
PRICE_FRACTION: float = 0.5


@dataclass(slots=True)
class _Sample:
    """One streamed generation."""

    ttft_s: float | None = None
    total_s: float = 0.0
    output_tokens: int | None = None
    chars: int = 0
    error: str | None = None

    @property
    def tokens_per_s(self) -> float | None:
        if not self.output_tokens:
            return None
        generating = self.total_s - (self.ttft_s or 0.0)
        return self.output_tokens / generating if generating > 0 else None

    @property
    def chars_per_s(self) -> float | None:
        generating = self.total_s - (self.ttft_s or 0.0)
        return self.chars / generating if generating > 0 and self.chars else None


@dataclass(slots=True)
class _Series:
    """Median and spread of one measured quantity."""

    values: list[float] = field(default_factory=list)

    @property
    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    @property
    def iqr(self) -> tuple[float, float] | None:
        """Interquartile range, or ``None`` when there are too few values.

        The inclusive method is used because these samples are the population
        being described, not a draw from a larger one.
        """
        if len(self.values) < 2:
            return None
        q1, _, q3 = statistics.quantiles(self.values, n=4, method="inclusive")
        return q1, q3

    def as_dict(self) -> dict[str, Any]:
        spread = self.iqr
        return {
            "n": len(self.values),
            "median": round(self.median, 3) if self.median is not None else None,
            "iqr_low": round(spread[0], 3) if spread else None,
            "iqr_high": round(spread[1], 3) if spread else None,
            "values": [round(v, 3) for v in self.values],
        }


@register_probe
class PerformanceProbe(Probe):
    """Streamed throughput and latency, against published rates and prices."""

    name: ClassVar[str] = "performance"
    layer: ClassVar[int] = 2
    family: ClassVar[str] = "performance"
    order: ClassVar[int] = 100
    estimated_requests: ClassVar[int] = SAMPLES
    description: ClassVar[str] = (
        "Time-to-first-token and output tokens per second over streamed samples, "
        "with advertised pricing, reported as suggestive economics rather than proof."
    )

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        samples, truncated, cost, tokens = await self._collect(ctx)
        elapsed = time.perf_counter() - started

        answered = [s for s in samples if s.error is None]
        ttft = _Series([s.ttft_s for s in answered if s.ttft_s is not None])
        tps = _Series([v for s in answered if (v := s.tokens_per_s) is not None])
        cps = _Series([v for s in answered if (v := s.chars_per_s) is not None])

        data: dict[str, Any] = {
            "samples_requested": SAMPLES,
            "samples_answered": len(answered),
            "streamed": bool(ctx.adapter.capabilities.streaming),
            "ttft_s": ttft.as_dict(),
            "output_tokens_per_s": tps.as_dict(),
            "chars_per_s": cps.as_dict(),
            "errors": [s.error for s in samples if s.error][:5],
        }
        charged = {"cost_usd": cost, "tokens": tokens, "duration_s": elapsed}

        if not answered:
            return [
                self._ev(
                    "throughput",
                    0.0,
                    status=EvidenceStatus.TRUNCATED if truncated else EvidenceStatus.ERROR,
                    detail="no streamed sample completed, so nothing was measured.",
                    data=data,
                    **charged,
                )
            ]

        evidence = [self._throughput(ctx, tps, cps, ttft, data, truncated, charged)]
        pricing = self._pricing(ctx, data)
        if pricing is not None:
            evidence.append(pricing)
        joint = self._joint(ctx, tps, data)
        if joint is not None:
            evidence.append(joint)
        efficiency = self._cost_per_correct(ctx)
        if efficiency is not None:
            evidence.append(efficiency)
        return evidence

    # ------------------------------------------------------------- measurement

    async def _collect(
        self, ctx: ProbeContext
    ) -> tuple[list[_Sample], bool, float, int]:
        """Stream :data:`SAMPLES` generations, one at a time."""
        samples: list[_Sample] = []
        cost = 0.0
        tokens = 0
        stream = bool(ctx.adapter.capabilities.streaming)

        for _ in range(SAMPLES):
            try:
                ctx.budget.check()
            except BudgetExhausted:
                return samples, True, cost, tokens

            response, error = await ctx.adapter.try_chat(
                ChatRequest(
                    messages=(Message(Role.USER, PROMPT),),
                    max_tokens=MAX_TOKENS,
                    stream=stream,
                )
            )
            if response is None:
                ctx.budget.charge(None, None)
                samples.append(_Sample(error=redact(str(error))[:200]))
                continue

            cost += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            tokens += response.usage.total_tokens or 0
            samples.append(
                _Sample(
                    ttft_s=response.timing.ttft_s,
                    total_s=response.timing.total_s,
                    output_tokens=response.usage.output_tokens,
                    chars=len(response.text or ""),
                )
            )
        return samples, False, cost, tokens

    # ------------------------------------------------------------ interpretation

    def _throughput(
        self,
        ctx: ProbeContext,
        tps: _Series,
        cps: _Series,
        ttft: _Series,
        data: dict[str, Any],
        truncated: bool,
        charged: dict[str, Any],
    ) -> Evidence:
        """Report the measured rates against the published one. Never weighed."""
        typical = ctx.reference.typical_output_tps if ctx.reference is not None else None
        measured = tps.median
        parts: list[str] = []
        if ttft.median is not None:
            spread = ttft.iqr
            window = f" (IQR {spread[0]:.2f}-{spread[1]:.2f}s)" if spread else ""
            parts.append(f"time to first token {ttft.median:.2f}s{window}")
        if measured is not None:
            spread = tps.iqr
            window = f" (IQR {spread[0]:.0f}-{spread[1]:.0f})" if spread else ""
            parts.append(f"{measured:.0f} output tokens/s{window}")
        elif cps.median is not None:
            parts.append(
                f"{cps.median:.0f} characters/s -- the endpoint streams no usage object, "
                "so a token rate cannot be computed without assuming a tokenizer"
            )

        detail = "; ".join(parts) or "no rate could be computed"
        if typical is not None and measured is not None:
            ratio = measured / typical if typical else None
            data["typical_output_tps"] = typical
            data["throughput_ratio"] = round(ratio, 3) if ratio else None
            detail += (
                f". The reference records {typical:.0f} tokens/s for "
                f"{ctx.provider.target_model!r} on first-party infrastructure, so this is "
                f"{ratio:.1f}x that figure"
            )
        detail += (
            ". Reported without weight: hardware, cluster load and serving-stack version "
            "move this number more than the choice of weights does, and vendors change "
            "their own infrastructure without announcing it."
        )

        incomplete = truncated and len(tps.values) < 2
        return self._ev(
            "throughput",
            0.0,
            status=EvidenceStatus.TRUNCATED if incomplete else EvidenceStatus.OK,
            detail=detail,
            data=data,
            **charged,
        )

    def _pricing(self, ctx: ProbeContext, data: dict[str, Any]) -> Evidence | None:
        """Compare advertised prices with the official ones. Never weighed."""
        reference = ctx.reference.pricing if ctx.reference is not None else None
        advertised_in = ctx.provider.price_in_per_mtok
        advertised_out = ctx.provider.price_out_per_mtok
        if reference is None or (advertised_in is None and advertised_out is None):
            return None

        ratios: dict[str, float] = {}
        for label, advertised, official in (
            ("input", advertised_in, reference.input_per_mtok),
            ("output", advertised_out, reference.output_per_mtok),
        ):
            if advertised is not None and official:
                ratios[label] = advertised / official

        if not ratios:
            return None
        summary = ", ".join(
            f"{label} at {ratio:.0%} of the official price" for label, ratio in ratios.items()
        )
        return self._ev(
            "advertised_pricing",
            0.0,
            detail=(
                f"the provider advertises {summary} for {ctx.provider.target_model!r} "
                f"(reference source: {reference.source or 'unstated'}). A low price is a "
                "commercial decision, not evidence: loss leaders, committed capacity and a "
                "different cost base all produce this on an entirely honest endpoint."
            ),
            data={**data, "price_ratios": {k: round(v, 4) for k, v in ratios.items()}},
        )

    def _joint(
        self, ctx: ProbeContext, tps: _Series, data: dict[str, Any]
    ) -> Evidence | None:
        """The one weighed observation: fast *and* cheap, at the same time."""
        typical = ctx.reference.typical_output_tps if ctx.reference is not None else None
        measured = tps.median
        reference = ctx.reference.pricing if ctx.reference is not None else None
        advertised_out = ctx.provider.price_out_per_mtok
        if not typical or measured is None or reference is None or advertised_out is None:
            return None
        official_out = reference.output_per_mtok
        if not official_out:
            return None

        throughput_ratio = measured / typical
        price_ratio = advertised_out / official_out
        if throughput_ratio < THROUGHPUT_FACTOR or price_ratio > PRICE_FRACTION:
            return None

        return self._ev(
            "economics_consistent_with_smaller_model",
            -0.6 * WEAK,
            detail=(
                f"the endpoint generates {throughput_ratio:.1f}x the tokens per second "
                f"recorded for {ctx.provider.target_model!r} on first-party infrastructure "
                f"while charging {price_ratio:.0%} of its official output price. Those two "
                "together are what serving a smaller or quantized model looks like -- and "
                "equally what better hardware, spare capacity or a loss leader looks like. "
                "It is weighed near the floor for that reason, and the performance family "
                "is capped so that this cannot reach a verdict however extreme it gets."
            ),
            data={
                **data,
                "throughput_ratio": round(throughput_ratio, 3),
                "price_ratio": round(price_ratio, 4),
                "throughput_factor_threshold": THROUGHPUT_FACTOR,
                "price_fraction_threshold": PRICE_FRACTION,
            },
        )

    def _cost_per_correct(self, ctx: ProbeContext) -> Evidence | None:
        """Dollars per correct benchmark answer, when a benchmark has run.

        Read from ``ctx.shared['benchmark_summary']``, a mapping carrying
        ``correct``, ``graded`` and ``cost_usd``. Absent or malformed, this is
        simply not reported: an economics figure is not worth guessing at.
        """
        summary = ctx.shared.get("benchmark_summary")
        if not isinstance(summary, dict):
            return None
        correct = summary.get("correct")
        graded = summary.get("graded")
        spent = summary.get("cost_usd")
        if not isinstance(correct, int) or not isinstance(graded, int) or graded <= 0:
            return None
        if not isinstance(spent, (int, float)) or spent <= 0:
            return None

        accuracy = correct / graded
        per_correct = spent / correct if correct else None
        return self._ev(
            "cost_per_correct_answer",
            0.0,
            detail=(
                f"the benchmark evidence in this run cost ${spent:.4f} for {correct} correct "
                f"answers out of {graded} ({accuracy:.1%}), or "
                + (f"${per_correct:.4f} per correct answer" if per_correct else "no answers")
                + ". An efficiency figure for the reader; it says nothing about identity."
            ),
            data={
                "benchmark_correct": correct,
                "benchmark_graded": graded,
                "benchmark_cost_usd": round(float(spent), 6),
                "cost_per_correct_usd": round(per_correct, 6) if per_correct else None,
            },
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
            # Hard-capped at the family ceiling: no measurement of speed or price
            # may exceed a 3:1 likelihood ratio, whatever its magnitude.
            cap=WEAK,
            family=self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )
