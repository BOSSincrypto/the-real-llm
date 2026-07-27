"""Repeat one prompt and measure how much the answer moves.

**Drift is expected. It is not evidence of substitution.** This has to be said
first because the opposite assumption is the single most common way an amateur
verifier accuses an honest provider.

A forward pass at a fixed batch composition is deterministic. Production
inference is not, because dynamic batching means your result depends on how many
other requests happened to be batched alongside it: floating-point addition is
not associative, and GPU reduction kernels pick their tiling from the runtime
batch shape, so the same matrix multiply returns different low-order bits
depending on how much company your request had. Every hosted provider disclaims
bitwise determinism, and Anthropic states plainly that even at temperature 0 the
results will not be fully deterministic. So variability here is a measurement of
someone else's load, not of which weights ran.

What follows for the weighting:

* High variability contributes an LLR of essentially zero. It is reported
  because a human reading the report wants the number, not because it argues
  either way.
* Perfect bitwise stability is very weak *positive* evidence at most, and only
  for an indirect reason: batch-invariant kernels are opt-in and cost 25-55% of
  throughput, which is not an expense a reseller cutting corners on weights
  would choose to pay. The ``determinism`` family caps at WEAK regardless.
* What is worth reporting with real weight is a *degenerate* kind of sameness:
  every sample byte-identical and suspiciously short, or repeat latency
  collapsing toward zero. That is the signature of a cache or a lookup table
  answering instead of a model, and it is filed under ``misc`` rather than
  ``determinism`` precisely so it is not damped along with the drift
  measurement, which is measuring something else entirely.

Output-length variance is recorded alongside, because a genuinely stochastic
decoder and a fixed response differ in length distribution even when their
average text looks similar.

Samples are sent sequentially rather than concurrently. Concurrent identical
requests tend to land in one batch, which is exactly the condition under which
outputs agree for reasons that have nothing to do with the model.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, ClassVar

from ..config import redact
from ..errors import BudgetExhausted, ProviderError
from ..evidence import MODERATE, WEAK, Evidence, EvidenceStatus
from ..types import ChatRequest, ChatResponse, FinishReason, Message, Role
from . import Probe, ProbeContext, register_probe

__all__ = ["DeterminismProbe"]

#: Long enough that agreement is informative, short enough that the pairwise
#: edit distance stays cheap in pure Python.
_PROMPT = "Describe a lighthouse at dawn in exactly two sentences."

#: Beyond this the quadratic edit distance stops being worth its runtime, and
#: the leading characters already carry the signal.
_EDIT_DISTANCE_LIMIT = 4000

#: A completion this short, repeated byte for byte, is not two sentences.
_DEGENERATE_TOKENS = 4
_DEGENERATE_CHARS = 24


@register_probe
class DeterminismProbe(Probe):
    """Measure repeat-to-repeat agreement, and say plainly what it does not mean."""

    name: ClassVar[str] = "determinism"
    layer: ClassVar[int] = 1
    family: ClassVar[str] = "determinism"
    order: ClassVar[int] = 40
    estimated_requests: ClassVar[int] = 5
    description: ClassVar[str] = (
        "Repeat agreement at temperature 0 with a pinned seed, reported as "
        "information; cached-response and lookup signatures reported as evidence."
    )

    samples: ClassVar[int] = 5
    max_tokens: ClassVar[int] = 160

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        responses, configuration, failure = await self._collect(ctx)
        elapsed = time.perf_counter() - started

        if failure is not None:
            return [failure]
        if len(responses) < 2:
            return [
                self._ev(
                    "output_stability",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=(
                        f"only {len(responses)} sample(s) completed, which is not enough "
                        "to measure agreement."
                    ),
                    duration_s=elapsed,
                )
            ]

        texts = [response.text for response in responses]
        metrics = _measure(responses, texts)
        metrics["request_configuration"] = configuration
        metrics["samples"] = len(responses)

        cost = sum(
            ctx.budget.charge(r.usage.input_tokens, r.usage.output_tokens) for r in responses
        )
        tokens = sum(r.usage.total_tokens or 0 for r in responses)

        evidence = [self._stability(metrics, cost, tokens, elapsed)]
        evidence.append(self._degenerate(metrics))
        evidence.append(self._length_variance(metrics))
        return evidence

    # -------------------------------------------------------------- collection

    async def _collect(
        self, ctx: ProbeContext
    ) -> tuple[list[ChatResponse], str, Evidence | None]:
        """Send the same prompt ``samples`` times, sequentially.

        The first call also settles which request shape the endpoint tolerates.
        Reasoning models commonly refuse an explicit temperature, and no
        protocol but the OpenAI-compatible one defines a seed, so a rejection is
        answered by dropping the offending field rather than by failing the
        probe -- and which shape survived is recorded, since it is informative
        in itself.
        """
        wanted = min(self.samples, ctx.budget.remaining_samples or self.samples)
        if wanted < 2:
            return [], "", self._ev(
                "output_stability",
                0.0,
                status=EvidenceStatus.TRUNCATED,
                detail="the sample budget left room for fewer than two repeats.",
            )

        responses: list[ChatResponse] = []
        configuration = ""
        last_error: Exception | None = None

        for label, fields in _configurations(ctx.run.seed):
            request = ChatRequest(
                messages=(Message(Role.USER, _PROMPT),),
                max_tokens=self.max_tokens,
                **fields,
            )
            response, error = await ctx.adapter.try_chat(request)
            if response is not None:
                responses.append(response)
                configuration = label
                break
            last_error = error
            if not _is_client_error(error):
                break

        if not responses:
            return [], "", self._ev(
                "output_stability",
                0.0,
                status=EvidenceStatus.ERROR,
                detail=f"no sample completed: {redact(str(last_error))[:300]}",
            )

        fields = dict(_configuration_fields(ctx.run.seed, configuration))
        for _ in range(wanted - 1):
            try:
                ctx.budget.check()
            except BudgetExhausted:
                break
            request = ChatRequest(
                messages=(Message(Role.USER, _PROMPT),),
                max_tokens=self.max_tokens,
                **fields,
            )
            response, _error = await ctx.adapter.try_chat(request)
            if response is not None:
                responses.append(response)

        return responses, configuration, None

    # ------------------------------------------------------------ evidence

    def _stability(
        self, metrics: dict[str, Any], cost: float, tokens: int, elapsed: float
    ) -> Evidence:
        if metrics["all_empty"]:
            return self._ev(
                "output_stability",
                0.0,
                status=EvidenceStatus.UNSUPPORTED,
                detail=(
                    "every sample returned empty text, so there is no output to compare. "
                    "This is what a reasoning model does when the token allowance is "
                    "spent before it starts writing."
                ),
                data=metrics,
                cost_usd=cost,
                tokens=tokens,
                duration_s=elapsed,
            )

        rate = metrics["exact_match_rate"]
        distance = metrics["mean_normalised_edit_distance"]

        if metrics["all_identical"] and not metrics["degenerate"]:
            return self._ev(
                "output_stability",
                0.25 * WEAK,
                detail=(
                    f"all {metrics['samples']} samples were byte-identical at "
                    f"{metrics['request_configuration']}. Weakly supportive, and only "
                    "indirectly: batch-invariant kernels are opt-in and cost 25-55% of "
                    "throughput, which is not a bill a reseller cutting corners pays."
                ),
                data=metrics,
                cost_usd=cost,
                tokens=tokens,
                duration_s=elapsed,
            )

        return self._ev(
            "output_stability",
            0.0,
            detail=(
                f"{rate:.0%} of sample pairs were identical, mean normalised edit "
                f"distance {distance:.3f}. Reported as information only: drift under a "
                "fixed seed at temperature 0 is expected, because dynamic batching makes "
                "a result depend on what else was in the batch."
            ),
            data=metrics,
            cost_usd=cost,
            tokens=tokens,
            duration_s=elapsed,
        )

    def _degenerate(self, metrics: dict[str, Any]) -> Evidence:
        """Report the cached-or-canned signature, separately from drift.

        Filed under ``misc`` rather than ``determinism`` on purpose. The
        determinism family is damped to WEAK because everything in it measures
        the same expected-to-be-noisy quantity; "this endpoint is not running
        inference for my request" is a different observation and would be
        wrongly muted by that damping.
        """
        if metrics["all_empty"] or metrics["all_truncated"]:
            return self._ev(
                "cached_response_signature",
                0.0,
                family="misc",
                status=EvidenceStatus.UNSUPPORTED,
                detail=(
                    "every sample hit the token ceiling, so short identical outputs "
                    "cannot be told apart from truncation."
                ),
                data=metrics,
            )

        degenerate = metrics["degenerate"]
        collapsed = metrics["latency_collapsed"]

        if degenerate and collapsed:
            return self._ev(
                "cached_response_signature",
                -MODERATE,
                cap=MODERATE,
                family="misc",
                detail=(
                    f"every sample was byte-identical and only "
                    f"{metrics['median_chars']:.0f} characters long, and repeat latency "
                    f"collapsed from {metrics['first_latency_s']:.2f}s to "
                    f"{metrics['median_repeat_latency_s']:.2f}s. Both together are what a "
                    "cache or a canned response looks like, not a model generating text."
                ),
                data=metrics,
            )
        if degenerate:
            return self._ev(
                "cached_response_signature",
                -WEAK,
                cap=MODERATE,
                family="misc",
                detail=(
                    f"every sample was byte-identical and only "
                    f"{metrics['median_chars']:.0f} characters long, for a prompt asking "
                    "for two sentences. Latency did not collapse, so this may be a very "
                    "terse model rather than a canned answer."
                ),
                data=metrics,
            )
        if collapsed:
            return self._ev(
                "cached_response_signature",
                -WEAK,
                cap=MODERATE,
                family="misc",
                detail=(
                    f"repeat latency collapsed from {metrics['first_latency_s']:.2f}s to "
                    f"{metrics['median_repeat_latency_s']:.2f}s. Something is answering "
                    "repeats faster than generation allows, most likely a response cache."
                ),
                data=metrics,
            )
        return self._ev(
            "cached_response_signature",
            0.0,
            family="misc",
            detail="no cached-response or canned-answer signature.",
            data=metrics,
        )

    def _length_variance(self, metrics: dict[str, Any]) -> Evidence:
        return self._ev(
            "output_length_variance",
            0.0,
            detail=(
                f"output length {metrics['mean_chars']:.0f} +/- {metrics['stdev_chars']:.1f} "
                f"characters across {metrics['samples']} samples"
                + (
                    f", {metrics['mean_output_tokens']:.0f} +/- "
                    f"{metrics['stdev_output_tokens']:.1f} tokens."
                    if metrics["mean_output_tokens"] is not None
                    else "."
                )
                + " Recorded to separate a stochastic decoder from a lookup; no weight "
                "either way."
            ),
            data=metrics,
        )

    # ------------------------------------------------------------------ helpers

    def _ev(
        self,
        label: str,
        llr: float,
        *,
        cap: float = WEAK,
        family: str | None = None,
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
            family=family or self.family,
            status=status,
            detail=detail,
            data=data or {},
            cost_usd=cost_usd,
            tokens=tokens,
            duration_s=duration_s,
        )


# --------------------------------------------------------------------------- #
# Request shapes
# --------------------------------------------------------------------------- #


def _configurations(seed: int) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Request shapes to try, most pinned first."""
    return (
        ("temperature 0 with a pinned seed", {"temperature": 0.0, "seed": seed}),
        ("temperature 0, no seed", {"temperature": 0.0}),
        ("provider defaults", {}),
    )


def _configuration_fields(seed: int, label: str) -> dict[str, Any]:
    for candidate, fields in _configurations(seed):
        if candidate == label:
            return fields
    return {}


def _is_client_error(error: Exception | None) -> bool:
    """Whether an error is the endpoint refusing the request, not the network failing."""
    status = getattr(error, "status", None)
    return isinstance(error, ProviderError) and isinstance(status, int) and 400 <= status < 500


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _measure(responses: list[ChatResponse], texts: list[str]) -> dict[str, Any]:
    """Agreement, length and latency statistics for one set of repeats."""
    pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1 :]]
    identical = sum(1 for a, b in pairs if a == b)
    distances = [_normalised_edit_distance(a, b) for a, b in pairs]

    lengths = [len(text) for text in texts]
    output_tokens = [
        r.usage.output_tokens for r in responses if r.usage.output_tokens is not None
    ]
    latencies = [r.timing.total_s for r in responses]
    repeats = latencies[1:]
    median_repeat = statistics.median(repeats) if repeats else latencies[0]

    all_identical = len(set(texts)) == 1
    all_empty = all(not text.strip() for text in texts)
    all_truncated = all(r.finish_reason is FinishReason.LENGTH for r in responses)
    median_chars = statistics.median(len(text.strip()) for text in texts)
    median_tokens = statistics.median(output_tokens) if output_tokens else None

    short = median_chars <= _DEGENERATE_CHARS or (
        median_tokens is not None and median_tokens <= _DEGENERATE_TOKENS
    )

    return {
        "exact_match_rate": identical / len(pairs) if pairs else 0.0,
        "mean_normalised_edit_distance": statistics.fmean(distances) if distances else 0.0,
        "max_normalised_edit_distance": max(distances) if distances else 0.0,
        "distinct_outputs": len(set(texts)),
        "all_identical": all_identical,
        "all_empty": all_empty,
        "all_truncated": all_truncated,
        "degenerate": all_identical and short and not all_empty,
        "mean_chars": statistics.fmean(lengths),
        "stdev_chars": statistics.pstdev(lengths),
        "median_chars": median_chars,
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else None,
        "stdev_output_tokens": statistics.pstdev(output_tokens) if output_tokens else None,
        "first_latency_s": latencies[0],
        "median_repeat_latency_s": median_repeat,
        # A repeat answered in a small fraction of the first call's time, and in
        # a time too short to have generated the tokens, is not generation.
        "latency_collapsed": bool(
            repeats and median_repeat < 0.25 * latencies[0] and median_repeat < 0.1
        ),
    }


def _normalised_edit_distance(left: str, right: str) -> float:
    """Levenshtein distance divided by the longer string's length.

    Normalising makes the number comparable across prompts and models: a
    ten-character difference means something quite different in a tweet and in
    an essay.
    """
    a, b = left[:_EDIT_DISTANCE_LIMIT], right[:_EDIT_DISTANCE_LIMIT]
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))
