"""Accuracy against published reference scores, with sequential early stopping.

This is the probe that compares an endpoint against what the official provider
of the claimed model reports. It is also the probe most likely to produce a
confident answer to a question it cannot actually settle, so most of what
follows is about not doing that.

**Benchmark choice decides everything, and the famous ones are useless.** GPQA
Diamond cannot separate 2026 frontier models. The published scores cluster
between 87 and 93, and separating Claude Opus 5 (91.8) from GPT-5.6 Sol (91.3)
at 95% confidence and 80% power needs about 48,500 items per arm against the 198
that exist -- an impossibility, not a large expense. SimpleQA Verified, whose
published scores span 67.7 points, and ARC-AGI-2, which spans 87.6, separate the
same pair with a few dozen items and a handful respectively. So benchmarks are
ranked by information per request using
:attr:`~llmverify.benchmarks.base.Benchmark.score_spread` and
:func:`~llmverify.stats.power.discriminative_power`, and when the only reference
score available for the claimed model sits on a saturated benchmark the user is
told so in as many words. A non-significant result there means the benchmark is
too coarse, never that the endpoint is genuine.

**Conditions, not just numbers.** A published score is meaningless without the
settings it was measured at. Reasoning effort alone moves scores by tens of
points -- DeepSeek reports V4-Pro at 90.1 on GPQA Diamond in Think-Max mode and
72.9 in Non-Think -- so when the provider pins an effort the reference score was
not measured at, the comparison is invalid as stated. Rather than refuse
outright, the probe widens its tolerance by
:data:`EFFORT_MISMATCH_ALLOWANCE_PP` and says why in the report. Scores
published under benchmark-optimised settings get a further
:data:`OPTIMIZED_ALLOWANCE_PP`. Both allowances are judgement calls, not
measurements, and are labelled as such in the evidence data.

Where several sources publish a score for the same model under the same
conditions they disagree -- Opus 5 on ARC-AGI-2 is 90.4 by ARC Prize and 88.3 by
Epoch AI -- so the null is set at the *lowest* published value in the matching
range. Disagreement between sources widens the benefit of the doubt instead of
silently becoming this tool's own bias.

**Sequential testing.** The null is that the endpoint achieves the reference
accuracy; the alternative that it is worse by
:attr:`~llmverify.config.BudgetConfig.min_effect_pp`. Items are drawn without
replacement from a seeded shuffle and fed to a :class:`~llmverify.stats.SPRT`
one at a time, which settles a blatant substitution in a few dozen questions and
keeps sampling when the difference is subtle. A test that hits the sample, cost
or wall-clock ceiling before crossing a boundary has not reached a decision:
that is reported as ``TRUNCATED``, never as whichever boundary happened to be
nearer, because forcing a decision destroys the error guarantees the whole
design exists to provide.

**Two kinds of wrong.** An item the grader could not extract an answer from is
not an item answered incorrectly. Extraction failures are excluded from the SPRT
and reported separately with their own rate, because a provider mangling the
output format is failing in a different way from one reasoning badly -- and
because the reference score was measured by a harness whose extractor worked.
Exclusion is not free: if an endpoint mangles output only on the items it finds
hard, excluding those inflates the observed accuracy. That is why the rate is
reported as evidence in its own right rather than quietly dropped.

**Anti-evasion.** Under :attr:`~llmverify.config.RunConfig.anti_evasion` the
items alternate between verbatim and paraphrased renderings, and the variant
used for each item is recorded in the per-item results left in
``ctx.shared["benchmark_results"]`` for the evasion probe and the report.
Interleaving costs a little: paraphrasing perturbs difficulty slightly, so a
half-paraphrased run is measured against a reference score that was not, which
the allowances above absorb.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..benchmarks.base import (
    Benchmark,
    BenchmarkItem,
    GradedResult,
    Variant,
    all_benchmarks,
    get_benchmark,
)
from ..benchmarks.datasets import DatasetLoader
from ..config import redact
from ..errors import ConfigError, DatasetError, LLMVerifyError
from ..evidence import DECISIVE, MODERATE, STRONG, WEAK, Evidence, EvidenceStatus
from ..stats.intervals import wilson_interval
from ..stats.power import discriminative_power
from ..stats.sequential import SPRT, SPRTDecision
from ..stats.tests import mcnemar
from ..types import ChatRequest, FinishReason
from . import Probe, ProbeContext, register_probe

if TYPE_CHECKING:
    from ..adapters.base import Adapter
    from ..reference.schema import BenchmarkScore

__all__ = [
    "EFFORT_MISMATCH_ALLOWANCE_PP",
    "MAX_ITEMS_PER_BENCHMARK",
    "OPTIMIZED_ALLOWANCE_PP",
    "BenchmarkProbe",
    "dataset_loader",
]

#: Percentage points of slack added when the reference score's reasoning effort
#: is not the one the provider pins, or is not stated at all. A judgement call
#: anchored on a published example of the size of the gap: DeepSeek V4-Pro
#: scores 90.1 on GPQA Diamond in Think-Max and 72.9 in Non-Think, a 17-point
#: swing from effort alone. Not a measured constant for any other model.
EFFORT_MISMATCH_ALLOWANCE_PP: float = 15.0

#: Further slack when the published score was measured with benchmark-optimised
#: settings, which this probe does not reproduce. Also a judgement call.
OPTIMIZED_ALLOWANCE_PP: float = 3.0

#: Ceiling on items per benchmark regardless of budget. The SPRT decides long
#: before this on any difference worth acting on; past it the run is buying
#: precision about a difference too small to be interpretable anyway.
MAX_ITEMS_PER_BENCHMARK: int = 150

#: Fewest items worth starting a benchmark with at all. A run that can only
#: afford a handful should say so rather than buy a measurement it cannot use.
MIN_GRADED_ITEMS: int = 12

#: Fewest graded items a sequential decision is honoured on. Wald's guarantee
#: holds at any stopping time, and the whole point of the design is that a
#: blatant substitution settles quickly, so this is deliberately low -- it
#: guards only against a decision resting on so few items that one mis-graded
#: answer would flip it. Truncation is still reported as inconclusive.
DECISION_FLOOR: int = 8

#: Benchmarks run when the user named none. Two independent benchmarks are worth
#: more than twice as many items of one, but each additional one costs a full
#: sequential test's worth of budget.
MAX_AUTO_BENCHMARKS: int = 2

#: Fraction of the run's cost and wall-clock ceilings this probe may consume,
#: leaving room for the evasion probe that follows it.
BUDGET_SHARE: float = 0.6

#: Consecutive request failures after which the benchmark is abandoned. A
#: provider that has started refusing will keep refusing, and each attempt
#: spends the clock.
CONSECUTIVE_ERROR_LIMIT: int = 3

#: Response text kept per item. Enough for a human to see how an answer was
#: graded, bounded so that a 150-item run does not carry a megabyte of prose
#: into the JSON report.
RAW_RESPONSE_CHARS: int = 2000

#: Extraction-failure rate above which the endpoint's output formatting is
#: reported as a finding rather than as noise.
EXTRACTION_ALARM_RATE: float = 0.20


def dataset_loader(ctx: ProbeContext) -> tuple[DatasetLoader, bool]:
    """Return a loader for this run, and whether the caller owns it.

    The runner may put a shared loader in ``ctx.shared["dataset_loader"]`` so
    that several probes reuse one connection pool; when it has, the caller must
    not close it. Otherwise a private loader is created and the caller closes it.
    """
    existing = ctx.shared.get("dataset_loader")
    if isinstance(existing, DatasetLoader):
        return existing, False

    token = os.environ.get(ctx.run.hf_token_env)
    return (
        DatasetLoader(ctx.run.cache_dir, hf_token=token, token_env=ctx.run.hf_token_env),
        True,
    )


@dataclass(slots=True)
class _Plan:
    """One benchmark, its reference score, and what that score can support."""

    benchmark: Benchmark
    score: BenchmarkScore | None = None
    #: Lowest published score under matching conditions, in percent.
    conservative_score: float | None = None
    power: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_reference(self) -> bool:
        return self.score is not None


@dataclass(slots=True)
class _Tally:
    """What one benchmark run produced."""

    results: list[GradedResult] = field(default_factory=list)
    requests: int = 0
    request_errors: int = 0
    extraction_failures: int = 0
    ceiling_truncations: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    stopped_for: str = ""

    @property
    def graded(self) -> int:
        return sum(1 for result in self.results if result.correct is not None)


@register_probe
class BenchmarkProbe(Probe):
    """Sequential accuracy test against published reference scores."""

    name: ClassVar[str] = "benchmark"
    layer: ClassVar[int] = 3
    family: ClassVar[str] = "benchmark"
    order: ClassVar[int] = 120
    estimated_requests: ClassVar[int] = 60
    description: ClassVar[str] = (
        "Benchmark accuracy against the claimed model's published score, tested "
        "sequentially so an obvious substitution stops after a few dozen items."
    )

    #: Room for a reasoning model to think and still reach its answer line.
    #: Responses that hit this ceiling are counted apart from ones that finished
    #: without a parseable answer -- the ceiling is ours, the mangling is theirs.
    max_tokens: ClassVar[int] = 2048

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        plans, failure = self._select(ctx)
        if failure is not None:
            return [failure]

        loader, owned = dataset_loader(ctx)
        evidence: list[Evidence] = []
        try:
            budget_n = self._items_per_benchmark(ctx, len(plans))
            for plan in plans:
                if self._must_stop(ctx):
                    break
                evidence.extend(await self._measure(ctx, plan, loader, budget_n))
        finally:
            if owned:
                await loader.aclose()

        if not evidence:
            return [
                ctx.skipped(
                    self.name,
                    "no benchmark could be loaded within the run's budget.",
                    family=self.family,
                )
            ]
        return evidence

    # ------------------------------------------------------------------ planning

    def _select(self, ctx: ProbeContext) -> tuple[list[_Plan], Evidence | None]:
        """Choose benchmarks and work out what each one's reference score supports."""
        try:
            candidates = self._candidates(ctx)
        except ConfigError as exc:
            return [], ctx.skipped(self.name, str(exc), family=self.family)

        if not candidates:
            return [], ctx.skipped(
                self.name,
                "no benchmark is available that this tool can grade without a code sandbox.",
                family=self.family,
            )

        plans = [self._plan(ctx, benchmark) for benchmark in candidates]
        with_reference = [plan for plan in plans if plan.has_reference]

        if not with_reference and not ctx.has_baseline:
            model = ctx.provider.target_model
            return [], ctx.skipped(
                self.name,
                (
                    f"the reference snapshot publishes no benchmark score for {model!r} on "
                    f"any benchmark this tool can run, and no baseline endpoint was "
                    "configured, so there is nothing to compare an accuracy against. "
                    "Supply --baseline to measure a first-party endpoint side by side."
                ),
                family=self.family,
            )

        # A benchmark with a published score is always preferred: comparing
        # against the vendor's own number is the question the tool was asked.
        # The baseline A/B path is a fallback for when no benchmark has one.
        chosen = with_reference or plans
        if not ctx.run.benchmarks:
            chosen = sorted(chosen, key=_rank_key)[:MAX_AUTO_BENCHMARKS]

        if with_reference and all(not plan.benchmark.discriminative for plan in with_reference):
            warning = (
                f"The only published score(s) for {ctx.provider.target_model!r} that this "
                f"tool can reproduce are on saturated benchmark(s) "
                f"({', '.join(plan.benchmark.name for plan in with_reference)}), where every "
                "current frontier model scores within a few points of every other. A result "
                "consistent with the reference there rules out a badly degraded endpoint and "
                "nothing more."
            )
            for plan in chosen:
                plan.notes.append(warning)
            _warn(ctx, warning)

        return chosen, None

    def _candidates(self, ctx: ProbeContext) -> list[Benchmark]:
        """Instantiate the benchmarks this run may use."""
        if ctx.run.benchmarks:
            return [get_benchmark(name) for name in ctx.run.benchmarks]
        return [
            cls()
            for cls in all_benchmarks().values()
            # Grading a sandboxed benchmark without a sandbox would score every
            # item wrong and read as a catastrophic substitution.
            if not cls.needs_sandbox
        ]

    def _plan(self, ctx: ProbeContext, benchmark: Benchmark) -> _Plan:
        """Attach the reference score and the discriminative-power summary."""
        plan = _Plan(benchmark=benchmark)
        if ctx.reference is None:
            return plan

        effort = ctx.provider.reasoning_effort
        plan.score = ctx.reference.score_for(benchmark.reference_key, effort=effort, tools=False)
        span = ctx.reference.score_range_for(benchmark.reference_key, effort=effort, tools=False)
        if span is not None:
            low, high, count = span
            plan.conservative_score = low
            if count > 1 and high - low > 0.05:
                plan.notes.append(
                    f"{count} sources publish this score under matching conditions, spanning "
                    f"{low:.1f} to {high:.1f}; the test uses the lowest."
                )

        published = [
            score.score
            for record in (ctx.snapshot.models if ctx.snapshot is not None else ())
            for score in record.scores
            if score.benchmark == benchmark.reference_key
        ]
        if len(published) >= 2:
            plan.power = discriminative_power(
                published,
                alpha=ctx.run.budget.alpha,
                power=1.0 - ctx.run.budget.beta,
                available_items=self._items_per_benchmark(ctx, 1),
            )
            if plan.power.get("warning"):
                plan.notes.append(str(plan.power["warning"]))
        return plan

    def _items_per_benchmark(self, ctx: ProbeContext, benchmarks: int) -> int:
        """How many items one benchmark may spend, given the run's ceilings."""
        remaining = ctx.budget.remaining_samples
        allowance = MAX_ITEMS_PER_BENCHMARK
        if remaining is not None:
            allowance = min(allowance, int(remaining * BUDGET_SHARE))
        return max(0, allowance // max(1, benchmarks))

    # --------------------------------------------------------------- measurement

    async def _measure(
        self, ctx: ProbeContext, plan: _Plan, loader: DatasetLoader, budget_n: int
    ) -> list[Evidence]:
        """Load, sample, grade and test one benchmark."""
        started = time.perf_counter()
        benchmark = plan.benchmark

        if budget_n < MIN_GRADED_ITEMS:
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=(
                        f"the sample budget left room for {budget_n} items of "
                        f"{benchmark.name}, below the {MIN_GRADED_ITEMS} this probe will "
                        "start a benchmark with."
                    ),
                    data={"benchmark": benchmark.name, "items_affordable": budget_n},
                )
            ]

        try:
            # Load more than will be used so that the seeded draw is a real
            # sample of the split rather than its first N rows in file order.
            items = await benchmark.load(loader, limit=min(600, max(60, budget_n * 3)))
        except (DatasetError, LLMVerifyError) as exc:
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    0.0,
                    status=EvidenceStatus.ERROR,
                    detail=f"{benchmark.name} could not be loaded: {redact(str(exc))[:300]}",
                    data={"benchmark": benchmark.name},
                )
            ]

        if len(items) < MIN_GRADED_ITEMS:
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        f"{benchmark.name} yielded only {len(items)} items, fewer than the "
                        f"{MIN_GRADED_ITEMS} needed for any conclusion."
                    ),
                    data={"benchmark": benchmark.name, "items_available": len(items)},
                )
            ]

        rng = ctx.rng(f"benchmark:{benchmark.name}")
        wanted = min(budget_n, len(items))
        chosen = rng.sample(items, wanted)

        if not plan.has_reference:
            return await self._baseline_ab(ctx, plan, chosen, started)

        p0, p1, hypothesis = self._hypotheses(ctx, plan)
        sprt = SPRT(p0=p0, p1=p1, alpha=ctx.run.budget.alpha, beta=ctx.run.budget.beta)
        tally = await self._grade(ctx, ctx.adapter, plan, chosen, sprt=sprt)

        _record(ctx, benchmark.name, tally.results)
        elapsed = time.perf_counter() - started
        return [
            self._accuracy_evidence(ctx, plan, sprt, tally, hypothesis, elapsed),
            self._extraction_evidence(plan, tally),
        ]

    def _hypotheses(self, ctx: ProbeContext, plan: _Plan) -> tuple[float, float, dict[str, Any]]:
        """Set the null and the alternative, widening for incomparable conditions."""
        score = plan.score
        assert score is not None  # guaranteed by the caller
        base = plan.conservative_score if plan.conservative_score is not None else score.score

        pinned = ctx.provider.reasoning_effort
        allowance = 0.0
        reasons: list[str] = []
        if score.effort is None:
            allowance += EFFORT_MISMATCH_ALLOWANCE_PP
            reasons.append(
                "the published score does not state the reasoning effort it was measured at"
            )
        elif pinned is None:
            allowance += EFFORT_MISMATCH_ALLOWANCE_PP
            reasons.append(
                f"the published score was measured at effort {score.effort!r} but this "
                "provider config pins no effort, so the endpoint's default is unknown"
            )
        elif pinned != score.effort:
            allowance += EFFORT_MISMATCH_ALLOWANCE_PP
            reasons.append(
                f"the published score was measured at effort {score.effort!r} and this run "
                f"pins {pinned!r}; comparing across efforts is not valid"
            )
        if score.optimized:
            allowance += OPTIMIZED_ALLOWANCE_PP
            reasons.append("the published score used benchmark-optimised settings")
        if ctx.run.anti_evasion:
            reasons.append(
                "half the items are paraphrased, which the reference score's items were not"
            )

        effective = max(2.0, min(99.0, base - allowance))
        p0 = effective / 100.0
        effect = ctx.run.budget.min_effect_pp / 100.0
        p1 = p0 - effect
        clamped = False
        if p1 <= 0.0:
            # Nothing below zero accuracy exists, so a reference score smaller
            # than the effect size gets a proportional alternative instead.
            p1 = p0 / 2.0
            clamped = True

        return (
            p0,
            p1,
            {
                "reference_score_pp": score.score,
                "reference_score_used_pp": base,
                "reference_source": score.source,
                "reference_as_of": score.as_of.isoformat(),
                "reference_confidence": score.confidence,
                "reference_effort": score.effort,
                "reference_optimized": score.optimized,
                "provider_effort": pinned,
                "tolerance_allowance_pp": allowance,
                "tolerance_allowance_is_a_judgement_call": allowance > 0.0,
                "tolerance_reasons": reasons,
                "null_accuracy": round(p0, 4),
                "alternative_accuracy": round(p1, 4),
                "alternative_clamped": clamped,
                "min_effect_pp": ctx.run.budget.min_effect_pp,
                "alpha": ctx.run.budget.alpha,
                "beta": ctx.run.budget.beta,
            },
        )

    async def _grade(
        self,
        ctx: ProbeContext,
        adapter: Adapter,
        plan: _Plan,
        items: list[BenchmarkItem],
        *,
        sprt: SPRT | None,
        variant_override: Variant | None = None,
    ) -> _Tally:
        """Ask one endpoint every item in order, updating the test as it goes."""
        benchmark = plan.benchmark
        tally = _Tally()
        consecutive_errors = 0

        for index, item in enumerate(items):
            if self._must_stop(ctx):
                tally.stopped_for = "budget"
                break

            variant = variant_override or self._variant(ctx, index)
            render_rng = ctx.rng(f"render:{benchmark.name}:{item.id}:{variant.value}")
            messages, state = benchmark.render(item, variant=variant, rng=render_rng)

            request = ChatRequest(
                messages=messages,
                max_tokens=self.max_tokens,
                reasoning_effort=ctx.provider.reasoning_effort,
                thinking_budget=ctx.provider.thinking_budget,
            )
            started = time.perf_counter()
            response, error = await adapter.try_chat(request)
            duration = time.perf_counter() - started
            tally.requests += 1

            if response is None:
                consecutive_errors += 1
                tally.request_errors += 1
                ctx.budget.charge(None, None)
                tally.results.append(
                    GradedResult(
                        item_id=item.id,
                        variant=variant,
                        correct=None,
                        raw_response="",
                        expected=item.answer,
                        duration_s=duration,
                        error=redact(str(error))[:200],
                    )
                )
                if consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
                    tally.stopped_for = "repeated request failures"
                    break
                continue

            consecutive_errors = 0
            tally.cost_usd += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            tally.tokens += response.usage.total_tokens or 0

            correct, extracted = benchmark.grade(item, response.text, state)
            at_ceiling = response.finish_reason is FinishReason.LENGTH
            if correct is None:
                if at_ceiling:
                    tally.ceiling_truncations += 1
                else:
                    tally.extraction_failures += 1

            tally.results.append(
                GradedResult(
                    item_id=item.id,
                    variant=variant,
                    correct=correct,
                    raw_response=response.text[:RAW_RESPONSE_CHARS],
                    extracted=extracted,
                    expected=item.answer,
                    input_tokens=response.usage.input_tokens or 0,
                    output_tokens=response.usage.output_tokens or 0,
                    duration_s=duration,
                    error=(
                        "no answer line before the output token ceiling" if at_ceiling else None
                    ),
                )
            )

            if (
                sprt is not None
                and correct is not None
                and sprt.update(correct) is not SPRTDecision.CONTINUE
            ):
                tally.stopped_for = "sequential decision"
                break

        return tally

    def _variant(self, ctx: ProbeContext, index: int) -> Variant:
        """Alternate renderings so both arms are balanced whatever ``n`` ends up being."""
        if not ctx.run.anti_evasion:
            return Variant.VERBATIM
        return Variant.VERBATIM if index % 2 == 0 else Variant.PARAPHRASED

    def _must_stop(self, ctx: ProbeContext) -> bool:
        """Whether this probe has spent its share of the run."""
        budget = ctx.budget
        if budget.exhausted:
            return True
        cost_cap, wall_cap = budget.max_cost_usd, budget.max_wall_s
        if cost_cap is not None and budget.spent_usd >= cost_cap * BUDGET_SHARE:
            return True
        return wall_cap is not None and budget.elapsed_s >= wall_cap * BUDGET_SHARE

    # ------------------------------------------------------------------ evidence

    def _accuracy_evidence(
        self,
        ctx: ProbeContext,
        plan: _Plan,
        sprt: SPRT,
        tally: _Tally,
        hypothesis: dict[str, Any],
        elapsed: float,
    ) -> Evidence:
        """The headline: what the SPRT concluded, or that it did not conclude."""
        benchmark = plan.benchmark
        low, high = wilson_interval(sprt.successes, sprt.n) if sprt.n else (0.0, 0.0)
        variants = _variant_counts(tally.results)
        data: dict[str, Any] = {
            "benchmark": benchmark.name,
            "reference_key": benchmark.reference_key,
            "discriminative": benchmark.discriminative,
            "score_spread_pp": benchmark.score_spread,
            "n_graded": sprt.n,
            "successes": sprt.successes,
            "observed_accuracy_pp": round(sprt.rate * 100.0, 2),
            "wilson_95_pp": [round(low * 100.0, 2), round(high * 100.0, 2)],
            "sprt_llr_nats": round(sprt.llr, 4),
            "sprt_decision": sprt.decision.value,
            "sprt_upper_bound": round(sprt.upper_bound, 4),
            "sprt_lower_bound": round(sprt.lower_bound, 4),
            "expected_n_h0": round(sprt.expected_n_h0, 1),
            "expected_n_h1": round(sprt.expected_n_h1, 1),
            "requests": tally.requests,
            "request_errors": tally.request_errors,
            "estimated_cost_usd": round(tally.cost_usd, 6),
            "variants": variants,
            "stopped_for": tally.stopped_for or "items exhausted",
            "notes": plan.notes,
            **hypothesis,
        }
        if plan.power is not None:
            data["discriminative_power"] = {
                key: value
                for key, value in plan.power.items()
                if key in ("closest_gap_pp", "n_to_separate", "sufficient", "spread", "sd")
            }
        charged = {
            "cost_usd": tally.cost_usd,
            "tokens": tally.tokens,
            "duration_s": elapsed,
        }
        suffix = (" " + " ".join(plan.notes)) if plan.notes else ""

        if sprt.n < DECISION_FLOOR or sprt.decision is SPRTDecision.CONTINUE:
            return self._ev(
                f"{benchmark.name}_accuracy",
                0.0,
                status=EvidenceStatus.TRUNCATED,
                detail=(
                    f"{sprt.successes}/{sprt.n} correct on {benchmark.name} "
                    f"({sprt.rate:.1%}, Wilson 95% {low:.1%}-{high:.1%}) against a reference "
                    f"of {hypothesis['reference_score_used_pp']:.1f}%. The sequential test "
                    f"stopped at a log-likelihood ratio of {sprt.llr:.2f}, inside its "
                    f"boundaries [{sprt.lower_bound:.2f}, {sprt.upper_bound:.2f}], because "
                    f"the run ran out of {tally.stopped_for or 'items'}. A truncated "
                    "sequential test is inconclusive and contributes nothing; forcing it to "
                    "the nearer boundary would void its error rates." + suffix
                ),
                data=data,
                **charged,
            )

        # The SPRT statistic is already a log-likelihood ratio between exactly
        # the two hypotheses in question, signed in favour of the degraded one,
        # so the evidence LLR is its negation and needs no further calibration.
        llr = -sprt.llr
        if sprt.decision is SPRTDecision.ACCEPT_H1:
            detail = (
                f"{sprt.successes}/{sprt.n} correct on {benchmark.name} ({sprt.rate:.1%}, "
                f"Wilson 95% {low:.1%}-{high:.1%}) against a reference of "
                f"{hypothesis['reference_score_used_pp']:.1f}% for "
                f"{ctx.provider.target_model!r}. The sequential test crossed its "
                f"lower-accuracy boundary at n={sprt.n} (LLR {sprt.llr:.2f} >= "
                f"{sprt.upper_bound:.2f}), so the endpoint is performing at least "
                f"{ctx.run.budget.min_effect_pp:.0f} points below the published score at "
                f"alpha={ctx.run.budget.alpha}." + suffix
            )
        else:
            detail = (
                f"{sprt.successes}/{sprt.n} correct on {benchmark.name} ({sprt.rate:.1%}, "
                f"Wilson 95% {low:.1%}-{high:.1%}), consistent with the published "
                f"{hypothesis['reference_score_used_pp']:.1f}% for "
                f"{ctx.provider.target_model!r}. The sequential test crossed its "
                f"reference-accuracy boundary at n={sprt.n} (LLR {sprt.llr:.2f} <= "
                f"{sprt.lower_bound:.2f}). This rules out a degradation of "
                f"{ctx.run.budget.min_effect_pp:.0f} points or more; it does not rule out a "
                "smaller one." + suffix
            )
            if plan.power is not None and plan.power.get("sufficient") is False:
                detail += (
                    " Read with the discriminative-power warning above: on this benchmark a "
                    "consistent result cannot separate the claimed model from its nearest "
                    "published neighbour."
                )
        return self._ev(
            f"{benchmark.name}_accuracy",
            llr,
            cap=DECISIVE,
            detail=detail,
            data=data,
            **charged,
        )

    def _extraction_evidence(self, plan: _Plan, tally: _Tally) -> Evidence:
        """Report unparseable answers apart from wrong ones."""
        attempted = tally.graded + tally.extraction_failures
        rate = tally.extraction_failures / attempted if attempted else 0.0
        data = {
            "benchmark": plan.benchmark.name,
            "graded": tally.graded,
            "extraction_failures": tally.extraction_failures,
            "extraction_failure_rate": round(rate, 4),
            "output_ceiling_truncations": tally.ceiling_truncations,
            "request_errors": tally.request_errors,
            "max_tokens": self.max_tokens,
        }

        if attempted < MIN_GRADED_ITEMS:
            return self._ev(
                f"{plan.benchmark.name}_answer_extraction",
                0.0,
                status=EvidenceStatus.TRUNCATED,
                detail=(
                    f"only {attempted} responses reached the grader, too few to say whether "
                    "the endpoint formats answers reliably."
                ),
                data=data,
            )

        if rate >= EXTRACTION_ALARM_RATE:
            severity = min(1.0, (rate - EXTRACTION_ALARM_RATE) / (1.0 - EXTRACTION_ALARM_RATE))
            return self._ev(
                f"{plan.benchmark.name}_answer_extraction",
                -(WEAK + (MODERATE - WEAK) * severity),
                cap=STRONG,
                detail=(
                    f"{tally.extraction_failures} of {attempted} responses finished without "
                    f"an answer in the requested format ({rate:.0%}), separately from "
                    f"{tally.ceiling_truncations} that hit the {self.max_tokens}-token "
                    "ceiling. Failing to follow a one-line output instruction is a different "
                    "failure from answering wrongly, and the published reference score was "
                    "measured by a harness whose extractor worked. These items are excluded "
                    "from the accuracy test rather than counted as wrong."
                ),
                data=data,
            )

        return self._ev(
            f"{plan.benchmark.name}_answer_extraction",
            0.0,
            detail=(
                f"{tally.extraction_failures} of {attempted} responses ({rate:.0%}) had no "
                f"extractable answer, plus {tally.ceiling_truncations} that hit the output "
                "token ceiling. Within the range a normal harness sees; excluded from the "
                "accuracy test rather than counted as wrong."
            ),
            data=data,
        )

    # ------------------------------------------------------------------ baseline

    async def _baseline_ab(
        self, ctx: ProbeContext, plan: _Plan, items: list[BenchmarkItem], started: float
    ) -> list[Evidence]:
        """Compare the two endpoints item by item when no reference score exists.

        Both endpoints see the same items in the same renderings, so the
        comparison is paired and :func:`~llmverify.stats.tests.mcnemar` is the
        right test: it conditions on the items where the two disagreed, which is
        where all the information about a difference lives.
        """
        benchmark = plan.benchmark
        if ctx.baseline is None:
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    0.0,
                    status=EvidenceStatus.SKIPPED,
                    detail=(
                        f"no published score for {ctx.provider.target_model!r} on "
                        f"{benchmark.name} and no baseline endpoint to measure instead."
                    ),
                    data={"benchmark": benchmark.name},
                )
            ]

        paired = items[: max(MIN_GRADED_ITEMS, len(items) // 2)]
        candidate = await self._grade(
            ctx, ctx.adapter, plan, paired, sprt=None, variant_override=Variant.VERBATIM
        )
        reference = await self._grade(
            ctx, ctx.baseline, plan, paired, sprt=None, variant_override=Variant.VERBATIM
        )
        _record(ctx, benchmark.name, candidate.results)

        left = {r.item_id: r.correct for r in candidate.results if r.correct is not None}
        right = {r.item_id: r.correct for r in reference.results if r.correct is not None}
        shared_ids = sorted(set(left) & set(right))
        elapsed = time.perf_counter() - started
        cost = candidate.cost_usd + reference.cost_usd
        charged = {
            "cost_usd": cost,
            "tokens": candidate.tokens + reference.tokens,
            "duration_s": elapsed,
        }

        if len(shared_ids) < MIN_GRADED_ITEMS:
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=(
                        f"only {len(shared_ids)} items were graded on both endpoints, too "
                        "few for a paired comparison."
                    ),
                    data={"benchmark": benchmark.name, "paired_items": len(shared_ids)},
                    **charged,
                )
            ]

        b = sum(1 for key in shared_ids if right[key] and not left[key])
        c = sum(1 for key in shared_ids if left[key] and not right[key])
        p_value = mcnemar(b, c)
        candidate_correct = sum(1 for key in shared_ids if left[key])
        baseline_correct = sum(1 for key in shared_ids if right[key])
        gap_pp = (baseline_correct - candidate_correct) / len(shared_ids) * 100.0

        data = {
            "benchmark": benchmark.name,
            "paired_items": len(shared_ids),
            "candidate_correct": candidate_correct,
            "baseline_correct": baseline_correct,
            "gap_pp": round(gap_pp, 2),
            "discordant_baseline_only": b,
            "discordant_candidate_only": c,
            "test": "exact McNemar on paired items",
            "p_value": p_value,
            "baseline_provider": ctx.baseline_provider.name if ctx.baseline_provider else None,
            "estimated_cost_usd": round(cost, 6),
            "notes": plan.notes,
        }

        if p_value < 0.05 and gap_pp > 0.0:
            severity = min(1.0, gap_pp / 30.0)
            return [
                self._ev(
                    f"{benchmark.name}_accuracy",
                    -(MODERATE + (STRONG - MODERATE) * severity),
                    cap=STRONG,
                    detail=(
                        f"on {len(shared_ids)} shared {benchmark.name} items the candidate "
                        f"answered {candidate_correct} correctly against the baseline's "
                        f"{baseline_correct}, a gap of {gap_pp:.1f} points (exact McNemar "
                        f"p={p_value:.3g} on {b}/{c} discordant pairs). No published score "
                        "was available, so this is a measured side-by-side rather than a "
                        "comparison against the vendor's own number."
                    ),
                    data=data,
                    **charged,
                )
            ]

        return [
            self._ev(
                f"{benchmark.name}_accuracy",
                MODERATE if p_value >= 0.05 else 0.0,
                cap=STRONG,
                detail=(
                    f"on {len(shared_ids)} shared {benchmark.name} items the candidate "
                    f"answered {candidate_correct} correctly against the baseline's "
                    f"{baseline_correct} (exact McNemar p={p_value:.3g}). The two endpoints "
                    "are not distinguishable at this sample size."
                ),
                data=data,
                **charged,
            )
        ]

    # ------------------------------------------------------------------- helpers

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
# Module helpers
# --------------------------------------------------------------------------- #


def _rank_key(plan: _Plan) -> tuple[Any, ...]:
    """Order benchmarks by information per request.

    A benchmark whose published scores are spread out separates models with
    fewer items, which is the whole of "information per request" here: a
    reference score means nothing without the ability to fall visibly short of
    it. Gated datasets sort last among equals because they fail entirely without
    an accepted licence and a token.
    """
    benchmark = plan.benchmark
    n_needed = math.inf
    if plan.power is not None:
        n_needed = float(plan.power.get("n_to_separate", math.inf))
    return (
        0 if plan.has_reference else 1,
        0 if benchmark.discriminative else 1,
        0 if not benchmark.gated else 1,
        -benchmark.score_spread,
        n_needed,
        benchmark.name,
    )


def _variant_counts(results: list[GradedResult]) -> dict[str, dict[str, int]]:
    """Per-variant totals, so the evasion probe's arms are visible in this report too."""
    counts: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = counts.setdefault(
            result.variant.value, {"items": 0, "correct": 0, "ungraded": 0}
        )
        bucket["items"] += 1
        if result.correct is None:
            bucket["ungraded"] += 1
        elif result.correct:
            bucket["correct"] += 1
    return counts


def _record(ctx: ProbeContext, benchmark: str, results: list[GradedResult]) -> None:
    """Publish per-item results for the evasion probe and the HTML report."""
    stored = ctx.shared.setdefault("benchmark_results", [])
    if isinstance(stored, list):
        stored.extend(results)
    names = ctx.shared.setdefault("benchmark_names", [])
    if isinstance(names, list) and benchmark not in names:
        names.append(benchmark)


def _warn(ctx: ProbeContext, message: str) -> None:
    """Add a run-level warning for the runner to surface outside the evidence list."""
    warnings = ctx.shared.setdefault("warnings", [])
    if isinstance(warnings, list) and message not in warnings:
        warnings.append(message)
