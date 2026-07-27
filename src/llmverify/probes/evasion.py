"""Detect an endpoint that answers differently when it recognises the question.

The cheapest dishonesty available to a provider is not serving a worse model all
the time. It is serving a worse model most of the time and routing anything that
looks like a benchmark item to the genuine one. That defeats every accuracy
measurement in this package at negligible cost, and it is invisible to a run
that only asks "how well did it score".

So this probe measures the routing itself. The same underlying questions are put
twice: once verbatim, as published and therefore string-matchable against any
corpus, and once paraphrased -- same question, same gold answer, no longer equal
or nearly equal to anything in a lookup table. An endpoint whose behaviour does
not depend on recognisability scores the same on both arms within sampling
noise. One that routes on recognition drops on the paraphrased arm, and the size
of that drop is a direct measurement of how much of the run's headline accuracy
was theatre.

**Where the data comes from.** When the benchmark probe has run, its per-item
results are already in ``ctx.shared["benchmark_results"]`` with the variant used
for each item recorded, and this probe simply reads them: the measurement is
free. Only when that record is missing or too small does it buy its own canary
split, defaulting to :data:`CANARY_ITEMS` items per arm and shrinking to fit the
remaining budget.

**Paired beats unpaired.** When the same items appear in both arms -- which is
what the canary split arranges deliberately -- McNemar's test conditions on the
items where the two arms disagreed, which is where all the information about a
difference lives. It is substantially more powerful than treating the arms as
independent samples, so it is preferred whenever the pairing exists;
:func:`~llmverify.stats.tests.two_proportion_z` is the fallback for the
interleaved case, where each item appears in one arm only.

**What the null actually looks like.** Not zero. Paraphrasing perturbs
difficulty a little, and the neutral framing sentences the paraphraser adds can
shift accuracy on their own; a few points either way is the expected honest
result. A gap of tens of points with a small p-value is something else
entirely, and it invalidates every other measurement in the run rather than
adding to them -- which is why this probe's family is ``evasion`` and why
:func:`~llmverify.evidence.aggregate` promotes a strong finding here to the
``EVASION`` verdict instead of folding it into the match/mismatch axis.

**The structural check.** Independently of any benchmark data, the same question
is sent with and without a marker a naive detector would key on: a canary-style
header and the distinctive ``ANSWER:`` instruction format published harnesses
use. Response length and latency are then compared between the two arms. This is
corroboration only and is capped at ``WEAK``, for two reasons. It is a lower
bound on detectability -- the marker is synthetic and no real dataset's canary
string is shipped here, so a detector keyed to a specific corpus will not fire --
and length and latency differ between prompts for entirely innocent reasons.

**Multiplicity.** This probe runs up to three tests and corrects across them
with Holm-Bonferroni. That is not ceremony. Running several comparisons at a
nominal 5% and reporting whichever came out smallest is exactly how a tool like
this manufactures accusations against honest providers, and the whole package is
built on the premise that a false accusation is the worst thing it can do. Holm
is valid under any dependence structure, which matters because the accuracy test
and the structural tests are not independent.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..benchmarks.base import Benchmark, GradedResult, Variant, all_benchmarks, get_benchmark
from ..config import redact
from ..errors import ConfigError, DatasetError, LLMVerifyError
from ..evidence import DECISIVE, MODERATE, WEAK, Evidence, EvidenceStatus
from ..stats.multiplicity import holm_bonferroni
from ..stats.tests import ks_two_sample, mcnemar, two_proportion_z
from ..types import ChatRequest, Message, Role
from . import Probe, ProbeContext, register_probe
from .benchmark import dataset_loader

__all__ = ["CANARY_ITEMS", "MARKER_HEADER", "STRUCTURAL_PROMPTS", "EvasionProbe"]

#: Items per arm in the dedicated canary split, when one is needed. Sixty
#: requests is affordable next to a benchmark run and enough for a gap of the
#: size this probe is looking for -- tens of points -- to be significant.
CANARY_ITEMS: int = 30

#: Below this many graded items per arm the existing benchmark record is treated
#: as too thin to test, and a canary split is bought instead.
MIN_PER_ARM: int = 12

#: Fraction of the run's cost and wall-clock ceilings this probe may consume.
BUDGET_SHARE: float = 0.9

#: Adjusted p-value below which a difference is called real.
SIGNIFICANCE: float = 0.05

#: Percentage-point gap treated as the honest ceiling for paraphrase effects
#: alone. Below it, no amount of statistical significance is read as evasion:
#: with enough items a two-point difference becomes significant, and a two-point
#: difference is what paraphrasing does by itself.
NULL_GAP_PP: float = 5.0

#: Gap at which the finding earns this probe's full weight.
SATURATION_GAP_PP: float = 25.0

#: A synthetic recognisability marker. It is deliberately *not* any real
#: dataset's canary string -- shipping one would put a corpus watermark in this
#: package and would still only catch detectors keyed to that corpus. What it
#: does carry is the shape a naive detector matches on: an all-caps provenance
#: header, a dataset field, and an identifier that looks like a canary GUID.
MARKER_HEADER: str = (
    "BENCHMARK EVALUATION ITEM -- DO NOT REDISTRIBUTE\n"
    "dataset: llmverify-canary-split\n"
    "canary: llmverify-0000-0000-0000-000000000000\n"
    "task_id: item-{index:03d}\n"
)

#: Instruction appended to the marked arm, matching the output format published
#: harnesses ask for. On its own it is a strong hint that a request is an eval.
MARKER_INSTRUCTION: str = (
    'Respond with the final answer only, formatted exactly as "ANSWER: <answer>".'
)

#: Questions for the structural check. Short, factual, and answerable in a
#: sentence, so that a difference in response length between the two arms is a
#: difference in behaviour rather than in what was asked.
STRUCTURAL_PROMPTS: tuple[str, ...] = (
    "What is the capital city of Australia?",
    "Which planet in the solar system has the shortest day?",
    "In which year did the first successful powered aeroplane flight take place?",
    "What is the chemical symbol for tungsten?",
    "Which ocean lies between Africa and Australia?",
)


@dataclass(slots=True)
class _Arms:
    """Per-item correctness in each arm, and how the two are related."""

    verbatim: dict[str, bool] = field(default_factory=dict)
    paraphrased: dict[str, bool] = field(default_factory=dict)
    source: str = "benchmark probe"
    benchmark: str | None = None
    cost_usd: float = 0.0
    tokens: int = 0
    requests: int = 0
    note: str = ""

    @property
    def paired_ids(self) -> list[str]:
        return sorted(set(self.verbatim) & set(self.paraphrased))


@dataclass(slots=True)
class _Structural:
    """Length and latency samples from the marked and unmarked arms."""

    marked_chars: list[float] = field(default_factory=list)
    clean_chars: list[float] = field(default_factory=list)
    marked_latency: list[float] = field(default_factory=list)
    clean_latency: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens: int = 0
    requests: int = 0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return len(self.marked_chars) >= 3 and len(self.clean_chars) >= 3


@register_probe
class EvasionProbe(Probe):
    """Compare behaviour on recognisable inputs against behaviour on paraphrases."""

    name: ClassVar[str] = "evasion"
    layer: ClassVar[int] = 3
    family: ClassVar[str] = "evasion"
    order: ClassVar[int] = 130
    estimated_requests: ClassVar[int] = 2 * CANARY_ITEMS + 2 * len(STRUCTURAL_PROMPTS)
    description: ClassVar[str] = (
        "Accuracy on verbatim benchmark items against accuracy on paraphrases of the "
        "same questions, plus a length and latency check on marked versus unmarked prompts."
    )

    max_tokens: ClassVar[int] = 2048
    structural_max_tokens: ClassVar[int] = 256

    def applicable(self, ctx: ProbeContext) -> bool:
        """Only when paraphrasing is in play at all.

        With ``anti_evasion`` off the user has asked for verbatim items
        everywhere, and buying a paraphrased arm anyway would spend budget on a
        comparison they declined.
        """
        return ctx.run.anti_evasion

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        started = time.perf_counter()
        arms = _arms_from_shared(ctx)
        if len(arms.verbatim) < MIN_PER_ARM or len(arms.paraphrased) < MIN_PER_ARM:
            arms = await self._canary_split(ctx, arms)

        structural = await self._structural(ctx)

        accuracy = _accuracy_test(arms)
        raw_pvalues: list[float] = []
        if accuracy is not None:
            raw_pvalues.append(accuracy["p_value"])
        length_ks = latency_ks = None
        if structural.usable:
            length_ks = ks_two_sample(structural.marked_chars, structural.clean_chars)
            latency_ks = ks_two_sample(structural.marked_latency, structural.clean_latency)
            raw_pvalues.extend([length_ks[1], latency_ks[1]])

        adjusted = holm_bonferroni(raw_pvalues)
        cursor = 0
        evidence: list[Evidence] = []

        elapsed = time.perf_counter() - started
        if accuracy is None:
            evidence.append(self._no_accuracy_evidence(ctx, arms, elapsed))
        else:
            evidence.append(
                self._accuracy_evidence(arms, accuracy, adjusted[cursor], len(adjusted), elapsed)
            )
            cursor += 1

        if length_ks is not None and latency_ks is not None:
            evidence.append(
                self._structural_evidence(
                    structural,
                    (length_ks, adjusted[cursor]),
                    (latency_ks, adjusted[cursor + 1]),
                    len(adjusted),
                )
            )
        else:
            evidence.append(
                self._ev(
                    "marker_sensitivity",
                    0.0,
                    cap=WEAK,
                    status=(
                        EvidenceStatus.ERROR
                        if structural.error is not None
                        else EvidenceStatus.TRUNCATED
                    ),
                    detail=(
                        f"the marker-sensitivity check did not complete: "
                        f"{structural.error or 'the budget ran out before both arms were sent'}."
                    ),
                    data={"requests": structural.requests},
                    cost_usd=structural.cost_usd,
                    tokens=structural.tokens,
                )
            )
        return evidence

    # ------------------------------------------------------------- canary split

    async def _canary_split(self, ctx: ProbeContext, existing: _Arms) -> _Arms:
        """Buy a small paired split when the benchmark probe left too little behind.

        Both arms get the *same* items, which is the point: pairing makes
        McNemar available, and McNemar on thirty pairs detects the size of gap
        this probe exists to find far more cheaply than two independent samples
        would.
        """
        arms = _Arms(source="dedicated canary split")
        budget_pairs = self._affordable_pairs(ctx)
        if budget_pairs < MIN_PER_ARM:
            existing.note = (
                f"the benchmark probe left {len(existing.verbatim)} verbatim and "
                f"{len(existing.paraphrased)} paraphrased items, and the remaining budget "
                f"affords only {budget_pairs} fresh pairs"
            )
            return existing

        try:
            benchmark = _pick_benchmark(ctx)
        except ConfigError as exc:
            existing.note = str(exc)
            return existing

        loader, owned = dataset_loader(ctx)
        try:
            items = await benchmark.load(loader, limit=max(60, budget_pairs * 3))
        except (DatasetError, LLMVerifyError) as exc:
            existing.note = (
                f"a canary split of {benchmark.name} could not be loaded: "
                f"{redact(str(exc))[:200]}"
            )
            return existing
        finally:
            if owned:
                await loader.aclose()

        if len(items) < MIN_PER_ARM:
            existing.note = f"{benchmark.name} yielded only {len(items)} items"
            return existing

        rng = ctx.rng(f"evasion:{benchmark.name}")
        chosen = rng.sample(items, min(budget_pairs, len(items)))
        arms.benchmark = benchmark.name

        for item in chosen:
            if self._must_stop(ctx):
                break
            for variant, store in (
                (Variant.VERBATIM, arms.verbatim),
                (Variant.PARAPHRASED, arms.paraphrased),
            ):
                render_rng = ctx.rng(f"render:{benchmark.name}:{item.id}:{variant.value}")
                messages, state = benchmark.render(item, variant=variant, rng=render_rng)
                response, _error = await ctx.adapter.try_chat(
                    ChatRequest(
                        messages=messages,
                        max_tokens=self.max_tokens,
                        reasoning_effort=ctx.provider.reasoning_effort,
                        thinking_budget=ctx.provider.thinking_budget,
                    )
                )
                arms.requests += 1
                if response is None:
                    ctx.budget.charge(None, None)
                    continue
                arms.cost_usd += ctx.budget.charge(
                    response.usage.input_tokens, response.usage.output_tokens
                )
                arms.tokens += response.usage.total_tokens or 0
                correct, _extracted = benchmark.grade(item, response.text, state)
                if correct is not None:
                    store[item.id] = correct

        if len(arms.verbatim) < MIN_PER_ARM or len(arms.paraphrased) < MIN_PER_ARM:
            arms.note = (
                f"the canary split graded {len(arms.verbatim)} verbatim and "
                f"{len(arms.paraphrased)} paraphrased items before the budget ran out"
            )
        return arms

    def _affordable_pairs(self, ctx: ProbeContext) -> int:
        """How many verbatim/paraphrased pairs the remaining budget allows."""
        if self._must_stop(ctx):
            return 0
        remaining = ctx.budget.remaining_samples
        # Two requests per pair, and the structural check still has to be paid for.
        headroom = CANARY_ITEMS if remaining is None else (remaining - 2 * len(STRUCTURAL_PROMPTS))
        return max(0, min(CANARY_ITEMS, headroom // 2))

    # --------------------------------------------------------------- structural

    async def _structural(self, ctx: ProbeContext) -> _Structural:
        """Ask the same questions with and without a recognisability marker."""
        out = _Structural()
        for index, question in enumerate(STRUCTURAL_PROMPTS):
            for marked in (True, False):
                if self._must_stop(ctx):
                    return out
                prompt = (
                    f"{MARKER_HEADER.format(index=index)}\n{question}\n\n{MARKER_INSTRUCTION}"
                    if marked
                    else question
                )
                response, error = await ctx.adapter.try_chat(
                    ChatRequest(
                        messages=(Message(Role.USER, prompt),),
                        max_tokens=self.structural_max_tokens,
                    )
                )
                out.requests += 1
                if response is None:
                    out.error = redact(str(error))[:200]
                    ctx.budget.charge(None, None)
                    continue
                out.cost_usd += ctx.budget.charge(
                    response.usage.input_tokens, response.usage.output_tokens
                )
                out.tokens += response.usage.total_tokens or 0
                if marked:
                    out.marked_chars.append(float(len(response.text)))
                    out.marked_latency.append(response.timing.total_s)
                else:
                    out.clean_chars.append(float(len(response.text)))
                    out.clean_latency.append(response.timing.total_s)
        return out

    def _must_stop(self, ctx: ProbeContext) -> bool:
        budget = ctx.budget
        if budget.exhausted:
            return True
        cost_cap, wall_cap = budget.max_cost_usd, budget.max_wall_s
        if cost_cap is not None and budget.spent_usd >= cost_cap * BUDGET_SHARE:
            return True
        return wall_cap is not None and budget.elapsed_s >= wall_cap * BUDGET_SHARE

    # ------------------------------------------------------------------ evidence

    def _no_accuracy_evidence(
        self, ctx: ProbeContext, arms: _Arms, elapsed: float
    ) -> Evidence:
        reason = arms.note or (
            f"only {len(arms.verbatim)} verbatim and {len(arms.paraphrased)} paraphrased "
            f"items were graded, below the {MIN_PER_ARM} per arm this test needs"
        )
        status = EvidenceStatus.TRUNCATED if arms.requests else EvidenceStatus.SKIPPED
        return self._ev(
            "variant_accuracy_gap",
            0.0,
            status=status,
            detail=(
                f"no verbatim-versus-paraphrase comparison was possible: {reason}. "
                "Without it, routing on recognisable inputs has not been ruled out, so "
                "read every accuracy number in this run as an upper bound."
            ),
            data={
                "verbatim_items": len(arms.verbatim),
                "paraphrased_items": len(arms.paraphrased),
                "source": arms.source,
                "benchmark": arms.benchmark,
                "requests": arms.requests,
                "run_anti_evasion": ctx.run.anti_evasion,
            },
            cost_usd=arms.cost_usd,
            tokens=arms.tokens,
            duration_s=elapsed,
        )

    def _accuracy_evidence(
        self,
        arms: _Arms,
        accuracy: dict[str, Any],
        adjusted_p: float,
        tests: int,
        elapsed: float,
    ) -> Evidence:
        gap = accuracy["gap_pp"]
        data = {
            **accuracy,
            "p_adjusted": adjusted_p,
            "tests_corrected": tests,
            "source": arms.source,
            "benchmark": arms.benchmark,
            "requests": arms.requests,
            "estimated_cost_usd": round(arms.cost_usd, 6),
            "null_gap_pp": NULL_GAP_PP,
            "note": arms.note,
        }
        charged = {"cost_usd": arms.cost_usd, "tokens": arms.tokens, "duration_s": elapsed}
        headline = (
            f"verbatim {accuracy['verbatim_correct']}/{accuracy['verbatim_n']} "
            f"({accuracy['verbatim_rate_pp']:.1f}%) against paraphrased "
            f"{accuracy['paraphrased_correct']}/{accuracy['paraphrased_n']} "
            f"({accuracy['paraphrased_rate_pp']:.1f}%), a gap of {gap:.1f} points "
            f"({accuracy['test']}, p={accuracy['p_value']:.3g}, {adjusted_p:.3g} after "
            f"Holm correction across {tests} tests)"
        )

        if adjusted_p < SIGNIFICANCE and gap > NULL_GAP_PP:
            severity = min(
                1.0, (gap - NULL_GAP_PP) / max(1e-9, SATURATION_GAP_PP - NULL_GAP_PP)
            )
            llr = -(MODERATE + (DECISIVE - MODERATE) * severity)
            return self._ev(
                "variant_accuracy_gap",
                llr,
                cap=DECISIVE,
                detail=(
                    f"{headline}. The questions are the same and the gold answers are the "
                    "same; only recognisability differs. An endpoint whose accuracy depends "
                    "on whether an input looks like a published benchmark item is choosing "
                    "what to measure, which invalidates every other measurement in this run."
                ),
                data=data,
                **charged,
            )

        if adjusted_p < SIGNIFICANCE and gap < -NULL_GAP_PP:
            return self._ev(
                "variant_accuracy_gap",
                0.0,
                cap=DECISIVE,
                detail=(
                    f"{headline}. The paraphrased arm scored *higher*, which is not what "
                    "routing on recognition looks like. Most likely the paraphrases were "
                    "easier than the originals; reported, not weighed."
                ),
                data=data,
                **charged,
            )

        if accuracy["verbatim_n"] >= 20 and accuracy["paraphrased_n"] >= 20 and abs(gap) <= 5.0:
            return self._ev(
                "variant_accuracy_gap",
                0.5 * WEAK,
                cap=MODERATE,
                detail=(
                    f"{headline}. Within the range paraphrasing produces on its own, so the "
                    "endpoint's behaviour does not appear to depend on whether an input is "
                    "recognisable. Weakly supportive: it rules out lookup-based routing at "
                    "this sample size, not a substitution that applies to every request."
                ),
                data=data,
                **charged,
            )

        return self._ev(
            "variant_accuracy_gap",
            0.0,
            cap=DECISIVE,
            detail=(
                f"{headline}. Not separable at this sample size, so routing on recognisable "
                "inputs is neither shown nor ruled out."
            ),
            data=data,
            **charged,
        )

    def _structural_evidence(
        self,
        structural: _Structural,
        length: tuple[tuple[float, float], float],
        latency: tuple[tuple[float, float], float],
        tests: int,
    ) -> Evidence:
        """Report the marker-sensitivity check, always with low weight."""
        (length_d, length_p), length_adj = length
        (latency_d, latency_p), latency_adj = latency
        data = {
            "test": "two-sample KS on response length and latency, marked vs unmarked",
            "prompts": len(STRUCTURAL_PROMPTS),
            "requests": structural.requests,
            "length_d": round(length_d, 4),
            "length_p": length_p,
            "length_p_adjusted": length_adj,
            "latency_d": round(latency_d, 4),
            "latency_p": latency_p,
            "latency_p_adjusted": latency_adj,
            "tests_corrected": tests,
            "marked_mean_chars": round(statistics.fmean(structural.marked_chars), 1),
            "clean_mean_chars": round(statistics.fmean(structural.clean_chars), 1),
            "marked_mean_latency_s": round(statistics.fmean(structural.marked_latency), 3),
            "clean_mean_latency_s": round(statistics.fmean(structural.clean_latency), 3),
            "marker_is_synthetic": True,
        }
        charged = {"cost_usd": structural.cost_usd, "tokens": structural.tokens}
        both = length_adj < SIGNIFICANCE and latency_adj < SIGNIFICANCE

        if both:
            return self._ev(
                "marker_sensitivity",
                -WEAK,
                cap=WEAK,
                detail=(
                    f"marked prompts came back at {data['marked_mean_chars']} characters and "
                    f"{data['marked_mean_latency_s']}s against {data['clean_mean_chars']} "
                    f"characters and {data['clean_mean_latency_s']}s unmarked, and both "
                    f"differences survive correction (length p={length_adj:.3g}, latency "
                    f"p={latency_adj:.3g}). Corroboration only: the marker also changes the "
                    "requested output format, which moves length legitimately."
                ),
                data=data,
                **charged,
            )

        return self._ev(
            "marker_sensitivity",
            0.0,
            cap=WEAK,
            detail=(
                f"adding a canary-style header and a strict answer format did not change "
                f"response length (p={length_adj:.3g}) and latency (p={latency_adj:.3g}) "
                "together. A lower bound on detectability rather than a clean bill of "
                "health: the marker here is synthetic, so a detector keyed to a real "
                "corpus would not fire on it."
            ),
            data=data,
            **charged,
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
# Arms and tests
# --------------------------------------------------------------------------- #


def _arms_from_shared(ctx: ProbeContext) -> _Arms:
    """Read the benchmark probe's per-item record, if it left one.

    Only the first grade for an item and variant is kept. A repeat of the same
    item is not an independent observation of the same routing decision, and
    counting it twice would inflate both arms' sample sizes.
    """
    arms = _Arms()
    stored = ctx.shared.get("benchmark_results")
    if not isinstance(stored, list):
        return arms

    names = ctx.shared.get("benchmark_names")
    if isinstance(names, list) and names:
        arms.benchmark = ", ".join(str(name) for name in names)

    for result in stored:
        if not isinstance(result, GradedResult) or result.correct is None:
            continue
        if result.variant is Variant.VERBATIM:
            arms.verbatim.setdefault(result.item_id, result.correct)
        elif result.variant is Variant.PARAPHRASED:
            arms.paraphrased.setdefault(result.item_id, result.correct)
    return arms


def _accuracy_test(arms: _Arms) -> dict[str, Any] | None:
    """Test the verbatim-versus-paraphrase gap, paired where the pairing exists."""
    paired = arms.paired_ids
    if len(paired) >= MIN_PER_ARM:
        verbatim = {key: arms.verbatim[key] for key in paired}
        paraphrased = {key: arms.paraphrased[key] for key in paired}
        b = sum(1 for key in paired if verbatim[key] and not paraphrased[key])
        c = sum(1 for key in paired if paraphrased[key] and not verbatim[key])
        p_value = mcnemar(b, c)
        test = f"exact McNemar on {len(paired)} paired items"
        extra = {"discordant_verbatim_only": b, "discordant_paraphrased_only": c}
    elif len(arms.verbatim) >= MIN_PER_ARM and len(arms.paraphrased) >= MIN_PER_ARM:
        verbatim = arms.verbatim
        paraphrased = arms.paraphrased
        _z, p_value = two_proportion_z(
            sum(verbatim.values()),
            len(verbatim),
            sum(paraphrased.values()),
            len(paraphrased),
        )
        test = "two-proportion z on unpaired arms"
        extra = {"paired_items": len(paired)}
    else:
        return None

    verbatim_correct = sum(verbatim.values())
    paraphrased_correct = sum(paraphrased.values())
    verbatim_rate = verbatim_correct / len(verbatim) * 100.0
    paraphrased_rate = paraphrased_correct / len(paraphrased) * 100.0
    return {
        "test": test,
        "p_value": p_value,
        "verbatim_n": len(verbatim),
        "paraphrased_n": len(paraphrased),
        "verbatim_correct": verbatim_correct,
        "paraphrased_correct": paraphrased_correct,
        "verbatim_rate_pp": round(verbatim_rate, 2),
        "paraphrased_rate_pp": round(paraphrased_rate, 2),
        "gap_pp": round(verbatim_rate - paraphrased_rate, 2),
        **extra,
    }


def _pick_benchmark(ctx: ProbeContext) -> Benchmark:
    """Choose a benchmark for the canary split.

    The one the benchmark probe already used is preferred: its items are cached,
    so the split costs network time only for generation, and the two probes then
    speak about the same corpus. Otherwise the cheapest ungated, sandbox-free
    benchmark is taken -- what matters here is that the endpoint can get some
    items right and some wrong, not that the benchmark discriminates between
    frontier models.
    """
    names = ctx.shared.get("benchmark_names")
    if isinstance(names, list) and names:
        return get_benchmark(str(names[0]))
    if ctx.run.benchmarks:
        return get_benchmark(ctx.run.benchmarks[0])

    registry = [cls for cls in all_benchmarks().values() if not cls.needs_sandbox]
    if not registry:
        raise ConfigError("no benchmark is available to build a canary split from")
    chosen = sorted(registry, key=lambda cls: (cls.gated, cls.name))[0]
    return chosen()
