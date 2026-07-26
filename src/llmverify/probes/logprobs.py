"""Distribution-level tests on the logprobs an endpoint reports about itself.

A logprob table says what the sampler was choosing between, not merely what it
chose, so where one is available it carries far more information per request
than the text does. That makes this the most query-efficient test in the
package -- when it is available at all, which is the exception. Anthropic's
protocol has no logprobs and no seed, Gemini's are unconfirmed, and reasoning
models across vendors decline them in practice, so ``UNSUPPORTED`` is the
ordinary outcome here and is reported as an ordinary outcome rather than as a
failure.

The probe runs in two modes.

**With a baseline.** The same prompts go to both endpoints and three tests
compare what comes back. The primary statistic is a two-sample
Kolmogorov-Smirnov test on the pooled chosen-token logprobs: two servings of one
model produce two samples from one distribution, and KS asks whether they look
like it. Secondary are a chi-square on which token the candidate actually
chose, categorised by that token's rank in the baseline's top-k, with
expectations taken from the baseline's own probabilities; and a rank-uniformity
check, since under the null the baseline's chosen token should sit at a
uniformly distributed percentile of the candidate's distribution -- randomised
within its own atom, because a discrete distribution has no exactly uniform
percentile without it.

Positions are compared only up to and including the first token where the two
completions diverge. After that point the two models are continuing different
prefixes, so position *i* is not the same random variable on both sides and
comparing it measures our alignment rather than their weights.

Sampling temperature is set to 1.0 on purpose. All three tests treat the emitted
token as a draw from the reported distribution; a greedy decode would make the
emitted token the argmax by construction and both rank tests would then reject
the null against every endpoint, honest ones included. Since an endpoint may
ignore the temperature it was sent, the assumption is checked rather than
trusted: if either side emits its most likely token materially more often than
its own probabilities predict, the rank tests are skipped and said to be
skipped, and only the KS test -- which compares the same quantity on both sides
and does not care how it was chosen -- is reported.

**Without a baseline.** There is nothing to compare against, so the probe checks
only that the endpoint's own numbers form a distribution: entries sorted
descending, the chosen token present among the alternatives, every value at or
below zero, and an entropy profile that is not obviously manufactured. OpenAI
reports ``-9999.0`` for tokens it did not track; that sentinel is excluded from
every statistic rather than averaged in as if it were a probability of
1e-4343.

**What a significant result is worth, and what it is not.** arXiv:2504.04715 is
blunt about the ceiling: methods that use log probabilities "are defeated by
inherent inference nondeterminism in production". Batch composition, kernel
selection, quantization and speculative decoding all move a logprob without
touching a weight, and providers change all four without notice. A significant
KS result therefore says the two endpoints are running *different serving
configurations* -- which may mean different weights, different precision, or
merely a different GPU generation on a Tuesday. It is not proof of substituted
weights, so every item this probe emits is capped at ``MODERATE`` and further
discounted by :data:`CONFIGURATION_DISCOUNT`, and the ``distribution`` family
caps at ``STRONG`` however many tests fire.

The complementary inference -- an endpoint claiming a Claude model while
returning logprobs at all, which Anthropic's protocol cannot express -- belongs
to the api-surface probe and is deliberately not repeated here. Counting one
observation in two families would defeat the damping that keeps correlated
evidence from compounding.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..config import redact
from ..evidence import MODERATE, WEAK, Evidence, EvidenceStatus
from ..stats.multiplicity import holm_bonferroni
from ..stats.tests import chi_square_gof, ks_two_sample
from ..types import ChatRequest, Message, Role, TokenLogprob
from . import Probe, ProbeContext, register_probe

if TYPE_CHECKING:
    from ..adapters.base import Adapter

__all__ = ["CONFIGURATION_DISCOUNT", "PROMPTS", "UNTRACKED_FLOOR", "LogprobsProbe"]

#: Fixed prompt set. Weighted towards constrained continuations for a practical
#: reason as much as a statistical one: the rank tests can only use positions
#: where both endpoints were continuing the same prefix, and two stochastic
#: decodes of an open-ended prompt diverge at the first token. A prompt whose
#: next few tokens are near-certain stays aligned long enough to be worth
#: comparing, while the open ones supply the high-entropy end of the range.
PROMPTS: tuple[str, ...] = (
    "Continue the list with the next three items: Monday, Tuesday, Wednesday,",
    "Continue the sequence with the next four numbers: 2, 4, 6, 8,",
    "Complete the sentence exactly: the capital city of France is",
    "Finish the proverb: a rolling stone gathers",
    "Continue the list: red, orange, yellow, green,",
    "Name the chemical element with atomic number 26. Answer with the name alone.",
    "Complete the sentence with a single word: the opposite of ancient is",
    "Continue the alphabet from here: p, q, r, s,",
    "Complete this line: the mitochondrion is the powerhouse of the",
    "Write one sentence about a harbour at night.",
    "In one short sentence, say what a compiler does.",
    "Describe the taste of a lemon in one short sentence.",
)

#: OpenAI reports ``-9999.0`` for tokens it did not track. Anything at or below
#: this floor is that marker rather than data: a genuine logprob of -9000 is a
#: probability around 1e-3909, which no sampler in existence produces.
UNTRACKED_FLOOR: float = -9000.0

#: Logprobs are samples from a *serving configuration*, and configurations
#: change for reasons that have nothing to do with the weights. Every LLR this
#: probe produces is multiplied by this factor before capping.
CONFIGURATION_DISCOUNT: float = 0.5

#: Slack when comparing logprobs for ordering and for the zero ceiling. Servers
#: round their float32 values on the way into JSON, so an exact comparison would
#: report an ordering violation on a tie.
_TOLERANCE: float = 1e-6

#: Minimum aligned positions before the rank tests are attempted at all. Below
#: this the chi-square approximation is worthless and the honest answer is that
#: the run bought too few tokens to say anything.
_MIN_ALIGNED: int = 20

#: Minimum expected count per chi-square cell before neighbouring cells are
#: pooled into the tail.
_MIN_EXPECTED: float = 5.0

#: Bins for the rank-uniformity check. Four keeps every expected count at or
#: above five once ``_MIN_ALIGNED`` positions exist.
_UNIFORMITY_BINS: int = 4

#: How far the share of positions where the emitted token was the most likely
#: one may exceed what the reported probabilities predict before the decode is
#: treated as effectively greedy. Both rank tests assume the emitted token is a
#: draw from the reported distribution; against a greedy decode they reject the
#: null on every endpoint, including an honest one.
_GREEDY_MARGIN: float = 0.25

#: Model of the p-value under the alternative: ``Beta(a, 1)``, whose density is
#: ``a * p ** (a - 1)``. Under the null a p-value is uniform, so the LLR in
#: favour of the claim is ``-ln(a) - (a - 1) * ln(p)``. Smaller ``a`` means a
#: sharper assumed concentration near zero and a harsher penalty for a small
#: p-value; 0.2 is deliberately mild.
_ALTERNATIVE_CONCENTRATION: float = 0.2


@dataclass(slots=True)
class _Sample:
    """The logprob table returned for one prompt."""

    prompt_index: int
    tokens: tuple[TokenLogprob, ...]
    text: str


@dataclass(slots=True)
class _Collection:
    """Everything one endpoint gave back, plus what it cost to ask."""

    samples: list[_Sample] = field(default_factory=list)
    requests: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    duration_s: float = 0.0
    #: The request shape that survived, for the report.
    configuration: str = ""
    #: Why no logprobs came back, when none did.
    refusal: str | None = None
    error: str | None = None

    @property
    def positions(self) -> int:
        return sum(len(sample.tokens) for sample in self.samples)


@register_probe
class LogprobsProbe(Probe):
    """Compare reported token distributions, or check that they are distributions."""

    name: ClassVar[str] = "logprobs"
    layer: ClassVar[int] = 3
    family: ClassVar[str] = "distribution"
    order: ClassVar[int] = 110
    estimated_requests: ClassVar[int] = 2 * len(PROMPTS)
    description: ClassVar[str] = (
        "Two-sample KS and rank tests on returned logprobs against a baseline "
        "endpoint, or self-consistency checks on the distribution when there is none."
    )

    #: Short completions. The information is in the per-token distributions, and
    #: a long generation buys mostly positions the two endpoints have already
    #: diverged at.
    max_tokens: ClassVar[int] = 24
    #: Within OpenAI's documented 0-20 range and within the narrower range some
    #: compatible front ends still enforce.
    top_k: ClassVar[int] = 5
    #: The tests read emitted tokens as draws from the reported distribution, so
    #: the decode must not be greedy.
    temperature: ClassVar[float] = 1.0

    async def run(self, ctx: ProbeContext) -> list[Evidence]:
        if not ctx.adapter.capabilities.logprobs:
            return [
                ctx.unsupported(
                    self.name,
                    f"the {ctx.adapter.name} protocol has no logprob fields, so there is "
                    "no distribution to test.",
                    family=self.family,
                )
            ]

        candidate = await self._collect(ctx, ctx.adapter, label="candidate")
        if candidate.error is not None:
            return [
                self._ev(
                    "logprob_availability",
                    0.0,
                    status=EvidenceStatus.ERROR,
                    detail=f"no prompt completed: {candidate.error}",
                    data={"requests": candidate.requests},
                    duration_s=candidate.duration_s,
                )
            ]
        if not candidate.samples:
            return [
                self._ev(
                    "logprob_availability",
                    0.0,
                    status=EvidenceStatus.UNSUPPORTED,
                    detail=(
                        f"{candidate.refusal or 'the endpoint returned no logprob table'}. "
                        "This is the common case and argues neither way: most reasoning "
                        "models decline logprobs even when the protocol allows them."
                    ),
                    data={"requests": candidate.requests, "configuration": candidate.configuration},
                    cost_usd=candidate.cost_usd,
                    tokens=candidate.tokens,
                    duration_s=candidate.duration_s,
                )
            ]

        validity = _validity(candidate.samples)
        profile = _entropy_profile(candidate.samples)
        shared = {
            "prompts": len(candidate.samples),
            "positions": candidate.positions,
            "configuration": candidate.configuration,
            "top_k_requested": self.top_k,
        }

        evidence = [
            self._validity_evidence(candidate, validity, shared),
            self._profile_evidence(profile, validity, shared),
        ]
        if ctx.has_baseline:
            evidence.extend(await self._compare(ctx, candidate, shared))
        return evidence

    # ---------------------------------------------------------------- collection

    async def _collect(self, ctx: ProbeContext, adapter: Adapter, *, label: str) -> _Collection:
        """Ask every prompt for its logprob table, settling the request shape first.

        The first prompt decides which shape the endpoint tolerates: reasoning
        models commonly refuse an explicit temperature, and refusing to fall back
        would report ``ERROR`` for an endpoint that is merely opinionated. Which
        shape survived is recorded, because a run at the provider's default
        temperature is not the run the rank tests assume.
        """
        started = time.perf_counter()
        out = _Collection()
        shapes: tuple[tuple[str, float | None], ...] = (
            (f"temperature {self.temperature}", self.temperature),
            ("provider default temperature", None),
        )

        settled: float | None = None
        last_error: Exception | None = None
        for shape_label, temperature in shapes:
            response, error = await adapter.try_chat(self._request(PROMPTS[0], temperature))
            out.requests += 1
            if response is None:
                last_error = error
                continue
            settled = temperature
            out.configuration = shape_label
            out.cost_usd += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            out.tokens += response.usage.total_tokens or 0
            if response.logprobs:
                out.samples.append(_Sample(0, response.logprobs, response.text))
            break

        if out.configuration == "":
            out.error = redact(str(last_error))[:300]
            out.duration_s = time.perf_counter() - started
            return out

        for index, prompt in enumerate(PROMPTS[1:], start=1):
            if ctx.budget.exhausted:
                break
            response, _error = await adapter.try_chat(self._request(prompt, settled))
            out.requests += 1
            if response is None:
                continue
            out.cost_usd += ctx.budget.charge(
                response.usage.input_tokens, response.usage.output_tokens
            )
            out.tokens += response.usage.total_tokens or 0
            if response.logprobs:
                out.samples.append(_Sample(index, response.logprobs, response.text))

        if not out.samples:
            out.refusal = (
                f"the {label} endpoint accepted logprobs=true and top_logprobs="
                f"{self.top_k} but returned no logprob entries"
            )
        out.duration_s = time.perf_counter() - started
        return out

    def _request(self, prompt: str, temperature: float | None) -> ChatRequest:
        return ChatRequest(
            messages=(Message(Role.USER, prompt),),
            max_tokens=self.max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=self.top_k,
        )

    # ------------------------------------------------------------ self-checking

    def _validity_evidence(
        self, collection: _Collection, validity: dict[str, Any], shared: dict[str, Any]
    ) -> Evidence:
        """Whether the numbers returned describe a probability distribution."""
        data = {**shared, **validity}
        charged = {
            "cost_usd": collection.cost_usd,
            "tokens": collection.tokens,
            "duration_s": collection.duration_s,
        }
        faults: list[str] = []
        if validity["positive_logprobs"]:
            faults.append(
                f"{validity['positive_logprobs']} entries report a logprob above zero, "
                "which is a probability above one"
            )
        if validity["unsorted_positions"]:
            faults.append(
                f"{validity['unsorted_positions']} positions list their alternatives out of "
                "descending order"
            )
        if validity["chosen_absent_positions"]:
            faults.append(
                f"{validity['chosen_absent_positions']} positions omit the emitted token from "
                f"their own top-{self.top_k}"
            )
        if validity["malformed_entries"]:
            faults.append(
                f"{validity['malformed_entries']} entries carry no usable numeric logprob"
            )

        if not faults:
            return self._ev(
                "logprob_self_consistency",
                _discounted(0.5 * WEAK),
                detail=(
                    f"{validity['usable_positions']} positions across "
                    f"{len(collection.samples)} prompts form a valid distribution: values "
                    "at or below zero, alternatives sorted, emitted token present among "
                    "them. Weakly supportive only -- a proxy relaying a real upstream's "
                    "logprobs passes this too."
                ),
                data=data,
                **charged,
            )

        # A malformed table is an argument about the software in front of the
        # model, not about the weights behind it, so it is capped low even when
        # several faults appear at once.
        magnitude = MODERATE if len(faults) > 1 else 0.6 * MODERATE
        return self._ev(
            "logprob_self_consistency",
            _discounted(-magnitude),
            detail=(
                "the returned logprobs are not a coherent distribution: "
                + "; ".join(faults)
                + ". A first-party endpoint emits these from the sampler itself, so the "
                "shape is a property of the serving software rather than of the weights."
            ),
            data=data,
            **charged,
        )

    def _profile_evidence(
        self, profile: dict[str, Any], validity: dict[str, Any], shared: dict[str, Any]
    ) -> Evidence:
        """Whether the entropy profile could have come from a running sampler."""
        data = {**shared, **profile}
        if validity["usable_positions"] < _MIN_ALIGNED:
            return self._ev(
                "logprob_entropy_profile",
                0.0,
                status=EvidenceStatus.TRUNCATED,
                detail=(
                    f"only {validity['usable_positions']} tracked positions came back, too "
                    "few to say whether the entropy profile is plausible."
                ),
                data=data,
            )

        if profile["all_certain"]:
            return self._ev(
                "logprob_entropy_profile",
                _discounted(-MODERATE),
                detail=(
                    "every tracked position reports the emitted token with probability 1. "
                    "Natural-language generation is not certain at every token, so these "
                    "numbers were not produced by a sampler choosing between candidates."
                ),
                data=data,
            )
        if profile["constant_logprob"]:
            return self._ev(
                "logprob_entropy_profile",
                _discounted(-MODERATE),
                detail=(
                    f"every tracked position reports the same logprob "
                    f"({profile['mean_chosen_logprob']:.4f}). A constant is not a "
                    "distribution; something is filling the field in rather than measuring it."
                ),
                data=data,
            )
        if profile["flat_alternatives"]:
            return self._ev(
                "logprob_entropy_profile",
                _discounted(-0.6 * MODERATE),
                detail=(
                    f"at every position the top-{self.top_k} alternatives are equally "
                    "likely to within rounding. A real next-token distribution is not flat "
                    "across its own head."
                ),
                data=data,
            )

        return self._ev(
            "logprob_entropy_profile",
            0.0,
            detail=(
                f"mean emitted-token logprob {profile['mean_chosen_logprob']:.3f}, median "
                f"{profile['median_chosen_logprob']:.3f}, mean top-{self.top_k} entropy "
                f"{profile['mean_truncated_entropy']:.3f} nats over "
                f"{profile['usable_positions']} positions. Reported as information: without "
                "a reference distribution for this model there is no threshold these "
                "numbers can be held to."
            ),
            data=data,
        )

    # -------------------------------------------------------------- comparison

    async def _compare(
        self, ctx: ProbeContext, candidate: _Collection, shared: dict[str, Any]
    ) -> list[Evidence]:
        """Run the three two-sample tests against the baseline endpoint."""
        assert ctx.baseline is not None  # guaranteed by ctx.has_baseline
        baseline = await self._collect(ctx, ctx.baseline, label="baseline")
        charged = {
            "cost_usd": baseline.cost_usd,
            "tokens": baseline.tokens,
            "duration_s": baseline.duration_s,
        }
        if not baseline.samples:
            return [
                self._ev(
                    "baseline_distribution",
                    0.0,
                    status=EvidenceStatus.UNSUPPORTED,
                    detail=(
                        f"{baseline.refusal or baseline.error or 'the baseline returned nothing'}"
                        ". Without logprobs from both endpoints there is no two-sample test "
                        "to run."
                    ),
                    data=shared,
                    **charged,
                )
            ]

        pairs = _aligned_pairs(candidate.samples, baseline.samples)
        candidate_values = _chosen_logprobs(candidate.samples)
        baseline_values = _chosen_logprobs(baseline.samples)

        tests: list[tuple[str, float, dict[str, Any]]] = []
        if candidate_values and baseline_values:
            statistic, p_value = ks_two_sample(candidate_values, baseline_values)
            tests.append(
                (
                    "top1_logprob_distribution",
                    p_value,
                    {
                        "test": "two-sample Kolmogorov-Smirnov",
                        "d_statistic": round(statistic, 4),
                        "candidate_positions": len(candidate_values),
                        "baseline_positions": len(baseline_values),
                        "candidate_mean": round(statistics.fmean(candidate_values), 4),
                        "baseline_mean": round(statistics.fmean(baseline_values), 4),
                    },
                )
            )

        candidate_decode = _sampling_check(candidate.samples)
        baseline_decode = _sampling_check(baseline.samples)
        sampled = candidate_decode["looks_sampled"] and baseline_decode["looks_sampled"]

        if sampled:
            rank_test = _rank_identity_test(pairs)
            if rank_test is not None:
                tests.append(("top_token_rank_distribution", rank_test[0], rank_test[1]))

            uniformity = _rank_uniformity_test(pairs, ctx.rng("logprobs:uniformity"))
            if uniformity is not None:
                tests.append(("chosen_token_rank_uniformity", uniformity[0], uniformity[1]))

        base = {
            **shared,
            "baseline_provider": ctx.baseline_provider.name if ctx.baseline_provider else None,
            "baseline_configuration": baseline.configuration,
            "aligned_positions": len(pairs),
            "baseline_prompts": len(baseline.samples),
            "candidate_decode": candidate_decode,
            "baseline_decode": baseline_decode,
        }

        if not tests:
            return [
                self._ev(
                    "baseline_distribution",
                    0.0,
                    status=EvidenceStatus.TRUNCATED,
                    detail=(
                        "the two endpoints produced too few comparable positions to test: "
                        f"{len(pairs)} aligned, {len(candidate_values)} tracked on the "
                        "candidate side."
                    ),
                    data=base,
                    **charged,
                )
            ]

        adjusted = holm_bonferroni([p for _, p, _ in tests])
        evidence: list[Evidence] = []
        if not sampled:
            evidence.append(
                self._ev(
                    "token_rank_tests",
                    0.0,
                    status=EvidenceStatus.UNSUPPORTED,
                    detail=(
                        "the rank tests were not run: at least one endpoint emitted its most "
                        "likely token far more often than its own probabilities predict "
                        f"(candidate {candidate_decode['observed_argmax_share']:.0%} against a "
                        f"predicted {candidate_decode['expected_argmax_share']:.0%}, baseline "
                        f"{baseline_decode['observed_argmax_share']:.0%} against "
                        f"{baseline_decode['expected_argmax_share']:.0%}), so the decode is "
                        "effectively greedy and the emitted token is not a draw from the "
                        "reported distribution. Both rank tests would reject the null against "
                        "any endpoint under that condition, including an honest one."
                    ),
                    data=base,
                )
            )
        for index, (label, raw_p, detail_data) in enumerate(tests):
            evidence.append(
                self._test_evidence(
                    label,
                    raw_p,
                    adjusted[index],
                    {**base, **detail_data, "p_value": raw_p, "p_adjusted": adjusted[index]},
                    charged if index == 0 else {},
                )
            )
        return evidence

    def _test_evidence(
        self,
        label: str,
        raw_p: float,
        adjusted_p: float,
        data: dict[str, Any],
        charged: dict[str, Any],
    ) -> Evidence:
        """Turn one adjusted p-value into a signed, discounted, capped LLR."""
        llr = _discounted(_llr_from_pvalue(adjusted_p))
        if adjusted_p < 0.05:
            detail = (
                f"{data['test']} separates the two endpoints (p={raw_p:.2g}, {adjusted_p:.2g} "
                "after Holm correction across this probe's tests). The two endpoints are "
                "running different serving configurations. That is not the same finding as "
                "different weights: quantization, batch composition and kernel choice all "
                "move a logprob, which is why this is capped at moderate."
            )
        else:
            detail = (
                f"{data['test']} finds no separation (p={raw_p:.2g}, {adjusted_p:.2g} "
                "adjusted). The endpoints' reported distributions agree as far as this "
                "test can see, which rules out a coarse substitution and nothing finer."
            )
        return self._ev(label, llr, detail=detail, data=data, **charged)

    # ------------------------------------------------------------------ helpers

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
# Reading the tables
# --------------------------------------------------------------------------- #


def _untracked(value: float) -> bool:
    """Whether a reported logprob is the provider's "not tracked" marker.

    ``NaN`` is deliberately not caught here: an unparseable value is malformed,
    which is a different finding from a value the provider declined to track.
    """
    return value <= UNTRACKED_FLOOR


def _usable_top(entry: TokenLogprob) -> list[tuple[str, float]]:
    return [(token, value) for token, value in entry.top if _is_number(value)]


def _is_number(value: float) -> bool:
    return math.isfinite(value) and not _untracked(value)


def _chosen_logprobs(samples: list[_Sample]) -> list[float]:
    """Every tracked emitted-token logprob, pooled across prompts."""
    return [
        entry.logprob
        for sample in samples
        for entry in sample.tokens
        if _is_number(entry.logprob)
    ]


def _validity(samples: list[_Sample]) -> dict[str, Any]:
    """Count the ways the returned tables fail to be distributions."""
    positions = usable = untracked = malformed = 0
    positive = unsorted = chosen_absent = with_alternatives = 0

    for sample in samples:
        for entry in sample.tokens:
            positions += 1
            if _untracked(entry.logprob):
                untracked += 1
                continue
            if not math.isfinite(entry.logprob):
                malformed += 1
                continue
            usable += 1
            if entry.logprob > _TOLERANCE:
                positive += 1
            top = _usable_top(entry)
            if not top:
                continue
            with_alternatives += 1
            values = [value for _, value in top]
            if any(a + _TOLERANCE < b for a, b in itertools.pairwise(values)):
                unsorted += 1
            if entry.token not in {token for token, _ in top}:
                chosen_absent += 1

    return {
        "total_positions": positions,
        "usable_positions": usable,
        "untracked_positions": untracked,
        "malformed_entries": malformed,
        "positive_logprobs": positive,
        "unsorted_positions": unsorted,
        "chosen_absent_positions": chosen_absent,
        "positions_with_alternatives": with_alternatives,
    }


def _entropy_profile(samples: list[_Sample]) -> dict[str, Any]:
    """Summarise the shape of the reported distributions.

    ``truncated_entropy`` is computed over the returned alternatives only and is
    therefore a lower bound on the true entropy -- the tail the endpoint did not
    send can only add to it. That is enough for the question asked here, which
    is whether the profile is degenerate rather than what its exact value is.
    """
    chosen = _chosen_logprobs(samples)
    entropies: list[float] = []
    flat_positions = 0
    positions_with_alternatives = 0

    for sample in samples:
        for entry in sample.tokens:
            top = _usable_top(entry)
            if len(top) < 2:
                continue
            positions_with_alternatives += 1
            probabilities = [math.exp(value) for _, value in top]
            entropies.append(
                -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)
            )
            values = [value for _, value in top]
            if max(values) - min(values) <= _TOLERANCE:
                flat_positions += 1

    return {
        "usable_positions": len(chosen),
        "mean_chosen_logprob": statistics.fmean(chosen) if chosen else 0.0,
        "median_chosen_logprob": statistics.median(chosen) if chosen else 0.0,
        "min_chosen_logprob": min(chosen) if chosen else 0.0,
        "distinct_chosen_logprobs": len({round(value, 9) for value in chosen}),
        "mean_truncated_entropy": statistics.fmean(entropies) if entropies else 0.0,
        "certain_position_share": (
            sum(1 for value in chosen if value > -_TOLERANCE) / len(chosen) if chosen else 0.0
        ),
        "all_certain": bool(chosen) and all(value > -_TOLERANCE for value in chosen),
        "constant_logprob": len({round(value, 9) for value in chosen}) == 1 and len(chosen) > 1,
        "flat_alternatives": (
            positions_with_alternatives > 0 and flat_positions == positions_with_alternatives
        ),
    }


# --------------------------------------------------------------------------- #
# Two-sample machinery
# --------------------------------------------------------------------------- #


def _aligned_pairs(
    candidate: list[_Sample], baseline: list[_Sample]
) -> list[tuple[TokenLogprob, TokenLogprob]]:
    """Positions where both endpoints were continuing the same prefix.

    The common prefix plus the *first* differing position: that position is
    still conditioned on identical context on both sides, and it is the only
    place a disagreement can be observed at all, so dropping it would leave the
    rank tests with nothing but agreements by construction.
    """
    by_prompt = {sample.prompt_index: sample for sample in baseline}
    pairs: list[tuple[TokenLogprob, TokenLogprob]] = []
    for sample in candidate:
        other = by_prompt.get(sample.prompt_index)
        if other is None:
            continue
        for left, right in zip(sample.tokens, other.tokens, strict=False):
            if not _is_number(left.logprob) or not _is_number(right.logprob):
                break
            pairs.append((left, right))
            if left.token != right.token:
                break
    return pairs


def _rank_of(token: str, entry: TokenLogprob) -> int | None:
    """Position of ``token`` in ``entry``'s alternatives, or ``None`` if absent."""
    for index, (candidate_token, _) in enumerate(_usable_top(entry)):
        if candidate_token == token:
            return index
    return None


def _rank_identity_test(
    pairs: list[tuple[TokenLogprob, TokenLogprob]],
) -> tuple[float, dict[str, Any]] | None:
    """Chi-square on which rank of the baseline's head the candidate emitted.

    Expected frequencies come from the baseline's own reported probabilities
    averaged over the aligned positions, so they are a prediction made before
    the candidate's tokens are looked at rather than a fit to them. The trailing
    cells are pooled until every expectation reaches ``_MIN_EXPECTED``.
    """
    if len(pairs) < _MIN_ALIGNED:
        return None

    width = max(len(_usable_top(right)) for _, right in pairs)
    if width < 2:
        return None

    observed = [0.0] * (width + 1)
    expected_mass = [0.0] * (width + 1)
    for left, right in pairs:
        rank = _rank_of(left.token, right)
        observed[rank if rank is not None else width] += 1.0
        top = _usable_top(right)
        head = 0.0
        for index, (_, value) in enumerate(top):
            probability = math.exp(value)
            expected_mass[index] += probability
            head += probability
        expected_mass[width] += max(0.0, 1.0 - head)

    total = math.fsum(expected_mass)
    if total <= 0.0:
        return None
    expected = [mass / total * len(pairs) for mass in expected_mass]

    observed, expected = _pool_cells(observed, expected)
    if len(observed) < 2 or min(expected) < _MIN_EXPECTED:
        return None

    statistic, p_value = chi_square_gof(observed, expected)
    return p_value, {
        "test": "chi-square on emitted-token rank in the baseline's top-k",
        "chi_square": round(statistic, 4),
        "categories": len(observed),
        "observed": [int(value) for value in observed],
        "expected": [round(value, 2) for value in expected],
    }


def _pool_cells(
    observed: list[float], expected: list[float]
) -> tuple[list[float], list[float]]:
    """Merge sparse cells into neighbours until every expectation reaches the floor.

    The thinly-populated cells are not always the trailing ones -- when most of
    the distribution's mass sits outside the reported head, the middle ranks are
    the sparse ones -- so the smallest cell is merged into its smaller neighbour
    repeatedly. Merging adjacent cells keeps the categories ordered by rank,
    which is what makes the pooled table interpretable.
    """
    obs = list(observed)
    exp = list(expected)
    while len(exp) > 2 and min(exp) < _MIN_EXPECTED:
        index = exp.index(min(exp))
        if index == 0:
            target = 1
        elif index == len(exp) - 1:
            target = index - 1
        else:
            target = index - 1 if exp[index - 1] <= exp[index + 1] else index + 1
        moved_expected = exp.pop(index)
        moved_observed = obs.pop(index)
        if target > index:
            target -= 1
        exp[target] += moved_expected
        obs[target] += moved_observed
    return obs, exp


def _sampling_check(samples: list[_Sample]) -> dict[str, Any]:
    """Whether the emitted tokens look like draws from the reported distribution.

    Compares how often the most likely token was actually emitted against how
    often the endpoint's own probabilities say it should have been. A greedy
    decode emits it every time whatever the probabilities say, and both rank
    tests assume otherwise.
    """
    observed = 0
    expected = 0.0
    counted = 0
    for sample in samples:
        for entry in sample.tokens:
            top = _usable_top(entry)
            if len(top) < 2:
                continue
            mass = math.fsum(math.exp(value) for _, value in top)
            if mass <= 0.0:
                continue
            counted += 1
            expected += math.exp(top[0][1]) / mass
            if entry.token == top[0][0]:
                observed += 1

    if counted == 0:
        return {
            "positions": 0,
            "observed_argmax_share": 0.0,
            "expected_argmax_share": 0.0,
            "looks_sampled": False,
        }

    observed_share = observed / counted
    expected_share = expected / counted
    return {
        "positions": counted,
        "observed_argmax_share": round(observed_share, 4),
        "expected_argmax_share": round(expected_share, 4),
        "looks_sampled": observed_share - expected_share <= _GREEDY_MARGIN,
    }


def _rank_uniformity_test(
    pairs: list[tuple[TokenLogprob, TokenLogprob]], rng: random.Random
) -> tuple[float, dict[str, Any]] | None:
    """Chi-square that the baseline's token sits at a uniform percentile of the candidate's.

    The percentile is the candidate's own probability mass above the baseline's
    emitted token, plus a *random* fraction of that token's own mass. The
    randomisation is not a flourish. A next-token distribution is discrete with
    a handful of large atoms, and the ordinary probability-integral transform is
    uniform only for a continuous variable: with five atoms the mid-point
    convention puts every observation on one of five values, which a uniformity
    test rejects on identical endpoints. Drawing the offset uniformly inside the
    observed token's own atom makes the transform exactly uniform under the
    null, at the cost of some power. The draw comes from the run's seeded RNG,
    so the result is still reproducible.

    Probabilities are renormalised over the reported head, which is legitimate
    because the comparison is conditioned on the token being in that head:
    positions where the baseline's token is outside it are excluded and counted
    separately, since turning "somewhere below rank k" into a percentile would
    mean inventing the tail the endpoint did not send.
    """
    percentiles: list[float] = []
    outside = 0
    for left, right in pairs:
        top = _usable_top(left)
        if len(top) < 2:
            continue
        mass = math.fsum(math.exp(value) for _, value in top)
        if mass <= 0.0:
            continue
        above = 0.0
        own: float | None = None
        for token, value in top:
            probability = math.exp(value) / mass
            if token == right.token:
                own = probability
                break
            above += probability
        if own is None:
            outside += 1
            continue
        percentiles.append(min(1.0, above + rng.random() * own))

    if len(percentiles) < _UNIFORMITY_BINS * _MIN_EXPECTED:
        return None

    counts = [0.0] * _UNIFORMITY_BINS
    for value in percentiles:
        index = min(_UNIFORMITY_BINS - 1, int(value * _UNIFORMITY_BINS))
        counts[index] += 1.0
    expected = [len(percentiles) / _UNIFORMITY_BINS] * _UNIFORMITY_BINS

    statistic, p_value = chi_square_gof(counts, expected)
    return p_value, {
        "test": "chi-square uniformity of the baseline token's percentile rank",
        "chi_square": round(statistic, 4),
        "bins": _UNIFORMITY_BINS,
        "observed": [int(value) for value in counts],
        "scored_positions": len(percentiles),
        "positions_outside_head": outside,
    }


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #


def _llr_from_pvalue(p_value: float, *, concentration: float = _ALTERNATIVE_CONCENTRATION) -> float:
    """Signed LLR from a p-value, under an explicit model of the alternative.

    Under the null a p-value is uniform, so its density is 1. The alternative is
    modelled as ``Beta(concentration, 1)``, whose density is
    ``a * p ** (a - 1)`` and which piles mass near zero. The log ratio is then
    ``-ln(a) - (a - 1) * ln(p)``: positive for a large p-value, negative for a
    small one, and finite everywhere except exactly zero, which is floored.

    Stating the alternative explicitly is the point. Thresholding a p-value at
    0.05 and calling the result evidence hides an assumption about the
    alternative rather than removing it.
    """
    p = min(1.0, max(1e-12, p_value))
    return -math.log(concentration) - (concentration - 1.0) * math.log(p)


def _discounted(llr: float) -> float:
    """Apply the serving-configuration discount and clamp to the probe's cap."""
    scaled = llr * CONFIGURATION_DISCOUNT
    return max(-MODERATE, min(MODERATE, scaled))
