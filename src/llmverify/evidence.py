"""Evidence accumulation and the final verdict.

Every probe emits :class:`Evidence`. Each piece carries a log-likelihood ratio

    llr = ln( P(observation | endpoint really serves the claimed model)
            / P(observation | it serves something else) )

so a positive value supports the provider's claim and a negative value refutes
it. Combining evidence is then addition, and the posterior follows from Bayes'
rule in odds form:

    ln(posterior odds) = ln(prior odds) + sum(llr)

Two deliberate departures from naive addition:

**Per-probe caps.** No single probe may contribute more than its declared
``cap`` nats. A model that emits an unexpected token should not be able to
single-handedly convict a provider.

**Per-family damping.** Probes sharing a ``family`` measure overlapping things
(five tokenizer probes are not five independent experiments). Within a family
the LLRs are summed and then damped toward the family cap, so correlated
evidence saturates instead of compounding.

Neither is statistically exact -- exactness would require a joint model of
probe correlations that nobody has. They are conservative in the direction that
matters: they make it harder, not easier, to accuse a provider of fraud.
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Evidence",
    "EvidenceStatus",
    "Verdict",
    "VerdictReport",
    "aggregate",
    "llr_from_probability",
]

#: Natural-log LLR magnitudes, named for readability at probe sites.
NEGLIGIBLE = 0.0
WEAK = math.log(3)  # ~1.1 nats, a 3:1 likelihood ratio
MODERATE = math.log(10)  # ~2.3 nats
STRONG = math.log(100)  # ~4.6 nats
DECISIVE = math.log(10_000)  # ~9.2 nats


class EvidenceStatus(str, enum.Enum):
    OK = "ok"
    #: The probe does not apply here (e.g. an Anthropic-only probe on OpenAI).
    SKIPPED = "skipped"
    #: The endpoint genuinely cannot support the probe. Often informative, but
    #: on its own not proof of anything.
    UNSUPPORTED = "unsupported"
    #: The probe failed for operational reasons. Contributes zero LLR.
    ERROR = "error"
    #: The probe stopped early because the run ran out of budget.
    TRUNCATED = "truncated"


class Verdict(str, enum.Enum):
    MATCH = "MATCH"
    LIKELY_MATCH = "LIKELY_MATCH"
    INCONCLUSIVE = "INCONCLUSIVE"
    LIKELY_MISMATCH = "LIKELY_MISMATCH"
    MISMATCH = "MISMATCH"
    #: The provider behaves differently on recognisable benchmark inputs than on
    #: semantically equivalent paraphrases. This is reported separately because
    #: it is not a point on the match/mismatch axis -- it means the measurement
    #: itself was manipulated, so every other number in the run is suspect.
    EVASION = "EVASION"

    @property
    def is_adverse(self) -> bool:
        return self in (Verdict.LIKELY_MISMATCH, Verdict.MISMATCH, Verdict.EVASION)


@dataclass(slots=True)
class Evidence:
    """One observation and what it implies.

    ``llr`` must be signed from the point of view of the provider's *claim*:
    positive supports "this really is the claimed model", negative refutes it.
    """

    probe: str
    label: str
    llr: float
    #: Maximum |llr| this single piece may contribute after clamping.
    cap: float = STRONG
    #: Probes measuring the same underlying property share a family, so their
    #: correlated evidence is damped rather than summed outright.
    family: str = "misc"
    status: EvidenceStatus = EvidenceStatus.OK
    #: One line a human can read without knowing how the probe works.
    detail: str = ""
    #: Machine-readable specifics: measurements, p-values, thresholds, samples.
    data: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens: int = 0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status is not EvidenceStatus.OK:
            self.llr = 0.0
        elif not math.isfinite(self.llr):
            self.llr = math.copysign(self.cap, self.llr)
        self.llr = max(-abs(self.cap), min(abs(self.cap), self.llr))

    @property
    def bans(self) -> float:
        """LLR expressed in base-10 "bans" -- each ban is a 10:1 likelihood ratio."""
        return self.llr / math.log(10)

    @property
    def supports(self) -> bool:
        return self.llr > 0

    @property
    def refutes(self) -> bool:
        return self.llr < 0


def llr_from_probability(
    p_if_genuine: float, p_if_substituted: float, *, floor: float = 1e-6
) -> float:
    """LLR from two explicitly modelled likelihoods.

    Both are clamped away from zero so that a single confidently-wrong model
    assumption cannot produce an infinite LLR.
    """
    a = max(floor, min(1.0, p_if_genuine))
    b = max(floor, min(1.0, p_if_substituted))
    return math.log(a / b)


@dataclass(slots=True)
class VerdictReport:
    verdict: Verdict
    #: Posterior probability that the provider serves the claimed model.
    probability: float
    total_llr: float
    prior_odds: float
    evidence: list[Evidence]
    family_totals: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def total_bans(self) -> float:
        return self.total_llr / math.log(10)

    @property
    def cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.evidence)

    @property
    def tokens(self) -> int:
        return sum(e.tokens for e in self.evidence)

    @property
    def duration_s(self) -> float:
        return sum(e.duration_s for e in self.evidence)

    def by_status(self, status: EvidenceStatus) -> list[Evidence]:
        return [e for e in self.evidence if e.status is status]

    @property
    def top_refuting(self) -> list[Evidence]:
        return sorted((e for e in self.evidence if e.refutes), key=lambda e: e.llr)

    @property
    def top_supporting(self) -> list[Evidence]:
        return sorted((e for e in self.evidence if e.supports), key=lambda e: -e.llr)


#: Damping ceiling per evidence family, in nats. Correlated probes within a
#: family saturate here instead of accumulating without bound.
FAMILY_CAPS: dict[str, float] = {
    "metadata": MODERATE,
    "api_surface": STRONG,
    "tokenizer": STRONG,
    "token_accounting": DECISIVE,
    "cryptographic": DECISIVE,
    "knowledge": MODERATE,
    "self_report": WEAK,
    "determinism": WEAK,
    "capability": STRONG,
    "long_context": STRONG,
    "tool_use": MODERATE,
    "multilingual": MODERATE,
    "vision": STRONG,
    "performance": WEAK,
    "distribution": STRONG,
    "benchmark": DECISIVE,
    "evasion": DECISIVE,
    "misc": MODERATE,
}

#: Posterior-probability boundaries for each verdict band.
THRESHOLDS: tuple[tuple[float, Verdict], ...] = (
    (0.99, Verdict.MATCH),
    (0.90, Verdict.LIKELY_MATCH),
    (0.10, Verdict.INCONCLUSIVE),
    (0.01, Verdict.LIKELY_MISMATCH),
    (0.00, Verdict.MISMATCH),
)


def _damp(total: float, cap: float) -> float:
    """Squash ``total`` into ``(-cap, cap)`` while staying linear near zero.

    ``cap * tanh(total / cap)`` leaves small amounts of evidence untouched and
    compresses large amounts, which is exactly the behaviour we want from a
    group of probes that all measure the same underlying property.
    """
    if cap <= 0:
        return 0.0
    return cap * math.tanh(total / cap)


def aggregate(
    evidence: list[Evidence],
    *,
    prior_odds: float = 1.0,
    family_caps: dict[str, float] | None = None,
    min_effective_probes: int = 3,
) -> VerdictReport:
    """Combine evidence into a posterior and a verdict.

    ``prior_odds`` is the prior odds that the claim is true. The default of 1.0
    (50/50) is deliberately uncommitted: this tool is used both on providers
    the user already distrusts and on ones being spot-checked.

    A run with fewer than ``min_effective_probes`` contributing probes is forced
    to ``INCONCLUSIVE`` regardless of the arithmetic. Two probes agreeing is not
    an audit.
    """
    caps = {**FAMILY_CAPS, **(family_caps or {})}
    notes: list[str] = []

    by_family: dict[str, float] = defaultdict(float)
    for e in evidence:
        if e.status is EvidenceStatus.OK:
            by_family[e.family] += e.llr

    family_totals = {
        fam: _damp(total, caps.get(fam, caps["misc"])) for fam, total in by_family.items()
    }
    total_llr = sum(family_totals.values())

    log_posterior_odds = math.log(max(prior_odds, 1e-12)) + total_llr
    # Numerically safe logistic.
    if log_posterior_odds >= 0:
        probability = 1.0 / (1.0 + math.exp(-log_posterior_odds))
    else:
        ex = math.exp(log_posterior_odds)
        probability = ex / (1.0 + ex)

    effective = sum(1 for e in evidence if e.status is EvidenceStatus.OK and abs(e.llr) > 1e-9)

    verdict = Verdict.INCONCLUSIVE
    for threshold, candidate in THRESHOLDS:
        if probability >= threshold:
            verdict = candidate
            break

    if effective < min_effective_probes and verdict is not Verdict.INCONCLUSIVE:
        notes.append(
            f"Only {effective} probe(s) produced usable evidence "
            f"(minimum {min_effective_probes}); verdict forced to INCONCLUSIVE."
        )
        verdict = Verdict.INCONCLUSIVE

    # Evasion is not a point on the match axis. If the provider treats
    # recognisable benchmark inputs differently from paraphrases, every other
    # measurement in the run was taken under conditions the provider chose.
    evading = [
        e
        for e in evidence
        if e.family == "evasion" and e.status is EvidenceStatus.OK and e.llr < -MODERATE
    ]
    if evading:
        notes.append(
            "Evasion detected: the endpoint's behaviour depends on whether an input "
            "is recognisable as a benchmark item. Treat all other measurements in "
            "this run as measured under the provider's control, not yours."
        )
        verdict = Verdict.EVASION

    return VerdictReport(
        verdict=verdict,
        probability=probability,
        total_llr=total_llr,
        prior_odds=prior_odds,
        evidence=list(evidence),
        family_totals=family_totals,
        notes=notes,
    )
