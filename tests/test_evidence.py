"""Evidence clamping, family damping, and the rules that override the arithmetic.

The aggregator is where a pile of observations becomes an accusation, so every
guard it carries is tested here directly rather than through a probe: the cap on
a single piece, the damping within a family, the refusal to decide on too few
probes, the exact threshold boundaries, and evasion taking the verdict off the
match/mismatch axis entirely.
"""

from __future__ import annotations

import math

import pytest

from llmverify.evidence import (
    DECISIVE,
    FAMILY_CAPS,
    MODERATE,
    STRONG,
    THRESHOLDS,
    WEAK,
    Evidence,
    EvidenceStatus,
    Verdict,
    aggregate,
    llr_from_probability,
)


def ev(
    llr: float,
    *,
    family: str = "misc",
    cap: float = STRONG,
    status: EvidenceStatus = EvidenceStatus.OK,
    label: str = "finding",
) -> Evidence:
    return Evidence(probe="p", label=label, llr=llr, cap=cap, family=family, status=status)


# --------------------------------------------------------------------------- #
# One piece of evidence
# --------------------------------------------------------------------------- #


def test_llr_is_clamped_to_the_declared_cap() -> None:
    assert ev(50.0, cap=MODERATE).llr == pytest.approx(MODERATE)
    assert ev(-50.0, cap=MODERATE).llr == pytest.approx(-MODERATE)
    assert ev(1.0, cap=MODERATE).llr == pytest.approx(1.0)


def test_infinite_llr_becomes_the_cap_with_its_sign() -> None:
    assert ev(math.inf, cap=STRONG).llr == pytest.approx(STRONG)
    assert ev(-math.inf, cap=STRONG).llr == pytest.approx(-STRONG)


def test_a_negative_cap_is_read_as_a_magnitude() -> None:
    assert ev(10.0, cap=-2.0).llr == pytest.approx(2.0)


def test_non_ok_status_contributes_nothing() -> None:
    for status in (
        EvidenceStatus.SKIPPED,
        EvidenceStatus.UNSUPPORTED,
        EvidenceStatus.ERROR,
        EvidenceStatus.TRUNCATED,
    ):
        assert ev(9.0, status=status).llr == 0.0


def test_bans_supports_and_refutes() -> None:
    item = ev(math.log(100))
    assert item.bans == pytest.approx(2.0)
    assert item.supports and not item.refutes
    assert ev(-1.0).refutes and not ev(-1.0).supports
    assert not ev(0.0).supports and not ev(0.0).refutes


def test_llr_from_probability_is_bounded_by_the_floor() -> None:
    assert llr_from_probability(0.9, 0.1) == pytest.approx(math.log(9.0))
    assert llr_from_probability(1.0, 0.0) == pytest.approx(math.log(1e6))
    assert llr_from_probability(0.0, 1.0) == pytest.approx(math.log(1e-6))


# --------------------------------------------------------------------------- #
# Damping
# --------------------------------------------------------------------------- #


def test_correlated_probes_in_one_family_saturate_instead_of_compounding() -> None:
    """Five tokenizer measurements are one opinion, not five."""
    many = [ev(2.0, family="tokenizer", cap=STRONG) for _ in range(5)]
    report = aggregate(many)
    cap = FAMILY_CAPS["tokenizer"]
    assert report.family_totals["tokenizer"] < 10.0
    assert report.family_totals["tokenizer"] < cap
    assert report.family_totals["tokenizer"] == pytest.approx(cap * math.tanh(10.0 / cap))


def test_damping_is_linear_near_zero() -> None:
    report = aggregate([ev(0.01, family="metadata")])
    assert report.family_totals["metadata"] == pytest.approx(0.01, rel=1e-3)


def test_families_accumulate_independently() -> None:
    report = aggregate(
        [
            ev(1.0, family="metadata"),
            ev(1.0, family="tokenizer"),
            ev(1.0, family="benchmark"),
        ]
    )
    assert set(report.family_totals) == {"metadata", "tokenizer", "benchmark"}
    assert report.total_llr == pytest.approx(sum(report.family_totals.values()))


def test_an_unknown_family_falls_back_to_the_misc_cap() -> None:
    report = aggregate([ev(100.0, family="not-a-family", cap=DECISIVE)])
    assert abs(report.family_totals["not-a-family"]) <= FAMILY_CAPS["misc"]


def test_custom_family_caps_override_the_defaults() -> None:
    report = aggregate(
        [ev(5.0, family="metadata")], family_caps={"metadata": 0.5}
    )
    assert report.family_totals["metadata"] <= 0.5


# --------------------------------------------------------------------------- #
# Minimum probes
# --------------------------------------------------------------------------- #


def test_two_agreeing_probes_are_not_an_audit() -> None:
    report = aggregate([ev(6.0, family="metadata"), ev(6.0, family="tokenizer")])
    assert report.probability > 0.99
    assert report.verdict is Verdict.INCONCLUSIVE
    assert any("minimum 3" in note for note in report.notes)


def test_three_contributing_probes_are_enough() -> None:
    report = aggregate(
        [
            ev(6.0, family="metadata"),
            ev(6.0, family="tokenizer"),
            ev(6.0, family="benchmark"),
        ]
    )
    assert report.verdict is Verdict.MATCH
    assert report.notes == []


def test_zero_weight_and_skipped_findings_do_not_count_towards_the_minimum() -> None:
    report = aggregate(
        [
            ev(6.0, family="metadata"),
            ev(6.0, family="tokenizer"),
            ev(0.0, family="benchmark"),
            ev(6.0, family="vision", status=EvidenceStatus.SKIPPED),
        ]
    )
    assert report.verdict is Verdict.INCONCLUSIVE


def test_the_minimum_can_be_relaxed_by_the_caller() -> None:
    evidence = [ev(6.0, family="metadata"), ev(6.0, family="tokenizer")]
    assert aggregate(evidence, min_effective_probes=2).verdict is Verdict.MATCH


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


#: Families whose cap is DECISIVE, so a third of any reachable posterior fits
#: inside one of them and the inversion below stays in the domain of atanh.
_WIDE_FAMILIES = ("token_accounting", "cryptographic", "benchmark")


def _report_at(probability: float):
    """Three probes whose damped total lands the posterior exactly on ``probability``."""
    share = math.log(probability / (1.0 - probability)) / 3.0
    evidence = [
        ev(_undamped(share, FAMILY_CAPS[family]), family=family, cap=DECISIVE)
        for family in _WIDE_FAMILIES
    ]
    return aggregate(evidence)


def _undamped(damped: float, cap: float) -> float:
    """Invert ``cap * tanh(x / cap)`` so a family total lands exactly on ``damped``."""
    return cap * math.atanh(damped / cap)


def test_the_threshold_table_is_the_documented_one() -> None:
    """0.99, 0.90, 0.10 and 0.01, descending, each owning the band above it."""
    assert THRESHOLDS == (
        (0.99, Verdict.MATCH),
        (0.90, Verdict.LIKELY_MATCH),
        (0.10, Verdict.INCONCLUSIVE),
        (0.01, Verdict.LIKELY_MISMATCH),
        (0.00, Verdict.MISMATCH),
    )
    assert [t for t, _v in THRESHOLDS] == sorted((t for t, _v in THRESHOLDS), reverse=True)


@pytest.mark.parametrize(
    ("probability", "verdict"),
    [
        (0.9999, Verdict.MATCH),
        (0.99 + 1e-9, Verdict.MATCH),
        (0.99 - 1e-9, Verdict.LIKELY_MATCH),
        (0.90 + 1e-9, Verdict.LIKELY_MATCH),
        (0.90 - 1e-9, Verdict.INCONCLUSIVE),
        (0.50, Verdict.INCONCLUSIVE),
        (0.10 + 1e-9, Verdict.INCONCLUSIVE),
        (0.10 - 1e-9, Verdict.LIKELY_MISMATCH),
        (0.01 + 1e-9, Verdict.LIKELY_MISMATCH),
        (0.01 - 1e-9, Verdict.MISMATCH),
        (0.0001, Verdict.MISMATCH),
    ],
)
def test_threshold_boundaries_split_the_bands_where_documented(
    probability: float, verdict: Verdict
) -> None:
    """Each band is entered within a nanoprobability of its stated boundary.

    Landing a posterior on a boundary *exactly* is not reachable through
    :func:`aggregate`: the value comes out of a logistic, and no representable
    log-odds maps to 0.90 in binary floating point. The table itself is pinned
    by the test above; this one pins the bands to a resolution nine orders of
    magnitude finer than any verdict depends on.
    """
    report = _report_at(probability)
    assert report.probability == pytest.approx(probability, abs=1e-12)
    assert report.verdict is verdict


def test_prior_odds_move_the_posterior() -> None:
    evidence = [ev(1.0, family=f) for f in ("metadata", "tokenizer", "benchmark")]
    sceptical = aggregate(evidence, prior_odds=0.01)
    neutral = aggregate(evidence, prior_odds=1.0)
    assert sceptical.probability < neutral.probability
    assert neutral.total_llr == pytest.approx(sceptical.total_llr)


def test_no_evidence_at_all_is_inconclusive_at_the_prior() -> None:
    report = aggregate([])
    assert report.verdict is Verdict.INCONCLUSIVE
    assert report.probability == pytest.approx(0.5)
    assert report.family_totals == {}


# --------------------------------------------------------------------------- #
# Evasion
# --------------------------------------------------------------------------- #


def test_evasion_overrides_a_match_verdict() -> None:
    """It is not a point on the match axis; it says the measurements were chosen."""
    report = aggregate(
        [
            ev(6.0, family="metadata"),
            ev(6.0, family="tokenizer"),
            ev(6.0, family="benchmark"),
            ev(-DECISIVE, family="evasion", cap=DECISIVE),
        ]
    )
    assert report.verdict is Verdict.EVASION
    assert report.probability > 0.5
    assert any("Evasion detected" in note for note in report.notes)


def test_evasion_overrides_a_mismatch_verdict_too() -> None:
    report = aggregate(
        [
            ev(-6.0, family="metadata"),
            ev(-6.0, family="tokenizer"),
            ev(-6.0, family="benchmark"),
            ev(-DECISIVE, family="evasion", cap=DECISIVE),
        ]
    )
    assert report.verdict is Verdict.EVASION


def test_evasion_overrides_even_below_the_minimum_probe_count() -> None:
    report = aggregate([ev(-DECISIVE, family="evasion", cap=DECISIVE)])
    assert report.verdict is Verdict.EVASION


def test_weak_evasion_evidence_does_not_trip_the_override() -> None:
    """Only a finding past MODERATE takes the verdict off the match axis."""
    report = aggregate(
        [
            ev(6.0, family="metadata"),
            ev(6.0, family="tokenizer"),
            ev(-MODERATE, family="evasion", cap=DECISIVE),
        ]
    )
    assert report.verdict is not Verdict.EVASION


def test_positive_evasion_evidence_never_triggers_the_override() -> None:
    report = aggregate(
        [
            ev(1.0, family="metadata"),
            ev(1.0, family="tokenizer"),
            ev(DECISIVE, family="evasion", cap=DECISIVE),
        ]
    )
    assert report.verdict is not Verdict.EVASION


def test_evasion_evidence_in_a_non_ok_status_is_ignored() -> None:
    report = aggregate(
        [
            ev(1.0, family="metadata"),
            ev(1.0, family="tokenizer"),
            ev(1.0, family="benchmark"),
            ev(-DECISIVE, family="evasion", status=EvidenceStatus.TRUNCATED),
        ]
    )
    assert report.verdict is not Verdict.EVASION


# --------------------------------------------------------------------------- #
# Report accessors
# --------------------------------------------------------------------------- #


def test_report_sorts_and_totals_its_evidence() -> None:
    report = aggregate(
        [
            Evidence(probe="a", label="x", llr=-3.0, family="metadata", cost_usd=0.5, tokens=10),
            Evidence(probe="b", label="y", llr=-1.0, family="tokenizer", cost_usd=0.25, tokens=5),
            Evidence(probe="c", label="z", llr=2.0, family="benchmark", duration_s=1.5),
            Evidence(
                probe="d",
                label="w",
                llr=0.0,
                family="vision",
                status=EvidenceStatus.UNSUPPORTED,
            ),
        ]
    )
    assert [e.probe for e in report.top_refuting] == ["a", "b"]
    assert [e.probe for e in report.top_supporting] == ["c"]
    assert report.cost_usd == pytest.approx(0.75)
    assert report.tokens == 15
    assert report.duration_s == pytest.approx(1.5)
    assert [e.probe for e in report.by_status(EvidenceStatus.UNSUPPORTED)] == ["d"]
    assert report.total_bans == pytest.approx(report.total_llr / math.log(10))


def test_verdict_adverse_classification() -> None:
    assert Verdict.MISMATCH.is_adverse
    assert Verdict.LIKELY_MISMATCH.is_adverse
    assert Verdict.EVASION.is_adverse
    assert not Verdict.MATCH.is_adverse
    assert not Verdict.LIKELY_MATCH.is_adverse
    assert not Verdict.INCONCLUSIVE.is_adverse


def test_named_llr_magnitudes_are_ordered() -> None:
    assert 0.0 < WEAK < MODERATE < STRONG < DECISIVE
    assert pytest.approx(math.log(3)) == WEAK
    assert pytest.approx(math.log(10_000)) == DECISIVE
