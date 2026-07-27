"""The statistical layer, against reference values and against its own promises.

Two kinds of test. The first pins each function to a value computed
independently -- from a table, from a textbook formula, or from an exact
enumeration -- so a rewrite that changes an answer cannot pass. The second is
property-based: interval containment, monotonicity, and the two claims the
module docstrings actually make about their own error rates.

Nothing here touches the network or a provider.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from llmverify.stats import (
    SPRT,
    SPRTDecision,
    agresti_coull,
    benjamini_hochberg,
    beta_ppf,
    binomial_test,
    chi2_sf,
    chi_square_gof,
    clopper_pearson,
    detectable_effect,
    discriminative_power,
    fisher_exact,
    hamming_kernel,
    holm_bonferroni,
    ks_two_sample,
    mcnemar,
    mmd_hamming,
    normal_cdf,
    normal_ppf,
    normal_sf,
    regularized_incomplete_beta,
    required_n,
    two_proportion_z,
    wilson_interval,
)

# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


def test_normal_ppf_matches_the_standard_critical_values() -> None:
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert normal_ppf(0.95) == pytest.approx(1.644854, abs=1e-6)
    assert normal_ppf(0.995) == pytest.approx(2.575829, abs=1e-6)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_normal_cdf_and_sf_are_complementary() -> None:
    for x in (-4.0, -1.0, 0.0, 0.5, 3.3):
        assert normal_cdf(x) + normal_sf(x) == pytest.approx(1.0, abs=1e-12)
    assert normal_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert normal_cdf(1.96) == pytest.approx(0.975002, abs=1e-6)


def test_normal_ppf_inverts_normal_cdf() -> None:
    for p in (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999):
        assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_chi2_survival_at_the_five_percent_critical_value() -> None:
    assert chi2_sf(3.84, 1) == pytest.approx(0.05, abs=5e-4)
    assert chi2_sf(5.991, 2) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(0.0, 3) == pytest.approx(1.0, abs=1e-12)


def test_regularized_incomplete_beta_is_symmetric() -> None:
    # I_x(a, b) = 1 - I_(1-x)(b, a) is the defining identity, and it exercises
    # both branches of the continued fraction.
    for x, a, b in ((0.3, 2.0, 5.0), (0.75, 4.5, 1.5), (0.5, 3.0, 3.0)):
        left = regularized_incomplete_beta(x, a, b)
        right = 1.0 - regularized_incomplete_beta(1.0 - x, b, a)
        assert left == pytest.approx(right, abs=1e-12)


def test_beta_ppf_inverts_the_incomplete_beta() -> None:
    for p, a, b in ((0.025, 3.0, 8.0), (0.5, 2.0, 2.0), (0.975, 9.0, 2.0)):
        assert regularized_incomplete_beta(beta_ppf(p, a, b), a, b) == pytest.approx(p, abs=1e-9)


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #


def test_wilson_interval_matches_the_published_value() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4901, abs=5e-4)
    assert high == pytest.approx(0.9433, abs=5e-4)


def test_clopper_pearson_at_zero_successes() -> None:
    low, high = clopper_pearson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.3085, abs=5e-4)


def test_clopper_pearson_is_the_widest_of_the_three() -> None:
    """Its coverage guarantee is bought with width, which is the whole trade."""
    for successes in (1, 5, 9):
        cp = clopper_pearson(successes, 10)
        wilson = wilson_interval(successes, 10)
        agresti = agresti_coull(successes, 10)
        assert cp[0] <= wilson[0] and cp[1] >= wilson[1]
        assert cp[1] - cp[0] >= agresti[1] - agresti[0]


def test_intervals_contain_the_point_estimate_and_stay_in_range() -> None:
    for n in (1, 5, 20, 97):
        for successes in (0, n // 3, n // 2, n):
            phat = successes / n
            for interval in (
                wilson_interval(successes, n),
                clopper_pearson(successes, n),
                agresti_coull(successes, n),
            ):
                low, high = interval
                assert 0.0 <= low <= high <= 1.0
                assert low <= phat <= high


def test_interval_width_shrinks_as_the_sample_grows() -> None:
    widths = [
        wilson_interval(round(0.8 * n), n)[1] - wilson_interval(round(0.8 * n), n)[0]
        for n in (10, 40, 160, 640)
    ]
    assert widths == sorted(widths, reverse=True)


def test_zero_observations_constrain_nothing() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    assert agresti_coull(0, 0) == (0.0, 1.0)


def test_intervals_reject_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 10)
    with pytest.raises(ValueError):
        agresti_coull(1, 10, confidence=1.0)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_two_proportion_z_uses_the_pooled_standard_error() -> None:
    """80/100 against 60/100.

    The pooled standard error is the one correct under the null being tested --
    that the two proportions are equal -- and is what this implementation
    documents. It gives z = 3.086 and p = 0.00203. An unpooled standard error
    would give p = 0.00157; that is the right quantity for an interval on the
    difference and the wrong one for a test of equality, so the smaller number
    is deliberately not what this returns.
    """
    z, p = two_proportion_z(80, 100, 60, 100)
    assert z == pytest.approx(3.086067, abs=1e-6)
    assert p == pytest.approx(0.0020282, abs=1e-7)


def test_two_proportion_z_is_symmetric_and_degenerate_cases_are_safe() -> None:
    forward = two_proportion_z(80, 100, 60, 100)
    backward = two_proportion_z(60, 100, 80, 100)
    assert forward[0] == pytest.approx(-backward[0], abs=1e-12)
    assert forward[1] == pytest.approx(backward[1], abs=1e-12)
    assert two_proportion_z(10, 10, 10, 10) == (0.0, 1.0)
    with pytest.raises(ValueError):
        two_proportion_z(1, 0, 1, 1)


def test_fisher_exact_matches_an_independent_enumeration() -> None:
    assert fisher_exact(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-6)
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0, abs=1e-12)
    # One-sided tails must bracket the two-sided value on this table.
    assert fisher_exact(1, 9, 11, 3, "less") < fisher_exact(1, 9, 11, 3)
    assert fisher_exact(1, 9, 11, 3, "greater") > 0.99


def test_fisher_exact_sums_to_one_over_the_hypergeometric_support() -> None:
    """The two one-sided tails overlap in exactly the observed table."""
    a, b, c, d = 3, 7, 8, 2
    less = fisher_exact(a, b, c, d, "less")
    greater = fisher_exact(a, b, c, d, "greater")
    point = fisher_exact(a, b, c, d, "less") + fisher_exact(a, b, c, d, "greater") - 1.0
    assert less + greater - point == pytest.approx(1.0, abs=1e-9)


def test_binomial_test_against_exact_arithmetic() -> None:
    # P(X <= 2) for n=10, p=0.5 is (1 + 10 + 45) / 1024.
    assert binomial_test(2, 10, 0.5, "less") == pytest.approx(56 / 1024, abs=1e-12)
    assert binomial_test(10, 10, 0.5, "greater") == pytest.approx(1 / 1024, abs=1e-12)
    assert binomial_test(5, 10, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert binomial_test(0, 0, 0.5) == 1.0


def test_mcnemar_is_the_exact_binomial_on_the_discordant_pairs() -> None:
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(1, 9) == pytest.approx(binomial_test(1, 10, 0.5), abs=1e-12)
    assert mcnemar(1, 9) == pytest.approx(0.021484, abs=1e-6)
    assert mcnemar(9, 1) == pytest.approx(mcnemar(1, 9), abs=1e-12)
    # Yates' correction gives (|b - c| - 1)^2 / (b + c) = 4.9 on one degree of
    # freedom. Close to the exact test but not equal, which is the point of
    # defaulting to exact at these sample sizes.
    assert mcnemar(1, 9, exact=False) == pytest.approx(chi2_sf(4.9, 1), abs=1e-12)
    assert mcnemar(1, 9, exact=False) == pytest.approx(0.026857, abs=1e-6)
    with pytest.raises(ValueError):
        mcnemar(-1, 3)


def test_ks_two_sample_on_identical_samples_finds_nothing() -> None:
    sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    d, p = ks_two_sample(sample, sample)
    assert d == 0.0
    assert p == pytest.approx(1.0, abs=1e-12)


def test_ks_two_sample_separates_disjoint_samples() -> None:
    d, p = ks_two_sample([float(i) for i in range(30)], [float(i) + 100 for i in range(30)])
    assert d == pytest.approx(1.0, abs=1e-12)
    assert p < 1e-6


def test_ks_two_sample_handles_ties_without_overcounting() -> None:
    d, _p = ks_two_sample([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 2.0, 2.0])
    assert d == pytest.approx(0.25, abs=1e-12)
    with pytest.raises(ValueError):
        ks_two_sample([], [1.0])


def test_chi_square_goodness_of_fit() -> None:
    statistic, p = chi_square_gof([10, 20, 30], [20, 20, 20])
    assert statistic == pytest.approx(10.0, abs=1e-12)
    assert p == pytest.approx(chi2_sf(10.0, 2), abs=1e-12)
    with pytest.raises(ValueError):
        chi_square_gof([1, 2], [1, 0])


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


def test_required_n_for_a_ten_point_gap() -> None:
    assert required_n(0.9, 0.8) == 199
    assert required_n(0.8, 0.9) == 199


def test_required_n_grows_as_the_gap_narrows() -> None:
    sizes = [required_n(0.918, other) for other in (0.874, 0.891, 0.913)]
    assert sizes == sorted(sizes)
    # The two leading 2026 models on GPQA Diamond, half a point apart. The
    # benchmark has 198 items.
    assert required_n(0.918, 0.913) > 40_000


def test_required_n_refuses_an_impossible_question() -> None:
    with pytest.raises(ValueError):
        required_n(0.9, 0.9)
    with pytest.raises(ValueError):
        required_n(1.5, 0.5)


def test_detectable_effect_inverts_required_n() -> None:
    effect = detectable_effect(199, 0.9)
    assert effect == pytest.approx(0.10, abs=0.01)
    assert required_n(0.9, 0.9 - effect) <= 200


def test_detectable_effect_reports_the_whole_range_when_nothing_is_detectable() -> None:
    assert detectable_effect(1, 0.9) == 0.9


def test_discriminative_power_flags_a_saturated_benchmark() -> None:
    """GPQA Diamond's published 2026 spread, against the 198 items that exist."""
    scores = [91.8, 91.3, 91.9, 92.0, 91.1, 90.8, 89.1, 87.4]
    summary = discriminative_power(scores, available_items=198)
    assert summary["closest_gap_pp"] == pytest.approx(0.1, abs=1e-9)
    assert summary["n_to_separate"] > 198
    assert summary["sufficient"] is False
    assert "too coarse" in summary["warning"]


def test_discriminative_power_accepts_a_wide_benchmark() -> None:
    summary = discriminative_power([92.5, 71.6, 43.1, 22.8], available_items=200)
    assert summary["sufficient"] is True
    assert summary["warning"] is None
    assert summary["spread"] == pytest.approx(69.7, abs=1e-9)


def test_discriminative_power_rejects_inputs_it_cannot_read() -> None:
    with pytest.raises(ValueError):
        discriminative_power([150.0, 90.0])
    with pytest.raises(ValueError):
        discriminative_power([90.0])


# --------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------- #


def test_holm_bonferroni_is_monotone_and_never_below_bonferroni_on_the_smallest() -> None:
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_bonferroni(raw)
    assert adjusted == [0.03, 0.06, 0.06]
    assert adjusted[0] == pytest.approx(min(1.0, 3 * raw[0]), abs=1e-12)
    ordered = sorted(zip(raw, adjusted, strict=True))
    assert [p for _r, p in ordered] == sorted(p for _r, p in ordered)


def test_benjamini_hochberg_is_never_more_conservative_than_holm() -> None:
    raw = [0.001, 0.008, 0.02, 0.04, 0.3]
    bh = benjamini_hochberg(raw)
    holm = holm_bonferroni(raw)
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm, strict=True))
    assert all(0.0 <= value <= 1.0 for value in bh)


def test_multiplicity_corrections_pass_empty_and_single_inputs_through() -> None:
    assert holm_bonferroni([]) == []
    assert benjamini_hochberg([]) == []
    assert holm_bonferroni([0.2]) == [0.2]
    with pytest.raises(ValueError):
        holm_bonferroni([1.5])


# --------------------------------------------------------------------------- #
# Sequential testing
# --------------------------------------------------------------------------- #


def _run_sprt(p_true: float, seed: int, *, cap: int = 5000) -> tuple[int, SPRTDecision]:
    rng = random.Random(seed)
    test = SPRT(p0=0.90, p1=0.75, alpha=0.01, beta=0.05)
    for _ in range(cap):
        decision = test.update(rng.random() < p_true)
        if decision is not SPRTDecision.CONTINUE:
            return test.n, decision
    return test.n, SPRTDecision.CONTINUE


def test_sprt_rejects_hypotheses_that_are_not_ordered() -> None:
    with pytest.raises(ValueError):
        SPRT(p0=0.75, p1=0.90)
    with pytest.raises(ValueError):
        SPRT(p0=0.9, p1=0.8, alpha=0.6, beta=0.5)


def test_sprt_boundaries_bracket_zero_and_the_steps_have_the_right_signs() -> None:
    test = SPRT(p0=0.90, p1=0.75, alpha=0.01, beta=0.05)
    assert test.lower_bound < 0.0 < test.upper_bound
    assert test.upper_bound == pytest.approx(math.log(0.95 / 0.01), abs=1e-12)
    assert test.lower_bound == pytest.approx(math.log(0.05 / 0.99), abs=1e-12)
    assert test.decision is SPRTDecision.CONTINUE
    test.update(True)
    assert test.llr < 0.0
    test.reset()
    test.update(False)
    assert test.llr > 0.0


def test_sprt_realised_error_rates_are_at_or_below_nominal() -> None:
    """The docstring's own claim: overshoot pushes the errors below nominal.

    Simulating this implementation at p0=0.90, p1=0.75, alpha=0.01, beta=0.05
    is documented as giving a false-accusation rate near 0.007 and a miss rate
    near 0.043. Both are checked here against generous ceilings, so the test
    fails on a real regression rather than on Monte Carlo noise.
    """
    trials = 400
    false_accusations = sum(
        1 for seed in range(trials) if _run_sprt(0.90, seed)[1] is SPRTDecision.ACCEPT_H1
    )
    misses = sum(
        1
        for seed in range(trials)
        if _run_sprt(0.75, 10_000 + seed)[1] is SPRTDecision.ACCEPT_H0
    )
    assert false_accusations / trials <= 0.03
    assert misses / trials <= 0.10


def test_sprt_terminates_near_walds_expected_sample_number() -> None:
    """Wald's average sample number is optimistic, and by a bounded amount."""
    test = SPRT(p0=0.90, p1=0.75, alpha=0.01, beta=0.05)
    under_h0 = [_run_sprt(0.90, seed)[0] for seed in range(200)]
    under_h1 = [_run_sprt(0.75, 20_000 + seed)[0] for seed in range(200)]

    assert all(n < 5000 for n in under_h0 + under_h1)
    # Optimistic, so the realised mean sits above the promise -- but not by more
    # than the roughly 20% the module docstring warns about, plus slack.
    assert test.expected_n_h0 <= statistics.fmean(under_h0) <= 1.5 * test.expected_n_h0
    assert test.expected_n_h1 <= statistics.fmean(under_h1) <= 1.5 * test.expected_n_h1


def test_sprt_confidence_interval_tracks_the_observed_rate() -> None:
    test = SPRT(p0=0.90, p1=0.75)
    for success in (True, True, True, False, True):
        test.update(success)
    assert test.n == 5
    assert test.successes == 4
    assert test.rate == pytest.approx(0.8, abs=1e-12)
    low, high = test.ci
    assert low <= 0.8 <= high


# --------------------------------------------------------------------------- #
# MMD
# --------------------------------------------------------------------------- #


def test_hamming_kernel_endpoints() -> None:
    assert hamming_kernel("abcd", "abcd") == 1.0
    assert hamming_kernel("abcd", "wxyz") == 0.0
    assert hamming_kernel("", "") == 1.0
    # A length difference is charged as a mismatch per missing position.
    assert hamming_kernel("abcd", "ab") == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(ValueError):
        hamming_kernel("a", "b", max_len=0)


def _strings(rng: random.Random, count: int, alphabet: str = "abcdefgh") -> list[str]:
    return ["".join(rng.choice(alphabet) for _ in range(24)) for _ in range(count)]


def test_mmd_separates_two_different_generators() -> None:
    rng = random.Random(11)
    a = _strings(rng, 12, "abcdefgh")
    b = _strings(rng, 12, "stuvwxyz")
    statistic, p = mmd_hamming(a, b, n_permutations=500, rng=random.Random(3))
    assert statistic > 0.0
    assert p < 0.01


def test_mmd_p_value_is_roughly_uniform_under_the_null() -> None:
    """Both samples from one generator, so the null is true by construction.

    A permutation p-value on discrete data is uniform only up to ties, so this
    checks the shape rather than the exact distribution: the mean sits near a
    half and the nominal 10% tail is not grossly inflated. A statistic that had
    drifted -- a sign error, a diagonal left in the U-statistic -- moves both.
    """
    pvalues: list[float] = []
    for replicate in range(40):
        rng = random.Random(1000 + replicate)
        a = _strings(rng, 8)
        b = _strings(rng, 8)
        _statistic, p = mmd_hamming(a, b, n_permutations=200, rng=random.Random(replicate))
        pvalues.append(p)

    assert 0.30 <= statistics.fmean(pvalues) <= 0.70
    assert sum(1 for p in pvalues if p <= 0.10) / len(pvalues) <= 0.30


def test_mmd_refuses_samples_it_cannot_estimate() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        mmd_hamming(["a"], ["b", "c"], rng=rng)
    with pytest.raises(ValueError):
        mmd_hamming(["a"] * 300, ["b"] * 300, rng=rng)
    with pytest.raises(ValueError):
        mmd_hamming(["a", "b"], ["c", "d"], n_permutations=0, rng=rng)


def test_mmd_p_value_never_reaches_zero() -> None:
    """The (1 + hits) / (1 + B) estimator has a floor, and it is documented."""
    rng = random.Random(5)
    a = ["aaaaaaaa"] * 6
    b = ["zzzzzzzz"] * 6
    _statistic, p = mmd_hamming(a, b, n_permutations=99, rng=rng)
    assert p >= 1.0 / 100.0
