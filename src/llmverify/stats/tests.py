"""Hypothesis tests used by the probes.

Every test here returns a p-value, and every p-value in this package means the
same thing: the probability of seeing a discrepancy at least this extreme if the
endpoint really is serving the claimed model. A small p-value is therefore
evidence against the provider's claim, and the probes convert it into a signed
log-likelihood ratio rather than thresholding it directly.

Exact tests are preferred over asymptotic ones wherever the exact version is
affordable, because the sample sizes here are small by statistical standards --
a few dozen benchmark items, a handful of paired flips -- and that is precisely
where normal approximations become optimistic. Since an optimistic p-value in
this tool means a stronger accusation than the data supports, the bias has to
run the other way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from .distributions import chi2_sf, ks_pvalue, normal_sf

__all__ = [
    "Alternative",
    "binomial_test",
    "chi_square_gof",
    "fisher_exact",
    "ks_two_sample",
    "mcnemar",
    "two_proportion_z",
]

#: Direction of the alternative hypothesis, shared by the exact tests.
Alternative = Literal["less", "greater", "two-sided"]

#: Slack allowed when comparing point probabilities for equality in the
#: two-sided exact tests. Without it, a table symmetric in exact arithmetic can
#: be excluded from its own tail by a one-ulp rounding difference.
_PMF_TOLERANCE = 1e-7


def _check_alternative(alternative: str) -> None:
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(
            f"alternative must be 'less', 'greater' or 'two-sided', got {alternative!r}"
        )


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test with pooled variance.

    Returns ``(z, p)`` where ``p`` is two-sided. The standard error uses the
    pooled proportion ``(k1 + k2) / (n1 + n2)``, which is the estimate that is
    correct *under the null hypothesis being tested* -- that the two endpoints
    have the same underlying accuracy. An unpooled standard error gives a
    slightly smaller p-value here and is appropriate for building a confidence
    interval on the difference, but not for testing equality.

    When both samples are entirely successes or entirely failures the pooled
    variance is zero and no test is possible; that is reported as ``(0.0, 1.0)``
    rather than as a division by zero.

    This is an asymptotic test. With fewer than about five expected successes
    and five expected failures per arm, prefer :func:`fisher_exact`.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("two_proportion_z requires positive sample sizes")
    if not 0 <= k1 <= n1 or not 0 <= k2 <= n2:
        raise ValueError("success counts must lie within their sample sizes")

    p1 = k1 / n1
    p2 = k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0.0:
        return (0.0, 1.0)

    z = (p1 - p2) / math.sqrt(variance)
    return (z, min(1.0, 2.0 * normal_sf(abs(z))))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    """Log of the binomial pmf, via ``lgamma`` so that large ``n`` is safe."""
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_choose + k * math.log(p) + (n - k) * math.log1p(-p)


def binomial_test(k: int, n: int, p0: float, alternative: Alternative = "two-sided") -> float:
    """Exact binomial test of ``H0: p == p0``.

    ``"less"`` returns ``P(X <= k)``, ``"greater"`` returns ``P(X >= k)``, and
    ``"two-sided"`` sums the probability of every outcome no more likely than the
    one observed. That last definition is the conventional one and is not the
    same as doubling the smaller tail; for an asymmetric binomial the two differ,
    and the point-probability rule is the one that generalises correctly.

    Probabilities are accumulated from exponentiated log-pmfs, so ``n`` in the
    thousands is fine. The cost is ``O(n)``.
    """
    _check_alternative(alternative)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n!r}")
    if not 0 <= k <= n:
        raise ValueError(f"k must lie in [0, {n}], got {k!r}")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError(f"p0 must lie in [0, 1], got {p0!r}")
    if n == 0:
        return 1.0

    pmf = [math.exp(_log_binom_pmf(i, n, p0)) for i in range(n + 1)]

    if alternative == "less":
        total = math.fsum(pmf[: k + 1])
    elif alternative == "greater":
        total = math.fsum(pmf[k:])
    else:
        threshold = pmf[k] * (1.0 + _PMF_TOLERANCE)
        total = math.fsum(value for value in pmf if value <= threshold)
    return min(1.0, max(0.0, total))


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int, alternative: Alternative = "two-sided") -> float:
    """Fisher's exact test on the 2x2 table ``[[a, b], [c, d]]``.

    Conditions on both margins and enumerates the hypergeometric distribution of
    the top-left cell. ``"greater"`` tests for an excess in that cell,
    ``"less"`` for a deficit, and ``"two-sided"`` sums every table no more likely
    than the observed one.

    Being exact, this is the right test for the small contingency tables the
    probes produce -- verbatim versus paraphrased answer counts on a few dozen
    items, say -- where the chi-square approximation is not trustworthy.
    """
    _check_alternative(alternative)
    if min(a, b, c, d) < 0:
        raise ValueError("fisher_exact requires non-negative counts")
    total = a + b + c + d
    if total == 0:
        return 1.0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    log_denom = _log_choose(total, col1)

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    support = range(lo, hi + 1)
    probs = {
        x: math.exp(_log_choose(row1, x) + _log_choose(row2, col1 - x) - log_denom)
        for x in support
    }

    if alternative == "less":
        chosen = [p for x, p in probs.items() if x <= a]
    elif alternative == "greater":
        chosen = [p for x, p in probs.items() if x >= a]
    else:
        threshold = probs[a] * (1.0 + _PMF_TOLERANCE)
        chosen = [p for p in probs.values() if p <= threshold]
    return min(1.0, max(0.0, math.fsum(chosen)))


def ks_two_sample(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns ``(D, p)`` where ``D`` is the largest absolute gap between the two
    empirical CDFs. The p-value comes from the asymptotic Kolmogorov
    distribution evaluated at the effective sample size ``n*m/(n+m)``.

    The statistic itself is exact and handles ties correctly by advancing both
    samples past every occurrence of a shared value before measuring the gap.
    The p-value is not exact: for samples smaller than roughly a dozen each it
    is anti-conservative, so treat a marginal result on tiny samples as
    inconclusive rather than as evidence.
    """
    if not x or not y:
        raise ValueError("ks_two_sample requires both samples to be non-empty")

    xs = sorted(x)
    ys = sorted(y)
    n, m = len(xs), len(ys)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        value = min(xs[i], ys[j])
        while i < n and xs[i] == value:
            i += 1
        while j < m and ys[j] == value:
            j += 1
        d = max(d, abs(i / n - j / m))

    return (d, ks_pvalue(d, n * m / (n + m)))


def chi_square_gof(
    observed: Sequence[float], expected: Sequence[float]
) -> tuple[float, float]:
    """Pearson chi-square goodness-of-fit test.

    Returns ``(statistic, p)`` with ``len(observed) - 1`` degrees of freedom,
    which assumes the expected counts were fixed in advance rather than fitted
    from the same data. Every probe that uses this supplies expectations from
    the reference snapshot, so that assumption holds.

    The approximation degrades when expected counts fall below about five per
    cell. That is not enforced here -- collapsing cells is a modelling decision
    the caller has to make -- but a low expected count is the first thing to
    check when this test disagrees with an exact one.
    """
    if len(observed) != len(expected):
        raise ValueError(
            f"observed and expected must have the same length, got "
            f"{len(observed)} and {len(expected)}"
        )
    if len(observed) < 2:
        raise ValueError("chi_square_gof needs at least two categories")
    if any(e <= 0.0 for e in expected):
        raise ValueError("expected counts must all be strictly positive")

    statistic = math.fsum((o - e) ** 2 / e for o, e in zip(observed, expected, strict=True))
    return (statistic, chi2_sf(statistic, len(observed) - 1))


def mcnemar(b: int, c: int, exact: bool = True) -> float:
    """McNemar's test for paired binary outcomes.

    ``b`` and ``c`` are the two discordant counts: the number of items the first
    condition got right and the second got wrong, and the reverse. Concordant
    items carry no information about a difference and are not passed in.

    This is the right test when the same items are put to two endpoints, or to
    one endpoint under two variants of the same prompt. Treating those as
    independent samples and reaching for :func:`two_proportion_z` throws away the
    pairing and loses most of the power -- which matters, because answer flips
    between a model and its quantized twin are common even when aggregate
    accuracy is unchanged.

    ``exact=True`` runs the two-sided binomial test on ``b`` out of ``b + c`` at
    ``p = 0.5``. ``exact=False`` uses the chi-square approximation with Yates'
    continuity correction, which is faster but unreliable when ``b + c`` is below
    about 25. With no discordant pairs at all the result is 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError("mcnemar requires non-negative discordant counts")
    n = b + c
    if n == 0:
        return 1.0
    if exact:
        return binomial_test(min(b, c), n, 0.5, "two-sided")
    statistic = (abs(b - c) - 1.0) ** 2 / n if abs(b - c) > 1 else 0.0
    return chi2_sf(statistic, 1)
