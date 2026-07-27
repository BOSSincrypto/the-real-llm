"""Distribution functions used by the tests and intervals in this package.

Everything here is a standard rational approximation or continued fraction
evaluated with :mod:`math`. The accuracy targets are stated per function and
were chosen so that the intervals and p-values built on top of them agree with
reference implementations to more digits than a verification verdict could
possibly depend on.

Two conventions hold throughout. Functions named ``*_sf`` return upper-tail
probabilities and are computed directly rather than as ``1 - cdf``, because a
p-value of 1e-12 loses every significant digit to cancellation if it is formed
by subtraction. Functions named ``*_ppf`` are quantile functions and return
``-inf`` / ``+inf`` at the closed ends of their support instead of raising, so
that a caller asking for a 100% confidence interval gets the mathematically
correct answer rather than an exception.
"""

from __future__ import annotations

import math

__all__ = [
    "beta_ppf",
    "chi2_sf",
    "ks_pvalue",
    "log_beta",
    "normal_cdf",
    "normal_ppf",
    "normal_sf",
    "regularized_incomplete_beta",
]

#: Relative convergence tolerance for the iterative expansions below. Set just
#: above double-precision epsilon so the loops terminate rather than spin.
_EPS = 3.0e-16
#: Stand-in for zero in continued fractions, where a zero denominator would
#: otherwise abort a recurrence that is perfectly well behaved either side.
_TINY = 1.0e-300
_MAX_ITER = 500

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Normal
# --------------------------------------------------------------------------- #


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function.

    Exact to within one or two units in the last place, since ``math.erf`` is
    the platform libm's own implementation.
    """
    return 0.5 * math.erfc(-x / _SQRT2)


def normal_sf(x: float) -> float:
    """Standard normal upper-tail probability, ``P(Z > x)``.

    Computed from ``erfc`` directly. This is what makes a two-sided p-value of
    1e-15 meaningful instead of rounding to zero.
    """
    return 0.5 * math.erfc(x / _SQRT2)


# Acklam's rational approximation to the normal quantile function. The bare
# approximation is accurate to about 1.15e-9 relative; the Halley step below
# takes it to full double precision.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def normal_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF).

    Acklam's rational approximation refined by one Halley iteration against
    ``erfc``, which brings the absolute error below 1e-12 across the whole
    representable range -- comfortably inside the 1e-9 the callers need.

    ``p`` of exactly 0 or 1 returns infinity of the appropriate sign rather
    than raising, because a degenerate confidence level is a legitimate, if
    useless, request. Anything outside ``[0, 1]`` is a caller bug.
    """
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        raise ValueError(f"normal_ppf expects a probability in [0, 1], got {p!r}")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )

    # Halley refinement. Skipped in the far tails where exp(x^2/2) overflows;
    # there the raw approximation is already better than the caller can use.
    if abs(x) < 37.0:
        err = normal_cdf(x) - p
        u = err * _SQRT_2PI * math.exp(x * x / 2.0)
        x -= u / (1.0 + x * u / 2.0)
    return x


# --------------------------------------------------------------------------- #
# Beta
# --------------------------------------------------------------------------- #


def log_beta(a: float, b: float) -> float:
    """Natural log of the beta function ``B(a, b)``.

    Taken through ``lgamma`` so that shape parameters in the hundreds -- routine
    for a Clopper-Pearson interval over a few hundred benchmark items -- do not
    overflow.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"log_beta requires positive shapes, got a={a!r}, b={b!r}")
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Lentz evaluation of the continued fraction for the incomplete beta."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d

    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + num / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c

        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + num / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``.

    This is the beta CDF, and through the usual identities it is also the CDF
    of the binomial, the F and the Student t. The continued fraction converges
    fast only for ``x`` below the distribution's mode, so the symmetry relation
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` is applied above it.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"regularized_incomplete_beta requires positive shapes, got {a!r}, {b!r}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """Quantile function of the beta distribution.

    Plain bisection on :func:`regularized_incomplete_beta`. Newton would
    converge faster, but the beta density is unbounded at the ends for shapes
    below one -- exactly the shapes a Clopper-Pearson interval uses when a model
    got every item right or every item wrong -- and Newton's step there is
    unreliable in a way bisection never is. The iteration count is bounded by
    the width of a double, so the cost is fixed and small.
    """
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        raise ValueError(f"beta_ppf expects a probability in [0, 1], got {p!r}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"beta_ppf requires positive shapes, got a={a!r}, b={b!r}")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if regularized_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 8.0 * math.ulp(hi):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Gamma / chi-square
# --------------------------------------------------------------------------- #


def _gamma_p_series(s: float, x: float) -> float:
    """Series expansion of the regularized lower incomplete gamma ``P(s, x)``."""
    ap = s
    term = 1.0 / s
    total = term
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _gamma_q_continued_fraction(s: float, x: float) -> float:
    """Lentz evaluation of the regularized upper incomplete gamma ``Q(s, x)``."""
    b = x + 1.0 - s
    c = 1.0 / _TINY
    d = 1.0 / b if abs(b) > _TINY else 1.0 / _TINY
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _EPS:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def _gamma_q(s: float, x: float) -> float:
    """Regularized upper incomplete gamma ``Q(s, x) = 1 - P(s, x)``."""
    if x <= 0.0:
        return 1.0
    if x < s + 1.0:
        return 1.0 - _gamma_p_series(s, x)
    return _gamma_q_continued_fraction(s, x)


def chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability of the chi-square distribution.

    ``chi2_sf(3.841, 1)`` is the familiar 0.05, which is a convenient way to
    remember that this is the survival function and not the CDF.
    """
    if df <= 0:
        raise ValueError(f"chi2_sf requires positive degrees of freedom, got {df!r}")
    if x <= 0.0:
        return 1.0
    return min(1.0, max(0.0, _gamma_q(df / 2.0, x / 2.0)))


# --------------------------------------------------------------------------- #
# Kolmogorov
# --------------------------------------------------------------------------- #

#: Below this value of the scaled statistic the alternating series needs an
#: impractical number of terms, and the true p-value is 1 to well beyond double
#: precision anyway.
_KS_LAMBDA_FLOOR = 0.04


def ks_pvalue(d: float, n_eff: float) -> float:
    """Asymptotic p-value for a Kolmogorov-Smirnov statistic.

    Evaluates the limiting Kolmogorov distribution
    ``Q(lambda) = 2 * sum (-1)^(j-1) exp(-2 j^2 lambda^2)`` at
    ``lambda = sqrt(n_eff) * d``. For a two-sample test ``n_eff`` is the
    effective size ``n*m/(n+m)``.

    Stephens' finite-sample correction, ``(sqrt(n) + 0.12 + 0.11/sqrt(n)) * d``,
    is deliberately *not* applied. It is derived for the one-sample statistic,
    and against the exactly enumerated two-sample null it over-corrects in every
    configuration tested: at n = m = 10 and D = 0.5 the exact tail is 0.168, the
    plain scaling gives 0.164 and the corrected one gives 0.111. Mean absolute
    error across a range of sample sizes is four times worse with the
    correction, and every one of its errors is in the anti-conservative
    direction. In a tool that converts small p-values into accusations, an
    approximation that only ever understates them is the wrong trade.

    What remains is still asymptotic. Below roughly ten observations per sample
    it is mildly anti-conservative for unequal sample sizes, so treat a marginal
    result on tiny samples as inconclusive rather than as evidence.
    """
    if n_eff <= 0.0 or d <= 0.0:
        return 1.0
    lam = math.sqrt(n_eff) * d
    if lam < _KS_LAMBDA_FLOOR:
        return 1.0

    total = 0.0
    sign = 2.0
    for j in range(1, 101):
        term = sign * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-16:
            break
        sign = -sign
    return min(1.0, max(0.0, total))
