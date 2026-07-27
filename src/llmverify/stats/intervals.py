"""Confidence intervals for a binomial proportion.

Three intervals are offered because they disagree in exactly the regime this
tool works in: small samples and proportions near the ends. A model that gets
198 of 198 GPQA items right is not "100% accurate with no uncertainty", and the
Wald interval that would say so is not implemented here at all.

Which to reach for:

**Wilson** is the default for reporting. It has good average coverage down to
very small ``n``, never leaves ``[0, 1]``, and its bounds stay strictly inside
the unit interval when they should. Use it for the accuracy figures shown to a
user and for the sequential test's running interval.

**Clopper-Pearson** is exact in the sense that its coverage is guaranteed to be
at least the nominal level for every true proportion. It pays for that with
conservatism -- intervals are wider than they need to be, sometimes markedly so.
Use it when an interval is load-bearing for an accusation, because being
conservative there means erring toward not accusing.

**Agresti-Coull** is the "add two successes and two failures" adjustment. It is
simpler than Wilson and behaves similarly for moderate ``n``, and is included
mainly so that a reported interval can be reproduced by a reader who expects it.

All three return ``(0.0, 1.0)`` for ``n == 0``. That is not a coverage claim; it
is the honest statement that zero observations constrain nothing.
"""

from __future__ import annotations

import math

from .distributions import beta_ppf, normal_ppf

__all__ = ["agresti_coull", "clopper_pearson", "wilson_interval"]


def _check(successes: int, n: int, confidence: float) -> None:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n!r}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes must lie in [0, {n}], got {successes!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence!r}")


def _z_for(confidence: float) -> float:
    return normal_ppf(1.0 - (1.0 - confidence) / 2.0)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Derived by inverting the score test rather than the Wald test, which is why
    it stays inside ``[0, 1]`` and why it remains sensible at ``k = 0`` and
    ``k = n``: at those points the interval collapses to one side on its own,
    with no clamping needed. The two endpoints are nevertheless set exactly
    there, because the algebra that makes them exactly 0 and 1 does not survive
    floating point and a lower bound of 3e-17 in a report is noise.
    """
    _check(successes, n, confidence)
    if n == 0:
        return (0.0, 1.0)

    z = _z_for(confidence)
    z2 = z * z
    phat = successes / n
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return (lower, upper)


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson "exact" interval, from the beta quantile function.

    The bounds are the beta quantiles implied by inverting the binomial CDF:
    the lower bound solves ``P(X >= k) = alpha/2`` and the upper solves
    ``P(X <= k) = alpha/2``. At ``k = 0`` the lower bound is exactly 0 and at
    ``k = n`` the upper bound is exactly 1, so those cases are returned directly
    instead of asking for a quantile of a degenerate beta.
    """
    _check(successes, n, confidence)
    if n == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta_ppf(alpha / 2.0, successes, n - successes + 1)
    upper = 1.0 if successes == n else beta_ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return (max(0.0, lower), min(1.0, upper))


def agresti_coull(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Agresti-Coull adjusted-Wald interval.

    Adds ``z^2/2`` pseudo-successes and the same number of pseudo-failures, then
    applies the ordinary Wald formula to the adjusted counts. At the default 95%
    level ``z^2/2`` is very close to 2, which is where the folk description "add
    two and two" comes from.

    Unlike Wilson and Clopper-Pearson, this interval has no natural boundary
    behaviour: at ``k = 0`` the raw lower bound is negative and at ``k = n`` the
    raw upper bound exceeds one. Both are clamped, so those endpoints are
    reported as exactly 0 and 1 respectively -- a truncation, not an exact
    result, and a reason to prefer one of the other two at the extremes.
    """
    _check(successes, n, confidence)
    if n == 0:
        return (0.0, 1.0)

    z = _z_for(confidence)
    z2 = z * z
    n_adj = n + z2
    p_adj = (successes + z2 / 2.0) / n_adj
    half = z * math.sqrt(p_adj * (1.0 - p_adj) / n_adj)
    return (max(0.0, p_adj - half), min(1.0, p_adj + half))
