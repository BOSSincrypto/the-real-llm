"""Sample-size and detectability arithmetic.

This module exists to stop the tool making claims it cannot support. Most famous
benchmarks no longer separate frontier models: the published 2026 scores on GPQA
Diamond span 71.3 to 92.8 with a standard deviation of about five points, and the
two leading models sit half a point apart. Separating those two at 95%
confidence and 80% power needs tens of thousands of items. GPQA Diamond has 198.

So before a benchmark probe spends anything, the runner asks
:func:`discriminative_power` what the benchmark could possibly show, and says so
in the report when the answer is "nothing". A benchmark that cannot discriminate
is still worth running -- a substituted model is often nowhere near frontier, and
a 40-point gap needs only a handful of items -- but its *negative* result must
never be presented as confirmation.

All three functions use the ordinary two-proportion normal approximation without
continuity correction, evaluated per arm. For the effect sizes that matter here
that is accurate enough; where it is not, it errs by asking for slightly fewer
items than an exact calculation would, so treat the numbers as lower bounds.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .distributions import normal_ppf

__all__ = ["detectable_effect", "discriminative_power", "required_n"]


def _validate(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must lie strictly in (0, 1), got {power!r}")


def _required_n_exact(p1: float, p2: float, alpha: float, power: float) -> float:
    """Unrounded per-arm sample size, so that bisection has something smooth."""
    delta = abs(p1 - p2)
    if delta == 0.0:
        raise ValueError("no finite sample size can separate two identical proportions")
    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    z_beta = normal_ppf(power)
    pbar = (p1 + p2) / 2.0
    under_null = z_alpha * math.sqrt(2.0 * pbar * (1.0 - pbar))
    under_alt = z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return ((under_null + under_alt) / delta) ** 2


def required_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Items needed *per arm* to distinguish accuracy ``p1`` from ``p2``.

    Both proportions are given on ``[0, 1]``, not in percent. The variance under
    the null is taken at the pooled proportion and the variance under the
    alternative at the two separate proportions, which is the standard two-sided
    formulation and matches what the two-proportion z-test actually does.

    Identical proportions raise :class:`ValueError`: no sample size detects a
    difference of zero, and returning a large sentinel would let that fact hide
    inside a report.
    """
    _validate(alpha, power)
    for name, value in (("p1", p1), ("p2", p2)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return max(1, math.ceil(_required_n_exact(p1, p2, alpha, power)))


def detectable_effect(
    n: int, p_base: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest accuracy drop from ``p_base`` that ``n`` items per arm can detect.

    Returned in proportion units: 0.08 means eight percentage points. The search
    is downward only, because the question this tool asks is always "has the
    endpoint got worse", never "has it got better".

    Found by bisecting :func:`required_n` rather than by inverting it in closed
    form, since the pooled variance term moves with the effect size. When even a
    collapse to zero accuracy would not be detectable at this ``n``, ``p_base``
    itself is returned -- the whole range is undetectable, and that is the most
    informative thing that can be said.
    """
    _validate(alpha, power)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    if not 0.0 < p_base <= 1.0:
        raise ValueError(f"p_base must lie in (0, 1], got {p_base!r}")

    if _required_n_exact(p_base, 0.0, alpha, power) > n:
        return p_base

    lo, hi = 0.0, p_base
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _required_n_exact(p_base, p_base - mid, alpha, power) > n:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12:
            break
    return hi


def discriminative_power(
    scores: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    available_items: int | None = None,
) -> dict[str, Any]:
    """Summarise whether a benchmark can tell a set of models apart.

    ``scores`` are published accuracies **in percent**, matching
    :attr:`llmverify.reference.schema.BenchmarkScore.score`. Pass every score the
    reference snapshot holds for the benchmark; the summary is about the
    benchmark, not about any one model.

    The headline number is ``n_to_separate``: the per-arm sample size needed to
    distinguish the two *closest* scores in the set. That is the hardest
    discrimination the benchmark is being asked to make, so it is the honest
    worst case. When ``available_items`` is supplied, ``sufficient`` compares the
    two and ``warning`` carries a sentence fit to put in a report.

    Spread and standard deviation are reported alongside because they are what a
    reader recognises, but they are the weaker signal: a benchmark can have a
    wide spread overall and still be useless for the specific pair of models
    under comparison.
    """
    _validate(alpha, power)
    values = [float(s) for s in scores]
    if len(values) < 2:
        raise ValueError("discriminative_power needs at least two scores")
    if any(not 0.0 <= v <= 100.0 for v in values):
        raise ValueError("scores must be percentages in [0, 100], not proportions")

    ordered = sorted(values)
    gap, lower, upper = min(
        (ordered[i + 1] - ordered[i], ordered[i], ordered[i + 1])
        for i in range(len(ordered) - 1)
    )

    if gap == 0.0:
        n_needed: float = math.inf
    else:
        n_needed = float(required_n(upper / 100.0, lower / 100.0, alpha, power))

    sufficient: bool | None = None
    warning: str | None = None
    if available_items is not None:
        sufficient = available_items >= n_needed
        if not sufficient:
            n_text = "infinitely many" if math.isinf(n_needed) else f"{int(n_needed):,}"
            warning = (
                f"The two closest published scores differ by {gap:.2f} points; "
                f"separating them at {round((1.0 - alpha) * 100)}% confidence and "
                f"{round(power * 100)}% power needs {n_text} items per arm, but only "
                f"{available_items:,} exist. A non-significant result from this benchmark "
                f"means the benchmark is too coarse, not that the endpoint is genuine."
            )

    return {
        "n_models": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "spread": ordered[-1] - ordered[0],
        "closest_pair": (lower, upper),
        "closest_gap_pp": gap,
        "n_to_separate": n_needed,
        "alpha": alpha,
        "power": power,
        "available_items": available_items,
        "sufficient": sufficient,
        "warning": warning,
    }
