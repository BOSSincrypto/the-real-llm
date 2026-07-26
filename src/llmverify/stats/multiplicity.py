"""Multiple-testing correction.

A full run fires on the order of seventeen probes, several of which produce a
p-value. Testing seventeen independent hypotheses at 0.05 apiece gives a 58%
chance that at least one comes up significant against a completely honest
provider. Reporting the smallest of those as "the endpoint failed a test" would
make the tool worse than useless: it would generate a stream of confident, wrong
accusations, and the first person to check one by hand would stop trusting all
of them.

Both procedures here take raw p-values and return *adjusted* p-values, which are
compared against the original threshold. Adjusted p-values are preferable to
returning a reject/accept vector because they survive being put in a report: a
reader can see how close a probe came to significance, and can apply their own
threshold without rerunning anything.

**Which to use.** Holm-Bonferroni controls the family-wise error rate -- the
probability of *any* false rejection. It is uniformly more powerful than plain
Bonferroni and needs no assumption about dependence between the probes, which
matters because llmverify's probes are emphatically not independent (five
tokenizer measurements are one experiment wearing five hats). Use it when a
single significant probe would drive an adverse verdict.

Benjamini-Hochberg controls the false discovery rate -- the expected proportion
of rejections that are false. It is more powerful, and appropriate when the
output is a ranked list of suspicious probes for a human to triage rather than a
single accusation. Its guarantee holds under independence and under positive
dependence; under arbitrary dependence it needs the Benjamini-Yekutieli
log-factor penalty, which is not implemented here. Prefer Holm when the
dependence structure is unknown, which is the usual case.

Both functions preserve input order and both are monotone: a smaller raw
p-value never receives a larger adjusted p-value than a bigger one.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["benjamini_hochberg", "holm_bonferroni"]


def _validated(pvalues: Sequence[float]) -> list[float]:
    values = [float(p) for p in pvalues]
    for p in values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must lie in [0, 1], got {p!r}")
    return values


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, controlling the family-wise error rate.

    The ``i``-th smallest p-value is multiplied by ``m - i`` (the number of
    hypotheses still untested at that step), then a running maximum enforces
    monotonicity: once a hypothesis fails to be rejected, nothing with a larger
    raw p-value may be rejected either. Results are clamped at 1.

    Valid under any dependence structure between the tests.
    """
    values = _validated(pvalues)
    m = len(values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjustment, controlling the false discovery rate.

    The ``i``-th smallest p-value (one-based) is multiplied by ``m / i``, then a
    running minimum taken from the largest downward enforces monotonicity.
    Results are clamped at 1.

    Guaranteed under independence and under positive regression dependence. It
    is anti-conservative under arbitrary negative dependence, so a probe designed
    to fire when another one does not is a reason to switch to
    :func:`holm_bonferroni`.
    """
    values = _validated(pvalues)
    m = len(values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: values[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for offset, idx in enumerate(order):
        rank = m - offset
        running = min(running, m / rank * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
