"""Statistics for llmverify, implemented against the standard library only.

Every distribution function, hypothesis test, confidence interval and power
calculation this package needs lives here, written in pure Python on top of
:mod:`math`, :mod:`statistics` and :mod:`random`. There is no numpy and no
scipy, and that is deliberate.

The reason is not minimalism for its own sake. This tool is run by people
auditing a provider they have reason to distrust, often from a laptop, a CI job
or a locked-down container where installing a compiled scientific stack is a
half-hour argument with a platform team. A verification tool that is hard to
install is a verification tool that does not get run. The whole dependency set
is httpx, pydantic, pyyaml and rich, and it stays that way.

The cost is real and worth stating. These implementations are slower than
scipy's by a wide margin, and they are not vectorised. That is affordable
because the sample sizes are set by what an API call costs, not by what a CPU
can do: a few hundred benchmark items, a few thousand permutations. The one
place where the constant factor bites is
:func:`~llmverify.stats.mmd.mmd_hamming`, whose permutation test is quadratic in
the pooled sample size and which refuses inputs large enough to matter.

Accuracy is not compromised anywhere. The rational approximations, continued
fractions and exact enumerations used here agree with reference implementations
to more digits than any verdict could depend on, and where an approximation is
known to be optimistic -- the asymptotic Kolmogorov p-value on small samples,
Wald's sequential boundaries -- the docstring says so explicitly rather than
leaving the reader to assume exactness.

Layout:

``distributions``
    Normal, beta and gamma CDFs and quantiles, plus the Kolmogorov
    distribution. Everything else is built on these.
``intervals``
    Wilson, Clopper-Pearson and Agresti-Coull intervals for a proportion.
``tests``
    Two-proportion z, exact binomial, Fisher exact, two-sample KS, chi-square
    goodness of fit, and McNemar for paired answer flips.
``power``
    Sample sizes, detectable effects, and the discriminative-power summary that
    stops the tool claiming a saturated benchmark proved anything.
``sequential``
    Wald's SPRT, which is what lets an obvious substitution be settled in a few
    dozen questions instead of a few hundred.
``mmd``
    Maximum Mean Discrepancy with a Hamming string kernel, the text-only
    distribution test from arXiv:2410.20247.
``multiplicity``
    Holm-Bonferroni and Benjamini-Hochberg, without which running seventeen
    probes at a nominal 5% would accuse honest providers more than half the time.
"""

from __future__ import annotations

from .distributions import (
    beta_ppf,
    chi2_sf,
    ks_pvalue,
    log_beta,
    normal_cdf,
    normal_ppf,
    normal_sf,
    regularized_incomplete_beta,
)
from .intervals import agresti_coull, clopper_pearson, wilson_interval
from .mmd import MAX_TOTAL_SAMPLES, hamming_kernel, mmd_hamming
from .multiplicity import benjamini_hochberg, holm_bonferroni
from .power import detectable_effect, discriminative_power, required_n
from .sequential import SPRT, SPRTDecision
from .tests import (
    Alternative,
    binomial_test,
    chi_square_gof,
    fisher_exact,
    ks_two_sample,
    mcnemar,
    two_proportion_z,
)

__all__ = [
    "MAX_TOTAL_SAMPLES",
    "SPRT",
    "Alternative",
    "SPRTDecision",
    "agresti_coull",
    "benjamini_hochberg",
    "beta_ppf",
    "binomial_test",
    "chi2_sf",
    "chi_square_gof",
    "clopper_pearson",
    "detectable_effect",
    "discriminative_power",
    "fisher_exact",
    "hamming_kernel",
    "holm_bonferroni",
    "ks_pvalue",
    "ks_two_sample",
    "log_beta",
    "mcnemar",
    "mmd_hamming",
    "normal_cdf",
    "normal_ppf",
    "normal_sf",
    "regularized_incomplete_beta",
    "required_n",
    "two_proportion_z",
    "wilson_interval",
]
