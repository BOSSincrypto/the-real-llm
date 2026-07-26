"""Maximum Mean Discrepancy over completions, with a Hamming string kernel.

This is the two-sample test from "Model Equality Testing: Which Model Is This
API Serving?" (Gao, Liang and Guestrin, arXiv:2410.20247, ICLR 2025). The paper
compares several statistics for deciding whether two sets of completions came
from the same distribution and finds MMD with a normalised Hamming string kernel
the strongest, reaching a median power of about 77% with only ten samples per
prompt. Applied to 31 commercial Llama endpoints, that test found 11 serving a
distribution measurably different from the reference weights.

The appeal for this package is that it needs nothing but text. No logprobs, no
seed support, no cooperation from the provider beyond answering. That makes it
the only distribution-level test available against an endpoint that exposes a
bare chat completion and nothing else -- which is most of them.

The kernel is

    k(s, t) = 1 - hamming(s, t) / max(len(s), len(t))

with positions past the end of the shorter string counted as mismatches, so
strings of different lengths are handled without padding or truncation to a
common size. It is 1 for identical strings and 0 for strings that share no
character position.

The statistic is the unbiased U-statistic form of MMD^2, which excludes the
diagonal ``k(x_i, x_i) = 1`` terms. Being unbiased it can come out slightly
negative when the two samples are drawn from the same distribution; that is
normal and means "no difference detected", not "an error occurred".

The null distribution comes from a permutation test rather than from asymptotic
theory, because the kernel is not one whose null distribution has a usable
closed form at these sample sizes.

**Cost.** Building the kernel matrix is ``O((n+m)^2 * L)`` character
comparisons, where ``L`` is ``max_len``. Each permutation then costs
``O((n+m)^2)`` float additions against that cached matrix, for
``O(B * (n+m)^2)`` overall. Measured at the default 2000 permutations, a pooled
sample of 24 takes about 0.05s, 100 about 0.4s, 200 about 1.5s and the
:data:`MAX_TOTAL_SAMPLES` ceiling of 400 about 9s. The quadratic term is what
the ceiling exists to bound: a verification run has a wall-clock budget, and
refusing an oversized sample is better than appearing to hang.

**What a small p-value does and does not mean.** It means the two sets of
completions did not come from the same distribution. Temperature, sampler
settings, a system prompt, a quantization change, a different serving stack and a
different model all produce that. Distinguishing among those causes is the job of
the other probes; this one only establishes that something differs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

__all__ = ["MAX_TOTAL_SAMPLES", "hamming_kernel", "mmd_hamming"]

#: Refuse pooled samples larger than this. The permutation loop is quadratic in
#: the pooled size, so the cost curve is steep enough that a limit is kinder
#: than a run that appears to hang.
MAX_TOTAL_SAMPLES = 400

#: Slack when counting permutation statistics at least as extreme as the
#: observed one, so that floating-point noise in an identical rearrangement does
#: not make the test look more significant than it is.
_STAT_TOLERANCE = 1e-12


def hamming_kernel(s: str, t: str, *, max_len: int = 512) -> float:
    """Normalised Hamming similarity between two strings, in ``[0, 1]``.

    Both strings are truncated to ``max_len`` characters first. Truncation costs
    some sensitivity to differences that only appear late in a long completion,
    and buys a bounded per-comparison cost; the paper's own setup uses short
    completions for the same reason.

    Positions beyond the end of the shorter string count as mismatches, so the
    kernel penalises a length difference directly. Two empty strings are defined
    as identical.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be at least 1, got {max_len!r}")
    a = s[:max_len]
    b = t[:max_len]
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    mismatches = longest - min(len(a), len(b))
    mismatches += sum(1 for x, y in zip(a, b, strict=False) if x != y)
    return 1.0 - mismatches / longest


def _kernel_matrix(samples: Sequence[str], max_len: int) -> list[list[float]]:
    """Full symmetric kernel matrix, computed once and reused by every permutation."""
    size = len(samples)
    matrix = [[1.0] * size for _ in range(size)]
    for i in range(size):
        row_i = matrix[i]
        for j in range(i + 1, size):
            value = hamming_kernel(samples[i], samples[j], max_len=max_len)
            row_i[j] = value
            matrix[j][i] = value
    return matrix


def _u_statistic(
    matrix: list[list[float]], a_idx: Sequence[int], b_idx: Sequence[int]
) -> float:
    """Unbiased MMD^2 for the split of ``matrix`` given by the two index lists."""
    m = len(a_idx)
    n = len(b_idx)

    within_a = 0.0
    for pos, i in enumerate(a_idx):
        row = matrix[i]
        within_a += sum(row[j] for j in a_idx[pos + 1 :])

    within_b = 0.0
    for pos, i in enumerate(b_idx):
        row = matrix[i]
        within_b += sum(row[j] for j in b_idx[pos + 1 :])

    cross = 0.0
    for i in a_idx:
        row = matrix[i]
        cross += sum(row[j] for j in b_idx)

    # within_* hold the strict upper triangle, so they are doubled to cover the
    # full i != j sum the U-statistic is defined over.
    return (
        2.0 * within_a / (m * (m - 1))
        + 2.0 * within_b / (n * (n - 1))
        - 2.0 * cross / (m * n)
    )


def mmd_hamming(
    sample_a: Sequence[str],
    sample_b: Sequence[str],
    *,
    n_permutations: int = 2000,
    rng: random.Random,
    max_len: int = 512,
) -> tuple[float, float]:
    """Test whether two sets of completions came from the same distribution.

    Returns ``(mmd_squared, p_value)``. The statistic is the unbiased U-statistic
    estimate of squared MMD under the normalised Hamming kernel, and the p-value
    is the fraction of random relabellings of the pooled sample whose statistic
    is at least as large.

    ``rng`` is required and has no default: a permutation p-value is only
    reproducible if the caller controls the stream, and a verification verdict
    that changes between runs of the same command is not a verdict. Pass a
    :class:`random.Random` seeded from
    :meth:`llmverify.probes.ProbeContext.rng`.

    The p-value uses the ``(1 + hits) / (1 + B)`` estimator rather than
    ``hits / B``. That includes the observed labelling as one of its own
    permutations, which keeps the test valid -- ``hits = 0`` would otherwise
    report an impossible p-value of exactly zero -- and sets the floor at
    ``1/(B+1)``, so 2000 permutations cannot report anything below about 5e-4
    however different the samples are.

    Both samples need at least two elements, since the unbiased estimator
    divides by ``m(m-1)``.
    """
    m = len(sample_a)
    n = len(sample_b)
    if m < 2 or n < 2:
        raise ValueError(
            f"mmd_hamming needs at least two completions per sample, got {m} and {n}"
        )
    total = m + n
    if total > MAX_TOTAL_SAMPLES:
        raise ValueError(
            f"pooled sample of {total} exceeds MAX_TOTAL_SAMPLES={MAX_TOTAL_SAMPLES}; "
            "the permutation test is quadratic in the pooled size, so subsample first"
        )
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations!r}")

    pooled = [*sample_a, *sample_b]
    matrix = _kernel_matrix(pooled, max_len)

    indices = list(range(total))
    observed = _u_statistic(matrix, indices[:m], indices[m:])

    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(indices)
        if _u_statistic(matrix, indices[:m], indices[m:]) >= observed - _STAT_TOLERANCE:
            at_least_as_extreme += 1

    return (observed, (1.0 + at_least_as_extreme) / (1.0 + n_permutations))
