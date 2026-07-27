"""Wald's sequential probability ratio test.

This is the mechanism that makes the tool affordable. A fixed-sample benchmark
has to buy enough items for the hardest case it might face, and then pays for
all of them even when the endpoint is obviously wrong. The SPRT stops as soon as
the evidence is decisive in either direction, which in practice means a blatant
substitution -- a small model answering for a frontier one -- is settled in on the
order of twenty questions, while a subtle degradation keeps sampling until it
either separates or runs out of budget.

The test compares two simple hypotheses about the endpoint's accuracy:

    H0: accuracy == p0    the endpoint serves the genuine model
    H1: accuracy == p1    it serves something materially worse (p1 < p0)

After each item the log-likelihood ratio in favour of H1 moves by
``ln(p1/p0)`` on a correct answer and ``ln((1-p1)/(1-p0))`` on a wrong one, and
the test stops when the running total leaves the interval
``[ln(beta/(1-alpha)), ln((1-beta)/alpha)]``.

**What the error rates actually guarantee.** Wald's bounds are an approximation.
They ignore overshoot -- the amount by which the statistic passes a boundary on
the step that crosses it -- and are exact only in the limit of small per-sample
information. Overshoot pushes the realised error rates *below* nominal rather
than above, so the approximation errs in the safe direction here: simulating
this implementation at ``p0=0.90, p1=0.75, alpha=0.01, beta=0.05`` gives a
realised false-accusation rate near 0.007 and a realised miss rate near 0.043.
It is still an approximation, and ``alpha = 0.01`` should be read as "about one
percent", not "at most one percent".

:attr:`SPRT.expected_n_h0` and :attr:`SPRT.expected_n_h1` inherit the same
weakness with none of the safety, because there the bias runs the other way:
they are optimistic, by roughly 3% when the two hypotheses are close and by
around 20% when they are far apart. Budget for more items than they promise.

**Truncation.** A run that reaches its sample or cost ceiling before crossing a
boundary has *not* reached a decision, and forcing one -- by taking whichever
boundary is closer, or by falling back to a fixed-sample test on the items
collected -- destroys the error guarantees, because the stopping rule then
depends on the data in a way the analysis never accounted for. A truncated test
must be reported as :attr:`~llmverify.evidence.Verdict.INCONCLUSIVE`. The
decision stays :attr:`SPRTDecision.CONTINUE` at truncation precisely so that a
caller cannot accidentally read it as an answer.

**Independence.** The log-likelihood ratios add only because the items are
assumed independent given the model. Repeated samples of the *same* prompt are
not independent, and neither are items whose difficulty correlates with whatever
the substitution changed. Draw items without replacement from a shuffled pool.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from .intervals import wilson_interval

__all__ = ["SPRT", "SPRTDecision"]


class SPRTDecision(str, enum.Enum):
    """Where the test stands after the observations so far."""

    #: Neither boundary reached. Keep sampling, or stop and report INCONCLUSIVE.
    CONTINUE = "continue"
    #: The evidence favours the genuine model at the requested error rates.
    ACCEPT_H0 = "accept_h0"
    #: The evidence favours a materially worse model.
    ACCEPT_H1 = "accept_h1"


@dataclass(slots=True)
class SPRT:
    """A running sequential test of accuracy ``p0`` against accuracy ``p1``.

    Feed it one :meth:`update` per graded item and stop when :attr:`decision` is
    no longer :attr:`SPRTDecision.CONTINUE`.

    Choosing ``p1`` is a modelling decision, not a statistical one: it is the
    accuracy the caller has decided counts as "materially worse", and the test's
    power is defined against that value and no other. Setting it too close to
    ``p0`` makes the test run forever; too far, and a real but moderate
    degradation slips through as ACCEPT_H0. The runner derives it from
    :attr:`~llmverify.config.BudgetConfig.min_effect_pp`.
    """

    p0: float
    p1: float
    alpha: float = 0.01
    beta: float = 0.05

    _n: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _llr: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.p1 < self.p0 < 1.0:
            raise ValueError(
                "SPRT requires 0 < p1 < p0 < 1; p1 is the degraded accuracy and must be "
                f"strictly below p0, got p0={self.p0!r}, p1={self.p1!r}"
            )
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.beta < 1.0:
            raise ValueError("alpha and beta must lie strictly in (0, 1)")
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be below 1 for the boundaries to bracket zero")

    # ------------------------------------------------------------------ state

    @property
    def n(self) -> int:
        """Items observed so far."""
        return self._n

    @property
    def successes(self) -> int:
        """Correct answers so far."""
        return self._successes

    @property
    def rate(self) -> float:
        """Observed accuracy, or 0.0 before any observation."""
        return self._successes / self._n if self._n else 0.0

    @property
    def llr(self) -> float:
        """Log-likelihood ratio in favour of H1, in nats.

        Signed the opposite way from :class:`~llmverify.evidence.Evidence.llr`,
        which is stated from the point of view of the provider's claim. A probe
        wrapping this test negates it.
        """
        return self._llr

    @property
    def ci(self) -> tuple[float, float]:
        """Wilson 95% interval for the observed accuracy.

        Purely descriptive. It is *not* a valid confidence interval for a
        sequentially stopped sample -- optional stopping biases the observed rate
        toward whichever boundary was crossed -- and is here so that a report can
        show the rate with some sense of its precision, not so that anyone can
        test with it.
        """
        return wilson_interval(self._successes, self._n)

    # -------------------------------------------------------------- boundaries

    @property
    def upper_bound(self) -> float:
        """Boundary above which H1 is accepted, ``ln((1-beta)/alpha)``."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower_bound(self) -> float:
        """Boundary below which H0 is accepted, ``ln(beta/(1-alpha))``."""
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def _step_success(self) -> float:
        return math.log(self.p1 / self.p0)

    @property
    def _step_failure(self) -> float:
        return math.log((1.0 - self.p1) / (1.0 - self.p0))

    @property
    def decision(self) -> SPRTDecision:
        """Current decision, recomputed from the running statistic."""
        if self._llr >= self.upper_bound:
            return SPRTDecision.ACCEPT_H1
        if self._llr <= self.lower_bound:
            return SPRTDecision.ACCEPT_H0
        return SPRTDecision.CONTINUE

    # ----------------------------------------------------------- expected cost

    @property
    def expected_n_h0(self) -> float:
        """Wald's approximate average sample number when H0 is true.

        Useful for planning a budget before the run starts. It ignores overshoot
        and can be optimistic by a wide margin when the two hypotheses are far
        apart, which is exactly the case where the run is cheap anyway.
        """
        drift = self.p0 * self._step_success + (1.0 - self.p0) * self._step_failure
        return self._expected_n(self.alpha, drift)

    @property
    def expected_n_h1(self) -> float:
        """Wald's approximate average sample number when H1 is true."""
        drift = self.p1 * self._step_success + (1.0 - self.p1) * self._step_failure
        return self._expected_n(1.0 - self.beta, drift)

    def _expected_n(self, p_accept_h1: float, drift: float) -> float:
        if drift == 0.0:
            return math.inf
        numerator = p_accept_h1 * self.upper_bound + (1.0 - p_accept_h1) * self.lower_bound
        return abs(numerator / drift)

    # ----------------------------------------------------------------- driving

    def update(self, success: bool) -> SPRTDecision:
        """Record one graded item and return the decision it leads to."""
        self._n += 1
        if success:
            self._successes += 1
            self._llr += self._step_success
        else:
            self._llr += self._step_failure
        return self.decision

    def reset(self) -> None:
        """Clear all observations, keeping the hypotheses and error rates.

        Restarting a test on the same endpoint and then reporting only the run
        that crossed a boundary is a garden of forking paths; this exists for
        reusing a configured test across independent benchmarks, not for retries.
        """
        self._n = 0
        self._successes = 0
        self._llr = 0.0
