"""Certified auto-execution: choose MNCD's threshold with a finite-sample guarantee.

Problem with a hand-set theta
-----------------------------
MNCD certifies a claim when its evidence-weighted share reaches theta. A
hand-set theta (0.5) gives a good *average* accepted-error rate but no
statement an examiner or a regulator can rely on. "Escalate on uncertainty" is
only a legitimate oversight design when confidence is calibrated.

Method: Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021;
Annals of Applied Statistics 2025) with fixed-sequence testing
----------------------------------------------------------------------------
For each candidate threshold lambda (strictest first) we test
H0(lambda): risk(lambda) > alpha, where risk(lambda) = P(wrong | executed at
lambda). The p-value is the exact binomial tail (Clopper-Pearson / binomial
test) on the calibration split. Walking the thresholds from strictest to
loosest and stopping at the first non-rejection controls the family-wise error
at delta without a multiplicity correction, because the hypotheses are
ordered. The selected lambda* therefore satisfies

    P( risk(lambda*) <= alpha ) >= 1 - delta

over the draw of the calibration set (exchangeability of calibration and
deployment sub-goals assumed - the assumption is stated, and tested by the
held-out check in ``experiments/governance_eval.py``).

Turns below lambda* are not discarded; they become REVIEW (a person decides),
so the guarantee buys a bounded auto-execution error at a measured review cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), exact, stable for n up to ~1e5."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    try:
        from scipy.stats import binom  # type: ignore

        return float(binom.cdf(k, n, p))
    except Exception:  # pragma: no cover - numpy-only fallback
        logp, log1p = math.log(p), math.log1p(-p)
        total = 0.0
        for i in range(k + 1):
            total += math.exp(
                math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * logp + (n - i) * log1p
            )
        return min(1.0, total)


def binomial_pvalue(errors: int, n: int, alpha: float) -> float:
    """p-value for H0: true error rate > alpha, given `errors` out of `n`."""
    if n == 0:
        return 1.0
    return _binom_cdf(errors, n, alpha)


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    try:
        from scipy.stats import beta  # type: ignore

        return float(beta.ppf(1 - delta, errors + 1, n - errors))
    except Exception:  # pragma: no cover
        return min(1.0, errors / n + math.sqrt(math.log(1 / delta) / (2 * n)))


@dataclass(frozen=True)
class CertifiedThreshold:
    threshold: float
    alpha: float
    delta: float
    calibration_n: int
    executed_n: int
    executed_errors: int
    coverage: float           # share of certified sub-goals auto-executed
    risk_hat: float
    risk_upper: float          # Clopper-Pearson (1-delta) upper bound at threshold
    valid: bool                # False -> no threshold controls risk; everything goes to REVIEW

    def to_dict(self) -> dict[str, float | int | bool]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def learn_then_test(
    scores: Sequence[float],
    wrong: Sequence[bool],
    alpha: float = 0.02,
    delta: float = 0.05,
    grid: Sequence[float] | None = None,
) -> CertifiedThreshold:
    """Pick the loosest threshold whose execution risk is certified <= alpha.

    ``scores``  MNCD weighted share for every *certified* calibration sub-goal
    ``wrong``   ground truth: True when the certified claim was not faithful
    """
    if len(scores) != len(wrong):
        raise ValueError("scores and labels must align")
    n = len(scores)
    grid = sorted(set(grid or [round(0.30 + 0.01 * i, 2) for i in range(71)]), reverse=True)
    # A threshold can only be certified if, even with zero errors, its executed
    # set is large enough to reject H0: n >= log(delta) / log(1 - alpha). The
    # sequence starts at the strictest such threshold. This uses only the
    # (unlabelled) scores, so the order of hypotheses is fixed given the
    # scores and the fixed-sequence argument holds conditionally on them.
    n_min = math.ceil(math.log(delta) / math.log(1.0 - alpha)) if 0 < alpha < 1 else 1
    grid = [lam for lam in grid if sum(1 for s in scores if s >= lam) >= n_min]
    chosen = None
    for lam in grid:  # strictest first: fixed-sequence testing
        idx = [i for i, s in enumerate(scores) if s >= lam]
        k = sum(1 for i in idx if wrong[i])
        if binomial_pvalue(k, len(idx), alpha) <= delta:
            chosen = (lam, len(idx), k)
        else:
            break
    if chosen is None:
        return CertifiedThreshold(1.01, alpha, delta, n, 0, 0, 0.0, 0.0, 1.0, False)
    lam, m, k = chosen
    return CertifiedThreshold(
        threshold=lam,
        alpha=alpha,
        delta=delta,
        calibration_n=n,
        executed_n=m,
        executed_errors=k,
        coverage=m / n if n else 0.0,
        risk_hat=k / m if m else 0.0,
        risk_upper=clopper_pearson_upper(k, m, delta),
        valid=True,
    )


def expected_calibration_error(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """ECE of a confidence score; reported so 'escalate on uncertainty' is earned, not assumed."""
    n = len(conf)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        avg = sum(conf[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - avg)
    return ece


__all__ = [
    "CertifiedThreshold",
    "binomial_pvalue",
    "clopper_pearson_upper",
    "expected_calibration_error",
    "learn_then_test",
]
