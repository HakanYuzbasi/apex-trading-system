"""
monitoring/stat_significance.py — Statistical Significance Gates

Prevents automated feedback loops (SignalAutoTuner, ThresholdCalibrator,
AdaptiveEntryGate) from overfit to noise by requiring that observed win-rate
deviations are statistically distinguishable from chance.

Two test modes are provided:
  - binomial_pvalue: exact two-sided binomial test (wins vs null H0 p=0.5)
  - wilson_ci: Wilson score confidence interval; check if null rate is outside CI

Both use scipy.stats when available and fall back to a normal approximation
so this module has no hard dependency.

Usage:
    from monitoring.stat_significance import is_significant

    if is_significant(wins=22, n=40, alpha=0.05):
        # safe to act on this win rate
        ...
"""
from __future__ import annotations

import math
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Null hypothesis: fair coin (50% win rate)
_NULL_P: float = 0.50


def binomial_pvalue(wins: int, n: int, null_p: float = _NULL_P) -> float:
    """
    Two-sided exact binomial p-value: P(X ≥ wins | H0: p=null_p).

    Falls back to normal approximation if scipy unavailable.
    """
    if n <= 0:
        return 1.0
    wins = max(0, min(wins, n))
    try:
        from scipy.stats import binom_test  # type: ignore
        return float(binom_test(wins, n, null_p, alternative="two-sided"))
    except ImportError:
        pass
    try:
        from scipy.stats import binomtest  # type: ignore
        return float(binomtest(wins, n, null_p, alternative="two-sided").pvalue)
    except (ImportError, Exception):
        pass
    # Normal approximation fallback
    p_hat = wins / n
    se = math.sqrt(null_p * (1 - null_p) / n)
    if se == 0:
        return 0.0
    z = abs(p_hat - null_p) / se
    # Two-sided p-value using complementary error function
    return float(math.erfc(z / math.sqrt(2)))


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for_confidence(confidence)
    p_hat = wins / n
    denominator = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _z_for_confidence(confidence: float) -> float:
    """Approximate z-score for two-sided confidence interval."""
    # Common values
    _TABLE = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if confidence in _TABLE:
        return _TABLE[confidence]
    # Use erfinv approximation
    try:
        from scipy.special import ndtri  # type: ignore
        return float(ndtri((1 + confidence) / 2))
    except ImportError:
        return 1.960  # default to 95%


def is_significant(
    wins: int,
    n: int,
    alpha: float = 0.05,
    null_p: float = _NULL_P,
    method: str = "wilson",
) -> bool:
    """
    Return True if the observed win rate is significantly different from null_p.

    Args:
        wins:    Number of winning trades
        n:       Total trades
        alpha:   Significance level (default 0.05 → 95% CI)
        null_p:  Null-hypothesis win rate (default 0.50)
        method:  "wilson" (CI-based) or "binomial" (p-value test)

    Returns:
        True → deviation is significant, safe to act
        False → could be noise, hold off on parameter changes
    """
    if n <= 0:
        return False
    if method == "wilson":
        lo, hi = wilson_interval(wins, n, confidence=1 - alpha)
        # Significant if null_p is OUTSIDE the confidence interval
        return null_p < lo or null_p > hi
    else:
        pval = binomial_pvalue(wins, n, null_p)
        return pval < alpha


def significance_summary(wins: int, n: int, alpha: float = 0.05) -> dict:
    """
    Full diagnostic dict — useful for logging and the walk-forward dashboard.
    """
    if n <= 0:
        return {"significant": False, "win_rate": 0.0, "n": 0, "pvalue": 1.0,
                "ci_lo": 0.0, "ci_hi": 1.0, "alpha": alpha}
    lo, hi = wilson_interval(wins, n, confidence=1 - alpha)
    pval = binomial_pvalue(wins, n)
    sig = is_significant(wins, n, alpha=alpha)
    return {
        "significant": sig,
        "win_rate": round(wins / n, 4),
        "wins": wins,
        "n": n,
        "pvalue": round(pval, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "alpha": alpha,
    }
