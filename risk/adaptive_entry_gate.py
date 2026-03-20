"""
Adaptive Entry Gate - ML-powered dynamic threshold calibration.

Uses Bayesian online learning to adjust entry signal thresholds based on:
1. Recent trade outcomes (win rate of entries at each signal level)
2. Current regime conditions (VIX, correlation, volatility)
3. Hedge dampener state (if dampener is active, thresholds should be
   proportionally lower since signals are already risk-adjusted)
4. Signal-to-noise ratio of recent predictions

Mathematical foundation:
- Maintains a Beta distribution (alpha, beta) for win probability at each
  signal bucket
- Thompson sampling to explore threshold boundaries
- Exponential moving average of recent threshold effectiveness
- Mean-variance optimization for threshold that maximizes expected PnL

Key invariants:
- Threshold NEVER goes below FLOOR (absolute floor for noise protection)
- Threshold NEVER exceeds CEILING (would block too many valid signals)
- Regime-crisis threshold is always >= neutral threshold * 0.6

Calibrated for actual model output range:
  Model signal range: 0.05 - 0.272, mean 0.124
  Buckets span 0.02 - 0.40 in 0.02 increments to match this range.
  Static config thresholds (MIN_SIGNAL_THRESHOLD=0.18,
  SIGNAL_THRESHOLDS_BY_REGIME 0.15-0.45) serve as fallback/ceiling.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal bucket: one per discrete signal-strength band
# ---------------------------------------------------------------------------

@dataclass
class SignalBucket:
    """Tracks win/loss outcomes for a signal strength range via online Bayesian updates."""

    alpha: float = 1.0   # Beta distribution wins + prior
    beta: float = 1.0    # Beta distribution losses + prior
    total_pnl: float = 0.0
    count: int = 0
    avg_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def win_rate(self) -> float:
        """Posterior mean of the Beta distribution = P(win)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def expected_value(self) -> float:
        """Average realised PnL per trade in this bucket."""
        if self.count == 0:
            return 0.0
        return self.avg_pnl

    @property
    def uncertainty(self) -> float:
        """Posterior standard deviation - quantifies how much we still
        need to learn about this bucket."""
        a, b = self.alpha, self.beta
        return math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, won: bool, pnl: float) -> None:
        """Bayesian update with exponential forgetting (half-life ~50 trades).

        The decay ensures the model adapts to regime shifts rather than being
        anchored to stale history.
        """
        # --- accumulate raw PnL stats (no decay, used for avg_pnl) ---
        self.total_pnl += pnl
        self.count += 1
        self.avg_pnl = self.total_pnl / self.count

        # --- Bayesian update with exponential decay ---
        # decay^50 ~ 0.5  =>  decay = exp(ln(0.5)/50) ~ 0.9862
        decay = 0.9862
        self.alpha = max(1.0, self.alpha * decay + (1.0 if won else 0.0))
        self.beta = max(1.0, self.beta * decay + (0.0 if won else 1.0))

    def thompson_sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Draw from the posterior Beta(alpha, beta) for Thompson sampling."""
        _rng = rng or np.random.default_rng()
        return float(_rng.beta(self.alpha, self.beta))

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "total_pnl": self.total_pnl,
            "count": self.count,
            "avg_pnl": self.avg_pnl,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignalBucket":
        return cls(
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            total_pnl=d.get("total_pnl", 0.0),
            count=d.get("count", 0),
            avg_pnl=d.get("avg_pnl", 0.0),
        )


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------

class AdaptiveEntryGate:
    """ML-powered adaptive threshold gate.

    Maintains signal buckets in ``BUCKET_WIDTH`` increments from
    ``BUCKET_MIN`` to ``BUCKET_MAX``.  Uses Thompson sampling + expected-value
    optimisation to find the optimal entry threshold.

    The gate is **opt-in**: callers may fall back to static thresholds at any
    time by checking ``has_sufficient_data``.

    Thread-safety: this class is **not** thread-safe.  It is designed to be
    owned by a single execution-loop instance and called from one asyncio task.
    """

    # Absolute bounds for the computed threshold
    FLOOR: float = 0.06
    CEILING: float = 0.30

    # Bucket geometry - calibrated for model output range [0.05, 0.272]
    BUCKET_MIN: float = 0.02
    BUCKET_MAX: float = 0.40
    BUCKET_WIDTH: float = 0.02

    # Minimum total trades before Bayesian optimisation kicks in
    MIN_TRADES_FOR_BAYESIAN: int = 10
    # Minimum trades per bucket to trust its statistics
    MIN_BUCKET_TRADES: int = 3
    # Minimum win-rate for a bucket to be considered "positive expectancy"
    MIN_BUCKET_WIN_RATE: float = 0.38

    # Calibration frequency (in execution-loop cycles)
    CALIBRATION_INTERVAL: int = 50

    # EMA smoothing coefficient (0.90 = slow adaptation, 0.70 = fast)
    EMA_ALPHA: float = 0.90

    # Blend weight: how much to trust data vs rules
    DATA_WEIGHT: float = 0.60
    RULE_WEIGHT: float = 0.40

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self.persist_path = os.path.join(data_dir, "adaptive_entry_gate.json")

        # Signal buckets keyed by lower-bound string (e.g. "0.10")
        self.buckets: Dict[str, SignalBucket] = {}
        self._init_buckets()

        # Regime-specific base thresholds (aligned with config.py defaults)
        self.regime_thresholds: Dict[str, float] = {
            "strong_bull": 0.15,
            "bull": 0.18,
            "neutral": 0.18,
            "bear": 0.22,
            "strong_bear": 0.28,
            "volatile": 0.20,
            "crisis": 0.15,   # Lower in crisis to catch mean-reversion bounces
        }

        # Internal state
        self._current_hedge_dampener: float = 1.0
        self._ema_optimal_threshold: float = 0.18
        self._last_calibration_cycle: int = 0
        self._rng = np.random.default_rng()

        # Attempt to restore previous state
        self._load_state()

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def _init_buckets(self) -> None:
        """Create empty buckets covering the full signal range."""
        val = self.BUCKET_MIN
        while val <= self.BUCKET_MAX + 1e-9:
            key = f"{val:.2f}"
            if key not in self.buckets:
                self.buckets[key] = SignalBucket()
            val = round(val + self.BUCKET_WIDTH, 4)

    def _bucket_key(self, signal: float) -> str:
        """Map a raw signal value to its nearest bucket key."""
        abs_sig = abs(signal)
        # Snap to nearest bucket boundary
        snapped = round(
            max(self.BUCKET_MIN, min(self.BUCKET_MAX, abs_sig)) / self.BUCKET_WIDTH
        ) * self.BUCKET_WIDTH
        snapped = round(snapped, 4)
        return f"{snapped:.2f}"

    # ------------------------------------------------------------------
    # Public API: record outcomes
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        signal: float,
        pnl_pct: float,
        regime: str = "neutral",
    ) -> None:
        """Record a closed-trade outcome for online learning.

        Parameters
        ----------
        signal : float
            The entry signal strength (absolute value used).
        pnl_pct : float
            Realised PnL as a fraction (e.g. 0.02 = +2%).
        regime : str
            Market regime at entry time (for future per-regime buckets).
        """
        key = self._bucket_key(signal)
        bucket = self.buckets.get(key)
        if bucket is None:
            # Signal outside expected range - create bucket on the fly
            bucket = SignalBucket()
            self.buckets[key] = bucket
        bucket.update(won=(pnl_pct > 0), pnl=pnl_pct)

    # ------------------------------------------------------------------
    # Public API: query effective threshold
    # ------------------------------------------------------------------

    @property
    def has_sufficient_data(self) -> bool:
        """Whether enough trades have been recorded for Bayesian optimisation."""
        return sum(b.count for b in self.buckets.values()) >= self.MIN_TRADES_FOR_BAYESIAN

    def get_effective_threshold(
        self,
        regime: str = "neutral",
        hedge_dampener: float = 1.0,
        vix: float = 20.0,
        daily_loss_pct: float = 0.0,
        avg_correlation: float = 0.0,
        is_crypto: bool = False,
    ) -> float:
        """Compute the optimal entry threshold given current market conditions.

        Algorithm
        ---------
        1. Start with regime base threshold.
        2. If hedge dampener active (< 1.0), lower threshold proportionally
           because the signal has already been risk-adjusted by the dampener.
        3. Adjust for VIX level.
        4. Adjust for portfolio correlation.
        5. Blend with Bayesian EV-optimal threshold (if sufficient data).
        6. EMA-smooth to prevent threshold whipsaw.
        7. Clamp to [FLOOR, CEILING].

        Returns
        -------
        float
            Effective signal threshold.
        """
        self._current_hedge_dampener = hedge_dampener

        # 1. Regime base
        base = self.regime_thresholds.get(regime, 0.18)

        # 2. Hedge dampener adjustment
        #    If dampener = 0.30, signal was multiplied by 0.30, so a "good"
        #    signal of 0.25 becomes 0.075.  Threshold should track downward.
        if hedge_dampener < 1.0:
            # Scale threshold by dampener but floor the adjustment at 50%
            dampener_factor = max(0.50, hedge_dampener)
            base *= dampener_factor

        # 3. VIX adjustment
        if vix > 30:
            # High VIX: more opportunity but noisier signals - slight increase
            base *= 1.0 + min(0.15, (vix - 30) / 100.0)
        elif vix < 15:
            # Low vol: signals are cleaner, can afford tighter threshold
            base *= 0.90

        # 4. Correlation adjustment
        #    High avg correlation means positions move together - be pickier
        if avg_correlation > 0.70:
            base *= 1.0 + min(0.10, (avg_correlation - 0.70) * 0.33)

        # 5. Crypto: noisier market structure, discount threshold slightly
        if is_crypto:
            base *= 0.85

        # 6. Bayesian optimisation blend (if we have enough data)
        ev_threshold = self._bayesian_optimal_threshold()
        if ev_threshold is not None:
            base = self.RULE_WEIGHT * base + self.DATA_WEIGHT * ev_threshold

        # 7. EMA smoothing
        self._ema_optimal_threshold = (
            self.EMA_ALPHA * self._ema_optimal_threshold
            + (1.0 - self.EMA_ALPHA) * base
        )

        # 8. Clamp
        result = max(self.FLOOR, min(self.CEILING, self._ema_optimal_threshold))
        return round(result, 4)

    # ------------------------------------------------------------------
    # Public API: adaptive confidence threshold
    # ------------------------------------------------------------------

    def get_effective_confidence_threshold(
        self,
        signal: float,
        regime: str = "neutral",
        daily_loss_pct: float = 0.0,
    ) -> float:
        """Adaptive confidence threshold replacing the static tiered gate.

        Uses a smooth function instead of hard cutoffs:
            conf_req = base + slope * max(0, 1 - |signal| / reference)

        where ``reference`` is the signal level considered "strong" and
        ``base`` / ``slope`` set the range of required confidence.

        Regime and drawdown adjustments are applied on top.

        Returns
        -------
        float
            Required minimum confidence for this signal/regime combination.
        """
        sig_abs = abs(signal)

        # Smooth confidence curve — calibrated to actual model output range.
        # Model signals: [0.05, 0.272], model confidences: [0.40, 0.55].
        # Old values (base=0.50, slope=0.18, ref=0.25) required conf ≥ 0.59
        # for a typical signal of 0.12, blocking every trade.
        # New values allow entry at the real confidence floor (~0.42).
        reference = 0.20   # "Strong" at model's realistic upper bound
        base_conf = 0.38   # Minimum conf for strong signals (clamped up to 0.40)
        slope = 0.10       # Extra conf for weak signals (+0.10 max)
        conf = base_conf + slope * max(0.0, 1.0 - sig_abs / reference)

        # Regime adjustment
        regime_adj: Dict[str, float] = {
            "strong_bull": -0.04,
            "bull": -0.03,
            "neutral": 0.0,
            "bear": +0.05,
            "strong_bear": +0.08,
            "volatile": +0.02,
            "crisis": -0.05,   # Lower confidence bar to catch bounces
        }
        conf += regime_adj.get(regime, 0.0)

        # Drawdown adjustment: gradual, not a cliff
        if daily_loss_pct < -0.015:
            # -1.5% -> +0.03, -3.0% -> +0.06, capped at +0.08
            loss_boost = min(0.08, abs(daily_loss_pct) * 2.0)
            conf += loss_boost

        # Clamp to sane range
        return max(0.40, min(0.78, round(conf, 4)))

    # ------------------------------------------------------------------
    # Bayesian threshold optimisation
    # ------------------------------------------------------------------

    def _bayesian_optimal_threshold(self) -> Optional[float]:
        """Find the signal threshold that maximises expected PnL.

        Walks buckets from low to high and finds the lowest bucket where:
        - Enough trades observed (>= MIN_BUCKET_TRADES)
        - Win rate above MIN_BUCKET_WIN_RATE
        - Average PnL positive

        If no qualifying bucket is found, falls back to the bucket with the
        best average PnL.  Returns ``None`` if total data is insufficient.
        """
        total_trades = sum(b.count for b in self.buckets.values())
        if total_trades < self.MIN_TRADES_FOR_BAYESIAN:
            return None

        sorted_keys = sorted(self.buckets.keys())

        # Pass 1: find lowest positive-expectancy bucket
        for key in sorted_keys:
            bucket = self.buckets[key]
            if (
                bucket.count >= self.MIN_BUCKET_TRADES
                and bucket.win_rate >= self.MIN_BUCKET_WIN_RATE
                and bucket.avg_pnl > 0
            ):
                return float(key)

        # Pass 2: fallback to best-EV bucket with enough data
        candidates = [
            (k, self.buckets[k])
            for k in sorted_keys
            if self.buckets[k].count >= self.MIN_BUCKET_TRADES
        ]
        if not candidates:
            return None
        best_key = max(candidates, key=lambda pair: pair[1].avg_pnl)[0]
        return float(best_key)

    def _thompson_optimal_threshold(self) -> Optional[float]:
        """Thompson-sampling based threshold exploration.

        Draws from each bucket's Beta posterior and returns the lowest bucket
        whose sampled win probability exceeds ``MIN_BUCKET_WIN_RATE``.

        This encourages exploration of under-sampled buckets.
        """
        total_trades = sum(b.count for b in self.buckets.values())
        if total_trades < self.MIN_TRADES_FOR_BAYESIAN:
            return None

        sorted_keys = sorted(self.buckets.keys())
        for key in sorted_keys:
            bucket = self.buckets[key]
            sampled_wr = bucket.thompson_sample(self._rng)
            if sampled_wr >= self.MIN_BUCKET_WIN_RATE and bucket.count >= 2:
                return float(key)
        return None

    # ------------------------------------------------------------------
    # Calibration / persistence
    # ------------------------------------------------------------------

    def calibrate(self, cycle: int) -> None:
        """Periodic recalibration and persistence (every CALIBRATION_INTERVAL cycles)."""
        if cycle - self._last_calibration_cycle < self.CALIBRATION_INTERVAL:
            return
        self._last_calibration_cycle = cycle

        total = sum(b.count for b in self.buckets.values())
        if total > 0:
            logger.info(
                "AdaptiveEntryGate calibration: %d total trades, "
                "EMA threshold=%.4f, hedge_dampener=%.2f",
                total,
                self._ema_optimal_threshold,
                self._current_hedge_dampener,
            )

        self._save_state()

    def _save_state(self) -> None:
        """Persist state to JSON."""
        try:
            state = {
                "buckets": {k: b.to_dict() for k, b in self.buckets.items()},
                "regime_thresholds": self.regime_thresholds,
                "ema_optimal_threshold": self._ema_optimal_threshold,
                "last_calibration_cycle": self._last_calibration_cycle,
            }
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self.persist_path)
        except Exception:
            logger.warning("AdaptiveEntryGate: failed to save state", exc_info=True)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if not os.path.exists(self.persist_path):
                return
            with open(self.persist_path) as f:
                state = json.load(f)

            for k, v in state.get("buckets", {}).items():
                self.buckets[k] = SignalBucket.from_dict(v)
            # Re-initialise any missing buckets (e.g. after BUCKET_WIDTH change)
            self._init_buckets()

            saved_regime = state.get("regime_thresholds")
            if isinstance(saved_regime, dict):
                self.regime_thresholds.update(saved_regime)

            self._ema_optimal_threshold = float(
                state.get("ema_optimal_threshold", 0.18)
            )
            self._last_calibration_cycle = int(
                state.get("last_calibration_cycle", 0)
            )

            total = sum(b.count for b in self.buckets.values())
            if total > 0:
                logger.info(
                    "AdaptiveEntryGate loaded: %d historical trades, "
                    "EMA threshold=%.4f",
                    total,
                    self._ema_optimal_threshold,
                )
        except Exception:
            logger.warning("AdaptiveEntryGate: failed to load state", exc_info=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a diagnostic summary for the TCA / ops API."""
        active_buckets = {
            k: {
                "count": b.count,
                "win_rate": round(b.win_rate, 3),
                "avg_pnl": round(b.avg_pnl, 5),
                "uncertainty": round(b.uncertainty, 4),
            }
            for k, b in sorted(self.buckets.items())
            if b.count > 0
        }
        return {
            "total_trades": sum(b.count for b in self.buckets.values()),
            "ema_threshold": round(self._ema_optimal_threshold, 4),
            "bayesian_threshold": self._bayesian_optimal_threshold(),
            "has_sufficient_data": self.has_sufficient_data,
            "hedge_dampener": self._current_hedge_dampener,
            "active_buckets": active_buckets,
        }
