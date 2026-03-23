"""
risk/correlation_early_warning.py — Correlation Regime Shift Early Warning

The existing HedgeManager reacts when equity-crypto correlation IS already high
(≥0.85). By then positions have already started correlated drawdowns.

This module detects the RISE in correlation — a velocity signal — and emits
a position multiplier BEFORE the correlation peaks.

Mechanism:
  - Maintain two rolling windows of SPY × BTC log-returns:
      short_window (~30 min): captures current correlation regime
      long_window  (~4 h):    captures baseline / recent-past correlation
  - correlation_velocity = short_corr − long_corr
  - When velocity > rising_threshold AND short_corr > shift_threshold:
      we are in a regime shift → reduce position sizes proactively
  - Multiplier tiers:
      normal  (vel < 0.15):  1.00
      caution (vel 0.15-0.30): 0.80
      warning (vel 0.30-0.50): 0.65
      alert   (vel > 0.50):  0.50

Integrated into execution_loop: replaces part of hedge_manager logic and
provides earlier position sizing reduction.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationStats:
    short_corr: float         # Rolling ~30-min correlation
    long_corr: float          # Rolling ~4h correlation (baseline)
    velocity: float           # short_corr − long_corr
    position_multiplier: float
    tier: str                 # "normal" | "caution" | "warning" | "alert"
    samples_short: int
    samples_long: int
    timestamp: datetime


class CorrelationEarlyWarning:
    """
    Proactive correlation regime shift detector.

    Call record_prices(spy, btc) each cycle.
    Call get_position_multiplier() to get the current sizing constraint.

    Requires at least short_window samples before producing non-trivial output.
    Returns multiplier=1.0 (normal) until enough data is accumulated.
    """

    def __init__(
        self,
        short_window: int = 12,          # bars for "current" correlation (~30 min at 2.5 min/cycle)
        long_window: int = 96,           # bars for "baseline" correlation (~4 h)
        rising_threshold: float = 0.20,  # velocity above this = regime shifting
        alert_threshold: float = 0.45,   # velocity above this = urgent
        shift_threshold: float = 0.35,   # short_corr must also exceed this
        min_samples: int = 8,            # minimum bars before producing a signal
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.rising_threshold = rising_threshold
        self.alert_threshold = alert_threshold
        self.shift_threshold = shift_threshold
        self.min_samples = min_samples

        self._spy_prices: Deque[float] = deque(maxlen=long_window + 1)
        self._btc_prices: Deque[float] = deque(maxlen=long_window + 1)
        self._last_stats: Optional[CorrelationStats] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def record_prices(self, spy_price: float, btc_price: float) -> None:
        """Record current SPY and BTC prices. Call once per main loop cycle."""
        if spy_price and spy_price > 0 and btc_price and btc_price > 0:
            self._spy_prices.append(float(spy_price))
            self._btc_prices.append(float(btc_price))
            self._last_stats = self._compute()

    def get_position_multiplier(self) -> float:
        """Return position sizing multiplier [0.50, 1.00]."""
        stats = self._last_stats
        if stats is None:
            return 1.0
        return stats.position_multiplier

    def get_stats(self) -> Optional[CorrelationStats]:
        """Return full diagnostics dict."""
        return self._last_stats

    def get_diagnostics(self) -> Dict:
        stats = self._last_stats
        if stats is None:
            return {
                "tier": "normal",
                "position_multiplier": 1.0,
                "short_corr": None,
                "long_corr": None,
                "velocity": None,
                "samples": len(self._spy_prices),
            }
        return {
            "tier": stats.tier,
            "position_multiplier": stats.position_multiplier,
            "short_corr": round(stats.short_corr, 3),
            "long_corr": round(stats.long_corr, 3),
            "velocity": round(stats.velocity, 3),
            "samples_short": stats.samples_short,
            "samples_long": stats.samples_long,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute(self) -> CorrelationStats:
        n = len(self._spy_prices)

        if n < self.min_samples + 1:
            return CorrelationStats(
                short_corr=0.0, long_corr=0.0, velocity=0.0,
                position_multiplier=1.0, tier="normal",
                samples_short=n, samples_long=n,
                timestamp=datetime.now(),
            )

        spy_arr = np.array(self._spy_prices)
        btc_arr = np.array(self._btc_prices)

        # Log returns
        spy_ret = np.diff(np.log(spy_arr))
        btc_ret = np.diff(np.log(btc_arr))

        short_n = min(self.short_window, len(spy_ret))
        long_n = len(spy_ret)

        short_corr = self._rolling_corr(spy_ret[-short_n:], btc_ret[-short_n:])
        long_corr = self._rolling_corr(spy_ret, btc_ret)

        velocity = short_corr - long_corr

        # Tier assignment
        tier, mult = self._classify(short_corr, velocity)

        return CorrelationStats(
            short_corr=short_corr,
            long_corr=long_corr,
            velocity=velocity,
            position_multiplier=mult,
            tier=tier,
            samples_short=short_n,
            samples_long=long_n,
            timestamp=datetime.now(),
        )

    def _classify(self, short_corr: float, velocity: float) -> Tuple[str, float]:
        """Map (corr, velocity) → (tier, multiplier)."""
        # Only trigger when correlation is meaningfully positive AND rising
        if short_corr < self.shift_threshold:
            return "normal", 1.0

        if velocity > self.alert_threshold:
            return "alert", 0.50
        if velocity > self.rising_threshold * 1.5:
            return "warning", 0.65
        if velocity > self.rising_threshold:
            return "caution", 0.80
        return "normal", 1.0

    @staticmethod
    def _rolling_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if x.std() < 1e-10 or y.std() < 1e-10:
            return 0.0
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
            return corr if math.isfinite(corr) else 0.0
        except Exception:
            return 0.0
