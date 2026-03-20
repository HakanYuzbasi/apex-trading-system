"""
risk/vol_targeting.py

Portfolio-level volatility targeting.

Rescales the common position-size multiplier each cycle so that the
portfolio's expected annualised volatility stays close to TARGET_VOL_ANN
(default 15 %).

Algorithm
---------
1. Pull the daily P&L (equity-curve returns) from the last LOOKBACK days.
2. Compute realised annualised vol = std(returns) × √ann_factor.
3. multiplier = TARGET_VOL_ANN / realised_vol   (clamped to [MIN, MAX]).
4. Return the multiplier for the execution loop to apply.

Falls back to 1.0 when fewer than MIN_DAYS data points are available.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class VolTargeting:
    """
    Compute a scalar size multiplier to keep realised portfolio vol near
    a target.

    Parameters
    ----------
    target_vol_ann : float
        Target annualised portfolio volatility (e.g. 0.15 for 15 %).
    lookback_days : int
        Rolling window of daily returns used for realised-vol estimate.
    min_days : int
        Minimum number of daily returns required before multiplier is
        applied (returns 1.0 otherwise).
    min_mult : float
        Hard floor on the multiplier (never go below this).
    max_mult : float
        Hard ceiling on the multiplier (never lever above this).
    ann_factor : int
        Annualisation factor: 252 for equities, 365 for crypto.
    """

    def __init__(
        self,
        target_vol_ann: float = 0.15,
        lookback_days: int = 20,
        min_days: int = 5,
        min_mult: float = 0.30,
        max_mult: float = 2.00,
        ann_factor: int = 252,
    ) -> None:
        self.target_vol_ann = float(target_vol_ann)
        self.lookback_days = int(lookback_days)
        self.min_days = int(min_days)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.ann_factor = int(ann_factor)
        self._daily_returns: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, portfolio_return_today: float) -> None:
        """
        Feed today's portfolio return (fractional, e.g. −0.012 for −1.2 %).
        Should be called once per trading day after P&L is known.
        """
        self._daily_returns.append(float(portfolio_return_today))
        # Keep only lookback window
        if len(self._daily_returns) > self.lookback_days:
            self._daily_returns = self._daily_returns[-self.lookback_days:]

    def get_multiplier(self) -> float:
        """
        Return the current vol-targeting size multiplier.

        Returns 1.0 if insufficient history. Clamps to [min_mult, max_mult].
        """
        if len(self._daily_returns) < self.min_days:
            return 1.0

        arr = np.array(self._daily_returns, dtype=float)
        realized_vol = float(arr.std()) * math.sqrt(self.ann_factor)

        if realized_vol < 1e-9:
            return 1.0

        mult = self.target_vol_ann / realized_vol
        mult = float(np.clip(mult, self.min_mult, self.max_mult))
        logger.debug(
            "VolTargeting: realised_vol=%.2f%% target=%.0f%% → mult=%.3f",
            realized_vol * 100, self.target_vol_ann * 100, mult,
        )
        return mult

    @property
    def realized_vol_ann(self) -> Optional[float]:
        """Current annualised realised vol, or None if insufficient data."""
        if len(self._daily_returns) < self.min_days:
            return None
        arr = np.array(self._daily_returns, dtype=float)
        return float(arr.std()) * math.sqrt(self.ann_factor)

    def feed_equity_curve(self, equity_values: Sequence[float]) -> None:
        """
        Bootstrap from a sequence of equity values (oldest→newest).
        Computes daily returns internally and fills the rolling window.
        """
        vals = [float(v) for v in equity_values if v and v > 0]
        if len(vals) < 2:
            return
        returns = [
            (vals[i] - vals[i - 1]) / max(vals[i - 1], 1e-9)
            for i in range(1, len(vals))
        ]
        for r in returns[-self.lookback_days:]:
            self._daily_returns.append(r)
        if len(self._daily_returns) > self.lookback_days:
            self._daily_returns = self._daily_returns[-self.lookback_days:]
