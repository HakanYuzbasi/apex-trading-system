"""
models/intraday_mr.py — Intraday Mean-Reversion Signal

Generates a counter-trend entry signal when price deviates significantly
from its intraday VWAP anchor AND RSI confirms an extreme condition.
Only activates in mean-reverting / neutral market regimes — suppressed
in trending or crisis environments.

Signal construction:
  1. VWAP deviation z-score: (price - VWAP) / σ(VWAP deviations)
  2. RSI extreme: RSI < 30 (oversold) or RSI > 70 (overbought)
  3. Combined score: deviation z-score × RSI confirmation multiplier
  4. Direction: negative deviation + oversold RSI → BUY signal (+)
                positive deviation + overbought RSI → SELL signal (-)

Output:
  MRSignal(signal, confidence, deviation_z, rsi, regime_eligible)

Wire-in (execution_loop.py after signal blend):
    from models.intraday_mr import IntradayMRSignal
    _mr = IntradayMRSignal()
    mr_result = _mr.compute(intraday_df, regime=self._current_regime)
    if mr_result.regime_eligible and abs(mr_result.signal) > 0.10:
        signal = signal * (1 - MR_WEIGHT) + mr_result.signal * MR_WEIGHT
        confidence *= mr_result.confidence_adj
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regimes where mean-reversion is expected (suppress in trending/crisis)
_MR_ELIGIBLE_REGIMES = {"neutral"}
_MR_SUPPRESS_REGIMES = {"crisis", "strong_bull", "strong_bear"}

# Signal thresholds
_VWAP_DEV_TRIGGER = 1.5     # z-score deviation from VWAP to trigger signal
_RSI_OVERSOLD = 35.0        # RSI below this → oversold (buy candidate)
_RSI_OVERBOUGHT = 65.0      # RSI above this → overbought (sell candidate)
_RSI_EXTREME_OVERSOLD = 25.0
_RSI_EXTREME_OVERBOUGHT = 75.0
_MIN_BARS = 20              # minimum bars for reliable VWAP + RSI


@dataclass
class MRSignal:
    signal: float           # [-1, 1] counter-trend signal
    confidence_adj: float   # [0.85, 1.10] — apply to confidence when fusing
    deviation_z: float      # VWAP deviation z-score
    rsi: float              # RSI value
    vwap: float             # current VWAP
    regime_eligible: bool   # True when regime supports mean-reversion


class IntradayMRSignal:
    """
    Stateless intraday mean-reversion signal generator.
    One instance can be shared across all symbols.
    """

    def __init__(
        self,
        vwap_dev_trigger: float = _VWAP_DEV_TRIGGER,
        rsi_oversold: float = _RSI_OVERSOLD,
        rsi_overbought: float = _RSI_OVERBOUGHT,
        min_bars: int = _MIN_BARS,
    ) -> None:
        self._vwap_trigger = vwap_dev_trigger
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._min_bars = min_bars

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        df: Optional[pd.DataFrame],
        regime: str = "neutral",
    ) -> MRSignal:
        """
        Compute intraday mean-reversion signal from OHLCV DataFrame.

        Args:
            df:     DataFrame with columns ['Close'] and optionally ['Volume'].
                    Should contain intraday bars (5-min or 1-hour).
            regime: Current market regime string.

        Returns:
            MRSignal with signal [-1, 1] and metadata.
        """
        regime_lower = regime.lower()
        eligible = (
            regime_lower in _MR_ELIGIBLE_REGIMES
            or (regime_lower not in _MR_SUPPRESS_REGIMES and regime_lower not in {"volatile"})
        )

        _null = MRSignal(0.0, 1.0, 0.0, 50.0, 0.0, eligible)

        if df is None or len(df) < self._min_bars:
            return _null

        if regime_lower in _MR_SUPPRESS_REGIMES:
            # Hard suppress — trending/crisis markets are NOT mean-reverting
            return MRSignal(0.0, 1.0, 0.0, 50.0, 0.0, False)

        try:
            close = df["Close"].astype(float).values

            # VWAP (volume-weighted if volume available, else equal-weighted)
            vwap = self._compute_vwap(df)

            # VWAP deviation z-score over recent bars
            deviations = (close - vwap) / (abs(vwap) + 1e-9)
            dev_std = float(np.std(deviations[-self._min_bars:]))
            if dev_std < 1e-9:
                return _null
            dev_z = float((close[-1] - vwap) / (dev_std * abs(vwap) + 1e-9))

            # RSI
            rsi = self._rsi(close, 14)

            # Only generate signal when BOTH conditions are met
            signal = self._combine_signal(dev_z, rsi)

            # Confidence adjustment
            if abs(signal) > 0.5:
                conf_adj = 1.05
            elif abs(signal) < 0.10:
                conf_adj = 0.95
            else:
                conf_adj = 1.0

            return MRSignal(
                signal=round(float(np.clip(signal, -1.0, 1.0)), 5),
                confidence_adj=conf_adj,
                deviation_z=round(dev_z, 4),
                rsi=round(rsi, 2),
                vwap=round(float(vwap), 6),
                regime_eligible=eligible,
            )

        except Exception as exc:
            logger.debug("IntradayMR compute error: %s", exc)
            return _null

    # ── Signal construction ───────────────────────────────────────────────────

    def _combine_signal(self, dev_z: float, rsi: float) -> float:
        """
        Combine VWAP deviation and RSI into a mean-reversion signal.

        Logic:
          - Price above VWAP (dev_z > 0) + RSI overbought → SELL (negative)
          - Price below VWAP (dev_z < 0) + RSI oversold   → BUY  (positive)
          - Magnitude: stronger deviation + more extreme RSI = stronger signal
        """
        # RSI confirmation factor [0, 1]
        if rsi <= _RSI_EXTREME_OVERSOLD:
            rsi_factor = 1.0   # strong buy
            rsi_direction = 1.0
        elif rsi <= self._rsi_oversold:
            rsi_factor = (self._rsi_oversold - rsi) / (self._rsi_oversold - _RSI_EXTREME_OVERSOLD)
            rsi_direction = 1.0
        elif rsi >= _RSI_EXTREME_OVERBOUGHT:
            rsi_factor = 1.0   # strong sell
            rsi_direction = -1.0
        elif rsi >= self._rsi_overbought:
            rsi_factor = (rsi - self._rsi_overbought) / (_RSI_EXTREME_OVERBOUGHT - self._rsi_overbought)
            rsi_direction = -1.0
        else:
            return 0.0  # RSI neutral — no mean-reversion signal

        # Deviation must confirm RSI direction (both pointing to same extreme)
        dev_direction = math.copysign(1.0, dev_z)
        if dev_direction == rsi_direction:
            return 0.0  # price already moving toward VWAP — no trade

        # Deviation magnitude factor (stronger when further from VWAP)
        dev_magnitude = min(1.0, max(0.0, (abs(dev_z) - self._vwap_trigger) / self._vwap_trigger))
        if dev_magnitude <= 0:
            return 0.0   # not far enough from VWAP to trigger

        raw = rsi_direction * rsi_factor * dev_magnitude
        return float(np.clip(raw, -1.0, 1.0))

    # ── Technical helpers ─────────────────────────────────────────────────────

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        """VWAP using volume if available, else equal-weighted mean of Close."""
        close = df["Close"].astype(float).values
        if "Volume" in df.columns:
            vol = df["Volume"].astype(float).values
            total_vol = vol.sum()
            if total_vol > 0:
                return float(np.dot(close, vol) / total_vol)
        return float(np.mean(close))

    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        if gains == 0 and losses == 0:
            return 50.0
        if losses == 0:
            return 100.0
        return float(100 - 100 / (1 + gains / losses))

    @staticmethod
    def regime_is_eligible(regime: str) -> bool:
        """True when regime allows mean-reversion entries."""
        r = regime.lower()
        return r not in _MR_SUPPRESS_REGIMES
