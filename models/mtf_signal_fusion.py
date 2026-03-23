"""
models/mtf_signal_fusion.py — Multi-Timeframe Signal Fusion

Combines signals derived from three timeframes:
  - 5-min bars  (intraday momentum / noise filter)
  - 1-hour bars (intermediate trend confirmation)
  - Daily bars  (regime-level directional bias)

The final fused signal is a weighted average where the timeframe weights
adapt to the current market regime:

  Trending (bull/strong_bull/bear/strong_bear):
      5m=0.20,  1h=0.45,  daily=0.35
  Mean-reverting / neutral:
      5m=0.35,  1h=0.35,  daily=0.30
  Volatile / crisis:
      5m=0.15,  1h=0.30,  daily=0.55   ← rely on higher-timeframe stability

The 1-hour signal is derived from EMA crossover + RSI of hourly bars.
The 5-min signal is a simple short-term momentum score.
The daily signal is the existing signal passed in by the caller.

Usage (inside execution_loop.process_symbol):
    from models.mtf_signal_fusion import MTFSignalFuser
    fuser = MTFSignalFuser()
    fused = fuser.fuse(
        daily_signal=signal,
        hourly_df=hourly_data,
        fivemin_df=fivemin_data,
        regime=current_regime,
    )
    # fused.signal, fused.confidence_adj, fused.dominant_tf
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime weight maps ────────────────────────────────────────────────────────
_TF_WEIGHTS: dict[str, tuple[float, float, float]] = {
    # (5m, 1h, daily)
    "strong_bull":  (0.20, 0.45, 0.35),
    "bull":         (0.20, 0.45, 0.35),
    "neutral":      (0.35, 0.35, 0.30),
    "bear":         (0.20, 0.45, 0.35),
    "strong_bear":  (0.20, 0.45, 0.35),
    "volatile":     (0.15, 0.30, 0.55),
    "crisis":       (0.10, 0.25, 0.65),
}
_DEFAULT_WEIGHTS = (0.25, 0.40, 0.35)

# Minimum bars needed to compute timeframe signal
_MIN_1H_BARS = 5
_MIN_5M_BARS = 6


@dataclass
class MTFFusedSignal:
    signal: float           # fused signal [-1, 1]
    confidence_adj: float   # multiplicative adj to apply to base confidence [0.8, 1.15]
    dominant_tf: str        # which timeframe contributed most
    tf_signals: dict        # {"5m": v, "1h": v, "daily": v}
    weights: dict           # {"5m": w, "1h": w, "daily": w}
    aligned: bool           # True if all three TFs agree on direction


class MTFSignalFuser:
    """
    Stateless multi-timeframe signal fuser.
    One instance can be shared (no mutable state).
    """

    def fuse(
        self,
        daily_signal: float,
        hourly_df: Optional[pd.DataFrame],
        fivemin_df: Optional[pd.DataFrame],
        regime: str = "neutral",
    ) -> MTFFusedSignal:
        """
        Fuse three timeframe signals into one.

        Args:
            daily_signal:  The existing daily-bar signal [-1, 1].
            hourly_df:     DataFrame of 1-hour OHLCV bars (indexed newest-last).
            fivemin_df:    DataFrame of 5-min OHLCV bars (indexed newest-last).
            regime:        Current market regime string.

        Returns:
            MTFFusedSignal with fused signal and metadata.
        """
        w5, w1h, wd = _TF_WEIGHTS.get(regime.lower(), _DEFAULT_WEIGHTS)

        sig_1h = self._hourly_signal(hourly_df)
        sig_5m = self._fivemin_signal(fivemin_df)
        sig_d = float(np.clip(daily_signal, -1.0, 1.0))

        tf_signals = {"5m": sig_5m, "1h": sig_1h, "daily": sig_d}

        # Fuse: weighted average, fall back to daily when higher-TF data absent
        if sig_1h is None and sig_5m is None:
            fused = sig_d
            w5, w1h, wd = 0.0, 0.0, 1.0
        elif sig_1h is None:
            w5_adj = w5 / (w5 + wd)
            wd_adj = wd / (w5 + wd)
            fused = w5_adj * (sig_5m or 0.0) + wd_adj * sig_d
            w5, w1h, wd = w5_adj, 0.0, wd_adj
        elif sig_5m is None:
            w1h_adj = w1h / (w1h + wd)
            wd_adj = wd / (w1h + wd)
            fused = w1h_adj * sig_1h + wd_adj * sig_d
            w5, w1h, wd = 0.0, w1h_adj, wd_adj
        else:
            fused = w5 * sig_5m + w1h * sig_1h + wd * sig_d

        fused = float(np.clip(fused, -1.0, 1.0))

        # Dominant timeframe
        contribs = {
            "5m": abs((sig_5m or 0.0) * w5),
            "1h": abs((sig_1h or 0.0) * w1h),
            "daily": abs(sig_d * wd),
        }
        dominant_tf = max(contribs, key=contribs.get)

        # Alignment check: all available TFs agree on sign
        available_sigs = [s for s in (sig_5m, sig_1h, sig_d) if s is not None]
        if len(available_sigs) >= 2:
            signs = [math.copysign(1, s) for s in available_sigs if abs(s) > 0.01]
            aligned = len(set(signs)) == 1
        else:
            aligned = True

        # Confidence adjustment
        if aligned and len(available_sigs) >= 2:
            conf_adj = 1.10  # all TFs agree → boost confidence
        elif not aligned:
            conf_adj = 0.85  # TFs disagree → reduce confidence
        else:
            conf_adj = 1.0

        # Additional: if fused diverges strongly from daily, dampen slightly
        divergence = abs(fused - sig_d)
        if divergence > 0.20:
            conf_adj = max(0.80, conf_adj - 0.05)

        tf_weights_out = {"5m": round(w5, 3), "1h": round(w1h, 3), "daily": round(wd, 3)}
        tf_signals_clean = {k: round(v, 4) if v is not None else None for k, v in tf_signals.items()}

        return MTFFusedSignal(
            signal=round(fused, 5),
            confidence_adj=round(conf_adj, 4),
            dominant_tf=dominant_tf,
            tf_signals=tf_signals_clean,
            weights=tf_weights_out,
            aligned=aligned,
        )

    # ── Timeframe-specific signal extractors ──────────────────────────────────

    def _hourly_signal(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        """EMA crossover + RSI on 1-hour bars → signal in [-1, 1]."""
        if df is None or len(df) < _MIN_1H_BARS:
            return None
        try:
            close = df["Close"].astype(float).values
            ema_fast = self._ema(close, 5)
            ema_slow = self._ema(close, 12)
            cross = (ema_fast[-1] - ema_slow[-1]) / (abs(ema_slow[-1]) + 1e-9)
            rsi = self._rsi(close, 10)
            # Combine: EMA cross direction + RSI deviation from 50
            rsi_signal = (rsi - 50.0) / 50.0  # [-1, 1]
            raw = 0.60 * cross * 5.0 + 0.40 * rsi_signal  # scale cross to comparable range
            return float(np.clip(raw, -1.0, 1.0))
        except Exception as exc:
            logger.debug("MTF hourly signal error: %s", exc)
            return None

    def _fivemin_signal(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        """Short-term momentum on 5-min bars → signal in [-1, 1]."""
        if df is None or len(df) < _MIN_5M_BARS:
            return None
        try:
            close = df["Close"].astype(float).values
            # 6-bar ROC as momentum proxy
            roc = (close[-1] - close[-6]) / (abs(close[-6]) + 1e-9)
            # Vol-adjusted: divide by realised vol of last 12 bars
            vol = float(np.std(np.diff(close[-13:]))) if len(close) >= 13 else 0.001
            vol = max(vol, 1e-6)
            z = roc / (vol + 1e-9)
            sig = float(np.clip(z * 0.3, -1.0, 1.0))
            return sig
        except Exception as exc:
            logger.debug("MTF 5m signal error: %s", exc)
            return None

    # ── Technical helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _ema(prices: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        ema = np.empty_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period
        if gains == 0 and losses == 0:
            return 50.0  # flat market → neutral
        if losses == 0:
            return 100.0
        rs = gains / losses
        return float(100 - 100 / (1 + rs))
