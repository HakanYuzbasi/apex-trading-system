"""
risk/entry_filters.py

Lightweight, zero-network entry quality filters using existing daily OHLCV bars.

1. VWAP Deviation Gate
   - 20-day rolling VWAP from daily bars (TP × Volume weighted)
   - Blocks long entries more than (ATR-adjusted)% above VWAP
   - Blocks short entries more than (ATR-adjusted)% below VWAP
   - ATR-adjusted: avoids penalising high-vol names for normal daily range
   - Crypto gets 2× threshold (higher baseline volatility)

2. RVOL (Relative Volume) Gate
   - RVOL = today's volume / 20-day average daily volume
   - RVOL < threshold → skip entry (low participation = unreliable signal)
   - Uses last bar in historical_data; zero new network calls

Both gates are pure-Python, synchronous, and use historical_data already in memory.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── VWAP Gate ────────────────────────────────────────────────────────────────

def _compute_20d_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    20-day typical-price VWAP from daily bars.
    TP = (High + Low + Close) / 3 ; VWAP = Σ(TP × Vol) / Σ(Vol)
    Returns None when volume data is absent or insufficient.
    """
    try:
        if df is None or len(df) < 5:
            return None
        if "Volume" not in df.columns:
            return None

        close = df["Close"].values.astype(float)
        high = df["High"].values.astype(float) if "High" in df.columns else close
        low = df["Low"].values.astype(float) if "Low" in df.columns else close
        vol = df["Volume"].values.astype(float)

        n = min(20, len(df))
        tp = (high[-n:] + low[-n:] + close[-n:]) / 3.0
        vol_n = vol[-n:]

        total_vol = float(vol_n.sum())
        if total_vol <= 0:
            return None

        return float((tp * vol_n).sum() / total_vol)
    except Exception:
        return None


def vwap_gate_check(
    price: float,
    df: pd.DataFrame,
    signal: float,
    atr_pct: float = 0.0,
    is_crypto: bool = False,
    max_deviation_pct: float = 2.0,
    atr_adjust: bool = True,
) -> Tuple[bool, float, str]:
    """
    Check whether the current price is too extended from 20-day VWAP for a clean entry.

    Args:
        price:            Current market price.
        df:               Historical daily OHLCV DataFrame (from historical_data[symbol]).
        signal:           Signal value; >0 = long intent, <0 = short intent.
        atr_pct:          20-day ATR as % of price (0 = unknown).
        is_crypto:        Crypto assets use 2× threshold.
        max_deviation_pct: Baseline max deviation from VWAP (%).
        atr_adjust:       Scale threshold up with ATR so volatile assets aren't penalised.

    Returns:
        (blocked, deviation_pct, reason)
        deviation_pct > 0 → price above VWAP; < 0 → price below VWAP.
    """
    vwap = _compute_20d_vwap(df)
    if vwap is None or vwap <= 0:
        return False, 0.0, ""

    deviation_pct = (price / vwap - 1.0) * 100.0

    # ATR-adjusted threshold: floor at max_deviation_pct, scale up with ATR
    if atr_adjust and atr_pct > 0:
        # Allow up to 1.5× ATR deviation; cap at 3× baseline
        threshold = max(max_deviation_pct, min(atr_pct * 1.5, max_deviation_pct * 3.0))
    else:
        threshold = max_deviation_pct

    # Crypto has 2× baseline (BTC/ETH easily move 3-5% intraday)
    if is_crypto:
        threshold *= 2.0

    is_long = signal > 0
    if is_long and deviation_pct > threshold:
        return (
            True,
            deviation_pct,
            f"VWAP gate: {deviation_pct:+.1f}% above 20d VWAP (limit {threshold:.1f}%)",
        )
    if not is_long and deviation_pct < -threshold:
        return (
            True,
            deviation_pct,
            f"VWAP gate: {abs(deviation_pct):.1f}% below 20d VWAP (limit {threshold:.1f}%)",
        )
    return False, deviation_pct, ""


# ─── RVOL Gate ────────────────────────────────────────────────────────────────

def rvol_check(
    df: pd.DataFrame,
    min_rvol: float = 0.30,
) -> Tuple[float, bool, str]:
    """
    Relative Volume check: current day's volume vs 20-day average.

    RVOL < min_rvol → low participation → signal is likely noise.

    Args:
        df:        Historical daily OHLCV DataFrame (latest row = today).
        min_rvol:  Block threshold (default 0.30 = 30% of average volume).

    Returns:
        (rvol_ratio, blocked, reason)
    """
    try:
        if df is None or "Volume" not in df.columns or len(df) < 5:
            return 1.0, False, ""

        vol = df["Volume"].dropna().values.astype(float)
        if len(vol) < 5:
            return 1.0, False, ""

        current_vol = float(vol[-1])
        # 20-day average excluding today
        lookback = vol[-21:-1] if len(vol) >= 21 else vol[:-1]
        avg_vol = float(lookback.mean()) if len(lookback) > 0 else 0.0

        if avg_vol <= 0:
            return 1.0, False, ""

        rvol = current_vol / avg_vol

        if rvol < min_rvol:
            return (
                rvol,
                True,
                f"RVOL={rvol:.2f}× avg (below {min_rvol:.2f}× threshold — low participation)",
            )
        return rvol, False, ""
    except Exception:
        return 1.0, False, ""
