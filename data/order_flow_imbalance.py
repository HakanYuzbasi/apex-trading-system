"""
data/order_flow_imbalance.py — Real-Time Order Flow Imbalance (OFI) Signal

Computes order flow imbalance using the tick-rule VPIN approximation:
    - Up-tick bars (Close > prev Close) → classified as buy volume
    - Down-tick bars (Close < prev Close) → classified as sell volume
    - OFI = (buy_vol - sell_vol) / (buy_vol + sell_vol) ∈ [-1, +1]

This is a standard proxy for true order flow when L2 bid/ask data is unavailable.

The signal is:
    +1.0 → pure aggressive buying (strong bullish flow)
    -1.0 → pure aggressive selling (strong bearish flow)
     0.0 → balanced flow

Blended at 6% weight into the final signal blend.

Config keys:
    OFI_ENABLED              = True
    OFI_WINDOW               = 20    # bars for rolling OFI calculation
    OFI_BLEND_WEIGHT         = 0.06
    OFI_MIN_VOLUME           = 1000  # minimum total volume to compute OFI
    OFI_SMOOTHING_ALPHA      = 0.30  # EMA smoothing on raw OFI per symbol
    OFI_CACHE_TTL_SECONDS    = 30    # cache computed OFI to avoid redundant computation
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "OFI_ENABLED":           True,
    "OFI_WINDOW":            20,
    "OFI_BLEND_WEIGHT":      0.06,
    "OFI_MIN_VOLUME":        1000,
    "OFI_SMOOTHING_ALPHA":   0.30,
    "OFI_CACHE_TTL_SECONDS": 30,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def compute_ofi(df: pd.DataFrame, window: Optional[int] = None) -> float:
    """
    Compute the rolling Order Flow Imbalance signal from OHLCV bar data.

    Args:
        df: DataFrame with columns ['Close', 'Volume'] (at least `window+1` rows)
        window: Number of bars to use (default from config)

    Returns:
        float in [-1.0, 1.0]: positive = net buying, negative = net selling
    """
    if df is None or "Close" not in df.columns or len(df) < 3:
        return 0.0

    n = int(window or _cfg("OFI_WINDOW"))
    min_vol = float(_cfg("OFI_MIN_VOLUME"))

    tail = df.tail(n + 1)
    closes = tail["Close"].values.astype(float)
    volumes = (
        tail["Volume"].values.astype(float)
        if "Volume" in tail.columns
        else np.ones(len(closes))
    )

    if len(closes) < 2:
        return 0.0

    # Tick rule: classify each bar as buy (up-tick) or sell (down-tick)
    buy_vol = 0.0
    sell_vol = 0.0
    for i in range(1, len(closes)):
        v = float(volumes[i]) if i < len(volumes) else 1.0
        if closes[i] > closes[i - 1]:
            buy_vol += v
        elif closes[i] < closes[i - 1]:
            sell_vol += v
        else:
            # Neutral tick: split evenly
            buy_vol += v * 0.5
            sell_vol += v * 0.5

    total = buy_vol + sell_vol
    if total < min_vol:
        return 0.0

    ofi = (buy_vol - sell_vol) / total
    return float(np.clip(ofi, -1.0, 1.0))


class OrderFlowSignal:
    """
    Maintains per-symbol EMA-smoothed OFI and a simple LRU cache.

    Usage:
        ofi_signal = OrderFlowSignal()
        signal = ofi_signal.get_signal("AAPL", df)
    """

    def __init__(self):
        self._ema: Dict[str, float] = {}           # symbol → smoothed OFI
        self._cache: Dict[str, tuple] = {}         # symbol → (ts, value)
        self._raw_buf: Dict[str, list] = defaultdict(list)  # tick buffer (future: live ticks)

    def get_signal(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Return the smoothed OFI signal for `symbol`.

        Uses a short-TTL cache to avoid recomputing on every call within a cycle.
        """
        if not _cfg("OFI_ENABLED"):
            return 0.0

        ttl = float(_cfg("OFI_CACHE_TTL_SECONDS"))
        cached = self._cache.get(symbol)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < ttl:
            return cached[1]

        raw = compute_ofi(df)
        alpha = float(_cfg("OFI_SMOOTHING_ALPHA"))
        prev_ema = self._ema.get(symbol, raw)
        smoothed = alpha * raw + (1.0 - alpha) * prev_ema
        self._ema[symbol] = smoothed
        self._cache[symbol] = (now, smoothed)

        if abs(smoothed) > 0.15:
            logger.debug("OFI %s: raw=%.3f smoothed=%.3f", symbol, raw, smoothed)

        return float(np.clip(smoothed, -1.0, 1.0))

    def record_tick(self, symbol: str, price: float, volume: float, side: str) -> None:
        """
        Record a live trade tick. Side: 'buy' | 'sell' | 'unknown'.
        Used for real-time OFI when websocket provides trade messages.
        """
        buf = self._raw_buf[symbol]
        buf.append({"price": price, "volume": volume, "side": side})
        if len(buf) > 200:
            del buf[:-200]
        # Invalidate cache so next get_signal() recomputes from ticks
        self._cache.pop(symbol, None)

    def get_tick_ofi(self, symbol: str, window: int = 20) -> float:
        """
        Compute OFI from live tick buffer (if available).
        Falls back to 0.0 if insufficient ticks.
        """
        buf = self._raw_buf.get(symbol, [])
        if len(buf) < 5:
            return 0.0

        recent = buf[-window:]
        buy_vol = sum(t["volume"] for t in recent if t["side"] == "buy")
        sell_vol = sum(t["volume"] for t in recent if t["side"] == "sell")
        neutral_vol = sum(t["volume"] for t in recent if t["side"] == "unknown")
        # Split neutral evenly
        buy_vol += neutral_vol * 0.5
        sell_vol += neutral_vol * 0.5
        total = buy_vol + sell_vol
        if total < float(_cfg("OFI_MIN_VOLUME")):
            return 0.0
        return float(np.clip((buy_vol - sell_vol) / total, -1.0, 1.0))

    def get_summary(self) -> Dict[str, float]:
        """Return current smoothed OFI for all tracked symbols."""
        return dict(self._ema)


# ── Module-level singleton ────────────────────────────────────────────────────

_ofi_signal: Optional[OrderFlowSignal] = None


def get_ofi_signal() -> OrderFlowSignal:
    global _ofi_signal
    if _ofi_signal is None:
        _ofi_signal = OrderFlowSignal()
    return _ofi_signal
