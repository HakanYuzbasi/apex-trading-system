"""
orb_signal.py — Opening Range Breakout (ORB) signal.

The first 30 minutes after NYSE open (9:30–10:00 ET) define the Opening Range.
A breakout above that range's high, confirmed with above-average volume, predicts
continuation with 62–68% accuracy on RVOL ≥ 1.5× days.

The pattern is one of the most consistent intraday edges in equities because:
  1. Institutional orders execute in the first 15-30 minutes.
  2. The breakout signals that one side (buyers or sellers) has absorbed the other.
  3. Confirmed by RVOL — random breakouts fail; volume-confirmed ones persist.

ONLY applies to equity symbols during US session (9:30–16:00 ET weekdays).
Does NOT apply to crypto (24/7 markets have no meaningful opening range).

Usage (called from execution_loop.py or a data pipeline):
    orb = ORBSignal()

    # During first 30 min: feed intraday bars as they arrive
    orb.update_opening_range(symbol, intraday_5min_df)

    # After 10:00 ET: get the directional signal
    ctx = orb.get_signal(symbol, current_price, current_rvol)
    print(ctx.signal, ctx.confidence, ctx.direction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Opening range is 9:30–10:00 ET (30 minutes)
_OR_START_ET = dt_time(9, 30)
_OR_END_ET = dt_time(10, 0)
# Earliest entry: 10:05 ET (give 5-min confirmation candle)
_SIGNAL_START_ET = dt_time(10, 5)
# No new entries after 15:00 ET (final hour too late for ORB momentum)
_SIGNAL_END_ET = dt_time(15, 0)


@dataclass
class ORBContext:
    symbol: str
    signal: float          # [-1, 1]: +1 = bullish breakout, -1 = bearish breakdown
    confidence: float      # [0, 1]: scales with RVOL and breakout magnitude
    direction: str         # "bullish_breakout" | "bearish_breakdown" | "inside" | "no_range"
    or_high: float         # Opening Range high
    or_low: float          # Opening Range low
    current_price: float
    breakout_pct: float    # how far price has moved past the range boundary
    rvol: float            # relative volume at time of check


def _neutral_orb(symbol: str, note: str = "no_range") -> ORBContext:
    return ORBContext(
        symbol=symbol, signal=0.0, confidence=0.0,
        direction=note, or_high=0.0, or_low=0.0,
        current_price=0.0, breakout_pct=0.0, rvol=1.0,
    )


@dataclass
class _RangeData:
    high: float
    low: float
    avg_volume_or: float    # average volume per bar during OR period
    date: str               # YYYY-MM-DD to invalidate stale data


class ORBSignal:
    """
    Tracks Opening Range per symbol and returns breakout signals.

    Typical integration pattern:
      1. Call `update_opening_range(symbol, intraday_df)` each cycle during US morning.
      2. Call `get_signal(symbol, current_price, rvol)` after 10:05 ET for new entries.

    Thread-safe reads; NOT safe for concurrent writes to the same symbol.
    """

    # Minimum RVOL to generate a signal (avoids false breakouts on thin volume)
    MIN_RVOL_FOR_SIGNAL: float = 1.20

    # Breakout must extend at least this far past the range boundary (% of range width)
    MIN_BREAKOUT_EXTENSION: float = 0.003   # 0.3% of price

    def __init__(
        self,
        min_rvol: float = 1.20,
        min_breakout_pct: float = 0.003,
    ) -> None:
        self._min_rvol = min_rvol
        self._min_breakout = min_breakout_pct
        self._ranges: Dict[str, _RangeData] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_opening_range(self, symbol: str, intraday_df: pd.DataFrame) -> None:
        """
        Compute or update the Opening Range for *symbol* from 5-min intraday bars.

        *intraday_df* should have a DatetimeIndex (timezone-aware, or naive UTC/ET)
        and columns: Open, High, Low, Close, Volume.

        Safe to call every cycle — it's idempotent once the range is established.
        """
        if intraday_df is None or intraday_df.empty:
            return
        if not all(c in intraday_df.columns for c in ("High", "Low", "Volume")):
            return

        try:
            import pytz
            et_tz = pytz.timezone("US/Eastern")

            df = intraday_df.copy()
            # Ensure index is timezone-aware
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC").tz_convert(et_tz)
            else:
                df.index = df.index.tz_convert(et_tz)

            today = datetime.now(et_tz).date()
            today_str = str(today)

            # Only compute for today's bars
            df_today = df[df.index.date == today]
            if df_today.empty:
                return

            # Opening Range bars: 9:30 ≤ time < 10:00
            or_mask = (df_today.index.time >= _OR_START_ET) & (df_today.index.time < _OR_END_ET)
            or_bars = df_today[or_mask]

            if len(or_bars) < 2:
                return  # Not enough bars to define the range

            or_high = float(or_bars["High"].max())
            or_low = float(or_bars["Low"].min())
            avg_vol = float(or_bars["Volume"].mean()) if "Volume" in or_bars.columns else 0.0

            if or_high <= or_low:
                return

            # Update range (overwrite with fresher data)
            self._ranges[symbol] = _RangeData(
                high=or_high,
                low=or_low,
                avg_volume_or=avg_vol,
                date=today_str,
            )
            logger.debug(
                "ORB %s: range [%.2f, %.2f] width=%.2f%% avg_vol=%.0f",
                symbol, or_low, or_high,
                (or_high - or_low) / or_low * 100, avg_vol,
            )

        except Exception as e:
            logger.debug("ORBSignal.update_opening_range error (%s): %s", symbol, e)

    def get_signal(
        self,
        symbol: str,
        current_price: float,
        rvol: float = 1.0,
        current_bar_volume: float = 0.0,
    ) -> ORBContext:
        """
        Return the ORB signal for *symbol* at the current moment.

        Only generates a signal:
          - After 10:05 ET (5-min confirmation beyond range formation)
          - Before 15:00 ET (too late for ORB continuation momentum)
          - When RVOL >= MIN_RVOL_FOR_SIGNAL
          - When price has broken the range by MIN_BREAKOUT_EXTENSION

        Returns neutral context outside of US hours / without valid range.
        """
        # Skip non-equity (crypto / FX identified by '/' in symbol)
        if "/" in symbol or symbol.startswith("CRYPTO:") or symbol.startswith("FX:"):
            return _neutral_orb(symbol, "non_equity")

        # Time gate
        try:
            import pytz
            now_et = datetime.now(pytz.timezone("US/Eastern"))
            # Only fire during US session
            if now_et.weekday() >= 5:
                return _neutral_orb(symbol, "weekend")
            now_t = now_et.time()
            if now_t < _SIGNAL_START_ET or now_t >= _SIGNAL_END_ET:
                return _neutral_orb(symbol, "outside_window")
        except Exception:
            pass  # If timezone fails, continue without time gate

        # Range check
        today_str = str(datetime.now().date())
        rdata = self._ranges.get(symbol)
        if rdata is None or rdata.date != today_str:
            return _neutral_orb(symbol, "no_range")

        or_high = rdata.high
        or_low = rdata.low
        or_width = or_high - or_low
        mid = (or_high + or_low) / 2.0

        if or_width <= 0 or mid <= 0 or current_price <= 0:
            return _neutral_orb(symbol, "invalid_range")

        # ── Bullish breakout: price above OR high ─────────────────────────────
        if current_price > or_high + or_width * 0.05:
            breakout_pct = (current_price - or_high) / current_price
            if breakout_pct < self._min_breakout:
                return _neutral_orb(symbol, "inside")  # Not enough extension yet

            # Confidence: scales with RVOL and breakout distance
            vol_conf = min(1.0, max(0.0, (rvol - self._min_rvol) / 1.5)) if rvol >= self._min_rvol else 0.0
            dist_conf = min(1.0, breakout_pct / 0.015)   # 1.5% move = full distance confidence
            confidence = float(vol_conf * 0.60 + dist_conf * 0.40)
            signal = confidence if rvol >= self._min_rvol else 0.0

            logger.debug(
                "ORB %s: BULLISH breakout price=%.2f > OR_high=%.2f, "
                "ext=%.2f%% rvol=%.2f conf=%.2f",
                symbol, current_price, or_high,
                breakout_pct * 100, rvol, confidence,
            )
            return ORBContext(
                symbol=symbol, signal=signal, confidence=confidence,
                direction="bullish_breakout", or_high=or_high, or_low=or_low,
                current_price=current_price, breakout_pct=breakout_pct, rvol=rvol,
            )

        # ── Bearish breakdown: price below OR low ─────────────────────────────
        elif current_price < or_low - or_width * 0.05:
            breakout_pct = (or_low - current_price) / current_price
            if breakout_pct < self._min_breakout:
                return _neutral_orb(symbol, "inside")

            vol_conf = min(1.0, max(0.0, (rvol - self._min_rvol) / 1.5)) if rvol >= self._min_rvol else 0.0
            dist_conf = min(1.0, breakout_pct / 0.015)
            confidence = float(vol_conf * 0.60 + dist_conf * 0.40)
            signal = -confidence if rvol >= self._min_rvol else 0.0

            logger.debug(
                "ORB %s: BEARISH breakdown price=%.2f < OR_low=%.2f, "
                "ext=%.2f%% rvol=%.2f conf=%.2f",
                symbol, current_price, or_low,
                breakout_pct * 100, rvol, confidence,
            )
            return ORBContext(
                symbol=symbol, signal=signal, confidence=confidence,
                direction="bearish_breakdown", or_high=or_high, or_low=or_low,
                current_price=current_price, breakout_pct=breakout_pct, rvol=rvol,
            )

        # Price still inside the range
        return ORBContext(
            symbol=symbol, signal=0.0, confidence=0.0,
            direction="inside", or_high=or_high, or_low=or_low,
            current_price=current_price, breakout_pct=0.0, rvol=rvol,
        )

    def clear_stale_ranges(self) -> None:
        """Drop ranges from previous days. Call once per day at session open."""
        today_str = str(datetime.now().date())
        stale = [s for s, r in self._ranges.items() if r.date != today_str]
        for s in stale:
            del self._ranges[s]
        if stale:
            logger.debug("ORBSignal: cleared %d stale ranges", len(stale))

    def has_range(self, symbol: str) -> bool:
        today_str = str(datetime.now().date())
        r = self._ranges.get(symbol)
        return r is not None and r.date == today_str
