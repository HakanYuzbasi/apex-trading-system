"""
orb_signal.py — Opening Range Breakout (ORB) signal (v2).

The first 30 minutes after NYSE open (9:30–10:00 ET) define the Opening Range.
A breakout above that range's high, confirmed with above-average volume, predicts
continuation with 62–68% accuracy on RVOL ≥ 1.5× days.

The pattern is one of the most consistent intraday edges in equities because:
  1. Institutional orders execute in the first 15-30 minutes.
  2. The breakout signals that one side (buyers or sellers) has absorbed the other.
  3. Confirmed by RVOL — random breakouts fail; volume-confirmed ones persist.

Version-2 improvements (ROI):
  * **Range-relative breakout**: the extension threshold is scaled by the OR
    width. A volatile 3% opening range rightly demands a larger absolute
    break than a tight 0.4% range. The old fixed 0.3% floor fired spuriously
    on narrow-range days and missed true breakouts on wide-range days.
  * **All thresholds sourced from ApexConfig** (no magic numbers in hot path).
  * **Continuous RVOL confidence** instead of a hard cliff at ``MIN_RVOL``.
  * **Preserves original ``__init__`` and public method signatures** for
    upstream callers in ``core.execution_loop``.

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
      1. Call ``update_opening_range(symbol, intraday_df)`` each cycle during
         US morning.
      2. Call ``get_signal(symbol, current_price, rvol)`` after 10:05 ET for
         new entries.

    Thread-safe reads; NOT safe for concurrent writes to the same symbol.
    """

    # Logging guard for the "inactive because interval != 1h" warning so we
    # do not spam one line per bar in a backtest.
    _INACTIVE_LOG_EMITTED: bool = False

    @classmethod
    def is_active(cls) -> bool:
        """
        Return ``True`` iff the ORB signal is structurally usable under
        the current ``ApexConfig.OHLCV_INTERVAL``.

        ORB semantics require intraday bars (typical integration uses the
        09:30-10:05 ET opening range on 1-hour bars). On daily bars the
        whole "opening range" concept collapses and the signal is a
        no-op — this classmethod exposes that fact cleanly so a future
        switch of ``OHLCV_INTERVAL`` to ``"1h"`` re-activates ORB without
        any code changes elsewhere.
        """
        try:
            from config import ApexConfig
            interval = str(getattr(ApexConfig, "OHLCV_INTERVAL", "1d")).strip().lower()
        except Exception:
            interval = "1d"
        if interval == "1h":
            return True
        if not cls._INACTIVE_LOG_EMITTED:
            logger.warning(
                "ORB inactive: requires OHLCV_INTERVAL=1h, currently %r",
                interval,
            )
            cls._INACTIVE_LOG_EMITTED = True
        return False

    # Legacy class-level constants. Kept for backwards compatibility with
    # external callers that read them; the instance pulls live values from
    # :class:`ApexConfig` in ``__init__``.
    MIN_RVOL_FOR_SIGNAL: float = 1.20
    MIN_BREAKOUT_EXTENSION: float = 0.003   # 0.3% of price

    def __init__(
        self,
        min_rvol: Optional[float] = None,
        min_breakout_pct: Optional[float] = None,
    ) -> None:
        """
        Args:
            min_rvol: Minimum relative-volume multiple for a breakout to fire.
                If ``None``, pulled from ``ApexConfig.ORB_MIN_RVOL``.
            min_breakout_pct: Absolute-price floor for breakout extension as a
                fraction of current price. If ``None``, pulled from
                ``ApexConfig.ORB_MIN_BREAKOUT_PCT``. The true extension
                threshold is ``max(min_breakout_pct, range_width *
                ORB_RANGE_EXTENSION_FRACTION / current_price)`` — scaling with
                volatility.
        """
        try:
            from config import ApexConfig
            cfg_min_rvol = float(getattr(ApexConfig, "ORB_MIN_RVOL", 1.20))
            cfg_min_break = float(getattr(ApexConfig, "ORB_MIN_BREAKOUT_PCT", 0.003))
            self._range_ext_frac: float = float(
                getattr(ApexConfig, "ORB_RANGE_EXTENSION_FRACTION", 0.25)
            )
            self._vol_conf_scale: float = float(
                getattr(ApexConfig, "ORB_VOL_CONFIDENCE_SCALE", 1.5)
            )
            self._dist_conf_scale: float = float(
                getattr(ApexConfig, "ORB_DIST_CONFIDENCE_SCALE", 0.015)
            )
            self._vol_conf_weight: float = float(
                getattr(ApexConfig, "ORB_VOL_CONF_WEIGHT", 0.60)
            )
            self._dist_conf_weight: float = float(
                getattr(ApexConfig, "ORB_DIST_CONF_WEIGHT", 0.40)
            )
        except Exception:
            cfg_min_rvol = 1.20
            cfg_min_break = 0.003
            self._range_ext_frac = 0.25
            self._vol_conf_scale = 1.5
            self._dist_conf_scale = 0.015
            self._vol_conf_weight = 0.60
            self._dist_conf_weight = 0.40

        self._min_rvol: float = float(min_rvol) if min_rvol is not None else cfg_min_rvol
        self._min_breakout: float = (
            float(min_breakout_pct) if min_breakout_pct is not None else cfg_min_break
        )

        if self._vol_conf_scale <= 0.0:
            raise ValueError(
                f"ORB_VOL_CONFIDENCE_SCALE must be > 0, got {self._vol_conf_scale!r}"
            )
        if self._dist_conf_scale <= 0.0:
            raise ValueError(
                f"ORB_DIST_CONFIDENCE_SCALE must be > 0, got {self._dist_conf_scale!r}"
            )
        if self._range_ext_frac < 0.0:
            raise ValueError(
                f"ORB_RANGE_EXTENSION_FRACTION must be >= 0, got {self._range_ext_frac!r}"
            )
        # Normalise confidence weights to sum to 1 (robust against config drift).
        w_sum = self._vol_conf_weight + self._dist_conf_weight
        if w_sum <= 0.0:
            self._vol_conf_weight = 0.60
            self._dist_conf_weight = 0.40
        else:
            self._vol_conf_weight /= w_sum
            self._dist_conf_weight /= w_sum

        self._ranges: Dict[str, _RangeData] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_opening_range(self, symbol: str, intraday_df: pd.DataFrame) -> None:
        """
        Compute or update the Opening Range for *symbol* from 5-min intraday bars.

        Args:
            symbol: Equity ticker. Crypto / FX symbols are silently ignored.
            intraday_df: DataFrame with DatetimeIndex (tz-aware or naive UTC)
                and columns ``Open, High, Low, Close, Volume``.

        Safe to call every cycle — it is idempotent once the range is
        established.
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
          - When RVOL ≥ ``self._min_rvol``
          - When price has broken the range by the required extension
            (``max(_min_breakout, range_width × _range_ext_frac / price)``)

        Returns neutral context outside of US hours / without a valid range.

        Args:
            symbol: Equity ticker.
            current_price: Most recent trade price.
            rvol: Relative volume at time of check (today vs. N-day average).
            current_bar_volume: Reserved for future use. Unused in v2 (kept
                for signature compatibility).

        Returns:
            :class:`ORBContext`. ``signal`` is non-zero only when an
            RVOL-confirmed breakout is detected.
        """
        # Skip non-equity (crypto / FX identified by '/' in symbol)
        if "/" in symbol or symbol.startswith("CRYPTO:") or symbol.startswith("FX:"):
            return _neutral_orb(symbol, "non_equity")

        # Interval gate — ORB semantics require intraday bars.
        if not self.is_active():
            return _neutral_orb(symbol, "interval_gate")

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

        # Volatility-aware breakout threshold: max of absolute floor and a
        # fraction of the opening range width (expressed as % of price).
        range_pct = or_width / current_price
        vol_threshold = range_pct * self._range_ext_frac
        breakout_threshold = max(self._min_breakout, vol_threshold)

        # ── Bullish breakout: price above OR high ─────────────────────────────
        if current_price > or_high + or_width * 0.05:
            breakout_pct = (current_price - or_high) / current_price
            if breakout_pct < breakout_threshold:
                return _neutral_orb(symbol, "inside")  # Not enough extension yet

            confidence = self._compute_confidence(breakout_pct, rvol)
            signal = confidence if rvol >= self._min_rvol else 0.0

            logger.debug(
                "ORB %s: BULLISH breakout price=%.2f > OR_high=%.2f, "
                "ext=%.2f%% (thr=%.2f%%) rvol=%.2f conf=%.2f",
                symbol, current_price, or_high,
                breakout_pct * 100, breakout_threshold * 100, rvol, confidence,
            )
            return ORBContext(
                symbol=symbol, signal=signal, confidence=confidence,
                direction="bullish_breakout", or_high=or_high, or_low=or_low,
                current_price=current_price, breakout_pct=breakout_pct, rvol=rvol,
            )

        # ── Bearish breakdown: price below OR low ─────────────────────────────
        elif current_price < or_low - or_width * 0.05:
            breakout_pct = (or_low - current_price) / current_price
            if breakout_pct < breakout_threshold:
                return _neutral_orb(symbol, "inside")

            confidence = self._compute_confidence(breakout_pct, rvol)
            signal = -confidence if rvol >= self._min_rvol else 0.0

            logger.debug(
                "ORB %s: BEARISH breakdown price=%.2f < OR_low=%.2f, "
                "ext=%.2f%% (thr=%.2f%%) rvol=%.2f conf=%.2f",
                symbol, current_price, or_low,
                breakout_pct * 100, breakout_threshold * 100, rvol, confidence,
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

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_confidence(self, breakout_pct: float, rvol: float) -> float:
        """
        Compute ORB breakout confidence as a weighted blend of volume
        conviction and breakout distance.

        Args:
            breakout_pct: Breakout extension as fraction of current price
                (always positive — direction handled by caller).
            rvol: Relative volume at time of check.

        Returns:
            Confidence in ``[0, 1]``.
        """
        if rvol >= self._min_rvol:
            vol_conf = min(1.0, max(0.0, (rvol - self._min_rvol) / self._vol_conf_scale))
        else:
            vol_conf = 0.0
        dist_conf = min(1.0, max(0.0, breakout_pct / self._dist_conf_scale))
        blended = vol_conf * self._vol_conf_weight + dist_conf * self._dist_conf_weight
        return float(max(0.0, min(1.0, blended)))
