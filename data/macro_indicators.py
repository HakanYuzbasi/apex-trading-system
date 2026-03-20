"""
data/macro_indicators.py

Real-time macro regime overlay for Apex Trading.
Pulls free, no-key-required data via yfinance:

  • Yield curve slope   — 10Y − 2Y US Treasury spread (^TNX minus ^IRX)
  • VIX term structure  — VX1/VX2 ratio (contango vs backwardation)
  • Dollar momentum     — DXY 20-day rate-of-change

The resulting MacroContext is injected into the execution loop once per cycle
(cached 15 minutes) to adjust position sizing and entry confidence.

Usage:
    macro = MacroIndicators()
    ctx = await macro.get_context()
    # ctx.risk_appetite ∈ [-1, 1]  (-1 = maximum risk-off, +1 = maximum risk-on)
    # ctx.yield_curve_inverted → reduce equity long sizes
    # ctx.vix_backwardation    → dampen all new entries
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_TTL = 900  # 15 minutes


@dataclass
class MacroContext:
    """Aggregated macro regime snapshot."""
    # Raw values
    yield_curve_slope: float      # 10Y - 2Y spread in percentage points
    vix_futures_ratio: float      # VX1 / VX2  (>1 = backwardation = stress)
    dxy_momentum_20d: float       # DXY 20-day ROC (positive = dollar strengthening)
    vix_spot: float               # Current VIX level

    # Derived flags
    yield_curve_inverted: bool    # slope < 0
    vix_backwardation: bool       # VX1 > VX2 (near-term vol premium = stress)
    dollar_risk_off: bool         # DXY ROC > +1% over 20 days

    # Composite score ∈ [-1, 1]
    # -1 = maximum risk-off (inverted curve + VIX backwardation + strong dollar)
    # +1 = maximum risk-on  (steep curve + VIX contango + weak dollar)
    risk_appetite: float

    # Human-readable label
    regime_signal: str            # "risk_on" | "neutral" | "risk_off" | "crisis"

    @property
    def equity_size_multiplier(self) -> float:
        """Multiplier for equity position sizes based on macro context."""
        from config import ApexConfig
        mult = 1.0
        if self.yield_curve_inverted:
            mult *= float(getattr(ApexConfig, "MACRO_YIELD_CURVE_INVERSION_SIZE_MULT", 0.75))
        if self.vix_backwardation:
            mult *= float(getattr(ApexConfig, "MACRO_VIX_BACKWARDATION_DAMPENER", 0.80))
        return max(0.40, min(1.0, mult))

    @property
    def crypto_size_multiplier(self) -> float:
        """Multiplier for crypto position sizes. Crypto is MORE sensitive to macro."""
        mult = self.equity_size_multiplier
        if self.dollar_risk_off:
            mult *= 0.85  # Strong dollar headwind for crypto
        if self.vix_spot > 30:
            mult *= 0.80
        return max(0.30, min(1.0, mult))


_NEUTRAL_MACRO = MacroContext(
    yield_curve_slope=1.0,
    vix_futures_ratio=1.0,
    dxy_momentum_20d=0.0,
    vix_spot=15.0,
    yield_curve_inverted=False,
    vix_backwardation=False,
    dollar_risk_off=False,
    risk_appetite=0.0,
    regime_signal="neutral",
)


class MacroIndicators:
    """
    Fetches and caches macro indicators. Thread-safe via asyncio.Lock.
    Returns _NEUTRAL_MACRO on any fetch failure so the system degrades gracefully.
    """

    # Tickers
    _TNX = "^TNX"   # 10-year Treasury yield
    _IRX = "^IRX"   # 13-week (3-month) T-bill, proxy for 2-year
    _VIX = "^VIX"
    _VX1 = "VX=F"   # VIX front-month futures (approximation via cboe products)
    _DXY = "DX-Y.NYB"  # US Dollar Index

    def __init__(self):
        self._cache: Optional[tuple] = None  # (timestamp, MacroContext)
        self._lock = asyncio.Lock()

    async def get_context(self) -> MacroContext:
        """Return macro context, using cache when fresh."""
        async with self._lock:
            if self._cache and (time.monotonic() - self._cache[0]) < _CACHE_TTL:
                return self._cache[1]
            ctx = await self._fetch()
            self._cache = (time.monotonic(), ctx)
            return ctx

    async def _fetch(self) -> MacroContext:
        """Fetch all macro indicators concurrently."""
        try:
            yc_task = asyncio.create_task(self._yield_curve())
            vix_task = asyncio.create_task(self._vix_structure())
            dxy_task = asyncio.create_task(self._dxy_momentum())

            yc_slope, vix_spot = await yc_task
            vx_ratio = await vix_task
            dxy_roc = await dxy_task

            return _build_context(yc_slope, vx_ratio, dxy_roc, vix_spot)
        except Exception as e:
            logger.warning("MacroIndicators fetch failed (%s), using neutral context", e)
            return _NEUTRAL_MACRO

    # ── Individual indicators ─────────────────────────────────────────────

    async def _yield_curve(self) -> tuple[float, float]:
        """Return (10Y-2Y slope in pct, VIX spot)."""
        try:
            import yfinance as yf
            import pandas as pd

            def _fetch_sync():
                tnx = yf.download(self._TNX, period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
                irx = yf.download(self._IRX, period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
                vix = yf.download(self._VIX, period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
                return tnx, irx, vix

            tnx, irx, vix_data = await asyncio.to_thread(_fetch_sync)

            tnx_val = _last_close(tnx)
            irx_val = _last_close(irx)
            vix_val = _last_close(vix_data)

            if tnx_val is None or irx_val is None:
                return 1.0, float(vix_val or 15.0)

            # IRX is annualized discount rate; approximate: 10Y - 3M as a proxy
            slope = tnx_val - irx_val
            return round(slope, 4), round(float(vix_val or 15.0), 2)

        except Exception as e:
            logger.debug("Yield curve fetch error: %s", e)
            return 1.0, 15.0

    async def _vix_structure(self) -> float:
        """
        Return VX1/VX2 ratio.
        Uses VX=F (front month) vs VXX ETF as a cheap proxy when futures data unavailable.
        Ratio > 1.05 = backwardation (stress). < 0.95 = contango (normal).
        """
        try:
            import yfinance as yf

            def _fetch_sync():
                # Try VIX front-month futures
                vx1 = yf.download("VX=F", period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
                # VXX ETF holds short-dated VIX futures; rolls forward → its own roll cost
                # but useful as a second-month proxy
                vxx = yf.download("VXX", period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
                vix = yf.download("^VIX", period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
                return vx1, vxx, vix

            vx1_data, vxx_data, vix_data = await asyncio.to_thread(_fetch_sync)

            vix_spot = _last_close(vix_data)
            vx1_val = _last_close(vx1_data)

            if vix_spot and vx1_val and vx1_val > 0:
                # VIX spot vs VX front-month: ratio > 1 = spot > futures = backwardation
                ratio = float(vix_spot) / float(vx1_val)
                return round(ratio, 4)

            # Fallback: VIX > 25 usually coincides with backwardation
            if vix_spot and float(vix_spot) > 25:
                return 1.05  # conservative: treat as mild backwardation
            return 0.95  # contango default

        except Exception as e:
            logger.debug("VIX structure fetch error: %s", e)
            return 0.95  # neutral / contango default

    async def _dxy_momentum(self) -> float:
        """Return DXY 20-day rate-of-change (%)."""
        try:
            import yfinance as yf

            def _fetch_sync():
                return yf.download("DX-Y.NYB", period="40d", interval="1d",
                                   progress=False, auto_adjust=True)

            dxy = await asyncio.to_thread(_fetch_sync)
            if dxy is None or len(dxy) < 22:
                return 0.0

            closes = dxy["Close"].dropna().values
            if len(closes) < 22:
                return 0.0

            roc = (closes[-1] / closes[-21] - 1) * 100  # 20-day ROC in %
            return round(float(roc), 4)

        except Exception as e:
            logger.debug("DXY momentum fetch error: %s", e)
            return 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _last_close(df) -> Optional[float]:
    """Extract the last valid close from a yfinance DataFrame."""
    try:
        if df is None or df.empty:
            return None
        closes = df["Close"].dropna()
        if closes.empty:
            return None
        val = closes.iloc[-1]
        # yfinance MultiIndex may return a Series for a single ticker
        if hasattr(val, 'iloc'):
            val = val.iloc[0]
        return float(val)
    except Exception:
        return None


def _build_context(
    yc_slope: float,
    vx_ratio: float,
    dxy_roc: float,
    vix_spot: float,
) -> MacroContext:
    """Convert raw indicator values into a MacroContext."""
    inverted = yc_slope < 0.0
    backwardation = vx_ratio > 1.03
    dollar_risk_off = dxy_roc > 1.0  # DXY rose >1% over 20 days

    # Risk appetite score: each component ∈ [-1, 1]
    # Yield curve: maps [-2, 3] spread → [-1, 1]
    yc_score = max(-1.0, min(1.0, yc_slope / 1.5))
    # VIX structure: 0.80 (deep contango) → +1, 1.20 (deep backwardation) → -1
    vx_score = max(-1.0, min(1.0, (1.0 - vx_ratio) / 0.20))
    # DXY: -5% → +0.5 (weak dollar = risk-on), +5% → -0.5 (strong dollar = risk-off)
    dxy_score = max(-1.0, min(1.0, -dxy_roc / 5.0))
    # VIX level: <15 → +0.3, >35 → -0.6
    vix_score = max(-1.0, min(1.0, -(vix_spot - 20) / 25.0))

    # Weighted composite
    risk_appetite = (
        0.35 * yc_score
        + 0.30 * vx_score
        + 0.20 * dxy_score
        + 0.15 * vix_score
    )
    risk_appetite = round(max(-1.0, min(1.0, risk_appetite)), 4)

    # Regime label
    if risk_appetite >= 0.30:
        regime = "risk_on"
    elif risk_appetite >= -0.10:
        regime = "neutral"
    elif risk_appetite >= -0.50:
        regime = "risk_off"
    else:
        regime = "crisis"

    ctx = MacroContext(
        yield_curve_slope=round(yc_slope, 4),
        vix_futures_ratio=round(vx_ratio, 4),
        dxy_momentum_20d=round(dxy_roc, 4),
        vix_spot=round(vix_spot, 2),
        yield_curve_inverted=inverted,
        vix_backwardation=backwardation,
        dollar_risk_off=dollar_risk_off,
        risk_appetite=risk_appetite,
        regime_signal=regime,
    )
    logger.info(
        "📊 MacroIndicators: yield_curve=%.2f%% vix_struct=%.2f dxy_roc=%.2f%% "
        "VIX=%.1f → risk_appetite=%.2f [%s]",
        yc_slope, vx_ratio, dxy_roc, vix_spot, risk_appetite, regime,
    )
    return ctx
