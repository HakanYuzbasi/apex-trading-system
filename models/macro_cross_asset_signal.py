"""
models/macro_cross_asset_signal.py — Macro Cross-Asset Signal Generator

Synthesises three independent macro regime signals into a single [-1, +1] scalar:

  1. VIX Velocity  — rate-of-change of CBOE Volatility Index over N days.
                     Rising VIX → risk-off → negative signal.
  2. Yield Curve   — 10Y - 2Y Treasury spread (TNX - TWO).
                     Steepening curve → risk-on → positive signal.
                     Inverted spread → recession signal → negative.
  3. DXY Momentum — Rolling momentum of the US Dollar Index.
                     USD strengthening → global risk-off → negative signal
                     (for equities; inverted for EM assets if needed).

Each sub-signal is independently scaled to [-1, +1] via np.tanh().
Final signal = weighted average with configurable weights, then tanh-clipped.

Usage:
    gen = MacroCrossAssetSignal()
    signal = gen.get_signal()          # overall macro signal
    report = gen.get_report()          # full sub-signal breakdown

Config keys:
    MACRO_ENABLED             = True
    MACRO_VIX_WEIGHT          = 0.40
    MACRO_YIELD_WEIGHT        = 0.35
    MACRO_DXY_WEIGHT          = 0.25
    MACRO_VIX_LOOKBACK        = 10     # bars for VIX velocity
    MACRO_YIELD_LOOKBACK      = 20     # bars for yield curve smoothing
    MACRO_DXY_LOOKBACK        = 20     # bars for DXY momentum
    MACRO_CACHE_TTL_SECONDS   = 300    # 5-min cache (expensive yfinance pulls)
    MACRO_BLEND_WEIGHT        = 0.08   # blend weight in GodLevel signal stack
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "MACRO_ENABLED":           True,
    "MACRO_VIX_WEIGHT":        0.40,
    "MACRO_YIELD_WEIGHT":      0.35,
    "MACRO_DXY_WEIGHT":        0.25,
    "MACRO_VIX_LOOKBACK":      10,
    "MACRO_YIELD_LOOKBACK":    20,
    "MACRO_DXY_LOOKBACK":      20,
    "MACRO_CACHE_TTL_SECONDS": 300,
    "MACRO_BLEND_WEIGHT":      0.08,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Sub-signal helpers ─────────────────────────────────────────────────────────

def _vix_velocity_signal(vix_series: list, lookback: int) -> float:
    """
    Negative VIX velocity (falling VIX) → positive risk-on signal.
    Rising VIX → negative risk-off signal.

    Output: tanh(-velocity × scale) ∈ [-1, +1]
    """
    if len(vix_series) < lookback + 1:
        return 0.0
    recent = vix_series[-lookback:]
    start = float(recent[0])
    end = float(recent[-1])
    if start < 0.1:
        return 0.0
    pct_change = (end - start) / start
    # Invert: rising VIX = negative signal
    return float(np.tanh(-pct_change * 10.0))


def _yield_curve_signal(spread_series: list, lookback: int) -> float:
    """
    10Y-2Y spread: positive (steep) → risk-on, negative (inverted) → risk-off.
    Output: tanh(spread × scale) ∈ [-1, +1]
    """
    if not spread_series:
        return 0.0
    recent = spread_series[-lookback:] if len(spread_series) >= lookback else spread_series
    avg_spread = float(np.mean(recent))
    # Spread in %, scale factor: 1% spread → tanh(1) ≈ 0.76
    return float(np.tanh(avg_spread * 1.0))


def _dxy_momentum_signal(dxy_series: list, lookback: int) -> float:
    """
    DXY strengthening (USD up) → risk-off for global equities → negative signal.
    Output: tanh(-momentum × scale) ∈ [-1, +1]
    """
    if len(dxy_series) < lookback + 1:
        return 0.0
    start = float(dxy_series[-lookback])
    end = float(dxy_series[-1])
    if start < 0.1:
        return 0.0
    pct_change = (end - start) / start
    # Invert: USD strength = negative for equities
    return float(np.tanh(-pct_change * 20.0))


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_series(ticker: str, period: str = "3mo", field: str = "Close") -> list:
    """
    Fetch a price/index series via yfinance.
    Returns a list of floats (empty list on failure).
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty or field not in df.columns:
            return []
        # yfinance ≥0.2 may return a multi-level column DataFrame rather than a plain
        # Series for a single ticker. squeeze() collapses (ticker, field) → field Series.
        col = df[field]
        if hasattr(col, 'squeeze'):
            col = col.squeeze()
        return [float(x) for x in col.dropna().to_numpy().tolist()]
    except Exception as e:
        logger.debug("MacroCrossAsset: fetch %s failed — %s", ticker, e)
        return []


# ── MacroCrossAssetSignal ──────────────────────────────────────────────────────

@dataclass
class MacroState:
    """Snapshot of macro sub-signals."""
    vix_signal: float = 0.0
    yield_signal: float = 0.0
    dxy_signal: float = 0.0
    composite_signal: float = 0.0
    vix_level: float = 0.0
    yield_spread: float = 0.0
    dxy_level: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "vix_signal": round(self.vix_signal, 4),
            "yield_signal": round(self.yield_signal, 4),
            "dxy_signal": round(self.dxy_signal, 4),
            "composite_signal": round(self.composite_signal, 4),
            "vix_level": round(self.vix_level, 2),
            "yield_spread": round(self.yield_spread, 4),
            "dxy_level": round(self.dxy_level, 2),
            "timestamp": self.timestamp,
            "error": self.error,
        }


class MacroCrossAssetSignal:
    """
    Macro cross-asset signal generator.

    Fetches VIX, 10Y/2Y yields, and DXY from Yahoo Finance (cached 5 min).
    Produces a composite macro signal [-1, +1] for blending into GodLevel.
    """

    def __init__(self):
        self._cache_ts: float = 0.0
        self._cached_state: Optional[MacroState] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_signal(self) -> float:
        """Return the composite macro signal (cached)."""
        if not _cfg("MACRO_ENABLED"):
            return 0.0
        state = self._get_state()
        return state.composite_signal

    def get_state(self) -> MacroState:
        """Return the full MacroState (cached)."""
        return self._get_state()

    def get_report(self) -> Dict:
        """Return JSON-serialisable report dict."""
        return self._get_state().to_dict()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_state(self) -> MacroState:
        ttl = float(_cfg("MACRO_CACHE_TTL_SECONDS"))
        now = time.monotonic()
        if self._cached_state is not None and (now - self._cache_ts) < ttl:
            return self._cached_state

        state = self._compute()
        self._cached_state = state
        self._cache_ts = now
        return state

    def _compute(self) -> MacroState:
        try:
            return self._compute_inner()
        except Exception as e:
            logger.warning("MacroCrossAsset: compute failed — %s", e)
            return MacroState(error=str(e))

    def _compute_inner(self) -> MacroState:
        vix_lb = int(_cfg("MACRO_VIX_LOOKBACK"))
        yield_lb = int(_cfg("MACRO_YIELD_LOOKBACK"))
        dxy_lb = int(_cfg("MACRO_DXY_LOOKBACK"))
        w_vix = float(_cfg("MACRO_VIX_WEIGHT"))
        w_yield = float(_cfg("MACRO_YIELD_WEIGHT"))
        w_dxy = float(_cfg("MACRO_DXY_WEIGHT"))

        # ── Fetch data ──────────────────────────────────────────────────────
        vix_data = _fetch_series("^VIX", period="3mo")
        tnx_data = _fetch_series("^TNX", period="3mo")   # 10Y yield
        two_data = _fetch_series("^IRX", period="3mo")   # 13-week T-bill as 2Y proxy
        dxy_data = _fetch_series("DX-Y.NYB", period="3mo")

        # ── Sub-signals ─────────────────────────────────────────────────────
        vix_sig = _vix_velocity_signal(vix_data, vix_lb)
        vix_level = float(vix_data[-1]) if vix_data else 0.0

        # Build spread series from TNX - IRX (convert IRX from annualised % to match TNX units)
        spread_series: list = []
        if tnx_data and two_data:
            min_len = min(len(tnx_data), len(two_data))
            for i in range(min_len):
                spread_series.append(tnx_data[-min_len + i] - two_data[-min_len + i])
        yield_sig = _yield_curve_signal(spread_series, yield_lb)
        yield_spread = float(spread_series[-1]) if spread_series else 0.0

        dxy_sig = _dxy_momentum_signal(dxy_data, dxy_lb)
        dxy_level = float(dxy_data[-1]) if dxy_data else 0.0

        # ── Composite ───────────────────────────────────────────────────────
        w_total = w_vix + w_yield + w_dxy
        if w_total < 1e-10:
            composite = 0.0
        else:
            composite = (w_vix * vix_sig + w_yield * yield_sig + w_dxy * dxy_sig) / w_total
        composite = float(np.clip(composite, -1.0, 1.0))

        logger.debug(
            "MacroCrossAsset: vix=%.3f yield=%.3f dxy=%.3f → composite=%.3f",
            vix_sig, yield_sig, dxy_sig, composite,
        )

        return MacroState(
            vix_signal=vix_sig,
            yield_signal=yield_sig,
            dxy_signal=dxy_sig,
            composite_signal=composite,
            vix_level=vix_level,
            yield_spread=yield_spread,
            dxy_level=dxy_level,
        )

    def invalidate_cache(self) -> None:
        """Force recompute on next get_signal() call."""
        self._cache_ts = 0.0


# ── Module-level singleton ────────────────────────────────────────────────────

_macro_signal: Optional[MacroCrossAssetSignal] = None


def get_macro_signal() -> MacroCrossAssetSignal:
    global _macro_signal
    if _macro_signal is None:
        _macro_signal = MacroCrossAssetSignal()
    return _macro_signal
