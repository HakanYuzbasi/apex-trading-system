"""
monitoring/liquidation_monitor.py — Crypto Liquidation Cascade Early Warning

Detects early-stage liquidation cascades in crypto markets using proxies
available without a paid data feed:

  1. Funding Rate Extremes — When perpetual futures funding rates spike above
     +0.10% or below -0.10% (hourly), leveraged positions are unstable.
     Extreme positive rate → longs at risk of cascade liquidation (bearish).
     Extreme negative rate → shorts at risk (bullish for spot).

  2. Open Interest Velocity — Rapid OI decline signals forced liquidation events.
     OI drop >5% in 1h = warning; >10% = critical.

  3. Price-OI Divergence — Price rising while OI falling = short squeeze reversal.
     Price falling while OI falling = long liquidation cascade (bearish).

Data sources (via ccxt/yfinance fallbacks):
  - Binance perpetuals funding rates (requires ccxt)
  - OI data from Binance futures API (via ccxt)
  - Falls back to 0.0 signal when data unavailable

Signal range: [-1, +1]
  -1.0 = high liquidation risk (cascading longs) → reduce crypto position
  +1.0 = short squeeze risk or liq-driven capitulation → potential mean-reversion entry
   0.0 = normal conditions

Config keys:
    LIQUIDATION_MONITOR_ENABLED           = True
    LIQUIDATION_MONITOR_FUNDING_EXTREME   = 0.10   # % per 8h
    LIQUIDATION_MONITOR_OI_DROP_WARNING   = 0.05   # 5% OI drop threshold
    LIQUIDATION_MONITOR_OI_DROP_CRITICAL  = 0.10   # 10% OI drop threshold
    LIQUIDATION_MONITOR_CACHE_TTL_SECONDS = 120    # 2-min cache
    LIQUIDATION_MONITOR_SIZING_MULT_FLOOR = 0.50   # min position multiplier during cascade
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "LIQUIDATION_MONITOR_ENABLED":           True,
    "LIQUIDATION_MONITOR_FUNDING_EXTREME":   0.10,
    "LIQUIDATION_MONITOR_OI_DROP_WARNING":   0.05,
    "LIQUIDATION_MONITOR_OI_DROP_CRITICAL":  0.10,
    "LIQUIDATION_MONITOR_CACHE_TTL_SECONDS": 120,
    "LIQUIDATION_MONITOR_SIZING_MULT_FLOOR": 0.50,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Sub-signal helpers ─────────────────────────────────────────────────────────

def funding_rate_signal(funding_rate: float, extreme_threshold: float) -> float:
    """
    Map funding rate to cascade risk signal.

    High positive rate → longs at cascade risk → bearish signal.
    High negative rate → shorts at cascade risk → slightly bullish (capitulation near).

    Args:
        funding_rate: Rate as decimal (e.g. 0.001 = 0.1% per 8h)
        extreme_threshold: Threshold as decimal (default 0.001 = 0.1%)

    Returns float in [-1, 0]: negative = bearish cascade risk.
    """
    if abs(funding_rate) < 1e-8:
        return 0.0

    # Normalise: at threshold → -0.5; at 2× threshold → -1.0
    normalised = funding_rate / extreme_threshold
    if funding_rate > 0:
        # Positive funding = longs overleveraged → cascade bearish
        sig = -min(1.0, normalised)
    else:
        # Negative funding = shorts overleveraged → capitulation reversal possible
        sig = min(0.5, abs(normalised) * 0.5)

    return float(np.clip(sig, -1.0, 1.0))


def oi_velocity_signal(oi_series: List[float], warning_thresh: float, critical_thresh: float) -> float:
    """
    Compute OI velocity signal from a time-series of open interest values.

    Returns:
        0.0  = no significant OI change
        -0.5 = warning-level OI drop (5%)
        -1.0 = critical OI drop (10%) → forced liquidation likely
        +0.3 = OI rising → new money entering (slightly positive)
    """
    if len(oi_series) < 2:
        return 0.0

    start = float(oi_series[0])
    end = float(oi_series[-1])
    if start <= 0:
        return 0.0

    change = (end - start) / start  # negative = OI dropping

    if change < -critical_thresh:
        return -1.0
    elif change < -warning_thresh:
        # Interpolate between -0.5 and -1.0
        frac = (abs(change) - warning_thresh) / (critical_thresh - warning_thresh)
        return float(-0.5 - 0.5 * min(frac, 1.0))
    elif change > warning_thresh:
        # OI growing: new money, slightly positive
        return float(min(0.3, change / critical_thresh * 0.3))

    return 0.0


def price_oi_divergence_signal(
    price_change: float,
    oi_change: float,
) -> float:
    """
    Detect price-OI divergence patterns.

    Price ↓ + OI ↓ = long liquidation cascade → bearish signal
    Price ↑ + OI ↓ = short squeeze → mild positive (reversal)
    Price ↑ + OI ↑ = healthy uptrend
    Price ↓ + OI ↑ = new shorts entering

    Returns float in [-1, +1].
    """
    if abs(price_change) < 0.005 or abs(oi_change) < 0.01:
        return 0.0

    if price_change < 0 and oi_change < 0:
        # Long liquidation: severity based on magnitude
        magnitude = min(abs(price_change) + abs(oi_change), 0.3) / 0.3
        return float(-0.8 * magnitude)
    elif price_change > 0 and oi_change < 0:
        # Short squeeze / capitulation reversal
        return float(0.4 * min(abs(price_change) / 0.05, 1.0))

    return 0.0


# ── LiquidationMonitor ────────────────────────────────────────────────────────

@dataclass
class LiquidationState:
    """Snapshot of liquidation risk indicators."""
    symbol: str
    funding_signal: float = 0.0
    oi_signal: float = 0.0
    divergence_signal: float = 0.0
    composite_signal: float = 0.0
    sizing_multiplier: float = 1.0
    funding_rate: float = 0.0
    oi_change_pct: float = 0.0
    risk_level: str = "normal"  # normal / warning / critical
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "funding_signal": round(self.funding_signal, 4),
            "oi_signal": round(self.oi_signal, 4),
            "divergence_signal": round(self.divergence_signal, 4),
            "composite_signal": round(self.composite_signal, 4),
            "sizing_multiplier": round(self.sizing_multiplier, 4),
            "funding_rate": round(self.funding_rate, 6),
            "oi_change_pct": round(self.oi_change_pct * 100, 2),
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class LiquidationMonitor:
    """
    Crypto liquidation cascade early warning system.

    Uses funding rates, OI velocity, and price-OI divergence to generate
    a defensive signal that reduces position sizing during cascade risk.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[LiquidationState, float]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> float:
        """Return composite liquidation risk signal for symbol."""
        if not _cfg("LIQUIDATION_MONITOR_ENABLED"):
            return 0.0
        return self._get_state(symbol).composite_signal

    def get_sizing_multiplier(self, symbol: str) -> float:
        """Return position sizing multiplier [FLOOR, 1.0]. < 1.0 during cascade risk."""
        if not _cfg("LIQUIDATION_MONITOR_ENABLED"):
            return 1.0
        return self._get_state(symbol).sizing_multiplier

    def get_state(self, symbol: str) -> LiquidationState:
        """Return full liquidation state for a symbol."""
        return self._get_state(symbol)

    def get_report(self, symbol: str) -> Dict:
        """Return JSON-serialisable state."""
        return self._get_state(symbol).to_dict()

    def get_all_report(self) -> Dict:
        """Return cached states for all monitored symbols."""
        return {sym: state.to_dict() for sym, (state, _) in self._cache.items()}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> LiquidationState:
        ttl = float(_cfg("LIQUIDATION_MONITOR_CACHE_TTL_SECONDS"))
        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached is not None and (now - cached[1]) < ttl:
            return cached[0]

        state = self._compute(symbol)
        self._cache[symbol] = (state, now)
        return state

    def _compute(self, symbol: str) -> LiquidationState:
        try:
            return self._compute_inner(symbol)
        except Exception as e:
            logger.debug("LiquidationMonitor: compute failed for %s — %s", symbol, e)
            return LiquidationState(symbol=symbol, error=str(e))

    def _compute_inner(self, symbol: str) -> LiquidationState:
        funding_extreme = float(_cfg("LIQUIDATION_MONITOR_FUNDING_EXTREME")) / 100.0
        oi_warn = float(_cfg("LIQUIDATION_MONITOR_OI_DROP_WARNING"))
        oi_crit = float(_cfg("LIQUIDATION_MONITOR_OI_DROP_CRITICAL"))
        mult_floor = float(_cfg("LIQUIDATION_MONITOR_SIZING_MULT_FLOOR"))

        # ── Fetch data ─────────────────────────────────────────────────────
        funding_rate = self._fetch_funding_rate(symbol)
        oi_series = self._fetch_oi_series(symbol)
        price_change, oi_change = self._fetch_price_oi_change(symbol)

        # ── Sub-signals ────────────────────────────────────────────────────
        fund_sig = funding_rate_signal(funding_rate, funding_extreme)
        oi_sig = oi_velocity_signal(oi_series, oi_warn, oi_crit)
        div_sig = price_oi_divergence_signal(price_change, oi_change)

        oi_change_pct = 0.0
        if len(oi_series) >= 2 and oi_series[0] > 0:
            oi_change_pct = (oi_series[-1] - oi_series[0]) / oi_series[0]

        # Composite: weighted average (funding most important)
        composite = 0.50 * fund_sig + 0.30 * oi_sig + 0.20 * div_sig
        composite = float(np.clip(composite, -1.0, 1.0))

        # Risk level classification
        if composite < -0.60:
            risk_level = "critical"
        elif composite < -0.30:
            risk_level = "warning"
        else:
            risk_level = "normal"

        # Sizing multiplier: 1.0 (normal) → FLOOR (critical cascade)
        if composite >= 0:
            sizing_mult = 1.0
        else:
            # Linear interpolation from 1.0 to mult_floor as composite goes from 0 to -1
            sizing_mult = 1.0 + composite * (1.0 - mult_floor)
        sizing_mult = float(np.clip(sizing_mult, mult_floor, 1.0))

        if risk_level != "normal":
            logger.warning(
                "LiquidationMonitor %s: RISK=%s fund=%.4f oi_chg=%.1f%% composite=%.3f mult=%.2f",
                symbol, risk_level.upper(), funding_rate, oi_change_pct * 100,
                composite, sizing_mult,
            )

        return LiquidationState(
            symbol=symbol,
            funding_signal=fund_sig,
            oi_signal=oi_sig,
            divergence_signal=div_sig,
            composite_signal=composite,
            sizing_multiplier=sizing_mult,
            funding_rate=funding_rate,
            oi_change_pct=oi_change_pct,
            risk_level=risk_level,
        )

    # ── Binance public REST helpers ──────────────────────────────────────────────
    # No API key required; uses free-tier public Futures endpoints.
    # Shared requests.Session for connection reuse across calls.

    @staticmethod
    def _to_binance_symbol(symbol: str) -> str:
        """
        Convert Apex symbol format to Binance perpetual contract format.
        CRYPTO:BTC/USD → BTCUSDT, ETH/USD → ETHUSDT
        """
        clean = (symbol
                 .replace("CRYPTO:", "")
                 .replace("/USDT", "USDT")   # must come before /USD to avoid BTCUSDTT
                 .replace("/USD", "USDT")
                 .replace("/", "")
                 .upper())
        return clean

    def _get_session(self):
        """Lazy-initialise a shared requests.Session (connection pool reuse)."""
        if not hasattr(self, "_requests_session") or self._requests_session is None:
            try:
                import requests
                self._requests_session = requests.Session()
                self._requests_session.headers.update({"User-Agent": "apex-trading/1.0"})
            except ImportError:
                self._requests_session = None
        return self._requests_session

    def _fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch latest perpetual funding rate from Binance public REST API.
        Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1
        Returns rate as decimal (e.g. 0.0001 = 0.01% per 8h).
        Falls back to 0.0 on any error.
        """
        session = self._get_session()
        if session is None:
            return 0.0
        try:
            bin_sym = self._to_binance_symbol(symbol)
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            resp = session.get(url, params={"symbol": bin_sym, "limit": 1}, timeout=4.0)
            if resp.status_code != 200:
                return 0.0
            data = resp.json()
            if isinstance(data, list) and data:
                return float(data[-1].get("fundingRate", 0.0))
        except Exception as e:
            logger.debug("LiquidationMonitor: funding rate fetch failed for %s — %s", symbol, e)
        return 0.0

    def _fetch_oi_series(self, symbol: str) -> List[float]:
        """
        Fetch recent open interest history from Binance public REST API.
        Endpoint: GET https://fapi.binance.com/futures/data/openInterestHist
        Returns list of OI values (oldest first), newest last.
        Falls back to empty list on any error.
        """
        session = self._get_session()
        if session is None:
            return []
        try:
            bin_sym = self._to_binance_symbol(symbol)
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            resp = session.get(
                url,
                params={"symbol": bin_sym, "period": "1h", "limit": 6},
                timeout=4.0,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            if isinstance(data, list):
                return [float(d.get("sumOpenInterest", 0.0)) for d in data]
        except Exception as e:
            logger.debug("LiquidationMonitor: OI series fetch failed for %s — %s", symbol, e)
        return []

    def _fetch_price_oi_change(self, symbol: str) -> Tuple[float, float]:
        """
        Returns (price_change_pct, oi_change_pct) over last ~4 hours.
        Uses Binance klines endpoint for price and OI history for OI change.
        Falls back to (0.0, 0.0).
        """
        session = self._get_session()
        if session is None:
            return 0.0, 0.0
        try:
            bin_sym = self._to_binance_symbol(symbol)
            # 4h price change from Binance klines (4 × 1h candles)
            klines_url = "https://fapi.binance.com/fapi/v1/klines"
            resp = session.get(
                klines_url,
                params={"symbol": bin_sym, "interval": "1h", "limit": 5},
                timeout=4.0,
            )
            price_change = 0.0
            if resp.status_code == 200:
                klines = resp.json()
                if klines and len(klines) >= 4:
                    _open = float(klines[-4][1])   # open price 4h ago
                    _close = float(klines[-1][4])  # current close
                    if _open > 0:
                        price_change = (_close - _open) / _open

            # OI change from OI history
            oi_series = self._fetch_oi_series(symbol)
            oi_change = 0.0
            if len(oi_series) >= 4 and oi_series[-4] > 0:
                oi_change = (oi_series[-1] - oi_series[-4]) / oi_series[-4]

            return price_change, oi_change
        except Exception as e:
            logger.debug("LiquidationMonitor: price/OI change failed for %s — %s", symbol, e)
        return 0.0, 0.0


# ── Module-level singleton ────────────────────────────────────────────────────

_liquidation_monitor: Optional[LiquidationMonitor] = None


def get_liquidation_monitor() -> LiquidationMonitor:
    global _liquidation_monitor
    if _liquidation_monitor is None:
        _liquidation_monitor = LiquidationMonitor()
    return _liquidation_monitor
