"""
signals/funding_rate_signal.py — Crypto Funding Rate Signal
============================================================
Reads Binance perpetuals funding rates (public API, no key).

When longs are overcrowded (funding > 0.10%/8h):
  → signal is BEARISH (fade the crowd)
When shorts are overcrowded (funding < -0.05%/8h):
  → signal is BULLISH (fade the crowd)

This is pure POSITIONING DATA — completely independent from price models.
Only applies to crypto symbols.

Usage:
    frs = FundingRateSignal()
    ctx = await frs.get_signal("BTC/USD")
    print(ctx.signal, ctx.direction)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via ApexConfig)
# ---------------------------------------------------------------------------
_FUNDING_EXTREME_THRESHOLD = 0.0010   # 0.10%/8h — very crowded
_FUNDING_SIGNAL_SCALE = 0.0015        # tanh saturation point
_CACHE_TTL = 900.0                     # 15 minutes
_BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Symbols Binance perpetuals supports (USDT margined)
_SUPPORTED_BASES = {
    "BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "LTC",
    "DOT", "ADA", "XRP", "MATIC", "DOGE", "UNI", "ATOM",
    "AAVE", "ALGO", "XLM", "BCH", "ETC",
}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class FundingRateContext:
    symbol: str
    current_rate: float    # latest rate, e.g. 0.0001 = 0.01%/8h
    rate_ma3: float        # 3-period moving average
    signal: float          # [-1, 1]: negative = bearish (fade longs)
    confidence: float      # [0, 1]
    is_extreme: bool       # abs(rate_ma3) > threshold
    direction: str         # "long_crowded" | "short_crowded" | "neutral"


def _neutral(symbol: str) -> FundingRateContext:
    return FundingRateContext(
        symbol=symbol,
        current_rate=0.0,
        rate_ma3=0.0,
        signal=0.0,
        confidence=0.0,
        is_extreme=False,
        direction="neutral",
    )


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------
def _to_binance_perp(symbol: str) -> Optional[str]:
    """
    Convert internal symbol to Binance USDT-margined perpetual ticker.
    "BTC/USD" | "CRYPTO:BTC/USD" | "BTC/USDT"  →  "BTCUSDT"
    Returns None if the base is not in the supported set.
    """
    bare = symbol.replace("CRYPTO:", "").replace("FX:", "")
    if "/" not in bare:
        return None
    base = bare.split("/")[0].upper()
    if base not in _SUPPORTED_BASES:
        return None
    return f"{base}USDT"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FundingRateSignal:
    """
    Async funding-rate signal source.  One instance is shared for the whole session.
    Thread-safe via per-symbol asyncio.Lock.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, FundingRateContext]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._client = None   # httpx.AsyncClient, lazy-initialised

        try:
            from config import ApexConfig
            self._extreme_threshold = float(
                getattr(ApexConfig, "FUNDING_EXTREME_THRESHOLD", _FUNDING_EXTREME_THRESHOLD)
            )
            self._signal_scale = float(
                getattr(ApexConfig, "FUNDING_SIGNAL_SCALE", _FUNDING_SIGNAL_SCALE)
            )
            self._cache_ttl = float(
                getattr(ApexConfig, "FUNDING_RATE_CACHE_TTL", _CACHE_TTL)
            )
        except Exception:
            self._extreme_threshold = _FUNDING_EXTREME_THRESHOLD
            self._signal_scale = _FUNDING_SIGNAL_SCALE
            self._cache_ttl = _CACHE_TTL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_signal(self, symbol: str) -> FundingRateContext:
        """Return funding-rate context for `symbol`, using cache when fresh."""
        binance_sym = _to_binance_perp(symbol)
        if binance_sym is None:
            return _neutral(symbol)

        if symbol not in self._locks:
            self._locks[symbol] = asyncio.Lock()

        async with self._locks[symbol]:
            cached = self._cache.get(symbol)
            if cached and (time.monotonic() - cached[0]) < self._cache_ttl:
                return cached[1]

            ctx = await self._fetch(symbol, binance_sym)
            self._cache[symbol] = (time.monotonic(), ctx)
            return ctx

    async def close(self) -> None:
        """Close the shared HTTP client (call on engine shutdown)."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal fetch
    # ------------------------------------------------------------------

    async def _get_client(self):
        """Lazy-init shared httpx client."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx not installed — pip install httpx")

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(6.0, connect=3.0),
                headers={"User-Agent": "ApexTrading/1.0"},
            )
        return self._client

    async def _fetch(self, symbol: str, binance_sym: str) -> FundingRateContext:
        """Fetch from Binance; return neutral on any failure."""
        try:
            client = await self._get_client()

            # Fetch last 3 historical rates + current mark premium
            hist_url = f"{_BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
            prem_url = f"{_BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"

            hist_task = client.get(hist_url, params={"symbol": binance_sym, "limit": 3})
            prem_task = client.get(prem_url, params={"symbol": binance_sym})

            hist_resp, prem_resp = await asyncio.gather(hist_task, prem_task, return_exceptions=True)

            rates: List[float] = []

            # Historical rates
            if not isinstance(hist_resp, Exception) and hist_resp.status_code == 200:
                for item in hist_resp.json():
                    try:
                        rates.append(float(item.get("fundingRate", 0)))
                    except (TypeError, ValueError):
                        pass

            # Current rate from premium index
            current_rate = 0.0
            if not isinstance(prem_resp, Exception) and prem_resp.status_code == 200:
                data = prem_resp.json()
                try:
                    current_rate = float(data.get("lastFundingRate", data.get("fundingRate", 0)))
                except (TypeError, ValueError):
                    pass

            if not rates and current_rate == 0.0:
                return _neutral(symbol)

            # Use current_rate as most recent; prepend to historical list
            all_rates = [current_rate] + rates if current_rate != 0.0 else rates
            rate_ma3 = float(np.mean(all_rates[:3])) if len(all_rates) >= 1 else 0.0

            return self._build_context(symbol, current_rate, rate_ma3)

        except Exception as exc:
            logger.debug("FundingRateSignal fetch failed (%s): %s", symbol, exc)
            return _neutral(symbol)

    def _build_context(
        self, symbol: str, current_rate: float, rate_ma3: float
    ) -> FundingRateContext:
        # Fade the crowd: positive funding → longs overcrowded → bearish signal
        signal = float(-np.tanh(rate_ma3 / max(self._signal_scale, 1e-8)))
        confidence = float(min(abs(rate_ma3) / max(self._extreme_threshold, 1e-8), 1.0))

        if rate_ma3 > self._extreme_threshold * 0.5:
            direction = "long_crowded"
        elif rate_ma3 < -self._extreme_threshold * 0.3:
            direction = "short_crowded"
        else:
            direction = "neutral"

        is_extreme = abs(rate_ma3) >= self._extreme_threshold

        return FundingRateContext(
            symbol=symbol,
            current_rate=current_rate,
            rate_ma3=rate_ma3,
            signal=float(np.clip(signal, -1.0, 1.0)),
            confidence=confidence,
            is_extreme=is_extreme,
            direction=direction,
        )
