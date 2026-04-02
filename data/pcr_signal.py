"""
pcr_signal.py — Put/Call Ratio (PCR) contrarian sentiment signal.

The CBOE Equity Put/Call Ratio measures how many put options are being bought
relative to call options. When retail traders are extremely bearish (PCR > 1.1),
markets tend to be near a bottom — the crowd is wrong at extremes.
When PCR < 0.5 (extreme complacency / call buying), markets are often near a top.

PCR is a MARKET-WIDE signal (not per-symbol). It acts as a macro filter:
  • PCR > 1.10 → contrarian bullish (extreme bearishness = buy signal)
  • PCR 0.90–1.10 → neutral
  • PCR < 0.50 → contrarian bearish (extreme complacency = sell/cautious)
  • PCR 0.50–0.90 → mildly bearish → slight confidence penalty for longs

Data source: CBOE free endpoint (no API key required)
  Primary:  https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_PCE.json
  Fallback: https://cdn.cboe.com/api/global/us_indices/daily_prices/PCE_History.json

This is equity-only (CBOE options are on US stocks/ETFs). Don't apply to crypto.

Caching: 1-hour TTL (PCR is published once per day, so hourly is fine).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# CBOE PCR API endpoints (no key needed)
_CBOE_PCR_URL_1 = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_PCE.json"
_CBOE_PCR_URL_2 = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCE_History.json"

# Thresholds (Hulbert, Sentimentrader research)
_PCR_EXTREME_BEARISH = 1.10   # crowd too bearish → contrarian bull
_PCR_ELEVATED = 0.90          # elevated concern
_PCR_NEUTRAL_LOW = 0.70       # below neutral (slight complacency)
_PCR_EXTREME_BULLISH = 0.50   # crowd too bullish → contrarian bear


@dataclass
class PCRContext:
    pcr: float             # raw put/call ratio (e.g. 0.82)
    signal: float          # [-1, 1]: +1 = contrarian bullish, -1 = contrarian bearish
    confidence: float      # [0, 1]: how extreme is the reading
    direction: str         # "extreme_bearish" | "elevated" | "neutral" | "complacent" | "extreme_bullish"
    date: str              # YYYY-MM-DD of the data point
    source: str            # "cboe" | "cached" | "fallback"


def _neutral_pcr() -> PCRContext:
    return PCRContext(
        pcr=0.80,
        signal=0.0,
        confidence=0.0,
        direction="neutral",
        date="",
        source="neutral",
    )


class PCRSignal:
    """
    Fetches CBOE Equity Put/Call Ratio and returns a contrarian market signal.

    Usage:
        pcr = PCRSignal()
        ctx = await pcr.get_signal()          # async version (httpx)
        ctx = pcr.get_signal_sync()           # sync version (requests fallback)
    """

    def __init__(self, cache_ttl_sec: int = 3600) -> None:
        self._cache_ttl = cache_ttl_sec
        self._cache: Optional[Tuple[float, PCRContext]] = None   # (timestamp, ctx)

    # ──────────────────────────────────────────────────────────────────────────
    # Async API (preferred — execution loop is async)
    # ──────────────────────────────────────────────────────────────────────────

    async def get_signal(self) -> PCRContext:
        """Return PCR signal (async). Never raises."""
        # Check cache
        if self._cache and (time.time() - self._cache[0]) < self._cache_ttl:
            ctx = self._cache[1]
            ctx.source = "cached"
            return ctx

        ctx = await self._fetch_async()
        if ctx is None:
            ctx = _neutral_pcr()
        self._cache = (time.time(), ctx)
        return ctx

    async def _fetch_async(self) -> Optional[PCRContext]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
                for url in (_CBOE_PCR_URL_1, _CBOE_PCR_URL_2):
                    try:
                        resp = await client.get(url, follow_redirects=True)
                        if resp.status_code == 200:
                            ctx = self._parse_response(resp.json(), url)
                            if ctx:
                                return ctx
                    except Exception:
                        continue
        except ImportError:
            pass
        # Fall back to sync
        return self._fetch_sync()

    # ──────────────────────────────────────────────────────────────────────────
    # Sync API (fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def get_signal_sync(self) -> PCRContext:
        """Synchronous version for use in non-async contexts."""
        if self._cache and (time.time() - self._cache[0]) < self._cache_ttl:
            return self._cache[1]
        ctx = self._fetch_sync() or _neutral_pcr()
        self._cache = (time.time(), ctx)
        return ctx

    def _fetch_sync(self) -> Optional[PCRContext]:
        # Try CBOE API first (may be blocked in some environments)
        try:
            import urllib.request
            import json as _json
            for url in (_CBOE_PCR_URL_1, _CBOE_PCR_URL_2):
                try:
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 ApexTrading/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        data = _json.loads(resp.read().decode())
                    ctx = self._parse_response(data, url)
                    if ctx:
                        return ctx
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback: VIX / VIX3M ratio as fear-structure proxy
        # VIX > VIX3M (ratio > 1) = near-term fear elevated → contrarian bull signal
        # VIX < VIX3M (ratio < 0.88) = deep contango = complacency → caution
        return self._fetch_vix_ratio()

    def _fetch_vix_ratio(self) -> Optional[PCRContext]:
        """
        Compute VIX/VIX3M ratio as a PCR proxy.
        VIX3M (^VIX3M) measures 3-month implied vol; VIX is 1-month.
        When VIX > VIX3M, short-term fear dominates — same signal as elevated PCR.
        """
        try:
            import yfinance as yf
            import datetime as _dt

            vix_df = yf.download("^VIX", period="3d", progress=False, auto_adjust=True)
            vxv_df = yf.download("^VIX3M", period="3d", progress=False, auto_adjust=True)

            if vix_df.empty or vxv_df.empty:
                return None

            # Extract latest scalar close
            def _last_close(df):
                col = "Close"
                if isinstance(df.columns, object) and col in df.columns:
                    s = df[col].dropna()
                    return float(s.iloc[-1]) if not s.empty else None
                return None

            vix_val = _last_close(vix_df)
            vxv_val = _last_close(vxv_df)

            if not vix_val or not vxv_val or vxv_val <= 0:
                return None

            ratio = vix_val / vxv_val
            date_str = str(_dt.date.today())

            # Map ratio to PCR-equivalent: ratio > 1.05 ≈ PCR > 1.10 (fear)
            # ratio < 0.88 ≈ PCR < 0.55 (complacency)
            pcr_equivalent = ratio * 0.80  # scale: ratio 1.0 → pcr 0.80 (neutral)
            logger.debug(
                "PCR proxy via VIX/VIX3M: VIX=%.1f VIX3M=%.1f ratio=%.3f → pcr_equiv=%.2f",
                vix_val, vxv_val, ratio, pcr_equivalent,
            )
            return self._build_context(pcr_equivalent, date_str, "vix_ratio")

        except Exception as e:
            logger.debug("PCRSignal VIX ratio fallback error: %s", e)
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_response(self, data: dict, url: str) -> Optional[PCRContext]:
        """
        Parse CBOE JSON response into a PCRContext.

        CBOE API format varies:
          _PCE endpoint: {"data": [{"date": "2026-03-19", "close": 0.82}, ...]}
          PCE_History endpoint: {"data": {"date": [...], "close": [...]}}
        """
        try:
            # Format 1: list of {date, close} dicts
            raw = data.get("data", [])
            if isinstance(raw, list) and len(raw) > 0:
                # Most recent valid entry
                for entry in reversed(raw[-20:]):  # last 20 entries
                    pcr = entry.get("close") or entry.get("pcr") or entry.get("value")
                    date = entry.get("date", "")
                    if pcr and 0.1 < float(pcr) < 5.0:
                        src = f"cboe:{url.split('/')[-1].replace('.json', '')}"
                        return self._build_context(float(pcr), str(date), src)

            # Format 2: {"date": [...], "close": [...]}
            if isinstance(raw, dict):
                dates = raw.get("date", [])
                closes = raw.get("close", [])
                if closes:
                    pcr = float(closes[-1])
                    date = str(dates[-1]) if dates else ""
                    if 0.1 < pcr < 5.0:
                        src = f"cboe:{url.split('/')[-1].replace('.json', '')}"
                        return self._build_context(pcr, date, src)

        except Exception as e:
            logger.debug("PCRSignal parse error: %s", e)
        return None

    def _build_context(self, pcr: float, date: str, source: str) -> PCRContext:
        """Convert raw PCR value into a directional signal."""
        # Signal and direction
        if pcr > _PCR_EXTREME_BEARISH:
            # Extreme bearishness → contrarian BUY
            signal = min(1.0, (pcr - _PCR_EXTREME_BEARISH) / 0.40)
            confidence = min(1.0, (pcr - _PCR_EXTREME_BEARISH) / 0.30)
            direction = "extreme_bearish"
        elif pcr > _PCR_ELEVATED:
            # Elevated concern → mild bullish lean
            signal = (pcr - _PCR_ELEVATED) / (_PCR_EXTREME_BEARISH - _PCR_ELEVATED) * 0.35
            confidence = signal * 0.50
            direction = "elevated"
        elif pcr > _PCR_NEUTRAL_LOW:
            # Normal range → neutral
            signal = 0.0
            confidence = 0.0
            direction = "neutral"
        elif pcr > _PCR_EXTREME_BULLISH:
            # Mild complacency → slight caution
            signal = -((_PCR_NEUTRAL_LOW - pcr) / (_PCR_NEUTRAL_LOW - _PCR_EXTREME_BULLISH)) * 0.30
            confidence = abs(signal) * 0.50
            direction = "complacent"
        else:
            # Extreme complacency → contrarian SELL signal
            signal = -min(1.0, (_PCR_EXTREME_BULLISH - pcr) / 0.20)
            confidence = min(1.0, (_PCR_EXTREME_BULLISH - pcr) / 0.15)
            direction = "extreme_bullish"

        logger.debug(
            "PCR %.2f [%s]: signal=%+.3f conf=%.3f (%s)",
            pcr, date, signal, confidence, direction,
        )
        return PCRContext(
            pcr=pcr,
            signal=float(signal),
            confidence=float(confidence),
            direction=direction,
            date=date,
            source=source,
        )
