"""
data/earnings_catalyst.py — Post-Earnings Announcement Drift (PEAD) Signal

One of the most durable market anomalies: stocks that beat earnings estimates
by >5% drift upward for 20-60 days; misses drift downward. This module
computes a decaying PEAD signal for any equity symbol.

Signal range: [-1.0, 1.0]
  +1.0 = strong beat, fresh (day 0 after report)
   0.0 = no recent earnings / neutral surprise
  -1.0 = strong miss, fresh

Signal decays linearly to 0 over drift_days (default: 20 trading days).

Usage:
    from data.earnings_catalyst import EarningsCatalystSignal
    catalyst = EarningsCatalystSignal()
    signal = catalyst.get_signal("AAPL")   # e.g. 0.72 (recent 10% beat)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# How many calendar days to consider "fresh" drift window
_DEFAULT_DRIFT_DAYS = 20
# Surprise magnitude thresholds
_SURPRISE_STRONG = 0.10   # 10%+ beat/miss → max signal
_SURPRISE_MODERATE = 0.04  # 4-10% beat/miss → partial signal
# Cache TTL: refresh once per day (earnings data is daily)
_CACHE_TTL_SEC = 86_400


@dataclass
class EarningsEvent:
    """Most recent earnings event for a symbol."""
    symbol: str
    report_date: datetime
    actual_eps: float
    estimate_eps: float
    surprise_pct: float      # (actual - estimate) / |estimate|
    days_since: int          # calendar days since report


class EarningsCatalystSignal:
    """
    Post-Earnings Announcement Drift signal generator.

    Computes a time-decaying signal based on the most recent earnings
    surprise. Signals are cached for 24h to avoid repeated yfinance calls.

    Thread-safe for read-only get_signal(). Cache is written lazily.
    """

    def __init__(
        self,
        drift_days: int = _DEFAULT_DRIFT_DAYS,
        surprise_strong: float = _SURPRISE_STRONG,
        surprise_moderate: float = _SURPRISE_MODERATE,
        cache_ttl_sec: float = _CACHE_TTL_SEC,
    ):
        self.drift_days = drift_days
        self.surprise_strong = surprise_strong
        self.surprise_moderate = surprise_moderate
        self.cache_ttl_sec = cache_ttl_sec

        # symbol → (signal_value, computed_at_epoch)
        self._cache: Dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_signal(self, symbol: str) -> float:
        """
        Return PEAD signal in [-1, 1] for symbol.

        Returns 0.0 for crypto, FX, unknown, or symbols with no recent
        material earnings surprise.
        """
        clean = self._clean_symbol(symbol)
        if not clean or not self._is_equity(symbol):
            return 0.0

        now = time.time()
        cached = self._cache.get(clean)
        if cached is not None and (now - cached[1]) < self.cache_ttl_sec:
            return cached[0]

        signal = self._compute(clean)
        self._cache[clean] = (signal, now)
        return signal

    def get_event(self, symbol: str) -> Optional[EarningsEvent]:
        """Return the most recent earnings event data for inspection."""
        clean = self._clean_symbol(symbol)
        if not clean or not self._is_equity(symbol):
            return None
        return self._fetch_event(clean)

    def prefetch(self, symbols: list[str]) -> None:
        """Warm cache for a list of symbols (call at session start)."""
        for sym in symbols:
            try:
                self.get_signal(sym)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute(self, symbol: str) -> float:
        event = self._fetch_event(symbol)
        if event is None:
            return 0.0

        # Outside the drift window → neutral
        if event.days_since < 0 or event.days_since > self.drift_days:
            return 0.0

        # Linear decay: 1.0 at day 0, 0.0 at drift_days
        decay = 1.0 - (event.days_since / self.drift_days)

        surprise = event.surprise_pct
        if abs(surprise) < self.surprise_moderate:
            return 0.0   # within ±4% → not material

        # Clamp raw signal to ±1 and apply decay
        if abs(surprise) >= self.surprise_strong:
            raw = min(1.0, abs(surprise) / self.surprise_strong)
        else:
            raw = 0.6   # moderate tier

        signal = raw * decay * (1.0 if surprise > 0 else -1.0)
        logger.debug(
            "EarningsCatalyst %s: surprise=%.1f%% days_since=%d decay=%.2f → signal=%.3f",
            symbol, surprise * 100, event.days_since, decay, signal,
        )
        return round(signal, 4)

    def _fetch_event(self, symbol: str) -> Optional[EarningsEvent]:
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                hist = ticker.get_earnings_history()

            if hist is None or hist.empty:
                return None

            # Most recent row (sorted by quarter ascending — take the latest)
            row = hist.iloc[-1]
            actual = row.get("epsActual")
            estimate = row.get("epsEstimate")
            if actual is None or estimate is None or estimate == 0:
                return None

            surprise = float((actual - estimate) / abs(estimate))
            # Quarter index is Timestamp representing report date
            report_date = hist.index[-1]
            if hasattr(report_date, "to_pydatetime"):
                report_date = report_date.to_pydatetime()
            days_since = (datetime.now() - report_date).days

            return EarningsEvent(
                symbol=symbol,
                report_date=report_date,
                actual_eps=float(actual),
                estimate_eps=float(estimate),
                surprise_pct=surprise,
                days_since=days_since,
            )
        except Exception as exc:
            logger.debug("EarningsCatalyst: fetch failed for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        return symbol.split(":")[-1].split("/")[0].upper().strip()

    @staticmethod
    def _is_equity(symbol: str) -> bool:
        """Exclude crypto, FX and index symbols."""
        s = symbol.upper()
        if s.startswith("CRYPTO:") or s.startswith("FX:"):
            return False
        if "/" in s or "BTC" in s or "ETH" in s or "USD" in s.split(":")[-1]:
            return False
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[EarningsCatalystSignal] = None


def get_earnings_catalyst() -> EarningsCatalystSignal:
    global _instance
    if _instance is None:
        _instance = EarningsCatalystSignal()
    return _instance


def get_earnings_signal(symbol: str) -> float:
    """Convenience function: return PEAD signal for symbol."""
    return get_earnings_catalyst().get_signal(symbol)
