"""
data/earnings_catalyst.py — Post-Earnings Announcement Drift (PEAD) Signal
                             + Earnings Revision Trends
                             + Multi-Quarter Surprise Persistence

Three independent earnings-intelligence signals:

  1. PEAD signal      — time-decaying post-earnings drift (original).
  2. Revision signal  — consensus estimate drift over the past 30 days.
                        Analysts upgrading estimates → positive; downgrading → negative.
                        Range: [-1, +1].
  3. Persistence      — consistency of beats/misses across the last N quarters.
                        4-quarter consecutive beat → +1.0; consecutive miss → -1.0.
                        Range: [-1, +1].

  Extended combined signal: 0.50×PEAD + 0.30×revision + 0.20×persistence.

Usage:
    from data.earnings_catalyst import EarningsCatalystSignal
    catalyst = EarningsCatalystSignal()
    signal = catalyst.get_signal("AAPL")           # PEAD only
    extended = catalyst.get_extended_signal("AAPL") # all three combined
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

    def get_revision_signal(self, symbol: str) -> float:
        """
        Earnings estimate revision signal.

        Compares current quarter EPS estimate vs estimate from 30 days ago using
        yfinance analyst price-target trend as a proxy (direct estimate history
        requires paid data). Falls back to 0.0 when unavailable.

        Returns [-1, +1]: positive = upgrades, negative = downgrades.
        """
        clean = self._clean_symbol(symbol)
        if not clean or not self._is_equity(symbol):
            return 0.0

        cache_key = f"rev_{clean}"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached[1]) < self.cache_ttl_sec:
            return cached[0]

        signal = self._compute_revision(clean)
        self._cache[cache_key] = (signal, now)
        return signal

    def get_persistence_signal(self, symbol: str, n_quarters: int = 4) -> float:
        """
        Multi-quarter earnings surprise persistence.

        Scores the consistency of beats/misses across the last n_quarters.
        Returns [-1, +1]: +1 = n consecutive beats, -1 = n consecutive misses.
        """
        clean = self._clean_symbol(symbol)
        if not clean or not self._is_equity(symbol):
            return 0.0

        cache_key = f"persist_{clean}"
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached[1]) < self.cache_ttl_sec:
            return cached[0]

        signal = self._compute_persistence(clean, n_quarters)
        self._cache[cache_key] = (signal, now)
        return signal

    def get_extended_signal(self, symbol: str) -> float:
        """
        Combined earnings intelligence signal.

        Blends:
          0.50 × PEAD (time-decaying post-announcement drift)
          0.30 × Revision trend (consensus estimate upgrades/downgrades)
          0.20 × Persistence (multi-quarter beat/miss consistency)

        Returns [-1, +1].
        """
        pead = self.get_signal(symbol)
        revision = self.get_revision_signal(symbol)
        persistence = self.get_persistence_signal(symbol)

        extended = 0.50 * pead + 0.30 * revision + 0.20 * persistence
        return float(max(-1.0, min(1.0, extended)))

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

    def _compute_revision(self, symbol: str) -> float:
        """
        Use analyst recommendation trends as a proxy for estimate revisions.
        Maps Strong Buy/Buy/Hold/Sell/Strong Sell to +1/-1 direction; computes
        net change between current month and 1-month-ago recommendations.
        Falls back to 0.0 if data unavailable.
        """
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                rec = ticker.recommendations

            if rec is None or rec.empty or len(rec) < 2:
                return 0.0

            def _score_row(row) -> float:
                counts = {}
                for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                    counts[col] = int(row.get(col, 0))
                total = sum(counts.values())
                if total == 0:
                    return 0.0
                # Weighted sentiment: SB=+2, B=+1, H=0, S=-1, SS=-2
                net = (counts.get("strongBuy", 0) * 2
                       + counts.get("buy", 0) * 1
                       + counts.get("hold", 0) * 0
                       + counts.get("sell", 0) * -1
                       + counts.get("strongSell", 0) * -2)
                return net / (total * 2.0)  # normalise to [-1, +1]

            # Use last two available monthly rows
            recent = _score_row(rec.iloc[-1])
            prior = _score_row(rec.iloc[-2])
            drift = recent - prior   # positive = upgrades, negative = downgrades
            signal = float(max(-1.0, min(1.0, drift * 5.0)))  # scale
            logger.debug("EarningsCatalyst revision %s: recent=%.3f prior=%.3f → %.3f",
                         symbol, recent, prior, signal)
            return signal
        except Exception as exc:
            logger.debug("EarningsCatalyst revision fetch failed for %s: %s", symbol, exc)
            return 0.0

    def _compute_persistence(self, symbol: str, n_quarters: int) -> float:
        """
        Multi-quarter surprise persistence scorer.

        Fetches earnings history and scores the last n_quarters of surprise signs.
        """
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                hist = ticker.get_earnings_history()

            if hist is None or hist.empty:
                return 0.0

            surprises = []
            for i in range(len(hist) - 1, -1, -1):
                row = hist.iloc[i]
                actual = row.get("epsActual")
                estimate = row.get("epsEstimate")
                if actual is None or estimate is None or estimate == 0:
                    continue
                surprise = float((actual - estimate) / abs(estimate))
                surprises.append(surprise)
                if len(surprises) >= n_quarters:
                    break

            if not surprises:
                return 0.0

            # Score: fraction of quarters with material positive surprise minus negative
            material_thresh = 0.02  # 2% threshold for materiality
            beats = sum(1 for s in surprises if s > material_thresh)
            misses = sum(1 for s in surprises if s < -material_thresh)
            total = len(surprises)

            net_score = (beats - misses) / total  # in [-1, +1]

            # Boost for consecutive streaks
            streak = 0
            for s in surprises:  # most recent first
                if net_score > 0 and s > material_thresh:
                    streak += 1
                elif net_score < 0 and s < -material_thresh:
                    streak += 1
                else:
                    break

            streak_boost = min(streak / n_quarters, 1.0) * 0.20  # up to +20% boost
            signal = float(max(-1.0, min(1.0, net_score + (streak_boost if net_score > 0 else -streak_boost))))
            logger.debug("EarningsCatalyst persistence %s: beats=%d misses=%d streak=%d → %.3f",
                         symbol, beats, misses, streak, signal)
            return signal
        except Exception as exc:
            logger.debug("EarningsCatalyst persistence fetch failed for %s: %s", symbol, exc)
            return 0.0

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


def get_extended_earnings_signal(symbol: str) -> float:
    """Convenience function: return extended earnings signal (PEAD + revision + persistence)."""
    return get_earnings_catalyst().get_extended_signal(symbol)


def get_earnings_revision_signal(symbol: str) -> float:
    """Convenience function: return analyst revision trend signal."""
    return get_earnings_catalyst().get_revision_signal(symbol)


def get_earnings_persistence_signal(symbol: str) -> float:
    """Convenience function: return multi-quarter persistence signal."""
    return get_earnings_catalyst().get_persistence_signal(symbol)
