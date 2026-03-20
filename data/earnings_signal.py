"""
earnings_signal.py — Post-Earnings Announcement Drift (PEAD) signal.

PEAD is a well-documented market anomaly (Ball & Brown 1968, Bernard & Thomas 1989):
  • Stocks with positive EPS surprise drift UP for 30-60 days after earnings.
  • Stocks with negative EPS surprise drift DOWN for 30-60 days after earnings.

This module fetches earnings data from yfinance, computes the surprise percentage,
then returns a signal strength that decays exponentially over time.

Formula:
  signal = surprise × exp(-ln(2) / halflife × days_since_earnings)
  confidence = min(1.0, abs(surprise) / 0.20) × recency_factor

Only applies to equity symbols (crypto has no EPS). Returns neutral context for
any symbol that doesn't have earnings data (indices, ETFs, crypto, FX).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EarningsContext:
    """Result from PEAD signal computation."""
    symbol: str
    signal: float               # [-1, 1]: positive = PEAD bullish, negative = PEAD bearish
    confidence: float           # [0, 1]: how strong/recent is the earnings signal
    surprise_pct: float         # raw EPS surprise: (actual - estimate) / |estimate|
    days_since_earnings: int    # how many days since the most recent earnings release
    direction: str              # "beat" | "miss" | "neutral" | "no_data"
    ticker_note: str = ""       # human-readable note for logging


def _neutral(symbol: str, note: str = "no_data") -> EarningsContext:
    return EarningsContext(
        symbol=symbol,
        signal=0.0,
        confidence=0.0,
        surprise_pct=0.0,
        days_since_earnings=9999,
        direction="no_data",
        ticker_note=note,
    )


class EarningsSignal:
    """
    Computes PEAD (Post-Earnings Announcement Drift) signal for equity symbols.

    Thread-safe read; uses a simple TTL dict cache — one entry per symbol.

    Usage:
        es = EarningsSignal()
        ctx = es.get_signal("AAPL")
        print(ctx.signal, ctx.confidence, ctx.direction)
    """

    def __init__(
        self,
        cache_ttl_sec: int = 3600,           # 1-hour TTL per symbol
        decay_halflife_days: int = 30,        # signal decays 50% every 30 days
        min_surprise: float = 0.05,           # 5% surprise minimum to generate signal
        max_signal_days: int = 90,            # ignore earnings older than 90 days
    ) -> None:
        self._cache_ttl = cache_ttl_sec
        self._decay_halflife = decay_halflife_days
        self._min_surprise = min_surprise
        self._max_signal_days = max_signal_days
        # Cache: symbol → (fetch_timestamp, EarningsContext)
        self._cache: Dict[str, Tuple[float, EarningsContext]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> EarningsContext:
        """
        Return the PEAD signal for *symbol*. Cached for cache_ttl_sec.
        Never raises — returns neutral context on any error.
        """
        # Quickly reject non-equity symbols
        clean = self._clean_symbol(symbol)
        if not clean:
            return _neutral(symbol, "non_equity")

        # Check cache
        cached = self._cache.get(clean)
        if cached and (time.time() - cached[0]) < self._cache_ttl:
            return cached[1]

        # Fetch fresh
        ctx = self._fetch(clean, symbol)
        self._cache[clean] = (time.time(), ctx)
        return ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _clean_symbol(self, symbol: str) -> str:
        """
        Convert trading symbol to yfinance ticker, or return '' to skip.
        Skips: crypto (BTC/, CRYPTO:), FX (FX:, /USD FX pairs), indices (^).
        """
        s = symbol.strip()
        # Skip crypto
        if (s.startswith("CRYPTO:")
                or s.startswith("BTC")
                or s.startswith("ETH")
                or "/" in s):   # any slash = not a plain equity ticker
            return ""
        # Skip FX
        if s.startswith("FX:"):
            return ""
        # Skip indices
        if s.startswith("^"):
            return ""
        # Remove broker prefixes
        for pfx in ("NASDAQ:", "NYSE:", "EQUITY:"):
            if s.startswith(pfx):
                s = s[len(pfx):]
        return s.upper()

    def _fetch(self, ticker: str, original_symbol: str) -> EarningsContext:
        """Fetch earnings data from yfinance and compute PEAD signal."""
        try:
            import yfinance as yf
            import pandas as pd

            t = yf.Ticker(ticker)
            # earnings_dates DataFrame: index=date, columns=EPS Estimate / Reported EPS
            ed = t.earnings_dates
            if ed is None or ed.empty:
                return _neutral(original_symbol, f"no_earnings_data:{ticker}")

            # Filter to past earnings only (some rows are future scheduled dates)
            now = pd.Timestamp.now(tz="UTC")
            past = ed[ed.index < now].copy()
            if past.empty:
                return _neutral(original_symbol, f"no_past_earnings:{ticker}")

            # Most recent earnings
            past_sorted = past.sort_index(ascending=False)
            most_recent = past_sorted.iloc[0]

            # Column names vary — try both formats
            est_col = None
            act_col = None
            for col in past_sorted.columns:
                cl = col.lower()
                if "estimate" in cl:
                    est_col = col
                if "reported" in cl or "actual" in cl or "eps" in cl and "report" in cl:
                    act_col = col

            if est_col is None or act_col is None:
                return _neutral(original_symbol, f"earnings_cols_missing:{list(past_sorted.columns)}")

            estimate = most_recent[est_col]
            actual = most_recent[act_col]

            if pd.isna(estimate) or pd.isna(actual):
                return _neutral(original_symbol, "earnings_nans")

            estimate = float(estimate)
            actual = float(actual)

            # Avoid division by near-zero
            if abs(estimate) < 0.001:
                return _neutral(original_symbol, "estimate_near_zero")

            surprise_pct = (actual - estimate) / abs(estimate)

            # Days since earnings
            earnings_ts = most_recent.name
            if hasattr(earnings_ts, "tzinfo") and earnings_ts.tzinfo is not None:
                days_since = (now - earnings_ts).days
            else:
                days_since = (pd.Timestamp.now() - pd.Timestamp(earnings_ts)).days

            # Too old — signal has fully decayed
            if days_since > self._max_signal_days:
                return _neutral(original_symbol, f"earnings_too_old:{days_since}d")

            # Surprise too small
            if abs(surprise_pct) < self._min_surprise:
                return EarningsContext(
                    symbol=original_symbol,
                    signal=0.0,
                    confidence=0.0,
                    surprise_pct=surprise_pct,
                    days_since_earnings=days_since,
                    direction="neutral",
                    ticker_note=f"surprise_below_min:{surprise_pct:.2%}",
                )

            # Exponential decay: signal(t) = surprise × e^(-ln2/halflife × t)
            decay = math.exp(-math.log(2) / self._decay_halflife * max(days_since, 0))
            raw_signal = float(surprise_pct) * decay

            # Clamp signal to [-1, 1]
            signal = max(-1.0, min(1.0, raw_signal))

            # Confidence: scales with surprise size and recency
            # abs(surprise) / 0.20 caps at 1.0 when surprise >= 20%
            surprise_conf = min(1.0, abs(surprise_pct) / 0.20)
            recency_conf = decay  # 1.0 on day 0, 0.5 on day 30, 0.25 on day 60
            confidence = float(surprise_conf * recency_conf)

            direction = "beat" if surprise_pct > 0 else "miss"

            logger.debug(
                "EarningsPEAD %s: surprise=%.1f%% days=%d signal=%.3f conf=%.3f [%s]",
                ticker, surprise_pct * 100, days_since, signal, confidence, direction,
            )

            return EarningsContext(
                symbol=original_symbol,
                signal=signal,
                confidence=confidence,
                surprise_pct=surprise_pct,
                days_since_earnings=days_since,
                direction=direction,
                ticker_note=f"{ticker}:{days_since}d",
            )

        except ImportError:
            return _neutral(original_symbol, "yfinance_not_available")
        except Exception as e:
            logger.debug("EarningsSignal._fetch error for %s: %s", ticker, e)
            return _neutral(original_symbol, f"fetch_error:{type(e).__name__}")
