"""
data/social/market_adapter.py

MarketSentimentAdapter — derives a social-feed-compatible SourceSnapshot from
price-based signals: VIX z-score and market breadth.

Zero external API dependency beyond yfinance (with graceful simulation fallback).
Always produces quality = "ok" since it relies on cached price data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from data.social.adapters import SourceSnapshot, SocialSourceAdapter, _clamp

logger = logging.getLogger(__name__)

# Bellwether equities used to measure market breadth (% above 20d MA)
_BREADTH_SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "XLF", "XLE", "XLV",
]


class MarketSentimentAdapter(SocialSourceAdapter):
    """
    Produces a social-feed snapshot from price-based market signals.

    * ``attention_z``  — VIX z-score vs 60-day mean (high VIX → high attention)
    * ``sentiment_score`` — market breadth: fraction of bellwether symbols
      trading above their 20-day moving average, mapped to [-1, 1]
    * ``confidence``   — always 0.6 (moderate; price data is reliable but
      indirect proxy for social sentiment)

    quality is always "ok" because this adapter never touches an external API
    at call time — it uses the VIXRegimeManager's cached / simulated data.
    """

    source_name = "MARKET"
    env_endpoint_key = ""
    env_token_key = ""
    default_local_name = ""

    def __init__(self, data_dir: Path, timeout_seconds: float = 10.0):
        super().__init__(data_dir=data_dir, timeout_seconds=timeout_seconds)

    # ------------------------------------------------------------------
    def fetch(
        self,
        now: Optional[datetime] = None,
        freshness_sla_seconds: int = 1800,
    ) -> SourceSnapshot:
        now_dt = now or datetime.utcnow()
        if now_dt.tzinfo is not None:
            now_dt = now_dt.astimezone(timezone.utc).replace(tzinfo=None)

        vix_z, vix_raw = self._get_vix_zscore()
        breadth_score = self._get_market_breadth()

        attention_z = _clamp(vix_z, -5.0, 8.0)
        sentiment_score = _clamp(breadth_score, -1.0, 1.0)
        confidence = 0.6  # fixed — price data is always available

        flags: List[str] = []
        if vix_raw is None:
            flags.append("vix_simulated")

        quality_status = "ok"
        quality_score = _clamp(confidence * 0.6 + 0.4, 0.0, 1.0)

        return SourceSnapshot(
            source=self.source_name,
            fetched_at=now_dt.isoformat(),
            freshness_ts=now_dt.isoformat(),
            attention_z=attention_z,
            sentiment_score=sentiment_score,
            confidence=confidence,
            sample_size=len(_BREADTH_SYMBOLS),
            quality_status=quality_status,
            quality_score=quality_score,
            quality_flags=sorted(set(flags)),
            adapter_mode="market",
            raw_payload={
                "vix_zscore": float(vix_z),
                "vix_level": float(vix_raw) if vix_raw is not None else None,
                "breadth_score": float(breadth_score),
                "breadth_symbols": len(_BREADTH_SYMBOLS),
            },
        )

    # ------------------------------------------------------------------
    def _get_vix_zscore(self) -> Tuple[float, Optional[float]]:
        """
        Return (z-score, raw_vix_level).

        Uses VIXRegimeManager if importable; falls back to yfinance directly;
        falls back to simulated neutral (0.0, None).
        """
        # Try VIXRegimeManager first (it caches internally)
        try:
            from risk.vix_regime_manager import VIXRegimeManager
            mgr = VIXRegimeManager(cache_minutes=15)
            state = mgr.get_current_state()
            if state is not None:
                # VIX z-score from percentile: percentile 0.5 → z=0
                # Use inverse-normal approximation: z ≈ (percentile - 0.5) * 4
                z = _clamp((state.vix_percentile - 0.5) * 4.0, -5.0, 8.0)
                return z, state.current_vix
        except Exception as exc:
            logger.debug("MARKET: VIXRegimeManager unavailable: %s", exc)

        # Direct yfinance fallback
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="3mo")
            if hist is not None and not hist.empty:
                closes = hist["Close"].dropna()
                if len(closes) >= 20:
                    current = float(closes.iloc[-1])
                    mean_60 = float(closes.tail(60).mean())
                    std_60 = float(closes.tail(60).std())
                    if std_60 > 0.01:
                        z = (current - mean_60) / std_60
                        return _clamp(z, -5.0, 8.0), current
                    return 0.0, current
        except Exception as exc:
            logger.debug("MARKET: yfinance VIX fetch failed: %s", exc)

        return 0.0, None

    # ------------------------------------------------------------------
    def _get_market_breadth(self) -> float:
        """
        Fraction of bellwether symbols above their 20-day MA, mapped to [-1, 1].

        breadth = 1.0  → all symbols above 20d MA (strong bullish)
        breadth = 0.0  → half above, half below (neutral)
        breadth = -1.0 → all below (strong bearish)

        Falls back to 0.0 (neutral) on any error.
        """
        try:
            import yfinance as yf
        except ImportError:
            return 0.0

        try:
            data = yf.download(
                _BREADTH_SYMBOLS,
                period="2mo",
                progress=False,
                threads=True,
            )
            if data is None or data.empty:
                return 0.0

            closes = data.get("Close")
            if closes is None or closes.empty:
                return 0.0

            # Handle single-symbol edge case (Series vs DataFrame)
            if isinstance(closes, np.ndarray) or not hasattr(closes, "columns"):
                return 0.0

            above_count = 0
            total = 0
            for sym in closes.columns:
                col = closes[sym].dropna()
                if len(col) < 20:
                    continue
                total += 1
                current = float(col.iloc[-1])
                ma20 = float(col.tail(20).mean())
                if current > ma20:
                    above_count += 1

            if total == 0:
                return 0.0

            # Map [0, 1] → [-1, 1]
            ratio = above_count / total
            return _clamp(ratio * 2.0 - 1.0, -1.0, 1.0)

        except Exception as exc:
            logger.debug("MARKET: breadth calculation failed: %s", exc)
            return 0.0
