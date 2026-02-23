"""
data/social/news_adapter.py

NewsAggregatorAdapter — wraps the existing SentimentAnalyzer to produce a
SourceSnapshot compatible with the social-risk pipeline.

Aggregates Yahoo Finance keyword-sentiment across bellwether symbols and
converts the result into attention_z / sentiment_score / confidence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from data.social.adapters import SourceSnapshot, SocialSourceAdapter, _clamp

logger = logging.getLogger(__name__)

# Small set of bellwether tickers — broad market coverage without hammering yfinance
_BELLWETHER_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]


class NewsAggregatorAdapter(SocialSourceAdapter):
    """
    Produces a social-feed snapshot from Yahoo Finance news sentiment.

    Uses the existing ``SentimentAnalyzer`` (keyword-based, free) to scan
    bellwether tickers and aggregate into a single market-level signal.

    quality = "ok"   when yfinance is available and news is found
    quality = "degraded" when yfinance works but no news returned (price fallback)
    quality = "missing"  when yfinance is not installed
    """

    source_name = "NEWS_AGG"
    # No env endpoint / token — this adapter runs locally
    env_endpoint_key = ""
    env_token_key = ""
    default_local_name = ""

    def __init__(
        self,
        data_dir: Path,
        bellwether_symbols: Optional[List[str]] = None,
        timeout_seconds: float = 10.0,
    ):
        super().__init__(data_dir=data_dir, timeout_seconds=timeout_seconds)
        self._symbols = list(bellwether_symbols or _BELLWETHER_SYMBOLS)

    # ------------------------------------------------------------------
    # Override fetch() entirely — we don't use file/http loading
    # ------------------------------------------------------------------
    def fetch(
        self,
        now: Optional[datetime] = None,
        freshness_sla_seconds: int = 1800,
    ) -> SourceSnapshot:
        now_dt = now or datetime.utcnow()
        if now_dt.tzinfo is not None:
            now_dt = now_dt.astimezone(timezone.utc).replace(tzinfo=None)

        try:
            from data.sentiment_analyzer import SentimentAnalyzer, YFINANCE_AVAILABLE
        except Exception:
            return self._missing_snapshot(now_dt, ["import_error"])

        if not YFINANCE_AVAILABLE:
            return self._missing_snapshot(now_dt, ["yfinance_unavailable"])

        try:
            analyzer = SentimentAnalyzer(cache_minutes=15)
            results = analyzer.analyze_batch(self._symbols)
            market = analyzer.get_market_sentiment(results)
        except Exception as exc:
            logger.warning("NEWS_AGG: sentiment analysis failed: %s", exc)
            return self._missing_snapshot(now_dt, ["fetch_error"])

        total_news = int(market.get("total_news", 0))
        avg_sentiment = float(market.get("average_sentiment", 0.0))
        bullish_pct = float(market.get("bullish_pct", 0.5))
        bearish_pct = float(market.get("bearish_pct", 0.5))

        # attention_z: z-score proxy from news volume
        # Baseline: ~3 headlines per symbol → 24 total for 8 symbols
        baseline_news = len(self._symbols) * 3.0
        if baseline_news > 0 and total_news > 0:
            attention_z = _clamp(
                (total_news - baseline_news) / max(1.0, baseline_news * 0.3),
                -5.0,
                8.0,
            )
        else:
            attention_z = 0.0

        sentiment_score = _clamp(avg_sentiment, -1.0, 1.0)

        # confidence: based on how many symbols returned news
        symbols_with_news = sum(1 for r in results.values() if r.news_count > 0)
        coverage = symbols_with_news / max(1, len(self._symbols))
        confidence = _clamp(0.2 + 0.6 * coverage, 0.0, 1.0)

        flags: List[str] = []
        if total_news == 0:
            flags.append("no_news_returned")
        if coverage < 0.3:
            flags.append("low_symbol_coverage")

        quality_status = "ok" if not flags else "degraded"
        quality_score = _clamp(confidence * (0.6 if quality_status == "ok" else 0.35) + 0.3, 0.0, 1.0)

        return SourceSnapshot(
            source=self.source_name,
            fetched_at=now_dt.isoformat(),
            freshness_ts=now_dt.isoformat(),
            attention_z=attention_z,
            sentiment_score=sentiment_score,
            confidence=confidence,
            sample_size=total_news,
            quality_status=quality_status,
            quality_score=quality_score,
            quality_flags=sorted(set(flags)),
            adapter_mode="aggregator",
            raw_payload={
                "total_news": total_news,
                "average_sentiment": avg_sentiment,
                "bullish_pct": bullish_pct,
                "bearish_pct": bearish_pct,
                "symbols_queried": len(self._symbols),
                "symbols_with_news": symbols_with_news,
            },
        )

    # ------------------------------------------------------------------
    def _missing_snapshot(self, now_dt: datetime, flags: List[str]) -> SourceSnapshot:
        return SourceSnapshot(
            source=self.source_name,
            fetched_at=now_dt.isoformat(),
            freshness_ts=now_dt.isoformat(),
            attention_z=0.0,
            sentiment_score=0.0,
            confidence=0.0,
            sample_size=0,
            quality_status="missing",
            quality_score=0.0,
            quality_flags=sorted(set(flags + ["source_unavailable"])),
            adapter_mode="aggregator",
            raw_payload={},
        )
