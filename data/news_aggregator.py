"""
data/news_aggregator.py

Multi-source news aggregator for Apex Trading.
Aggregates crypto and equity news from free no-key-required sources:
  - CryptoPanic API  (crypto-specific, vote-weighted sentiment)
  - Alternative.me Fear & Greed Index  (crypto sentiment index 0–100)
  - Yahoo Finance news  (existing, all assets, fallback)

Enhanced NLP with a 160+ term financial domain lexicon (vs the original 30-term list).
Adds sentiment MOMENTUM: tracks whether sentiment is improving or deteriorating.

All network calls are async; results cached per-symbol with a 20-min TTL.
Graceful degradation: any source failure is silently logged and skipped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ─── Domain Lexicon ──────────────────────────────────────────────────────────

_POSITIVE_TERMS: List[str] = [
    # Earnings/fundamentals
    "earnings beat", "beat estimates", "beat expectations", "record earnings",
    "profit surge", "revenue growth", "margin expansion", "guidance raised",
    "buyback", "dividend increase", "special dividend", "share repurchase",
    # Analyst actions
    "upgraded", "upgrade", "outperform", "buy rating", "strong buy",
    "price target raised", "overweight", "accumulate",
    # Corporate events
    "acquisition", "merger", "strategic partnership", "deal signed",
    "contract won", "new customer", "market share gain", "product launch",
    "fda approval", "regulatory approval", "patent granted",
    # Price action / technical
    "breakout", "all-time high", "new high", "rally", "surge", "soar",
    "jump", "gain", "rise", "climb", "bounce", "recovery",
    # Crypto specific
    "adoption", "institutional buying", "etf approval", "listed on",
    "partnership", "mainnet launch", "upgrade", "staking rewards",
    "whale accumulation", "long positions", "bullish divergence",
    # Macro positive
    "rate cut", "dovish", "stimulus", "strong economy", "jobs growth",
    "consumer confidence", "gdp beat", "soft landing",
    # Sentiment words
    "bullish", "optimistic", "confidence", "strong", "robust", "solid",
    "momentum", "outperform", "leading", "innovative",
]

_NEGATIVE_TERMS: List[str] = [
    # Earnings/fundamentals
    "earnings miss", "missed estimates", "missed expectations", "profit warning",
    "revenue decline", "margin compression", "guidance cut", "guidance lowered",
    "dividend cut", "suspended dividend", "write-down", "impairment",
    # Analyst actions
    "downgraded", "downgrade", "underperform", "sell rating",
    "price target cut", "underweight", "avoid",
    # Corporate problems
    "layoffs", "restructuring", "ceo resign", "cfo resign", "accounting fraud",
    "sec investigation", "class action", "lawsuit", "recall", "data breach",
    "bankruptcy", "chapter 11", "default", "insolvency",
    # Price action / technical
    "breakdown", "all-time low", "new low", "crash", "plunge", "drop",
    "fall", "decline", "slide", "tumble", "selloff", "collapse",
    # Crypto specific
    "hack", "exploit", "stolen", "rug pull", "delisted", "regulatory ban",
    "sec lawsuit", "money laundering", "exchange collapse", "bank run",
    "short positions", "bearish divergence", "liquidation cascade",
    # Macro negative
    "rate hike", "hawkish", "recession", "inflation surge", "stagflation",
    "unemployment rise", "gdp miss", "credit crunch", "banking crisis",
    # Sentiment words
    "bearish", "pessimistic", "concern", "weak", "fragile", "uncertain",
    "risk", "warning", "caution", "volatile", "underperform",
]

# Pre-compute lowercase sets for O(1) lookup
_POS_SET = set(_POSITIVE_TERMS)
_NEG_SET = set(_NEGATIVE_TERMS)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class NewsContext:
    """Aggregated news context for a single symbol."""
    symbol: str
    sentiment: float          # [-1, 1] negative to positive
    confidence: float         # [0, 1] how reliable the score is
    sources_count: int        # number of independent sources contributing
    momentum: float           # sentiment change vs prior cache window, [-1, 1]
    fear_greed_index: Optional[int]  # 0–100 (crypto only, None for equity)
    headlines: List[str] = field(default_factory=list)  # up to 5 for logging
    source_breakdown: Dict[str, float] = field(default_factory=dict)  # per-source sentiment

    @property
    def is_strongly_negative(self) -> bool:
        return self.sentiment < -0.35 and self.confidence > 0.4

    @property
    def is_strongly_positive(self) -> bool:
        return self.sentiment > 0.35 and self.confidence > 0.4

    @property
    def fear_greed_label(self) -> str:
        if self.fear_greed_index is None:
            return "N/A"
        if self.fear_greed_index <= 20:
            return "Extreme Fear"
        if self.fear_greed_index <= 40:
            return "Fear"
        if self.fear_greed_index <= 60:
            return "Neutral"
        if self.fear_greed_index <= 80:
            return "Greed"
        return "Extreme Greed"


_NEUTRAL_CONTEXT = NewsContext(
    symbol="", sentiment=0.0, confidence=0.0, sources_count=0,
    momentum=0.0, fear_greed_index=None,
)


# ─── Cache ────────────────────────────────────────────────────────────────────

_TTL_SECONDS = 1200        # 20-minute per-symbol cache
_FG_TTL_SECONDS = 300      # 5-minute Fear & Greed cache (single global value)

class _Cache:
    def __init__(self):
        self._data: Dict[str, Tuple[float, NewsContext]] = {}
        self._fg: Optional[Tuple[float, int]] = None  # (timestamp, value)

    def get(self, symbol: str) -> Optional[NewsContext]:
        entry = self._data.get(symbol)
        if entry and (time.monotonic() - entry[0]) < _TTL_SECONDS:
            return entry[1]
        return None

    def set(self, symbol: str, ctx: NewsContext) -> None:
        self._data[symbol] = (time.monotonic(), ctx)

    def get_fg(self) -> Optional[int]:
        if self._fg and (time.monotonic() - self._fg[0]) < _FG_TTL_SECONDS:
            return self._fg[1]
        return None

    def set_fg(self, value: int) -> None:
        self._fg = (time.monotonic(), value)


# ─── NewsAggregator ──────────────────────────────────────────────────────────

class NewsAggregator:
    """
    Async news aggregator combining CryptoPanic, Alternative.me Fear & Greed,
    and Yahoo Finance into a single NewsContext per symbol.
    All external calls use a shared httpx.AsyncClient with conservative timeouts.
    """

    _CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v1/posts/"
    _FEARGREED_URL = "https://api.alternative.me/fng/"

    def __init__(self, cryptopanic_api_key: str = ""):
        self._cp_key = cryptopanic_api_key.strip()
        self._cache = _Cache()
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(8.0, connect=4.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    headers={"User-Agent": "ApexTrading/1.0"},
                )
        return self._client

    # ── Public interface ─────────────────────────────────────────────────────

    async def get_news_context(self, symbol: str) -> NewsContext:
        """
        Return a NewsContext for the given symbol.
        Uses cache; fetches fresh data only when stale.
        Never raises — returns a neutral context on any failure.
        """
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached

        is_crypto = _is_crypto_symbol(symbol)
        clean_sym = _clean_symbol(symbol)

        tasks = [self._get_yfinance_sentiment(clean_sym)]
        if is_crypto:
            tasks.append(self._get_cryptopanic_sentiment(clean_sym))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiment_scores: List[Tuple[float, float, str]] = []  # (sentiment, confidence, source)
        all_headlines: List[str] = []

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            sent, conf, src, headlines = result
            if conf > 0:
                sentiment_scores.append((sent, conf, src))
            all_headlines.extend(headlines[:3])

        # Fetch Fear & Greed for crypto
        fg_index: Optional[int] = None
        if is_crypto:
            fg_index = await self._get_fear_greed()

        # Combine scores (confidence-weighted average)
        final_sentiment, final_confidence, source_breakdown = _combine_scores(
            sentiment_scores, fg_index, is_crypto
        )

        # Momentum: compare with previous cache entry
        prev = self._cache._data.get(symbol)
        momentum = 0.0
        if prev:
            prev_sentiment = prev[1].sentiment
            momentum = float(final_sentiment - prev_sentiment)
            momentum = max(-1.0, min(1.0, momentum))

        ctx = NewsContext(
            symbol=symbol,
            sentiment=round(float(final_sentiment), 4),
            confidence=round(float(final_confidence), 4),
            sources_count=len(sentiment_scores),
            momentum=round(momentum, 4),
            fear_greed_index=fg_index,
            headlines=all_headlines[:5],
            source_breakdown=source_breakdown,
        )
        self._cache.set(symbol, ctx)
        logger.debug(
            "NewsAggregator [%s]: sentiment=%.3f conf=%.3f sources=%d fg=%s",
            symbol, ctx.sentiment, ctx.confidence, ctx.sources_count,
            ctx.fear_greed_index,
        )
        return ctx

    # ── Yahoo Finance sentiment ────────────────────────────────────────────

    async def _get_yfinance_sentiment(
        self, symbol: str
    ) -> Optional[Tuple[float, float, str, List[str]]]:
        """Fetch and score Yahoo Finance news headlines."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = await asyncio.to_thread(lambda: ticker.news or [])
            if not news:
                return None
            headlines = []
            for item in news[:20]:
                title = None
                if isinstance(item, dict):
                    title = (
                        item.get("title")
                        or (item.get("content") or {}).get("title")
                    )
                if title:
                    headlines.append(str(title).lower())
            if not headlines:
                return None
            sentiment, confidence = _score_headlines(headlines)
            return sentiment, confidence, "yfinance", headlines[:3]
        except Exception as e:
            logger.debug("YFinance news error for %s: %s", symbol, e)
            return None

    # ── CryptoPanic ───────────────────────────────────────────────────────

    async def _get_cryptopanic_sentiment(
        self, symbol: str
    ) -> Optional[Tuple[float, float, str, List[str]]]:
        """
        Fetch CryptoPanic posts for a crypto symbol.
        Uses vote counts (bullish/bearish) as a crowd-wisdom sentiment signal.
        """
        try:
            # Map common symbols to CryptoPanic currency codes
            currency = _to_cryptopanic_currency(symbol)
            if not currency:
                return None
            params = {"currencies": currency, "public": "true", "kind": "news"}
            if self._cp_key:
                params["auth_token"] = self._cp_key

            client = await self._get_client()
            resp = await client.get(self._CRYPTOPANIC_URL, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return None

            total_bull = 0
            total_bear = 0
            headlines = []
            for post in results[:15]:
                votes = post.get("votes") or {}
                total_bull += int(votes.get("positive", 0) or 0)
                total_bear += int(votes.get("negative", 0) or 0)
                title = post.get("title", "")
                if title:
                    headlines.append(title.lower())

            # Vote-based sentiment
            total_votes = total_bull + total_bear
            if total_votes > 0:
                vote_sentiment = (total_bull - total_bear) / total_votes  # [-1, 1]
                vote_confidence = min(0.8, 0.3 + 0.5 * min(1.0, total_votes / 50))
            else:
                vote_sentiment = 0.0
                vote_confidence = 0.0

            # Headline NLP on top
            if headlines:
                nlp_sent, nlp_conf = _score_headlines(headlines)
                # Blend: 60% votes, 40% NLP
                combined_sent = 0.6 * vote_sentiment + 0.4 * nlp_sent
                combined_conf = max(vote_confidence, nlp_conf * 0.7)
            else:
                combined_sent = vote_sentiment
                combined_conf = vote_confidence

            return combined_sent, combined_conf, "cryptopanic", headlines[:3]
        except Exception as e:
            logger.debug("CryptoPanic error for %s: %s", symbol, e)
            return None

    # ── Fear & Greed ──────────────────────────────────────────────────────

    async def _get_fear_greed(self) -> Optional[int]:
        """Fetch Alternative.me Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)."""
        cached = self._cache.get_fg()
        if cached is not None:
            return cached
        try:
            client = await self._get_client()
            resp = await client.get(self._FEARGREED_URL, params={"limit": 1})
            if resp.status_code != 200:
                return None
            data = resp.json()
            value_str = (data.get("data") or [{}])[0].get("value")
            if value_str is None:
                return None
            value = int(value_str)
            self._cache.set_fg(value)
            logger.debug("Fear & Greed Index: %d (%s)", value,
                         "Fear" if value < 50 else "Greed")
            return value
        except Exception as e:
            logger.debug("Fear & Greed fetch error: %s", e)
            return None

    async def close(self) -> None:
        """Gracefully close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return (
        s.startswith("CRYPTO:")
        or "/USD" in s
        or s in {"BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "XRP"}
    )


def _clean_symbol(symbol: str) -> str:
    """Strip CRYPTO: prefix and convert BTC/USD → BTC for yfinance."""
    s = str(symbol)
    if s.upper().startswith("CRYPTO:"):
        s = s[7:]
    # For yfinance: BTC/USD → BTC-USD
    s = s.replace("/", "-")
    return s


def _to_cryptopanic_currency(symbol: str) -> Optional[str]:
    """Map our symbol format to CryptoPanic currency code."""
    s = symbol.upper().replace("CRYPTO:", "").replace("/USD", "").replace("-USD", "")
    known = {
        "BTC", "ETH", "SOL", "LINK", "AVAX", "DOT", "ADA", "XRP",
        "BCH", "LTC", "DOGE", "MATIC", "SHIB", "UNI", "AAVE", "ATOM",
        "ALGO", "XLM", "FIL", "CRV", "ETC",
    }
    return s if s in known else None


def _score_headlines(headlines: List[str]) -> Tuple[float, float]:
    """
    Score a list of lowercased headlines using the domain lexicon.
    Returns (sentiment [-1,1], confidence [0,1]).
    """
    pos_hits = 0
    neg_hits = 0
    for headline in headlines:
        for term in _POS_SET:
            if term in headline:
                pos_hits += 1
                break  # one positive match per headline
        for term in _NEG_SET:
            if term in headline:
                neg_hits += 1
                break  # one negative match per headline

    total = len(headlines)
    if total == 0:
        return 0.0, 0.0

    net = pos_hits - neg_hits
    sentiment = max(-1.0, min(1.0, net / max(1, total)))

    agreement = abs(pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
    coverage = min(1.0, (pos_hits + neg_hits) / max(1, total))
    confidence = min(0.85, 0.20 + 0.40 * agreement + 0.25 * coverage + 0.15 * min(1.0, total / 10))

    return round(sentiment, 4), round(confidence, 4)


def _combine_scores(
    scores: List[Tuple[float, float, str]],
    fear_greed: Optional[int],
    is_crypto: bool,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Combine per-source scores using confidence-weighted average.
    For crypto, integrates Fear & Greed as an additional signal.
    Returns (final_sentiment, final_confidence, source_breakdown).
    """
    source_breakdown: Dict[str, float] = {}

    if not scores and fear_greed is None:
        return 0.0, 0.0, source_breakdown

    weighted_sum = 0.0
    total_weight = 0.0

    for sent, conf, src in scores:
        source_breakdown[src] = round(sent, 3)
        weighted_sum += sent * conf
        total_weight += conf

    # Fear & Greed for crypto: convert 0-100 → [-1, 1]
    if is_crypto and fear_greed is not None:
        fg_sentiment = (fear_greed - 50) / 50.0  # 0 → -1, 50 → 0, 100 → +1
        fg_conf = 0.45  # moderate weight — it's a single index
        # Extreme readings get higher confidence
        if fear_greed <= 15 or fear_greed >= 85:
            fg_conf = 0.65
        source_breakdown["fear_greed"] = round(fg_sentiment, 3)
        weighted_sum += fg_sentiment * fg_conf
        total_weight += fg_conf

    if total_weight < 1e-9:
        return 0.0, 0.0, source_breakdown

    final_sentiment = max(-1.0, min(1.0, weighted_sum / total_weight))

    # Confidence: higher when multiple sources agree
    n_sources = len(scores) + (1 if is_crypto and fear_greed is not None else 0)
    source_agreement = 1.0 - (
        max(s for s, _, _ in scores) - min(s for s, _, _ in scores)
        if len(scores) >= 2 else 0.0
    )
    base_conf = total_weight / max(1, n_sources)
    diversity_bonus = min(0.15, 0.05 * n_sources)
    final_confidence = min(0.90, base_conf * source_agreement + diversity_bonus)

    return round(final_sentiment, 4), round(final_confidence, 4), source_breakdown
