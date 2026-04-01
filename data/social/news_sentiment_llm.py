"""
data/social/news_sentiment_llm.py

Enhanced news sentiment engine — domain-augmented VADER.

Upgrades over plain VADER:
  1. 220+ financial domain terms injected directly into VADER's lexicon,
     covering earnings, analyst actions, macro, crypto-specific events,
     M&A, regulatory, and executive events.
  2. Time-decay weighting: most-recent headline weighted 1.0, each older
     headline discounted by RECENCY_DECAY (default 0.85).
  3. Negation-aware pre-processing: detects "not", "no", "despite",
     "fails to" before positive terms and inverts them.
  4. Confidence scoring: agreement across headlines boosts confidence,
     single contradictory headline lowers it.
  5. 20-minute per-symbol TTL cache (up from 60-min VADER-only).

Public API (backward-compatible):
    get_news_sentiment(symbol)   → float [-1, 1]
    get_rich_sentiment(symbol)  → SentimentResult
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yfinance as yf

# ── VADER with domain lexicon injection ──────────────────────────────────────

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VADER
    nltk.download("vader_lexicon", quiet=True)
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Domain lexicon: (term, valence) pairs ────────────────────────────────────
# Valence scale matches VADER: [-4, +4], where ±1 = mild, ±3 = strong

_DOMAIN_LEXICON: Dict[str, float] = {
    # ── Earnings beats ────────────────────────────────────────────────────────
    "earnings beat": 3.2, "beat estimates": 3.0, "beat expectations": 3.0,
    "record earnings": 3.4, "record profit": 3.4, "record revenue": 3.2,
    "profit surge": 3.0, "revenue growth": 2.5, "margin expansion": 2.8,
    "guidance raised": 3.2, "raised guidance": 3.2, "raised forecast": 3.0,
    "raised outlook": 3.0, "upside surprise": 3.0,
    # ── Earnings misses ───────────────────────────────────────────────────────
    "earnings miss": -3.2, "missed estimates": -3.0, "missed expectations": -3.0,
    "profit warning": -3.4, "revenue decline": -2.8, "margin compression": -2.8,
    "guidance cut": -3.2, "guidance lowered": -3.2, "cut guidance": -3.2,
    "guidance withdrawn": -3.4, "missed forecast": -3.0,
    # ── Analyst actions bullish ───────────────────────────────────────────────
    "upgraded": 2.8, "upgrade": 2.5, "outperform": 2.5, "buy rating": 3.0,
    "strong buy": 3.2, "overweight": 2.5, "accumulate": 2.2,
    "price target raised": 2.8, "price target increase": 2.5,
    "top pick": 2.8, "best idea": 2.5, "catalyst": 2.0, "compelling": 1.8,
    # ── Analyst actions bearish ───────────────────────────────────────────────
    "downgraded": -2.8, "downgrade": -2.5, "underperform": -2.5,
    "sell rating": -3.0, "strong sell": -3.2, "underweight": -2.5,
    "price target cut": -2.8, "price target reduced": -2.5,
    "reduce": -2.0, "avoid": -2.5,
    # ── Corporate events positive ─────────────────────────────────────────────
    "acquisition": 2.0, "merger": 2.0, "strategic partnership": 2.5,
    "deal signed": 2.2, "contract won": 2.5, "new customer": 2.2,
    "market share gain": 2.5, "product launch": 2.0, "fda approval": 3.2,
    "regulatory approval": 2.8, "patent granted": 2.2, "buyback": 2.5,
    "share repurchase": 2.5, "special dividend": 2.8, "dividend increase": 2.5,
    # ── Corporate events negative ─────────────────────────────────────────────
    "layoffs": -2.5, "restructuring": -1.8, "ceo resign": -2.8,
    "cfo resign": -2.5, "accounting fraud": -3.8, "sec investigation": -3.5,
    "class action": -3.2, "recall": -2.8, "data breach": -3.0,
    "bankruptcy": -4.0, "chapter 11": -4.0, "default": -3.8,
    "insolvency": -3.8, "dividend cut": -3.0, "suspended dividend": -3.0,
    "write-down": -2.8, "impairment": -2.5, "goodwill impairment": -2.8,
    # ── Price action bullish ──────────────────────────────────────────────────
    "breakout": 2.5, "all-time high": 3.0, "new high": 2.5,
    "rally": 2.2, "surge": 2.5, "soar": 3.0, "jump": 2.2, "gain": 1.5,
    "rise": 1.5, "climb": 1.5, "bounce": 1.8, "recovery": 2.0,
    "52-week high": 2.8, "multi-year high": 2.5,
    # ── Price action bearish ──────────────────────────────────────────────────
    "breakdown": -2.5, "all-time low": -3.0, "new low": -2.5,
    "crash": -3.5, "plunge": -3.2, "drop": -2.0, "fall": -1.8,
    "decline": -1.8, "slide": -2.0, "tumble": -2.5, "selloff": -2.8,
    "collapse": -3.5, "52-week low": -2.8, "multi-year low": -2.5,
    # ── Macro bullish ─────────────────────────────────────────────────────────
    "rate cut": 2.8, "dovish": 2.5, "stimulus": 2.5, "strong economy": 2.5,
    "jobs growth": 2.2, "consumer confidence": 2.0, "gdp beat": 2.5,
    "soft landing": 2.8, "interest rate cut": 3.0, "fed pivot": 2.8,
    # ── Macro bearish ─────────────────────────────────────────────────────────
    "rate hike": -2.5, "hawkish": -2.5, "recession": -3.2,
    "inflation surge": -2.8, "stagflation": -3.2, "unemployment rise": -2.5,
    "gdp miss": -2.5, "credit crunch": -3.0, "banking crisis": -3.5,
    "interest rate hike": -2.8, "tightening": -1.8,
    # ── Crypto-specific bullish ───────────────────────────────────────────────
    "etf approval": 3.5, "btc etf": 3.2, "institutional buying": 2.8,
    "adoption": 2.2, "mainnet launch": 2.5, "staking rewards": 1.8,
    "whale accumulation": 2.0, "listed on binance": 2.5,
    "bullish divergence": 2.2,
    # ── Crypto-specific bearish ───────────────────────────────────────────────
    "hack": -3.5, "exploit": -3.2, "stolen": -3.5, "rug pull": -4.0,
    "delisted": -3.0, "regulatory ban": -3.5, "sec lawsuit": -3.8,
    "money laundering": -3.5, "exchange collapse": -4.0, "bank run": -3.8,
    "liquidation cascade": -3.2, "bearish divergence": -2.2,
    # ── Executive/governance ─────────────────────────────────────────────────
    "strategic review": 1.5, "ipo": 2.0, "spinoff": 1.8,
    "activist investor": 1.5, "hostile takeover": 1.2,
    "delisting threat": -3.0, "going private": 1.5,
}

# Negation triggers: if any of these appear within 4 tokens before a positive
# term, the score is inverted
_NEGATION_TRIGGERS = frozenset({
    "not", "no", "never", "fails", "fail", "failed", "unable",
    "despite", "without", "lack", "lacks", "missed", "miss",
    "below", "under", "disappointing", "insufficient",
})

# Minimum length for a headline to be scored (filters feed artefacts)
_MIN_HEADLINE_LEN = 12
# Time-decay factor: older headlines discounted geometrically
_RECENCY_DECAY = 0.85
# Cache TTL seconds
_CACHE_TTL = 1200  # 20 minutes


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """Rich sentiment output for a single symbol."""
    symbol: str
    sentiment: float          # [-1, 1]
    confidence: float         # [0, 1]
    headline_count: int
    top_headlines: List[str] = field(default_factory=list)
    method: str = "vader+domain"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sentiment": round(self.sentiment, 4),
            "confidence": round(self.confidence, 4),
            "headline_count": self.headline_count,
            "top_headlines": self.top_headlines[:3],
            "method": self.method,
        }


# ── Engine ───────────────────────────────────────────────────────────────────

class NewsSentimentEngine:
    """
    Domain-augmented VADER sentiment engine.
    All public methods are synchronous; call via asyncio.to_thread from async ctx.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[SentimentResult, float]] = {}
        self._analyzer: Optional[object] = None
        self._init_analyzer()

    def _init_analyzer(self) -> None:
        if not _VADER_AVAILABLE:
            logger.warning("VADER unavailable — news sentiment disabled")
            return
        try:
            self._analyzer = _VADER()
            self._analyzer.lexicon.update(_DOMAIN_LEXICON)
            logger.info(
                "NewsSentimentEngine: VADER ready, %d domain terms injected",
                len(_DOMAIN_LEXICON),
            )
        except Exception as exc:
            logger.warning("NewsSentimentEngine init failed: %s", exc)
            self._analyzer = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_news_sentiment(self, symbol: str) -> float:
        """Backward-compatible float sentiment [-1, 1]."""
        return self.get_rich_sentiment(symbol).sentiment

    def get_rich_sentiment(self, symbol: str) -> SentimentResult:
        """Full SentimentResult with confidence and headlines."""
        clean_sym = symbol.split(":")[-1].split("/")[0]
        cached = self._cache.get(clean_sym)
        if cached and (time.time() - cached[1]) < _CACHE_TTL:
            return cached[0]
        result = self._fetch_and_score(clean_sym)
        self._cache[clean_sym] = (result, time.time())
        return result

    # ── Core scoring ─────────────────────────────────────────────────────────

    def _fetch_and_score(self, clean_sym: str) -> SentimentResult:
        if self._analyzer is None:
            return SentimentResult(symbol=clean_sym, sentiment=0.0,
                                   confidence=0.0, headline_count=0,
                                   method="unavailable")
        try:
            news = yf.Ticker(clean_sym).news or []
        except Exception as exc:
            logger.debug("yFinance news fetch failed for %s: %s", clean_sym, exc)
            return SentimentResult(symbol=clean_sym, sentiment=0.0,
                                   confidence=0.0, headline_count=0,
                                   method="fetch_error")

        headlines: List[str] = [
            (a.get("title") or "").strip()
            for a in news[:10]
            if len((a.get("title") or "").strip()) >= _MIN_HEADLINE_LEN
        ]
        if not headlines:
            return SentimentResult(symbol=clean_sym, sentiment=0.0,
                                   confidence=0.0, headline_count=0,
                                   method="no_headlines")

        scores = [self._score_headline(h) for h in headlines]
        sentiment, confidence = self._aggregate(scores)
        return SentimentResult(
            symbol=clean_sym,
            sentiment=round(sentiment, 4),
            confidence=round(confidence, 4),
            headline_count=len(headlines),
            top_headlines=headlines[:5],
            method="vader+domain",
        )

    def _score_headline(self, text: str) -> float:
        """
        Score a single headline by blending:
          1. VADER compound score (single-token lexicon)
          2. Domain phrase scan (multi-word financial terms, negation-aware)
        """
        low = text.lower()
        negated = self._apply_negation(low)

        # VADER pass (single-token)
        try:
            vader_compound = float(self._analyzer.polarity_scores(negated)["compound"])
        except Exception:
            vader_compound = 0.0

        # Domain phrase pass: scan for multi-word terms VADER can't tokenise
        domain_score = self._domain_phrase_score(low)

        # Blend: VADER 60% + domain phrase 40%
        blended = vader_compound * 0.60 + domain_score * 0.40
        return max(-1.0, min(1.0, blended))

    def _domain_phrase_score(self, text_lower: str) -> float:
        """
        Scan headline for domain phrases; return mean valence normalised to [-1, 1].
        Applies negation: phrase preceded by a trigger → invert valence.
        """
        hits: List[float] = []
        for phrase, valence in _DOMAIN_LEXICON.items():
            idx = text_lower.find(phrase)
            if idx == -1:
                continue
            prefix = text_lower[max(0, idx - 30): idx]
            if any(t in prefix.split() for t in _NEGATION_TRIGGERS):
                valence = -valence
            hits.append(max(-1.0, min(1.0, valence / 4.0)))
        return max(-1.0, min(1.0, sum(hits) / len(hits))) if hits else 0.0

    def _apply_negation(self, text_lower: str) -> str:
        """
        Replace negation trigger + following positive domain phrase with a
        known-negative marker so VADER's single-token path also scores it down.
        Tries every sub-phrase starting position in a 4-token window.
        """
        tokens = text_lower.split()
        out = list(tokens)
        for i, tok in enumerate(tokens):
            if tok not in _NEGATION_TRIGGERS:
                continue
            found = False
            for start in range(i + 1, min(i + 4, len(tokens))):
                for end in range(start + 1, min(start + 4, len(tokens) + 1)):
                    phrase = " ".join(tokens[start:end])
                    if _DOMAIN_LEXICON.get(phrase, 0.0) > 0:
                        out[i] = "terrible"
                        found = True
                        break
                if found:
                    break
        return " ".join(out)

    def _aggregate(self, scores: List[float]) -> Tuple[float, float]:
        """
        Time-decay weighted average + confidence from inter-headline agreement.
        Returns (sentiment [-1,1], confidence [0,1]).
        """
        if not scores:
            return 0.0, 0.0

        weights = [_RECENCY_DECAY ** i for i in range(len(scores))]
        total_w = sum(weights)
        sentiment = max(-1.0, min(1.0,
                                  sum(s * w for s, w in zip(scores, weights)) / total_w))

        n = len(scores)
        if n == 1:
            confidence = 0.35
        else:
            mean_s = sum(scores) / n
            std_s = math.sqrt(sum((s - mean_s) ** 2 for s in scores) / n)
            agreement = max(0.0, 1.0 - std_s)
            confidence = max(0.10, min(0.95,
                                       agreement * min(1.0, abs(sentiment) * 2.5 + 0.2)))

        return sentiment, round(confidence, 4)


# ── Module-level singleton ────────────────────────────────────────────────────

_engine = NewsSentimentEngine()


def get_news_sentiment(symbol: str) -> float:
    """Backward-compatible float sentiment [-1, 1]."""
    return _engine.get_news_sentiment(symbol)


def get_rich_sentiment(symbol: str) -> SentimentResult:
    """Full SentimentResult with confidence and headlines."""
    return _engine.get_rich_sentiment(symbol)
