"""
data/sentiment_analyzer.py - Free News Sentiment Analysis

Uses free data sources to extract market sentiment:
- Yahoo Finance news (free API)
- Simple keyword-based sentiment (no paid NLP API)
- Volume/price sentiment signals

No paid subscriptions required.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available for news")


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float       # 0 to 1
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    top_headlines: List[str]
    analyzed_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'news_count': self.news_count,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'top_headlines': self.top_headlines[:3],
            'analyzed_at': self.analyzed_at.isoformat()
        }


class SentimentAnalyzer:
    """
    Simple sentiment analysis using free data.
    
    Features:
    - Yahoo Finance news scraping
    - Keyword-based sentiment scoring
    - Volume/price momentum sentiment
    - Earnings surprise sentiment
    """
    
    # Positive sentiment keywords
    POSITIVE_KEYWORDS = [
        'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'climb',
        'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'bullish',
        'strong', 'record', 'high', 'growth', 'profit', 'positive',
        'innovation', 'breakthrough', 'success', 'win', 'deal',
        'partnership', 'expansion', 'launch', 'approve', 'dividend'
    ]
    
    # Negative sentiment keywords
    NEGATIVE_KEYWORDS = [
        'fall', 'drop', 'plunge', 'crash', 'decline', 'slip', 'sink',
        'miss', 'disappoint', 'underperform', 'downgrade', 'sell', 'bearish',
        'weak', 'low', 'loss', 'negative', 'warning', 'risk',
        'lawsuit', 'investigation', 'fraud', 'recall', 'layoff',
        'bankruptcy', 'default', 'cancel', 'delay', 'cut'
    ]
    
    def __init__(self, cache_minutes: int = 30):
        """
        Initialize sentiment analyzer.
        
        Args:
            cache_minutes: Cache duration for news
        """
        self.cache_minutes = cache_minutes
        self._cache: Dict[str, Tuple[SentimentResult, datetime]] = {}
        
        logger.info("SentimentAnalyzer initialized (keyword-based)")
    
    def analyze(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> SentimentResult:
        """
        Analyze sentiment for a symbol.
        
        Args:
            symbol: Stock ticker
            force_refresh: Force refresh cache
        
        Returns:
            SentimentResult
        """
        # Check cache
        if not force_refresh and symbol in self._cache:
            result, cached_at = self._cache[symbol]
            cache_age = (datetime.now() - cached_at).total_seconds() / 60
            if cache_age < self.cache_minutes:
                return result
        
        # Fetch news
        headlines = self._fetch_news(symbol)
        
        if not headlines:
            # ✅ FALLBACK: Use Price/Volume Action if news fails
            action_sentiment = VolumePriceSentiment.calculate(self._get_recent_prices(symbol))
            fallback_score = action_sentiment.get('combined', 0.0)
            
            logger.info(f"🌑 Sentiment fallback for {symbol} (News failed): {fallback_score:+.2f}")
            
            return SentimentResult(
                symbol=symbol,
                sentiment_score=float(fallback_score),
                confidence=0.25,  # Moderate confidence for action-based
                news_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                top_headlines=["[News Failed] Price/Volume Fallback"],
                analyzed_at=datetime.now()
            )
        
        # Analyze sentiment
        positive = 0
        negative = 0
        neutral = 0
        
        for headline in headlines:
            score = self._score_headline(headline)
            if score > 0:
                positive += 1
            elif score < 0:
                negative += 1
            else:
                neutral += 1
        
        total = len(headlines)
        
        # Calculate overall sentiment
        if total > 0:
            sentiment_score = (positive - negative) / total
        else:
            sentiment_score = 0.0
        
        # Confidence based on volume and agreement
        if total >= 5:
            if positive > 0 or negative > 0:
                agreement = abs(positive - negative) / (positive + negative + 0.1)
            else:
                agreement = 0.5
            confidence = min(0.9, 0.3 + 0.4 * agreement + 0.2 * (total / 10))
        else:
            confidence = 0.3
        
        result = SentimentResult(
            symbol=symbol,
            sentiment_score=float(np.clip(sentiment_score, -1, 1)),
            confidence=confidence,
            news_count=total,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            top_headlines=headlines[:5],
            analyzed_at=datetime.now()
        )
        
        # Cache result
        self._cache[symbol] = (result, datetime.now())
        
        return result

    def _get_recent_prices(self, symbol: str) -> pd.Series:
        """Helper to get recent prices for action-based sentiment fallback."""
        try:
            # First try Yahoo Finance since it's already used here
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                return hist['Close']
            return pd.Series()
        except Exception as e:
            logger.debug(f"Error getting recent prices for {symbol}: {e}")
            return pd.Series()
    
    def _fetch_news(self, symbol: str) -> List[str]:
        """Fetch news headlines from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            return []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return []
            
            headlines = []
            for item in news[:20]:  # Limit to 20 most recent
                title = item.get('title', '')
                if title:
                    headlines.append(title)
            
            return headlines
            
        except Exception as e:
            logger.debug(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _score_headline(self, headline: str) -> float:
        """
        Score a headline based on keywords.
        
        Returns:
            Score: positive (>0), negative (<0), or neutral (0)
        """
        headline_lower = headline.lower()
        
        positive_matches = sum(
            1 for word in self.POSITIVE_KEYWORDS
            if word in headline_lower
        )
        
        negative_matches = sum(
            1 for word in self.NEGATIVE_KEYWORDS
            if word in headline_lower
        )
        
        return positive_matches - negative_matches
    
    def _neutral_result(self, symbol: str) -> SentimentResult:
        """Return neutral result when no news available."""
        return SentimentResult(
            symbol=symbol,
            sentiment_score=0.0,
            confidence=0.1,
            news_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
            top_headlines=[],
            analyzed_at=datetime.now()
        )
    
    def analyze_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, SentimentResult]:
        """
        Analyze sentiment for multiple symbols.
        
        Args:
            symbols: List of tickers
        
        Returns:
            Dict of {symbol: SentimentResult}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                results[symbol] = self._neutral_result(symbol)
        
        return results
    
    def get_market_sentiment(
        self,
        results: Dict[str, SentimentResult]
    ) -> Dict[str, float]:
        """
        Calculate aggregate market sentiment.
        
        Args:
            results: Sentiment results for symbols
        
        Returns:
            Dict with market-level sentiment metrics
        """
        if not results:
            return {
                'average_sentiment': 0.0,
                'bullish_pct': 0.5,
                'bearish_pct': 0.5,
                'total_news': 0
            }
        
        sentiments = [r.sentiment_score for r in results.values()]
        confidences = [r.confidence for r in results.values()]
        
        # Weighted average by confidence
        if sum(confidences) > 0:
            weighted_avg = sum(s * c for s, c in zip(sentiments, confidences)) / sum(confidences)
        else:
            weighted_avg = np.mean(sentiments)
        
        bullish = len([s for s in sentiments if s > 0.1]) / len(sentiments)
        bearish = len([s for s in sentiments if s < -0.1]) / len(sentiments)
        
        return {
            'average_sentiment': float(weighted_avg),
            'bullish_pct': float(bullish),
            'bearish_pct': float(bearish),
            'total_news': sum(r.news_count for r in results.values())
        }


class VolumePriceSentiment:
    """
    Derive sentiment from price/volume action.
    
    No external data required - uses historical prices.
    """
    
    @staticmethod
    def calculate(prices: pd.Series, volumes: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate price/volume sentiment signals.
        
        Args:
            prices: Price series
            volumes: Optional volume series
        
        Returns:
            Dict of sentiment signals
        """
        if len(prices) < 20:
            return {'price_sentiment': 0.0, 'volume_sentiment': 0.0, 'combined': 0.0}
        
        returns = prices.pct_change().dropna()
        
        # Price sentiment: recent momentum
        ret_5d = (prices.iloc[-1] / prices.iloc[-5]) - 1 if len(prices) >= 5 else 0
        ret_20d = (prices.iloc[-1] / prices.iloc[-20]) - 1 if len(prices) >= 20 else 0
        
        price_sentiment = np.tanh((ret_5d * 2 + ret_20d) * 10)
        
        # Up days vs down days
        recent_returns = returns.iloc[-10:]
        up_ratio = (recent_returns > 0).mean()
        direction_sentiment = (up_ratio - 0.5) * 2
        
        # Volume sentiment (if available)
        volume_sentiment = 0.0
        if volumes is not None and len(volumes) >= 20:
            # Up-volume vs down-volume ratio
            vol_ret = pd.DataFrame({'vol': volumes, 'ret': returns}).dropna()
            if len(vol_ret) >= 10:
                recent = vol_ret.iloc[-10:]
                up_vol = recent[recent['ret'] > 0]['vol'].sum()
                down_vol = recent[recent['ret'] <= 0]['vol'].sum()
                
                if up_vol + down_vol > 0:
                    volume_sentiment = (up_vol - down_vol) / (up_vol + down_vol)
        
        # Combined
        combined = price_sentiment * 0.5 + direction_sentiment * 0.3 + volume_sentiment * 0.2
        
        return {
            'price_sentiment': float(np.clip(price_sentiment, -1, 1)),
            'direction_sentiment': float(np.clip(direction_sentiment, -1, 1)),
            'volume_sentiment': float(np.clip(volume_sentiment, -1, 1)),
            'combined': float(np.clip(combined, -1, 1))
        }
