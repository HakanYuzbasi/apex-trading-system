"""
data/social/news_sentiment_llm.py

Fetches real-time financial news headlines for a specific ticker and processes them
using a native NLTK VADER sentiment analyzer to derive a macroeconomic alpha score.
"""

import yfinance as yf
import logging
import time

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Ensure the lexicon is downloaded the first time
    nltk.download('vader_lexicon', quiet=True)
    _analyzer = SentimentIntensityAnalyzer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class NewsSentimentEngine:
    def __init__(self):
        self._cache = {}
        self.CACHE_TTL = 3600  # Cache news scores for 1 hour to prevent API bans

    def get_news_sentiment(self, symbol: str) -> float:
        """
        Returns a normalized score between -1.0 (Extreme Negative News)
        and 1.0 (Extreme Bullish News).
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK/VADER is not available. News sentiment disabled.")
            return 0.0

        clean_sym = symbol.split(":")[-1].split("/")[0]
        
        # Check cache
        if clean_sym in self._cache:
            data, timestamp = self._cache[clean_sym]
            if time.time() - timestamp < self.CACHE_TTL:
                return data

        try:
            ticker = yf.Ticker(clean_sym)
            news = ticker.news
            
            if not news:
                # No recent news detected
                return 0.0

            compound_scores = []
            
            # Extract headlines and score them using VADER
            for article in news[:5]:  # Process the top 5 most recent articles
                title = article.get("title", "")
                if title:
                    sentiment = _analyzer.polarity_scores(title)
                    # Use the compound score which is normalized between -1.0 and +1.0
                    compound_scores.append(sentiment['compound'])
            
            if not compound_scores:
                return 0.0
                
            # Average the sentiment across the recent headlines
            avg_sentiment = sum(compound_scores) / len(compound_scores)
            
            # Normalize to strictly bounds
            score = max(-1.0, min(1.0, float(avg_sentiment)))

            self._cache[clean_sym] = (score, time.time())
            return score

        except Exception as e:
            logger.warning(f"Failed to fetch or process news sentiment for {clean_sym}: {e}")
            return 0.0

_engine = NewsSentimentEngine()

def get_news_sentiment(symbol: str) -> float:
    return _engine.get_news_sentiment(symbol)
