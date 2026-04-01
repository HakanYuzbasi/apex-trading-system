"""
data/crypto/on_chain_flow.py

Fetches live public Crypto Market Sentiment (Fear & Greed Index via alternative.me)
and normalizes it into a -1.0 to 1.0 structural edge for Alpaca pairs.
"""

import requests
import logging
import time

logger = logging.getLogger(__name__)

class OnChainTracker:
    def __init__(self):
        self._cache_val = 0.0
        self._last_update = 0
        self.CACHE_TTL = 3600  # 1 Hour

    def get_crypto_flow_sentiment(self) -> float:
        """
        Calculates normalized whale flow using Fear & Greed Index.
        Extreme Greed (75-100) indicates euphoric retail flow -> Institutional Dump Imminent -> -1.0
        Extreme Fear (0-25) indicates retail capitulation -> Institutional Squeeze Imminent -> +1.0
        """
        now = time.time()
        if now - self._last_update < self.CACHE_TTL and self._cache_val != 0.0:
            return self._cache_val

        try:
            # 100% Free, reliable endpoint requiring no API keys
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    fng_value = int(data["data"][0]["value"]) # 0 to 100
                    
                    # Normalize:
                    # FNG 50 is Neutral (0.0)
                    # FNG 100 is Extreme Greed (-1.0 Alpha signal, contrarian)
                    # FNG 0 is Extreme Fear (+1.0 Alpha signal, contrarian squeeze)
                    normalized = (50 - fng_value) / 50.0
                    self._cache_val = max(-1.0, min(1.0, float(normalized)))
                    self._last_update = now
                    return self._cache_val
        except Exception as e:
            logger.warning(f"Failed to fetch On-Chain Crypto Flow: {e}. Defaulting to 0.0")
        
        return 0.0

_tracker = OnChainTracker()

def get_on_chain_sentiment() -> float:
    return _tracker.get_crypto_flow_sentiment()
