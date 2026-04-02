"""
data/crypto/on_chain_flow.py - On-Chain Crypto Whale Tracker

Uses the free alternative.me Fear & Greed Index API as a contrarian
sentiment signal. Converts FNG values (0-100) to a bounded [-1, 1] score:
  - FNG=0 (Extreme Fear)  → +1.0 (contrarian buy)
  - FNG=50 (Neutral)      → 0.0
  - FNG=100 (Extreme Greed) → -1.0 (contrarian sell)

Results are cached for a configurable TTL to avoid rate limits.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://api.alternative.me/fng/?limit=1"
_DEFAULT_TTL = 300  # 5 minutes


class OnChainTracker:
    def __init__(self, cache_ttl: int = _DEFAULT_TTL):
        self._cache_ttl = cache_ttl
        self._cached_score: Optional[float] = None
        self._cache_ts: float = 0.0

    def get_crypto_flow_sentiment(self) -> float:
        now = time.time()
        if self._cached_score is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cached_score

        try:
            resp = requests.get(_API_URL, timeout=10)
            if resp.status_code != 200:
                logger.warning("FNG API returned %d", resp.status_code)
                return 0.0

            data = resp.json().get("data", [])
            if not data:
                return 0.0

            fng_value = int(data[0].get("value", 50))
            # Convert 0-100 → contrarian score [-1, 1]
            # FNG=0 → +1.0 (fear = buy), FNG=100 → -1.0 (greed = sell)
            score = float(1.0 - (fng_value / 50.0))
            score = max(-1.0, min(1.0, score))

            self._cached_score = score
            self._cache_ts = now
            return score

        except Exception as e:
            logger.warning("FNG API error: %s", e)
            return 0.0


# Module-level singleton
_default_tracker = OnChainTracker()


def get_on_chain_sentiment() -> float:
    return _default_tracker.get_crypto_flow_sentiment()
