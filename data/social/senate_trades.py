"""
data/social/senate_trades.py

Fetches and caches US Congressional (Senate) stock trades to use as an institutional alpha signal.
"Usually whatever they invest goes up!" -> High-probability insider tracking.
"""

import json
import logging
import time
import requests
from pathlib import Path
from config import ApexConfig

logger = logging.getLogger(__name__)

# Cache file to avoid spamming the free API
SENATE_CACHE_FILE = ApexConfig.DATA_DIR / "senate_trades_cache.json"
CACHE_TTL_SECONDS = 86400  # 24 hours

# For synthetic testing and fallback, assume recent heavy buys in these names:
SYNTHETIC_SENATE_BUYS = {"NVDA": 1.0, "PLTR": 0.8, "MSFT": 0.5, "LMT": 0.6, "META": 0.4, "AAPL": 0.3}
SYNTHETIC_SENATE_SELLS = {"DIS": -0.5, "INTC": -0.6, "BA": -0.7}

class SenateTradesFetcher:
    def __init__(self):
        self._cache = {}
        self._last_update = 0
        self._load_cache()

    def _load_cache(self):
        """Load from local JSON cache if still valid."""
        if SENATE_CACHE_FILE.exists():
            try:
                with open(SENATE_CACHE_FILE, "r") as f:
                    data = json.load(f)
                    self._cache = data.get("sentiment", {})
                    self._last_update = data.get("timestamp", 0)
            except Exception as e:
                logger.warning(f"Failed to load Senate trades cache: {e}")

    def _save_cache(self):
        """Save the loaded sentiment scores to JSON cache."""
        try:
            with open(SENATE_CACHE_FILE, "w") as f:
                json.dump({"timestamp": self._last_update, "sentiment": self._cache}, f)
        except Exception as e:
            logger.warning(f"Failed to save Senate trades cache: {e}")

    def update_trades_sync(self):
        """Synchronous update of Senate trades from public API (with synthetic fallback)."""
        now = time.time()
        if now - self._last_update < CACHE_TTL_SECONDS and self._cache:
            return

        sentiment = {}
        target_url = "https://senatestockwatcher.com/api/all.json"

        # If offline/synthetic history mode is active, skip the remote API for reliability
        allow_synthetic = getattr(ApexConfig, "MARKET_DATA_ALLOW_SYNTHETIC_HISTORY", False)

        # Whitelist of top-performing senators (proven to beat S&P 500 cleanly based on historical tracking)
        TOP_PERFORMING_SENATORS = [
            "Thomas R Carper", "Thomas R. Carper",
            "Tommy Tuberville", 
            "Markwayne Mullin",
            "Sheldon Whitehouse",
            "Dan Sullivan",
            "Rick Scott",
            "John Boozman",
            "Gary C Peters", "Gary C. Peters"
        ]

        try:
            if not allow_synthetic:
                response = requests.get(target_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Calculate net buy/sell pressure over the recent transactions
                    for tx in data[:300]:  # Look at the most recent 300 disclosures for filtering
                        ticker = tx.get("ticker", "").upper()
                        if not ticker or ticker == "N/A":
                            continue
                        
                        # Only track proven winners!
                        senator = str(tx.get("senator", "")).strip()
                        is_top_performer = any(top_s.lower() in senator.lower() for top_s in TOP_PERFORMING_SENATORS)
                        if not is_top_performer:
                            continue
                            
                        tx_type = str(tx.get("type", "")).lower()
                        # Weight roughly by 'Purchase' vs 'Sale'
                        impact = 0.1
                        if "purchase" in tx_type:
                            sentiment[ticker] = sentiment.get(ticker, 0.0) + impact
                        elif "sale" in tx_type:
                            sentiment[ticker] = sentiment.get(ticker, 0.0) - impact
            
            # Normalize signals between -1.0 and 1.0
            if sentiment:
                max_val = max(abs(v) for v in sentiment.values()) or 1.0
                sentiment = {k: max(-1.0, min(1.0, v / max_val)) for k, v in sentiment.items()}
            else:
                raise ValueError("No viable data returned/extracted from API")
                
        except Exception as e:
            # The system must rely strictly on real data! No synthetic overrides allowed.
            logger.warning(f"Senate API fetch failed ({e}), defaulting to neutral sentiment for loop safety.")
            sentiment = {}

        self._cache = sentiment
        self._last_update = now
        self._save_cache()

    def get_sentiment(self, symbol: str) -> float:
        """
        Get the [-1.0, 1.0] sentiment score for a given ticker driven by US Senate trading activity.
        +1.0 = heavy buying, -1.0 = heavy selling.
        """
        # Periodic refresh if expired
        if time.time() - self._last_update >= CACHE_TTL_SECONDS:
            self.update_trades_sync()
            
        # Strip prefixes like EQUITY: or FX:
        clean_sym = symbol.split(":")[-1].split("/")[0]
        return self._cache.get(clean_sym, 0.0)

# Singleton fetcher instance
_fetcher = SenateTradesFetcher()

def get_senate_sentiment(symbol: str) -> float:
    return _fetcher.get_sentiment(symbol)
