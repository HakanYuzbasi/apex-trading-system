"""
data/social/options_flow.py

Fetches real-time Options Chain data to calculate Put/Call Volatility Ratios,
acting as a robust proxy for Smart Money & Institutional block trades implicitly.
"""

import yfinance as yf
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class OptionsFlowFetcher:
    def __init__(self):
        self._cache = {}
        self.CACHE_TTL = 3600  # Cache options flow for 1 hour to avoid API rate limits

    def get_smart_money_sentiment(self, symbol: str) -> float:
        """
        Returns a normalized score between -1.0 (Extreme Bearish Options Flow/Heavy Puts)
        and 1.0 (Extreme Bullish Options Flow/Heavy Calls).
        """
        clean_sym = symbol.split(":")[-1].split("/")[0]
        
        # Check cache
        if clean_sym in self._cache:
            data, timestamp = self._cache[clean_sym]
            if time.time() - timestamp < self.CACHE_TTL:
                return data

        try:
            ticker = yf.Ticker(clean_sym)
            expirations = ticker.options
            
            if not expirations:
                # No options market for this ticker (e.g. Crypto or micro-caps)
                return 0.0

            total_call_vol = 0
            total_put_vol = 0
            
            # Aggregate volume across the 3 nearest expirations to catch immediate gamma positioning
            for exp in expirations[:3]:
                opt_chain = ticker.option_chain(exp)
                total_call_vol += opt_chain.calls['volume'].sum(skipna=True)
                total_put_vol += opt_chain.puts['volume'].sum(skipna=True)

            if total_call_vol + total_put_vol == 0:
                score = 0.0
            else:
                # Put/Call Ratio
                # PCR > 1 means more puts (bearish), PCR < 1 means more calls (bullish)
                pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 2.0
                
                # Normalize PCR to a -1.0 to 1.0 range
                # A balanced market is PCR ~ 0.7 (markets skew call-heavy structurally).
                # We map: PCR 0.5 = +1.0 (Very Bullish), PCR 1.5 = -1.0 (Very Bearish)
                score = (0.7 - pcr) / 0.5
                score = max(-1.0, min(1.0, float(score)))

            self._cache[clean_sym] = (score, time.time())
            return score

        except Exception as e:
            logger.warning(f"Options Flow fetch failed native resolution for {clean_sym}: {e}")
            return 0.0

_fetcher = OptionsFlowFetcher()

def get_smart_money_sentiment(symbol: str) -> float:
    return _fetcher.get_smart_money_sentiment(symbol)
