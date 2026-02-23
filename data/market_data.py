"""
data/market_data.py - Market Data Fetcher
Fetches real market data using Yahoo Finance (yfinance)
"""

import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import threading
import time

logger = logging.getLogger(__name__)

from core.symbols import AssetClass, parse_symbol, to_yfinance_ticker
from config import ApexConfig

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")


class MarketDataFetcher:
    """
    Fetches market data from Yahoo Finance.

    Features:
    - Current price fetching with caching
    - Historical OHLCV data
    - Batch downloading for efficiency
    - Rate limiting to avoid API blocks
    - Fallback mechanisms for reliability
    """

    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Initialize the market data fetcher.

        Args:
            cache_ttl_seconds: Time-to-live for price cache in seconds
        """
        self.cache_ttl = cache_ttl_seconds
        self._price_cache: Dict[str, tuple] = {}  # symbol -> (price, timestamp)
        self._historical_cache: Dict[str, pd.DataFrame] = {}  # symbol_days -> DataFrame
        self._historical_cache_times: Dict[str, datetime] = {}  # cache_key -> timestamp
        self._lock = threading.Lock()
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

        if not YFINANCE_AVAILABLE:
            logger.error("yfinance is not installed. Market data will not be available.")

    def _rate_limit(self):
        """Apply rate limiting to avoid API blocks."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _map_for_data(self, parsed):
        """Map symbols for data providers (e.g., USDT/USDC -> USD)."""
        data_map = getattr(ApexConfig, "DATA_PAIR_MAP", {}) or {}
        candidates = [
            parsed.raw,
            parsed.normalized,
            parsed.normalized.replace("CRYPTO:", "").replace("FX:", ""),
        ]
        for key in candidates:
            if key in data_map:
                try:
                    return parse_symbol(data_map[key])
                except ValueError:
                    return parsed
        # Fallback: map USDT/USDC to USD for data providers like yfinance
        try:
            if (
                parsed.asset_class == AssetClass.CRYPTO
                and parsed.quote in {"USDT", "USDC"}
                and getattr(ApexConfig, "DATA_MAP_STABLECOINS_TO_USD", False)
            ):
                return parse_symbol(f"{parsed.base}/USD")
        except Exception as e:
            logger.debug("Data map stablecoin fallback failed: %s", e)
        return parsed

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol with caching.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or 0.0 if unavailable
        """
        if not YFINANCE_AVAILABLE:
            return 0.0

        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            logger.warning(f"Invalid symbol format for price fetch: {symbol}")
            return 0.0

        parsed = self._map_for_data(parsed)

        cache_key = parsed.normalized
        yf_symbol = to_yfinance_ticker(parsed)

        # Check cache first
        with self._lock:
            if cache_key in self._price_cache:
                cached_price, cached_time = self._price_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                    return cached_price

        try:
            self._rate_limit()

            ticker = yf.Ticker(yf_symbol)

            # Try fast_info first (faster)
            try:
                price = ticker.fast_info.get('lastPrice', 0)
                if price and price > 0:
                    with self._lock:
                        self._price_cache[cache_key] = (float(price), datetime.now())
                    return float(price)
            except Exception as e:
                logger.debug("Fast info price fetch failed for %s: %s", parsed.normalized, e)

            # Fallback to history
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                with self._lock:
                    self._price_cache[cache_key] = (price, datetime.now())
                return price

            # Second fallback: daily history
            hist = ticker.history(period='5d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                with self._lock:
                    self._price_cache[cache_key] = (price, datetime.now())
                return price

            logger.warning(f"No price data available for {parsed.normalized}")
            return 0.0

        except Exception as e:
            logger.debug(f"Error fetching price for {parsed.normalized}: {e}")
            return 0.0

    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols efficiently.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary of {symbol: price}
        """
        if not YFINANCE_AVAILABLE:
            return {s: 0.0 for s in symbols}

        prices = {}

        parsed_map = {}
        for symbol in symbols:
            try:
                parsed_map[symbol] = parse_symbol(symbol)
            except ValueError:
                logger.warning(f"Invalid symbol format for price fetch: {symbol}")
                prices[symbol] = 0.0

        # Check cache first
        symbols_to_fetch = []
        for symbol in symbols:
            if symbol not in parsed_map:
                continue
            parsed = parsed_map[symbol]
            cache_key = parsed.normalized
            with self._lock:
                if cache_key in self._price_cache:
                    cached_price, cached_time = self._price_cache[cache_key]
                    if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                        prices[symbol] = cached_price
                        continue
            symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return prices

        try:
            self._rate_limit()

            # Batch download
            yf_map = {s: to_yfinance_ticker(parsed_map[s]) for s in symbols_to_fetch}
            tickers_str = ' '.join(yf_map.values())
            data = yf.download(
                tickers_str,
                period='1d',
                interval='1m',
                progress=False,
                threads=True
            )

            if data.empty:
                # Fallback to individual fetching
                for symbol in symbols_to_fetch:
                    prices[symbol] = self.get_current_price(symbol)
                return prices

            # Extract prices
            if len(symbols_to_fetch) == 1:
                symbol = symbols_to_fetch[0]
                if 'Close' in data.columns:
                    price = float(data['Close'].iloc[-1])
                    prices[symbol] = price
                    with self._lock:
                        self._price_cache[parsed_map[symbol].normalized] = (price, datetime.now())
            else:
                for symbol in symbols_to_fetch:
                    try:
                        yf_symbol = yf_map[symbol]
                        if ('Close', yf_symbol) in data.columns:
                            price = float(data[('Close', yf_symbol)].dropna().iloc[-1])
                            prices[symbol] = price
                            with self._lock:
                                self._price_cache[parsed_map[symbol].normalized] = (price, datetime.now())
                        else:
                            prices[symbol] = self.get_current_price(symbol)
                    except Exception as e:
                        logger.debug("Batch price extraction failed for %s: %s", symbol, e)
                        prices[symbol] = 0.0

            return prices

        except Exception as e:
            logger.error(f"Error batch fetching prices: {e}")
            # Fallback to individual fetching
            for symbol in symbols_to_fetch:
                prices[symbol] = self.get_current_price(symbol)
            return prices

    def fetch_historical_data(
        self,
        symbol: str,
        days: int = 252,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            days: Number of days of history (default: 252 = 1 trading year)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            logger.warning(f"Invalid symbol format for history fetch: {symbol}")
            return pd.DataFrame()

        parsed = self._map_for_data(parsed)

        cache_key = f"{parsed.normalized}_{days}"

        # Check cache
        if use_cache and cache_key in self._historical_cache:
            cached_df = self._historical_cache[cache_key]
            # Check if cache is recent (within 1 hour)
            if hasattr(cached_df, '_cache_time'):
                if (datetime.now() - cached_df._cache_time).total_seconds() < 3600:
                    return cached_df.copy()

        try:
            self._rate_limit()

            yf_symbol = to_yfinance_ticker(parsed)
            ticker = yf.Ticker(yf_symbol)

            # Calculate period string
            if days <= 5:
                period = '5d'
            elif days <= 30:
                period = '1mo'
            elif days <= 90:
                period = '3mo'
            elif days <= 180:
                period = '6mo'
            elif days <= 365:
                period = '1y'
            elif days <= 730:
                period = '2y'
            else:
                period = '5y'

            # Fetch data
            df = ticker.history(period=period)

            if df.empty:
                logger.warning(f"No historical data for {parsed.normalized}")
                return pd.DataFrame()

            # Standardize column names
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            # Keep only OHLCV columns
            columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[c for c in columns_to_keep if c in df.columns]]

            # Remove timezone info for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Cache the result
            df._cache_time = datetime.now()
            self._historical_cache[cache_key] = df.copy()

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {parsed.normalized}: {e}")
            return pd.DataFrame()

    def fetch_historical_batch(
        self,
        symbols: List[str],
        days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols efficiently.

        Args:
            symbols: List of ticker symbols
            days: Number of days of history

        Returns:
            Dictionary of {symbol: DataFrame}
        """
        if not YFINANCE_AVAILABLE:
            return {s: pd.DataFrame() for s in symbols}

        results = {}
        parsed_map = {}
        for symbol in symbols:
            try:
                parsed_map[symbol] = parse_symbol(symbol)
            except ValueError:
                logger.warning(f"Invalid symbol format for history fetch: {symbol}")
                results[symbol] = pd.DataFrame()
                continue
            parsed_map[symbol] = self._map_for_data(parsed_map[symbol])

        try:
            self._rate_limit()

            # Calculate period string
            if days <= 30:
                period = '1mo'
            elif days <= 90:
                period = '3mo'
            elif days <= 180:
                period = '6mo'
            elif days <= 365:
                period = '1y'
            else:
                period = '2y'

            # Batch download
            yf_map = {s: to_yfinance_ticker(parsed_map[s]) for s in symbols if s in parsed_map}
            if not yf_map:
                return results

            tickers_str = ' '.join(yf_map.values())
            data = yf.download(
                tickers_str,
                period=period,
                progress=False,
                threads=True,
                group_by='ticker'
            )

            if data.empty:
                # Fallback to individual fetching
                for symbol in symbols:
                    results[symbol] = self.fetch_historical_data(symbol, days)
                return results

            # Extract data for each symbol
            for symbol in symbols:
                if symbol not in parsed_map:
                    continue
                try:
                    if len(yf_map) == 1:
                        df = data.copy()
                    else:
                        yf_symbol = yf_map[symbol]
                        if yf_symbol in data.columns.get_level_values(0):
                            df = data[yf_symbol].copy()
                        else:
                            results[symbol] = self.fetch_historical_data(symbol, days)
                            continue

                    # Clean up
                    df = df.dropna()
                    if not df.empty:
                        # Remove timezone
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        results[symbol] = df
                    else:
                        results[symbol] = pd.DataFrame()

                except Exception as e:
                    logger.debug(f"Error extracting data for {symbol}: {e}")
                    results[symbol] = self.fetch_historical_data(symbol, days)

            return results

        except Exception as e:
            logger.error(f"Error batch fetching historical data: {e}")
            # Fallback to individual fetching
            for symbol in symbols:
                results[symbol] = self.fetch_historical_data(symbol, days)
            return results

    def get_market_info(self, symbol: str) -> Dict:
        """
        Get market info for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with market cap, sector, industry, etc.
        """
        if not YFINANCE_AVAILABLE:
            return {}

        try:
            parsed = parse_symbol(symbol)
            if parsed.asset_class != AssetClass.EQUITY:
                return {'symbol': parsed.normalized, 'sector': 'Unknown'}

            self._rate_limit()
            ticker = yf.Ticker(to_yfinance_ticker(parsed))
            info = ticker.info

            return {
                'symbol': parsed.normalized,
                'name': info.get('shortName', parsed.base),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'avg_volume': info.get('averageVolume', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }

        except Exception as e:
            logger.debug(f"Error fetching market info for {symbol}: {e}")
            return {'symbol': symbol, 'sector': 'Unknown'}

    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        now = datetime.utcnow()

        # Market hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC during EST)
        # Adjust for DST as needed

        # Check weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Simplified check: 14:30 - 21:00 UTC
        market_open_utc = now.replace(hour=14, minute=30, second=0, microsecond=0)
        market_close_utc = now.replace(hour=21, minute=0, second=0, microsecond=0)

        return market_open_utc <= now <= market_close_utc

    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._price_cache.clear()
            self._historical_cache.clear()
        logger.info("Market data cache cleared")
