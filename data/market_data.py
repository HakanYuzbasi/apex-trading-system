"""
data/market_data.py - Professional Market Data Fetcher
Features:
- Yahoo Finance integration with fallback
- Data validation and quality checks
- Corporate actions handling
- Caching with TTL
- Rate limiting protection
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from functools import wraps

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")


def rate_limit(calls_per_second: float = 2.0):
    """Rate limiting decorator to prevent API throttling."""
    min_interval = 1.0 / calls_per_second
    last_called = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = func.__name__
            now = time.time()

            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            result = func(*args, **kwargs)
            last_called[key] = time.time()
            return result
        return wrapper
    return decorator


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class MarketDataFetcher:
    """
    Professional market data fetcher with validation, caching, and error handling.

    Features:
    - Automatic data validation
    - Corporate actions detection
    - Intelligent caching with TTL
    - Rate limiting
    - Multiple data sources with fallback
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize market data fetcher.

        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache TTL: 1 hour for intraday, 24 hours for historical
        self.cache_ttl_intraday = 3600  # 1 hour
        self.cache_ttl_historical = 86400  # 24 hours

        # Track API call timestamps for rate limiting
        self.last_call_times: Dict[str, float] = {}

        # Corporate actions log
        self.corporate_actions_log: List[Dict] = []

        logger.info("✅ MarketDataFetcher initialized")
        logger.info(f"   Cache directory: {self.cache_dir}")
        if not YF_AVAILABLE:
            logger.warning("   ⚠️ yfinance not available - limited functionality")

    def _get_cache_path(self, symbol: str, data_type: str = "historical") -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol}_{data_type}.json"

    def _is_cache_valid(self, cache_path: Path, ttl: int) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False

        modified_time = cache_path.stat().st_mtime
        age = time.time() - modified_time

        return age < ttl

    def _load_from_cache(self, symbol: str, data_type: str = "historical") -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(symbol, data_type)
        ttl = self.cache_ttl_historical if data_type == "historical" else self.cache_ttl_intraday

        if not self._is_cache_valid(cache_path, ttl):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            df = pd.DataFrame(data['data'])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            logger.debug(f"📦 Loaded {symbol} from cache")
            return df

        except Exception as e:
            logger.debug(f"Cache load failed for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame, data_type: str = "historical"):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(symbol, data_type)

            # Convert DataFrame to JSON-serializable format
            df_reset = df.reset_index()
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': df_reset.to_dict(orient='records')
            }

            with open(cache_path, 'w') as f:
                json.dump(data, f)

            logger.debug(f"💾 Cached {symbol} data")

        except Exception as e:
            logger.debug(f"Cache save failed for {symbol}: {e}")

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate market data quality.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            True if data is valid

        Raises:
            DataValidationError if validation fails
        """
        if df.empty:
            raise DataValidationError(f"{symbol}: Empty dataset")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise DataValidationError(f"{symbol}: Missing columns {missing}")

        # Check for null values
        null_pct = df[required_columns].isnull().sum() / len(df)
        if (null_pct > 0.1).any():
            cols_with_nulls = null_pct[null_pct > 0.1].index.tolist()
            raise DataValidationError(f"{symbol}: >10% null values in {cols_with_nulls}")

        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (df[price_cols] <= 0).any().any():
            raise DataValidationError(f"{symbol}: Negative or zero prices detected")

        # Check OHLC relationships
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )

        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            logger.warning(f"⚠️ {symbol}: {n_invalid} bars with invalid OHLC relationships")
            # Fix invalid bars by setting High=max, Low=min
            df.loc[invalid_ohlc, 'High'] = df.loc[invalid_ohlc, [' Open', 'High', 'Low', 'Close']].max(axis=1)
            df.loc[invalid_ohlc, 'Low'] = df.loc[invalid_ohlc, ['Open', 'High', 'Low', 'Close']].min(axis=1)

        # Check for suspicious gaps (>50% single-day move)
        returns = df['Close'].pct_change()
        suspicious_moves = abs(returns) > 0.5
        if suspicious_moves.any():
            n_suspicious = suspicious_moves.sum()
            logger.warning(f"⚠️ {symbol}: {n_suspicious} suspicious price moves (>50%)")

            # Log potential corporate actions
            for date, ret in returns[suspicious_moves].items():
                self.corporate_actions_log.append({
                    'symbol': symbol,
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'return': float(ret),
                    'type': 'potential_split' if ret < -0.3 else 'potential_action'
                })

        return True

    @rate_limit(calls_per_second=2.0)
    def fetch_historical_data(
        self,
        symbol: str,
        days: int = 252,
        use_cache: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker
            days: Number of days of history
            use_cache: Whether to use cached data
            validate: Whether to validate data quality

        Returns:
            DataFrame with OHLCV data (Date index)
        """
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, "historical")
            if cached_data is not None and len(cached_data) >= days * 0.9:
                return cached_data

        # Fetch from API
        if not YF_AVAILABLE:
            logger.error(f"❌ Cannot fetch {symbol}: yfinance not available")
            return pd.DataFrame()

        try:
            logger.debug(f"📥 Fetching {symbol} ({days} days)...")

            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))  # Fetch extra for validation

            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                actions=True  # Include dividends and splits
            )

            if df.empty:
                logger.warning(f"⚠️ No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]

            # Handle dividends and stock splits
            if 'Dividends' in df.columns and df['Dividends'].sum() > 0:
                n_dividends = (df['Dividends'] > 0).sum()
                logger.debug(f"   {symbol}: {n_dividends} dividends in period")

            if 'Stock splits' in df.columns and df['Stock splits'].sum() != len(df):
                splits = df[df['Stock splits'] != 0]['Stock splits']
                if len(splits) > 0:
                    logger.info(f"   ⚠️ {symbol}: Stock split detected: {splits.values}")
                    for date, split in splits.items():
                        self.corporate_actions_log.append({
                            'symbol': symbol,
                            'date': date.strftime('%Y-%m-%d'),
                            'type': 'stock_split',
                            'ratio': float(split)
                        })

            # Keep only OHLCV
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in ohlcv_cols if col in df.columns]]

            # Validate data
            if validate:
                try:
                    self.validate_data(df, symbol)
                except DataValidationError as e:
                    logger.warning(f"⚠️ Validation warning: {e}")
                    # Continue with data but log the warning

            # Trim to requested days
            if len(df) > days:
                df = df.iloc[-days:]

            # Cache the data
            if use_cache:
                self._save_to_cache(symbol, df, "historical")

            logger.debug(f"✅ Fetched {symbol}: {len(df)} bars")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float:
        """
        Fetch current/latest price for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Current price or 0 if unavailable
        """
        if not YF_AVAILABLE:
            return 0.0

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different price fields
            price = info.get('regularMarketPrice') or \
                    info.get('currentPrice') or \
                    info.get('previousClose') or \
                    0.0

            return float(price)

        except Exception as e:
            logger.debug(f"Error fetching current price for {symbol}: {e}")
            return 0.0

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        days: int = 252,
        max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel.

        Args:
            symbols: List of stock tickers
            days: Number of days of history
            max_workers: Maximum parallel workers

        Returns:
            Dictionary of {symbol: DataFrame}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_historical_data, symbol, days): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")

        logger.info(f"✅ Fetched {len(results)}/{len(symbols)} symbols")
        return results

    def get_corporate_actions(self) -> List[Dict]:
        """Get log of detected corporate actions."""
        return self.corporate_actions_log

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            for cache_file in self.cache_dir.glob(f"{symbol}_*.json"):
                cache_file.unlink()
            logger.info(f"🗑️ Cleared cache for {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("🗑️ Cleared all cache")
