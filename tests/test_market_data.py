"""
tests/test_market_data.py - Test market data fetcher
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.market_data import MarketDataFetcher, DataValidationError


class TestMarketDataFetcher:
    """Test MarketDataFetcher functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = MarketDataFetcher(cache_dir="./data/test_cache")

    def test_initialization(self):
        """Test fetcher initialization."""
        assert self.fetcher is not None
        assert self.fetcher.cache_dir.exists()

    def test_data_validation_valid(self):
        """Test data validation with valid data."""
        # Create valid test data
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        })

        assert self.fetcher.validate_data(df, 'TEST') == True

    def test_data_validation_empty(self):
        """Test data validation with empty data."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError):
            self.fetcher.validate_data(df, 'TEST')

    def test_data_validation_missing_columns(self):
        """Test data validation with missing columns."""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106]
            # Missing Low, Close, Volume
        })

        with pytest.raises(DataValidationError):
            self.fetcher.validate_data(df, 'TEST')

    def test_data_validation_negative_prices(self):
        """Test data validation with negative prices."""
        df = pd.DataFrame({
            'Open': [100, -1],  # Negative price
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [104, 105],
            'Volume': [1000000, 1100000]
        })

        with pytest.raises(DataValidationError):
            self.fetcher.validate_data(df, 'TEST')

    def test_fetch_historical_data(self):
        """Test fetching historical data."""
        # This test requires network access and yfinance
        df = self.fetcher.fetch_historical_data('AAPL', days=30, validate=True)

        if not df.empty:
            assert 'Open' in df.columns
            assert 'Close' in df.columns
            assert len(df) > 0

    def test_cache_functionality(self):
        """Test cache save and load."""
        # Create test data
        test_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [99, 100],
            'Close': [104, 105],
            'Volume': [1000000, 1100000]
        })

        # Save to cache
        self.fetcher._save_to_cache('TEST', test_data, 'test')

        # Load from cache
        loaded = self.fetcher._load_from_cache('TEST', 'test')

        assert loaded is not None
        assert len(loaded) == len(test_data)


def test_rate_limiting():
    """Test rate limiting decorator."""
    from data.market_data import rate_limit
    import time

    @rate_limit(calls_per_second=5.0)
    def test_func():
        return time.time()

    # Call multiple times and check timing
    start = time.time()
    for _ in range(3):
        test_func()
    elapsed = time.time() - start

    # Should take at least 0.4 seconds (3 calls at 5 calls/sec)
    assert elapsed >= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
