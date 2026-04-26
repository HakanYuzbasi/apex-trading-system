# tests/conftest.py - Pytest configuration and fixtures
#
# websockets compat: alpaca-trade-api pins websockets<11 but yfinance==1.0
# imports websockets.sync and websockets.asyncio (added in websockets 12).
# We inject empty stub modules here — at pytest_configure time — so they are
# present before any test file or plugin tries to import yfinance.live.
import sys
import types as _types

def _stub_websockets_subpkgs() -> None:
    """Register websockets.sync and websockets.asyncio stubs if missing."""
    _stubs = {
        "websockets.sync":           {},
        "websockets.sync.client":    {"connect": lambda *a, **kw: None},
        "websockets.asyncio":        {},
        "websockets.asyncio.client": {"connect": lambda *a, **kw: None},
    }
    for mod_name, attrs in _stubs.items():
        if mod_name not in sys.modules:
            mod = _types.ModuleType(mod_name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[mod_name] = mod

_stub_websockets_subpkgs()

import os
import tempfile
from pathlib import Path

# Force Matplotlib to use a writable temp cache during pytest collection/imports.
_mplconfigdir = Path(tempfile.gettempdir()) / "apex_pytest_mplconfig"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

# Async support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Sample data fixtures
@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    close_prices = np.random.normal(100, 5, 252)
    return pd.DataFrame({
        "Open": close_prices * 0.99,
        "High": close_prices * 1.02,
        "Low": close_prices * 0.98,
        "Close": close_prices,
        "Volume": np.random.randint(1000000, 10000000, 252)
    }, index=dates).sort_index()

@pytest.fixture
def uptrend_data() -> pd.Series:
    """Clear uptrend data."""
    return pd.Series(np.linspace(100, 120, 100), name="UPTREND")

@pytest.fixture
def downtrend_data() -> pd.Series:
    """Clear downtrend data."""
    return pd.Series(np.linspace(120, 100, 100), name="DOWNTREND")

@pytest.fixture
def sideways_data() -> pd.Series:
    """Sideways/choppy data."""
    return pd.Series(
        np.array([100 + 2*np.sin(i/5) for i in range(100)]),
        name="SIDEWAYS"
    )

# Mock fixtures
@pytest.fixture
def mock_ibkr():
    """Mock IBKR connector."""
    ibkr = AsyncMock()
    ibkr.connect = AsyncMock(return_value=None)
    ibkr.disconnect = AsyncMock(return_value=None)
    ibkr.get_market_price = AsyncMock(return_value=100.0)
    ibkr.get_portfolio_value = AsyncMock(return_value=1_100_000.0)
    ibkr.get_all_positions = AsyncMock(return_value={})
    ibkr.execute_order = AsyncMock(return_value={"status": "FILLED"})
    return ibkr

@pytest.fixture
def mock_market_data():
    """Mock market data fetcher."""
    fetcher = MagicMock()
    fetcher.fetch_historical_data = MagicMock(return_value=pd.DataFrame())
    return fetcher

# Configure pytest
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "asyncio: async tests")
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "integration: integration tests")
