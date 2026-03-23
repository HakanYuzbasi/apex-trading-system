from __future__ import annotations

import json

from config import ApexConfig
from data import market_data as market_data_module
from data.market_data import MarketDataFetcher


def test_fetch_historical_data_uses_synthetic_fallback_when_enabled(tmp_path, monkeypatch):
    cache_file = tmp_path / "price_cache.json"
    cache_file.write_text(json.dumps({"AAPL": 212.34}), encoding="utf-8")

    monkeypatch.setattr(ApexConfig, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ApexConfig, "MARKET_DATA_ALLOW_SYNTHETIC_HISTORY", True)
    monkeypatch.setattr(market_data_module, "YFINANCE_AVAILABLE", False)

    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data("AAPL", days=45)

    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert abs(float(df["Close"].iloc[-1]) - 212.34) < 10.0
    assert len(df) >= 30


def test_fetch_historical_data_returns_empty_without_synthetic_fallback(monkeypatch):
    monkeypatch.setattr(ApexConfig, "MARKET_DATA_ALLOW_SYNTHETIC_HISTORY", False)
    monkeypatch.setattr(market_data_module, "YFINANCE_AVAILABLE", False)

    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data("AAPL", days=30)

    assert df.empty
