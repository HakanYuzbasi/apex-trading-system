"""Tests for OptionsFlowFetcher — PCR normalisation, cache, symbol cleanup."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from data.social.options_flow import OptionsFlowFetcher, get_smart_money_sentiment


def _make_chain(call_vol: float, put_vol: float):
    """Build a minimal mock option chain."""
    import pandas as pd
    calls = pd.DataFrame({"volume": [call_vol]})
    puts  = pd.DataFrame({"volume": [put_vol]})
    chain = MagicMock()
    chain.calls = calls
    chain.puts  = puts
    return chain


class TestPCRNormalization:

    def test_balanced_market_near_zero(self):
        """PCR ≈ 0.7 → score ≈ 0."""
        fetcher = OptionsFlowFetcher()
        # PCR = 0.7 → score = (0.7 - 0.7) / 0.5 = 0.0
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(call_vol=1000, put_vol=700)
            score = fetcher.get_smart_money_sentiment("AAPL")
        assert score == pytest.approx(0.0, abs=0.05)

    def test_heavy_calls_bullish_positive(self):
        """PCR = 0.5 → score = +0.4 (bullish)."""
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(call_vol=2000, put_vol=1000)
            score = fetcher.get_smart_money_sentiment("MSFT")
        assert score > 0

    def test_heavy_puts_bearish_negative(self):
        """PCR = 1.5 → score ≈ -1.0 (bearish)."""
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(call_vol=1000, put_vol=1500)
            score = fetcher.get_smart_money_sentiment("TSLA")
        assert score < 0

    def test_score_clamped_to_minus_one_plus_one(self):
        """Score is always in [-1, 1]."""
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            # Extreme put dominance
            inst.option_chain.return_value = _make_chain(call_vol=100, put_vol=100000)
            score = fetcher.get_smart_money_sentiment("TSLA")
        assert -1.0 <= score <= 1.0

    def test_zero_volume_returns_zero(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(0, 0)
            score = fetcher.get_smart_money_sentiment("NVDA")
        assert score == 0.0

    def test_no_options_returns_zero(self):
        """Crypto/micro-caps with no options → 0."""
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = []
            score = fetcher.get_smart_money_sentiment("BTC/USD")
        assert score == 0.0


class TestSymbolCleanup:

    def test_crypto_prefix_stripped(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = []
            fetcher.get_smart_money_sentiment("CRYPTO:BTC/USD")
            # Should have called Ticker with "BTC" (after stripping CRYPTO: and /USD)
            call_sym = mock_ticker.call_args[0][0]
            assert "/" not in call_sym
            assert "CRYPTO:" not in call_sym

    def test_namespace_colon_stripped(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = []
            fetcher.get_smart_money_sentiment("nasdaq:AAPL")
            call_sym = mock_ticker.call_args[0][0]
            assert ":" not in call_sym

    def test_plain_symbol_unchanged(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = []
            fetcher.get_smart_money_sentiment("AAPL")
            call_sym = mock_ticker.call_args[0][0]
            assert call_sym == "AAPL"


class TestCaching:

    def test_second_call_uses_cache(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(1000, 700)
            fetcher.get_smart_money_sentiment("AAPL")
            fetcher.get_smart_money_sentiment("AAPL")
            # Ticker only called once (second hit from cache)
            assert mock_ticker.call_count == 1

    def test_expired_cache_refetches(self):
        fetcher = OptionsFlowFetcher()
        fetcher.CACHE_TTL = 0  # immediate expiry
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(1000, 700)
            fetcher.get_smart_money_sentiment("AAPL")
            fetcher.get_smart_money_sentiment("AAPL")
            assert mock_ticker.call_count == 2

    def test_different_symbols_independent_cache(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.return_value = _make_chain(1000, 700)
            fetcher.get_smart_money_sentiment("AAPL")
            fetcher.get_smart_money_sentiment("MSFT")
            assert mock_ticker.call_count == 2


class TestErrorHandling:

    def test_yfinance_exception_returns_zero(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            score = fetcher.get_smart_money_sentiment("AAPL")
        assert score == 0.0

    def test_option_chain_exception_returns_zero(self):
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01"]
            inst.option_chain.side_effect = Exception("API error")
            score = fetcher.get_smart_money_sentiment("AAPL")
        assert score == 0.0


class TestModuleSingleton:

    def test_convenience_function_returns_float(self):
        with patch("data.social.options_flow._fetcher") as mock_fetcher:
            mock_fetcher.get_smart_money_sentiment.return_value = 0.42
            result = get_smart_money_sentiment("AAPL")
            assert result == pytest.approx(0.42)

    def test_multiple_expirations_aggregated(self):
        """Confirm volumes are summed across up to 3 expirations."""
        fetcher = OptionsFlowFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            inst = mock_ticker.return_value
            inst.options = ["2026-04-01", "2026-04-08", "2026-04-15"]
            inst.option_chain.return_value = _make_chain(call_vol=1000, put_vol=700)
            fetcher.get_smart_money_sentiment("GOOG")
            assert inst.option_chain.call_count == 3
