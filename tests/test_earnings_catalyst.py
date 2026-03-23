"""Tests for EarningsCatalystSignal (PEAD signal)."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.earnings_catalyst import (
    EarningsCatalystSignal,
    EarningsEvent,
    get_earnings_catalyst,
    get_earnings_signal,
)


def _make_earnings_df(actual: float, estimate: float, days_ago: int) -> pd.DataFrame:
    """Build a minimal yfinance-style earnings DataFrame."""
    report_date = datetime.now() - timedelta(days=days_ago)
    idx = pd.DatetimeIndex([report_date])
    return pd.DataFrame(
        {"epsActual": [actual], "epsEstimate": [estimate]},
        index=idx,
    )


class TestEarningsCatalystSignal:

    def setup_method(self):
        self.cat = EarningsCatalystSignal(drift_days=20, cache_ttl_sec=0)

    def test_crypto_symbol_returns_zero(self):
        assert self.cat.get_signal("CRYPTO:BTC/USD") == 0.0

    def test_btc_usd_slash_returns_zero(self):
        assert self.cat.get_signal("BTC/USD") == 0.0

    def test_fx_prefix_returns_zero(self):
        assert self.cat.get_signal("FX:EUR/USD") == 0.0

    def test_strong_beat_fresh(self):
        """10%+ beat on day 0 → signal near +1."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date=datetime.now(),
            actual_eps=1.10,
            estimate_eps=1.00,
            surprise_pct=0.10,
            days_since=0,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("AAPL")
        assert sig > 0.90

    def test_strong_miss_fresh(self):
        """10%+ miss on day 0 → signal near -1."""
        event = EarningsEvent(
            symbol="MSFT",
            report_date=datetime.now() - timedelta(days=1),
            actual_eps=0.90,
            estimate_eps=1.00,
            surprise_pct=-0.10,
            days_since=1,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("MSFT")
        assert sig < -0.85

    def test_signal_decays_to_zero_at_drift_end(self):
        """Signal should be ~0 at drift_days."""
        event = EarningsEvent(
            symbol="GOOG",
            report_date=datetime.now() - timedelta(days=20),
            actual_eps=2.20,
            estimate_eps=2.00,
            surprise_pct=0.10,
            days_since=20,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("GOOG")
        assert sig == pytest.approx(0.0, abs=0.01)

    def test_signal_beyond_drift_window_zero(self):
        """Past drift window → 0."""
        event = EarningsEvent(
            symbol="TSLA",
            report_date=datetime.now() - timedelta(days=25),
            actual_eps=2.20,
            estimate_eps=2.00,
            surprise_pct=0.15,
            days_since=25,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("TSLA")
        assert sig == 0.0

    def test_moderate_beat_returns_0_6_base(self):
        """5-10% beat → raw signal = 0.6 × decay."""
        event = EarningsEvent(
            symbol="AMZN",
            report_date=datetime.now(),
            actual_eps=1.05,
            estimate_eps=1.00,
            surprise_pct=0.05,   # moderate tier
            days_since=0,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("AMZN")
        assert sig == pytest.approx(0.6, abs=0.01)

    def test_tiny_surprise_below_moderate_threshold_zero(self):
        """< 4% surprise → 0."""
        event = EarningsEvent(
            symbol="NVDA",
            report_date=datetime.now(),
            actual_eps=1.02,
            estimate_eps=1.00,
            surprise_pct=0.02,
            days_since=0,
        )
        with patch.object(self.cat, '_fetch_event', return_value=event):
            sig = self.cat.get_signal("NVDA")
        assert sig == 0.0

    def test_no_event_returns_zero(self):
        with patch.object(self.cat, '_fetch_event', return_value=None):
            sig = self.cat.get_signal("XYZ")
        assert sig == 0.0

    def test_signal_cached(self):
        """Second call within TTL should NOT re-fetch."""
        cat = EarningsCatalystSignal(drift_days=20, cache_ttl_sec=3600)
        event = EarningsEvent(
            symbol="AAPL",
            report_date=datetime.now(),
            actual_eps=1.10,
            estimate_eps=1.00,
            surprise_pct=0.10,
            days_since=0,
        )
        with patch.object(cat, '_fetch_event', return_value=event) as mock_fetch:
            cat.get_signal("AAPL")
            cat.get_signal("AAPL")
        assert mock_fetch.call_count == 1

    def test_clean_symbol_strips_prefix(self):
        assert EarningsCatalystSignal._clean_symbol("CRYPTO:AAPL") == "AAPL"
        assert EarningsCatalystSignal._clean_symbol("nasdaq:AAPL") == "AAPL"

    def test_is_equity_filters_crypto(self):
        assert EarningsCatalystSignal._is_equity("CRYPTO:BTC/USD") is False
        assert EarningsCatalystSignal._is_equity("BTC/USD") is False
        assert EarningsCatalystSignal._is_equity("ETHUSDT") is False

    def test_is_equity_accepts_equity(self):
        assert EarningsCatalystSignal._is_equity("AAPL") is True
        assert EarningsCatalystSignal._is_equity("MSFT") is True
        assert EarningsCatalystSignal._is_equity("GOOG") is True

    def test_prefetch_does_not_raise(self):
        with patch.object(self.cat, '_fetch_event', return_value=None):
            self.cat.prefetch(["AAPL", "MSFT", "CRYPTO:BTC/USD"])

    def test_module_singleton(self):
        a = get_earnings_catalyst()
        b = get_earnings_catalyst()
        assert a is b

    def test_convenience_function_returns_float(self):
        with patch("data.earnings_catalyst._instance") as mock_inst:
            mock_inst.get_signal.return_value = 0.42
            # Call directly by resetting singleton
            import data.earnings_catalyst as ec_mod
            original = ec_mod._instance
            ec_mod._instance = mock_inst
            try:
                result = get_earnings_signal("AAPL")
                assert result == 0.42
            finally:
                ec_mod._instance = original
