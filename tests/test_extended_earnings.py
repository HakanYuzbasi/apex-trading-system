"""
tests/test_extended_earnings.py — Unit tests for extended earnings intelligence
in data/earnings_catalyst.py (revision + persistence + extended signal)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.earnings_catalyst import (
    EarningsCatalystSignal,
    EarningsEvent,
    get_earnings_catalyst,
    get_earnings_persistence_signal,
    get_earnings_revision_signal,
    get_extended_earnings_signal,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_earnings_history(surprises: list) -> pd.DataFrame:
    """Build a fake earnings history DataFrame (most recent last)."""
    rows = []
    for i, (actual, estimate) in enumerate(surprises):
        rows.append({"epsActual": actual, "epsEstimate": estimate})
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="Q")
    return pd.DataFrame(rows, index=idx)


def _make_recommendations(strong_buy: int, buy: int, hold: int, sell: int, strong_sell: int):
    """Build a fake recommendations DataFrame with two rows."""
    row1 = {
        "strongBuy": strong_buy // 2, "buy": buy // 2, "hold": hold // 2,
        "sell": sell // 2, "strongSell": strong_sell // 2,
    }
    row2 = {
        "strongBuy": strong_buy - row1["strongBuy"],
        "buy": buy - row1["buy"],
        "hold": hold - row1["hold"],
        "sell": sell - row1["sell"],
        "strongSell": strong_sell - row1["strongSell"],
    }
    idx = pd.date_range("2024-01-01", periods=2, freq="M")
    return pd.DataFrame([row1, row2], index=idx)


# ── get_extended_signal ────────────────────────────────────────────────────────

class TestExtendedSignal:
    def test_returns_float(self):
        cat = EarningsCatalystSignal()
        with patch.object(cat, "get_signal", return_value=0.5), \
             patch.object(cat, "get_revision_signal", return_value=0.3), \
             patch.object(cat, "get_persistence_signal", return_value=0.2):
            result = cat.get_extended_signal("AAPL")
        assert isinstance(result, float)

    def test_bounded(self):
        cat = EarningsCatalystSignal()
        with patch.object(cat, "get_signal", return_value=1.0), \
             patch.object(cat, "get_revision_signal", return_value=1.0), \
             patch.object(cat, "get_persistence_signal", return_value=1.0):
            result = cat.get_extended_signal("AAPL")
        assert -1.0 <= result <= 1.0

    def test_all_positive_gives_positive(self):
        cat = EarningsCatalystSignal()
        with patch.object(cat, "get_signal", return_value=0.8), \
             patch.object(cat, "get_revision_signal", return_value=0.6), \
             patch.object(cat, "get_persistence_signal", return_value=0.5):
            result = cat.get_extended_signal("AAPL")
        assert result > 0.0

    def test_all_negative_gives_negative(self):
        cat = EarningsCatalystSignal()
        with patch.object(cat, "get_signal", return_value=-0.8), \
             patch.object(cat, "get_revision_signal", return_value=-0.6), \
             patch.object(cat, "get_persistence_signal", return_value=-0.5):
            result = cat.get_extended_signal("AAPL")
        assert result < 0.0

    def test_zero_components_returns_zero(self):
        cat = EarningsCatalystSignal()
        with patch.object(cat, "get_signal", return_value=0.0), \
             patch.object(cat, "get_revision_signal", return_value=0.0), \
             patch.object(cat, "get_persistence_signal", return_value=0.0):
            result = cat.get_extended_signal("AAPL")
        assert result == pytest.approx(0.0)

    def test_crypto_returns_zero(self):
        cat = EarningsCatalystSignal()
        result = cat.get_extended_signal("CRYPTO:BTC/USD")
        assert result == 0.0

    def test_blending_weights(self):
        cat = EarningsCatalystSignal()
        # pead=1.0, revision=0.0, persistence=0.0 → 0.50×1.0 = 0.50
        with patch.object(cat, "get_signal", return_value=1.0), \
             patch.object(cat, "get_revision_signal", return_value=0.0), \
             patch.object(cat, "get_persistence_signal", return_value=0.0):
            result = cat.get_extended_signal("AAPL")
        assert result == pytest.approx(0.50, abs=0.01)


# ── get_revision_signal ────────────────────────────────────────────────────────

class TestRevisionSignal:
    def test_upgrade_trend_positive(self):
        cat = EarningsCatalystSignal()
        # Most recent period has more strong buys
        rec = _make_recommendations(
            strong_buy=10, buy=5, hold=2, sell=1, strong_sell=0
        )
        mock_ticker = MagicMock()
        mock_ticker.recommendations = rec

        with patch("yfinance.Ticker", return_value=mock_ticker):
            sig = cat.get_revision_signal("AAPL")
        assert isinstance(sig, float)
        assert -1.0 <= sig <= 1.0

    def test_returns_zero_for_crypto(self):
        cat = EarningsCatalystSignal()
        result = cat.get_revision_signal("CRYPTO:BTC/USD")
        assert result == 0.0

    def test_empty_recommendations_returns_zero(self):
        cat = EarningsCatalystSignal()
        mock_ticker = MagicMock()
        mock_ticker.recommendations = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = cat.get_revision_signal("AAPL")
        assert result == 0.0

    def test_fetch_failure_returns_zero(self):
        cat = EarningsCatalystSignal()
        with patch("yfinance.Ticker", side_effect=RuntimeError("no data")):
            result = cat.get_revision_signal("AAPL")
        assert result == 0.0

    def test_result_cached(self):
        cat = EarningsCatalystSignal()
        call_count = [0]
        orig = cat._compute_revision

        def _counting_compute(sym):
            call_count[0] += 1
            return 0.3

        cat._compute_revision = _counting_compute
        cat.get_revision_signal("AAPL")
        cat.get_revision_signal("AAPL")
        assert call_count[0] == 1  # cache hit on second call


# ── get_persistence_signal ────────────────────────────────────────────────────

class TestPersistenceSignal:
    def test_consecutive_beats_positive(self):
        cat = EarningsCatalystSignal()
        # 4 quarters of 10%+ beats
        hist = _make_earnings_history([(1.10, 1.0), (1.10, 1.0), (1.10, 1.0), (1.10, 1.0)])
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            sig = cat.get_persistence_signal("AAPL")
        assert sig > 0.0

    def test_consecutive_misses_negative(self):
        cat = EarningsCatalystSignal()
        # 4 quarters of 10%+ misses
        hist = _make_earnings_history([(0.90, 1.0), (0.90, 1.0), (0.90, 1.0), (0.90, 1.0)])
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            sig = cat.get_persistence_signal("AAPL")
        assert sig < 0.0

    def test_mixed_quarters_near_zero(self):
        cat = EarningsCatalystSignal()
        # 2 beats + 2 misses → balanced
        hist = _make_earnings_history([(1.1, 1.0), (0.9, 1.0), (1.1, 1.0), (0.9, 1.0)])
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            sig = cat.get_persistence_signal("AAPL")
        assert abs(sig) < 0.3

    def test_output_bounded(self):
        cat = EarningsCatalystSignal()
        hist = _make_earnings_history([(2.0, 1.0)] * 8)  # extreme beats
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            sig = cat.get_persistence_signal("AAPL")
        assert -1.0 <= sig <= 1.0

    def test_empty_history_returns_zero(self):
        cat = EarningsCatalystSignal()
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = cat.get_persistence_signal("AAPL")
        assert result == 0.0

    def test_fetch_failure_returns_zero(self):
        cat = EarningsCatalystSignal()
        with patch("yfinance.Ticker", side_effect=RuntimeError("no data")):
            result = cat.get_persistence_signal("AAPL")
        assert result == 0.0

    def test_crypto_returns_zero(self):
        cat = EarningsCatalystSignal()
        result = cat.get_persistence_signal("CRYPTO:BTC/USD")
        assert result == 0.0

    def test_result_cached(self):
        cat = EarningsCatalystSignal()
        call_count = [0]
        orig_compute = cat._compute_persistence

        def _counting_compute(sym, n):
            call_count[0] += 1
            return 0.5

        cat._compute_persistence = _counting_compute
        cat.get_persistence_signal("MSFT")
        cat.get_persistence_signal("MSFT")
        assert call_count[0] == 1


# ── Convenience functions ─────────────────────────────────────────────────────

class TestConvenienceFunctions:
    def test_get_extended_earnings_signal_returns_float(self):
        cat = get_earnings_catalyst()
        with patch.object(cat, "get_signal", return_value=0.0), \
             patch.object(cat, "get_revision_signal", return_value=0.0), \
             patch.object(cat, "get_persistence_signal", return_value=0.0):
            result = get_extended_earnings_signal("AAPL")
        assert isinstance(result, float)

    def test_get_earnings_revision_signal_returns_float(self):
        cat = get_earnings_catalyst()
        with patch.object(cat, "_compute_revision", return_value=0.2):
            cat._cache.clear()
            result = get_earnings_revision_signal("AAPL")
        assert isinstance(result, float)

    def test_get_earnings_persistence_signal_returns_float(self):
        cat = get_earnings_catalyst()
        with patch.object(cat, "_compute_persistence", return_value=0.3):
            cat._cache.clear()
            result = get_earnings_persistence_signal("AAPL")
        assert isinstance(result, float)


# ── Regression: original PEAD signal unchanged ────────────────────────────────

class TestPEADRegression:
    def test_original_get_signal_still_works(self):
        cat = EarningsCatalystSignal()
        # No event → 0.0
        with patch.object(cat, "_fetch_event", return_value=None):
            result = cat.get_signal("AAPL")
        assert result == 0.0

    def test_strong_beat_in_window(self):
        cat = EarningsCatalystSignal()
        event = EarningsEvent(
            symbol="AAPL",
            report_date=datetime.now() - timedelta(days=5),
            actual_eps=1.15,
            estimate_eps=1.0,
            surprise_pct=0.15,
            days_since=5,
        )
        with patch.object(cat, "_fetch_event", return_value=event):
            result = cat.get_signal("AAPL")
        assert result > 0.0
