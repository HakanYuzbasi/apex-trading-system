"""Tests for DeltaHedger — beta calculation, hedge sizing, notional cap."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from risk.market_neutral_hedge import DeltaHedger


def _make_hedger(beta_map: dict | None = None) -> DeltaHedger:
    mdf = MagicMock()
    mdf.get_market_info = lambda sym: {"beta": (beta_map or {}).get(sym, 1.0)}
    mdf.get_current_price = lambda sym: 500.0
    return DeltaHedger(market_data_fetcher=mdf)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDeltaHedgerBasics:

    def test_empty_positions_returns_zero(self):
        dh = _make_hedger()
        qty = _run(dh.calculate_hedge_order({}, {}, "SPY"))
        assert qty == 0

    def test_zero_qty_positions_returns_zero(self):
        dh = _make_hedger()
        qty = _run(dh.calculate_hedge_order({"AAPL": 0, "MSFT": 0}, {"AAPL": 200.0, "MSFT": 300.0}, "SPY"))
        assert qty == 0

    def test_below_min_imbalance_returns_zero(self):
        """Small portfolio with beta=1 but total delta < $5000 threshold."""
        dh = _make_hedger({"AAPL": 1.0})
        prices = {"AAPL": 100.0, "SPY": 500.0}
        # 10 shares × $100 × beta1.0 = $1000 exposure → below $5000 threshold
        qty = _run(dh.calculate_hedge_order({"AAPL": 10}, prices, "SPY"))
        assert qty == 0

    def test_long_portfolio_returns_negative_spy_qty(self):
        """Long $100k AAPL (beta=1) → should SHORT SPY to neutralise."""
        dh = _make_hedger({"AAPL": 1.0})
        prices = {"AAPL": 200.0, "SPY": 500.0}
        # 500 shares × $200 × beta1 = $100k → need -$100k SPY = -200 shares
        qty = _run(dh.calculate_hedge_order({"AAPL": 500}, prices, "SPY"))
        assert qty < 0  # SHORT SPY
        assert abs(qty) == pytest.approx(200, abs=1)

    def test_high_beta_amplifies_hedge_size(self):
        """Beta=2 stock requires 2x more SPY short."""
        dh = _make_hedger({"TSLA": 2.0})
        prices = {"TSLA": 250.0, "SPY": 500.0}
        # 200 shares × $250 × beta2 = $100k → need -200 shares SPY
        qty = _run(dh.calculate_hedge_order({"TSLA": 200}, prices, "SPY"))
        assert qty < 0
        assert abs(qty) == pytest.approx(200, abs=1)

    def test_crypto_beta_assumed_1_2(self):
        """Crypto gets assumed beta of 1.2, not yfinance lookup."""
        dh = _make_hedger()
        prices = {"CRYPTO:BTC/USD": 50000.0, "SPY": 500.0}
        # 1 BTC × $50000 × 1.2 = $60000 → need -120 shares SPY
        qty = _run(dh.calculate_hedge_order({"CRYPTO:BTC/USD": 1}, prices, "SPY"))
        assert qty < 0
        assert abs(qty) == pytest.approx(120, abs=1)

    def test_existing_spy_position_offsets_hedge(self):
        """If we already hold -100 SPY, reduce new hedge by that amount."""
        dh = _make_hedger({"AAPL": 1.0})
        prices = {"AAPL": 200.0, "SPY": 500.0}
        # AAPL: 500 shares × $200 × beta1 = $100k → need -200 SPY
        # But we pass SPY separately as existing spy_exposure (-100 × $500 = -$50k)
        # The hedger will see spy_exposure from its own loop only when SPY is in positions
        # Per the code: hedge_symbol is skipped in the delta calc and spy_exposure is tracked
        # So: target_delta = $100k, spy_required = -$100k, current spy_exposure = -100×$500 = -$50k
        # diff_usd = -$100k - (-$50k) = -$50k → qty = -$50k / $500 = -100 shares
        positions = {"AAPL": 500, "SPY": -100}
        qty = _run(dh.calculate_hedge_order(positions, prices, "SPY"))
        assert qty < 0
        assert abs(qty) == pytest.approx(100, abs=1)

    def test_beta_cache_populated(self):
        dh = _make_hedger({"AAPL": 1.5})
        prices = {"AAPL": 200.0, "SPY": 500.0}
        _run(dh.calculate_hedge_order({"AAPL": 500}, prices, "SPY"))
        assert "AAPL" in dh._beta_cache
        assert dh._beta_cache["AAPL"] == pytest.approx(1.5)

    def test_missing_price_symbol_skipped(self):
        """Symbol with no price → skipped, no crash."""
        dh = _make_hedger({"AAPL": 1.0})
        # AAPL has no price in cache; mdf.get_current_price returns 500 as fallback
        qty = _run(dh.calculate_hedge_order({"AAPL": 500}, {}, "SPY"))
        # Should still compute (uses mdf.get_current_price fallback)
        assert isinstance(qty, (int, float))

    def test_returns_integer(self):
        """SPY qty must be whole shares."""
        dh = _make_hedger({"AAPL": 1.0})
        prices = {"AAPL": 200.0, "SPY": 500.0}
        qty = _run(dh.calculate_hedge_order({"AAPL": 500}, prices, "SPY"))
        assert qty == int(qty)

    def test_beta_fetch_failure_defaults_to_1(self):
        """If yfinance fails, beta defaults to 1.0 gracefully."""
        mdf = MagicMock()
        mdf.get_market_info = MagicMock(side_effect=Exception("network error"))
        mdf.get_current_price = lambda sym: 500.0
        dh = DeltaHedger(market_data_fetcher=mdf)
        prices = {"MSFT": 300.0, "SPY": 500.0}
        # Should not raise; beta defaults to 1.0
        qty = _run(dh.calculate_hedge_order({"MSFT": 200}, prices, "SPY"))
        assert isinstance(qty, (int, float))
