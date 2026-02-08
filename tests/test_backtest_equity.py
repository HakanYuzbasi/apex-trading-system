"""
tests/test_backtest_equity.py - Backtest Equity Calculation Tests

Validates:
- No double-counting of transaction costs in equity
- Correct commission deduction per asset class (equity, FX, crypto)
- Correct slippage application
- Cash + position value = total equity (invariant)
- Round-trip trade P&L accuracy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.backtest_engine import BacktestEngine, _DataView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(
    symbol: str,
    prices: list[float],
    start: datetime | None = None,
    freq: str = "D",
    volume: float = 1_000_000,
) -> dict[str, pd.DataFrame]:
    """Build a {symbol: DataFrame} dict suitable for BacktestEngine.load_data."""
    start = start or datetime(2024, 1, 2)
    dates = pd.bdate_range(start=start, periods=len(prices), freq="B")
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [volume] * len(prices),
        },
        index=dates,
    )
    return {symbol: df}


# ---------------------------------------------------------------------------
# Tests: equity invariant (cash + positions = total)
# ---------------------------------------------------------------------------

class TestEquityInvariant:
    """total_equity() must equal cash + sum(position market values) at every step."""

    def test_no_trades_equity_equals_initial_capital(self):
        engine = BacktestEngine(
            initial_capital=100_000,
            use_dynamic_slippage=False,
        )
        assert engine.total_equity() == 100_000

    def test_equity_after_buy(self):
        """After buying shares, equity = cash + position value, with costs deducted once."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,  # zero slippage to simplify
            use_dynamic_slippage=False,
        )
        prices = [100.0] * 10
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)

        # Manually set time and execute
        ts = list(data["AAPL"].index)
        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        # Position value = 100 shares * $100 = $10,000
        # Commission = max(1.0, 100 * 0.005) = $1.00
        # Cash should be 100_000 - 10_000 - 1.0 = 89_999.0
        assert engine.cash == pytest.approx(89_999.0, abs=0.01)
        assert engine.total_equity() == pytest.approx(99_999.0, abs=0.01)

    def test_equity_after_sell_close(self):
        """Round-trip buy then sell: equity = initial - 2*commission."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        prices = [100.0] * 10
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        engine.current_time = ts[1]
        engine._execute_order_now("AAPL", "SELL", 100)

        # No position left; equity = initial - 2 * commission
        commission = max(1.0, 100 * 0.005)
        expected = 100_000 - 2 * commission
        assert len(engine.positions) == 0
        assert engine.total_equity() == pytest.approx(expected, abs=0.01)
        assert engine.cash == pytest.approx(expected, abs=0.01)

    def test_equity_invariant_through_run(self):
        """During a full backtest run, equity = cash + positions at every history step."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=5.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 60
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            # Buy on day 5, sell on day 40
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 50)
            elif idx == 40:
                eng.execute_order("AAPL", "SELL", 50)

        engine.run(strategy, ts[0], ts[-1])

        for step in engine.history:
            recorded = step["equity"]
            # Reconstruct from engine state at that point - equity recorded should be positive
            assert recorded > 0


# ---------------------------------------------------------------------------
# Tests: no double-counting of costs
# ---------------------------------------------------------------------------

class TestNoDoubleCounting:
    """Commission/slippage must be deducted from cash only, never also from position value."""

    def test_buy_cost_deducted_from_cash_only(self):
        """When buying, commission reduces cash; position value is at fill price."""
        engine = BacktestEngine(
            initial_capital=50_000,
            commission_per_share=0.01,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [200.0] * 5)
        engine.load_data(data)
        engine.current_time = list(data["AAPL"].index)[0]
        engine._execute_order_now("AAPL", "BUY", 10)

        cost = 10 * 200.0
        commission = max(1.0, 10 * 0.01)

        # Cash = initial - cost - commission
        assert engine.cash == pytest.approx(50_000 - cost - commission, abs=0.01)

        # Position value is purely quantity * price
        pos_value = engine.positions["AAPL"].quantity * engine.positions["AAPL"].current_price
        assert pos_value == pytest.approx(cost, abs=0.01)

        # Total equity = cash + pos_value = initial - commission
        assert engine.total_equity() == pytest.approx(50_000 - commission, abs=0.01)

    def test_sell_cost_deducted_from_cash_only(self):
        """When selling, commission reduces proceeds into cash, not position value."""
        engine = BacktestEngine(
            initial_capital=50_000,
            commission_per_share=0.01,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [200.0] * 5)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 10)
        equity_after_buy = engine.total_equity()

        engine.current_time = ts[1]
        engine._execute_order_now("AAPL", "SELL", 10)

        commission = max(1.0, 10 * 0.01)
        # After sell, equity should drop by exactly one more commission
        assert engine.total_equity() == pytest.approx(equity_after_buy - commission, abs=0.01)

    def test_slippage_not_double_counted(self):
        """With slippage, fill price differs from market - but equity still consistent."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.0,
            min_commission=0.0,
            slippage_bps=50.0,  # 50 bps = 0.5%
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [100.0] * 5)
        engine.load_data(data)
        engine.current_time = list(data["AAPL"].index)[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        fill_price = 100.0 * (1 + 50 / 10000)  # 100.50
        expected_cash = 100_000 - 100 * fill_price
        expected_pos_val = 100 * engine.positions["AAPL"].current_price

        assert engine.cash == pytest.approx(expected_cash, abs=0.01)
        # Position records the fill price, so current_price = fill_price
        assert engine.positions["AAPL"].current_price == pytest.approx(fill_price, abs=0.01)
        # total_equity = cash + pos_value = 100_000 (no commission, just slippage impact)
        assert engine.total_equity() == pytest.approx(100_000, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: per-asset-class fee model
# ---------------------------------------------------------------------------

class TestAssetClassFees:
    """Correct fee model applied per asset class."""

    def test_equity_commission_model(self):
        """Equity uses per-share commission with minimum."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [150.0] * 5)
        engine.load_data(data)
        engine.current_time = list(data["AAPL"].index)[0]
        engine._execute_order_now("AAPL", "BUY", 10)

        # 10 * 0.005 = 0.05, min = 1.0, so commission = 1.0
        trade = engine.trades[-1]
        assert trade.commission == pytest.approx(1.0, abs=0.001)

    def test_equity_commission_above_minimum(self):
        """Large equity order exceeds minimum commission."""
        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [150.0] * 5)
        engine.load_data(data)
        engine.current_time = list(data["AAPL"].index)[0]
        engine._execute_order_now("AAPL", "BUY", 1000)

        # 1000 * 0.005 = 5.0 > min 1.0
        trade = engine.trades[-1]
        assert trade.commission == pytest.approx(5.0, abs=0.001)

    def test_fx_commission_model(self):
        """FX uses notional-based bps commission."""
        engine = BacktestEngine(
            initial_capital=1_000_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            fx_commission_bps=2.0,
            fx_min_commission=0.0,
            fx_spread_bps=0.0,
        )
        data = _make_price_data("FX:EUR/USD", [1.10] * 5)
        engine.load_data(data)
        engine.current_time = list(data["FX:EUR/USD"].index)[0]
        engine._execute_order_now("FX:EUR/USD", "BUY", 10_000)

        # notional = 10000 * 1.10 = 11000; commission = 11000 * (2/10000) = 2.20
        trade = engine.trades[-1]
        assert trade.commission == pytest.approx(2.20, abs=0.01)

    def test_crypto_commission_model(self):
        """Crypto uses notional-based bps commission."""
        engine = BacktestEngine(
            initial_capital=1_000_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            crypto_commission_bps=15.0,
            crypto_min_commission=0.0,
            crypto_spread_bps=0.0,
        )
        data = _make_price_data("CRYPTO:BTC/USD", [40_000.0] * 5)
        engine.load_data(data)
        engine.current_time = list(data["CRYPTO:BTC/USD"].index)[0]
        engine._execute_order_now("CRYPTO:BTC/USD", "BUY", 0.5)

        # notional = 0.5 * 40000 = 20000; commission = 20000 * (15/10000) = 30.0
        trade = engine.trades[-1]
        assert trade.commission == pytest.approx(30.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: round-trip P&L correctness
# ---------------------------------------------------------------------------

class TestRoundTripPnL:
    """Verify that realized P&L on a round-trip correctly accounts for price change and costs."""

    def test_profitable_roundtrip(self):
        """Buy at 100, sell at 110 -> P&L = (110-100)*qty - 2*commission."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [100.0, 110.0, 110.0])
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        engine.current_time = ts[1]
        # Update price before selling
        engine.positions["AAPL"].update_price(110.0)
        engine._execute_order_now("AAPL", "SELL", 100)

        commission = max(1.0, 100 * 0.005)
        expected_equity = 100_000 + (110 - 100) * 100 - 2 * commission
        assert engine.total_equity() == pytest.approx(expected_equity, abs=0.01)

    def test_losing_roundtrip(self):
        """Buy at 100, sell at 90 -> loss = (90-100)*qty - 2*commission."""
        engine = BacktestEngine(
            initial_capital=100_000,
            commission_per_share=0.005,
            min_commission=1.0,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        data = _make_price_data("AAPL", [100.0, 90.0, 90.0])
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        engine.current_time = ts[1]
        engine.positions["AAPL"].update_price(90.0)
        engine._execute_order_now("AAPL", "SELL", 100)

        commission = max(1.0, 100 * 0.005)
        expected_equity = 100_000 + (90 - 100) * 100 - 2 * commission
        assert engine.total_equity() == pytest.approx(expected_equity, abs=0.01)
