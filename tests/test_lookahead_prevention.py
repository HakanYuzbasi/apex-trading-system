"""
tests/test_lookahead_prevention.py - Lookahead Prevention Tests

Validates:
- _DataView only exposes data up to current timestamp
- Strategy cannot see future prices
- execute_order() queues for next bar (t+1), not current bar
- Pending orders execute with next bar's price
"""

import pytest
import pandas as pd
from datetime import datetime

from backtesting.backtest_engine import BacktestEngine, _DataView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(
    symbol: str,
    prices: list[float],
    start: datetime | None = None,
    volume: float = 1_000_000,
) -> dict[str, pd.DataFrame]:
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
# Tests: _DataView
# ---------------------------------------------------------------------------

class TestDataView:
    """_DataView must slice data to prevent lookahead."""

    def test_getitem_slices_to_current_time(self):
        """data_view[symbol] should only contain rows up to current_time."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        data = _make_price_data("AAPL", prices)
        ts = list(data["AAPL"].index)

        # View at time t=2 (third bar)
        view = _DataView(data, ts[2])
        visible = view["AAPL"]

        assert len(visible) == 3  # bars 0, 1, 2
        assert visible.index[-1] == ts[2]
        # Future bars should not be present
        assert ts[3] not in visible.index
        assert ts[4] not in visible.index

    def test_getitem_at_first_bar(self):
        """At the very first bar, only one row should be visible."""
        prices = [100.0, 200.0, 300.0]
        data = _make_price_data("AAPL", prices)
        ts = list(data["AAPL"].index)

        view = _DataView(data, ts[0])
        visible = view["AAPL"]

        assert len(visible) == 1
        assert visible["Close"].iloc[0] == 100.0

    def test_get_method_slices(self):
        """view.get() should also slice data."""
        prices = [10.0, 20.0, 30.0]
        data = _make_price_data("AAPL", prices)
        ts = list(data["AAPL"].index)

        view = _DataView(data, ts[1])
        result = view.get("AAPL")

        assert result is not None
        assert len(result) == 2
        assert result["Close"].iloc[-1] == 20.0

    def test_get_returns_default_for_missing(self):
        """view.get() should return default for missing symbols."""
        data = _make_price_data("AAPL", [100.0])
        ts = list(data["AAPL"].index)

        view = _DataView(data, ts[0])
        assert view.get("MSFT") is None
        assert view.get("MSFT", "fallback") == "fallback"

    def test_items_slices_all_symbols(self):
        """view.items() must slice every symbol."""
        data_aapl = _make_price_data("AAPL", [100.0, 200.0, 300.0])
        data_msft = _make_price_data("MSFT", [50.0, 60.0, 70.0])
        combined = {**data_aapl, **data_msft}
        ts = list(data_aapl["AAPL"].index)

        view = _DataView(combined, ts[1])

        for sym, df in view.items():
            assert len(df) == 2, f"Symbol {sym} should have exactly 2 rows at t=1"

    def test_contains_works(self):
        """'in' operator should work on _DataView."""
        data = _make_price_data("AAPL", [100.0])
        ts = list(data["AAPL"].index)
        view = _DataView(data, ts[0])

        assert "AAPL" in view
        assert "MSFT" not in view

    def test_keys_returns_all_symbols(self):
        data_aapl = _make_price_data("AAPL", [100.0])
        data_msft = _make_price_data("MSFT", [50.0])
        combined = {**data_aapl, **data_msft}
        ts = list(data_aapl["AAPL"].index)

        view = _DataView(combined, ts[0])
        assert set(view.keys()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# Tests: order execution timing (signal at t, fill at t+1)
# ---------------------------------------------------------------------------

class TestOrderTiming:
    """Orders submitted via execute_order() must fill at the next bar, not current."""

    def test_execute_order_queues_not_fills(self):
        """execute_order() should not immediately create a trade."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            enable_stop_management=False,
            max_participation_rate=0,
        )
        data = _make_price_data("AAPL", [100.0, 105.0, 110.0])
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine.execute_order("AAPL", "BUY", 10)

        # No trade should exist yet
        assert len(engine.trades) == 0
        assert len(engine.pending_orders) == 1

    def test_pending_order_fills_at_next_bar(self):
        """Pending order should execute at the next _process_step."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            enable_stop_management=False,
            max_participation_rate=0,
        )
        # Price goes 100 -> 110
        data = _make_price_data("AAPL", [100.0, 110.0, 120.0])
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        # At t=0, queue a buy
        engine.current_time = ts[0]
        engine.execute_order("AAPL", "BUY", 10)

        # Advance to t=1 (as run() does) and process: pending fills at t=1's price
        engine.current_time = ts[1]
        engine._process_step(ts[1])

        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(110.0, abs=0.01)
        assert engine.trades[0].timestamp == ts[1]

    def test_strategy_sees_only_past_data(self):
        """During backtest run, strategy function receives only past data."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            enable_stop_management=False,
            max_participation_rate=0,
        )
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        seen_lengths = []

        def strategy(eng, timestamp):
            # eng.data is a _DataView at this point
            df = eng.data["AAPL"]
            seen_lengths.append(len(df))

        engine.run(strategy, ts[0], ts[-1])

        # At t=0 strategy sees 1 bar, at t=1 sees 2, etc.
        assert seen_lengths == [1, 2, 3, 4, 5]

    def test_full_run_order_executes_next_bar(self):
        """In a full run(), order placed at bar t fills at bar t+1."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            enable_stop_management=False,
            max_participation_rate=0,
        )
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 1:
                eng.execute_order("AAPL", "BUY", 10)

        engine.run(strategy, ts[0], ts[-1])
        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(110.0, abs=0.01)
        assert engine.trades[0].timestamp == ts[2]
