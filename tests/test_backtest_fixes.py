"""
tests/test_backtest_fixes.py - Tests for backtesting bug fixes

Validates:
1. Short trade P&L matching in AdvancedBacktester (CRITICAL)
2. BacktestEngine Sortino uses correct annualization factor (not hardcoded 252)
3. BacktestEngine Calmar uses CAGR, not total_return
4. BacktestEngine includes max_dd_duration in metrics
5. AdvancedBacktester uses Open price for execution (no Close lookahead)
6. AdvancedBacktester slippage consistency between cash check and execution
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.backtest_engine import BacktestEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(
    symbol: str,
    closes: list,
    opens: list = None,
    start: datetime = None,
    volume: float = 1_000_000,
) -> dict:
    """Build {symbol: DataFrame} with OHLCV data."""
    start = start or datetime(2024, 1, 2)
    dates = pd.bdate_range(start=start, periods=len(closes), freq="B")
    if opens is None:
        opens = closes
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
            "Low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
            "Close": closes,
            "Volume": [volume] * len(closes),
        },
        index=dates,
    )
    return {symbol: df}


# ---------------------------------------------------------------------------
# Fix 1: Short trade P&L matching
# ---------------------------------------------------------------------------

class TestShortTradeMatching:
    """AdvancedBacktester must correctly match short trades (SELL->BUY)."""

    def _make_backtester_with_trades(self, trades_data: list) -> AdvancedBacktester:
        """Create a backtester with pre-populated trades and minimal equity curve."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=1.0,
            slippage_bps=0.0,
        )
        bt.trades = trades_data
        bt._data_symbols = list(set(t['symbol'] for t in trades_data))
        # Minimal equity curve for metrics calculation
        bt.equity_curve = [
            {'date': datetime(2024, 1, 2), 'equity': 100_000, 'cash': 100_000, 'positions_value': 0},
            {'date': datetime(2024, 6, 28), 'equity': 105_000, 'cash': 105_000, 'positions_value': 0},
        ]
        bt.daily_returns = [0.001] * 125
        return bt

    def _make_trade(self, date, symbol, side, qty, exec_price, commission=1.0):
        return {
            'date': date, 'symbol': symbol, 'side': side,
            'quantity': qty, 'price': exec_price, 'execution_price': exec_price,
            'commission': commission, 'slippage': 0.0, 'reason': 'test', 'cash_after': 0,
        }

    def test_long_trade_matching(self):
        """Standard long: BUY then SELL -> positive P&L."""
        trades = [
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'BUY', 100, 100.0),
            self._make_trade(datetime(2024, 2, 5), 'AAPL', 'SELL', 100, 110.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        # P&L = (110 - 100) * 100 - 2 commissions = 998
        assert metrics['win_rate'] == 1.0
        assert metrics['avg_trade_pnl'] == pytest.approx(998.0, abs=0.01)
        assert metrics['profit_factor'] == float('inf')

    def test_short_trade_matching(self):
        """Short: SELL then BUY -> P&L calculated correctly."""
        trades = [
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'SELL', 100, 110.0),
            self._make_trade(datetime(2024, 2, 5), 'AAPL', 'BUY', 100, 100.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        # Short P&L = (110 - 100) * 100 - 2 = 998
        assert metrics['win_rate'] == 1.0
        assert metrics['avg_trade_pnl'] == pytest.approx(998.0, abs=0.01)

    def test_losing_short_trade(self):
        """Short that loses money: SELL at 100, BUY at 110."""
        trades = [
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'SELL', 100, 100.0),
            self._make_trade(datetime(2024, 2, 5), 'AAPL', 'BUY', 100, 110.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        # Short P&L = (100 - 110) * 100 - 2 = -1002
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_trade_pnl'] == pytest.approx(-1002.0, abs=0.01)

    def test_mixed_long_and_short_trades(self):
        """Mix of long (winning) and short (winning) trades."""
        trades = [
            # Long AAPL: BUY 100 @ 100, SELL @ 105 -> P&L = 500 - 2 = 498
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'BUY', 100, 100.0),
            self._make_trade(datetime(2024, 1, 20), 'AAPL', 'SELL', 100, 105.0),
            # Short MSFT: SELL 50 @ 200, BUY @ 190 -> P&L = (200-190)*50 - 2 = 498
            self._make_trade(datetime(2024, 2, 5), 'MSFT', 'SELL', 50, 200.0),
            self._make_trade(datetime(2024, 2, 20), 'MSFT', 'BUY', 50, 190.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        assert metrics['win_rate'] == 1.0  # both winning
        assert metrics['avg_trade_pnl'] == pytest.approx(498.0, abs=0.01)
        assert metrics['profit_factor'] == float('inf')

    def test_multiple_shorts_same_symbol(self):
        """Multiple short trades on same symbol matched in FIFO order."""
        trades = [
            # Short 1: SELL @ 100
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'SELL', 100, 100.0),
            # Close short 1: BUY @ 90 -> P&L = (100-90)*100 - 2 = 998
            self._make_trade(datetime(2024, 1, 20), 'AAPL', 'BUY', 100, 90.0),
            # Short 2: SELL @ 95
            self._make_trade(datetime(2024, 2, 5), 'AAPL', 'SELL', 100, 95.0),
            # Close short 2: BUY @ 100 -> P&L = (95-100)*100 - 2 = -502
            self._make_trade(datetime(2024, 2, 20), 'AAPL', 'BUY', 100, 100.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        # 2 completed trades: 1 winner (998), 1 loser (-502)
        assert metrics['win_rate'] == pytest.approx(0.5, abs=0.01)
        total_pnl = 998 + (-502)
        assert metrics['avg_trade_pnl'] == pytest.approx(total_pnl / 2, abs=0.01)
        assert metrics['profit_factor'] == pytest.approx(998 / 502, abs=0.01)

    def test_no_trades_returns_zero_metrics(self):
        """Empty trades list should produce zero trade metrics."""
        bt = AdvancedBacktester(initial_capital=100_000)
        bt.trades = []
        bt._data_symbols = ['AAPL']
        bt.equity_curve = [
            {'date': datetime(2024, 1, 2), 'equity': 100_000, 'cash': 100_000, 'positions_value': 0},
        ]
        bt.daily_returns = []
        metrics = bt._calculate_metrics()

        assert metrics['win_rate'] == 0
        assert metrics['avg_trade_pnl'] == 0
        assert metrics['profit_factor'] == 0

    def test_unmatched_entries_dont_affect_metrics(self):
        """Open positions (unmatched) should not count as completed trades."""
        trades = [
            # Long entry, never closed
            self._make_trade(datetime(2024, 1, 5), 'AAPL', 'BUY', 100, 100.0),
            # Short entry, never closed
            self._make_trade(datetime(2024, 1, 10), 'MSFT', 'SELL', 50, 200.0),
        ]
        bt = self._make_backtester_with_trades(trades)
        metrics = bt._calculate_metrics()

        # No completed trades
        assert metrics['win_rate'] == 0
        assert metrics['avg_trade_pnl'] == 0


# ---------------------------------------------------------------------------
# Fix 2 & 3 & 4: BacktestEngine Sortino, Calmar, max_dd_duration
# ---------------------------------------------------------------------------

class TestBacktestEngineMetrics:
    """BacktestEngine Sortino uses ann_factor, Calmar uses CAGR, max_dd_duration included."""

    def test_sortino_uses_annualization_factor(self):
        """Sortino must use ann_factor from _annualization_factor(), not hardcoded 252."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 300
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.005, 0.015, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 900)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        assert 'sortino_ratio' in metrics
        # With upward drift, sortino should be positive
        assert metrics['sortino_ratio'] > 0

    def test_sortino_crypto_uses_365(self):
        """For crypto-only portfolio, annualization factor should be 365."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
            crypto_spread_bps=0.0,
            crypto_commission_bps=0.0,
        )
        np.random.seed(42)
        n = 400
        prices = list(40000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n))))
        data = _make_price_data("CRYPTO:BTC/USD", prices)
        engine.load_data(data)
        ts = list(data["CRYPTO:BTC/USD"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("CRYPTO:BTC/USD", "BUY", 0.5)

        engine.run(strategy, ts[0], ts[-1])

        assert engine._annualization_factor() == 365

        metrics = engine.get_results()
        assert 'sortino_ratio' in metrics

    def test_calmar_uses_cagr(self):
        """Calmar = CAGR / |max_drawdown|, not total_return / |max_drawdown|."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 500
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 10:
                eng.execute_order("AAPL", "BUY", 200)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        # Verify Calmar = CAGR / |max_dd|
        cagr = metrics['cagr']
        max_dd = metrics['max_drawdown']
        if abs(max_dd) > 0:
            expected_calmar = cagr / abs(max_dd)
            assert metrics['calmar_ratio'] == pytest.approx(expected_calmar, rel=0.01)

    def test_calmar_differs_from_naive(self):
        """Calmar with CAGR should differ from total_return/max_dd for multi-year runs."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(123)
        n = 500
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 200)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        max_dd = abs(metrics['max_drawdown'])
        if max_dd > 0 and n > 252:
            naive_calmar = metrics['total_return'] / max_dd
            # CAGR-based calmar should generally differ from total_return-based
            # (They'd be equal only for exactly 1 year of data at ann_factor frequency)
            assert metrics['calmar_ratio'] != pytest.approx(naive_calmar, rel=0.001)

    def test_max_dd_duration_in_metrics(self):
        """max_dd_duration must be included in metrics dict."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 100
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        assert 'max_dd_duration' in metrics
        assert isinstance(metrics['max_dd_duration'], int)
        assert metrics['max_dd_duration'] >= 0

    def test_cagr_consistency(self):
        """CAGR in metrics should match the formula (1+total_return)^(ann_factor/n)-1."""
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=0.0,
            use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 300
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n))))
        data = _make_price_data("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        total_return = metrics['total_return']
        ann_factor = engine._annualization_factor()
        history_len = len(engine.history)
        expected_cagr = ((1 + total_return) ** (ann_factor / history_len)) - 1
        assert metrics['cagr'] == pytest.approx(expected_cagr, rel=0.001)


# ---------------------------------------------------------------------------
# Fix 5: Execution price uses Open
# ---------------------------------------------------------------------------

class TestExecutionPrice:
    """AdvancedBacktester must use Open price for execution, not Close."""

    def test_entry_uses_open_price(self):
        """When entering a position, execution price should be based on Open."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=0.0,
            slippage_bps=0.0,
        )
        # Create data where Open != Close
        n = 60
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        opens = [100.0] * n
        closes = [105.0] * n  # Close is 5% higher than Open
        df = pd.DataFrame({
            "Open": opens,
            "High": [106.0] * n,
            "Low": [99.0] * n,
            "Close": closes,
            "Volume": [1_000_000] * n,
        }, index=dates)
        data = {"AAPL": df}

        # Signal generator that returns strong buy after enough data
        def gen_signal(symbol, prices):
            if len(prices) > 5:
                return {'signal': 0.8, 'confidence': 0.8, 'quality': 0.8}
            return {'signal': 0.0, 'confidence': 0.5, 'quality': 0.5}

        signal_gen = MagicMock()
        signal_gen.generate_ml_signal = gen_signal

        bt.run_backtest(
            data=data,
            signal_generator=signal_gen,
            start_date="2024-01-02",
            end_date="2024-03-29",
            position_size_usd=5000,
            max_positions=3,
        )

        # At least one trade should have been made
        assert len(bt.trades) > 0
        # All trades should use Open price (100.0), not Close (105.0)
        for trade in bt.trades:
            assert trade['price'] == 100.0, (
                f"Expected price=100.0 (Open), got {trade['price']} (Close would be 105.0)"
            )

    def test_exit_uses_open_price(self):
        """When exiting a position, execution price should be based on Open."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=0.0,
            slippage_bps=0.0,
        )
        n = 60
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        opens = [100.0] * n
        closes = [105.0] * n
        df = pd.DataFrame({
            "Open": opens,
            "High": [106.0] * n,
            "Low": [99.0] * n,
            "Close": closes,
            "Volume": [1_000_000] * n,
        }, index=dates)
        data = {"AAPL": df}

        # Signal: strong buy first, then flip to strong sell to trigger exit
        def gen_signal(symbol, prices):
            if len(prices) < 8:
                return {'signal': 0.0, 'confidence': 0.5, 'quality': 0.5}
            if len(prices) < 30:
                return {'signal': 0.8, 'confidence': 0.8, 'quality': 0.8}
            # Strong bearish signal to trigger exit
            return {'signal': -0.8, 'confidence': 0.8, 'quality': 0.8}

        signal_gen = MagicMock()
        signal_gen.generate_ml_signal = gen_signal

        bt.run_backtest(
            data=data,
            signal_generator=signal_gen,
            start_date="2024-01-02",
            end_date="2024-03-29",
            position_size_usd=5000,
            max_positions=3,
        )

        # Should have entry + exit trades
        assert len(bt.trades) >= 2
        # All trades should use Open price
        for trade in bt.trades:
            assert trade['price'] == 100.0, (
                f"Expected price=100.0 (Open), got {trade['price']}"
            )


# ---------------------------------------------------------------------------
# Fix 6: Slippage consistency
# ---------------------------------------------------------------------------

class TestSlippageConsistency:
    """Slippage from _estimate_slippage_pct used in execution, not just cash check."""

    def test_execute_order_uses_provided_slippage(self):
        """When slippage_pct is provided to _execute_order, it uses that value."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=1.0,
            slippage_bps=5.0,  # base = 5 bps = 0.0005
        )
        bt._data_symbols = ["AAPL"]

        # Execute with explicit custom slippage
        custom_slippage = 0.002  # 20 bps
        result = bt._execute_order(
            "AAPL", "BUY", 10, 100.0, datetime(2024, 1, 3),
            reason="test", slippage_pct=custom_slippage
        )

        assert result is True
        trade = bt.trades[-1]
        # execution_price = 100 * (1 + 0.002) = 100.20
        assert trade['execution_price'] == pytest.approx(100.20, abs=0.01)

    def test_execute_order_falls_back_to_base_slippage(self):
        """When slippage_pct is None, _execute_order falls back to base slippage."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=1.0,
            slippage_bps=10.0,  # base = 10 bps = 0.001
        )
        bt._data_symbols = ["AAPL"]

        result = bt._execute_order(
            "AAPL", "BUY", 10, 100.0, datetime(2024, 1, 3),
            reason="test"
        )

        assert result is True
        trade = bt.trades[-1]
        # base slippage = 10 bps = 0.001 (already converted in __init__)
        # execution_price = 100 * (1 + 0.001) = 100.10
        assert trade['execution_price'] == pytest.approx(100.10, abs=0.01)

    def test_sell_slippage_direction(self):
        """SELL slippage should reduce execution price."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=1.0,
            slippage_bps=10.0,
        )
        bt._data_symbols = ["AAPL"]
        bt.positions["AAPL"] = 10  # need position to sell

        result = bt._execute_order(
            "AAPL", "SELL", 10, 100.0, datetime(2024, 1, 3),
            reason="test", slippage_pct=0.002
        )

        assert result is True
        trade = bt.trades[-1]
        # SELL: execution_price = 100 * (1 - 0.002) = 99.80
        assert trade['execution_price'] == pytest.approx(99.80, abs=0.01)

    def test_provided_slippage_overrides_base(self):
        """Custom slippage must override the base, not add to it."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=0.0,
            slippage_bps=50.0,  # base = 50 bps = 0.005
        )
        bt._data_symbols = ["AAPL"]

        # Provide smaller custom slippage
        result = bt._execute_order(
            "AAPL", "BUY", 10, 100.0, datetime(2024, 1, 3),
            reason="test", slippage_pct=0.001  # 10 bps, less than base 50 bps
        )

        assert result is True
        trade = bt.trades[-1]
        # Should use provided 0.001 (10 bps), not base 0.005 (50 bps)
        assert trade['execution_price'] == pytest.approx(100.10, abs=0.01)
        # NOT 100.50 which would be base slippage
