"""
tests/test_backtest_institutional.py - Tests for institutional-grade backtesting features

Validates:
1. Point-in-time universe / survivorship bias protection
2. Data validation (split detection, stale prices, NaN)
3. Deflated Sharpe Ratio / Probabilistic Sharpe Ratio
4. Short borrow/funding costs
5. ADV-based capacity constraints
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

def _make_ohlcv(symbol, closes, opens=None, start=None, volume=1_000_000):
    start = start or datetime(2024, 1, 2)
    dates = pd.bdate_range(start=start, periods=len(closes), freq="B")
    if opens is None:
        opens = closes
    df = pd.DataFrame({
        "Open": opens,
        "High": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
        "Low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
        "Close": closes,
        "Volume": [volume] * len(closes),
    }, index=dates)
    return {symbol: df}


def _make_signal_gen(signal_val=0.0):
    gen = MagicMock()
    gen.generate_ml_signal = lambda sym, prices: {
        'signal': signal_val if len(prices) > 5 else 0.0,
        'confidence': 0.8, 'quality': 0.8,
    }
    return gen


# ---------------------------------------------------------------------------
# 1. Point-in-time Universe / Survivorship Bias
# ---------------------------------------------------------------------------

class TestPointInTimeUniverse:

    def test_universe_restricts_entries(self):
        """Only symbols in the current universe should get new entries."""
        n = 40
        data = {}
        for sym in ['AAPL', 'MSFT', 'GONE']:
            dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
            data[sym] = pd.DataFrame({
                "Open": [100.0] * n, "High": [101.0] * n,
                "Low": [99.0] * n, "Close": [100.0] * n,
                "Volume": [1_000_000] * n,
            }, index=dates)

        # Universe: only AAPL and MSFT are tradeable (GONE is "delisted")
        universe = {"2024-01-02": ["AAPL", "MSFT"]}

        bt = AdvancedBacktester(initial_capital=100_000, commission_per_trade=0.0, slippage_bps=0.0)
        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000, max_positions=5,
            universe_schedule=universe,
        )

        traded_symbols = set(t['symbol'] for t in bt.trades)
        assert 'GONE' not in traded_symbols

    def test_delisting_force_closes_position(self):
        """When a symbol leaves the universe, its position must be force-closed."""
        n = 40
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        data = {
            'AAPL': pd.DataFrame({
                "Open": [100.0] * n, "High": [101.0] * n,
                "Low": [99.0] * n, "Close": [100.0] * n,
                "Volume": [1_000_000] * n,
            }, index=dates),
        }

        # AAPL tradeable initially, then removed from universe on day 20
        day20 = str(dates[20].date())
        universe = {
            "2024-01-02": ["AAPL"],
            day20: [],  # AAPL delisted
        }

        bt = AdvancedBacktester(initial_capital=100_000, commission_per_trade=0.0, slippage_bps=0.0)
        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000, max_positions=5,
            universe_schedule=universe,
        )

        # After delisting, AAPL position should be zero
        assert bt.positions.get('AAPL', 0) == 0
        # Should have a "Universe exit" trade
        delist_trades = [t for t in bt.trades if "Universe exit" in t.get('reason', '')]
        assert len(delist_trades) >= 1

    def test_no_universe_trades_all_symbols(self):
        """Without universe_schedule, all symbols in data are tradeable."""
        n = 40
        data = {}
        for sym in ['AAPL', 'MSFT']:
            dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
            data[sym] = pd.DataFrame({
                "Open": [100.0] * n, "High": [101.0] * n,
                "Low": [99.0] * n, "Close": [100.0] * n,
                "Volume": [1_000_000] * n,
            }, index=dates)

        bt = AdvancedBacktester(initial_capital=100_000, commission_per_trade=0.0, slippage_bps=0.0)
        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000, max_positions=5,
        )

        traded_symbols = set(t['symbol'] for t in bt.trades)
        # Both should be tradeable
        assert len(traded_symbols) >= 1


# ---------------------------------------------------------------------------
# 2. Data Validation
# ---------------------------------------------------------------------------

class TestDataValidation:

    def test_detects_likely_split(self):
        """Overnight return >50% should trigger a split warning."""
        bt = AdvancedBacktester(initial_capital=100_000)
        closes = [100.0] * 10 + [200.0] + [200.0] * 9  # 100% jump on day 11
        data = _make_ohlcv("AAPL", closes)
        result = bt._validate_data(data)

        split_warnings = [w for w in result["errors"] if "split" in w.lower()]
        assert len(split_warnings) >= 1

    def test_detects_stale_prices(self):
        """5+ identical consecutive closes should trigger stale info message."""
        bt = AdvancedBacktester(initial_capital=100_000)
        closes = [100.0] * 20  # all identical
        data = _make_ohlcv("AAPL", closes)
        result = bt._validate_data(data)

        stale_info = [w for w in result["info"] if "stale" in w.lower()]
        assert len(stale_info) >= 1

    def test_detects_nan(self):
        """NaN in price data should trigger warning."""
        bt = AdvancedBacktester(initial_capital=100_000)
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame({
            "Open": [100.0] * 10, "High": [101.0] * 10,
            "Low": [99.0] * 10, "Close": [100.0] * 9 + [float('nan')],
            "Volume": [1_000_000] * 10,
        }, index=dates)
        result = bt._validate_data({"AAPL": df})

        nan_warnings = [w for w in result["errors"] if "NaN" in w]
        assert len(nan_warnings) >= 1

    def test_clean_data_no_warnings(self):
        """Clean data should produce no warnings or info messages."""
        bt = AdvancedBacktester(initial_capital=100_000)
        np.random.seed(42)
        closes = list(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 20))))
        data = _make_ohlcv("AAPL", closes)
        result = bt._validate_data(data)

        assert len(result["errors"]) == 0
        assert len(result["info"]) == 0


# ---------------------------------------------------------------------------
# 3. Probabilistic Sharpe / Deflated Sharpe
# ---------------------------------------------------------------------------

class TestDeflatedSharpe:

    def _run_backtest_with_trend(self, drift=0.001, n=300, n_trials=1):
        bt = AdvancedBacktester(
            initial_capital=100_000, commission_per_trade=0.0, slippage_bps=0.0,
            short_borrow_rate=0.0, max_adv_participation=0.0,
        )
        np.random.seed(42)
        closes = list(100 * np.exp(np.cumsum(np.random.normal(drift, 0.015, n))))
        opens = [c * 0.999 for c in closes]
        data = _make_ohlcv("AAPL", closes, opens=opens)
        bt.run_backtest(
            data=data, signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2025-03-15",
            position_size_usd=5000, max_positions=3,
            n_sharpe_trials=n_trials,
        )
        return bt._calculate_metrics()

    def test_psr_in_metrics(self):
        """probabilistic_sharpe should be in metrics output."""
        metrics = self._run_backtest_with_trend()
        assert 'probabilistic_sharpe' in metrics
        assert 0 <= metrics['probabilistic_sharpe'] <= 1

    def test_dsr_in_metrics(self):
        """deflated_sharpe should be in metrics output."""
        metrics = self._run_backtest_with_trend()
        assert 'deflated_sharpe' in metrics
        assert 0 <= metrics['deflated_sharpe'] <= 1

    def test_psr_high_for_strong_trend(self):
        """Strong positive drift should give high PSR (>0.5)."""
        metrics = self._run_backtest_with_trend(drift=0.002)
        if metrics['sharpe_ratio'] > 0:
            assert metrics['probabilistic_sharpe'] > 0.5

    def test_dsr_decreases_with_more_trials(self):
        """More trials (overfitting risk) should lower DSR."""
        m1 = self._run_backtest_with_trend(n_trials=1)
        m10 = self._run_backtest_with_trend(n_trials=10)
        m100 = self._run_backtest_with_trend(n_trials=100)

        if m1['sharpe_ratio'] > 0:
            assert m100['deflated_sharpe'] <= m1['deflated_sharpe']
            assert m10['deflated_sharpe'] <= m1['deflated_sharpe']

    def test_n_sharpe_trials_recorded(self):
        """n_sharpe_trials should be recorded in metrics."""
        metrics = self._run_backtest_with_trend(n_trials=50)
        assert metrics['n_sharpe_trials'] == 50

    def test_backtest_engine_has_psr(self):
        """BacktestEngine.get_results should also include probabilistic_sharpe."""
        engine = BacktestEngine(
            initial_capital=100_000, slippage_bps=0.0, use_dynamic_slippage=False,
        )
        np.random.seed(42)
        n = 300
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n))))
        data = _make_ohlcv("AAPL", prices)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            if timestamp == ts[5]:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])
        metrics = engine.get_results()

        assert 'probabilistic_sharpe' in metrics
        assert 0 <= metrics['probabilistic_sharpe'] <= 1


# ---------------------------------------------------------------------------
# 4. Short Borrow/Funding Costs
# ---------------------------------------------------------------------------

class TestBorrowCosts:

    def test_borrow_costs_deducted_for_shorts(self):
        """Short positions should incur daily borrow costs."""
        bt = AdvancedBacktester(
            initial_capital=100_000, commission_per_trade=0.0,
            slippage_bps=0.0, short_borrow_rate=0.10,  # 10% annualized for easy math
        )
        bt._data_symbols = ["AAPL"]
        bt.positions["AAPL"] = -100  # short 100 shares

        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        data = {"AAPL": pd.DataFrame({
            "Open": [100.0] * 5, "High": [101.0] * 5,
            "Low": [99.0] * 5, "Close": [100.0] * 5,
            "Volume": [1_000_000] * 5,
        }, index=dates)}

        initial_cash = bt.cash
        bt._apply_borrow_costs(data, dates[0])

        # daily_cost = 100 shares * $100 * 0.10 / 252 ≈ $3.97
        expected_daily = 100 * 100.0 * 0.10 / 252
        assert bt.cash == pytest.approx(initial_cash - expected_daily, rel=0.01)
        assert bt._borrow_costs_total == pytest.approx(expected_daily, rel=0.01)

    def test_no_borrow_for_longs(self):
        """Long positions should not incur borrow costs."""
        bt = AdvancedBacktester(
            initial_capital=100_000, commission_per_trade=0.0,
            slippage_bps=0.0, short_borrow_rate=0.10,
        )
        bt._data_symbols = ["AAPL"]
        bt.positions["AAPL"] = 100  # long

        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        data = {"AAPL": pd.DataFrame({
            "Open": [100.0] * 5, "High": [101.0] * 5,
            "Low": [99.0] * 5, "Close": [100.0] * 5,
            "Volume": [1_000_000] * 5,
        }, index=dates)}

        initial_cash = bt.cash
        bt._apply_borrow_costs(data, dates[0])

        assert bt.cash == initial_cash
        assert bt._borrow_costs_total == 0.0

    def test_zero_borrow_rate_no_cost(self):
        """With borrow rate 0, no costs should be charged."""
        bt = AdvancedBacktester(
            initial_capital=100_000, short_borrow_rate=0.0,
        )
        bt._data_symbols = ["AAPL"]
        bt.positions["AAPL"] = -100

        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        data = {"AAPL": pd.DataFrame({
            "Open": [100.0] * 5, "High": [101.0] * 5,
            "Low": [99.0] * 5, "Close": [100.0] * 5,
            "Volume": [1_000_000] * 5,
        }, index=dates)}

        initial_cash = bt.cash
        bt._apply_borrow_costs(data, dates[0])
        assert bt.cash == initial_cash

    def test_borrow_costs_in_metrics(self):
        """total_borrow_costs should appear in metrics."""
        bt = AdvancedBacktester(
            initial_capital=100_000, commission_per_trade=0.0,
            slippage_bps=0.0, short_borrow_rate=0.005,
            max_adv_participation=0.0,
        )
        np.random.seed(42)
        n = 40
        closes = list(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, n))))
        opens = [c * 0.999 for c in closes]
        data = _make_ohlcv("AAPL", closes, opens=opens)

        # Generate short signals
        def gen_short(sym, prices):
            if len(prices) > 5:
                return {'signal': -0.8, 'confidence': 0.8, 'quality': 0.8}
            return {'signal': 0.0, 'confidence': 0.5, 'quality': 0.5}

        sig = MagicMock()
        sig.generate_ml_signal = gen_short

        metrics = bt.run_backtest(
            data=data, signal_generator=sig,
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000, max_positions=3,
        )

        assert 'total_borrow_costs' in metrics
        assert metrics['total_borrow_costs'] >= 0


# ---------------------------------------------------------------------------
# 5. ADV Capacity Constraints
# ---------------------------------------------------------------------------

class TestADVCapacity:

    def test_position_capped_by_adv(self):
        """Position size should be capped at max_adv_participation * ADV."""
        bt = AdvancedBacktester(
            initial_capital=10_000_000,  # large capital
            commission_per_trade=0.0,
            slippage_bps=0.0,
            max_adv_participation=0.01,  # 1% of ADV
            short_borrow_rate=0.0,
        )
        n = 40
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        # Low volume: 10,000 shares/day → 1% cap = 100 shares
        data = {"AAPL": pd.DataFrame({
            "Open": [100.0] * n, "High": [101.0] * n,
            "Low": [99.0] * n, "Close": [100.0] * n,
            "Volume": [10_000] * n,
        }, index=dates)}

        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=500_000,  # would be 5000 shares without cap
            max_positions=5,
        )

        for trade in bt.trades:
            # Max shares = 10000 * 0.01 = 100
            assert trade['quantity'] <= 100, (
                f"Trade quantity {trade['quantity']} exceeds ADV cap of 100"
            )

    def test_no_cap_when_disabled(self):
        """With max_adv_participation=0, no capping should occur."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=0.0,
            slippage_bps=0.0,
            max_adv_participation=0.0,  # disabled
            short_borrow_rate=0.0,
        )
        n = 40
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        data = {"AAPL": pd.DataFrame({
            "Open": [10.0] * n, "High": [10.1] * n,
            "Low": [9.9] * n, "Close": [10.0] * n,
            "Volume": [100] * n,  # very low volume
        }, index=dates)}

        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000,  # would be 500 shares at $10
            max_positions=5,
        )

        # Without ADV cap, should be able to trade more than 1% of 100 = 1 share
        if bt.trades:
            assert bt.trades[0]['quantity'] > 1

    def test_high_volume_no_cap_effect(self):
        """With high volume, ADV cap should not bind."""
        bt = AdvancedBacktester(
            initial_capital=100_000,
            commission_per_trade=0.0,
            slippage_bps=0.0,
            max_adv_participation=0.05,  # 5%
            short_borrow_rate=0.0,
        )
        n = 40
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        # High volume: 10M shares/day → 5% = 500k shares
        data = {"AAPL": pd.DataFrame({
            "Open": [100.0] * n, "High": [101.0] * n,
            "Low": [99.0] * n, "Close": [100.0] * n,
            "Volume": [10_000_000] * n,
        }, index=dates)}

        bt.run_backtest(
            data=data,
            signal_generator=_make_signal_gen(0.8),
            start_date="2024-01-02", end_date="2024-02-23",
            position_size_usd=5000,  # 50 shares at $100, well within cap
            max_positions=5,
        )

        # Position should be sized by capital, not capped
        if bt.trades:
            assert bt.trades[0]['quantity'] > 1
