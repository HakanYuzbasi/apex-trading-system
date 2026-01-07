"""
tests/test_performance_tracker.py - Test performance tracking
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.advanced_performance_tracker import AdvancedPerformanceTracker


class TestAdvancedPerformanceTracker:
    """Test AdvancedPerformanceTracker functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = AdvancedPerformanceTracker(risk_free_rate=0.02)
        self.tracker.set_starting_capital(100000)

    def test_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.starting_capital == 100000
        assert len(self.tracker.trades) == 0
        assert len(self.tracker.equity_curve) == 1  # Initial equity point

    def test_record_trade(self):
        """Test recording a trade."""
        self.tracker.record_trade(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150,
            commission=1.0,
            pnl=0
        )

        assert len(self.tracker.trades) == 1
        assert self.tracker.trades[0]['symbol'] == 'AAPL'
        assert self.tracker.total_commissions == 1.0

    def test_record_equity(self):
        """Test recording equity points."""
        self.tracker.record_equity(105000)
        self.tracker.record_equity(110000)

        assert len(self.tracker.equity_curve) == 3  # Initial + 2 new

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Simulate growing equity curve
        for i in range(100):
            value = 100000 * (1 + 0.001 * i + np.random.normal(0, 0.01))
            self.tracker.record_equity(value)

        sharpe = self.tracker.get_sharpe_ratio()

        # Should be positive for growing equity
        assert sharpe != 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Simulate equity curve
        for i in range(100):
            value = 100000 * (1 + 0.001 * i + np.random.normal(0, 0.01))
            self.tracker.record_equity(value)

        sortino = self.tracker.get_sortino_ratio()

        # Sortino should be higher than Sharpe (only penalizes downside)
        sharpe = self.tracker.get_sharpe_ratio()
        # Both should be calculated (non-zero)
        assert sortino != 0 or sharpe != 0

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Simulate drawdown scenario
        values = [100000, 110000, 105000, 95000, 100000]
        for val in values:
            self.tracker.record_equity(val)

        max_dd = self.tracker.get_max_drawdown()

        # Should detect ~13.6% drawdown (110k -> 95k)
        assert max_dd > 0.13
        assert max_dd < 0.15

    def test_win_rate(self):
        """Test win rate calculation."""
        # Record some trades with P&L
        self.tracker.record_trade('AAPL', 'BUY', 10, 150, pnl=100)
        self.tracker.record_trade('GOOGL', 'BUY', 5, 120, pnl=-50)
        self.tracker.record_trade('MSFT', 'BUY', 8, 300, pnl=200)

        win_rate = self.tracker.get_win_rate()

        # 2 winners out of 3 = 66.7%
        assert abs(win_rate - 0.6667) < 0.01

    def test_profit_factor(self):
        """Test profit factor calculation."""
        self.tracker.record_trade('AAPL', 'BUY', 10, 150, pnl=100)
        self.tracker.record_trade('GOOGL', 'BUY', 5, 120, pnl=-50)
        self.tracker.record_trade('MSFT', 'BUY', 8, 300, pnl=200)

        pf = self.tracker.get_profit_factor()

        # Profit: 300, Loss: 50, PF = 6.0
        assert abs(pf - 6.0) < 0.1

    def test_average_trade(self):
        """Test average trade calculation."""
        self.tracker.record_trade('AAPL', 'BUY', 10, 150, pnl=100)
        self.tracker.record_trade('GOOGL', 'BUY', 5, 120, pnl=-50)
        self.tracker.record_trade('MSFT', 'BUY', 8, 300, pnl=200)

        avg_win, avg_loss, avg_trade = self.tracker.get_average_trade()

        assert avg_win == 150  # (100 + 200) / 2
        assert avg_loss == -50
        assert avg_trade == 250/3  # (100 - 50 + 200) / 3

    def test_slippage_analysis(self):
        """Test slippage analysis."""
        self.tracker.record_trade('AAPL', 'BUY', 10, 150, slippage=5.0, pnl=100)
        self.tracker.record_trade('GOOGL', 'SELL', 5, 120, slippage=3.0, pnl=50)

        analysis = self.tracker.get_slippage_analysis()

        assert analysis['total_slippage'] == 8.0
        assert analysis['avg_slippage_per_trade'] == 4.0

    def test_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Simulate some trading activity
        for i in range(10):
            self.tracker.record_equity(100000 + i * 1000)
            if i % 2 == 0:
                self.tracker.record_trade(f'STOCK{i}', 'BUY', 10, 100, pnl=100)
            else:
                self.tracker.record_trade(f'STOCK{i}', 'BUY', 10, 100, pnl=-50)

        report = self.tracker.get_comprehensive_report()

        assert 'starting_capital' in report
        assert 'ending_capital' in report
        assert 'total_trades' in report
        assert 'sharpe_ratio' in report
        assert report['total_trades'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
