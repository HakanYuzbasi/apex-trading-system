"""
tests/test_risk_management.py - Test advanced risk management
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.advanced_risk_manager import AdvancedRiskManager


class TestAdvancedRiskManager:
    """Test AdvancedRiskManager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = AdvancedRiskManager(
            max_daily_loss=0.02,
            max_drawdown=0.10,
            confidence_level=0.95
        )
        self.risk_manager.set_starting_capital(100000)

    def test_initialization(self):
        """Test risk manager initialization."""
        assert self.risk_manager.starting_capital == 100000
        assert self.risk_manager.max_daily_loss == 0.02
        assert self.risk_manager.confidence_level == 0.95

    def test_var_historical(self):
        """Test historical VaR calculation."""
        # Generate random returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        var = self.risk_manager.calculate_var(returns, method='historical')

        assert var > 0
        assert var < 0.1  # Reasonable range

    def test_var_parametric(self):
        """Test parametric VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        var = self.risk_manager.calculate_var(returns, method='parametric')

        assert var > 0
        assert var < 0.1

    def test_cvar(self):
        """Test CVaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        cvar = self.risk_manager.calculate_cvar(returns)

        assert cvar > 0
        # CVaR should be >= VaR
        var = self.risk_manager.calculate_var(returns)
        assert cvar >= var

    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        # Win rate 60%, avg win 2%, avg loss 1%
        kelly = self.risk_manager.calculate_kelly_fraction(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01
        )

        assert kelly > 0
        assert kelly <= 0.25  # Capped at max

    def test_position_sizing(self):
        """Test position size calculation."""
        shares = self.risk_manager.calculate_position_size(
            capital=100000,
            price=100,
            volatility=0.20,
            confidence=0.8,
            max_position_value=10000,
            max_shares=200
        )

        assert shares > 0
        assert shares <= 200  # Respect max shares
        assert shares * 100 <= 10000  # Respect max value

    def test_position_sizing_high_volatility(self):
        """Test position sizing scales down for high volatility."""
        shares_low_vol = self.risk_manager.calculate_position_size(
            capital=100000,
            price=100,
            volatility=0.10,
            confidence=0.8,
            max_position_value=10000,
            max_shares=200
        )

        shares_high_vol = self.risk_manager.calculate_position_size(
            capital=100000,
            price=100,
            volatility=0.40,  # Higher volatility
            confidence=0.8,
            max_position_value=10000,
            max_shares=200
        )

        # High volatility should result in smaller position
        assert shares_high_vol < shares_low_vol

    def test_risk_limits_daily_loss(self):
        """Test daily loss limit checking."""
        # Simulate a 3% loss (should breach 2% limit)
        result = self.risk_manager.check_risk_limits(97000)

        assert result['daily_return'] < 0
        assert result['daily_loss_breached'] == True

    def test_risk_limits_drawdown(self):
        """Test drawdown limit checking."""
        # Simulate growth then drawdown
        self.risk_manager.check_risk_limits(120000)  # New peak
        result = self.risk_manager.check_risk_limits(105000)  # 12.5% drawdown

        assert result['drawdown'] > 0.10
        assert result['drawdown_breached'] == True

    def test_portfolio_var(self):
        """Test portfolio-level VaR calculation."""
        # Create mock positions and returns
        positions = {
            'AAPL': (10, 150),
            'GOOGL': (5, 120),
            'MSFT': (8, 300)
        }

        np.random.seed(42)
        returns_data = {
            'AAPL': pd.Series(np.random.normal(0.001, 0.02, 252)),
            'GOOGL': pd.Series(np.random.normal(0.001, 0.025, 252)),
            'MSFT': pd.Series(np.random.normal(0.001, 0.018, 252))
        }

        portfolio_var = self.risk_manager.calculate_portfolio_var(
            positions, returns_data
        )

        assert portfolio_var > 0
        assert portfolio_var < 100000  # Should be reasonable


def test_correlation_matrix_update():
    """Test correlation matrix update."""
    risk_manager = AdvancedRiskManager()

    np.random.seed(42)
    returns_data = {
        'AAPL': pd.Series(np.random.normal(0.001, 0.02, 252)),
        'GOOGL': pd.Series(np.random.normal(0.001, 0.025, 252)),
        'MSFT': pd.Series(np.random.normal(0.001, 0.018, 252))
    }

    risk_manager.update_correlation_matrix(returns_data)

    assert risk_manager.correlation_matrix is not None
    assert len(risk_manager.correlation_matrix) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
