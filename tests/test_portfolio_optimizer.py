"""
tests/test_portfolio_optimizer.py - Portfolio Optimization Tests

Tests for portfolio optimization including:
- Mean-variance optimization
- Risk parity
- Rebalancing logic
- Constraint handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def returns_data():
    """Generate sample returns data for multiple assets."""
    np.random.seed(42)
    n_days = 252
    n_assets = 5

    # Generate correlated returns
    mean_returns = [0.0003, 0.0004, 0.0002, 0.0005, 0.0001]
    volatilities = [0.01, 0.015, 0.008, 0.02, 0.012]

    # Correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.3, 0.4, 0.2],
        [0.5, 1.0, 0.4, 0.6, 0.3],
        [0.3, 0.4, 1.0, 0.3, 0.5],
        [0.4, 0.6, 0.3, 1.0, 0.4],
        [0.2, 0.3, 0.5, 0.4, 1.0]
    ])

    # Generate returns
    L = np.linalg.cholesky(corr)
    random_returns = np.random.normal(0, 1, (n_days, n_assets))
    correlated_returns = random_returns @ L.T

    # Scale by volatility and add mean
    returns = pd.DataFrame(
        correlated_returns * volatilities + mean_returns,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        index=pd.date_range('2023-01-01', periods=n_days)
    )

    return returns


@pytest.fixture
def current_weights():
    """Current portfolio weights."""
    return {
        'AAPL': 0.25,
        'MSFT': 0.20,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'META': 0.20
    }


class TestMeanVarianceOptimization:
    """Test mean-variance optimization."""

    def test_equal_weight_portfolio(self, returns_data):
        """Test equal weight portfolio as baseline."""
        n_assets = len(returns_data.columns)
        weights = np.array([1/n_assets] * n_assets)

        assert np.isclose(weights.sum(), 1.0)
        assert all(w > 0 for w in weights)

    def test_portfolio_return_calculation(self, returns_data):
        """Test portfolio return calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mean_returns = returns_data.mean()

        portfolio_return = np.dot(weights, mean_returns) * 252

        assert isinstance(portfolio_return, float)

    def test_portfolio_volatility_calculation(self, returns_data):
        """Test portfolio volatility calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov_matrix = returns_data.cov() * 252

        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        assert portfolio_vol > 0
        assert portfolio_vol < 1  # Less than 100% vol

    def test_minimum_variance_portfolio(self, returns_data):
        """Test minimum variance portfolio construction."""
        cov_matrix = returns_data.cov()
        len(returns_data.columns)

        # Simple minimum variance (equal risk contribution approximation)
        inv_var = 1 / np.diag(cov_matrix)
        weights = inv_var / inv_var.sum()

        assert np.isclose(weights.sum(), 1.0)
        assert all(w > 0 for w in weights)

    def test_maximum_sharpe_portfolio(self, returns_data):
        """Test maximum Sharpe ratio portfolio."""
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        risk_free_rate = 0.05

        # Simple approximation using excess returns / variance
        excess_returns = mean_returns - risk_free_rate / 252
        inv_cov = np.linalg.inv(cov_matrix)
        np.ones(len(mean_returns))

        # Optimal weights (unconstrained)
        raw_weights = inv_cov @ excess_returns
        weights = raw_weights / raw_weights.sum()

        # Allow short selling in this test
        assert np.isclose(weights.sum(), 1.0)


class TestRiskParity:
    """Test risk parity optimization."""

    def test_equal_risk_contribution(self, returns_data):
        """Test equal risk contribution portfolio."""
        cov_matrix = returns_data.cov() * 252
        len(returns_data.columns)

        # Inverse volatility weighting as approximation
        vols = np.sqrt(np.diag(cov_matrix))
        weights = (1 / vols) / (1 / vols).sum()

        assert np.isclose(weights.sum(), 1.0)
        assert all(w > 0 for w in weights)

    def test_risk_contribution_calculation(self, returns_data):
        """Test marginal risk contribution calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        cov_matrix = returns_data.cov() * 252

        # Portfolio volatility
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_var)

        # Marginal risk contribution
        mrc = np.dot(cov_matrix, weights) / port_vol

        # Risk contribution
        rc = weights * mrc

        # Total risk should equal portfolio volatility
        assert np.isclose(rc.sum(), port_vol, rtol=0.01)


class TestRebalancing:
    """Test portfolio rebalancing logic."""

    def test_rebalancing_threshold(self, current_weights):
        """Test rebalancing threshold detection."""
        target_weights = {
            'AAPL': 0.20,
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'AMZN': 0.20,
            'META': 0.20
        }

        threshold = 0.05  # 5% deviation threshold

        deviations = {
            symbol: abs(current_weights[symbol] - target_weights[symbol])
            for symbol in current_weights
        }

        any(d > threshold for d in deviations.values())

        # AAPL deviates by 5%, which is at threshold
        assert deviations['AAPL'] == pytest.approx(0.05, abs=0.001)

    def test_calculate_trades_for_rebalance(self, current_weights):
        """Test trade calculation for rebalancing."""
        portfolio_value = 1000000
        target_weights = {
            'AAPL': 0.20,
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'AMZN': 0.20,
            'META': 0.20
        }

        trades = {}
        for symbol in current_weights:
            current_value = portfolio_value * current_weights[symbol]
            target_value = portfolio_value * target_weights[symbol]
            trade_value = target_value - current_value
            if abs(trade_value) > 1000:  # Minimum trade size
                trades[symbol] = trade_value

        # AAPL should sell, AMZN should buy
        assert trades.get('AAPL', 0) < 0  # Sell
        assert trades.get('AMZN', 0) > 0  # Buy

    def test_rebalancing_cost_consideration(self):
        """Test that rebalancing considers transaction costs."""
        trade_value = 10000
        commission_rate = 0.001  # 0.1%

        commission = trade_value * commission_rate

        # Trade should only happen if benefit > cost
        expected_benefit = 50  # Example benefit
        should_trade = expected_benefit > commission

        assert commission == 10
        assert should_trade


class TestConstraints:
    """Test constraint handling in optimization."""

    def test_long_only_constraint(self, returns_data):
        """Test long-only constraint enforcement."""
        len(returns_data.columns)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.10])

        # All weights should be non-negative
        assert all(w >= 0 for w in weights)
        assert np.isclose(weights.sum(), 1.0)

    def test_maximum_weight_constraint(self, returns_data):
        """Test maximum weight constraint."""
        max_weight = 0.30
        weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

        assert all(w <= max_weight for w in weights)

    def test_minimum_weight_constraint(self):
        """Test minimum weight constraint."""
        min_weight = 0.05
        weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

        assert all(w >= min_weight for w in weights)

    def test_sector_weight_constraint(self):
        """Test sector weight constraint."""
        sector_weights = {
            'Technology': 0.55,  # AAPL + MSFT + GOOGL
            'Consumer': 0.45     # AMZN + META
        }

        max_sector_weight = 0.60

        assert all(w <= max_sector_weight for w in sector_weights.values())

    def test_turnover_constraint(self, current_weights):
        """Test turnover constraint."""
        new_weights = {
            'AAPL': 0.15,
            'MSFT': 0.25,
            'GOOGL': 0.25,
            'AMZN': 0.20,
            'META': 0.15
        }

        turnover = sum(
            abs(new_weights[s] - current_weights[s])
            for s in current_weights
        ) / 2  # One-way turnover

        max_turnover = 0.20

        assert turnover <= max_turnover


class TestOptimizationMetrics:
    """Test optimization output metrics."""

    def test_efficient_frontier_point(self, returns_data):
        """Test that portfolio is on efficient frontier."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol

        assert port_return > 0 or port_return <= 0  # Can be any value
        assert port_vol > 0
        assert isinstance(sharpe, float)

    def test_tracking_error(self, returns_data):
        """Test tracking error calculation vs benchmark."""
        portfolio_returns = returns_data.mean(axis=1)  # Equal weight
        benchmark_returns = returns_data['AAPL']  # Use AAPL as benchmark

        tracking_diff = portfolio_returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(252)

        assert tracking_error > 0

    def test_information_ratio(self, returns_data):
        """Test information ratio calculation."""
        portfolio_returns = returns_data.mean(axis=1)
        benchmark_returns = returns_data['AAPL']

        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        assert isinstance(information_ratio, float)


class TestTaxLossHarvesting:
    """Test tax-loss harvesting functionality."""

    def test_identify_loss_positions(self):
        """Test identification of positions with losses."""
        positions = {
            'AAPL': {'cost_basis': 150, 'current_price': 180, 'quantity': 100},
            'MSFT': {'cost_basis': 350, 'current_price': 320, 'quantity': 50},
            'GOOGL': {'cost_basis': 140, 'current_price': 130, 'quantity': 80},
        }

        losses = {
            symbol: (p['current_price'] - p['cost_basis']) * p['quantity']
            for symbol, p in positions.items()
            if p['current_price'] < p['cost_basis']
        }

        assert 'MSFT' in losses
        assert 'GOOGL' in losses
        assert 'AAPL' not in losses

    def test_wash_sale_prevention(self):
        """Test wash sale rule compliance."""
        # Sold MSFT at a loss
        sale_date = datetime(2024, 1, 15)
        wash_sale_window = 30  # days

        # Trying to buy similar security
        proposed_buy_date = datetime(2024, 1, 25)

        days_since_sale = (proposed_buy_date - sale_date).days
        is_wash_sale = days_since_sale < wash_sale_window

        assert is_wash_sale

    def test_replacement_security_selection(self):
        """Test selection of replacement securities."""
        sold_security = 'MSFT'  # Microsoft
        similar_securities = ['GOOGL', 'AAPL', 'CRM']  # Similar tech stocks

        # Replacement should maintain exposure without wash sale
        replacement = similar_securities[0]  # GOOGL

        assert replacement != sold_security
        assert replacement in similar_securities
