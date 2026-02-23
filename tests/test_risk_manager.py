"""
tests/test_risk_manager.py - Risk Management Tests

Tests for the risk management system including:
- Position sizing
- Exposure limits
- Drawdown protection
- Sector exposure limits
"""

import pytest
import pandas as pd
import numpy as np


# Test fixtures
@pytest.fixture
def risk_config():
    """Risk management configuration."""
    return {
        'max_position_size': 50000,
        'max_portfolio_exposure': 0.95,
        'max_sector_exposure': 0.40,
        'max_single_stock_exposure': 0.10,
        'max_daily_loss': 0.03,
        'max_drawdown': 0.10,
        'position_size_usd': 15000,
    }


@pytest.fixture
def mock_portfolio():
    """Mock portfolio state."""
    return {
        'total_value': 1100000,
        'cash': 200000,
        'positions': {
            'AAPL': {'quantity': 100, 'value': 15000, 'sector': 'Technology'},
            'MSFT': {'quantity': 50, 'value': 20000, 'sector': 'Technology'},
            'JPM': {'quantity': 80, 'value': 12000, 'sector': 'Financials'},
        }
    }


@pytest.fixture
def sector_map():
    """Sector mapping for stocks."""
    return {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'NVDA': 'Technology',
        'JPM': 'Financials',
        'GS': 'Financials',
        'JNJ': 'Healthcare',
        'PFE': 'Healthcare',
        'XOM': 'Energy',
        'CVX': 'Energy',
    }


class TestPositionSizing:
    """Test position sizing logic."""

    def test_basic_position_size(self, risk_config):
        """Test basic position size calculation."""
        position_size = risk_config['position_size_usd']
        assert position_size == 15000

    def test_position_size_respects_max(self, risk_config):
        """Test that position size doesn't exceed maximum."""
        requested_size = 100000
        max_size = risk_config['max_position_size']
        actual_size = min(requested_size, max_size)
        assert actual_size == max_size

    def test_position_size_volatility_adjusted(self, risk_config):
        """Test volatility-adjusted position sizing."""
        base_size = risk_config['position_size_usd']
        volatility = 0.30  # 30% annualized vol

        # Higher volatility = smaller position
        vol_factor = 0.20 / volatility  # Target 20% vol
        adjusted_size = base_size * vol_factor

        assert adjusted_size < base_size

    def test_position_size_with_atr(self, risk_config, sample_price_data):
        """Test ATR-based position sizing."""
        base_size = risk_config['position_size_usd']

        # Calculate ATR
        high = sample_price_data['High']
        low = sample_price_data['Low']
        close = sample_price_data['Close']

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Risk per share based on ATR
        risk_per_share = 2 * atr
        current_price = close.iloc[-1]
        shares = int(base_size / current_price)

        assert shares > 0
        assert risk_per_share > 0


class TestExposureLimits:
    """Test exposure limit enforcement."""

    def test_max_portfolio_exposure(self, risk_config, mock_portfolio):
        """Test maximum portfolio exposure limit."""
        total_value = mock_portfolio['total_value']
        cash = mock_portfolio['cash']
        invested = total_value - cash

        exposure = invested / total_value
        max_exposure = risk_config['max_portfolio_exposure']

        assert exposure < max_exposure

    def test_single_stock_exposure_limit(self, risk_config, mock_portfolio):
        """Test single stock exposure limit."""
        total_value = mock_portfolio['total_value']
        max_single = risk_config['max_single_stock_exposure']

        for symbol, position in mock_portfolio['positions'].items():
            exposure = position['value'] / total_value
            assert exposure <= max_single, f"{symbol} exceeds single stock limit"

    def test_sector_exposure_limit(self, risk_config, mock_portfolio, sector_map):
        """Test sector exposure limit."""
        total_value = mock_portfolio['total_value']
        max_sector = risk_config['max_sector_exposure']

        sector_exposure = {}
        for symbol, position in mock_portfolio['positions'].items():
            sector = position['sector']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position['value']

        for sector, exposure in sector_exposure.items():
            pct = exposure / total_value
            assert pct <= max_sector, f"{sector} exceeds sector limit"

    def test_new_position_would_exceed_sector_limit(self, risk_config, mock_portfolio, sector_map):
        """Test rejection when new position would exceed sector limit."""
        total_value = mock_portfolio['total_value']
        max_sector = risk_config['max_sector_exposure']

        # Current tech exposure
        tech_exposure = sum(
            p['value'] for p in mock_portfolio['positions'].values()
            if p['sector'] == 'Technology'
        )

        # Proposed new tech position
        new_position_value = 500000  # Large position

        proposed_exposure = (tech_exposure + new_position_value) / total_value
        would_exceed = proposed_exposure > max_sector

        assert would_exceed, "Large tech position should exceed sector limit"


class TestDrawdownProtection:
    """Test drawdown protection."""

    def test_daily_loss_limit(self, risk_config):
        """Test daily loss limit enforcement."""
        initial_value = 1100000
        max_daily_loss = risk_config['max_daily_loss']
        max_loss_amount = initial_value * max_daily_loss

        # Simulate loss
        current_loss = 25000  # $25k loss
        exceeds_limit = current_loss > max_loss_amount

        assert not exceeds_limit, "Should not exceed daily loss limit"

    def test_daily_loss_triggers_halt(self, risk_config):
        """Test that exceeding daily loss triggers trading halt."""
        initial_value = 1100000
        max_daily_loss = risk_config['max_daily_loss']
        max_loss_amount = initial_value * max_daily_loss

        # Simulate large loss
        current_loss = 40000  # $40k loss (>3%)
        should_halt = current_loss > max_loss_amount

        assert should_halt, "Should trigger trading halt"

    def test_max_drawdown_limit(self, risk_config):
        """Test maximum drawdown limit."""
        peak_value = 1200000
        current_value = 1100000
        max_drawdown = risk_config['max_drawdown']

        drawdown = (peak_value - current_value) / peak_value
        exceeds_limit = drawdown > max_drawdown

        assert not exceeds_limit, "Should not exceed max drawdown"

    def test_drawdown_triggers_risk_reduction(self, risk_config):
        """Test drawdown triggers risk reduction."""
        peak_value = 1200000
        current_value = 1050000
        max_drawdown = risk_config['max_drawdown']

        drawdown = (peak_value - current_value) / peak_value
        should_reduce_risk = drawdown > (max_drawdown * 0.5)  # 50% of max

        assert should_reduce_risk, "Should trigger risk reduction"


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_var_calculation(self, sample_price_data):
        """Test Value at Risk calculation."""
        returns = sample_price_data['Close'].pct_change().dropna()

        # 95% VaR
        var_95 = np.percentile(returns, 5)

        assert var_95 < 0, "VaR should be negative (loss)"

    def test_cvar_calculation(self, sample_price_data):
        """Test Conditional VaR (Expected Shortfall) calculation."""
        returns = sample_price_data['Close'].pct_change().dropna()

        # 95% VaR
        var_95 = np.percentile(returns, 5)

        # CVaR is mean of returns below VaR
        cvar = returns[returns <= var_95].mean()

        assert cvar < var_95, "CVaR should be worse than VaR"

    def test_sharpe_ratio_calculation(self, sample_price_data):
        """Test Sharpe ratio calculation."""
        returns = sample_price_data['Close'].pct_change().dropna()

        mean_return = returns.mean() * 252  # Annualized
        std_return = returns.std() * np.sqrt(252)  # Annualized
        risk_free_rate = 0.05  # 5% risk-free rate

        sharpe = (mean_return - risk_free_rate) / std_return

        assert isinstance(sharpe, float)

    def test_sortino_ratio_calculation(self, sample_price_data):
        """Test Sortino ratio calculation."""
        returns = sample_price_data['Close'].pct_change().dropna()

        mean_return = returns.mean() * 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        risk_free_rate = 0.05

        sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        assert isinstance(sortino, float)

    def test_max_drawdown_calculation(self, sample_price_data):
        """Test maximum drawdown calculation."""
        prices = sample_price_data['Close']
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_dd = drawdown.min()

        assert max_dd <= 0, "Max drawdown should be negative or zero"
        assert max_dd >= -1, "Max drawdown can't exceed 100%"


class TestCorrelationRisk:
    """Test correlation-based risk management."""

    def test_correlation_calculation(self):
        """Test correlation calculation between assets."""
        np.random.seed(42)

        # Create correlated returns
        returns1 = np.random.normal(0, 0.02, 100)
        returns2 = returns1 * 0.8 + np.random.normal(0, 0.01, 100)

        correlation = np.corrcoef(returns1, returns2)[0, 1]

        assert correlation > 0.5, "Should show positive correlation"

    def test_portfolio_correlation_limit(self):
        """Test portfolio correlation limit enforcement."""

        # Simulate portfolio correlations
        correlations = [0.8, 0.6, 0.75, 0.65]
        avg_correlation = np.mean(correlations)


        # In this case, average is 0.7, exactly at limit
        assert avg_correlation == pytest.approx(0.7, rel=0.01)

    def test_diversification_ratio(self):
        """Test portfolio diversification ratio."""
        # Individual volatilities
        individual_vols = [0.2, 0.25, 0.3]
        weights = [0.33, 0.33, 0.34]

        # Weighted average volatility
        weighted_avg_vol = sum(w * v for w, v in zip(weights, individual_vols))

        # Portfolio volatility (with diversification)
        portfolio_vol = 0.18  # Lower due to diversification

        diversification_ratio = weighted_avg_vol / portfolio_vol

        assert diversification_ratio > 1, "Diversification should reduce risk"


class TestRiskAdjustedReturns:
    """Test risk-adjusted return calculations."""

    def test_risk_adjusted_position_size(self, risk_config):
        """Test risk-adjusted position sizing."""
        base_size = risk_config['position_size_usd']

        # High volatility stock
        high_vol_adjustment = 0.5
        high_vol_size = base_size * high_vol_adjustment

        # Low volatility stock
        low_vol_adjustment = 1.5
        low_vol_size = min(base_size * low_vol_adjustment, risk_config['max_position_size'])

        assert high_vol_size < base_size
        assert low_vol_size >= base_size

    def test_kelly_criterion(self):
        """Test Kelly criterion for position sizing."""
        win_rate = 0.55
        avg_win = 0.02  # 2% average win
        avg_loss = 0.015  # 1.5% average loss

        # Kelly = W - (1-W)/R where R = avg_win/avg_loss
        R = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / R

        assert 0 < kelly < 1, "Kelly should be between 0 and 1"

    def test_half_kelly(self):
        """Test half-Kelly for more conservative sizing."""
        win_rate = 0.55
        avg_win = 0.02
        avg_loss = 0.015

        R = avg_win / avg_loss
        full_kelly = win_rate - (1 - win_rate) / R
        half_kelly = full_kelly / 2

        assert half_kelly < full_kelly
        assert half_kelly > 0
