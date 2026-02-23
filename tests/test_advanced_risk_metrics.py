"""
Advanced Risk Metrics Tests

Tests for CVaR, Sortino, Calmar, Omega ratios and other advanced risk metrics.
"""
import pytest
import numpy as np
import pandas as pd


class TestAdvancedRiskMetrics:
    """Test suite for advanced risk metrics calculations."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series for testing."""
        np.random.seed(42)
        # Generate returns with known properties
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        return pd.Series(returns, index=pd.date_range('2025-01-01', periods=252))

    @pytest.fixture
    def negative_skew_returns(self):
        """Generate negatively skewed returns (crash risk)."""
        np.random.seed(123)
        # Mix of normal days and crash days
        normal_returns = np.random.normal(0.002, 0.01, 240)
        crash_returns = np.random.normal(-0.05, 0.02, 12)
        returns = np.concatenate([normal_returns, crash_returns])
        np.random.shuffle(returns)
        return pd.Series(returns, index=pd.date_range('2025-01-01', periods=252))

    def test_cvar_calculation(self, sample_returns):
        """Test Conditional Value at Risk (CVaR) calculation."""
        confidence_level = 0.95
        
        # Calculate VaR threshold
        var_threshold = np.percentile(sample_returns, (1 - confidence_level) * 100)
        
        # CVaR is the average of returns below VaR
        cvar = sample_returns[sample_returns <= var_threshold].mean()
        
        # CVaR should be more negative than VaR
        assert cvar < var_threshold
        assert cvar < 0  # Should be negative (tail loss)
        
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino Ratio calculation."""
        # Sortino focuses on downside deviation
        risk_free_rate = 0.0001  # Daily risk-free rate
        
        excess_returns = sample_returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        sortino_ratio = excess_returns.mean() / downside_deviation
        
        assert isinstance(sortino_ratio, (int, float))
        assert not np.isnan(sortino_ratio)
        # Sortino should typically be higher than Sharpe for positive returns
        
    def test_calmar_ratio(self, sample_returns):
        """Test Calmar Ratio (Return / Max Drawdown)."""
        # Calculate cumulative returns
        cum_returns = (1 + sample_returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Annualized return
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(sample_returns)) - 1
        
        calmar_ratio = annual_return / abs(max_drawdown)
        
        assert isinstance(calmar_ratio, (int, float))
        assert not np.isnan(calmar_ratio)
        assert max_drawdown < 0  # Drawdown should be negative
        
    def test_omega_ratio(self, sample_returns):
        """Test Omega Ratio calculation."""
        threshold = 0.0  # MAR (Minimum Acceptable Return)
        
        # Probability-weighted gains vs losses
        returns_above = sample_returns[sample_returns > threshold]
        returns_below = sample_returns[sample_returns <= threshold]
        
        gains = returns_above.sum()
        losses = abs(returns_below.sum())
        
        omega_ratio = gains / losses if losses > 0 else np.inf
        
        assert isinstance(omega_ratio, (int, float))
        assert omega_ratio > 0  # Should be positive
        # Omega > 1 indicates gains > losses
        
    def test_max_drawdown_duration(self, sample_returns):
        """Test Maximum Drawdown Duration calculation."""
        cum_returns = (1 + sample_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find underwater periods
        is_underwater = drawdown < 0
        
        # Calculate duration
        max_duration = 0
        current_duration = 0
        
        for underwater in is_underwater:
            if underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        assert isinstance(max_duration, int)
        assert max_duration >= 0
        assert max_duration <= len(sample_returns)
        
    def test_downside_deviation(self, sample_returns):
        """Test downside deviation calculation."""
        target = 0.0
        downside_returns = sample_returns[sample_returns < target]
        downside_dev = np.sqrt(np.mean((downside_returns - target) ** 2))
        
        # Standard deviation for comparison
        std_dev = sample_returns.std()
        
        assert downside_dev > 0
        assert downside_dev <= std_dev  # Downside dev should be <= total vol
        
    def test_tail_ratio(self, negative_skew_returns):
        """Test Tail Ratio (right tail / left tail)."""
        # 95th percentile vs 5th percentile
        right_tail = np.percentile(negative_skew_returns, 95)
        left_tail = abs(np.percentile(negative_skew_returns, 5))
        
        tail_ratio = right_tail / left_tail if left_tail > 0 else np.inf
        
        assert tail_ratio > 0
        # For negatively skewed returns, ratio should be < 1
        assert tail_ratio < 1
        
    def test_var_confidence_levels(self, sample_returns):
        """Test VaR at different confidence levels."""
        confidence_levels = [0.90, 0.95, 0.99]
        vars = []
        
        for cl in confidence_levels:
            var = np.percentile(sample_returns, (1 - cl) * 100)
            vars.append(var)
        
        # VaR should be more negative at higher confidence
        assert vars[0] > vars[1] > vars[2]
        
    def test_conditional_sharpe(self, sample_returns):
        """Test Conditional Sharpe Ratio (CSR)."""
        # Sharpe ratio in different market conditions
        median_return = sample_returns.median()
        
        bull_returns = sample_returns[sample_returns > median_return]
        bear_returns = sample_returns[sample_returns <= median_return]
        
        bull_sharpe = bull_returns.mean() / bull_returns.std() if len(bull_returns) > 1 else 0
        bear_sharpe = bear_returns.mean() / bear_returns.std() if len(bear_returns) > 1 else 0
        
        # Bull market Sharpe should typically be higher
        assert bull_sharpe > bear_sharpe
        
    def test_skewness_kurtosis(self, negative_skew_returns):
        """Test skewness and kurtosis calculations."""
        from scipy import stats
        
        skew = stats.skew(negative_skew_returns)
        kurt = stats.kurtosis(negative_skew_returns)
        
        # Negative skew indicates left tail risk
        assert skew < 0
        # Excess kurtosis > 0 indicates fat tails
        assert kurt > 0
        
    def test_value_at_risk_historical(self, sample_returns):
        """Test Historical VaR method."""
        confidence = 0.95
        var_95 = -np.percentile(sample_returns, (1 - confidence) * 100)
        
        # Number of breaches should be approximately 5%
        breaches = (sample_returns < -var_95).sum()
        breach_rate = breaches / len(sample_returns)
        
        # Allow 2% margin of error
        assert 0.03 <= breach_rate <= 0.07
        
    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall (ES) = CVaR calculation."""
        confidence = 0.95
        var = np.percentile(sample_returns, (1 - confidence) * 100)
        es = sample_returns[sample_returns <= var].mean()
        
        # ES should be subadditive (coherent risk measure)
        assert es <= var
        assert es < 0


class TestKellyCriterion:
    """Test suite for Kelly Criterion position sizing."""

    def test_kelly_formula_basic(self):
        """Test basic Kelly Criterion formula."""
        # Win probability
        p = 0.60
        # Win/loss ratio
        b = 2.0  # Win twice what you risk
        
        # Kelly = (bp - q) / b, where q = 1 - p
        kelly_fraction = (b * p - (1 - p)) / b
        
        assert 0 < kelly_fraction < 1
        assert kelly_fraction == pytest.approx(0.40, abs=0.01)
        
    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge (should be 0)."""
        p = 0.45  # Losing edge
        b = 1.0
        
        kelly_fraction = max(0, (b * p - (1 - p)) / b)
        
        assert kelly_fraction == 0  # Don't bet with negative edge
        
    def test_half_kelly(self):
        """Test Half-Kelly for reduced volatility."""
        p = 0.60
        b = 2.0
        
        full_kelly = (b * p - (1 - p)) / b
        half_kelly = full_kelly / 2
        
        assert half_kelly < full_kelly
        assert half_kelly == pytest.approx(0.20, abs=0.01)
        
    def test_kelly_from_sharpe(self):
        """Test Kelly approximation from Sharpe ratio."""
        sharpe = 1.5
        
        # Kelly ≈ Sharpe / 2 for log-normal returns
        kelly_approx = sharpe / 2
        
        assert kelly_approx == 0.75
        
    def test_kelly_with_correlation(self):
        """Test Kelly adjustment for correlated positions."""
        # Two positions with correlation
        kelly_1 = 0.20
        kelly_2 = 0.20
        correlation = 0.5
        
        # Adjust for correlation
        adjusted_total = (kelly_1 + kelly_2) / (1 + correlation)
        
        assert adjusted_total < (kelly_1 + kelly_2)
        assert adjusted_total == pytest.approx(0.267, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
