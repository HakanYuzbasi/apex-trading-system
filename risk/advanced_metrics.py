"""
Advanced Risk Metrics

Implementation of advanced risk and performance metrics including:
- CVaR (Conditional Value at Risk)
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Kelly Criterion
- Tail Risk Analysis
"""
import numpy as np
import pandas as pd
from typing import Union
from scipy import stats


class AdvancedRiskMetrics:
    """Advanced risk and performance metrics calculator."""

    @staticmethod
    def calculate_cvar(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        CVaR is the expected loss given that the loss exceeds VaR.
        Also known as Expected Shortfall (ES).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            CVaR value (negative number representing tail loss)
        """
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
        cvar = returns_array[returns_array <= var_threshold].mean()
        return cvar

    @staticmethod
    def calculate_sortino_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino ratio focuses on downside deviation rather than total volatility.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Sortino ratio
        """
        returns_array = np.array(returns)
        daily_rf = risk_free_rate / periods_per_year
        
        excess_returns = returns_array - daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return np.inf
        
        annualized_return = excess_returns.mean() * periods_per_year
        annualized_downside_dev = downside_deviation * np.sqrt(periods_per_year)
        
        return annualized_return / annualized_downside_dev

    @staticmethod
    def calculate_calmar_ratio(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar ratio is the annualized return divided by maximum drawdown.
        
        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        if max_drawdown == 0:
            return np.inf
        
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (periods_per_year / len(returns_series)) - 1
        
        return annual_return / abs(max_drawdown)

    @staticmethod
    def calculate_omega_ratio(
        returns: Union[pd.Series, np.ndarray],
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega Ratio.
        
        Omega ratio is the probability-weighted ratio of gains versus losses.
        
        Args:
            returns: Return series
            threshold: Minimum acceptable return (MAR)
            
        Returns:
            Omega ratio
        """
        returns_array = np.array(returns)
        
        returns_above = returns_array[returns_array > threshold]
        returns_below = returns_array[returns_array <= threshold]
        
        gains = returns_above.sum() if len(returns_above) > 0 else 0
        losses = abs(returns_below.sum()) if len(returns_below) > 0 else 0
        
        if losses == 0:
            return np.inf if gains > 0 else 1.0
        
        return gains / losses

    @staticmethod
    def calculate_max_drawdown_duration(
        returns: Union[pd.Series, np.ndarray]
    ) -> int:
        """
        Calculate Maximum Drawdown Duration.
        
        Returns the longest period (in number of returns) spent underwater.
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown duration in periods
        """
        returns_series = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        is_underwater = drawdown < 0
        
        max_duration = 0
        current_duration = 0
        
        for underwater in is_underwater:
            if underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration

    @staticmethod
    def calculate_downside_deviation(
        returns: Union[pd.Series, np.ndarray],
        target: float = 0.0
    ) -> float:
        """
        Calculate Downside Deviation.
        
        Measures volatility of returns below a target return.
        
        Args:
            returns: Return series
            target: Target return
            
        Returns:
            Downside deviation
        """
        returns_array = np.array(returns)
        downside_returns = returns_array[returns_array < target]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return np.sqrt(np.mean((downside_returns - target) ** 2))

    @staticmethod
    def calculate_tail_ratio(
        returns: Union[pd.Series, np.ndarray],
        percentile: int = 95
    ) -> float:
        """
        Calculate Tail Ratio.
        
        Ratio of right tail (gains) to left tail (losses).
        
        Args:
            returns: Return series
            percentile: Percentile for tail analysis (default 95)
            
        Returns:
            Tail ratio
        """
        returns_array = np.array(returns)
        
        right_tail = np.percentile(returns_array, percentile)
        left_tail = abs(np.percentile(returns_array, 100 - percentile))
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 1.0
        
        return right_tail / left_tail

    @staticmethod
    def calculate_var(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            VaR value (negative number)
        """
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence_level) * 100)

    @staticmethod
    def calculate_skewness(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate skewness of returns.
        
        Args:
            returns: Return series
            
        Returns:
            Skewness value
        """
        return stats.skew(returns)

    @staticmethod
    def calculate_kurtosis(returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate excess kurtosis of returns.
        
        Args:
            returns: Return series
            
        Returns:
            Excess kurtosis value
        """
        return stats.kurtosis(returns)


class KellyCriterion:
    """Kelly Criterion position sizing calculator."""

    @staticmethod
    def calculate_kelly_fraction(
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction.
        
        Formula: f = (bp - q) / b
        where:
            f = Kelly fraction
            b = win/loss ratio
            p = win probability
            q = loss probability (1 - p)
        
        Args:
            win_probability: Probability of winning
            win_loss_ratio: Ratio of win to loss
            
        Returns:
            Kelly fraction (0 to 1)
        """
        if win_probability <= 0 or win_probability >= 1:
            raise ValueError("Win probability must be between 0 and 1")
        
        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive")
        
        loss_probability = 1 - win_probability
        kelly = (win_loss_ratio * win_probability - loss_probability) / win_loss_ratio
        
        return max(0, kelly)  # Never bet with negative edge

    @staticmethod
    def calculate_half_kelly(
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Half-Kelly fraction for reduced volatility.
        
        Args:
            win_probability: Probability of winning
            win_loss_ratio: Ratio of win to loss
            
        Returns:
            Half-Kelly fraction
        """
        full_kelly = KellyCriterion.calculate_kelly_fraction(
            win_probability,
            win_loss_ratio
        )
        return full_kelly / 2

    @staticmethod
    def kelly_from_sharpe(
        sharpe_ratio: float
    ) -> float:
        """
        Approximate Kelly fraction from Sharpe ratio.
        
        For log-normal returns: Kelly ≈ Sharpe / 2
        
        Args:
            sharpe_ratio: Sharpe ratio
            
        Returns:
            Approximate Kelly fraction
        """
        return sharpe_ratio / 2

    @staticmethod
    def adjust_for_correlation(
        kelly_fraction_1: float,
        kelly_fraction_2: float,
        correlation: float
    ) -> float:
        """
        Adjust Kelly fraction for correlated positions.
        
        Args:
            kelly_fraction_1: Kelly fraction for position 1
            kelly_fraction_2: Kelly fraction for position 2  
            correlation: Correlation between positions
            
        Returns:
            Adjusted total Kelly fraction
        """
        if not -1 <= correlation <= 1:
            raise ValueError("Correlation must be between -1 and 1")
        
        total_kelly = kelly_fraction_1 + kelly_fraction_2
        adjusted = total_kelly / (1 + correlation)
        
        return adjusted


def calculate_all_metrics(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate all advanced risk metrics at once.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = AdvancedRiskMetrics()
    
    return {
        'cvar_95': metrics.calculate_cvar(returns, 0.95),
        'cvar_99': metrics.calculate_cvar(returns, 0.99),
        'sortino_ratio': metrics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': metrics.calculate_calmar_ratio(returns, periods_per_year),
        'omega_ratio': metrics.calculate_omega_ratio(returns),
        'max_dd_duration': metrics.calculate_max_drawdown_duration(returns),
        'downside_deviation': metrics.calculate_downside_deviation(returns),
        'tail_ratio': metrics.calculate_tail_ratio(returns),
        'var_95': metrics.calculate_var(returns, 0.95),
        'var_99': metrics.calculate_var(returns, 0.99),
        'skewness': metrics.calculate_skewness(returns),
        'kurtosis': metrics.calculate_kurtosis(returns)
    }
