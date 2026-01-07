"""
risk/advanced_risk_manager.py - Professional Risk Management
Features:
- Value-at-Risk (VaR) calculation (Historical, Parametric, Monte Carlo)
- Kelly Criterion position sizing
- Volatility-based position scaling
- Correlation tracking and risk decomposition
- Portfolio-level risk metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class AdvancedRiskManager:
    """
    Professional-grade risk management system.

    Features:
    - VaR calculation (multiple methods)
    - Kelly Criterion position sizing
    - Volatility scaling
    - Correlation tracking
    - Risk budgeting
    - Stress testing
    """

    def __init__(
        self,
        max_daily_loss: float = 0.02,
        max_drawdown: float = 0.10,
        confidence_level: float = 0.95,
        var_method: str = "historical"
    ):
        """
        Initialize advanced risk manager.

        Args:
            max_daily_loss: Maximum daily loss as fraction (0.02 = 2%)
            max_drawdown: Maximum drawdown from peak (0.10 = 10%)
            confidence_level: Confidence level for VaR (0.95 = 95%)
            var_method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.confidence_level = confidence_level
        self.var_method = var_method

        # State tracking
        self.starting_capital = 0.0
        self.peak_capital = 0.0
        self.day_start_capital = 0.0
        self.current_day = datetime.now().strftime('%Y-%m-%d')

        # Returns history for VaR calculation
        self.returns_history: List[float] = []
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)

        # Correlation matrix
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None

        # Position-level risk
        self.position_vars: Dict[str, float] = {}

        logger.info(f"🛡️ Advanced Risk Manager initialized")
        logger.info(f"   VaR method: {var_method}")
        logger.info(f"   Confidence level: {confidence_level*100}%")

    def set_starting_capital(self, capital: float):
        """Set starting capital."""
        self.starting_capital = float(capital)
        self.peak_capital = float(capital)
        self.day_start_capital = float(capital)
        logger.info(f"💰 Starting capital: ${capital:,.2f}")

    def update_correlation_matrix(self, returns_data: Dict[str, pd.Series]):
        """
        Update correlation and covariance matrices.

        Args:
            returns_data: Dict of {symbol: returns_series}
        """
        try:
            if not returns_data or len(returns_data) < 2:
                return

            # Create DataFrame from returns
            df = pd.DataFrame(returns_data)

            # Drop symbols with insufficient data
            df = df.dropna(axis=1, how='all')
            df = df.fillna(0)  # Fill remaining NaNs with 0

            if len(df.columns) < 2:
                return

            # Calculate correlation and covariance
            self.correlation_matrix = df.corr()
            self.covariance_matrix = df.cov()

            logger.debug(f"📊 Updated correlation matrix: {len(self.correlation_matrix)} symbols")

        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = None,
        method: str = None
    ) -> float:
        """
        Calculate Value-at-Risk.

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (default: self.confidence_level)
            method: VaR method (default: self.var_method)

        Returns:
            VaR as positive number (e.g., 0.02 = 2% loss)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if method is None:
            method = self.var_method

        if len(returns) < 10:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0

        try:
            if method == "historical":
                return self._var_historical(returns, confidence_level)
            elif method == "parametric":
                return self._var_parametric(returns, confidence_level)
            elif method == "monte_carlo":
                return self._var_monte_carlo(returns, confidence_level)
            else:
                logger.warning(f"Unknown VaR method: {method}, using historical")
                return self._var_historical(returns, confidence_level)

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def _var_historical(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Historical VaR."""
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        return abs(float(var))

    def _var_parametric(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Parametric VaR (assumes normal distribution)."""
        mean = returns.mean()
        std = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # VaR = mean + z_score * std (for negative tail)
        var = mean + z_score * std
        return abs(float(var))

    def _var_monte_carlo(
        self,
        returns: pd.Series,
        confidence_level: float,
        n_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        mean = returns.mean()
        std = returns.std()

        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, n_simulations)

        # Calculate VaR from simulations
        percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, percentile)
        return abs(float(var))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = None) -> float:
        """
        Calculate Conditional Value-at-Risk (Expected Shortfall).

        CVaR is the expected loss given that VaR is exceeded.

        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (default: self.confidence_level)

        Returns:
            CVaR as positive number
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if len(returns) < 10:
            return 0.0

        try:
            var = self.calculate_var(returns, confidence_level)
            # CVaR = mean of returns below VaR threshold
            threshold = -var
            tail_returns = returns[returns <= threshold]

            if len(tail_returns) == 0:
                return var

            cvar = abs(float(tail_returns.mean()))
            return cvar

        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.

        Kelly% = W - (1-W)/R
        Where: W = win rate, R = avg_win/avg_loss

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            max_kelly: Maximum Kelly fraction (cap for safety)

        Returns:
            Kelly fraction (0-max_kelly)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            return 0.0

        try:
            # Kelly formula
            win_loss_ratio = avg_win / avg_loss
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

            # Cap Kelly fraction for safety (full Kelly is often too aggressive)
            kelly = max(0.0, min(kelly, max_kelly))

            return float(kelly)

        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float,
        confidence: float = 0.5,
        max_position_value: float = 10000,
        max_shares: int = 200,
        target_volatility: float = 0.15
    ) -> int:
        """
        Calculate optimal position size with multiple constraints.

        Methods:
        1. Volatility scaling: Scale by (target_vol / current_vol)
        2. Confidence scaling: Scale by signal confidence
        3. Hard limits: Respect max dollars and max shares

        Args:
            capital: Available capital
            price: Current stock price
            volatility: Stock volatility (annualized)
            confidence: Signal confidence (0-1)
            max_position_value: Max dollar value per position
            max_shares: Max shares per position
            target_volatility: Target portfolio volatility

        Returns:
            Number of shares to trade
        """
        try:
            # Base position size (dollar amount)
            base_size = max_position_value

            # 1. Volatility scaling
            if volatility > 0:
                vol_scaling = min(target_volatility / volatility, 2.0)  # Cap at 2x
            else:
                vol_scaling = 1.0

            # 2. Confidence scaling
            confidence_scaling = max(0.5, min(confidence, 1.0))  # 50% to 100%

            # Calculate scaled dollar amount
            scaled_value = base_size * vol_scaling * confidence_scaling

            # Convert to shares
            shares = int(scaled_value / price)

            # Apply hard limits
            shares = min(shares, max_shares)
            shares = max(1, shares)  # At least 1 share

            # Check capital constraint
            total_cost = shares * price
            if total_cost > capital * 0.9:  # Don't use more than 90% capital
                shares = int((capital * 0.9) / price)

            logger.debug(f"Position size: {shares} shares (vol_scale={vol_scaling:.2f}, conf={confidence:.2f})")
            return max(1, shares)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1

    def calculate_portfolio_var(
        self,
        positions: Dict[str, Tuple[int, float]],
        returns_data: Dict[str, pd.Series],
        confidence_level: float = None
    ) -> float:
        """
        Calculate portfolio-level VaR considering correlations.

        Args:
            positions: Dict of {symbol: (quantity, current_price)}
            returns_data: Dict of {symbol: returns_series}
            confidence_level: Confidence level

        Returns:
            Portfolio VaR in dollars
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        try:
            # Calculate position values
            position_values = {}
            total_value = 0.0

            for symbol, (qty, price) in positions.items():
                value = abs(qty * price)
                position_values[symbol] = value
                total_value += value

            if total_value == 0:
                return 0.0

            # Calculate position weights
            weights = {symbol: value / total_value for symbol, value in position_values.items()}

            # Get aligned returns
            symbols = list(positions.keys())
            returns_df = pd.DataFrame({s: returns_data.get(s, pd.Series()) for s in symbols})
            returns_df = returns_df.dropna()

            if len(returns_df) < 10:
                logger.warning("Insufficient data for portfolio VaR")
                return 0.0

            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            for symbol in symbols:
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol] * weights.get(symbol, 0)

            # Calculate VaR on portfolio returns
            var_pct = self.calculate_var(portfolio_returns, confidence_level)

            # Convert to dollars
            portfolio_var = total_value * var_pct

            logger.debug(f"Portfolio VaR: ${portfolio_var:,.2f} ({var_pct*100:.2f}%)")
            return float(portfolio_var)

        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0

    def calculate_marginal_var(
        self,
        symbol: str,
        position_value: float,
        portfolio_var: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate marginal VaR contribution of a position.

        Marginal VaR shows how much each position contributes to portfolio risk.

        Args:
            symbol: Stock symbol
            position_value: Dollar value of position
            portfolio_var: Current portfolio VaR
            portfolio_value: Total portfolio value

        Returns:
            Marginal VaR contribution (0-1)
        """
        if portfolio_value == 0 or portfolio_var == 0:
            return 0.0

        try:
            # Simple approximation: position_weight * portfolio_var
            weight = position_value / portfolio_value
            marginal_var = weight * portfolio_var

            return float(marginal_var)

        except Exception as e:
            logger.error(f"Error calculating marginal VaR: {e}")
            return 0.0

    def check_risk_limits(
        self,
        current_value: float,
        portfolio_var: float = None
    ) -> Dict:
        """
        Comprehensive risk limit check.

        Args:
            current_value: Current portfolio value
            portfolio_var: Portfolio VaR in dollars

        Returns:
            Dict with risk metrics and breach flags
        """
        try:
            current_value = float(current_value)

            # Daily loss check
            today = datetime.now().strftime('%Y-%m-%d')
            if today != self.current_day:
                self.current_day = today
                self.day_start_capital = current_value

            daily_pnl = current_value - self.day_start_capital
            daily_return = daily_pnl / self.day_start_capital if self.day_start_capital > 0 else 0
            daily_loss_breached = daily_return < -self.max_daily_loss

            # Drawdown check
            if current_value > self.peak_capital:
                self.peak_capital = current_value

            drawdown = (current_value - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
            drawdown_breached = drawdown < -self.max_drawdown

            # VaR check
            var_ratio = 0.0
            if portfolio_var and current_value > 0:
                var_ratio = portfolio_var / current_value

            return {
                'current_value': current_value,
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'daily_loss_breached': daily_loss_breached,
                'drawdown': abs(drawdown),
                'drawdown_breached': drawdown_breached,
                'peak_capital': self.peak_capital,
                'portfolio_var': portfolio_var or 0.0,
                'var_ratio': var_ratio,
                'limits': {
                    'max_daily_loss': self.max_daily_loss,
                    'max_drawdown': self.max_drawdown
                }
            }

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {
                'current_value': current_value,
                'daily_pnl': 0,
                'daily_return': 0,
                'daily_loss_breached': False,
                'drawdown': 0,
                'drawdown_breached': False
            }

    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.

        Returns:
            Dict with all risk metrics
        """
        return {
            'starting_capital': self.starting_capital,
            'peak_capital': self.peak_capital,
            'day_start_capital': self.day_start_capital,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'confidence_level': self.confidence_level,
            'var_method': self.var_method,
            'has_correlation_data': self.correlation_matrix is not None
        }
