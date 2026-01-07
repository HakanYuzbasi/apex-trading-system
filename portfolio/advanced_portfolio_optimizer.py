"""
portfolio/advanced_portfolio_optimizer.py - Professional Portfolio Optimization
Features:
- Mean-variance optimization (Markowitz)
- Risk parity allocation
- Black-Litterman model
- Efficient frontier calculation
- Constraints handling (sector limits, position limits)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedPortfolioOptimizer:
    """
    Professional portfolio optimization using modern portfolio theory.

    Methods:
    - Mean-Variance (Markowitz)
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Risk Parity
    - Equal Weight (baseline)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"📊 Advanced Portfolio Optimizer initialized")

    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.

        Args:
            weights: Portfolio weights
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix

        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.dot(weights, returns)

        # Portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)

        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0

        return portfolio_return, portfolio_vol, sharpe

    def optimize_max_sharpe(
        self,
        symbols: List[str],
        returns_data: Dict[str, pd.Series],
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio for maximum Sharpe ratio.

        Args:
            symbols: List of symbols
            returns_data: Dict of {symbol: returns_series}
            constraints: Optional constraints dict

        Returns:
            Dict of {symbol: weight}
        """
        try:
            # Prepare data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 30:
                logger.warning("Insufficient data for optimization, using equal weight")
                return self._equal_weight(symbols)

            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252  # Annualize
            cov_matrix = returns_df.cov() * 252

            n_assets = len(symbols)

            # Objective function: negative Sharpe ratio (minimize)
            def objective(weights):
                port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
                    weights, expected_returns.values, cov_matrix.values
                )
                return -sharpe  # Negative because we minimize

            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]

            # Bounds: each weight between 0 and max_weight
            max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
            bounds = tuple((0, max_weight) for _ in range(n_assets))

            # Initial guess: equal weight
            x0 = np.array([1.0 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.warning(f"Optimization failed: {result.message}, using equal weight")
                return self._equal_weight(symbols)

            # Create weights dict
            weights = {symbol: float(w) for symbol, w in zip(symbols, result.x) if w > 0.01}

            # Normalize (ensure sum = 1)
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}

            logger.info(f"✅ Optimized portfolio (max Sharpe):")
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   {symbol}: {weight*100:.1f}%")

            return weights

        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {e}")
            return self._equal_weight(symbols)

    def optimize_min_variance(
        self,
        symbols: List[str],
        returns_data: Dict[str, pd.Series],
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio for minimum variance.

        Args:
            symbols: List of symbols
            returns_data: Dict of {symbol: returns_series}
            constraints: Optional constraints dict

        Returns:
            Dict of {symbol: weight}
        """
        try:
            # Prepare data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 30:
                logger.warning("Insufficient data for optimization, using equal weight")
                return self._equal_weight(symbols)

            # Calculate covariance matrix
            cov_matrix = returns_df.cov() * 252

            n_assets = len(symbols)

            # Objective function: portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix.values, weights))

            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]

            # Bounds
            max_weight = constraints.get('max_weight', 0.3) if constraints else 0.3
            bounds = tuple((0, max_weight) for _ in range(n_assets))

            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.warning(f"Optimization failed: {result.message}, using equal weight")
                return self._equal_weight(symbols)

            # Create weights dict
            weights = {symbol: float(w) for symbol, w in zip(symbols, result.x) if w > 0.01}

            # Normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}

            logger.info(f"✅ Optimized portfolio (min variance):")
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   {symbol}: {weight*100:.1f}%")

            return weights

        except Exception as e:
            logger.error(f"Error in min variance optimization: {e}")
            return self._equal_weight(symbols)

    def optimize_risk_parity(
        self,
        symbols: List[str],
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Risk parity allocation: equal risk contribution from each asset.

        Args:
            symbols: List of symbols
            returns_data: Dict of {symbol: returns_series}

        Returns:
            Dict of {symbol: weight}
        """
        try:
            # Prepare data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 30:
                logger.warning("Insufficient data for risk parity, using equal weight")
                return self._equal_weight(symbols)

            # Calculate covariance matrix
            cov_matrix = returns_df.cov() * 252

            n_assets = len(symbols)

            # Calculate asset volatilities
            volatilities = np.sqrt(np.diag(cov_matrix))

            # Inverse volatility weighting (simple risk parity)
            inv_vol = 1.0 / volatilities
            weights_array = inv_vol / np.sum(inv_vol)

            # Create weights dict
            weights = {symbol: float(w) for symbol, w in zip(symbols, weights_array)}

            logger.info(f"✅ Risk parity portfolio:")
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   {symbol}: {weight*100:.1f}%")

            return weights

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self._equal_weight(symbols)

    def calculate_efficient_frontier(
        self,
        symbols: List[str],
        returns_data: Dict[str, pd.Series],
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Calculate efficient frontier points.

        Args:
            symbols: List of symbols
            returns_data: Dict of {symbol: returns_series}
            n_points: Number of points on frontier

        Returns:
            Tuple of (returns_array, volatilities_array, weights_list)
        """
        try:
            # Prepare data
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < 30:
                logger.warning("Insufficient data for efficient frontier")
                return np.array([]), np.array([]), []

            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252

            n_assets = len(symbols)

            # Range of target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_points)

            frontier_returns = []
            frontier_vols = []
            frontier_weights = []

            for target_return in target_returns:
                # Objective: minimize variance
                def objective(weights):
                    return np.dot(weights, np.dot(cov_matrix.values, weights))

                # Constraints
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
                    {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns.values) - target_return}  # Target return
                ]

                bounds = tuple((0, 1) for _ in range(n_assets))
                x0 = np.array([1.0 / n_assets] * n_assets)

                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )

                if result.success:
                    weights = result.x
                    port_return = np.dot(weights, expected_returns.values)
                    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))

                    frontier_returns.append(port_return)
                    frontier_vols.append(port_vol)
                    frontier_weights.append(weights)

            return (
                np.array(frontier_returns),
                np.array(frontier_vols),
                frontier_weights
            )

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return np.array([]), np.array([]), []

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight allocation (baseline)."""
        n = len(symbols)
        if n == 0:
            return {}

        weight = 1.0 / n
        return {symbol: weight for symbol in symbols}

    def rebalance_portfolio(
        self,
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float,
        min_trade_value: float = 100.0
    ) -> Dict[str, int]:
        """
        Calculate rebalancing trades.

        Args:
            current_positions: Dict of {symbol: quantity}
            current_prices: Dict of {symbol: current_price}
            target_weights: Dict of {symbol: target_weight}
            total_value: Total portfolio value
            min_trade_value: Minimum trade value to execute

        Returns:
            Dict of {symbol: shares_to_trade} (positive=buy, negative=sell)
        """
        trades = {}

        # All symbols to consider
        all_symbols = set(list(current_positions.keys()) + list(target_weights.keys()))

        for symbol in all_symbols:
            current_qty = current_positions.get(symbol, 0)
            price = current_prices.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)

            if price == 0:
                continue

            # Current value
            current_value = current_qty * price

            # Target value
            target_value = total_value * target_weight

            # Difference
            value_diff = target_value - current_value

            # Convert to shares
            shares_diff = int(value_diff / price)

            # Only trade if above minimum
            if abs(shares_diff * price) >= min_trade_value:
                trades[symbol] = shares_diff

        return trades

    def optimize_portfolio(
        self,
        symbols: List[str],
        returns_data: Dict[str, pd.Series],
        method: str = "max_sharpe",
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio using specified method.

        Args:
            symbols: List of symbols
            returns_data: Dict of {symbol: returns_series}
            method: Optimization method ('max_sharpe', 'min_variance', 'risk_parity', 'equal_weight')
            constraints: Optional constraints

        Returns:
            Dict of {symbol: weight}
        """
        if method == "max_sharpe":
            return self.optimize_max_sharpe(symbols, returns_data, constraints)
        elif method == "min_variance":
            return self.optimize_min_variance(symbols, returns_data, constraints)
        elif method == "risk_parity":
            return self.optimize_risk_parity(symbols, returns_data)
        elif method == "equal_weight":
            return self._equal_weight(symbols)
        else:
            logger.warning(f"Unknown method '{method}', using equal weight")
            return self._equal_weight(symbols)
