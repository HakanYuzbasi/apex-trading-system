"""
portfolio/portfolio_optimizer.py - Portfolio Optimization and Rebalancing

Features:
- Equal-weight and risk-parity optimization
- Drift-based rebalancing triggers
- Scheduled rebalancing support
- Sector exposure constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from config import ApexConfig

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimize portfolio allocation and handle rebalancing.

    Supports:
    - Equal-weight optimization
    - Risk-parity optimization (coming soon)
    - Drift-based rebalancing
    - Time-based rebalancing
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        self.risk_free_rate = risk_free_rate
        self.target_weights: Dict[str, float] = {}
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_history: List[Dict] = []

        logger.info(f"📊 Portfolio Optimizer initialized")
        if ApexConfig.REBALANCE_ENABLED:
            logger.info(f"   Rebalancing: ENABLED")
            logger.info(f"   Drift Threshold: {ApexConfig.REBALANCE_DRIFT_THRESHOLD*100:.1f}%")
            logger.info(f"   Min Interval: {ApexConfig.REBALANCE_MIN_INTERVAL_HOURS}h")

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of daily returns

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - self.risk_free_rate / 252
        return float(excess_returns / returns.std() * np.sqrt(252))

    def calculate_current_weights(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current portfolio weights.

        Args:
            positions: Dict of {symbol: quantity}
            prices: Dict of {symbol: current_price}

        Returns:
            Dict of {symbol: weight} where weights sum to 1
        """
        total_value = 0.0
        position_values = {}

        for symbol, qty in positions.items():
            if qty != 0:
                price = prices.get(symbol, 0)
                value = abs(qty) * price
                position_values[symbol] = value
                total_value += value

        if total_value == 0:
            return {}

        return {symbol: value / total_value for symbol, value in position_values.items()}

    def optimize_weights(
        self,
        symbols: List[str],
        returns_data: Optional[Dict[str, pd.Series]] = None,
        method: str = 'equal'
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of symbols to include
            returns_data: Optional historical returns for each symbol
            method: Optimization method ('equal', 'risk_parity', 'min_variance')

        Returns:
            Dict of {symbol: target_weight}
        """
        n = len(symbols)
        if n == 0:
            return {}

        if method == 'equal':
            # Equal weight allocation
            weights = {symbol: 1.0 / n for symbol in symbols}

        elif method == 'risk_parity' and returns_data:
            # Risk parity: weight inversely proportional to volatility
            volatilities = {}
            for symbol in symbols:
                if symbol in returns_data:
                    returns = returns_data[symbol]
                    vol = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.2
                    volatilities[symbol] = max(vol, 0.01)  # Minimum volatility
                else:
                    volatilities[symbol] = 0.2  # Default volatility

            # Inverse volatility weights
            inv_vols = {s: 1.0 / v for s, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}

        else:
            # Default to equal weight
            weights = {symbol: 1.0 / n for symbol in symbols}

        self.target_weights = weights
        logger.debug(f"📊 Optimized weights ({method}): {len(weights)} positions")

        return weights

    def calculate_drift(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate drift from target weights.

        Args:
            positions: Current positions
            prices: Current prices
            target_weights: Target weights (uses stored if None)

        Returns:
            Dict of {symbol: drift} where drift = current_weight - target_weight
        """
        if target_weights is None:
            target_weights = self.target_weights

        if not target_weights:
            return {}

        current_weights = self.calculate_current_weights(positions, prices)

        drift = {}
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift[symbol] = current - target

        return drift

    def needs_rebalance(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if portfolio needs rebalancing.

        Args:
            positions: Current positions
            prices: Current prices
            target_weights: Target weights

        Returns:
            Tuple of (needs_rebalance: bool, reason: str)
        """
        if not ApexConfig.REBALANCE_ENABLED:
            return False, "Rebalancing disabled"

        if not target_weights and not self.target_weights:
            return False, "No target weights set"

        # Check minimum interval
        if self.last_rebalance_time:
            hours_since = (datetime.now() - self.last_rebalance_time).total_seconds() / 3600
            if hours_since < ApexConfig.REBALANCE_MIN_INTERVAL_HOURS:
                return False, f"Too soon ({hours_since:.1f}h < {ApexConfig.REBALANCE_MIN_INTERVAL_HOURS}h)"

        # Check drift
        drift = self.calculate_drift(positions, prices, target_weights)

        if not drift:
            return False, "No positions to rebalance"

        max_drift = max(abs(d) for d in drift.values())

        if max_drift > ApexConfig.REBALANCE_DRIFT_THRESHOLD:
            max_drift_symbol = max(drift.keys(), key=lambda s: abs(drift[s]))
            return True, f"Max drift {max_drift*100:.1f}% on {max_drift_symbol}"

        return False, f"Drift OK (max {max_drift*100:.1f}%)"

    def calculate_rebalance_trades(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        total_value: float,
        target_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, int]:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            positions: Current positions {symbol: qty}
            prices: Current prices {symbol: price}
            total_value: Total portfolio value
            target_weights: Target weights {symbol: weight}

        Returns:
            Dict of {symbol: trade_qty} where positive=buy, negative=sell
        """
        if target_weights is None:
            target_weights = self.target_weights

        if not target_weights:
            return {}

        trades = {}

        for symbol, target_weight in target_weights.items():
            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            target_value = total_value * target_weight
            target_shares = int(target_value / price)

            current_shares = positions.get(symbol, 0)
            trade_qty = target_shares - current_shares

            # Only include significant trades (> $500 or > 5 shares)
            trade_value = abs(trade_qty * price)
            if abs(trade_qty) >= 5 or trade_value >= 500:
                trades[symbol] = trade_qty

        # Also check for positions we should exit (not in target)
        for symbol, qty in positions.items():
            if symbol not in target_weights and qty != 0:
                trades[symbol] = -qty  # Close position

        return trades

    def should_rebalance_now(self, est_hour: float) -> bool:
        """
        Check if current time is appropriate for rebalancing.

        Args:
            est_hour: Current hour in EST

        Returns:
            True if good time to rebalance
        """
        if ApexConfig.REBALANCE_AT_MARKET_CLOSE:
            # Prefer rebalancing 30 min before close (3:30 PM EST = 15.5)
            return 15.0 <= est_hour <= 15.75

        # Otherwise any market hour is fine
        return ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END

    def record_rebalance(
        self,
        trades: Dict[str, int],
        before_weights: Dict[str, float],
        after_weights: Dict[str, float]
    ):
        """Record rebalancing event for history."""
        self.last_rebalance_time = datetime.now()

        record = {
            'timestamp': self.last_rebalance_time.isoformat(),
            'trades': trades,
            'before_weights': before_weights,
            'after_weights': after_weights,
            'num_trades': len(trades)
        }

        self.rebalance_history.append(record)

        # Keep only last 50 rebalances
        if len(self.rebalance_history) > 50:
            self.rebalance_history = self.rebalance_history[-50:]

        logger.info(f"📊 Rebalance recorded: {len(trades)} trades")

    def get_rebalance_summary(self) -> Dict:
        """Get rebalancing summary for dashboard."""
        return {
            'enabled': ApexConfig.REBALANCE_ENABLED,
            'drift_threshold': ApexConfig.REBALANCE_DRIFT_THRESHOLD,
            'min_interval_hours': ApexConfig.REBALANCE_MIN_INTERVAL_HOURS,
            'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
            'total_rebalances': len(self.rebalance_history),
            'target_weights': self.target_weights
        }
