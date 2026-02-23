"""
backtesting/market_impact.py - Dynamic Slippage and Market Impact Model

Realistic execution cost modeling that accounts for:
- Order size relative to volume
- Market volatility
- Time of day liquidity
- Bid-ask spread estimation
- Permanent and temporary market impact

Based on academic research:
- Almgren-Chriss market impact model
- Kyle's lambda for price impact
- Square-root law for execution costs
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCosts:
    """Breakdown of execution costs for an order."""
    total_cost_bps: float        # Total cost in basis points
    spread_cost_bps: float       # Bid-ask spread cost
    temporary_impact_bps: float  # Temporary price impact
    permanent_impact_bps: float  # Permanent price impact
    slippage_bps: float         # Random slippage component
    effective_price: float       # Actual execution price

    def total_cost_dollars(self, notional: float) -> float:
        """Get total cost in dollars."""
        return notional * (self.total_cost_bps / 10000.0)


@dataclass
class MarketConditions:
    """Current market conditions for impact calculation."""
    avg_daily_volume: float     # Average daily volume (shares)
    avg_daily_turnover: float   # Average daily turnover (dollars)
    volatility: float           # Daily volatility (decimal)
    bid_ask_spread_bps: float   # Typical bid-ask spread in bps
    current_volume_ratio: float # Current vs avg volume (1.0 = normal)
    time_of_day: Optional[time] = None


class MarketImpactModel:
    """
    Realistic market impact and slippage model for backtesting.

    Implements the Almgren-Chriss model with modifications for
    practical trading scenarios.

    Key principles:
    - Impact scales with sqrt(order_size / volume)
    - Volatility amplifies impact
    - Low liquidity periods have higher costs
    - Large orders have both temporary and permanent impact
    """

    # Model parameters (calibrated to typical US equity markets)
    TEMPORARY_IMPACT_COEF = 0.1     # Coefficient for temporary impact
    PERMANENT_IMPACT_COEF = 0.05   # Coefficient for permanent impact
    VOLATILITY_SCALING = 0.5       # How much volatility affects impact
    MIN_SPREAD_BPS = 1.0           # Minimum spread assumption
    MAX_IMPACT_BPS = 500.0         # Cap on total impact

    # Liquidity schedule (time of day factors)
    # Based on typical U-shaped intraday volume pattern
    LIQUIDITY_SCHEDULE = {
        (9, 30): 0.7,   # Market open - lower liquidity
        (10, 0): 1.0,   # Normal
        (11, 0): 1.1,   # Good liquidity
        (12, 0): 0.9,   # Lunch lull
        (13, 0): 0.85,  # Lunch lull
        (14, 0): 0.95,  # Picking up
        (15, 0): 1.1,   # Good liquidity
        (15, 30): 1.2,  # Pre-close liquidity
        (16, 0): 0.8,   # Close - can be erratic
    }

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        impact_multiplier: float = 1.0,
        random_slippage_std: float = 2.0
    ):
        """
        Initialize the market impact model.

        Args:
            base_spread_bps: Default bid-ask spread when unknown
            impact_multiplier: Scale factor for impact (1.0 = normal)
            random_slippage_std: Standard deviation of random slippage (bps)
        """
        self.base_spread_bps = base_spread_bps
        self.impact_multiplier = impact_multiplier
        self.random_slippage_std = random_slippage_std

        logger.info(f"MarketImpactModel initialized: "
                   f"base_spread={base_spread_bps}bps, "
                   f"multiplier={impact_multiplier}")

    def calculate_execution_costs(
        self,
        order_size_shares: int,
        price: float,
        side: str,  # 'BUY' or 'SELL'
        conditions: MarketConditions
    ) -> ExecutionCosts:
        """
        Calculate total execution costs for an order.

        Args:
            order_size_shares: Number of shares to trade
            price: Current market price
            side: 'BUY' or 'SELL'
            conditions: Market conditions for the stock

        Returns:
            ExecutionCosts with detailed breakdown
        """
        abs(order_size_shares) * price

        # 1. Spread Cost (half spread for crossing)
        spread_bps = max(conditions.bid_ask_spread_bps, self.MIN_SPREAD_BPS)
        spread_cost = spread_bps / 2.0  # Pay half spread

        # 2. Market Impact using square-root law
        # Impact = sigma * sqrt(Q/V) where Q is order size, V is volume
        if conditions.avg_daily_volume > 0:
            participation_rate = abs(order_size_shares) / conditions.avg_daily_volume
        else:
            participation_rate = 0.01  # Assume 1% if unknown

        # Temporary impact (reverts after execution)
        temp_impact = (
            self.TEMPORARY_IMPACT_COEF *
            conditions.volatility *
            np.sqrt(participation_rate) *
            10000  # Convert to bps
        )

        # Permanent impact (persists in the market)
        perm_impact = (
            self.PERMANENT_IMPACT_COEF *
            conditions.volatility *
            participation_rate *
            10000  # Convert to bps
        )

        # 3. Liquidity adjustment based on time of day
        liquidity_factor = self._get_liquidity_factor(conditions.time_of_day)
        if conditions.current_volume_ratio < 1.0:
            # Lower volume = higher impact
            liquidity_factor *= (1.0 / max(conditions.current_volume_ratio, 0.3))

        temp_impact *= liquidity_factor
        perm_impact *= liquidity_factor

        # 4. Random slippage (execution uncertainty)
        random_slippage = np.random.normal(0, self.random_slippage_std)

        # 5. Apply multiplier and cap
        temp_impact *= self.impact_multiplier
        perm_impact *= self.impact_multiplier

        total_impact = spread_cost + temp_impact + perm_impact + abs(random_slippage)
        total_impact = min(total_impact, self.MAX_IMPACT_BPS)

        # 6. Calculate effective execution price
        impact_pct = total_impact / 10000.0
        if side == 'BUY':
            effective_price = price * (1 + impact_pct)
        else:
            effective_price = price * (1 - impact_pct)

        return ExecutionCosts(
            total_cost_bps=total_impact,
            spread_cost_bps=spread_cost,
            temporary_impact_bps=temp_impact,
            permanent_impact_bps=perm_impact,
            slippage_bps=random_slippage,
            effective_price=effective_price
        )

    def calculate_slippage_bps(
        self,
        order_size_usd: float,
        avg_daily_volume_usd: float,
        volatility: float = 0.02,
        spread_bps: float = 5.0,
        time_of_day: Optional[time] = None
    ) -> float:
        """
        Simplified interface for slippage calculation.

        Args:
            order_size_usd: Order value in dollars
            avg_daily_volume_usd: Average daily turnover in dollars
            volatility: Daily volatility (default 2%)
            spread_bps: Bid-ask spread in basis points
            time_of_day: Time for liquidity adjustment

        Returns:
            Estimated slippage in basis points
        """
        # Estimate shares from dollar amounts (assuming $100 avg price)
        avg_price = 100.0
        order_shares = order_size_usd / avg_price
        avg_volume = avg_daily_volume_usd / avg_price

        conditions = MarketConditions(
            avg_daily_volume=avg_volume,
            avg_daily_turnover=avg_daily_volume_usd,
            volatility=volatility,
            bid_ask_spread_bps=spread_bps,
            current_volume_ratio=1.0,
            time_of_day=time_of_day
        )

        costs = self.calculate_execution_costs(
            order_size_shares=int(order_shares),
            price=avg_price,
            side='BUY',
            conditions=conditions
        )

        return costs.total_cost_bps

    def _get_liquidity_factor(self, t: Optional[time]) -> float:
        """Get liquidity factor for time of day."""
        if t is None:
            return 1.0


        # Find nearest scheduled time
        closest_factor = 1.0
        min_diff = float('inf')

        for sched_time, factor in self.LIQUIDITY_SCHEDULE.items():
            diff = abs(sched_time[0] * 60 + sched_time[1] - t.hour * 60 - t.minute)
            if diff < min_diff:
                min_diff = diff
                closest_factor = factor

        return closest_factor

    def estimate_from_historical(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        order_size_shares: int,
        price: float,
        side: str
    ) -> ExecutionCosts:
        """
        Estimate execution costs using historical data.

        Args:
            symbol: Stock symbol
            historical_data: DataFrame with OHLCV data
            order_size_shares: Number of shares
            price: Current price
            side: 'BUY' or 'SELL'

        Returns:
            ExecutionCosts estimate
        """
        # Calculate average daily volume
        if 'volume' in historical_data.columns:
            avg_volume = historical_data['volume'].tail(20).mean()
        elif 'Volume' in historical_data.columns:
            avg_volume = historical_data['Volume'].tail(20).mean()
        else:
            avg_volume = 1_000_000  # Default 1M shares

        # Calculate volatility
        if 'close' in historical_data.columns:
            close = historical_data['close']
        elif 'Close' in historical_data.columns:
            close = historical_data['Close']
        else:
            close = pd.Series([price])

        returns = close.pct_change().dropna()
        volatility = returns.tail(20).std() if len(returns) >= 20 else 0.02

        # Estimate spread from high-low range
        if 'high' in historical_data.columns and 'low' in historical_data.columns:
            high = historical_data['high'].tail(20)
            low = historical_data['low'].tail(20)
            avg_range = ((high - low) / close.tail(20)).mean()
            spread_bps = max(avg_range * 10000 / 4, self.MIN_SPREAD_BPS)
        else:
            spread_bps = self.base_spread_bps

        conditions = MarketConditions(
            avg_daily_volume=avg_volume,
            avg_daily_turnover=avg_volume * price,
            volatility=volatility,
            bid_ask_spread_bps=spread_bps,
            current_volume_ratio=1.0,
            time_of_day=datetime.now().time()
        )

        return self.calculate_execution_costs(
            order_size_shares=order_size_shares,
            price=price,
            side=side,
            conditions=conditions
        )


class AdaptiveSlippageModel:
    """
    Adaptive slippage model that learns from actual execution data.

    Maintains running estimates of execution quality by symbol
    and adjusts predictions based on recent fills.
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize adaptive model.

        Args:
            learning_rate: How fast to adapt to new observations
        """
        self.learning_rate = learning_rate
        self.base_model = MarketImpactModel()

        # Per-symbol adjustments learned from execution
        self.symbol_adjustments: Dict[str, float] = {}
        self.execution_history: Dict[str, list] = {}

    def predict_slippage(
        self,
        symbol: str,
        order_size_usd: float,
        avg_daily_volume_usd: float,
        volatility: float = 0.02
    ) -> float:
        """
        Predict slippage using base model + learned adjustments.

        Returns:
            Predicted slippage in basis points
        """
        base_slippage = self.base_model.calculate_slippage_bps(
            order_size_usd=order_size_usd,
            avg_daily_volume_usd=avg_daily_volume_usd,
            volatility=volatility
        )

        # Apply symbol-specific adjustment
        adjustment = self.symbol_adjustments.get(symbol, 1.0)
        return base_slippage * adjustment

    def record_execution(
        self,
        symbol: str,
        predicted_slippage_bps: float,
        actual_slippage_bps: float
    ):
        """
        Record an execution to update the model.

        Args:
            symbol: Stock symbol
            predicted_slippage_bps: What we predicted
            actual_slippage_bps: What actually happened
        """
        if symbol not in self.execution_history:
            self.execution_history[symbol] = []

        self.execution_history[symbol].append({
            'predicted': predicted_slippage_bps,
            'actual': actual_slippage_bps
        })

        # Keep last 50 executions
        self.execution_history[symbol] = self.execution_history[symbol][-50:]

        # Update adjustment factor
        if predicted_slippage_bps > 0:
            ratio = actual_slippage_bps / predicted_slippage_bps
            current_adj = self.symbol_adjustments.get(symbol, 1.0)
            new_adj = current_adj * (1 - self.learning_rate) + ratio * self.learning_rate
            self.symbol_adjustments[symbol] = np.clip(new_adj, 0.5, 2.0)

    def get_accuracy_stats(self, symbol: Optional[str] = None) -> Dict:
        """Get prediction accuracy statistics."""
        history = self.execution_history

        if symbol:
            history = {symbol: history.get(symbol, [])}

        stats = {}
        for sym, executions in history.items():
            if not executions:
                continue

            predicted = [e['predicted'] for e in executions]
            actual = [e['actual'] for e in executions]

            errors = [abs(p - a) for p, a in zip(predicted, actual)]

            stats[sym] = {
                'num_executions': len(executions),
                'mean_error_bps': np.mean(errors),
                'median_error_bps': np.median(errors),
                'mean_actual_bps': np.mean(actual),
                'adjustment_factor': self.symbol_adjustments.get(sym, 1.0)
            }

        return stats


# Convenience function for backtest integration
def calculate_realistic_slippage(
    order_size_usd: float,
    avg_daily_volume_usd: float,
    volatility: float = 0.02,
    spread_bps: float = 5.0
) -> float:
    """
    Calculate realistic slippage for backtesting.

    This is the main entry point for backtest engines.

    Args:
        order_size_usd: Order size in dollars
        avg_daily_volume_usd: Average daily volume in dollars
        volatility: Daily volatility (decimal)
        spread_bps: Estimated bid-ask spread

    Returns:
        Slippage in basis points
    """
    model = MarketImpactModel()
    return model.calculate_slippage_bps(
        order_size_usd=order_size_usd,
        avg_daily_volume_usd=avg_daily_volume_usd,
        volatility=volatility,
        spread_bps=spread_bps
    )
