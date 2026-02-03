"""
execution/arrival_price_benchmark.py - Implementation Shortfall Analysis

Tracks execution quality by comparing:
- Arrival price (decision price) vs fill price
- Delay cost (price movement before execution)
- Market impact cost

State-of-the-art execution analysis for trading systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single execution for analysis."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    arrival_price: float  # Price at decision time
    fill_price: float  # Actual fill price
    fill_time: datetime
    decision_time: datetime
    order_type: str = 'MARKET'
    venue: str = 'SMART'
    
    @property
    def implementation_shortfall_bps(self) -> float:
        """Calculate implementation shortfall in basis points."""
        if self.arrival_price == 0:
            return 0.0
        
        if self.side == 'BUY':
            # For buys, shortfall = (fill - arrival) / arrival
            shortfall = (self.fill_price - self.arrival_price) / self.arrival_price
        else:
            # For sells, shortfall = (arrival - fill) / arrival
            shortfall = (self.arrival_price - self.fill_price) / self.arrival_price
        
        return shortfall * 10000  # Convert to bps
    
    @property
    def delay_seconds(self) -> float:
        """Time between decision and fill."""
        return (self.fill_time - self.decision_time).total_seconds()


@dataclass
class ExecutionSummary:
    """Summary statistics for execution quality."""
    total_trades: int
    total_volume: float
    avg_shortfall_bps: float
    median_shortfall_bps: float
    std_shortfall_bps: float
    worst_shortfall_bps: float
    best_shortfall_bps: float
    avg_delay_seconds: float
    total_cost_estimate: float  # Estimated dollar cost of poor execution
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'avg_shortfall_bps': round(self.avg_shortfall_bps, 2),
            'median_shortfall_bps': round(self.median_shortfall_bps, 2),
            'std_shortfall_bps': round(self.std_shortfall_bps, 2),
            'worst_shortfall_bps': round(self.worst_shortfall_bps, 2),
            'best_shortfall_bps': round(self.best_shortfall_bps, 2),
            'avg_delay_seconds': round(self.avg_delay_seconds, 2),
            'total_cost_estimate': round(self.total_cost_estimate, 2)
        }


class ArrivalPriceBenchmark:
    """
    Track and analyze execution quality using arrival price benchmark.
    
    The arrival price is the market price at the moment the trading
    decision is made. Comparing this to actual fill price reveals
    the true cost of execution.
    
    Components of implementation shortfall:
    1. Delay cost: Price movement between decision and order placement
    2. Market impact: Price movement caused by the order itself
    3. Timing cost: Price movement during execution
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize benchmark tracker.
        
        Args:
            max_history: Maximum number of executions to store
        """
        self.max_history = max_history
        self.executions: List[ExecutionRecord] = []
        self.daily_summaries: Dict[str, ExecutionSummary] = {}
        
        logger.info("ArrivalPriceBenchmark initialized")
    
    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        arrival_price: float,
        fill_price: float,
        fill_time: Optional[datetime] = None,
        decision_time: Optional[datetime] = None,
        order_type: str = 'MARKET',
        venue: str = 'SMART'
    ):
        """
        Record an execution for analysis.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            arrival_price: Price at decision time
            fill_price: Actual fill price
            fill_time: When order was filled (default: now)
            decision_time: When decision was made (default: now)
            order_type: Order type used
            venue: Execution venue
        """
        now = datetime.now()
        
        record = ExecutionRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            arrival_price=arrival_price,
            fill_price=fill_price,
            fill_time=fill_time or now,
            decision_time=decision_time or now,
            order_type=order_type,
            venue=venue
        )
        
        self.executions.append(record)
        
        # Trim history if needed
        if len(self.executions) > self.max_history:
            self.executions = self.executions[-self.max_history:]
        
        # Log significant shortfall
        shortfall = record.implementation_shortfall_bps
        if abs(shortfall) > 10:  # More than 10 bps
            logger.warning(
                f"High shortfall: {symbol} {side} {quantity} shares, "
                f"{shortfall:.1f} bps (arrival: ${arrival_price:.2f}, fill: ${fill_price:.2f})"
            )
        else:
            logger.debug(
                f"Execution: {symbol} {side}, shortfall: {shortfall:.1f} bps"
            )
    
    def get_summary(
        self,
        days: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> ExecutionSummary:
        """
        Get execution quality summary.
        
        Args:
            days: Number of days to include (None = all)
            symbol: Filter by symbol (None = all)
        
        Returns:
            ExecutionSummary with statistics
        """
        # Filter executions
        filtered = self.executions
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            filtered = [e for e in filtered if e.fill_time >= cutoff]
        
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        
        if not filtered:
            return ExecutionSummary(
                total_trades=0,
                total_volume=0,
                avg_shortfall_bps=0,
                median_shortfall_bps=0,
                std_shortfall_bps=0,
                worst_shortfall_bps=0,
                best_shortfall_bps=0,
                avg_delay_seconds=0,
                total_cost_estimate=0
            )
        
        # Calculate statistics
        shortfalls = [e.implementation_shortfall_bps for e in filtered]
        delays = [e.delay_seconds for e in filtered]
        volumes = [e.quantity * e.fill_price for e in filtered]
        
        total_volume = sum(volumes)
        
        # Cost estimate: shortfall * volume / 10000
        costs = [
            e.implementation_shortfall_bps * e.quantity * e.fill_price / 10000
            for e in filtered
        ]
        
        return ExecutionSummary(
            total_trades=len(filtered),
            total_volume=total_volume,
            avg_shortfall_bps=float(np.mean(shortfalls)),
            median_shortfall_bps=float(np.median(shortfalls)),
            std_shortfall_bps=float(np.std(shortfalls)),
            worst_shortfall_bps=float(max(shortfalls)),
            best_shortfall_bps=float(min(shortfalls)),
            avg_delay_seconds=float(np.mean(delays)),
            total_cost_estimate=sum(costs)
        )
    
    def get_summary_by_symbol(self) -> Dict[str, ExecutionSummary]:
        """Get execution summary grouped by symbol."""
        symbols = set(e.symbol for e in self.executions)
        return {symbol: self.get_summary(symbol=symbol) for symbol in symbols}
    
    def get_summary_by_venue(self) -> Dict[str, ExecutionSummary]:
        """Get execution summary grouped by venue."""
        summaries = {}
        venues = set(e.venue for e in self.executions)
        
        for venue in venues:
            filtered = [e for e in self.executions if e.venue == venue]
            if filtered:
                shortfalls = [e.implementation_shortfall_bps for e in filtered]
                summaries[venue] = {
                    'trades': len(filtered),
                    'avg_shortfall_bps': float(np.mean(shortfalls)),
                    'std_shortfall_bps': float(np.std(shortfalls))
                }
        
        return summaries
    
    def estimate_pre_trade_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        urgency: str = 'medium'
    ) -> Dict[str, float]:
        """
        Estimate transaction costs before trading.
        
        Uses historical execution data to predict costs.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            current_price: Current market price
            urgency: 'low', 'medium', 'high'
        
        Returns:
            Dict with cost estimates in basis points
        """
        # Get historical shortfall for this symbol
        symbol_executions = [e for e in self.executions if e.symbol == symbol]
        
        if symbol_executions:
            historical_shortfall = np.mean([
                e.implementation_shortfall_bps for e in symbol_executions
            ])
        else:
            # Use overall average
            if self.executions:
                historical_shortfall = np.mean([
                    e.implementation_shortfall_bps for e in self.executions
                ])
            else:
                historical_shortfall = 5.0  # Default 5 bps
        
        # Urgency adjustment
        urgency_multipliers = {
            'low': 0.7,    # Patient execution
            'medium': 1.0,  # Normal
            'high': 1.5     # Aggressive execution
        }
        urgency_mult = urgency_multipliers.get(urgency, 1.0)
        
        # Size adjustment (larger orders have more impact)
        order_value = quantity * current_price
        size_impact = min(1.0, order_value / 50000) * 2  # 2 bps per $50k
        
        # Estimate components
        spread_cost = 2.0  # Assume 2 bps half-spread
        market_impact = historical_shortfall * urgency_mult + size_impact
        
        total_cost_bps = spread_cost + market_impact
        total_cost_dollars = total_cost_bps * order_value / 10000
        
        return {
            'spread_cost_bps': spread_cost,
            'market_impact_bps': market_impact,
            'total_cost_bps': total_cost_bps,
            'total_cost_dollars': total_cost_dollars,
            'order_value': order_value
        }
    
    def should_use_algo(
        self,
        quantity: int,
        price: float,
        avg_daily_volume: float = 1_000_000
    ) -> Tuple[bool, str]:
        """
        Recommend whether to use algorithmic execution.
        
        Args:
            quantity: Order size
            price: Current price
            avg_daily_volume: Average daily volume
        
        Returns:
            Tuple of (should_use_algo, recommended_algo)
        """
        order_value = quantity * price
        participation_rate = quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        
        if participation_rate > 0.10:  # >10% of daily volume
            return True, 'VWAP'
        elif participation_rate > 0.05:  # >5% of daily volume
            return True, 'TWAP'
        elif order_value > 100_000:  # >$100k
            return True, 'TWAP'
        else:
            return False, 'MARKET'
    
    def generate_report(self) -> str:
        """Generate text report of execution quality."""
        summary = self.get_summary()
        
        report = [
            "=" * 60,
            "EXECUTION QUALITY REPORT",
            "=" * 60,
            f"Total Trades: {summary.total_trades}",
            f"Total Volume: ${summary.total_volume:,.2f}",
            "",
            "Implementation Shortfall:",
            f"  Average: {summary.avg_shortfall_bps:.2f} bps",
            f"  Median:  {summary.median_shortfall_bps:.2f} bps",
            f"  Std Dev: {summary.std_shortfall_bps:.2f} bps",
            f"  Worst:   {summary.worst_shortfall_bps:.2f} bps",
            f"  Best:    {summary.best_shortfall_bps:.2f} bps",
            "",
            f"Average Delay: {summary.avg_delay_seconds:.1f} seconds",
            f"Estimated Cost: ${summary.total_cost_estimate:,.2f}",
            "=" * 60
        ]
        
        return "\n".join(report)
