"""
execution/execution_shield.py - Smart Execution & Slippage Protection

Unified execution wrapper that:
1. Selects optimal execution algorithm based on order size
2. Monitors slippage per symbol
3. Adjusts position sizing based on execution quality

Algorithm selection:
< $10K      → MARKET (instant)
$10K-$50K   → LIMIT (30-60s)
$50K-$200K  → TWAP (5 min)
> $200K     → VWAP (15 min)
Emergency   → MARKET (accept any slippage)

Slippage monitoring:
- Avg slippage > 15bps → reduce size 20%
- Avg slippage > 30bps → flag as expensive, require higher conviction
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class ExecutionAlgo(Enum):
    MARKET = "market"
    LIMIT = "limit"
    AGGRESSIVE_LIMIT = "aggressive_limit"
    TWAP = "twap"
    VWAP = "vwap"


class Urgency(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionStrategy:
    """Selected execution strategy for an order."""
    algo: ExecutionAlgo
    urgency: Urgency
    max_slippage_bps: float
    time_horizon_seconds: int
    use_dark_pool: bool = False


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    fill_price: float
    slippage_bps: float
    market_impact_bps: float
    execution_time_seconds: float
    algo_used: ExecutionAlgo
    venue: str = "SMART"
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutionShield:
    """
    Smart execution wrapper with slippage monitoring.

    Automatically selects the best execution algorithm based on
    order characteristics and tracks execution quality per symbol.
    """

    def __init__(
        self,
        twap_threshold: float = 50_000,
        vwap_threshold: float = 200_000,
        max_slippage_bps: float = 15.0,
        critical_slippage_bps: float = 30.0,
        slippage_history_size: int = 50,
    ):
        self.twap_threshold = twap_threshold
        self.vwap_threshold = vwap_threshold
        self.max_slippage_bps = max_slippage_bps
        self.critical_slippage_bps = critical_slippage_bps

        # Per-symbol slippage history
        self._slippage_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=slippage_history_size)
        )

        # Expensive execution list
        self._expensive_symbols: set = set()

        # Execution stats
        self._total_executions = 0
        self._total_slippage_bps = 0.0

        logger.info(
            f"ExecutionShield initialized: "
            f"twap_threshold=${twap_threshold:,.0f}, "
            f"vwap_threshold=${vwap_threshold:,.0f}"
        )

    def select_strategy(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        urgency: Urgency = Urgency.NORMAL,
    ) -> ExecutionStrategy:
        """
        Select optimal execution strategy for an order.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            shares: Number of shares
            price: Current price
            urgency: Order urgency level

        Returns:
            ExecutionStrategy with algorithm and parameters
        """
        order_value = shares * price

        # Critical urgency always uses market order
        if urgency == Urgency.CRITICAL:
            return ExecutionStrategy(
                algo=ExecutionAlgo.MARKET,
                urgency=urgency,
                max_slippage_bps=100.0,  # Accept high slippage
                time_horizon_seconds=0,
            )

        # Select algo based on order value
        if order_value >= self.vwap_threshold:
            algo = ExecutionAlgo.VWAP
            horizon = 900  # 15 min
            max_slip = 10.0
        elif order_value >= self.twap_threshold:
            algo = ExecutionAlgo.TWAP
            horizon = 300  # 5 min
            max_slip = 12.0
        elif order_value >= 10_000:
            if urgency == Urgency.HIGH:
                algo = ExecutionAlgo.AGGRESSIVE_LIMIT
                horizon = 30
            else:
                algo = ExecutionAlgo.LIMIT
                horizon = 60
            max_slip = 15.0
        else:
            algo = ExecutionAlgo.MARKET
            horizon = 0
            max_slip = 20.0

        # Use dark pool for large orders
        use_dark = order_value >= 100_000

        return ExecutionStrategy(
            algo=algo,
            urgency=urgency,
            max_slippage_bps=max_slip,
            time_horizon_seconds=horizon,
            use_dark_pool=use_dark,
        )

    def record_execution(
        self,
        symbol: str,
        expected_price: float,
        fill_price: float,
        shares: int,
        algo: ExecutionAlgo,
        execution_time: float = 0.0,
    ) -> ExecutionResult:
        """
        Record an execution and compute slippage.

        Args:
            symbol: Stock symbol
            expected_price: Price at order submission
            fill_price: Actual fill price
            shares: Shares executed
            algo: Algorithm used
            execution_time: Time to fill in seconds

        Returns:
            ExecutionResult with slippage metrics
        """
        if expected_price <= 0:
            slippage_bps = 0.0
        else:
            slippage_bps = abs(fill_price - expected_price) / expected_price * 10000

        # Record in history
        self._slippage_history[symbol].append(slippage_bps)

        # Update totals
        self._total_executions += 1
        self._total_slippage_bps += slippage_bps

        # Check if symbol should be flagged as expensive
        avg_slip = self.get_avg_slippage(symbol)
        if avg_slip > self.critical_slippage_bps:
            if symbol not in self._expensive_symbols:
                self._expensive_symbols.add(symbol)
                logger.warning(
                    f"Symbol {symbol} flagged as expensive: "
                    f"avg slippage={avg_slip:.1f}bps"
                )
        elif symbol in self._expensive_symbols and avg_slip < self.max_slippage_bps:
            self._expensive_symbols.discard(symbol)

        # Estimate market impact (simplified)
        market_impact = slippage_bps * 0.6  # Rough estimate

        result = ExecutionResult(
            success=True,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact,
            execution_time_seconds=execution_time,
            algo_used=algo,
        )

        if slippage_bps > self.max_slippage_bps:
            logger.warning(
                f"High slippage on {symbol}: {slippage_bps:.1f}bps "
                f"(expected={expected_price:.2f}, fill={fill_price:.2f})"
            )

        return result

    def get_slippage_adjustment(self, symbol: str) -> float:
        """
        Get position size adjustment based on execution quality.

        Returns multiplier 0.8-1.0:
        - Avg slippage > 30bps → 0.8
        - Avg slippage > 15bps → 0.9
        - Otherwise → 1.0
        """
        avg_slip = self.get_avg_slippage(symbol)

        if avg_slip > self.critical_slippage_bps:
            return 0.8
        elif avg_slip > self.max_slippage_bps:
            return 0.9
        return 1.0

    def get_avg_slippage(self, symbol: str) -> float:
        """Get average slippage for a symbol in basis points."""
        history = self._slippage_history.get(symbol)
        if not history or len(history) == 0:
            return 0.0
        return float(np.mean(list(history)))

    def is_expensive_symbol(self, symbol: str) -> bool:
        """Check if a symbol has expensive execution costs."""
        return symbol in self._expensive_symbols

    def get_execution_quality_report(self) -> Dict:
        """
        Get comprehensive execution quality report.

        Returns:
            Dict with per-symbol and aggregate statistics
        """
        report = {
            "total_executions": self._total_executions,
            "avg_slippage_bps": (
                self._total_slippage_bps / max(self._total_executions, 1)
            ),
            "expensive_symbols": list(self._expensive_symbols),
            "per_symbol": {},
        }

        for symbol, history in self._slippage_history.items():
            if len(history) > 0:
                report["per_symbol"][symbol] = {
                    "executions": len(history),
                    "avg_slippage_bps": round(float(np.mean(list(history))), 2),
                    "max_slippage_bps": round(float(max(history)), 2),
                    "is_expensive": symbol in self._expensive_symbols,
                }

        return report

    def get_diagnostics(self) -> Dict:
        """Return shield state for monitoring."""
        return {
            "total_executions": self._total_executions,
            "symbols_tracked": len(self._slippage_history),
            "expensive_symbols": list(self._expensive_symbols),
            "avg_slippage_overall": round(
                self._total_slippage_bps / max(self._total_executions, 1), 2
            ),
        }
