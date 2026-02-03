"""
execution/adaptive_twap.py - Adaptive TWAP Execution Algorithm

Improves on basic TWAP by:
- Adjusting participation based on real-time volume
- Speeding up when price favorable
- Slowing down when price adverse
- Canceling when market conditions change

State-of-the-art execution algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class ExecutionUrgency(Enum):
    """Order urgency levels."""
    PASSIVE = "passive"      # Minimize impact, take time
    NORMAL = "normal"        # Balance cost and time
    AGGRESSIVE = "aggressive" # Prioritize completion
    VERY_AGGRESSIVE = "very_aggressive"  # Complete ASAP


@dataclass
class TWAPSlice:
    """Single time slice of TWAP order."""
    slice_num: int
    target_qty: int
    filled_qty: int
    target_price: float
    avg_fill_price: float
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'pending', 'active', 'filled', 'cancelled'
    
    @property
    def fill_rate(self) -> float:
        return self.filled_qty / self.target_qty if self.target_qty > 0 else 0


@dataclass
class TWAPOrder:
    """Complete TWAP order with all slices."""
    symbol: str
    side: str
    total_qty: int
    num_slices: int
    duration_minutes: int
    urgency: ExecutionUrgency
    slices: List[TWAPSlice] = field(default_factory=list)
    arrival_price: float = 0.0
    status: str = 'pending'  # 'pending', 'active', 'completed', 'cancelled'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def filled_qty(self) -> int:
        return sum(s.filled_qty for s in self.slices)
    
    @property
    def avg_fill_price(self) -> float:
        total_value = sum(s.filled_qty * s.avg_fill_price for s in self.slices)
        return total_value / self.filled_qty if self.filled_qty > 0 else 0
    
    @property
    def implementation_shortfall_bps(self) -> float:
        if self.arrival_price == 0 or self.avg_fill_price == 0:
            return 0
        if self.side == 'BUY':
            return (self.avg_fill_price - self.arrival_price) / self.arrival_price * 10000
        else:
            return (self.arrival_price - self.avg_fill_price) / self.arrival_price * 10000


class AdaptiveTWAP:
    """
    Adaptive Time-Weighted Average Price execution.
    
    Features:
    - Dynamic participation rate based on volume
    - Price improvement seeking
    - Adverse selection avoidance
    - Urgency-based adaptation
    """
    
    # Participation rate multipliers by urgency
    URGENCY_MULTIPLIERS = {
        ExecutionUrgency.PASSIVE: 0.5,
        ExecutionUrgency.NORMAL: 1.0,
        ExecutionUrgency.AGGRESSIVE: 1.5,
        ExecutionUrgency.VERY_AGGRESSIVE: 2.5
    }
    
    def __init__(
        self,
        default_slices: int = 10,
        max_participation_rate: float = 0.15,
        price_improvement_threshold: float = 0.001
    ):
        """
        Initialize adaptive TWAP.
        
        Args:
            default_slices: Default number of time slices
            max_participation_rate: Maximum volume participation
            price_improvement_threshold: Price improvement threshold (bps)
        """
        self.default_slices = default_slices
        self.max_participation_rate = max_participation_rate
        self.price_improvement_threshold = price_improvement_threshold
        
        self.active_orders: Dict[str, TWAPOrder] = {}
        self.completed_orders: List[TWAPOrder] = []
        
        logger.info("AdaptiveTWAP initialized")
    
    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        duration_minutes: int = 30,
        urgency: ExecutionUrgency = ExecutionUrgency.NORMAL,
        arrival_price: Optional[float] = None
    ) -> TWAPOrder:
        """
        Create a new TWAP order.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Total shares to execute
            duration_minutes: Time to execute over
            urgency: Execution urgency
            arrival_price: Price at decision time
        
        Returns:
            TWAPOrder instance
        """
        # Calculate number of slices
        num_slices = max(3, min(self.default_slices, duration_minutes // 3))
        
        # Calculate per-slice quantity
        base_qty_per_slice = quantity // num_slices
        remainder = quantity % num_slices
        
        slices = []
        for i in range(num_slices):
            slice_qty = base_qty_per_slice + (1 if i < remainder else 0)
            slices.append(TWAPSlice(
                slice_num=i + 1,
                target_qty=slice_qty,
                filled_qty=0,
                target_price=0.0,
                avg_fill_price=0.0,
                start_time=datetime.now() + timedelta(minutes=i * duration_minutes / num_slices),
                end_time=None,
                status='pending'
            ))
        
        order = TWAPOrder(
            symbol=symbol,
            side=side,
            total_qty=quantity,
            num_slices=num_slices,
            duration_minutes=duration_minutes,
            urgency=urgency,
            slices=slices,
            arrival_price=arrival_price or 0.0,
            status='pending'
        )
        
        self.active_orders[symbol] = order
        
        logger.info(
            f"Created TWAP order: {side} {quantity} {symbol} "
            f"over {duration_minutes}m in {num_slices} slices"
        )
        
        return order
    
    def calculate_adaptive_size(
        self,
        order: TWAPOrder,
        current_slice: TWAPSlice,
        current_price: float,
        current_volume: float,
        avg_volume: float
    ) -> int:
        """
        Calculate adaptive slice size.
        
        Adjusts based on:
        - Current volume vs average
        - Price movement direction
        - Urgency level
        
        Args:
            order: Parent TWAP order
            current_slice: Current slice being executed
            current_price: Current market price
            current_volume: Current period volume
            avg_volume: Average period volume
        
        Returns:
            Adjusted quantity for this slice
        """
        base_qty = current_slice.target_qty
        
        # Volume adjustment
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 1.5:
                # High volume - can execute more
                base_qty = int(base_qty * min(volume_ratio, 2.0))
            elif volume_ratio < 0.5:
                # Low volume - execute less
                base_qty = int(base_qty * max(volume_ratio, 0.3))
        
        # Price adjustment
        if order.arrival_price > 0:
            price_move = (current_price - order.arrival_price) / order.arrival_price
            
            if order.side == 'BUY':
                if price_move < -self.price_improvement_threshold:
                    # Price improved for buys - accelerate
                    base_qty = int(base_qty * 1.3)
                elif price_move > self.price_improvement_threshold:
                    # Price worsened - slow down
                    base_qty = int(base_qty * 0.7)
            else:  # SELL
                if price_move > self.price_improvement_threshold:
                    # Price improved for sells - accelerate
                    base_qty = int(base_qty * 1.3)
                elif price_move < -self.price_improvement_threshold:
                    # Price worsened - slow down
                    base_qty = int(base_qty * 0.7)
        
        # Urgency adjustment
        urgency_mult = self.URGENCY_MULTIPLIERS.get(order.urgency, 1.0)
        base_qty = int(base_qty * urgency_mult)
        
        # Ensure within limits
        remaining = order.total_qty - order.filled_qty
        base_qty = min(base_qty, remaining)
        base_qty = max(1, base_qty)  # At least 1 share
        
        return base_qty
    
    def should_pause(
        self,
        order: TWAPOrder,
        current_price: float,
        bid_ask_spread: float
    ) -> Tuple[bool, str]:
        """
        Determine if execution should pause.
        
        Pause conditions:
        - Price moved significantly against
        - Spread widened excessively
        - Market volatility spike
        
        Returns:
            Tuple of (should_pause, reason)
        """
        # Check price movement
        if order.arrival_price > 0:
            price_move = (current_price - order.arrival_price) / order.arrival_price
            threshold = 0.02  # 2%
            
            if order.side == 'BUY' and price_move > threshold:
                return True, f"Price adverse: +{price_move*100:.1f}%"
            elif order.side == 'SELL' and price_move < -threshold:
                return True, f"Price adverse: {price_move*100:.1f}%"
        
        # Check spread
        if bid_ask_spread > 0.01:  # >1% spread
            return True, f"Wide spread: {bid_ask_spread*100:.1f}%"
        
        return False, ""
    
    def should_accelerate(
        self,
        order: TWAPOrder,
        current_price: float,
        remaining_time_pct: float
    ) -> bool:
        """
        Determine if execution should accelerate.
        
        Accelerate when:
        - Falling behind schedule
        - Price moving favorably
        """
        # Behind schedule check
        fill_pct = order.filled_qty / order.total_qty
        time_pct = 1 - remaining_time_pct
        
        if time_pct > fill_pct + 0.2:  # More than 20% behind
            return True
        
        # Price improvement check
        if order.arrival_price > 0:
            price_move = (current_price - order.arrival_price) / order.arrival_price
            
            if order.side == 'BUY' and price_move < -0.005:  # 50bps better
                return True
            elif order.side == 'SELL' and price_move > 0.005:
                return True
        
        return False
    
    def update_slice(
        self,
        order: TWAPOrder,
        slice_num: int,
        filled_qty: int,
        fill_price: float
    ):
        """Update slice with fill information."""
        if slice_num <= 0 or slice_num > len(order.slices):
            return
        
        slice_idx = slice_num - 1
        current_slice = order.slices[slice_idx]
        
        current_slice.filled_qty = filled_qty
        current_slice.avg_fill_price = fill_price
        current_slice.end_time = datetime.now()
        current_slice.status = 'filled' if filled_qty >= current_slice.target_qty else 'partial'
        
        # Check if order complete
        if order.filled_qty >= order.total_qty:
            order.status = 'completed'
            order.end_time = datetime.now()
            
            # Move to completed
            if order.symbol in self.active_orders:
                del self.active_orders[order.symbol]
            self.completed_orders.append(order)
            
            shortfall = order.implementation_shortfall_bps
            logger.info(
                f"TWAP completed: {order.symbol} {order.side} {order.filled_qty} shares, "
                f"shortfall: {shortfall:.1f} bps"
            )
    
    def cancel_order(self, symbol: str) -> bool:
        """Cancel an active TWAP order."""
        if symbol in self.active_orders:
            order = self.active_orders[symbol]
            order.status = 'cancelled'
            order.end_time = datetime.now()
            
            del self.active_orders[symbol]
            self.completed_orders.append(order)
            
            logger.info(f"TWAP cancelled: {symbol}, filled {order.filled_qty}/{order.total_qty}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        if not self.completed_orders:
            return {
                'total_orders': 0,
                'avg_shortfall_bps': 0,
                'completion_rate': 0
            }
        
        shortfalls = [o.implementation_shortfall_bps for o in self.completed_orders]
        completions = [o.filled_qty / o.total_qty for o in self.completed_orders]
        
        return {
            'total_orders': len(self.completed_orders),
            'avg_shortfall_bps': float(np.mean(shortfalls)),
            'median_shortfall_bps': float(np.median(shortfalls)),
            'completion_rate': float(np.mean(completions)),
            'active_orders': len(self.active_orders)
        }
