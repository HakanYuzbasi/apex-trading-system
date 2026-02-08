"""
risk/position_aging_manager.py - Time-Based Position Management

Manages positions based on holding time to prevent capital from
being tied up in stale, non-performing positions.

Key concepts:
- Mean reversion trades should resolve within 5-10 days
- Momentum trades need room to run but not forever
- Positions that don't move are opportunity cost

Aging tiers:
Day 1-5:   Fresh position, normal rules
Day 5-10:  Maturing, tighten stops 20%
Day 10-15: Stale, require positive P&L or exit
Day 15-20: Critical, require 2%+ P&L or exit
Day 20+:   Force exit (capital recycling)

Also tracks:
- Time-weighted P&L (did position make money early or late?)
- Velocity of gains (accelerating or decelerating?)
- Days since last new high
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
import numpy as np

logger = logging.getLogger(__name__)


class AgingTier(IntEnum):
    FRESH = 0       # Day 1-5
    MATURING = 1    # Day 5-10
    STALE = 2       # Day 10-15
    CRITICAL = 3    # Day 15-20
    EXPIRED = 4     # Day 20+


@dataclass
class PositionAge:
    """Age and performance metrics for a position."""
    symbol: str
    entry_time: datetime
    days_held: int
    tier: AgingTier
    current_pnl_pct: float
    high_water_pnl_pct: float
    days_since_new_high: int
    velocity: float  # Rate of P&L change
    stop_tightening: float  # Multiplier for stops (e.g., 0.8 = 20% tighter)
    should_exit: bool
    exit_reason: Optional[str]
    min_pnl_required: float  # Min P&L to keep position


@dataclass
class AgingRecommendation:
    """Recommendation for position based on age."""
    symbol: str
    action: str  # "hold", "tighten_stop", "reduce", "exit"
    urgency: str  # "low", "medium", "high"
    reason: str
    days_remaining: Optional[int]  # Days until forced exit


class PositionAgingManager:
    """
    Time-based position management.

    Tracks how long positions have been held and enforces
    time limits to prevent capital from being tied up in
    non-performing positions.
    """

    def __init__(
        self,
        fresh_days: int = 5,
        maturing_days: int = 10,
        stale_days: int = 15,
        critical_days: int = 20,
        max_days: int = 30,
        stale_min_pnl: float = 0.0,      # Must be positive
        critical_min_pnl: float = 0.02,  # Must be 2%+
        max_days_since_high: int = 7,    # Exit if no new high in 7 days
        stop_tightening_per_tier: float = 0.20,  # 20% tighter per tier
    ):
        self.fresh_days = fresh_days
        self.maturing_days = maturing_days
        self.stale_days = stale_days
        self.critical_days = critical_days
        self.max_days = max_days
        self.stale_min_pnl = stale_min_pnl
        self.critical_min_pnl = critical_min_pnl
        self.max_days_since_high = max_days_since_high
        self.stop_tightening = stop_tightening_per_tier

        # Position tracking
        self._positions: Dict[str, Dict] = {}

        logger.info(
            f"PositionAgingManager initialized: "
            f"max_days={max_days}, critical_min_pnl={critical_min_pnl:.1%}"
        )

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: Optional[datetime] = None,
    ):
        """Register a new position for aging tracking."""
        now = entry_time or datetime.now()
        self._positions[symbol] = {
            "entry_price": entry_price,
            "entry_time": now,
            "high_water_mark_pnl": 0.0,
            "high_water_time": now,
            "pnl_history": [],  # (timestamp, pnl_pct) for velocity
        }
        logger.debug(f"PositionAgingManager: Registered {symbol}")

    def update_position(
        self,
        symbol: str,
        current_price: float,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
    ) -> Optional[PositionAge]:
        """
        Update position with current price and compute age metrics.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            entry_price: Entry price (if not registered)
            entry_time: Entry time (if not registered)

        Returns:
            PositionAge with age metrics and recommendations
        """
        if symbol not in self._positions:
            if entry_price is not None:
                self.register_position(symbol, entry_price, entry_time)
            else:
                return None

        pos = self._positions[symbol]
        now = datetime.now()

        # Calculate P&L
        entry = pos["entry_price"]
        pnl_pct = (current_price - entry) / entry if entry > 0 else 0

        # Update P&L history (for velocity)
        pos["pnl_history"].append((now, pnl_pct))
        # Keep last 20 observations
        if len(pos["pnl_history"]) > 20:
            pos["pnl_history"] = pos["pnl_history"][-20:]

        # Update high water mark
        if pnl_pct > pos["high_water_mark_pnl"]:
            pos["high_water_mark_pnl"] = pnl_pct
            pos["high_water_time"] = now

        # Calculate days held
        days_held = (now - pos["entry_time"]).days

        # Calculate days since new high
        days_since_high = (now - pos["high_water_time"]).days

        # Calculate velocity (P&L change rate)
        velocity = self._calculate_velocity(pos["pnl_history"])

        # Determine tier
        tier = self._get_tier(days_held)

        # Stop tightening factor
        stop_tight = 1.0 - (tier.value * self.stop_tightening)
        stop_tight = max(stop_tight, 0.4)  # Never tighter than 60%

        # Minimum P&L required
        if tier == AgingTier.STALE:
            min_pnl = self.stale_min_pnl
        elif tier == AgingTier.CRITICAL:
            min_pnl = self.critical_min_pnl
        elif tier == AgingTier.EXPIRED:
            min_pnl = self.critical_min_pnl * 1.5  # 3% for expired
        else:
            min_pnl = -1.0  # No minimum for fresh positions

        # Should exit?
        should_exit = False
        exit_reason = None

        if tier == AgingTier.EXPIRED:
            should_exit = True
            exit_reason = f"Max holding period ({self.max_days} days) reached"
        elif tier >= AgingTier.STALE and pnl_pct < min_pnl:
            should_exit = True
            exit_reason = f"P&L {pnl_pct:.1%} below {tier.name} minimum {min_pnl:.1%}"
        elif days_since_high >= self.max_days_since_high and pnl_pct < 0.05:
            should_exit = True
            exit_reason = f"No new high in {days_since_high} days"
        elif velocity < -0.005 and tier >= AgingTier.MATURING:  # Losing 0.5%/day
            should_exit = True
            exit_reason = f"Negative momentum: {velocity:.2%}/day"

        return PositionAge(
            symbol=symbol,
            entry_time=pos["entry_time"],
            days_held=days_held,
            tier=tier,
            current_pnl_pct=pnl_pct,
            high_water_pnl_pct=pos["high_water_mark_pnl"],
            days_since_new_high=days_since_high,
            velocity=velocity,
            stop_tightening=stop_tight,
            should_exit=should_exit,
            exit_reason=exit_reason,
            min_pnl_required=min_pnl,
        )

    def _get_tier(self, days_held: int) -> AgingTier:
        """Determine aging tier from days held."""
        if days_held >= self.max_days:
            return AgingTier.EXPIRED
        elif days_held >= self.critical_days:
            return AgingTier.CRITICAL
        elif days_held >= self.stale_days:
            return AgingTier.STALE
        elif days_held >= self.maturing_days:
            return AgingTier.MATURING
        else:
            return AgingTier.FRESH

    def _calculate_velocity(self, pnl_history: List[Tuple[datetime, float]]) -> float:
        """Calculate rate of P&L change (% per day)."""
        if len(pnl_history) < 2:
            return 0.0

        # Use last 5 observations
        recent = pnl_history[-5:] if len(pnl_history) >= 5 else pnl_history

        if len(recent) < 2:
            return 0.0

        first_time, first_pnl = recent[0]
        last_time, last_pnl = recent[-1]

        days_diff = (last_time - first_time).total_seconds() / 86400
        if days_diff < 0.01:  # Less than ~15 min
            return 0.0

        pnl_change = last_pnl - first_pnl
        velocity = pnl_change / days_diff

        return velocity

    def get_recommendations(
        self,
        positions: Dict[str, Dict],
    ) -> List[AgingRecommendation]:
        """
        Get aging-based recommendations for all positions.

        Args:
            positions: Dict of symbol -> position info with current_price

        Returns:
            List of recommendations sorted by urgency
        """
        recommendations = []

        for symbol, pos_info in positions.items():
            current_price = pos_info.get("current_price", pos_info.get("price", 0))
            entry_price = pos_info.get("entry_price", pos_info.get("avg_cost", 0))

            if current_price <= 0 or entry_price <= 0:
                continue

            age = self.update_position(symbol, current_price, entry_price)
            if age is None:
                continue

            # Determine recommendation
            if age.should_exit:
                recommendations.append(AgingRecommendation(
                    symbol=symbol,
                    action="exit",
                    urgency="high",
                    reason=age.exit_reason or "Aging limit reached",
                    days_remaining=0,
                ))
            elif age.tier == AgingTier.CRITICAL:
                days_left = self.max_days - age.days_held
                recommendations.append(AgingRecommendation(
                    symbol=symbol,
                    action="reduce" if age.current_pnl_pct > 0 else "exit",
                    urgency="high",
                    reason=f"Critical age: {age.days_held} days, need {age.min_pnl_required:.1%}+ P&L",
                    days_remaining=days_left,
                ))
            elif age.tier == AgingTier.STALE:
                days_left = self.critical_days - age.days_held
                recommendations.append(AgingRecommendation(
                    symbol=symbol,
                    action="tighten_stop",
                    urgency="medium",
                    reason=f"Stale position: {age.days_held} days",
                    days_remaining=days_left,
                ))
            elif age.tier == AgingTier.MATURING and age.velocity < 0:
                recommendations.append(AgingRecommendation(
                    symbol=symbol,
                    action="tighten_stop",
                    urgency="low",
                    reason=f"Maturing with negative velocity: {age.velocity:.2%}/day",
                    days_remaining=self.stale_days - age.days_held,
                ))

        # Sort by urgency (high first)
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: urgency_order.get(r.urgency, 3))

        return recommendations

    def get_positions_to_exit(
        self,
        positions: Dict[str, Dict],
    ) -> List[Tuple[str, str]]:
        """
        Get list of positions that should be exited due to aging.

        Returns:
            List of (symbol, reason) tuples
        """
        exits = []

        for symbol, pos_info in positions.items():
            current_price = pos_info.get("current_price", pos_info.get("price", 0))
            if current_price <= 0:
                continue

            age = self.update_position(symbol, current_price)
            if age and age.should_exit:
                exits.append((symbol, age.exit_reason or "Age limit"))

        return exits

    def get_stop_multiplier(self, symbol: str) -> float:
        """Get stop tightening multiplier for a position."""
        if symbol not in self._positions:
            return 1.0

        pos = self._positions[symbol]
        days_held = (datetime.now() - pos["entry_time"]).days
        tier = self._get_tier(days_held)

        multiplier = 1.0 - (tier.value * self.stop_tightening)
        return max(multiplier, 0.4)

    def get_days_held(self, symbol: str) -> int:
        """Get days held for a position."""
        if symbol not in self._positions:
            return 0
        return (datetime.now() - self._positions[symbol]["entry_time"]).days

    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self._positions:
            del self._positions[symbol]

    def get_diagnostics(self) -> Dict:
        """Return manager state for monitoring."""
        tier_counts = {tier.name: 0 for tier in AgingTier}
        total_days = 0
        exit_candidates = 0

        for symbol, pos in self._positions.items():
            days = (datetime.now() - pos["entry_time"]).days
            total_days += days
            tier = self._get_tier(days)
            tier_counts[tier.name] += 1
            if tier >= AgingTier.STALE:
                exit_candidates += 1

        avg_days = total_days / len(self._positions) if self._positions else 0

        return {
            "positions_tracked": len(self._positions),
            "tier_distribution": tier_counts,
            "average_days_held": round(avg_days, 1),
            "exit_candidates": exit_candidates,
            "tier_thresholds": {
                "fresh": self.fresh_days,
                "maturing": self.maturing_days,
                "stale": self.stale_days,
                "critical": self.critical_days,
                "max": self.max_days,
            },
        }
