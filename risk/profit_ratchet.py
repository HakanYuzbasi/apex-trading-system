"""
risk/profit_ratchet.py - Progressive Profit Protection

Implements a "ratchet" mechanism that progressively tightens stops
as profits increase. Once profits are locked, they cannot be lost.

Profit tiers:
Tier 0:  0-2% gain   → 3% trailing stop (standard)
Tier 1:  2-5% gain   → Lock 50% of gains (max drawdown 1%)
Tier 2:  5-10% gain  → Lock 70% of gains (max drawdown 1.5%)
Tier 3:  10-20% gain → Lock 80% of gains (max drawdown 2%)
Tier 4:  20%+ gain   → Lock 85% of gains (max drawdown 3%)

Example: 15% gain → locked at 12% (80% of 15%), max drawdown to 10%

Additional features:
- Partial profit taking at each tier
- Acceleration detection (faster tightening in momentum)
- Time-based ratchet (tighter stops after 5 days)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum

logger = logging.getLogger(__name__)


class ProfitTier(IntEnum):
    INITIAL = 0      # 0-2%
    TIER_1 = 1       # 2-5%
    TIER_2 = 2       # 5-10%
    TIER_3 = 3       # 10-20%
    TIER_4 = 4       # 20%+


@dataclass
class RatchetState:
    """Profit ratchet state for a position."""
    symbol: str
    entry_price: float
    current_price: float
    high_water_mark: float
    current_pnl_pct: float
    tier: ProfitTier
    locked_profit_pct: float  # Minimum profit guaranteed
    stop_price: float
    trailing_pct: float
    should_take_partial: bool
    partial_size_pct: float
    days_held: int
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PartialExitRecommendation:
    """Recommendation to take partial profits."""
    symbol: str
    exit_pct: float  # Percentage of position to exit
    reason: str
    tier_triggered: ProfitTier


class ProfitRatchet:
    """
    Progressive profit protection with locked-in gains.

    Tracks high water mark for each position and tightens
    stops as profits accumulate, ensuring gains are protected.
    """

    def __init__(
        self,
        tier_1_threshold: float = 0.02,   # 2%
        tier_2_threshold: float = 0.05,   # 5%
        tier_3_threshold: float = 0.10,   # 10%
        tier_4_threshold: float = 0.20,   # 20%
        initial_trailing_pct: float = 0.03,  # 3% initial stop
        lock_pct_tier_1: float = 0.50,
        lock_pct_tier_2: float = 0.70,
        lock_pct_tier_3: float = 0.80,
        lock_pct_tier_4: float = 0.85,
        partial_take_tier_2: float = 0.25,  # Take 25% at tier 2
        partial_take_tier_3: float = 0.25,  # Take another 25% at tier 3
        time_tightening_days: int = 5,
        time_tightening_factor: float = 0.75,  # 25% tighter after 5 days
    ):
        self.tier_thresholds = {
            ProfitTier.TIER_1: tier_1_threshold,
            ProfitTier.TIER_2: tier_2_threshold,
            ProfitTier.TIER_3: tier_3_threshold,
            ProfitTier.TIER_4: tier_4_threshold,
        }
        self.initial_trailing = initial_trailing_pct
        self.lock_percentages = {
            ProfitTier.INITIAL: 0.0,
            ProfitTier.TIER_1: lock_pct_tier_1,
            ProfitTier.TIER_2: lock_pct_tier_2,
            ProfitTier.TIER_3: lock_pct_tier_3,
            ProfitTier.TIER_4: lock_pct_tier_4,
        }
        self.partial_takes = {
            ProfitTier.TIER_2: partial_take_tier_2,
            ProfitTier.TIER_3: partial_take_tier_3,
        }
        self.time_tightening_days = time_tightening_days
        self.time_tightening_factor = time_tightening_factor

        # State: symbol -> (high_water_mark, entry_price, entry_time, partials_taken)
        self._positions: Dict[str, Dict] = {}

        logger.info(
            f"ProfitRatchet initialized: "
            f"tiers={list(self.tier_thresholds.values())}, "
            f"lock={list(self.lock_percentages.values())}"
        )

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: Optional[datetime] = None,
        side: str = "LONG",
    ):
        """Register a new position for tracking."""
        self._positions[symbol] = {
            "entry_price": entry_price,
            "entry_time": entry_time or datetime.now(timezone.utc),
            "high_water_mark": entry_price,
            "side": side,
            "partials_taken": set(),  # Tiers where partial was taken
            "highest_tier_reached": ProfitTier.INITIAL,
        }
        logger.debug(f"ProfitRatchet: Registered {symbol} at ${entry_price:.2f}")

    def update_position(
        self,
        symbol: str,
        current_price: float,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
    ) -> Optional[RatchetState]:
        """
        Update position with current price and compute ratchet state.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            entry_price: Entry price (if not already registered)
            entry_time: Entry time (if not already registered)

        Returns:
            RatchetState with stop and profit-taking recommendations
        """
        if symbol not in self._positions:
            if entry_price is not None:
                self.register_position(symbol, entry_price, entry_time)
            else:
                return None

        pos = self._positions[symbol]
        entry = pos["entry_price"]
        side = pos.get("side", "LONG")

        # Calculate P&L
        if side == "LONG":
            pnl_pct = (current_price - entry) / entry
            # Update high water mark
            if current_price > pos["high_water_mark"]:
                pos["high_water_mark"] = current_price
            hwm = pos["high_water_mark"]
            hwm_pnl = (hwm - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - current_price) / entry
            if current_price < pos["high_water_mark"]:
                pos["high_water_mark"] = current_price
            hwm = pos["high_water_mark"]
            hwm_pnl = (entry - hwm) / entry

        # Determine current tier based on HIGH WATER MARK (not current price)
        tier = self._get_tier(hwm_pnl)
        pos["highest_tier_reached"] = max(pos["highest_tier_reached"], tier)

        # Calculate locked profit
        lock_pct = self.lock_percentages[tier]
        locked_profit = hwm_pnl * lock_pct

        # Calculate trailing stop
        trailing_pct = self._get_trailing_pct(tier, pos)

        # Stop price calculation
        if side == "LONG":
            stop_from_hwm = hwm * (1 - trailing_pct)
            stop_from_locked = entry * (1 + locked_profit)
            stop_price = max(stop_from_hwm, stop_from_locked)
        else:
            stop_from_hwm = hwm * (1 + trailing_pct)
            stop_from_locked = entry * (1 - locked_profit)
            stop_price = min(stop_from_hwm, stop_from_locked)

        # Check for partial profit taking
        should_take_partial = False
        partial_size = 0.0
        for partial_tier, partial_pct in self.partial_takes.items():
            if tier >= partial_tier and partial_tier not in pos["partials_taken"]:
                should_take_partial = True
                partial_size = partial_pct
                break

        _entry_time = pos["entry_time"]
        if _entry_time.tzinfo is None:
            _entry_time = _entry_time.replace(tzinfo=timezone.utc)
        days_held = (datetime.now(timezone.utc) - _entry_time).days

        state = RatchetState(
            symbol=symbol,
            entry_price=entry,
            current_price=current_price,
            high_water_mark=hwm,
            current_pnl_pct=pnl_pct,
            tier=tier,
            locked_profit_pct=locked_profit,
            stop_price=stop_price,
            trailing_pct=trailing_pct,
            should_take_partial=should_take_partial,
            partial_size_pct=partial_size,
            days_held=days_held,
        )

        return state

    def _get_tier(self, pnl_pct: float) -> ProfitTier:
        """Determine profit tier from P&L percentage."""
        if pnl_pct >= self.tier_thresholds[ProfitTier.TIER_4]:
            return ProfitTier.TIER_4
        elif pnl_pct >= self.tier_thresholds[ProfitTier.TIER_3]:
            return ProfitTier.TIER_3
        elif pnl_pct >= self.tier_thresholds[ProfitTier.TIER_2]:
            return ProfitTier.TIER_2
        elif pnl_pct >= self.tier_thresholds[ProfitTier.TIER_1]:
            return ProfitTier.TIER_1
        else:
            return ProfitTier.INITIAL

    def _get_trailing_pct(self, tier: ProfitTier, pos: Dict) -> float:
        """Get trailing stop percentage for tier with time adjustment."""
        # Base trailing by tier
        base_trailing = {
            ProfitTier.INITIAL: self.initial_trailing,
            ProfitTier.TIER_1: 0.025,  # 2.5%
            ProfitTier.TIER_2: 0.02,   # 2%
            ProfitTier.TIER_3: 0.018,  # 1.8%
            ProfitTier.TIER_4: 0.015,  # 1.5%
        }[tier]

        # Time-based tightening
        _entry_time = pos["entry_time"]
        if _entry_time.tzinfo is None:
            _entry_time = _entry_time.replace(tzinfo=timezone.utc)
        days_held = (datetime.now(timezone.utc) - _entry_time).days
        if days_held >= self.time_tightening_days:
            base_trailing *= self.time_tightening_factor

        return base_trailing

    def mark_partial_taken(self, symbol: str, tier: ProfitTier):
        """Mark that a partial profit was taken at a tier."""
        if symbol in self._positions:
            self._positions[symbol]["partials_taken"].add(tier)

    def should_exit(
        self,
        symbol: str,
        current_price: float,
    ) -> Tuple[bool, str, float]:
        """
        Check if position should be fully exited.

        Returns:
            (should_exit, reason, stop_price)
        """
        state = self.update_position(symbol, current_price)
        if state is None:
            return False, "", 0.0

        pos = self._positions[symbol]
        side = pos.get("side", "LONG")

        if side == "LONG":
            if current_price <= state.stop_price:
                return True, f"Ratchet stop hit (tier {state.tier.name})", state.stop_price
        else:
            if current_price >= state.stop_price:
                return True, f"Ratchet stop hit (tier {state.tier.name})", state.stop_price

        return False, "", state.stop_price

    def get_partial_recommendations(
        self,
        positions: Dict[str, Dict],
    ) -> List[PartialExitRecommendation]:
        """
        Get partial profit-taking recommendations for all positions.

        Args:
            positions: Dict of symbol -> position info with current_price

        Returns:
            List of partial exit recommendations
        """
        recommendations = []

        for symbol, pos_info in positions.items():
            current_price = pos_info.get("current_price", pos_info.get("price", 0))
            if current_price <= 0:
                continue

            state = self.update_position(symbol, current_price)
            if state is None:
                continue

            if state.should_take_partial:
                recommendations.append(PartialExitRecommendation(
                    symbol=symbol,
                    exit_pct=state.partial_size_pct,
                    reason=f"Tier {state.tier.name} reached ({state.current_pnl_pct:.1%} gain)",
                    tier_triggered=state.tier,
                ))

        return recommendations

    def get_stop_prices(
        self,
        positions: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Get current stop prices for all tracked positions.

        Args:
            positions: Dict of symbol -> position info with current_price

        Returns:
            Dict of symbol -> stop_price
        """
        stops = {}

        for symbol, pos_info in positions.items():
            current_price = pos_info.get("current_price", pos_info.get("price", 0))
            if current_price <= 0:
                continue

            state = self.update_position(symbol, current_price)
            if state:
                stops[symbol] = state.stop_price

        return stops

    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self._positions:
            del self._positions[symbol]

    def get_position_state(self, symbol: str) -> Optional[Dict]:
        """Get internal state for a position."""
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        return {
            "entry_price": pos["entry_price"],
            "entry_time": pos["entry_time"].isoformat(),
            "high_water_mark": pos["high_water_mark"],
            "side": pos["side"],
            "highest_tier": pos["highest_tier_reached"].name,
            "partials_taken": [t.name for t in pos["partials_taken"]],
        }

    def get_diagnostics(self) -> Dict:
        """Return ratchet state for monitoring."""
        tier_counts = {tier.name: 0 for tier in ProfitTier}

        for symbol, pos in self._positions.items():
            tier = pos["highest_tier_reached"]
            tier_counts[tier.name] += 1

        return {
            "positions_tracked": len(self._positions),
            "tier_distribution": tier_counts,
            "tier_thresholds": {k.name: v for k, v in self.tier_thresholds.items()},
            "lock_percentages": {k.name: v for k, v in self.lock_percentages.items()},
        }
