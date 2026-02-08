"""
risk/drawdown_cascade_breaker.py - Multi-Tier Drawdown Response

Implements 5 escalating tiers of defense based on portfolio drawdown:

Tier 0: NORMAL   (0-2%)  → 100% sizing, full trading
Tier 1: CAUTION  (2-4%)  → 75% sizing, tighter thresholds
Tier 2: DEFENSIVE(4-6%)  → 50% sizing, high-conviction only, close worst 2
Tier 3: SURVIVAL (6-8%)  → 25% sizing, entries blocked, max 5 positions
Tier 4: EMERGENCY(>8%)   → 0% sizing, close ALL, 24h cooldown

Also monitors drawdown velocity:
- >1% per day → jump up one tier
- >2% per day → jump up two tiers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DrawdownTier(IntEnum):
    NORMAL = 0
    CAUTION = 1
    DEFENSIVE = 2
    SURVIVAL = 3
    EMERGENCY = 4


@dataclass
class DrawdownState:
    """Current drawdown assessment."""
    tier: DrawdownTier
    current_drawdown: float
    drawdown_velocity: float  # Daily rate of change
    peak_capital: float
    current_capital: float
    days_in_tier: int
    max_new_positions: int
    entry_allowed: bool
    min_confidence: float
    size_multiplier: float
    force_close_count: int  # Number of positions to force close
    timestamp: datetime = field(default_factory=datetime.now)


class DrawdownCascadeBreaker:
    """
    Multi-tier drawdown response with automatic deleveraging.

    Monitors portfolio drawdown and velocity, escalating through
    defensive tiers to protect capital during adverse conditions.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        tier_1_threshold: float = 0.02,
        tier_2_threshold: float = 0.04,
        tier_3_threshold: float = 0.06,
        tier_4_threshold: float = 0.08,
        velocity_jump_threshold: float = 0.01,
        recovery_buffer: float = 0.015,
        cooldown_hours_tier_3: int = 12,
        cooldown_hours_tier_4: int = 24,
    ):
        self.initial_capital = initial_capital
        self.tier_thresholds = {
            DrawdownTier.CAUTION: tier_1_threshold,
            DrawdownTier.DEFENSIVE: tier_2_threshold,
            DrawdownTier.SURVIVAL: tier_3_threshold,
            DrawdownTier.EMERGENCY: tier_4_threshold,
        }
        self.velocity_jump = velocity_jump_threshold
        self.recovery_buffer = recovery_buffer
        self.cooldown_hours = {
            DrawdownTier.SURVIVAL: cooldown_hours_tier_3,
            DrawdownTier.EMERGENCY: cooldown_hours_tier_4,
        }

        # State
        self._peak_capital = initial_capital
        self._current_tier = DrawdownTier.NORMAL
        self._tier_entry_time: Optional[datetime] = None
        self._last_update: Optional[datetime] = None

        # Velocity tracking
        self._capital_history: deque = deque(maxlen=30)  # 30-day history

        logger.info(
            f"DrawdownCascadeBreaker initialized: "
            f"tiers={list(self.tier_thresholds.values())}"
        )

    def update(
        self,
        current_capital: float,
        peak_capital: Optional[float] = None,
    ) -> DrawdownState:
        """
        Update drawdown state with current capital.

        Args:
            current_capital: Current portfolio value
            peak_capital: High-water mark (if None, tracked internally)

        Returns:
            DrawdownState with tier and recommended actions
        """
        now = datetime.now()

        # Update peak
        if peak_capital is not None:
            self._peak_capital = max(self._peak_capital, peak_capital)
        self._peak_capital = max(self._peak_capital, current_capital)

        # Track capital history for velocity
        self._capital_history.append((now, current_capital))

        # Compute drawdown
        if self._peak_capital > 0:
            drawdown = (self._peak_capital - current_capital) / self._peak_capital
        else:
            drawdown = 0.0

        # Compute velocity (daily rate of change)
        velocity = self._compute_velocity()

        # Determine tier based on drawdown and velocity
        new_tier = self._determine_tier(drawdown, velocity)

        # Handle tier transitions
        self._handle_tier_transition(new_tier, now)

        # Compute days in tier
        days_in_tier = 0
        if self._tier_entry_time:
            days_in_tier = (now - self._tier_entry_time).days

        self._last_update = now

        state = DrawdownState(
            tier=self._current_tier,
            current_drawdown=drawdown,
            drawdown_velocity=velocity,
            peak_capital=self._peak_capital,
            current_capital=current_capital,
            days_in_tier=days_in_tier,
            max_new_positions=self._get_max_positions(),
            entry_allowed=self.get_entry_allowed(),
            min_confidence=self.get_min_confidence(),
            size_multiplier=self.get_position_size_multiplier(),
            force_close_count=self._get_force_close_count(),
        )

        if new_tier != self._current_tier:
            logger.warning(
                f"Drawdown tier changed: {self._current_tier.name} -> {new_tier.name} "
                f"(DD={drawdown:.1%}, velocity={velocity:.2%}/day)"
            )

        return state

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier for current tier."""
        return {
            DrawdownTier.NORMAL: 1.0,
            DrawdownTier.CAUTION: 0.75,
            DrawdownTier.DEFENSIVE: 0.50,
            DrawdownTier.SURVIVAL: 0.25,
            DrawdownTier.EMERGENCY: 0.0,
        }[self._current_tier]

    def get_entry_allowed(self) -> bool:
        """Check if new entries are allowed."""
        if self._current_tier >= DrawdownTier.SURVIVAL:
            return False
        if self._current_tier >= DrawdownTier.EMERGENCY:
            return False
        return True

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold for current tier."""
        return {
            DrawdownTier.NORMAL: 0.25,
            DrawdownTier.CAUTION: 0.35,
            DrawdownTier.DEFENSIVE: 0.70,
            DrawdownTier.SURVIVAL: 1.0,  # Effectively blocks all
            DrawdownTier.EMERGENCY: 1.0,
        }[self._current_tier]

    def get_positions_to_close(
        self,
        positions: Dict[str, Dict],
        prices: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Get positions that should be force-closed.

        Args:
            positions: Dict of symbol -> position info
            prices: Current prices for P&L calculation

        Returns:
            List of symbols to close, worst P&L first
        """
        n_close = self._get_force_close_count()
        if n_close == 0 or not positions:
            return []

        # Score by P&L (worst first)
        scored = []
        for symbol, pos in positions.items():
            pnl = pos.get("pnl", pos.get("unrealized_pnl", 0.0))
            if prices and symbol in prices and "entry_price" in pos:
                entry = pos["entry_price"]
                current = prices[symbol]
                qty = pos.get("quantity", pos.get("shares", 1))
                pnl = (current - entry) * qty
            scored.append((symbol, pnl))

        scored.sort(key=lambda x: x[1])

        if self._current_tier == DrawdownTier.EMERGENCY:
            return [s for s, _ in scored]  # Close all

        return [s for s, _ in scored[:n_close]]

    def should_force_deleverage(self) -> bool:
        """Check if forced deleveraging should occur."""
        return self._current_tier >= DrawdownTier.DEFENSIVE

    def reset_peak(self, new_peak: Optional[float] = None):
        """
        Reset the peak capital (e.g., after deposit/withdrawal).

        Args:
            new_peak: New peak value, or None to use current capital
        """
        if new_peak is not None:
            self._peak_capital = new_peak
        self._current_tier = DrawdownTier.NORMAL
        self._tier_entry_time = None
        logger.info(f"Peak capital reset to {self._peak_capital:,.0f}")

    # ─── Internal Methods ─────────────────────────────────────────

    def _compute_velocity(self) -> float:
        """Compute daily rate of drawdown change."""
        if len(self._capital_history) < 2:
            return 0.0

        history = list(self._capital_history)
        now_ts, now_cap = history[-1]

        # Find capital from ~1 day ago
        one_day_ago = now_ts - timedelta(days=1)
        prev_cap = now_cap
        for ts, cap in reversed(history[:-1]):
            if ts <= one_day_ago:
                prev_cap = cap
                break

        if prev_cap <= 0:
            return 0.0

        daily_change = (prev_cap - now_cap) / prev_cap
        return daily_change

    def _determine_tier(self, drawdown: float, velocity: float) -> DrawdownTier:
        """Determine tier based on drawdown level and velocity."""
        # Base tier from drawdown level
        if drawdown >= self.tier_thresholds[DrawdownTier.EMERGENCY]:
            base_tier = DrawdownTier.EMERGENCY
        elif drawdown >= self.tier_thresholds[DrawdownTier.SURVIVAL]:
            base_tier = DrawdownTier.SURVIVAL
        elif drawdown >= self.tier_thresholds[DrawdownTier.DEFENSIVE]:
            base_tier = DrawdownTier.DEFENSIVE
        elif drawdown >= self.tier_thresholds[DrawdownTier.CAUTION]:
            base_tier = DrawdownTier.CAUTION
        else:
            base_tier = DrawdownTier.NORMAL

        # Velocity can escalate tier
        if velocity >= 2 * self.velocity_jump:
            # Jump up 2 tiers
            base_tier = DrawdownTier(min(base_tier + 2, DrawdownTier.EMERGENCY))
        elif velocity >= self.velocity_jump:
            # Jump up 1 tier
            base_tier = DrawdownTier(min(base_tier + 1, DrawdownTier.EMERGENCY))

        return base_tier

    def _handle_tier_transition(self, new_tier: DrawdownTier, now: datetime):
        """Handle transitions between tiers with cooldowns."""
        if new_tier == self._current_tier:
            return

        if new_tier > self._current_tier:
            # Escalating — always allowed
            self._current_tier = new_tier
            self._tier_entry_time = now

        else:
            # De-escalating — check cooldown
            cooldown_hours = self.cooldown_hours.get(self._current_tier, 0)
            if self._tier_entry_time and cooldown_hours > 0:
                time_in_tier = (now - self._tier_entry_time).total_seconds() / 3600
                if time_in_tier < cooldown_hours:
                    # Still in cooldown, don't de-escalate
                    return

            # Check recovery buffer: only de-escalate if DD is sufficiently below threshold
            # (prevents oscillation at boundaries)
            self._current_tier = new_tier
            self._tier_entry_time = now

    def _get_max_positions(self) -> int:
        """Get maximum allowed positions for current tier."""
        return {
            DrawdownTier.NORMAL: 20,
            DrawdownTier.CAUTION: 15,
            DrawdownTier.DEFENSIVE: 10,
            DrawdownTier.SURVIVAL: 5,
            DrawdownTier.EMERGENCY: 0,
        }[self._current_tier]

    def _get_force_close_count(self) -> int:
        """Get number of positions to force close."""
        return {
            DrawdownTier.NORMAL: 0,
            DrawdownTier.CAUTION: 0,
            DrawdownTier.DEFENSIVE: 2,
            DrawdownTier.SURVIVAL: 99,  # Close down to max 5
            DrawdownTier.EMERGENCY: 999,  # Close all
        }[self._current_tier]

    def get_diagnostics(self) -> Dict:
        """Return breaker state for monitoring."""
        return {
            "tier": self._current_tier.name,
            "peak_capital": self._peak_capital,
            "size_multiplier": self.get_position_size_multiplier(),
            "entry_allowed": self.get_entry_allowed(),
            "min_confidence": self.get_min_confidence(),
            "max_positions": self._get_max_positions(),
            "force_close_count": self._get_force_close_count(),
            "tier_entry_time": (
                self._tier_entry_time.isoformat() if self._tier_entry_time else None
            ),
        }
