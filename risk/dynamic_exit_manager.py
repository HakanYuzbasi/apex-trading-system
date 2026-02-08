"""
risk/dynamic_exit_manager.py - Dynamic Exit Strategy Manager

Adapts exit thresholds based on:
- Market regime (bull/bear/volatile/neutral)
- VIX level
- Signal strength at entry
- Position P&L trajectory
- Holding time
- Momentum decay

Exit philosophy:
- Cut losers fast, let winners run
- Tighter stops in volatile markets
- Wider targets in trending markets
- Time decay on stale positions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from config import ApexConfig

logger = logging.getLogger(__name__)


class ExitUrgency(Enum):
    """Exit urgency levels."""
    IMMEDIATE = "immediate"      # Exit now at market
    HIGH = "high"                # Exit within minutes
    MODERATE = "moderate"        # Exit within hours
    LOW = "low"                  # Monitor, no rush
    HOLD = "hold"                # Keep position


@dataclass
class DynamicExitLevels:
    """Dynamic exit levels for a position."""
    stop_loss_pct: float         # Stop loss as % from entry
    take_profit_pct: float       # Take profit as % from entry
    trailing_activation_pct: float  # When to activate trailing stop
    trailing_distance_pct: float    # Trailing stop distance
    max_hold_days: int           # Maximum holding period
    signal_exit_threshold: float # Signal level to trigger exit
    urgency: ExitUrgency         # Current exit urgency
    reason: str                  # Explanation


class DynamicExitManager:
    """
    Manages dynamic exit levels based on market conditions.

    Key principles:
    1. Volatile markets = tighter stops, faster exits
    2. Trending markets = wider targets, let profits run
    3. Losing positions = faster exits, lower thresholds
    4. Winning positions = trailing stops, protect gains
    5. Stale positions = time decay, lower thresholds
    """

    # Base parameters (will be adjusted dynamically)
    BASE_STOP_LOSS_PCT = 0.03          # 3% stop - cut losers faster
    BASE_TAKE_PROFIT_PCT = 0.06        # 6% target - lock in profits (2:1 R/R)
    BASE_TRAILING_ACTIVATION = 0.025  # Activate trailing at 2.5% gain
    BASE_TRAILING_DISTANCE = 0.02     # 2% trailing distance - tighter protection
    BASE_MAX_HOLD_DAYS = 14           # 2 weeks max hold - force stale cleanup
    BASE_SIGNAL_EXIT = 0.15           # More sensitive to signal decay

    # Regime multipliers
    REGIME_ADJUSTMENTS = {
        'strong_bull': {
            'stop_mult': 1.2,      # Wider stops in strong trends
            'target_mult': 1.5,    # Much wider targets
            'hold_mult': 1.5,      # Hold longer
            'signal_mult': 0.8     # Higher threshold to exit (less sensitive)
        },
        'bull': {
            'stop_mult': 1.1,
            'target_mult': 1.3,
            'hold_mult': 1.2,
            'signal_mult': 0.9
        },
        'neutral': {
            'stop_mult': 0.9,      # Tighter stops in chop
            'target_mult': 0.8,    # Lower targets (take what you can)
            'hold_mult': 0.7,      # Shorter holds
            'signal_mult': 1.2     # More sensitive to signals
        },
        'bear': {
            'stop_mult': 0.8,      # Tight stops
            'target_mult': 1.2,    # Good targets on shorts
            'hold_mult': 0.8,
            'signal_mult': 1.1
        },
        'strong_bear': {
            'stop_mult': 0.7,      # Very tight stops for longs
            'target_mult': 1.4,    # Wide targets for shorts
            'hold_mult': 0.6,
            'signal_mult': 1.3
        },
        'high_volatility': {
            'stop_mult': 0.6,      # Much tighter stops
            'target_mult': 0.7,    # Take profits quickly
            'hold_mult': 0.5,      # Short holds
            'signal_mult': 1.5     # Very sensitive
        }
    }

    def __init__(self):
        self.position_history: Dict[str, list] = {}  # Track P&L trajectory
        self.base_signal_exit = getattr(ApexConfig, "SIGNAL_EXIT_BASE", self.BASE_SIGNAL_EXIT)
        logger.info("DynamicExitManager initialized")

    def calculate_exit_levels(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,  # 'LONG' or 'SHORT'
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime
    ) -> DynamicExitLevels:
        """
        Calculate dynamic exit levels for a position.

        Args:
            symbol: Stock ticker
            entry_price: Entry price
            current_price: Current price
            side: LONG or SHORT
            entry_signal: Signal strength at entry
            current_signal: Current signal strength
            confidence: Current confidence
            regime: Market regime
            vix_level: Current VIX (optional)
            atr: Current ATR (optional)
            entry_time: When position was opened

        Returns:
            DynamicExitLevels with adjusted thresholds
        """
        # Start with base values
        stop_pct = self.BASE_STOP_LOSS_PCT
        target_pct = self.BASE_TAKE_PROFIT_PCT
        trail_activation = self.BASE_TRAILING_ACTIVATION
        trail_distance = self.BASE_TRAILING_DISTANCE
        max_hold = self.BASE_MAX_HOLD_DAYS
        signal_exit = self.base_signal_exit

        # Calculate current P&L
        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
        else:
            pnl_pct = (entry_price / current_price - 1)

        holding_days = (datetime.now() - entry_time).days
        holding_hours = (datetime.now() - entry_time).total_seconds() / 3600

        # === 1. REGIME ADJUSTMENTS ===
        regime_adj = self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS['neutral'])

        # Adjust for side in bearish regimes
        if side == 'LONG' and regime in ['bear', 'strong_bear']:
            # Long in bear market = tighter stops
            stop_pct *= regime_adj['stop_mult'] * 0.8
            target_pct *= regime_adj['target_mult'] * 0.7
        elif side == 'SHORT' and regime in ['bull', 'strong_bull']:
            # Short in bull market = tighter stops
            stop_pct *= regime_adj['stop_mult'] * 0.8
            target_pct *= regime_adj['target_mult'] * 0.7
        else:
            stop_pct *= regime_adj['stop_mult']
            target_pct *= regime_adj['target_mult']

        max_hold = int(max_hold * regime_adj['hold_mult'])
        signal_exit *= regime_adj['signal_mult']

        # === 2. VIX ADJUSTMENTS ===
        if vix_level is not None:
            if vix_level > 30:
                # High fear - very tight stops
                stop_pct *= 0.6
                target_pct *= 0.6
                max_hold = min(max_hold, 5)
                signal_exit *= 1.5
            elif vix_level > 25:
                stop_pct *= 0.75
                target_pct *= 0.75
                max_hold = min(max_hold, 7)
                signal_exit *= 1.3
            elif vix_level > 20:
                stop_pct *= 0.9
                target_pct *= 0.9
                signal_exit *= 1.1
            elif vix_level < 12:
                # Complacency - can hold longer
                stop_pct *= 1.1
                target_pct *= 1.2
                max_hold = int(max_hold * 1.3)

        # === 3. ATR-BASED ADJUSTMENTS ===
        if atr is not None and entry_price > 0:
            atr_pct = atr / entry_price
            # Use ATR to set more realistic stops
            stop_pct = max(stop_pct, atr_pct * 2.0)  # At least 2x ATR
            target_pct = max(target_pct, atr_pct * 2.5)  # At least 2.5x ATR
            trail_distance = max(trail_distance, atr_pct * 1.5)

        # === 4. SIGNAL STRENGTH ADJUSTMENTS ===
        # Strong entry signal = more conviction = can hold longer
        if abs(entry_signal) > 0.7:
            stop_pct *= 1.15
            target_pct *= 1.2
            max_hold = int(max_hold * 1.2)
        elif abs(entry_signal) < 0.5:
            stop_pct *= 0.9
            target_pct *= 0.85
            max_hold = int(max_hold * 0.8)

        # === 5. P&L TRAJECTORY ADJUSTMENTS ===
        if pnl_pct > 0.06:
            # Winning 6%+ - protect with tight trailing
            trail_activation = 0.02
            trail_distance *= 0.7
            # IMPORTANT: Don't exit winners on signal decay
            signal_exit = 0.0  # Disable signal-based exit for big winners
        elif pnl_pct > 0.03:
            # Winning 3%+ - activate trailing but give room
            trail_activation = 0.02
            trail_distance *= 0.85
            signal_exit *= 0.5  # Much less sensitive to signal decay on winners
        elif pnl_pct > 0.015:
            # Small winner - be patient, let it grow
            signal_exit *= 0.7  # Less sensitive
        elif pnl_pct < -0.02:
            # Losing more than 2% - tighten
            signal_exit *= 1.3
            max_hold = min(max_hold, 7)

        # === 6. TIME DECAY ADJUSTMENTS ===
        # Only apply time decay to LOSING positions
        if holding_days >= 7 and pnl_pct < -0.01:
            # Stale losing position - start tightening
            decay_factor = max(0.6, 1 - (holding_days - 7) / max_hold * 0.3)
            signal_exit *= (1 / decay_factor)

        if holding_days >= 10 and pnl_pct < -0.02:
            # 10 days with significant loss - lower expectations
            target_pct *= 0.8

        if holding_days >= max_hold * 0.9:
            # Approaching max hold
            signal_exit *= 0.8

        # === 7. SIGNAL REVERSAL CHECK ===
        # Only care about reversals on losing positions
        signal_reversed = (entry_signal > 0 and current_signal < -0.3) or \
                         (entry_signal < 0 and current_signal > 0.3)

        if signal_reversed and pnl_pct < 0.005:  # Trigger earlier than 0%
            signal_exit *= 0.5  # Much more sensitive when signal flips AND losing or barely winning

        # === DETERMINE EXIT URGENCY ===
        urgency = ExitUrgency.HOLD
        reason = "Position healthy"

        # Check stop loss
        if pnl_pct <= -stop_pct:
            urgency = ExitUrgency.IMMEDIATE
            reason = f"Stop loss hit ({pnl_pct*100:+.1f}%)"

        # Check take profit
        elif pnl_pct >= target_pct:
            urgency = ExitUrgency.HIGH
            reason = f"Take profit reached ({pnl_pct*100:+.1f}%)"

        # Check signal reversal
        elif signal_reversed and pnl_pct < 0:
            urgency = ExitUrgency.HIGH
            reason = f"Signal reversed while losing ({current_signal:+.2f})"

        # Check holding time
        elif holding_days >= max_hold:
            urgency = ExitUrgency.HIGH
            reason = f"Max holding period ({holding_days}d)"

        # Check signal decay
        elif abs(current_signal) < signal_exit and holding_days >= 3:
            if pnl_pct < 0:
                urgency = ExitUrgency.HIGH
                reason = f"Signal weak on loser ({current_signal:+.2f})"
            else:
                urgency = ExitUrgency.MODERATE
                reason = f"Signal decayed ({current_signal:+.2f})"

        # Approaching limits
        elif holding_days >= max_hold * 0.8:
            urgency = ExitUrgency.MODERATE
            reason = f"Approaching max hold ({holding_days}/{max_hold}d)"

        elif pnl_pct <= -stop_pct * 0.7:
            urgency = ExitUrgency.MODERATE
            reason = f"Approaching stop ({pnl_pct*100:+.1f}%)"

        return DynamicExitLevels(
            stop_loss_pct=float(np.clip(stop_pct, 0.02, 0.15)),  # 2-15%
            take_profit_pct=float(np.clip(target_pct, 0.03, 0.25)),  # 3-25%
            trailing_activation_pct=float(np.clip(trail_activation, 0.01, 0.05)),
            trailing_distance_pct=float(np.clip(trail_distance, 0.01, 0.05)),
            max_hold_days=int(np.clip(max_hold, 3, 30)),  # 3-30 days
            signal_exit_threshold=float(np.clip(signal_exit, 0.15, 0.60)),
            urgency=urgency,
            reason=reason
        )

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime,
        peak_price: Optional[float] = None
    ) -> Tuple[bool, str, ExitUrgency]:
        """
        Determine if position should be exited.

        Returns:
            (should_exit, reason, urgency)
        """
        levels = self.calculate_exit_levels(
            symbol, entry_price, current_price, side,
            entry_signal, current_signal, confidence,
            regime, vix_level, atr, entry_time
        )

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
        else:
            pnl_pct = (entry_price / current_price - 1)

        holding_days = (datetime.now() - entry_time).days

        # === HARD EXITS ===

        # Stop loss
        if pnl_pct <= -levels.stop_loss_pct:
            return True, f"Stop loss: {pnl_pct*100:+.1f}% <= -{levels.stop_loss_pct*100:.1f}%", ExitUrgency.IMMEDIATE

        # Take profit
        if pnl_pct >= levels.take_profit_pct:
            return True, f"Take profit: {pnl_pct*100:+.1f}% >= +{levels.take_profit_pct*100:.1f}%", ExitUrgency.HIGH

        # Max holding period
        if holding_days >= levels.max_hold_days:
            return True, f"Max hold: {holding_days}d >= {levels.max_hold_days}d", ExitUrgency.HIGH

        # === TRAILING STOP ===
        if peak_price is not None and pnl_pct > levels.trailing_activation_pct:
            if side == 'LONG':
                trailing_stop = peak_price * (1 - levels.trailing_distance_pct)
                if current_price <= trailing_stop:
                    return True, f"Trailing stop: ${current_price:.2f} <= ${trailing_stop:.2f}", ExitUrgency.IMMEDIATE
            else:
                trailing_stop = peak_price * (1 + levels.trailing_distance_pct)
                if current_price >= trailing_stop:
                    return True, f"Trailing stop: ${current_price:.2f} >= ${trailing_stop:.2f}", ExitUrgency.IMMEDIATE

        # === SIGNAL-BASED EXITS ===

        # Strong reversal signal - only exit if NOT winning big
        if side == 'LONG' and current_signal < -0.50 and confidence > 0.40 and pnl_pct < 0.03:
            return True, f"Strong bearish reversal: {current_signal:+.2f}", ExitUrgency.HIGH

        if side == 'SHORT' and current_signal > 0.50 and confidence > 0.40 and pnl_pct < 0.03:
            return True, f"Strong bullish reversal: {current_signal:+.2f}", ExitUrgency.HIGH

        # Moderate reversal + losing - cut losses fast
        if side == 'LONG' and current_signal < -0.30 and pnl_pct < 0.0:
            return True, f"Bearish on loser: {current_signal:+.2f}, P&L={pnl_pct*100:+.1f}%", ExitUrgency.HIGH

        if side == 'SHORT' and current_signal > 0.30 and pnl_pct < 0.0:
            return True, f"Bullish on loser: {current_signal:+.2f}, P&L={pnl_pct*100:+.1f}%", ExitUrgency.HIGH

        # Signal decay - ONLY on losing positions after extended hold time
        if holding_days >= 5 and pnl_pct < -0.01:
            if side == 'LONG' and current_signal < levels.signal_exit_threshold:
                return True, f"Signal decay on loser: {current_signal:+.2f} after {holding_days}d", ExitUrgency.MODERATE

            if side == 'SHORT' and current_signal > -levels.signal_exit_threshold:
                return True, f"Signal decay on loser: {current_signal:+.2f} after {holding_days}d", ExitUrgency.MODERATE

        # WINNERS: Only exit via trailing stop or take profit (handled above)
        # Don't exit winners on signal decay - let the trailing stop do its job

        # No exit needed
        return False, levels.reason, levels.urgency

    def get_position_status(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime
    ) -> Dict:
        """Get comprehensive status for a position."""
        levels = self.calculate_exit_levels(
            symbol, entry_price, current_price, side,
            entry_signal, current_signal, confidence,
            regime, vix_level, atr, entry_time
        )

        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
            stop_price = entry_price * (1 - levels.stop_loss_pct)
            target_price = entry_price * (1 + levels.take_profit_pct)
        else:
            pnl_pct = (entry_price / current_price - 1)
            stop_price = entry_price * (1 + levels.stop_loss_pct)
            target_price = entry_price * (1 - levels.take_profit_pct)

        holding_days = (datetime.now() - entry_time).days

        return {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl_pct': pnl_pct,
            'holding_days': holding_days,
            'stop_price': stop_price,
            'target_price': target_price,
            'stop_pct': levels.stop_loss_pct,
            'target_pct': levels.take_profit_pct,
            'max_hold_days': levels.max_hold_days,
            'days_remaining': levels.max_hold_days - holding_days,
            'signal_exit_threshold': levels.signal_exit_threshold,
            'current_signal': current_signal,
            'urgency': levels.urgency.value,
            'status': levels.reason,
            'regime': regime,
            'vix': vix_level
        }


# Singleton instance
_exit_manager: Optional[DynamicExitManager] = None


def get_exit_manager() -> DynamicExitManager:
    """Get or create the dynamic exit manager singleton."""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = DynamicExitManager()
    return _exit_manager
