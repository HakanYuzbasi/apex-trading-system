"""
risk/trading_excellence.py - Trading Excellence Module

Implements critical excellence features:
1. Signal-Position Mismatch Detection - Alert/exit when signal contradicts position
2. Automatic Profit-Taking - Take profits on large winners (>5%)
3. Position Size Scaling - Scale size by signal strength and conviction
4. Winner Management - Trailing stops and profit protection for winners

These features transform good trading into excellent trading.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MismatchSeverity(Enum):
    """Severity levels for signal-position mismatches."""
    NONE = "none"           # No mismatch
    WEAK = "weak"           # Minor divergence (signal weakening)
    MODERATE = "moderate"   # Signal neutral, position active
    STRONG = "strong"       # Signal opposite to position
    CRITICAL = "critical"   # Strong opposite signal with high confidence


class ProfitAction(Enum):
    """Actions for profit-taking."""
    HOLD = "hold"           # Continue holding
    PARTIAL = "partial"     # Take partial profits (50%)
    FULL = "full"           # Take full profits
    TRAIL = "trail"         # Activate/tighten trailing stop


@dataclass
class MismatchAlert:
    """Alert for signal-position mismatch."""
    symbol: str
    severity: MismatchSeverity
    position_side: str          # 'LONG' or 'SHORT'
    signal_direction: str       # 'bullish', 'bearish', 'neutral'
    signal_value: float
    confidence: float
    position_pnl_pct: float
    recommendation: str         # What to do
    urgency: str                # 'immediate', 'high', 'moderate', 'low'


@dataclass
class ProfitDecision:
    """Decision for profit-taking."""
    symbol: str
    action: ProfitAction
    current_pnl_pct: float
    profit_tier: int            # 0=none, 1=2%, 2=5%, 3=10%, 4=20%
    trailing_stop_pct: float    # Current trailing stop level
    reason: str


@dataclass
class SizeRecommendation:
    """Position size recommendation based on signal strength."""
    base_shares: int
    recommended_shares: int
    scaling_factor: float       # 0.5 to 1.5
    reasons: List[str]


class TradingExcellenceManager:
    """
    Manager for trading excellence features.

    Key principles:
    - Never fight the signal (exit when signal turns against you)
    - Take profits when you have them (winners can become losers)
    - Size according to conviction (bigger size for stronger signals)
    - Protect gains aggressively (trailing stops tighten as profits grow)
    """

    # Signal mismatch thresholds
    WEAK_MISMATCH_THRESHOLD = 0.10      # Signal drops below 10%
    MODERATE_MISMATCH_THRESHOLD = 0.05  # Signal drops below 5%
    STRONG_MISMATCH_THRESHOLD = -0.15   # Signal flips opposite
    CRITICAL_MISMATCH_THRESHOLD = -0.30 # Strong opposite signal
    CRITICAL_CONFIDENCE_THRESHOLD = 0.70 # High confidence opposite signal

    # Profit-taking tiers (profit % -> trailing stop %)
    PROFIT_TIERS = {
        0.02: 0.50,   # 2% profit = lock 50% (trail at 1%)
        0.05: 0.70,   # 5% profit = lock 70% (trail at 1.5%)
        0.10: 0.80,   # 10% profit = lock 80% (trail at 2%)
        0.20: 0.85,   # 20% profit = lock 85% (trail at 3%)
        0.30: 0.90,   # 30% profit = lock 90% (trail at 3%)
    }

    # Auto profit-take thresholds
    AUTO_PROFIT_TAKE_THRESHOLD = 0.05   # Auto-take at 5%+ when signal weakens
    AGGRESSIVE_PROFIT_THRESHOLD = 0.10  # Always partial take at 10%+

    # Position sizing by signal strength
    SIGNAL_SIZE_SCALING = {
        (0.0, 0.25): 0.50,    # Weak signal = 50% size
        (0.25, 0.40): 0.75,   # Moderate signal = 75% size
        (0.40, 0.60): 1.00,   # Good signal = 100% size
        (0.60, 0.80): 1.20,   # Strong signal = 120% size
        (0.80, 1.00): 1.30,   # Very strong signal = 130% size
    }

    # Confidence adjustment
    MIN_CONFIDENCE_FOR_FULL_SIZE = 0.60
    CONFIDENCE_SIZE_PENALTY = 0.30  # Reduce by 30% if confidence is low

    def __init__(self):
        self.mismatch_history: Dict[str, List[MismatchAlert]] = {}
        self.profit_history: Dict[str, List[float]] = {}  # Track profit trajectory
        self.last_mismatch_check: Dict[str, datetime] = {}
        logger.info("TradingExcellenceManager initialized")

    def check_signal_mismatch(
        self,
        symbol: str,
        position_side: str,
        position_qty: int,
        signal: float,
        confidence: float,
        entry_price: float,
        current_price: float
    ) -> Optional[MismatchAlert]:
        """
        Check for signal-position mismatch.

        A mismatch occurs when:
        - LONG position but signal is negative/bearish
        - SHORT position but signal is positive/bullish
        - Any position but signal is very weak/neutral

        Args:
            symbol: Stock ticker
            position_side: 'LONG' or 'SHORT'
            position_qty: Current position quantity
            signal: Current signal value (-1 to +1)
            confidence: Signal confidence (0 to 1)
            entry_price: Position entry price
            current_price: Current price

        Returns:
            MismatchAlert if mismatch detected, None otherwise
        """
        if position_qty == 0:
            return None

        # Calculate P&L
        if position_side == 'LONG':
            pnl_pct = (current_price / entry_price - 1) * 100
            signal_aligned = signal > 0
        else:  # SHORT
            pnl_pct = (entry_price / current_price - 1) * 100
            signal_aligned = signal < 0

        # Determine signal direction
        if signal > 0.20:
            signal_direction = 'bullish'
        elif signal < -0.20:
            signal_direction = 'bearish'
        else:
            signal_direction = 'neutral'

        # Check for mismatches
        severity = MismatchSeverity.NONE
        recommendation = "Hold position"
        urgency = "low"

        # Strong opposite signal with high confidence = CRITICAL
        if position_side == 'LONG' and signal < self.CRITICAL_MISMATCH_THRESHOLD and confidence > self.CRITICAL_CONFIDENCE_THRESHOLD:
            severity = MismatchSeverity.CRITICAL
            recommendation = "EXIT IMMEDIATELY - Strong bearish signal with high confidence"
            urgency = "immediate"

        elif position_side == 'SHORT' and signal > -self.CRITICAL_MISMATCH_THRESHOLD and confidence > self.CRITICAL_CONFIDENCE_THRESHOLD:
            severity = MismatchSeverity.CRITICAL
            recommendation = "EXIT IMMEDIATELY - Strong bullish signal with high confidence"
            urgency = "immediate"

        # Signal flipped opposite = STRONG
        elif position_side == 'LONG' and signal < self.STRONG_MISMATCH_THRESHOLD:
            severity = MismatchSeverity.STRONG
            recommendation = "EXIT - Signal has turned bearish"
            urgency = "high"

        elif position_side == 'SHORT' and signal > -self.STRONG_MISMATCH_THRESHOLD:
            severity = MismatchSeverity.STRONG
            recommendation = "EXIT - Signal has turned bullish"
            urgency = "high"

        # Signal is neutral/weak = MODERATE
        elif abs(signal) < self.MODERATE_MISMATCH_THRESHOLD:
            severity = MismatchSeverity.MODERATE
            if pnl_pct > 3:
                recommendation = f"TAKE PROFITS - Signal neutral, sitting on {pnl_pct:.1f}% gain"
                urgency = "moderate"
            elif pnl_pct < -1:
                recommendation = f"EXIT - Signal neutral, position losing {abs(pnl_pct):.1f}%"
                urgency = "high"
            else:
                recommendation = "Monitor closely - signal has weakened significantly"
                urgency = "moderate"

        # Signal weakening = WEAK
        elif not signal_aligned and abs(signal) < self.WEAK_MISMATCH_THRESHOLD:
            severity = MismatchSeverity.WEAK
            recommendation = "Signal weakening - consider tightening stops"
            urgency = "low"

        if severity == MismatchSeverity.NONE:
            return None

        alert = MismatchAlert(
            symbol=symbol,
            severity=severity,
            position_side=position_side,
            signal_direction=signal_direction,
            signal_value=signal,
            confidence=confidence,
            position_pnl_pct=pnl_pct,
            recommendation=recommendation,
            urgency=urgency
        )

        # Log the alert
        if severity in [MismatchSeverity.CRITICAL, MismatchSeverity.STRONG]:
            logger.warning(f"🚨 {symbol}: SIGNAL MISMATCH [{severity.value.upper()}]")
            logger.warning(f"   Position: {position_side} | Signal: {signal:+.3f} ({signal_direction})")
            logger.warning(f"   P&L: {pnl_pct:+.2f}% | Confidence: {confidence:.2f}")
            logger.warning(f"   ➡️ {recommendation}")
        elif severity == MismatchSeverity.MODERATE:
            logger.info(f"⚠️ {symbol}: Signal mismatch [{severity.value}] - {recommendation}")

        # Track history
        if symbol not in self.mismatch_history:
            self.mismatch_history[symbol] = []
        self.mismatch_history[symbol].append(alert)

        return alert

    def get_profit_decision(
        self,
        symbol: str,
        position_side: str,
        entry_price: float,
        current_price: float,
        peak_price: float,
        signal: float,
        confidence: float
    ) -> ProfitDecision:
        """
        Determine profit-taking action for a position.

        Rules:
        1. At 2% profit - activate trailing stop at 50% of gains
        2. At 5% profit - tighten trailing to 70% of gains
        3. At 10% profit - tighten trailing to 80% of gains
        4. If signal weakens significantly at 5%+ profit - take full profits
        5. At 10%+ profit with neutral signal - take partial profits (50%)

        Args:
            symbol: Stock ticker
            position_side: 'LONG' or 'SHORT'
            entry_price: Entry price
            current_price: Current price
            peak_price: Peak price since entry
            signal: Current signal value
            confidence: Signal confidence

        Returns:
            ProfitDecision with recommended action
        """
        # Calculate P&L
        if position_side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
            peak_pnl_pct = (peak_price / entry_price - 1)
            drawdown_from_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0
        else:
            pnl_pct = (entry_price / current_price - 1)
            peak_pnl_pct = (entry_price / peak_price - 1) if peak_price > 0 else pnl_pct
            drawdown_from_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0

        # Determine current profit tier
        profit_tier = 0
        trailing_stop_pct = 0.03  # Default 3% trailing stop

        for threshold, lock_pct in sorted(self.PROFIT_TIERS.items()):
            if peak_pnl_pct >= threshold:
                profit_tier = list(self.PROFIT_TIERS.keys()).index(threshold) + 1
                # Calculate trailing stop level (lock this % of gains)
                trailing_stop_pct = peak_pnl_pct * (1 - lock_pct)

        # Default action
        action = ProfitAction.HOLD
        reason = "Position within normal parameters"

        # Check signal alignment
        signal_weak = abs(signal) < 0.15
        signal_opposite = (position_side == 'LONG' and signal < -0.10) or (position_side == 'SHORT' and signal > 0.10)

        # Rule 1: At 10%+ profit with signal turning against - FULL exit
        if pnl_pct >= 0.10 and signal_opposite:
            action = ProfitAction.FULL
            reason = f"10%+ profit ({pnl_pct*100:.1f}%) with signal reversal - take full profits"

        # Rule 2: At 5%+ profit with weak/neutral signal - PARTIAL exit
        elif pnl_pct >= self.AUTO_PROFIT_TAKE_THRESHOLD and signal_weak:
            action = ProfitAction.PARTIAL
            reason = f"5%+ profit ({pnl_pct*100:.1f}%) with weak signal ({signal:+.2f}) - take partial profits"

        # Rule 3: At 10%+ profit regardless of signal - PARTIAL exit (lock gains)
        elif pnl_pct >= self.AGGRESSIVE_PROFIT_THRESHOLD:
            action = ProfitAction.PARTIAL
            reason = f"10%+ profit ({pnl_pct*100:.1f}%) - taking partial profits to lock gains"

        # Rule 4: Drawdown from peak exceeds trailing stop - FULL exit
        elif profit_tier > 0 and drawdown_from_peak > trailing_stop_pct:
            action = ProfitAction.FULL
            reason = f"Trailing stop triggered - {drawdown_from_peak*100:.1f}% drawdown from {peak_pnl_pct*100:.1f}% peak"

        # Rule 5: At profit tier - activate/tighten trailing
        elif profit_tier > 0:
            action = ProfitAction.TRAIL
            reason = f"Tier {profit_tier} profit ({pnl_pct*100:.1f}%) - trailing stop at {trailing_stop_pct*100:.2f}%"

        decision = ProfitDecision(
            symbol=symbol,
            action=action,
            current_pnl_pct=pnl_pct * 100,
            profit_tier=profit_tier,
            trailing_stop_pct=trailing_stop_pct * 100,
            reason=reason
        )

        # Log significant decisions
        if action in [ProfitAction.FULL, ProfitAction.PARTIAL]:
            logger.info(f"💰 {symbol}: PROFIT ACTION [{action.value.upper()}]")
            logger.info(f"   P&L: {pnl_pct*100:+.2f}% | Peak: {peak_pnl_pct*100:.2f}%")
            logger.info(f"   ➡️ {reason}")

        return decision

    def calculate_size_scaling(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        regime: str,
        base_shares: int
    ) -> SizeRecommendation:
        """
        Calculate position size scaling based on signal strength and confidence.

        Scaling rules:
        - Strong signals (>0.6) = up to 130% of base size
        - Moderate signals (0.4-0.6) = 100% of base size
        - Weak signals (<0.25) = 50% of base size
        - Low confidence (<0.6) = additional 30% reduction
        - Volatile regime = 25% reduction

        Args:
            symbol: Stock ticker
            signal: Signal value (-1 to +1)
            confidence: Signal confidence (0 to 1)
            regime: Market regime
            base_shares: Base position size in shares

        Returns:
            SizeRecommendation with scaling factor and reasons
        """
        abs_signal = abs(signal)
        scaling_factor = 1.0
        reasons = []

        # 1. Signal strength scaling
        for (low, high), scale in self.SIGNAL_SIZE_SCALING.items():
            if low <= abs_signal < high:
                scaling_factor = scale
                if scale < 1.0:
                    reasons.append(f"Weak signal ({abs_signal:.2f}) = {scale:.0%} size")
                elif scale > 1.0:
                    reasons.append(f"Strong signal ({abs_signal:.2f}) = {scale:.0%} size")
                break

        # 2. Confidence adjustment
        if confidence < self.MIN_CONFIDENCE_FOR_FULL_SIZE:
            confidence_penalty = self.CONFIDENCE_SIZE_PENALTY
            scaling_factor *= (1 - confidence_penalty)
            reasons.append(f"Low confidence ({confidence:.2f}) = -{confidence_penalty:.0%}")

        # 3. Regime adjustment
        if regime in ['high_volatility', 'volatile']:
            scaling_factor *= 0.75
            reasons.append("Volatile regime = -25%")
        elif regime in ['strong_bear'] and signal > 0:
            scaling_factor *= 0.70
            reasons.append("Long in bear market = -30%")
        elif regime in ['strong_bull'] and signal < 0:
            scaling_factor *= 0.70
            reasons.append("Short in bull market = -30%")

        # Clamp scaling factor
        scaling_factor = max(0.25, min(1.50, scaling_factor))

        recommended_shares = max(1, int(base_shares * scaling_factor))

        return SizeRecommendation(
            base_shares=base_shares,
            recommended_shares=recommended_shares,
            scaling_factor=scaling_factor,
            reasons=reasons if reasons else ["Standard sizing"]
        )

    def should_exit_on_mismatch(self, alert: Optional[MismatchAlert]) -> Tuple[bool, str]:
        """
        Determine if a position should be exited based on mismatch alert.

        Args:
            alert: MismatchAlert from check_signal_mismatch

        Returns:
            Tuple of (should_exit, reason)
        """
        if alert is None:
            return False, ""

        if alert.severity == MismatchSeverity.CRITICAL:
            return True, f"CRITICAL mismatch: {alert.recommendation}"

        if alert.severity == MismatchSeverity.STRONG:
            return True, f"STRONG mismatch: {alert.recommendation}"

        if alert.severity == MismatchSeverity.MODERATE:
            # Exit if losing money with neutral signal
            if alert.position_pnl_pct < -1:
                return True, f"Moderate mismatch with loss: {alert.recommendation}"
            # Exit if decent profit with neutral signal (take profit)
            if alert.position_pnl_pct > 5:
                return True, f"Moderate mismatch with profit: {alert.recommendation}"

        return False, ""

    def get_all_position_alerts(
        self,
        positions: Dict[str, int],
        entry_prices: Dict[str, float],
        current_prices: Dict[str, float],
        signals: Dict[str, float],
        confidences: Dict[str, float]
    ) -> List[MismatchAlert]:
        """
        Check all positions for signal mismatches.

        Args:
            positions: Dict of symbol -> quantity
            entry_prices: Dict of symbol -> entry price
            current_prices: Dict of symbol -> current price
            signals: Dict of symbol -> current signal
            confidences: Dict of symbol -> current confidence

        Returns:
            List of MismatchAlerts for all positions with issues
        """
        alerts = []

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            side = 'LONG' if qty > 0 else 'SHORT'
            entry_price = entry_prices.get(symbol, 0)
            current_price = current_prices.get(symbol, 0)
            signal = signals.get(symbol, 0)
            confidence = confidences.get(symbol, 0.5)

            if entry_price <= 0 or current_price <= 0:
                continue

            alert = self.check_signal_mismatch(
                symbol=symbol,
                position_side=side,
                position_qty=qty,
                signal=signal,
                confidence=confidence,
                entry_price=entry_price,
                current_price=current_price
            )

            if alert:
                alerts.append(alert)

        # Sort by severity (CRITICAL first)
        severity_order = {
            MismatchSeverity.CRITICAL: 0,
            MismatchSeverity.STRONG: 1,
            MismatchSeverity.MODERATE: 2,
            MismatchSeverity.WEAK: 3
        }
        alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

        return alerts

    def get_positions_to_take_profit(
        self,
        positions: Dict[str, int],
        entry_prices: Dict[str, float],
        current_prices: Dict[str, float],
        peak_prices: Dict[str, float],
        signals: Dict[str, float],
        confidences: Dict[str, float]
    ) -> List[ProfitDecision]:
        """
        Get all positions that should have profits taken.

        Args:
            positions: Dict of symbol -> quantity
            entry_prices: Dict of symbol -> entry price
            current_prices: Dict of symbol -> current price
            peak_prices: Dict of symbol -> peak price since entry
            signals: Dict of symbol -> current signal
            confidences: Dict of symbol -> current confidence

        Returns:
            List of ProfitDecisions for positions needing action
        """
        decisions = []

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            side = 'LONG' if qty > 0 else 'SHORT'
            entry_price = entry_prices.get(symbol, 0)
            current_price = current_prices.get(symbol, 0)
            peak_price = peak_prices.get(symbol, current_price)
            signal = signals.get(symbol, 0)
            confidence = confidences.get(symbol, 0.5)

            if entry_price <= 0 or current_price <= 0:
                continue

            decision = self.get_profit_decision(
                symbol=symbol,
                position_side=side,
                entry_price=entry_price,
                current_price=current_price,
                peak_price=peak_price,
                signal=signal,
                confidence=confidence
            )

            if decision.action in [ProfitAction.FULL, ProfitAction.PARTIAL]:
                decisions.append(decision)

        return decisions


# Convenience function for quick checks
def quick_mismatch_check(
    position_side: str,
    signal: float,
    confidence: float,
    pnl_pct: float
) -> Tuple[bool, str]:
    """
    Quick check for signal-position mismatch without full object.

    Returns:
        Tuple of (should_exit, reason)
    """
    # LONG position checks
    if position_side == 'LONG':
        if signal < -0.30 and confidence > 0.70:
            return True, f"Strong bearish signal ({signal:+.2f}) with high confidence"
        if signal < -0.15:
            return True, f"Signal turned bearish ({signal:+.2f})"
        if signal < 0.05 and pnl_pct > 5:
            return True, f"Weak signal ({signal:+.2f}) with {pnl_pct:.1f}% profit - take profits"
        if signal < 0.05 and pnl_pct < -1:
            return True, f"Weak signal ({signal:+.2f}) with {pnl_pct:.1f}% loss - exit"

    # SHORT position checks
    elif position_side == 'SHORT':
        if signal > 0.30 and confidence > 0.70:
            return True, f"Strong bullish signal ({signal:+.2f}) with high confidence"
        if signal > 0.15:
            return True, f"Signal turned bullish ({signal:+.2f})"
        if signal > -0.05 and pnl_pct > 5:
            return True, f"Weak signal ({signal:+.2f}) with {pnl_pct:.1f}% profit - take profits"
        if signal > -0.05 and pnl_pct < -1:
            return True, f"Weak signal ({signal:+.2f}) with {pnl_pct:.1f}% loss - exit"

    return False, ""
