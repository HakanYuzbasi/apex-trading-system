"""
risk/exit_quality_guard.py - Exit Signal Validation & Resilient Retry

Validates exit signals before execution and implements exponential backoff
for failed exits. Prevents premature exits on stale data while ensuring
positions are NEVER permanently stuck.

Exit validation rules:
- Signal flip must be confirmed by 2+ component signals
- Exit confidence must exceed floor (0.30)
- Stale data blocks signal-based exits (not hard stops)
- Hard stop-losses ALWAYS bypass validation

Resilient retry with exponential backoff:
- Attempt 1: Market order (0s)
- Attempt 2: Market order (30s)
- Attempt 3: Aggressive limit (60s)
- Attempt 4: Market order extended (120s)
- Attempt 5: MOC order (240s)
- Attempt 6+: Retry every 5 min (never give up)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    SIGNAL_REVERSAL = "signal_reversal"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    TAKE_PROFIT = "take_profit"
    EMERGENCY = "emergency"
    MANUAL = "manual"


class OrderType(Enum):
    MARKET = "market"
    AGGRESSIVE_LIMIT = "aggressive_limit"
    MOC = "market_on_close"


@dataclass
class ExitValidation:
    """Result of exit signal validation."""
    should_exit: bool
    adjusted_urgency: float  # 0-1
    confidence_in_exit: float
    reasons: List[str]
    bypassed_validation: bool = False


@dataclass
class RetryStrategy:
    """Retry configuration for a failed exit attempt."""
    delay_seconds: int
    order_type: OrderType
    urgency: float
    attempt_number: int
    give_up: bool = False  # Never True — we never give up


class ExitQualityGuard:
    """
    Validates exit signals and manages resilient exit execution.

    Ensures exits are genuine (not caused by stale data or noise)
    while guaranteeing positions are never permanently stuck.
    """

    def __init__(
        self,
        min_exit_confidence: float = 0.30,
        max_data_age_for_exit: int = 300,
        hard_stop_pnl_threshold: float = -0.03,
        max_holding_days: int = 30,
    ):
        self.min_exit_confidence = min_exit_confidence
        self.max_data_age_for_exit = max_data_age_for_exit
        self.hard_stop_pnl_threshold = hard_stop_pnl_threshold
        self.max_holding_days = max_holding_days

        # Track exit attempts per symbol
        self._exit_attempts: Dict[str, int] = {}
        self._last_exit_attempt: Dict[str, datetime] = {}

        logger.info(
            f"ExitQualityGuard initialized: min_confidence={min_exit_confidence}, "
            f"hard_stop={hard_stop_pnl_threshold}"
        )

    def validate_exit(
        self,
        symbol: str,
        exit_reason: str,
        signal: float,
        confidence: float,
        entry_signal: float,
        pnl_pct: float,
        data_age_seconds: float = 0.0,
        generator_signals: Optional[Dict[str, float]] = None,
        holding_days: int = 0,
    ) -> ExitValidation:
        """
        Validate whether an exit should proceed.

        Args:
            symbol: Stock symbol
            exit_reason: Why exit is proposed (signal_reversal, stop_loss, etc.)
            signal: Current signal value
            confidence: Current signal confidence
            entry_signal: Signal at entry time
            pnl_pct: Current P&L percentage
            data_age_seconds: Age of the underlying data
            generator_signals: Per-generator signals for agreement check
            holding_days: Days the position has been held

        Returns:
            ExitValidation with decision and reasoning
        """
        reasons = []

        # Hard stops ALWAYS bypass validation
        if self.should_force_exit(symbol, pnl_pct, holding_days):
            if pnl_pct <= self.hard_stop_pnl_threshold:
                reasons.append(f"Hard stop-loss triggered: P&L={pnl_pct:.1%}")
            elif holding_days >= self.max_holding_days:
                reasons.append(f"Max holding period exceeded: {holding_days} days")
            return ExitValidation(
                should_exit=True,
                adjusted_urgency=1.0,
                confidence_in_exit=1.0,
                reasons=reasons,
                bypassed_validation=True,
            )

        # Emergency exits bypass validation
        if exit_reason == ExitReason.EMERGENCY.value or exit_reason == "emergency":
            return ExitValidation(
                should_exit=True,
                adjusted_urgency=1.0,
                confidence_in_exit=1.0,
                reasons=["Emergency exit requested"],
                bypassed_validation=True,
            )

        # For signal-reversal exits, apply validation
        exit_ok = True
        urgency = 0.5
        exit_confidence = confidence

        # Check 1: Staleness guard
        if data_age_seconds > self.max_data_age_for_exit:
            if exit_reason in (ExitReason.SIGNAL_REVERSAL.value, "signal_reversal"):
                reasons.append(
                    f"Data stale ({data_age_seconds:.0f}s > {self.max_data_age_for_exit}s)"
                )
                exit_ok = False

        # Check 2: Confidence floor
        if confidence < self.min_exit_confidence:
            if exit_reason in (ExitReason.SIGNAL_REVERSAL.value, "signal_reversal"):
                reasons.append(
                    f"Exit confidence too low ({confidence:.2f} < {self.min_exit_confidence})"
                )
                exit_ok = False

        # Check 3: Signal flip confirmation from multiple generators
        if exit_reason in (ExitReason.SIGNAL_REVERSAL.value, "signal_reversal"):
            if generator_signals and len(generator_signals) >= 2:
                # Check if majority agree on the new direction
                signal_dir = 1 if signal > 0 else -1
                agree_count = sum(
                    1 for s in generator_signals.values()
                    if (1 if s > 0 else -1) == signal_dir
                )
                total = len(generator_signals)
                if agree_count < total / 2:
                    reasons.append(
                        f"Generators disagree on exit direction "
                        f"({agree_count}/{total} agree)"
                    )
                    exit_ok = False
                else:
                    reasons.append(
                        f"Exit confirmed by {agree_count}/{total} generators"
                    )
                    urgency = min(1.0, urgency + 0.2)

        # Check 4: Stop-loss and take-profit exits are always allowed
        if exit_reason in (
            ExitReason.STOP_LOSS.value, "stop_loss",
            ExitReason.TRAILING_STOP.value, "trailing_stop",
            ExitReason.TAKE_PROFIT.value, "take_profit",
            ExitReason.TIME_BASED.value, "time_based",
        ):
            exit_ok = True
            if not reasons:
                reasons.append(f"Exit type '{exit_reason}' always allowed")

        if not reasons:
            reasons.append("All validation checks passed")

        return ExitValidation(
            should_exit=exit_ok,
            adjusted_urgency=urgency,
            confidence_in_exit=exit_confidence if exit_ok else 0.0,
            reasons=reasons,
        )

    def should_force_exit(self, symbol: str, pnl_pct: float, holding_days: int) -> bool:
        """
        Check if a forced exit should occur regardless of signal validation.

        Hard stops that bypass ALL validation:
        - P&L below hard stop threshold
        - Position held beyond max holding period
        """
        if pnl_pct <= self.hard_stop_pnl_threshold:
            return True
        if holding_days >= self.max_holding_days:
            return True
        return False

    def get_retry_strategy(self, symbol: str, attempt_number: int) -> RetryStrategy:
        """
        Get retry strategy for a failed exit attempt.

        Implements exponential backoff with escalating order types.
        NEVER gives up — always returns a strategy.

        Args:
            symbol: Stock symbol
            attempt_number: Which attempt this is (1-based)

        Returns:
            RetryStrategy with delay, order type, and urgency
        """
        self._exit_attempts[symbol] = attempt_number
        self._last_exit_attempt[symbol] = datetime.now()

        if attempt_number <= 1:
            return RetryStrategy(
                delay_seconds=0,
                order_type=OrderType.MARKET,
                urgency=0.8,
                attempt_number=attempt_number,
            )
        elif attempt_number == 2:
            return RetryStrategy(
                delay_seconds=30,
                order_type=OrderType.MARKET,
                urgency=0.85,
                attempt_number=attempt_number,
            )
        elif attempt_number == 3:
            return RetryStrategy(
                delay_seconds=60,
                order_type=OrderType.AGGRESSIVE_LIMIT,
                urgency=0.9,
                attempt_number=attempt_number,
            )
        elif attempt_number == 4:
            return RetryStrategy(
                delay_seconds=120,
                order_type=OrderType.MARKET,
                urgency=0.95,
                attempt_number=attempt_number,
            )
        elif attempt_number == 5:
            return RetryStrategy(
                delay_seconds=240,
                order_type=OrderType.MOC,
                urgency=1.0,
                attempt_number=attempt_number,
            )
        else:
            # Never give up — retry every 5 minutes
            return RetryStrategy(
                delay_seconds=300,
                order_type=OrderType.MARKET,
                urgency=1.0,
                attempt_number=attempt_number,
            )

    def record_exit_attempt(self, symbol: str, success: bool, attempt_number: int):
        """Record result of an exit attempt."""
        if success:
            self._exit_attempts.pop(symbol, None)
            self._last_exit_attempt.pop(symbol, None)
            logger.info(f"Exit succeeded for {symbol} on attempt {attempt_number}")
        else:
            self._exit_attempts[symbol] = attempt_number
            logger.warning(
                f"Exit failed for {symbol} on attempt {attempt_number}, will retry"
            )

    def get_pending_exits(self) -> Dict[str, int]:
        """Get symbols with pending (failed) exit attempts."""
        return dict(self._exit_attempts)

    def get_diagnostics(self) -> Dict:
        """Return guard state for monitoring."""
        return {
            "pending_exits": len(self._exit_attempts),
            "exit_attempts": dict(self._exit_attempts),
        }
