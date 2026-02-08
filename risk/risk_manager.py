"""
risk/risk_manager.py - Enhanced Risk Management

Provides comprehensive risk controls including:
- Daily loss limits with automatic enforcement
- Drawdown monitoring and alerts
- Circuit breaker for automatic trading halts
- Position sizing with configurable limits

The circuit breaker trips when:
- Daily loss exceeds threshold
- Drawdown exceeds threshold
- Consecutive losing trades exceed limit

After tripping, trading halts for a configurable cooldown period.
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

from config import ApexConfig

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker to halt trading during adverse conditions.

    Triggers:
    - Daily loss exceeds threshold
    - Drawdown exceeds threshold
    - Consecutive losing trades exceed limit
    """

    def __init__(self):
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.consecutive_losses = 0
        self.recent_trades: List[Dict] = []  # Track recent trade P&L

    def record_trade(self, pnl: float):
        """Record a trade for consecutive loss tracking."""
        self.recent_trades.append({
            'timestamp': datetime.now(),
            'pnl': pnl
        })

        # Keep only last 20 trades
        if len(self.recent_trades) > 20:
            self.recent_trades = self.recent_trades[-20:]

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check consecutive loss limit
        if ApexConfig.CIRCUIT_BREAKER_ENABLED:
            if self.consecutive_losses >= ApexConfig.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
                self.trip(f"Consecutive losses: {self.consecutive_losses}")

    def trip(self, reason: str):
        """Trip the circuit breaker."""
        if not self.is_tripped:
            self.is_tripped = True
            self.trip_reason = reason
            self.trip_time = datetime.now()
            logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
            logger.error(f"   Trading halted at {self.trip_time}")
            logger.error(f"   Cooldown: {ApexConfig.CIRCUIT_BREAKER_COOLDOWN_HOURS} hours")

    def check_and_reset(self) -> bool:
        """
        Check if circuit breaker can be reset.

        Returns:
            True if trading is allowed, False if still tripped
        """
        if not self.is_tripped:
            return True

        if self.trip_time is None:
            return True

        # Check cooldown period
        cooldown = timedelta(hours=ApexConfig.CIRCUIT_BREAKER_COOLDOWN_HOURS)
        if datetime.now() - self.trip_time >= cooldown:
            logger.info("✅ Circuit breaker cooldown complete - trading resumed")
            self.reset()
            return True

        remaining = cooldown - (datetime.now() - self.trip_time)
        logger.warning(
            f"⏳ Circuit breaker active - {remaining.total_seconds() / 3600:.1f}h remaining"
        )
        return False

    def reset(self):
        """Reset the circuit breaker."""
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.consecutive_losses = 0
        logger.info("🔄 Circuit breaker reset")

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            'is_tripped': self.is_tripped,
            'reason': self.trip_reason,
            'trip_time': self.trip_time.isoformat() if self.trip_time else None,
            'consecutive_losses': self.consecutive_losses,
            'recent_trades': len(self.recent_trades)
        }


class RiskManager:
    """Manage risk limits, position sizing, and circuit breaker."""

    def __init__(self, max_daily_loss: float = 0.02, max_drawdown: float = 0.10):
        """
        Initialize risk manager.

        Args:
            max_daily_loss: Max daily loss as fraction of capital (e.g., 0.02 = 2%)
            max_drawdown: Max drawdown from peak as fraction (e.g., 0.10 = 10%)
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        self.starting_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.day_start_capital: float = 0.0
        self.current_day: str = datetime.now().strftime('%Y-%m-%d')

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()

        logger.info(f"🛡️  Risk Manager initialized:")
        logger.info(f"   Max Daily Loss: {max_daily_loss*100:.1f}%")
        logger.info(f"   Max Drawdown: {max_drawdown*100:.1f}%")
        if ApexConfig.CIRCUIT_BREAKER_ENABLED:
            logger.info(f"   Circuit Breaker: ENABLED")
            logger.info(f"      Daily Loss Trigger: {ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS*100:.1f}%")
            logger.info(f"      Drawdown Trigger: {ApexConfig.CIRCUIT_BREAKER_DRAWDOWN*100:.1f}%")
            logger.info(f"      Consecutive Loss Trigger: {ApexConfig.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES}")

    def save_state(self):
        """Save risk state to disk."""
        try:
            state = {
                'day_start_capital': self.day_start_capital,
                'peak_capital': self.peak_capital,
                'starting_capital': self.starting_capital,
                'current_day': self.current_day,
                'circuit_breaker': self.circuit_breaker.get_status()
            }
            
            ApexConfig.DATA_DIR.mkdir(exist_ok=True)
            
            with open(ApexConfig.DATA_DIR / "risk_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
            logger.debug("💾 Risk state saved")
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")

    def load_state(self):
        """Load risk state from disk."""
        try:
            state_file = ApexConfig.DATA_DIR / "risk_state.json"
            if not state_file.exists():
                return
            
            with open(state_file, "r") as f:
                state = json.load(f)
            
            # Only restore day_start_capital if it's the same day
            today = datetime.now().strftime('%Y-%m-%d')
            if state.get('current_day') == today:
                self.day_start_capital = float(state.get('day_start_capital', 0))
                self.current_day = today
                logger.info(f"💾 Restored daily start capital: ${self.day_start_capital:,.2f}")
            else:
                logger.info("📅 New day detected, not restoring daily start capital")
            
            self.peak_capital = float(state.get('peak_capital', self.peak_capital))
            self.starting_capital = float(state.get('starting_capital', self.starting_capital))
            
            cb_state = state.get('circuit_breaker', {})
            if cb_state.get('is_tripped'):
                self.circuit_breaker.is_tripped = True
                self.circuit_breaker.trip_reason = cb_state.get('reason')
                self.circuit_breaker.trip_time = datetime.fromisoformat(cb_state['trip_time']) if cb_state.get('trip_time') else None
                logger.warning(f"🚨 Restored TRIPPED circuit breaker: {self.circuit_breaker.trip_reason}")

            logger.info("💾 Risk state loaded")
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed (circuit breaker not tripped).

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        if not ApexConfig.CIRCUIT_BREAKER_ENABLED:
            return True, "Circuit breaker disabled"

        if self.circuit_breaker.check_and_reset():
            return True, "OK"
        else:
            return False, f"Circuit breaker tripped: {self.circuit_breaker.trip_reason}"

    def record_trade_result(self, pnl: float):
        """Record trade result for circuit breaker tracking."""
        self.circuit_breaker.record_trade(pnl)

    def set_starting_capital(self, capital: float):
        """Set starting capital and initialize tracking."""
        # Only set overall starting capital if not already set (absolute system start)
        if getattr(self, 'starting_capital', 0) == 0:
            self.starting_capital = float(capital)
            logger.info(f"💰 Account starting capital initialized: ${self.starting_capital:,.2f}")
        
        # Only initialize peak if not already set
        if getattr(self, 'peak_capital', 0) == 0:
            self.peak_capital = float(capital)
            
        today = datetime.now().strftime('%Y-%m-%d')
        # CRITICAL: Only overwrite day_start_capital if it's 0 OR it's a new day
        if getattr(self, 'day_start_capital', 0) == 0 or self.current_day != today:
            self.day_start_capital = float(capital)
            self.current_day = today
            logger.info(f"💰 Day start capital initialized: ${self.day_start_capital:,.2f}")
        else:
            logger.info(f"💰 Using restored day start capital: ${self.day_start_capital:,.2f}")

        logger.info(f"💰 Active day start capital: ${self.day_start_capital:,.2f}")

    def check_daily_loss(self, current_value: float) -> Dict:
        """
        Check if daily loss limit breached.

        Returns:
            Dict with daily_pnl, daily_return, breached
        """
        try:
            current_value = float(current_value)

            # Reset daily tracking if new day
            today = datetime.now().strftime('%Y-%m-%d')
            if today != self.current_day:
                self.current_day = today
                self.day_start_capital = current_value
                logger.info(f"📅 New trading day: {today}")
                logger.info(f"   Starting capital: ${current_value:,.2f}")

            daily_pnl = current_value - self.day_start_capital
            daily_return = daily_pnl / self.day_start_capital if self.day_start_capital > 0 else 0

            breached = daily_return < -self.max_daily_loss

            # Check circuit breaker trigger
            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if daily_return < -ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS:
                    self.circuit_breaker.trip(
                        f"Daily loss {daily_return*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 DAILY LOSS LIMIT BREACHED!")
                logger.error(f"   Loss: ${daily_pnl:,.2f} ({daily_return*100:.2f}%)")
                logger.error(f"   Limit: {self.max_daily_loss*100:.1f}%")

            return {
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'breached': breached,
                'limit': self.max_daily_loss,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return {'daily_pnl': 0, 'daily_return': 0, 'breached': False, 'limit': self.max_daily_loss}

    def check_drawdown(self, current_value: float) -> Dict:
        """
        Check if drawdown limit breached.

        Returns:
            Dict with drawdown, breached, peak
        """
        try:
            current_value = float(current_value)

            # Update peak
            if current_value > self.peak_capital:
                self.peak_capital = current_value
                logger.debug(f"🎯 New equity peak: ${current_value:,.2f}")

            drawdown = (current_value - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
            breached = drawdown < -self.max_drawdown

            # Check circuit breaker trigger
            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if drawdown < -ApexConfig.CIRCUIT_BREAKER_DRAWDOWN:
                    self.circuit_breaker.trip(
                        f"Drawdown {abs(drawdown)*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DRAWDOWN*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 MAX DRAWDOWN BREACHED!")
                logger.error(f"   Drawdown: {drawdown*100:.2f}%")
                logger.error(f"   Peak: ${self.peak_capital:,.2f}")
                logger.error(f"   Current: ${current_value:,.2f}")
                logger.error(f"   Limit: {self.max_drawdown*100:.1f}%")

            return {
                'drawdown': abs(drawdown),
                'breached': breached,
                'peak': self.peak_capital,
                'current': current_value,
                'limit': self.max_drawdown,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return {'drawdown': 0, 'breached': False, 'peak': self.peak_capital, 'current': current_value, 'limit': self.max_drawdown}

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        max_position_value: float,
        max_shares: int = 200
    ) -> int:
        """
        Calculate position size with limits.

        Args:
            capital: Available capital
            price: Current stock price
            max_position_value: Max dollar value per position
            max_shares: Max shares per position

        Returns:
            Number of shares to buy
        """
        try:
            # Calculate based on dollar value
            shares_by_value = int(max_position_value / price)

            # Apply max shares limit
            shares = min(shares_by_value, max_shares)

            # Ensure at least 1 share
            if shares < 1:
                return 0

            # Check if we have enough capital
            total_cost = shares * price
            if total_cost > capital * 0.9:  # Don't use more than 90% of capital per trade
                shares = int((capital * 0.9) / price)

            return max(1, shares)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def get_circuit_breaker_status(self) -> Dict:
        """Get circuit breaker status for dashboard."""
        return self.circuit_breaker.get_status()
