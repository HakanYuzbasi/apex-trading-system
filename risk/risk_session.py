"""
risk/risk_session.py - User-Specific Risk State

Encapsulates the risk state for a single user/session, including:
- Capital tracking (starting, peak, day_start)
- Circuit breaker state
- Daily loss and drawdown logic
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from pathlib import Path

from config import ApexConfig

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit breaker to halt trading during adverse conditions.
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


class RiskSession:
    """
    Manages risk state for a specific user session.
    """

    def __init__(self, user_id: str, max_daily_loss: float = 0.02, max_drawdown: float = 0.10):
        self.user_id = user_id
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        self.starting_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.day_start_capital: float = 0.0
        self.current_day: str = datetime.now().strftime('%Y-%m-%d')

        self.circuit_breaker = CircuitBreaker()

        # Crypto uses a rolling 24h loss window (not calendar-day) because crypto trades 24/7
        self._crypto_24h_ref_capital: float = 0.0
        self._crypto_24h_ref_time: datetime = datetime.utcnow()

        # Load state on init
        self.load_state()

    def heal_baselines(self, current_capital: float, source: str = "runtime") -> bool:
        """Self-heal invalid baseline state."""
        try:
            value = float(current_capital)
        except Exception:
            return False

        if value <= 0:
            return False

        changed = False
        today = datetime.now().strftime('%Y-%m-%d')

        if self.starting_capital <= 0:
            self.starting_capital = value
            changed = True

        if self.peak_capital <= 0:
            self.peak_capital = value
            changed = True

        if self.current_day != today:
            self.current_day = today

        if self.day_start_capital <= 0:
            self.day_start_capital = value
            changed = True
        else:
            # Sanity check: if persisted day_start_capital is more than 60% away from
            # current capital it means it was saved under a completely different equity
            # level (e.g., a paper-startup value of $8k vs real $100k) — reset it.
            ratio = self.day_start_capital / value
            if ratio < 0.40 or ratio > 2.50:
                logger.warning(
                    f"[RiskSession/{self.user_id}] day_start_capital {self.day_start_capital:,.0f} "
                    f"is implausible vs current {value:,.0f} (ratio={ratio:.2f}). "
                    f"Resetting to current value to prevent false circuit trip."
                )
                self.day_start_capital = value
                changed = True

        return changed

    def _get_state_file(self) -> Path:
        """Get path to the risk state file for this session."""
        if self.user_id == "default":
             return ApexConfig.DATA_DIR / "risk_state.json"
        
        # Ensure user directory exists
        user_dir = ApexConfig.DATA_DIR / "users" / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "risk_state.json"

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
            
            state_file = self._get_state_file()
            # Ensure parent dir exists (redundant for default but good for users)
            state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving risk state for {self.user_id}: {e}")

    def load_state(self):
        """Load risk state from disk."""
        try:
            state_file = self._get_state_file()
            if not state_file.exists():
                return
            
            with open(state_file, "r") as f:
                state = json.load(f)
            
            today = datetime.now().strftime('%Y-%m-%d')
            if state.get('current_day') == today:
                self.day_start_capital = float(state.get('day_start_capital', 0))
                self.current_day = today
            else:
                pass # New day logic happens in heal or check methods
            
            self.peak_capital = float(state.get('peak_capital', self.peak_capital))
            self.starting_capital = float(state.get('starting_capital', self.starting_capital))
            
            cb_state = state.get('circuit_breaker', {})
            if cb_state.get('is_tripped'):
                self.circuit_breaker.is_tripped = True
                self.circuit_breaker.trip_reason = cb_state.get('reason')
                self.circuit_breaker.trip_time = datetime.fromisoformat(cb_state['trip_time']) if cb_state.get('trip_time') else None

        except Exception as e:
            logger.error(f"Error loading risk state for {self.user_id}: {e}")

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if not ApexConfig.CIRCUIT_BREAKER_ENABLED:
            return True, "Circuit breaker disabled"

        if self.circuit_breaker.check_and_reset():
            return True, "OK"
        else:
            return False, f"Circuit breaker tripped: {self.circuit_breaker.trip_reason}"

    def record_trade_result(self, pnl: float):
        """Record trade result."""
        self.circuit_breaker.record_trade(pnl)
        self.save_state()

    def manual_reset_circuit_breaker(self, requested_by: str = "admin", reason: str = "manual_reset") -> bool:
        """Manually reset the circuit breaker."""
        logger.info(f"🔄 Manual circuit breaker reset requested by {requested_by} for {self.user_id}. Reason: {reason}")
        self.circuit_breaker.reset()
        self.save_state()
        return True

    def set_starting_capital(self, capital: float):
        """Set starting capital and initialize tracking."""
        try:
            capital = float(capital)
        except Exception:
            return
        if capital <= 0:
            return

        self.starting_capital = capital
        
        if self.peak_capital == 0:
            self.peak_capital = capital
            
        today = datetime.now().strftime('%Y-%m-%d')
        if self.day_start_capital == 0 or self.current_day != today:
            self.day_start_capital = capital
            self.current_day = today
        
        self.save_state()

    def check_daily_loss(self, current_value: float) -> Dict:
        """Check if daily loss limit breached."""
        try:
            current_value = float(current_value)
            self.heal_baselines(current_capital=current_value, source="check_daily_loss")

            today = datetime.now().strftime('%Y-%m-%d')
            if today != self.current_day:
                self.current_day = today
                self.day_start_capital = current_value

            daily_pnl = current_value - self.day_start_capital
            daily_return = daily_pnl / self.day_start_capital if self.day_start_capital > 0 else 0

            # Guard: if daily_return is implausibly extreme (>50% loss or gain in one day)
            # it means day_start_capital is stale/corrupt — reset and skip the check.
            if abs(daily_return) > 0.50:
                logger.warning(
                    f"[RiskSession/{self.user_id}] Implausible daily_return {daily_return:.2%} "
                    f"(start={self.day_start_capital:,.0f}, current={current_value:,.0f}). "
                    f"Treating as stale baseline — resetting day_start_capital."
                )
                self.day_start_capital = current_value
                self.save_state()
                return {
                    'daily_pnl': 0.0,
                    'daily_return': 0.0,
                    'breached': False,
                    'limit': self.max_daily_loss,
                    'circuit_breaker_tripped': self.circuit_breaker.is_tripped
                }

            breached = daily_return < -self.max_daily_loss

            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if daily_return < -ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS:
                    self.circuit_breaker.trip(
                        f"Daily loss {daily_return*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 DAILY LOSS LIMIT BREACHED for {self.user_id}!")

            self.save_state()

            return {
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'breached': breached,
                'limit': self.max_daily_loss,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking daily loss for {self.user_id}: {e}")
            return {'daily_pnl': 0, 'daily_return': 0, 'breached': False, 'limit': self.max_daily_loss}

    def check_crypto_rolling_loss(self, current_value: float) -> Dict:
        """
        Check crypto P&L against a rolling 24-hour window.
        Unlike equities, crypto trades 24/7 so a calendar-day reset at midnight is meaningless.
        The reference capital resets every 24h from the first time this is called.
        """
        try:
            current_value = float(current_value)
            now = datetime.utcnow()
            limit = getattr(ApexConfig, "CRYPTO_MAX_DAILY_LOSS", 0.05)

            # Initialise reference on first call or after 24h window expires
            if self._crypto_24h_ref_capital <= 0:
                self._crypto_24h_ref_capital = current_value
                self._crypto_24h_ref_time = now

            elapsed_hours = (now - self._crypto_24h_ref_time).total_seconds() / 3600
            if elapsed_hours >= 24:
                logger.info(
                    "[RiskSession/%s] Crypto 24h window expired (%.1fh); resetting reference capital.",
                    self.user_id, elapsed_hours,
                )
                self._crypto_24h_ref_capital = current_value
                self._crypto_24h_ref_time = now

            pnl_24h = current_value - self._crypto_24h_ref_capital
            ret_24h = pnl_24h / self._crypto_24h_ref_capital if self._crypto_24h_ref_capital > 0 else 0.0
            breached = ret_24h < -limit

            if breached:
                logger.error(
                    "🚨 CRYPTO 24H LOSS LIMIT BREACHED for %s: %.2f%% (limit %.1f%%)",
                    self.user_id, ret_24h * 100, limit * 100,
                )

            return {
                "crypto_pnl_24h": pnl_24h,
                "crypto_return_24h": ret_24h,
                "breached": breached,
                "limit": limit,
                "window_start": self._crypto_24h_ref_time.isoformat(),
                "hours_elapsed": round(elapsed_hours, 2),
            }
        except Exception as e:
            logger.error(f"Error checking crypto rolling loss for {self.user_id}: {e}")
            return {"crypto_pnl_24h": 0.0, "crypto_return_24h": 0.0, "breached": False,
                    "limit": getattr(ApexConfig, "CRYPTO_MAX_DAILY_LOSS", 0.05)}

    def check_drawdown(self, current_value: float) -> Dict:
        """Check if drawdown limit breached."""
        try:
            current_value = float(current_value)
            self.heal_baselines(current_capital=current_value, source="check_drawdown")

            if current_value > self.peak_capital:
                self.peak_capital = current_value

            drawdown = (current_value - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
            breached = drawdown < -self.max_drawdown

            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if drawdown < -ApexConfig.CIRCUIT_BREAKER_DRAWDOWN:
                    self.circuit_breaker.trip(
                        f"Drawdown {abs(drawdown)*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DRAWDOWN*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 MAX DRAWDOWN BREACHED for {self.user_id}!")

            self.save_state()

            return {
                'drawdown': abs(drawdown),
                'breached': breached,
                'peak': self.peak_capital,
                'current': current_value,
                'limit': self.max_drawdown,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking drawdown for {self.user_id}: {e}")
            return {'drawdown': 0, 'breached': False, 'peak': self.peak_capital, 'current': current_value, 'limit': self.max_drawdown}
