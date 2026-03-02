"""
risk/risk_manager.py - Enhanced Risk Management (Multi-Session Refactor)

Provides comprehensive risk controls including:
- Daily loss limits with automatic enforcement
- Drawdown monitoring and alerts
- Circuit breaker for automatic trading halts
- Position sizing with configurable limits

Refactored to support Multi-User/Multi-Broker sessions while maintaining
backward compatibility for the singleton "system" account.
"""

import asyncio
import logging
from typing import Dict, Tuple
import pandas as pd

from risk.risk_session import RiskSession, CircuitBreaker

# Services
from services.broker.service import broker_service

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manage risk limits, position sizing, and circuit breakers for multiple users.
    
    Acts as a facade for RiskSession objects.
    Methods called without user_id default to the 'system'/'default' session
    to maintain backward compatibility with the existing trading engine.
    """

    def __init__(self, max_daily_loss: float = 0.02, max_drawdown: float = 0.10, user_id: str = "default"):
        """
        Initialize risk manager.

        Args:
            max_daily_loss: Max daily loss as fraction of capital (e.g., 0.02 = 2%)
            max_drawdown: Max drawdown from peak as fraction (e.g., 0.10 = 10%)
            user_id: The tenant ID this manager represents by default.
        """
        self.default_user_id = user_id
        self.default_max_daily_loss = max_daily_loss
        self.default_max_drawdown = max_drawdown
        
        # Session storage: user_id -> RiskSession
        self.sessions: Dict[str, RiskSession] = {}
        
        # Initialize default session
        self._get_or_create_session(self.default_user_id)

        logger.info(f"🛡️  Risk Manager initialized (Multi-Session Mode, default={self.default_user_id})")

    def _get_or_create_session(self, user_id: str) -> RiskSession:
        """Get existing session or create a new one."""
        if user_id not in self.sessions:
            session = RiskSession(
                user_id=user_id,
                max_daily_loss=self.default_max_daily_loss,
                max_drawdown=self.default_max_drawdown
            )
            self.sessions[user_id] = session
            logger.info(f"🛡️  Created risk session for user: {user_id}")
        return self.sessions[user_id]
        
    # ----------------------------------------------------------------
    # Backward Compatibility / Default Session Proxies
    # ----------------------------------------------------------------

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Access default session's circuit breaker."""
        return self.sessions[self.default_user_id].circuit_breaker

    @property
    def starting_capital(self) -> float:
        return self.sessions[self.default_user_id].starting_capital
    
    @starting_capital.setter
    def starting_capital(self, value: float):
        self.sessions[self.default_user_id].set_starting_capital(value)

    @property
    def peak_capital(self) -> float:
        return self.sessions[self.default_user_id].peak_capital

    @peak_capital.setter
    def peak_capital(self, value: float):
        self.sessions[self.default_user_id].peak_capital = value

    @property
    def day_start_capital(self) -> float:
        return self.sessions[self.default_user_id].day_start_capital
    
    @day_start_capital.setter
    def day_start_capital(self, value: float):
        # We allow direct setting for backward compat, but ideally should use set_starting_capital
        self.sessions[self.default_user_id].day_start_capital = value

    @property
    def current_day(self) -> str:
        return self.sessions[self.default_user_id].current_day

    def heal_baselines(self, current_capital: float, source: str = "runtime") -> bool:
        return self.sessions[self.default_user_id].heal_baselines(current_capital, source)

    def save_state(self):
        self.sessions[self.default_user_id].save_state()

    async def save_state_async(self, user_id: str = None):
        await self._get_or_create_session(user_id or self.default_user_id).save_state_async()

    def load_state(self):
        self.sessions[self.default_user_id].load_state()

    def can_trade(self, user_id: str = None) -> Tuple[bool, str]:
        return self._get_or_create_session(user_id or self.default_user_id).can_trade()

    def record_trade_result(self, pnl: float, user_id: str = None):
        self._get_or_create_session(user_id or self.default_user_id).record_trade_result(pnl)

    def set_starting_capital(self, capital: float, user_id: str = None):
         self._get_or_create_session(user_id or self.default_user_id).set_starting_capital(capital)

    def check_daily_loss(self, current_value: float, user_id: str = None) -> Dict:
        return self._get_or_create_session(user_id or self.default_user_id).check_daily_loss(current_value)

    def check_crypto_rolling_loss(self, current_value: float, user_id: str = None) -> Dict:
        """Check crypto P&L over a rolling 24h window (not calendar-day reset)."""
        return self._get_or_create_session(user_id or self.default_user_id).check_crypto_rolling_loss(current_value)

    def check_drawdown(self, current_value: float, user_id: str = None) -> Dict:
        return self._get_or_create_session(user_id or self.default_user_id).check_drawdown(current_value)

    def get_circuit_breaker_status(self, user_id: str = None) -> Dict:
        return self._get_or_create_session(user_id or self.default_user_id).circuit_breaker.get_status()

    def manual_reset_circuit_breaker(self, requested_by: str = "admin", reason: str = "manual_reset", user_id: str = None) -> bool:
        """Manually reset the circuit breaker for a user."""
        return self._get_or_create_session(user_id or self.default_user_id).manual_reset_circuit_breaker(requested_by, reason)

    # ----------------------------------------------------------------
    # Shared Logic (Stateless)
    # ----------------------------------------------------------------

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        max_position_value: float,
        max_shares: int = 200
    ) -> int:
        """
        Calculate position size with limits. Stateless logic, shared across users.
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

    def analyze_risk_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate advanced risk metrics. Stateless.
        """
        try:
            from risk.advanced_metrics import AdvancedRiskMetrics
            metrics = AdvancedRiskMetrics()
            
            return {
                'sortino_ratio': metrics.calculate_sortino_ratio(returns),
                'cvar_95': metrics.calculate_cvar(returns, 0.95),
                'max_dd_duration': metrics.calculate_max_drawdown_duration(returns),
                'omega_ratio': metrics.calculate_omega_ratio(returns),
                'tail_ratio': metrics.calculate_tail_ratio(returns)
            }
        except ImportError:
            # logger.warning("AdvancedRiskMetrics not available")
            return {}
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return {}

    # ----------------------------------------------------------------
    # Multi-Broker Aggregated Risk
    # ----------------------------------------------------------------

    async def check_aggregate_risk(self, user_id: str, proposed_trade_amount: float = 0.0) -> Dict:
        """
        Check risk against the user's AGGREGATED equity across all brokers.
        This is a pre-trade check.

        Args:
            user_id: The user to check
            proposed_trade_amount: Notional value of the proposed trade (for pre-trade impact check)
        
        Returns:
            Dict with 'allowed': bool, 'reason': str, 'metrics': Dict
        """
        try:
            # 1. Get Aggregated Equity
            total_equity = await broker_service.get_total_equity(user_id)
            
            # If we can't get equity (e.g. no brokers connected), we might default to 0 or handle error
            if total_equity <= 0:
                 # Check if default session has manual capital set?
                 session = self._get_or_create_session(user_id)
                 if session.starting_capital > 0:
                     total_equity = session.starting_capital # Fallback to manual config
                 else:
                     return {"allowed": False, "reason": "No equity available", "metrics": {}}

            # 2. Get Session (loads state like day_start_capital)
            session = self._get_or_create_session(user_id)
            
            # 3. Check Daily Loss (Preview)
            # We don't want to record a loss here, just check if we ARE currently in a loss state
            # that prevents trading.
            # But wait, check_daily_loss updates state. We likely want a 'preview' mode or just check the current state.
            
            # Actually, we should sync the session with the latest equity FIRST
            # This 'ticks' the risk manager state
            loss_status = session.check_daily_loss(total_equity) # This updates state!
            dd_status = session.check_drawdown(total_equity)     # This updates state!
            
            # 4. Check Circuit Breaker
            can_trade, reason = session.can_trade()
            if not can_trade:
                return {"allowed": False, "reason": reason, "metrics": {
                    "daily_loss": loss_status, "drawdown": dd_status
                }}
            
            # 5. Check if proposed trade puts us over limits (Pre-trade logic)
            # Simple check: do we have enough *buying power*? 
            # In a real system, we'd check margin. Here we check loose "cash" proxy or just raw equity caps.
            # For now, we trust the broker for margin, we check *risk limits*.
            
            if loss_status['breached']:
                 return {"allowed": False, "reason": "Daily loss limit breached", "metrics": loss_status}
                 
            if dd_status['breached']:
                 return {"allowed": False, "reason": "Max drawdown breached", "metrics": dd_status}

            return {"allowed": True, "reason": "OK", "metrics": {
                    "total_equity": total_equity,
                    "daily_loss": loss_status, 
                    "drawdown": dd_status
            }}

        except Exception as e:
            logger.error(f"Error checking aggregate risk for {user_id}: {e}")
            return {"allowed": False, "reason": f"Error: {e}", "metrics": {}}

