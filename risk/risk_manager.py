"""
risk/risk_manager.py
Enhanced risk management with better tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """Manage risk limits and position sizing."""
    
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
        
        logger.info(f"🛡️  Risk Manager initialized:")
        logger.info(f"   Max Daily Loss: {max_daily_loss*100:.1f}%")
        logger.info(f"   Max Drawdown: {max_drawdown*100:.1f}%")
    
    def set_starting_capital(self, capital: float):
        """Set starting capital and initialize tracking."""
        self.starting_capital = float(capital)
        self.peak_capital = float(capital)
        self.day_start_capital = float(capital)
        
        logger.info(f"💰 Starting capital set: ${capital:,.2f}")
    
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
            
            if breached:
                logger.error(f"🚨 DAILY LOSS LIMIT BREACHED!")
                logger.error(f"   Loss: ${daily_pnl:,.2f} ({daily_return*100:.2f}%)")
                logger.error(f"   Limit: {self.max_daily_loss*100:.1f}%")
            
            return {
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'breached': breached,
                'limit': self.max_daily_loss
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
                'limit': self.max_drawdown
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
