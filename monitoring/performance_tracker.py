"""
monitoring/performance_tracker.py
FIXED: Proper equity curve tracking with float conversion
"""

import logging
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track trading performance metrics."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[str, float]] = []  # ✅ Always store as float
        self.starting_capital: float = 0.0
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float, commission: float = 0.0):
        """Record a trade with commission."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': float(price),  # ✅ Force float
            'commission': float(commission),  # ✅ Force float
            'pnl': 0.0  # Calculated on exit
        }
        
        self.trades.append(trade)
        logger.debug(f"Trade recorded: {side} {quantity} {symbol} @ ${price:.2f}")
    
    def record_equity(self, value: float):
        """✅ FIXED: Record equity point with proper float conversion."""
        try:
            value = float(value)  # ✅ Force conversion
            timestamp = datetime.now().isoformat()
            self.equity_curve.append((timestamp, value))
            logger.debug(f"Equity recorded: ${value:,.2f}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid equity value: {value} ({type(value)}): {e}")
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        try:
            # Extract values and ensure float
            values = [float(v) for _, v in self.equity_curve]
            
            # Calculate returns
            returns = np.diff(values) / values[:-1]
            
            if len(returns) == 0:
                return 0.0
            
            # Annualize (assuming daily data)
            excess_returns = returns - (risk_free_rate / 252)
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return float(sharpe)
        
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def get_win_rate(self) -> float:
        """Calculate win rate from completed trades."""
        if len(self.trades) < 2:
            return 0.0
        
        try:
            # Match buys and sells
            positions = {}
            completed_trades = []
            
            for trade in self.trades:
                symbol = trade['symbol']
                side = trade['side']
                qty = trade['quantity']
                price = trade['price']
                commission = trade.get('commission', 0)
                
                if side == 'BUY':
                    if symbol not in positions:
                        positions[symbol] = []
                    positions[symbol].append({
                        'qty': qty,
                        'price': price,
                        'commission': commission
                    })
                
                elif side == 'SELL':
                    if symbol in positions and len(positions[symbol]) > 0:
                        entry = positions[symbol].pop(0)
                        pnl = (price - entry['price']) * qty - entry['commission'] - commission
                        completed_trades.append({'pnl': pnl})
            
            if len(completed_trades) == 0:
                return 0.0
            
            winners = sum(1 for t in completed_trades if t['pnl'] > 0)
            win_rate = winners / len(completed_trades)
            
            return win_rate
        
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        try:
            values = [float(v) for _, v in self.equity_curve]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                dd = (value - peak) / peak
                if dd < max_dd:
                    max_dd = dd
            
            return abs(max_dd)
        
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def print_summary(self):
        """Print performance summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Total Trades: {len(self.trades)}")
        
        if len(self.trades) > 0:
            logger.info(f"Win Rate: {self.get_win_rate()*100:.1f}%")
        
        if len(self.equity_curve) > 1:
            start_value = float(self.equity_curve[0][1])
            end_value = float(self.equity_curve[-1][1])
            total_return = (end_value / start_value - 1) * 100
            
            logger.info(f"Starting Capital: ${start_value:,.2f}")
            logger.info(f"Ending Capital: ${end_value:,.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info(f"Sharpe Ratio: {self.get_sharpe_ratio():.2f}")
            logger.info(f"Max Drawdown: {self.get_max_drawdown()*100:.2f}%")
        
        logger.info("=" * 80)
