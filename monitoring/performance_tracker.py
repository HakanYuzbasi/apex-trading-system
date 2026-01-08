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

    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (focuses on downside volatility only)."""
        if len(self.equity_curve) < 10:
            return 0.0

        try:
            values = [float(v) for _, v in self.equity_curve]
            returns = np.diff(values) / values[:-1]

            if len(returns) == 0:
                return 0.0

            # Calculate downside returns only
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return 0.0

            downside_std = np.std(downside_returns)
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
            return float(sortino)

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def get_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(self.equity_curve) < 20:
            return 0.0

        try:
            values = [float(v) for _, v in self.equity_curve]
            total_return = (values[-1] / values[0]) - 1

            # Annualize (assuming ~252 trading days)
            days = len(values)
            annual_return = total_return * (252 / max(days, 1))

            max_dd = self.get_max_drawdown()
            if max_dd == 0:
                return 0.0

            return float(annual_return / max_dd)

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.trades) < 2:
            return 0.0

        try:
            gross_profit = 0.0
            gross_loss = 0.0

            positions = {}

            for trade in self.trades:
                symbol = trade['symbol']
                side = trade['side']
                qty = trade['quantity']
                price = trade['price']
                commission = trade.get('commission', 0)

                if side == 'BUY':
                    if symbol not in positions:
                        positions[symbol] = []
                    positions[symbol].append({'qty': qty, 'price': price, 'commission': commission})

                elif side == 'SELL':
                    if symbol in positions and len(positions[symbol]) > 0:
                        entry = positions[symbol].pop(0)
                        pnl = (price - entry['price']) * qty - entry['commission'] - commission

                        if pnl > 0:
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)

            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0

            return float(gross_profit / gross_loss)

        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0

    def get_avg_trade_pnl(self) -> Tuple[float, float]:
        """Calculate average winning and losing trade P&L."""
        if len(self.trades) < 2:
            return (0.0, 0.0)

        try:
            winners = []
            losers = []
            positions = {}

            for trade in self.trades:
                symbol = trade['symbol']
                side = trade['side']
                qty = trade['quantity']
                price = trade['price']
                commission = trade.get('commission', 0)

                if side == 'BUY':
                    if symbol not in positions:
                        positions[symbol] = []
                    positions[symbol].append({'qty': qty, 'price': price, 'commission': commission})

                elif side == 'SELL':
                    if symbol in positions and len(positions[symbol]) > 0:
                        entry = positions[symbol].pop(0)
                        pnl = (price - entry['price']) * qty - entry['commission'] - commission

                        if pnl > 0:
                            winners.append(pnl)
                        else:
                            losers.append(pnl)

            avg_win = np.mean(winners) if winners else 0.0
            avg_loss = np.mean(losers) if losers else 0.0

            return (float(avg_win), float(avg_loss))

        except Exception as e:
            logger.error(f"Error calculating avg trade P&L: {e}")
            return (0.0, 0.0)
    
    def print_summary(self):
        """Print performance summary with full quant metrics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Total Trades: {len(self.trades)}")

        if len(self.trades) > 0:
            logger.info(f"Win Rate: {self.get_win_rate()*100:.1f}%")
            logger.info(f"Profit Factor: {self.get_profit_factor():.2f}")
            avg_win, avg_loss = self.get_avg_trade_pnl()
            logger.info(f"Avg Win: ${avg_win:+,.2f} | Avg Loss: ${avg_loss:+,.2f}")

        if len(self.equity_curve) > 1:
            start_value = float(self.equity_curve[0][1])
            end_value = float(self.equity_curve[-1][1])
            total_return = (end_value / start_value - 1) * 100

            logger.info(f"Starting Capital: ${start_value:,.2f}")
            logger.info(f"Ending Capital: ${end_value:,.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info("─" * 40)
            logger.info("📈 RISK-ADJUSTED METRICS:")
            logger.info(f"   Sharpe Ratio: {self.get_sharpe_ratio():.2f}")
            logger.info(f"   Sortino Ratio: {self.get_sortino_ratio():.2f}")
            logger.info(f"   Calmar Ratio: {self.get_calmar_ratio():.2f}")
            logger.info(f"   Max Drawdown: {self.get_max_drawdown()*100:.2f}%")

        logger.info("=" * 80)
