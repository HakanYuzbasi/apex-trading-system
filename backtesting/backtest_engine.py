"""
backtesting/backtest_engine.py - Event-Driven Backtesting Engine

Simulates:
- Market data events (bar-by-bar)
- Order execution (latency, slippage, commission)
- Portfolio tracking
- Signal generation replay

Supports:
- Transaction costs
- Slippage limits
- Partial fills (simulated)
- Monte Carlo Simulation
- Advanced Risk Metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a simulated trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    pnl: float = 0.0
    entry_id: Optional[str] = None
    
    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Position:
    """Simulated position."""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    max_price: float  # For trailing stop logic
    
    def update_price(self, price: float):
        self.current_price = price
        if self.quantity > 0:
            self.max_price = max(self.max_price, price)
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.max_price = min(self.max_price, price)
            self.unrealized_pnl = (self.avg_entry_price - price) * abs(self.quantity)


class BacktestEngine:
    """
    Event-driven backtesting engine with Monte Carlo capability.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_pct = slippage_bps / 10000.0
        
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_time: datetime = datetime.min
        
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[Dict] = []
        
        self.strategy: Optional[Callable] = None
        
        logger.info("BacktestEngine initialized")

    def load_data(self, data: Dict[str, pd.DataFrame]):
        """Load historical data."""
        self.data = data
        # Ensure data is sorted
        for symbol in self.data:
            self.data[symbol] = self.data[symbol].sort_index()

    def run(self, strategy_func: Callable, start_date: datetime, end_date: datetime):
        """
        Run backtest.
        
        Args:
            strategy_func: Function(engine, symbol, data) -> None
            start_date: Start datetime
            end_date: End datetime
        """
        self.strategy = strategy_func
        
        # Collect all timestamps
        timestamps = set()
        for df in self.data.values():
            timestamps.update(df.index)
        
        sorted_timestamps = sorted([t for t in timestamps if start_date <= t <= end_date])
        
        logger.info(f"Running backtest from {start_date} to {end_date} (steps: {len(sorted_timestamps)})")
        
        for timestamp in sorted_timestamps:
            self.current_time = timestamp
            self._process_step(timestamp)
            
        return self.get_results()
    
    def _process_step(self, timestamp: datetime):
        """Process a single time step."""
        # 1. Update prices & equity
        for symbol, pos in self.positions.items():
            if symbol in self.data and timestamp in self.data[symbol].index:
                price = self.data[symbol].loc[timestamp]['Close']
                pos.update_price(price)
        
        # 2. Run strategy
        if self.strategy:
            self.strategy(self, timestamp)
            
        # 3. Record history
        self.history.append({
            'timestamp': timestamp,
            'equity': self.total_equity(),
            'cash': self.cash,
            'positions': len(self.positions),
            'drawdown': 0.0 # Placeholder
        })
    
    def total_equity(self) -> float:
        """Calculate total equity."""
        pos_value = sum(
            p.quantity * p.current_price if p.quantity > 0 
            else p.quantity * p.current_price + (p.avg_entry_price - p.current_price) * abs(p.quantity)
            for p in self.positions.values()
        )
        
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized 
    
    def execute_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None):
        """Execute simulation order."""
        if quantity <= 0:
            return
            
        # Get price if not provided
        if price is None:
            if symbol in self.data and self.current_time in self.data[symbol].index:
                raw_price = self.data[symbol].loc[self.current_time]['Close']
            else:
                return # No price
        else:
            raw_price = price
            
        # Apply slippage
        if side == 'BUY':
            fill_price = raw_price * (1 + self.slippage_pct)
        else:
            fill_price = raw_price * (1 - self.slippage_pct)
            
        # Calculate commission
        commission = max(self.min_commission, quantity * self.commission_per_share)
        cost = quantity * fill_price
        
        # Capture P&L for closing trades
        pnl = 0.0
        
        # Update accounting
        if side == 'BUY':
            self.cash -= (cost + commission)
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                
                # Increasing long
                if pos.quantity > 0:
                    total_cost = pos.quantity * pos.avg_entry_price + cost
                    pos.avg_entry_price = total_cost / (pos.quantity + quantity)
                
                # Closing short
                elif pos.quantity < 0:
                    covered = min(quantity, abs(pos.quantity))
                    trade_pnl = (pos.avg_entry_price - fill_price) * covered - (commission * (covered/quantity))
                    pos.realized_pnl += trade_pnl
                    pnl = trade_pnl
                    
                    if quantity > abs(pos.quantity):
                        # Flip to long
                        pos.avg_entry_price = fill_price
                        pos.max_price = fill_price
                
                pos.quantity += quantity
                if pos.quantity == 0:
                    del self.positions[symbol]
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    max_price=fill_price
                )
                
        elif side == 'SELL':
            self.cash += (cost - commission)
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                
                # Closing long
                if pos.quantity > 0:
                    sold = min(quantity, pos.quantity)
                    trade_pnl = (fill_price - pos.avg_entry_price) * sold - (commission * (sold/quantity))
                    pos.realized_pnl += trade_pnl
                    pnl = trade_pnl
                    
                    if quantity > pos.quantity:
                        # Flip to short
                        pos.avg_entry_price = fill_price
                        pos.max_price = fill_price
                
                # Increasing short
                elif pos.quantity < 0:
                    total_val = abs(pos.quantity) * pos.avg_entry_price + cost
                    pos.avg_entry_price = total_val / (abs(pos.quantity) + quantity)

                pos.quantity -= quantity
                if pos.quantity == 0:
                    del self.positions[symbol]
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    max_price=fill_price
                )
                
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            timestamp=self.current_time,
            pnl=pnl
        )
        self.trades.append(trade)
    
    def get_results(self) -> Dict:
        """Calculate advanced backtest metrics."""
        if not self.history:
            return {}
            
        df = pd.DataFrame(self.history).set_index('timestamp')
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # Basic Metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        volatility = df['returns'].std() * np.sqrt(252)
        sharpe = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252) if df['returns'].std() > 0 else 0
        
        # Drawdown Stats
        peaks = df['equity'].cummax()
        drawdown = (df['equity'] / peaks) - 1
        max_drawdown = drawdown.min()
        max_dd_duration = (drawdown < 0).astype(int).groupby(drawdown.eq(0).cumsum()).cumsum().max()
        
        # Advanced Metrics
        sortino = 0
        downside_returns = df['returns'][df['returns'] < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (df['returns'].mean() / downside_returns.std()) * np.sqrt(252)
            
        calmar = 0
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)  # Annualized usually, simplistic here
            
        # Trade Analysis
        win_rate = 0
        profit_factor = 0
        avg_trade = 0
        
        closed_trades = [t for t in self.trades if t.pnl != 0]
        if closed_trades:
            winners = [t for t in closed_trades if t.pnl > 0]
            losers = [t for t in closed_trades if t.pnl <= 0]
            
            win_rate = len(winners) / len(closed_trades)
            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_trade = sum(t.pnl for t in closed_trades) / len(closed_trades)
        
        metrics = {
            'total_return': total_return,
            'cagr': ((1 + total_return) ** (252/len(df))) - 1 if len(df) > 200 else total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_equity': df['equity'].iloc[-1],
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'history': df
        }
        
        return metrics

    def run_monte_carlo(self, n_sims: int = 1000) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on the equity curve returns.
        
        Args:
            n_sims: Number of simulations
            
        Returns:
            Dict of risk metrics (VaR, CVaR at various confidence levels)
        """
        if not self.history:
            return {}
            
        df = pd.DataFrame(self.history)
        returns = df['equity'].pct_change().dropna().values
        
        if len(returns) < 10:
            return {}
            
        final_values = []
        days = len(returns)
        start_equity = df['equity'].iloc[-1]
        
        # Bootstrap resampling
        for _ in range(n_sims):
            sim_returns = np.random.choice(returns, size=days, replace=True)
            # Apply path
            cum_returns = np.cumprod(1 + sim_returns)
            final_values.append(start_equity * cum_returns[-1])
            
        final_values = np.array(final_values)
        
        # Calculate VaR of the simulation distribution
        # Note: This is VaR of the FINAL EQUITY, essentially worst case scenarios for future path
        sorted_outcomes = np.sort(final_values)
        
        return {
            'mc_min_equity': sorted_outcomes[0],
            'mc_median_equity': np.median(sorted_outcomes),
            'mc_95_pct_equity': sorted_outcomes[int(n_sims * 0.05)],  # 5th percentile worst case
            'mc_99_pct_equity': sorted_outcomes[int(n_sims * 0.01)],  # 1st percentile worst case
            'mc_max_equity': sorted_outcomes[-1]
        }
