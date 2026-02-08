"""
backtesting/backtest_engine.py - Event-Driven Backtesting Engine

Simulates:
- Market data events (bar-by-bar)
- Order execution (latency, slippage, commission)
- Portfolio tracking
- Signal generation replay

Supports:
- Transaction costs
- Dynamic slippage (market impact model)
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
import inspect

from core.symbols import AssetClass, parse_symbol, is_market_open
from config import ApexConfig

# Import dynamic slippage model
try:
    from backtesting.market_impact import MarketImpactModel, MarketConditions
    MARKET_IMPACT_AVAILABLE = True
except ImportError:
    MARKET_IMPACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class _DataView:
    """Read-only time-sliced data view to prevent lookahead."""
    def __init__(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        self._data = data
        self._current_time = current_time

    def __getitem__(self, key: str) -> pd.DataFrame:
        df = self._data[key]
        return df.loc[: self._current_time]

    def get(self, key: str, default=None):
        if key not in self._data:
            return default
        return self.__getitem__(key)

    def items(self):
        for k, v in self._data.items():
            yield k, v.loc[: self._current_time]

    def keys(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data


@dataclass
class Trade:
    """Record of a simulated trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
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
    quantity: float
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

    Supports dynamic slippage modeling based on:
    - Order size relative to volume
    - Market volatility
    - Bid-ask spread estimation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0,
        use_dynamic_slippage: bool = True,
        fx_commission_bps: Optional[float] = None,
        crypto_commission_bps: Optional[float] = None,
        fx_min_commission: float = 0.0,
        crypto_min_commission: float = 0.0,
        fx_spread_bps: Optional[float] = None,
        crypto_spread_bps: Optional[float] = None
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.base_slippage_bps = slippage_bps
        self.slippage_pct = slippage_bps / 10000.0
        self.use_dynamic_slippage = use_dynamic_slippage and MARKET_IMPACT_AVAILABLE
        self.fx_commission_bps = fx_commission_bps if fx_commission_bps is not None else ApexConfig.FX_COMMISSION_BPS
        self.crypto_commission_bps = crypto_commission_bps if crypto_commission_bps is not None else ApexConfig.CRYPTO_COMMISSION_BPS
        self.fx_min_commission = fx_min_commission
        self.crypto_min_commission = crypto_min_commission
        self.fx_spread_bps = fx_spread_bps if fx_spread_bps is not None else ApexConfig.FX_SPREAD_BPS
        self.crypto_spread_bps = crypto_spread_bps if crypto_spread_bps is not None else ApexConfig.CRYPTO_SPREAD_BPS

        self.data: Dict[str, pd.DataFrame] = {}
        self.current_time: datetime = datetime.min

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[Dict] = []
        self.pending_orders: List[Dict[str, Any]] = []

        self.strategy: Optional[Callable] = None

        # Initialize market impact model for dynamic slippage
        if self.use_dynamic_slippage:
            self.market_impact_model = MarketImpactModel(
                base_spread_bps=slippage_bps,
                impact_multiplier=1.0,
                random_slippage_std=2.0
            )
            logger.info("BacktestEngine initialized with dynamic slippage model")
        else:
            self.market_impact_model = None
            logger.info("BacktestEngine initialized with fixed slippage")

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
        try:
            self._strategy_arity = len(inspect.signature(strategy_func).parameters)
        except Exception:
            self._strategy_arity = 2
        
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
        # 1. Execute any pending orders from previous step (t-1 -> t)
        if self.pending_orders:
            pending = self.pending_orders
            self.pending_orders = []
            for order in pending:
                self._execute_order_now(
                    order["symbol"],
                    order["side"],
                    order["quantity"],
                    order.get("price")
                )

        # 2. Update prices & equity
        for symbol, pos in self.positions.items():
            if symbol in self.data and timestamp in self.data[symbol].index:
                price = self.data[symbol].loc[timestamp]['Close']
                pos.update_price(price)
        
        # 3. Run strategy (signals at t, executes at t+1)
        if self.strategy:
            data_view = _DataView(self.data, timestamp)
            original_data = self.data
            self.data = data_view
            try:
                if self._strategy_arity >= 3:
                    self.strategy(self, timestamp, data_view)
                else:
                    self.strategy(self, timestamp)
            finally:
                self.data = original_data
            
        # 4. Record history
        self.history.append({
            'timestamp': timestamp,
            'equity': self.total_equity(),
            'cash': self.cash,
            'positions': len(self.positions),
            'drawdown': 0.0 # Placeholder
        })
    
    def total_equity(self) -> float:
        """Calculate total equity."""
        pos_value = sum(p.quantity * p.current_price for p in self.positions.values())
        return self.cash + pos_value

    def _annualization_factor(self) -> int:
        classes = set()
        for symbol in self.data.keys():
            try:
                classes.add(parse_symbol(symbol).asset_class)
            except ValueError:
                continue
        if classes == {AssetClass.CRYPTO}:
            return 365
        if classes == {AssetClass.FOREX}:
            return 260
        return 252

    def _get_slippage_pct(self, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.FOREX:
            return self.fx_spread_bps / 10000.0
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_spread_bps / 10000.0
        return self.slippage_pct

    def _calculate_dynamic_fill_price(
        self,
        symbol: str,
        raw_price: float,
        quantity: float,
        side: str
    ) -> float:
        """
        Calculate fill price using dynamic market impact model.

        Considers:
        - Order size relative to average volume
        - Recent volatility
        - Estimated bid-ask spread
        """
        if not self.market_impact_model:
            # Fallback to fixed slippage
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        # Get historical data for this symbol
        if symbol not in self.data:
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        df = self.data[symbol]

        # Get data up to current time
        hist = df[df.index <= self.current_time].tail(20)

        if len(hist) < 5:
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        # Calculate market conditions
        close = hist['Close'] if 'Close' in hist.columns else hist.get('close', pd.Series([raw_price]))
        volume = hist['Volume'] if 'Volume' in hist.columns else hist.get('volume', pd.Series([1000000]))
        high = hist['High'] if 'High' in hist.columns else hist.get('high', close)
        low = hist['Low'] if 'Low' in hist.columns else hist.get('low', close)

        # Average daily volume
        avg_volume = volume.mean() if len(volume) > 0 else 1000000

        # Daily volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.02

        # Estimate spread from high-low range
        if len(close) > 0 and close.iloc[-1] > 0:
            avg_range = ((high - low) / close).mean()
            spread_bps = max(avg_range * 10000 / 4, self.base_slippage_bps)
        else:
            spread_bps = self.base_slippage_bps

        # Create market conditions
        conditions = MarketConditions(
            avg_daily_volume=avg_volume,
            avg_daily_turnover=avg_volume * raw_price,
            volatility=volatility,
            bid_ask_spread_bps=spread_bps,
            current_volume_ratio=1.0,
            time_of_day=self.current_time.time() if hasattr(self.current_time, 'time') else None
        )

        # Calculate execution costs
        costs = self.market_impact_model.calculate_execution_costs(
            order_size_shares=quantity,
            price=raw_price,
            side=side,
            conditions=conditions
        )

        return costs.effective_price

    def execute_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        """Queue simulation order for next bar execution (t+1)."""
        self.pending_orders.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "submitted_at": self.current_time
        })
        logger.info("event=order_queued symbol=%s side=%s qty=%s submitted_at=%s", symbol, side, quantity, self.current_time)

    def _execute_order_now(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        """Execute simulation order immediately with realistic slippage."""
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            logger.warning("event=order_rejected symbol=%s reason=invalid_symbol", symbol)
            return
        logger.info(
            "event=symbol_normalization input=%s normalized=%s broker=%s",
            symbol,
            parsed.normalized,
            parsed.normalized,
        )

        if quantity <= 0:
            logger.warning("event=order_rejected symbol=%s reason=non_positive_quantity quantity=%s", symbol, quantity)
            return

        # Market hours gating (24/7 crypto, 24/5 FX, equity hours)
        if not is_market_open(parsed, self.current_time, assume_daily=True):
            logger.warning("event=order_rejected symbol=%s reason=market_closed", parsed.normalized)
            return

        if parsed.asset_class == AssetClass.EQUITY and isinstance(quantity, float) and not quantity.is_integer():
            logger.warning("event=order_rejected symbol=%s reason=fractional_equity quantity=%s", parsed.normalized, quantity)
            return

        # Get price if not provided
        if price is None:
            if symbol in self.data and self.current_time in self.data[symbol].index:
                raw_price = self.data[symbol].loc[self.current_time]['Close']
            else:
                logger.warning("event=order_rejected symbol=%s reason=no_price", parsed.normalized)
                return  # No price
        else:
            raw_price = price

        # Calculate slippage
        slippage_pct = self._get_slippage_pct(parsed.asset_class)

        if self.use_dynamic_slippage and self.market_impact_model:
            # Use dynamic market impact model
            fill_price = self._calculate_dynamic_fill_price(
                symbol, raw_price, quantity, side
            )
        else:
            # Fixed slippage
            if side == 'BUY':
                fill_price = raw_price * (1 + slippage_pct)
            else:
                fill_price = raw_price * (1 - slippage_pct)
            
        # Calculate commission
        notional = abs(quantity) * fill_price
        if parsed.asset_class == AssetClass.EQUITY:
            commission = max(self.min_commission, abs(quantity) * self.commission_per_share)
        elif parsed.asset_class == AssetClass.FOREX:
            commission = max(self.fx_min_commission, notional * (self.fx_commission_bps / 10000.0))
        else:
            commission = max(self.crypto_min_commission, notional * (self.crypto_commission_bps / 10000.0))

        logger.info(
            "event=fee_model asset=%s symbol=%s notional=%.2f commission=%.4f slippage_bps=%.2f",
            parsed.asset_class.value,
            parsed.normalized,
            notional,
            commission,
            slippage_pct * 10000,
        )

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
        ann_factor = self._annualization_factor()
        volatility = df['returns'].std() * np.sqrt(ann_factor)
        sharpe = (df['returns'].mean() / df['returns'].std()) * np.sqrt(ann_factor) if df['returns'].std() > 0 else 0
        
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
            'cagr': ((1 + total_return) ** (ann_factor/len(df))) - 1 if len(df) > 200 else total_return,
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

# PSEUDO-TESTS (fee model)
# FX: qty=10000, price=1.10, FX_COMMISSION_BPS=0.2 -> commission=10000*1.10*0.00002=0.22
# Crypto: qty=0.5, price=20000, CRYPTO_COMMISSION_BPS=15 -> commission=0.5*20000*0.0015=15.0
