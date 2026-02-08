"""
backtesting/advanced_backtester.py
PROFESSIONAL BACKTESTING ENGINE
- Proper time-series handling
- Transaction costs
- Slippage modeling
- Realistic order fills
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from core.symbols import AssetClass, parse_symbol, is_market_open
from config import ApexConfig
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Professional-grade backtesting engine.
    
    Features:
    - Walk-forward analysis
    - Transaction costs modeling
    - Realistic slippage
    - Position tracking
    - Portfolio-level statistics
    - Risk-adjusted returns
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0,
        fx_commission_bps: Optional[float] = None,
        crypto_commission_bps: Optional[float] = None,
        fx_spread_bps: Optional[float] = None,
        crypto_spread_bps: Optional[float] = None
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.fx_commission_bps = fx_commission_bps if fx_commission_bps is not None else ApexConfig.FX_COMMISSION_BPS
        self.crypto_commission_bps = crypto_commission_bps if crypto_commission_bps is not None else ApexConfig.CRYPTO_COMMISSION_BPS
        self.fx_spread_bps = fx_spread_bps if fx_spread_bps is not None else ApexConfig.FX_SPREAD_BPS
        self.crypto_spread_bps = crypto_spread_bps if crypto_spread_bps is not None else ApexConfig.CRYPTO_SPREAD_BPS
        
        self.reset()
        
        logger.info(f"✅ Advanced Backtester initialized")
        logger.info(f"   Initial Capital: ${initial_capital:,.0f}")
        logger.info(f"   Commission: ${commission_per_trade:.2f} per trade")
        logger.info(f"   Slippage: {slippage_bps:.1f} bps")
    
    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: quantity}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.current_date = None
        self._data_symbols = []

    def _annualization_factor(self, symbols: List[str]) -> int:
        classes = set()
        for symbol in symbols:
            try:
                classes.add(parse_symbol(symbol).asset_class)
            except ValueError:
                continue
        if classes == {AssetClass.CRYPTO}:
            return 365
        if classes == {AssetClass.FOREX}:
            return 260
        return 252

    def _is_market_open(self, symbol: str, date: datetime) -> bool:
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            return False
        return is_market_open(parsed, date, assume_daily=True)

    def _get_slippage_pct(self, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.FOREX:
            return self.fx_spread_bps / 10000.0
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_spread_bps / 10000.0
        return self.slippage_bps

    def _get_prev_date(self, data: Dict[str, pd.DataFrame], symbol: str, date: datetime) -> Optional[datetime]:
        if symbol not in data:
            return None
        idx = data[symbol].index
        if len(idx) == 0:
            return None
        if date in idx:
            pos = idx.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start
            if pos == 0:
                return None
            return idx[pos - 1]
        prior = idx[idx < date]
        if len(prior) == 0:
            return None
        return prior[-1]

    def _estimate_slippage_pct(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        asset_class: AssetClass,
        quantity: float
    ) -> float:
        base = self._get_slippage_pct(asset_class)
        if symbol not in data or date not in data[symbol].index:
            return base
        hist = data[symbol].loc[:date].tail(20)
        if hist.empty:
            return base
        returns = hist['Close'].pct_change().dropna() if 'Close' in hist.columns else pd.Series(dtype=float)
        vol = returns.std() if len(returns) > 0 else 0.0
        vol_mult = getattr(ApexConfig, "SLIPPAGE_VOL_MULT", 2.0)
        adv_mult = getattr(ApexConfig, "SLIPPAGE_ADV_MULT", 5.0)
        slip = base * (1 + vol * vol_mult)
        if 'Volume' in hist.columns:
            adv = hist['Volume'].mean()
            if adv and adv > 0:
                ratio = abs(quantity) / adv
                slip += base * ratio * adv_mult
        max_bps = getattr(ApexConfig, "BACKTEST_MAX_SLIPPAGE_BPS", 50)
        slip = min(slip, max_bps / 10000.0)
        return slip

    def _calculate_commission(self, symbol: str, notional: float) -> float:
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            return self.commission_per_trade

        if parsed.asset_class == AssetClass.EQUITY:
            return self.commission_per_trade
        if parsed.asset_class == AssetClass.FOREX:
            return notional * (self.fx_commission_bps / 10000.0)
        return notional * (self.crypto_commission_bps / 10000.0)
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        signal_generator,
        start_date: str,
        end_date: str,
        position_size_usd: float = 5000,
        max_positions: int = 15
    ) -> Dict:
        """
        Run complete backtest.
        
        Args:
            data: {symbol: DataFrame with OHLCV}
            signal_generator: Signal generator instance
            start_date: Backtest start date
            end_date: Backtest end date
            position_size_usd: Position size in dollars
            max_positions: Maximum concurrent positions
        
        Returns:
            Backtest results with metrics
        """
        logger.info(f"🔄 Running backtest: {start_date} to {end_date}")
        
        self.reset()
        self._data_symbols = list(data.keys())
        
        # Get date range
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Track metrics
        peak_equity = self.initial_capital
        
        for date in dates:
            self.current_date = date
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(data, date)
            
            # Record equity
            self.equity_curve.append({
                'date': date,
                'equity': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                daily_return = (portfolio_value / prev_equity - 1) if prev_equity > 0 else 0
                self.daily_returns.append(daily_return)
            
            # Update peak
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value
            
            # Generate signals and execute trades
            self._process_trading_day(
                data,
                date,
                signal_generator,
                position_size_usd,
                max_positions,
                portfolio_value
            )
        
        # Calculate final metrics
        results = self._calculate_metrics()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 BACKTEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${results['final_value']:,.0f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"Annual Return: {results['annual_return']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Win Rate: {results['win_rate']*100:.1f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Avg Trade: ${results['avg_trade_pnl']:,.2f}")
        logger.info(f"Total Commissions: ${results['total_commissions']:,.2f}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _process_trading_day(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator,
        position_size_usd: float,
        max_positions: int,
        portfolio_value: float
    ):
        """Process one trading day."""
        
        # Exit positions first
        self._check_exits(data, date, signal_generator)
        
        # Enter new positions
        self._check_entries(
            data,
            date,
            signal_generator,
            position_size_usd,
            max_positions,
            portfolio_value
        )
    
    def _check_exits(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator
    ):
        """Check exit conditions for existing positions."""
        
        for symbol in list(self.positions.keys()):
            if self.positions[symbol] == 0:
                continue

            if not self._is_market_open(symbol, date):
                continue

            prev_date = self._get_prev_date(data, symbol, date)
            if not prev_date:
                continue
            
            # Get current data
            if symbol not in data or date not in data[symbol].index:
                continue
            
            current_bar = data[symbol].loc[date]
            price = current_bar['Close']
            
            # Generate signal
            try:
                prices = data[symbol].loc[:prev_date, 'Close']
                signal_data = signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data['signal']
            except:
                signal = 0
            
            # Simple exit logic (can be enhanced)
            should_exit = False
            exit_reason = ""
            
            if self.positions[symbol] > 0 and signal < -0.30:
                should_exit = True
                exit_reason = "Bearish signal"
            
            elif self.positions[symbol] < 0 and signal > 0.30:
                should_exit = True
                exit_reason = "Bullish signal"
            
            if should_exit:
                self._execute_order(
                    symbol,
                    'SELL' if self.positions[symbol] > 0 else 'BUY',
                    abs(self.positions[symbol]),
                    price,
                    date,
                    exit_reason
                )
    
    def _check_entries(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator,
        position_size_usd: float,
        max_positions: int,
        portfolio_value: float
    ):
        """Check entry conditions for new positions."""
        
        # Count current positions
        current_positions = sum(1 for qty in self.positions.values() if qty != 0)
        
        if current_positions >= max_positions:
            return
        
        # Check each symbol
        for symbol in data.keys():
            if date not in data[symbol].index:
                continue

            if not self._is_market_open(symbol, date):
                continue

            prev_date = self._get_prev_date(data, symbol, date)
            if not prev_date:
                continue
            
            # Skip if already have position
            if symbol in self.positions and self.positions[symbol] != 0:
                continue
            
            current_bar = data[symbol].loc[date]
            price = current_bar['Close']
            
            # Generate signal
            try:
                prices = data[symbol].loc[:prev_date, 'Close']
                signal_data = signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data.get('signal', 0.0)
                confidence = signal_data.get('confidence', 0.5)
                quality = signal_data.get('quality', confidence)
            except:
                continue
            
            # Entry logic
            # Dynamic quality gating for better Sharpe
            min_conf = 0.35
            if confidence < min_conf or quality < 0.35:
                continue

            dynamic_threshold = 0.45 + (0.10 if confidence < 0.55 else 0.0)
            if abs(signal) > dynamic_threshold:  # Signal threshold
                # Calculate position size
                try:
                    asset_class = parse_symbol(symbol).asset_class
                except ValueError:
                    continue

                if asset_class == AssetClass.EQUITY:
                    shares = int(position_size_usd / price)
                    shares = min(shares, 200)  # Max shares limit
                else:
                    shares = position_size_usd / price

                # Scale by confidence/quality
                size_mult = 0.5 + 0.5 * float(np.clip(quality, 0.0, 1.0))
                if asset_class == AssetClass.EQUITY:
                    shares = max(1, int(shares * size_mult))
                else:
                    shares = shares * size_mult

                if asset_class != AssetClass.EQUITY and shares < 0.0001:
                    continue
                
                if asset_class == AssetClass.EQUITY and shares < 1:
                    continue
                
                # Check if we have enough cash
                slippage_pct = self._estimate_slippage_pct(symbol, data, date, asset_class, shares)
                notional = shares * price * (1 + slippage_pct)
                commission = self._calculate_commission(symbol, notional)
                cost = notional + commission
                
                if cost > self.cash:
                    continue
                
                # Execute order
                side = 'BUY' if signal > 0 else 'SELL'
                self._execute_order(
                    symbol,
                    side,
                    shares,
                    price,
                    date,
                    f"Signal: {signal:.3f}"
                )
                
                current_positions += 1
                
                if current_positions >= max_positions:
                    break
    
    def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        date: datetime,
        reason: str = ""
    ):
        """Execute an order with realistic fills."""

        if not self._is_market_open(symbol, date):
            logger.debug("event=order_rejected symbol=%s reason=market_closed", symbol)
            return False

        try:
            parsed = parse_symbol(symbol)
            asset_class = parsed.asset_class
        except ValueError:
            logger.debug("event=order_rejected symbol=%s reason=invalid_symbol", symbol)
            return False

        logger.info(
            "event=symbol_normalization input=%s normalized=%s broker=%s",
            symbol,
            parsed.normalized,
            parsed.normalized,
        )

        if asset_class == AssetClass.EQUITY and isinstance(quantity, float) and not quantity.is_integer():
            logger.debug("event=order_rejected symbol=%s reason=fractional_equity quantity=%s", symbol, quantity)
            return False
        
        # Apply slippage
        slippage_pct = self._get_slippage_pct(asset_class)
        if side == 'BUY':
            execution_price = price * (1 + slippage_pct)
        else:
            execution_price = price * (1 - slippage_pct)
        
        # Calculate costs
        gross_value = quantity * execution_price
        commission = self._calculate_commission(symbol, gross_value)
        logger.info(
            "event=fee_model asset=%s symbol=%s notional=%.2f commission=%.4f slippage_bps=%.2f",
            asset_class.value,
            symbol,
            gross_value,
            commission,
            slippage_pct * 10000,
        )
        
        if side == 'BUY':
            total_cost = gross_value + commission
            
            # Check cash
            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {symbol}: need ${total_cost:,.2f}, have ${self.cash:,.2f}")
                return False
            
            # Execute
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        else:  # SELL
            total_proceeds = gross_value - commission
            
            # Execute
            self.cash += total_proceeds
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'slippage': abs(execution_price - price) * quantity,
            'reason': reason,
            'cash_after': self.cash
        }
        
        self.trades.append(trade)
        
        logger.debug(f"{date.date()}: {side} {quantity} {symbol} @ ${execution_price:.2f}")
        
        return True
    
    def _calculate_portfolio_value(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> float:
        """Calculate total portfolio value."""
        
        positions_value = 0
        
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            
            if symbol not in data or date not in data[symbol].index:
                continue
            
            price = data[symbol].loc[date, 'Close']
            positions_value += qty * price
        
        return self.cash + positions_value
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        final_value = equity_df['equity'].iloc[-1]
        total_return = (final_value / self.initial_capital) - 1
        
        # Annualized return
        days = len(equity_df)
        symbols_for_annual = self._data_symbols or list(self.positions.keys()) + [t['symbol'] for t in self.trades]
        ann_factor = self._annualization_factor(symbols_for_annual)
        years = days / ann_factor
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility and Sharpe
        if len(self.daily_returns) > 1:
            daily_vol = np.std(self.daily_returns)
            annual_vol = daily_vol * np.sqrt(ann_factor)
            sharpe_ratio = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
        else:
            annual_vol = 0
            sharpe_ratio = 0
        
        # Max Drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            total_commissions = trades_df['commission'].sum()
            total_slippage = trades_df['slippage'].sum()
            
            # Calculate P&L per trade (simplified)
            winning_trades = 0
            total_pnl = 0
            
            # Match buys and sells
            positions = defaultdict(list)
            
            for _, trade in trades_df.iterrows():
                symbol = trade['symbol']
                
                if trade['side'] == 'BUY':
                    positions[symbol].append({
                        'qty': trade['quantity'],
                        'price': trade['execution_price'],
                        'commission': trade['commission']
                    })
                
                elif trade['side'] == 'SELL' and positions[symbol]:
                    entry = positions[symbol].pop(0)
                    pnl = (trade['execution_price'] - entry['price']) * trade['quantity']
                    pnl -= (entry['commission'] + trade['commission'])
                    
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
            
            completed_trades = len(trades_df[trades_df['side'] == 'SELL'])
            win_rate = winning_trades / completed_trades if completed_trades > 0 else 0
            avg_trade_pnl = total_pnl / completed_trades if completed_trades > 0 else 0
        
        else:
            total_trades = 0
            total_commissions = 0
            total_slippage = 0
            win_rate = 0
            avg_trade_pnl = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def plot_results(self, results: Dict):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            
            equity_df = results['equity_curve']
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            axes[0].plot(equity_df['date'], equity_df['equity'], label='Portfolio Value')
            axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            axes[0].set_title('Equity Curve')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Drawdown
            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak * 100
            axes[1].fill_between(equity_df['date'], drawdown, 0, alpha=0.3, color='red')
            axes[1].set_title('Drawdown')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=150)
            logger.info("📊 Results saved to backtest_results.png")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Test backtester
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        # Generate random OHLCV
        close = 100 + np.random.randn(len(dates)).cumsum()
        data[symbol] = pd.DataFrame({
            'Open': close + np.random.randn(len(dates)) * 0.5,
            'High': close + abs(np.random.randn(len(dates))),
            'Low': close - abs(np.random.randn(len(dates))),
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Mock signal generator
    class MockSignalGenerator:
        def generate_ml_signal(self, symbol, prices):
            return {'signal': np.random.randn() * 0.5}
    
    # Run backtest
    backtester = AdvancedBacktester(initial_capital=100000)
    
    results = backtester.run_backtest(
        data=data,
        signal_generator=MockSignalGenerator(),
        start_date='2023-01-01',
        end_date='2023-12-31',
        position_size_usd=5000,
        max_positions=3
    )
    
    print("\n✅ Backtester tests complete!")
