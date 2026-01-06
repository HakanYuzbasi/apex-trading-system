"""
scripts/backtest.py - APEX Trading System Backtester
FIXED VERSION - Timezone handling corrected
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_generator import SignalGenerator
from data.market_data import MarketDataFetcher
from risk.risk_manager import RiskManager
from config import ApexConfig


class ApexBacktester:
    """Backtesting engine for APEX Trading System."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        
        self.signal_generator = SignalGenerator()
        self.market_data = MarketDataFetcher()
        self.risk_manager = RiskManager()
        self.risk_manager.set_starting_capital(initial_capital)
        
        # Tracking
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a symbol - FIXED timezone handling."""
        try:
            # Fetch data
            data = self.market_data.fetch_historical_data(symbol, days=500)
            if data.empty:
                return pd.DataFrame()
            
            # Fix: Reset index and handle timezone properly
            data = data.reset_index()
            
            # Convert date columns to timezone-naive for comparison
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            else:
                # Use index as Date
                data['Date'] = pd.to_datetime(data.index).tz_localize(None)
            
            # Parse input dates as timezone-naive
            start = pd.to_datetime(start_date).tz_localize(None)
            end = pd.to_datetime(end_date).tz_localize(None)
            
            # Filter by date range
            data = data[(data['Date'] >= start) & (data['Date'] <= end)]
            
            if not data.empty:
                data = data.set_index('Date')
            
            return data
            
        except Exception as e:
            return pd.DataFrame()
    
    def generate_signal_for_date(self, symbol: str, data: pd.DataFrame, date_idx: int) -> float:
        """Generate signal for a specific date."""
        if date_idx < 20 or len(data) < 20:
            return 0.0
        
        try:
            # Get prices up to this date
            prices = data['Close'].iloc[:date_idx+1]
            
            # Generate signal
            signal_data = self.signal_generator.generate_ml_signal(symbol)
            
            # Use historical data for better signals
            momentum = self.signal_generator.generate_momentum_signal(prices)
            mean_rev = self.signal_generator.generate_mean_reversion_signal(prices)
            signal = 0.7 * momentum + 0.3 * mean_rev
            
            return float(np.clip(signal, -1, 1))
        except:
            return 0.0
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run backtest for date range."""
        print("\n" + "="*70)
        print("APEX BACKTESTER - HISTORICAL PERFORMANCE")
        print("="*70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Symbols: {len(ApexConfig.SYMBOLS)}")
        print("="*70)
        print()
        
        # Load all historical data
        print("Loading historical data...")
        all_data = {}
        for i, symbol in enumerate(ApexConfig.SYMBOLS):
            if i % 10 == 0:
                print(f"  Loaded {i}/{len(ApexConfig.SYMBOLS)} symbols...")
            
            data = self.fetch_historical_data(symbol, start_date, end_date)
            if not data.empty and len(data) > 20:
                all_data[symbol] = data
        
        print(f"✅ Loaded {len(all_data)} symbols with data")
        print()
        
        if not all_data:
            print("❌ No data available for backtest")
            print("   Try using more recent dates or checking internet connection")
            return
        
        # Get all trading dates from first symbol
        first_symbol = list(all_data.keys())[0]
        all_dates = all_data[first_symbol].index.tolist()
        
        print(f"Trading days: {len(all_dates)}")
        if len(all_dates) > 0:
            print(f"Start: {all_dates[0]}")
            print(f"End: {all_dates[-1]}")
        print()
        print("-"*70)
        
        if len(all_dates) < 20:
            print("❌ Not enough trading days for backtest (need 20+)")
            return
        
        # Simulate each day
        for day_idx, date in enumerate(all_dates[20:], 1):
            daily_pnl = 0.0
            date_str = str(date).split()[0]  # Format as YYYY-MM-DD
            
            # Process each symbol
            for symbol in ApexConfig.SYMBOLS:
                if symbol not in all_data:
                    continue
                
                data = all_data[symbol]
                
                # Find this date in the data
                try:
                    date_idx = data.index.get_loc(date)
                except:
                    continue
                
                if date_idx >= len(data):
                    continue
                
                current_price = float(data['Close'].iloc[date_idx])
                if current_price <= 0:
                    continue
                
                # Get current position
                current_qty = self.positions.get(symbol, 0)
                
                # Generate signal
                signal = self.generate_signal_for_date(symbol, data, date_idx)
                
                # Trading logic
                if signal > ApexConfig.MIN_SIGNAL_THRESHOLD and current_qty == 0:
                    # BUY signal
                    position_size = ApexConfig.POSITION_SIZE_USD
                    if ApexConfig.is_commodity(symbol):
                        position_size = int(position_size * 0.8)
                    
                    qty = int(position_size / current_price)
                    if qty > 0 and self.capital >= position_size:
                        self.positions[symbol] = qty
                        self.capital -= position_size
                        self.trades.append({
                            'date': date_str,
                            'symbol': symbol,
                            'side': 'BUY',
                            'qty': qty,
                            'price': current_price,
                            'value': position_size
                        })
                
                elif signal < -ApexConfig.MIN_SIGNAL_THRESHOLD and current_qty > 0:
                    # SELL signal
                    proceeds = current_qty * current_price
                    self.capital += proceeds
                    
                    self.trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'side': 'SELL',
                        'qty': current_qty,
                        'price': current_price,
                        'value': proceeds
                    })
                    
                    del self.positions[symbol]
            
            # Calculate total equity
            portfolio_value = self.capital
            for symbol, qty in self.positions.items():
                if symbol in all_data:
                    data = all_data[symbol]
                    try:
                        date_idx = data.index.get_loc(date)
                        if date_idx < len(data):
                            price = float(data['Close'].iloc[date_idx])
                            portfolio_value += qty * price
                    except:
                        pass
            
            # Track equity
            self.equity_curve.append(portfolio_value)
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)
            
            # Update peak for drawdown
            self.peak_capital = max(self.peak_capital, portfolio_value)
            
            # Print every 5 days
            if day_idx % 5 == 0:
                print(f"Day {day_idx:4d}: ${portfolio_value:12,.2f} ({daily_return:+7.2%})")
        
        print("-"*70)
        print()
        self.print_summary()
    
    def print_summary(self):
        """Print backtest summary."""
        if not self.equity_curve:
            print("❌ No trading data")
            return
        
        equity_array = np.array(self.equity_curve)
        returns_array = np.array(self.daily_returns)
        
        # Metrics
        final_value = equity_array[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Daily metrics
        if len(equity_array) > 1:
            daily_returns_pct = np.diff(equity_array) / equity_array[:-1]
        else:
            daily_returns_pct = np.array([0.0])
        
        # Sharpe ratio (252 trading days per year)
        avg_daily_return = np.mean(daily_returns_pct)
        std_daily_return = np.std(daily_returns_pct)
        
        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return * 252) / (std_daily_return * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        # Drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        win_trades = len([t for t in self.trades if t['side'] == 'SELL'])
        win_rate = (win_trades / max(1, len([t for t in self.trades if t['side'] == 'BUY']))) * 100
        
        print("="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print()
        print(f"Initial Capital:        ${self.initial_capital:>15,.2f}")
        print(f"Final Value:            ${final_value:>15,.2f}")
        print(f"Total Return:           {total_return:>16.2%}")
        print()
        print(f"Total Trades:           {len(self.trades):>16}")
        print(f"Buy Signals:            {len([t for t in self.trades if t['side']=='BUY']):>16}")
        print(f"Sell Signals:           {len([t for t in self.trades if t['side']=='SELL']):>16}")
        print()
        print(f"Sharpe Ratio:           {sharpe_ratio:>16.2f}")
        print(f"Max Drawdown:           {max_drawdown:>16.2%}")
        print()
        print(f"Avg Daily Return:       {avg_daily_return:>16.2%}")
        print(f"Daily Volatility:       {std_daily_return:>16.2%}")
        print()
        print("="*70)
        print()
        
        if total_return > 0:
            print(f"✅ Backtest profitable: +{total_return:.2%}")
        else:
            print(f"⚠️  Backtest unprofitable: {total_return:.2%}")
            print("   Strategy needs tuning")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='APEX Backtester')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    
    args = parser.parse_args()
    
    backtester = ApexBacktester(initial_capital=args.capital)
    backtester.run_backtest(args.start, args.end)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Backtest stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
