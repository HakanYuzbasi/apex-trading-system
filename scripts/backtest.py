"""
scripts/backtest.py - APEX Trading System Backtester
IMPROVED VERSION - Enhanced performance with risk management
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_generator import SignalGenerator
from data.market_data import MarketDataFetcher
from risk.risk_manager import RiskManager
from config import ApexConfig


class ApexBacktester:
    """Enhanced backtesting engine for APEX Trading System."""

    # Backtest-specific configuration
    WARMUP_PERIOD = 50  # Days required before trading (increased from 20)
    MIN_HOLD_DAYS = 5  # Minimum days to hold a position (increased from 3)
    STOP_LOSS_PCT = 0.10  # 10% stop loss (increased from 7%)
    TRAILING_STOP_PCT = 0.08  # 8% trailing stop from peak (increased from 5%)
    EXIT_THRESHOLD_MULTIPLIER = 0.7  # Exit threshold = entry threshold * 0.7

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital

        self.signal_generator = SignalGenerator()
        self.market_data = MarketDataFetcher()
        self.risk_manager = RiskManager()
        self.risk_manager.set_starting_capital(initial_capital)

        # Tracking
        self.positions = {}  # symbol -> {'qty': int, 'entry_price': float, 'entry_date': str, 'peak_price': float}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.benchmark_curve = []  # SPY benchmark

        # Sector exposure tracking
        self.sector_exposure = {}

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a symbol - FIXED timezone handling."""
        try:
            # Fetch data
            data = self.market_data.fetch_historical_data(symbol, days=600)
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

    def calculate_volatility(self, data: pd.DataFrame, date_idx: int, lookback: int = 20) -> float:
        """Calculate historical volatility for position sizing."""
        if date_idx < lookback:
            return 0.02  # Default volatility

        try:
            prices = data['Close'].iloc[max(0, date_idx-lookback):date_idx+1]
            returns = prices.pct_change().dropna()
            if len(returns) < 5:
                return 0.02
            return float(returns.std())
        except:
            return 0.02

    def get_dynamic_position_size(self, symbol: str, volatility: float, current_price: float) -> int:
        """Calculate position size adjusted for volatility."""
        base_size = ApexConfig.POSITION_SIZE_USD

        # Reduce position size for commodities
        if ApexConfig.is_commodity(symbol):
            base_size = int(base_size * 0.8)

        # Volatility adjustment: higher vol = smaller position
        # Target volatility of 2% daily
        target_vol = 0.02
        vol_scalar = min(target_vol / max(volatility, 0.005), 1.5)
        adjusted_size = base_size * vol_scalar

        # Apply limits
        adjusted_size = min(adjusted_size, ApexConfig.POSITION_SIZE_USD * 1.5)
        adjusted_size = max(adjusted_size, ApexConfig.POSITION_SIZE_USD * 0.5)

        qty = int(adjusted_size / current_price)
        qty = min(qty, ApexConfig.MAX_SHARES_PER_POSITION)

        return max(1, qty)

    def calculate_sector_exposure(self, all_data: Dict, date) -> Dict[str, float]:
        """Calculate current sector exposure as percentage of portfolio."""
        portfolio_value = self.capital
        sector_values = {}

        for symbol, pos in self.positions.items():
            if symbol in all_data:
                try:
                    data = all_data[symbol]
                    date_idx = data.index.get_loc(date)
                    price = float(data['Close'].iloc[date_idx])
                    value = pos['qty'] * price
                    portfolio_value += value

                    sector = ApexConfig.get_sector(symbol)
                    sector_values[sector] = sector_values.get(sector, 0) + value
                except:
                    pass

        if portfolio_value <= 0:
            return {}

        return {sector: value / portfolio_value for sector, value in sector_values.items()}

    def check_sector_limit(self, symbol: str, position_value: float, all_data: Dict, date) -> bool:
        """Check if adding a position would breach sector limits."""
        sector = ApexConfig.get_sector(symbol)
        if sector == "Unknown":
            return True

        current_exposure = self.calculate_sector_exposure(all_data, date)
        current_sector_value = current_exposure.get(sector, 0)

        # Calculate new exposure
        portfolio_value = self.capital
        for sym, pos in self.positions.items():
            if sym in all_data:
                try:
                    data = all_data[sym]
                    date_idx = data.index.get_loc(date)
                    price = float(data['Close'].iloc[date_idx])
                    portfolio_value += pos['qty'] * price
                except:
                    pass

        new_exposure = (current_sector_value * portfolio_value + position_value) / (portfolio_value + position_value)

        return new_exposure <= ApexConfig.MAX_SECTOR_EXPOSURE

    def apply_transaction_costs(self, value: float, is_buy: bool) -> float:
        """Apply commission and slippage to transaction."""
        # Commission
        commission = ApexConfig.COMMISSION_PER_TRADE

        # Slippage (worse price for us)
        slippage_pct = ApexConfig.SLIPPAGE_BPS / 10000
        slippage = value * slippage_pct

        if is_buy:
            # Pay more when buying
            return value + commission + slippage
        else:
            # Receive less when selling
            return value - commission - slippage

    def check_stop_loss(self, symbol: str, current_price: float, date_str: str) -> bool:
        """Check if position should be stopped out."""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        peak_price = pos.get('peak_price', entry_price)

        # Update peak price
        if current_price > peak_price:
            self.positions[symbol]['peak_price'] = current_price
            peak_price = current_price

        # Check hard stop loss
        loss_pct = (current_price - entry_price) / entry_price
        if loss_pct <= -self.STOP_LOSS_PCT:
            return True

        # Check trailing stop (only after profit)
        if peak_price > entry_price:
            drawdown_from_peak = (current_price - peak_price) / peak_price
            if drawdown_from_peak <= -self.TRAILING_STOP_PCT:
                return True

        return False

    def check_min_hold_period(self, symbol: str, current_date_str: str) -> bool:
        """Check if minimum hold period has passed."""
        if symbol not in self.positions:
            return True

        entry_date = self.positions[symbol].get('entry_date', '')
        if not entry_date:
            return True

        try:
            entry = pd.to_datetime(entry_date)
            current = pd.to_datetime(current_date_str)
            days_held = (current - entry).days
            return days_held >= self.MIN_HOLD_DAYS
        except:
            return True

    def generate_signal_for_date(self, symbol: str, data: pd.DataFrame, date_idx: int) -> Tuple[float, float]:
        """Generate signal and confidence for a specific date."""
        if date_idx < self.WARMUP_PERIOD or len(data) < self.WARMUP_PERIOD:
            return 0.0, 0.0

        try:
            # Get prices up to this date (not including future data)
            prices = data['Close'].iloc[:date_idx+1]

            # Use the signal generator with actual price data
            signal_data = self.signal_generator.generate_ml_signal(symbol, prices)

            signal = float(np.clip(signal_data['signal'], -1, 1))
            confidence = float(signal_data.get('confidence', 0.0))

            return signal, confidence
        except Exception as e:
            return 0.0, 0.0

    def run_backtest(self, start_date: str, end_date: str):
        """Run backtest for date range."""
        print("\n" + "="*70)
        print("APEX BACKTESTER - ENHANCED PERFORMANCE")
        print("="*70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Symbols: {len(ApexConfig.SYMBOLS)}")
        print()
        print("Enhancements Active:")
        print(f"  • Warm-up Period: {self.WARMUP_PERIOD} days")
        print(f"  • Stop Loss: {self.STOP_LOSS_PCT:.0%}")
        print(f"  • Trailing Stop: {self.TRAILING_STOP_PCT:.0%}")
        print(f"  • Min Hold Period: {self.MIN_HOLD_DAYS} days")
        print(f"  • Confidence Filter: {ApexConfig.MIN_CONFIDENCE:.0%}")
        print(f"  • Sector Limit: {ApexConfig.MAX_SECTOR_EXPOSURE:.0%}")
        print(f"  • Transaction Costs: ${ApexConfig.COMMISSION_PER_TRADE} + {ApexConfig.SLIPPAGE_BPS}bps")
        print("="*70)
        print()

        # Load all historical data
        print("Loading historical data...")
        all_data = {}
        for i, symbol in enumerate(ApexConfig.SYMBOLS):
            if i % 10 == 0:
                print(f"  Loaded {i}/{len(ApexConfig.SYMBOLS)} symbols...")

            data = self.fetch_historical_data(symbol, start_date, end_date)
            if not data.empty and len(data) > self.WARMUP_PERIOD:
                all_data[symbol] = data

        print(f"✅ Loaded {len(all_data)} symbols with data")
        print()

        if not all_data:
            print("❌ No data available for backtest")
            print("   Try using more recent dates or checking internet connection")
            return

        # Get benchmark data (SPY)
        spy_data = all_data.get('SPY', pd.DataFrame())

        # Get all trading dates from first symbol
        first_symbol = list(all_data.keys())[0]
        all_dates = all_data[first_symbol].index.tolist()

        print(f"Trading days: {len(all_dates)}")
        if len(all_dates) > 0:
            print(f"Start: {all_dates[0]}")
            print(f"End: {all_dates[-1]}")
        print()
        print("-"*70)

        if len(all_dates) < self.WARMUP_PERIOD:
            print(f"❌ Not enough trading days for backtest (need {self.WARMUP_PERIOD}+)")
            return

        # Track benchmark
        if not spy_data.empty:
            spy_start_price = float(spy_data['Close'].iloc[self.WARMUP_PERIOD])
        else:
            spy_start_price = 1.0

        # Simulate each day
        for day_idx, date in enumerate(all_dates[self.WARMUP_PERIOD:], 1):
            date_str = str(date).split()[0]  # Format as YYYY-MM-DD

            # Check stop losses first
            symbols_to_close = []
            for symbol in list(self.positions.keys()):
                if symbol not in all_data:
                    continue

                data = all_data[symbol]
                try:
                    date_idx_data = data.index.get_loc(date)
                    current_price = float(data['Close'].iloc[date_idx_data])

                    if self.check_stop_loss(symbol, current_price, date_str):
                        symbols_to_close.append((symbol, current_price, 'STOP'))
                except:
                    pass

            # Execute stop losses
            for symbol, price, reason in symbols_to_close:
                pos = self.positions[symbol]
                proceeds = pos['qty'] * price
                proceeds = self.apply_transaction_costs(proceeds, is_buy=False)
                self.capital += proceeds

                self.trades.append({
                    'date': date_str,
                    'symbol': symbol,
                    'side': 'SELL',
                    'qty': pos['qty'],
                    'price': price,
                    'value': proceeds,
                    'reason': reason
                })

                del self.positions[symbol]

            # Process each symbol for new signals
            for symbol in ApexConfig.SYMBOLS:
                if symbol not in all_data:
                    continue

                data = all_data[symbol]

                # Find this date in the data
                try:
                    date_idx_data = data.index.get_loc(date)
                except:
                    continue

                if date_idx_data >= len(data):
                    continue

                current_price = float(data['Close'].iloc[date_idx_data])
                if current_price <= 0:
                    continue

                # Get current position
                current_pos = self.positions.get(symbol, None)
                has_position = current_pos is not None

                # Generate signal with confidence
                signal, confidence = self.generate_signal_for_date(symbol, data, date_idx_data)

                # Calculate volatility for position sizing
                volatility = self.calculate_volatility(data, date_idx_data)

                # Trading logic with enhanced filters
                entry_threshold = ApexConfig.MIN_SIGNAL_THRESHOLD
                exit_threshold = entry_threshold * self.EXIT_THRESHOLD_MULTIPLIER

                if signal > entry_threshold and not has_position:
                    # BUY signal - check confidence filter
                    if confidence < ApexConfig.MIN_CONFIDENCE:
                        continue

                    # Check max positions
                    if len(self.positions) >= ApexConfig.MAX_POSITIONS:
                        continue

                    # Dynamic position sizing
                    qty = self.get_dynamic_position_size(symbol, volatility, current_price)
                    position_value = qty * current_price

                    # Check sector exposure limit
                    if not self.check_sector_limit(symbol, position_value, all_data, date):
                        continue

                    # Apply transaction costs
                    cost = self.apply_transaction_costs(position_value, is_buy=True)

                    if self.capital >= cost:
                        self.positions[symbol] = {
                            'qty': qty,
                            'entry_price': current_price,
                            'entry_date': date_str,
                            'peak_price': current_price
                        }
                        self.capital -= cost
                        self.trades.append({
                            'date': date_str,
                            'symbol': symbol,
                            'side': 'BUY',
                            'qty': qty,
                            'price': current_price,
                            'value': cost,
                            'reason': 'SIGNAL'
                        })

                elif signal < -exit_threshold and has_position:
                    # SELL signal - check minimum hold period
                    if not self.check_min_hold_period(symbol, date_str):
                        continue

                    pos = self.positions[symbol]
                    proceeds = pos['qty'] * current_price
                    proceeds = self.apply_transaction_costs(proceeds, is_buy=False)
                    self.capital += proceeds

                    self.trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'side': 'SELL',
                        'qty': pos['qty'],
                        'price': current_price,
                        'value': proceeds,
                        'reason': 'SIGNAL'
                    })

                    del self.positions[symbol]

            # Calculate total equity
            portfolio_value = self.capital
            for symbol, pos in self.positions.items():
                if symbol in all_data:
                    data = all_data[symbol]
                    try:
                        date_idx_data = data.index.get_loc(date)
                        if date_idx_data < len(data):
                            price = float(data['Close'].iloc[date_idx_data])
                            portfolio_value += pos['qty'] * price
                    except:
                        pass

            # Track equity
            self.equity_curve.append(portfolio_value)
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)

            # Track benchmark
            if not spy_data.empty:
                try:
                    spy_idx = spy_data.index.get_loc(date)
                    spy_price = float(spy_data['Close'].iloc[spy_idx])
                    spy_return = (spy_price - spy_start_price) / spy_start_price
                    self.benchmark_curve.append(spy_return)
                except:
                    self.benchmark_curve.append(self.benchmark_curve[-1] if self.benchmark_curve else 0)

            # Update peak for drawdown
            self.peak_capital = max(self.peak_capital, portfolio_value)

            # Print every 5 days
            if day_idx % 5 == 0:
                pos_count = len(self.positions)
                print(f"Day {day_idx:4d}: ${portfolio_value:12,.2f} ({daily_return:+7.2%}) [{pos_count} pos]")

        print("-"*70)
        print()
        self.print_summary()

    def _calculate_win_rate(self) -> Tuple[float, int, int]:
        """
        Calculate win rate based on completed round-trip trades.

        Returns:
            Tuple of (win_rate, winners, total_trades)
        """
        # Match buys with sells to calculate round-trip P&L
        positions = {}  # symbol -> list of (qty, price)
        completed_trades = []

        for trade in self.trades:
            symbol = trade['symbol']
            side = trade['side']
            qty = trade['qty']
            price = trade['price']

            if side == 'BUY':
                if symbol not in positions:
                    positions[symbol] = []
                positions[symbol].append({'qty': qty, 'price': price})

            elif side == 'SELL':
                if symbol in positions and len(positions[symbol]) > 0:
                    # FIFO matching
                    entry = positions[symbol].pop(0)
                    pnl = (price - entry['price']) * qty
                    completed_trades.append({'symbol': symbol, 'pnl': pnl})

        if not completed_trades:
            return 0.0, 0, 0

        winners = sum(1 for t in completed_trades if t['pnl'] > 0)
        return (winners / len(completed_trades)) * 100, winners, len(completed_trades)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        positions = {}
        gross_profit = 0.0
        gross_loss = 0.0

        for trade in self.trades:
            symbol = trade['symbol']
            side = trade['side']
            qty = trade['qty']
            price = trade['price']

            if side == 'BUY':
                if symbol not in positions:
                    positions[symbol] = []
                positions[symbol].append({'qty': qty, 'price': price})

            elif side == 'SELL':
                if symbol in positions and len(positions[symbol]) > 0:
                    entry = positions[symbol].pop(0)
                    pnl = (price - entry['price']) * qty
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

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

        # Sortino ratio (downside deviation only)
        downside_returns = daily_returns_pct[daily_returns_pct < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (avg_daily_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        else:
            sortino_ratio = float('inf') if avg_daily_return > 0 else 0.0

        # Drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        win_rate, winners, total_completed = self._calculate_win_rate()

        # Profit factor
        profit_factor = self._calculate_profit_factor()

        # Trade breakdown
        buy_trades = [t for t in self.trades if t['side'] == 'BUY']
        sell_trades = [t for t in self.trades if t['side'] == 'SELL']
        stop_trades = [t for t in self.trades if t.get('reason') == 'STOP']
        signal_exits = [t for t in self.trades if t['side'] == 'SELL' and t.get('reason') == 'SIGNAL']

        # Benchmark comparison
        benchmark_return = self.benchmark_curve[-1] if self.benchmark_curve else 0.0
        alpha = total_return - benchmark_return

        # Calculate CAGR
        years = len(equity_array) / 252
        if years > 0 and final_value > 0:
            cagr = (final_value / self.initial_capital) ** (1 / years) - 1
        else:
            cagr = 0.0

        print("="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print()
        print(f"Initial Capital:        ${self.initial_capital:>15,.2f}")
        print(f"Final Value:            ${final_value:>15,.2f}")
        print(f"Total Return:           {total_return:>16.2%}")
        print(f"CAGR:                   {cagr:>16.2%}")
        print()
        print("-"*70)
        print("RISK METRICS")
        print("-"*70)
        print(f"Sharpe Ratio:           {sharpe_ratio:>16.2f}")
        print(f"Sortino Ratio:          {sortino_ratio:>16.2f}")
        print(f"Max Drawdown:           {max_drawdown:>16.2%}")
        print(f"Daily Volatility:       {std_daily_return:>16.2%}")
        print(f"Avg Daily Return:       {avg_daily_return:>16.4%}")
        print()
        print("-"*70)
        print("TRADE STATISTICS")
        print("-"*70)
        print(f"Total Trades:           {len(self.trades):>16}")
        print(f"Buy Signals:            {len(buy_trades):>16}")
        print(f"Sell Signals:           {len(sell_trades):>16}")
        print(f"  - Signal Exits:       {len(signal_exits):>16}")
        print(f"  - Stop Loss Exits:    {len(stop_trades):>16}")
        print()
        print(f"Completed Round Trips:  {total_completed:>16}")
        print(f"Win Rate:               {win_rate:>15.1f}%")
        print(f"Profit Factor:          {profit_factor:>16.2f}")
        print()
        print("-"*70)
        print("BENCHMARK COMPARISON (SPY)")
        print("-"*70)
        print(f"Strategy Return:        {total_return:>16.2%}")
        print(f"Benchmark Return:       {benchmark_return:>16.2%}")
        print(f"Alpha (vs SPY):         {alpha:>16.2%}")
        print()
        print("="*70)
        print()

        if total_return > 0:
            print(f"✅ Backtest profitable: +{total_return:.2%}")
            if alpha > 0:
                print(f"✅ Outperformed SPY by {alpha:.2%}")
            else:
                print(f"⚠️  Underperformed SPY by {abs(alpha):.2%}")
        else:
            print(f"⚠️  Backtest unprofitable: {total_return:.2%}")
            print("   Strategy needs tuning")

        if sharpe_ratio >= 1.0:
            print(f"✅ Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio >= 0.5:
            print(f"⚠️  Moderate risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        else:
            print(f"❌ Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

        print()


def main():
    parser = argparse.ArgumentParser(description='APEX Backtester - Enhanced Version')
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
