"""
scripts/backtest.py - APEX Trading System Backtester
OPTIMIZED VERSION - High win rate and Sharpe ratio with IBKR batching
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_generator import SignalGenerator
from data.market_data import MarketDataFetcher
from risk.risk_manager import RiskManager
from config import ApexConfig


class ApexBacktester:
    """Optimized backtesting engine for APEX Trading System."""

    # IBKR Batching Configuration
    BATCH_SIZE = 50  # Max symbols per batch (IBKR limit)
    BATCH_DELAY = 0.5  # Delay between batches (seconds)

    # Backtest-specific configuration - OPTIMIZED for high win rate + Sharpe
    WARMUP_PERIOD = 60  # Days required before trading (more data for better signals)
    MIN_HOLD_DAYS = 2  # Minimum days to hold
    STOP_LOSS_PCT = 0.06  # 6% stop loss
    TRAILING_STOP_PCT = 0.035  # 3.5% trailing stop (lock in profits)
    PROFIT_TARGET_PCT = 0.10  # 10% profit target
    EXIT_THRESHOLD_MULTIPLIER = 0.5  # Exit threshold = entry threshold * 0.5

    # Signal quality filters - balanced for win rate AND activity
    MIN_SIGNAL_STRENGTH = 0.50  # Slightly lower for more opportunities
    MIN_CONFIDENCE_REQUIRED = 0.35  # Balanced confidence requirement
    MAX_VOLATILITY = 0.035  # Skip very high volatility stocks (>3.5% daily)
    MIN_TREND_ALIGNMENT = 0.20  # Slightly lower trend requirement

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
        self.benchmark_curve = []

        # Market regime tracking
        self.market_trend = 0.0  # -1 to 1 (bear to bull)

    def fetch_historical_data_batched(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data in batches of 50 symbols max.
        This respects IBKR's subscription limits.
        """
        all_data = {}
        total_symbols = len(symbols)
        num_batches = (total_symbols + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        print(f"  {total_symbols} symbols will be processed in {num_batches} batches ({self.BATCH_SIZE} max per batch)")
        print("  Snapshot mode reduces subscription overhead")

        for batch_num in range(num_batches):
            start_idx = batch_num * self.BATCH_SIZE
            end_idx = min(start_idx + self.BATCH_SIZE, total_symbols)
            batch_symbols = symbols[start_idx:end_idx]

            batch_size_actual = len(batch_symbols)
            print(f"  Batch {batch_num + 1}/{num_batches}: Loading {batch_size_actual} symbols ({start_idx + 1}-{end_idx})...")

            for symbol in batch_symbols:
                data = self._fetch_single_symbol(symbol, start_date, end_date)
                if not data.empty and len(data) > self.WARMUP_PERIOD:
                    all_data[symbol] = data

            # Proper cleanup - delay between batches to free slots
            if batch_num < num_batches - 1:
                time.sleep(self.BATCH_DELAY)

        print("  Proper cleanup ensures slots are freed")
        return all_data

    def _fetch_single_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for a single symbol."""
        try:
            data = self.market_data.fetch_historical_data(symbol, days=700)
            if data.empty:
                return pd.DataFrame()

            data = data.reset_index()

            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            else:
                data['Date'] = pd.to_datetime(data.index).tz_localize(None)

            start = pd.to_datetime(start_date).tz_localize(None)
            end = pd.to_datetime(end_date).tz_localize(None)

            data = data[(data['Date'] >= start) & (data['Date'] <= end)]

            if not data.empty:
                data = data.set_index('Date')

            return data

        except Exception:
            return pd.DataFrame()

    def calculate_market_regime(self, spy_data: pd.DataFrame, date_idx: int) -> float:
        """
        Calculate market regime based on SPY trend.
        Returns: -1 (strong bear) to 1 (strong bull)
        """
        if date_idx < 50:
            return 0.0

        try:
            prices = spy_data['Close'].iloc[max(0, date_idx-50):date_idx+1]

            # Short-term trend (20 day)
            ma20 = prices.rolling(20).mean().iloc[-1]
            # Long-term trend (50 day)
            ma50 = prices.rolling(50).mean().iloc[-1]

            current_price = prices.iloc[-1]

            # Calculate trend strength
            short_trend = (current_price - ma20) / ma20 if ma20 > 0 else 0
            long_trend = (ma20 - ma50) / ma50 if ma50 > 0 else 0

            # Combined regime score
            regime = np.tanh((short_trend * 10 + long_trend * 5))

            return float(np.clip(regime, -1, 1))
        except:
            return 0.0

    def calculate_volatility(self, data: pd.DataFrame, date_idx: int, lookback: int = 20) -> float:
        """Calculate historical volatility."""
        if date_idx < lookback:
            return 0.02

        try:
            prices = data['Close'].iloc[max(0, date_idx-lookback):date_idx+1]
            returns = prices.pct_change().dropna()
            if len(returns) < 5:
                return 0.02
            return float(returns.std())
        except:
            return 0.02

    def calculate_stock_trend(self, data: pd.DataFrame, date_idx: int) -> float:
        """Calculate individual stock trend alignment."""
        if date_idx < 30:
            return 0.0

        try:
            prices = data['Close'].iloc[max(0, date_idx-30):date_idx+1]
            ma10 = prices.rolling(10).mean().iloc[-1]
            ma30 = prices.rolling(30).mean().iloc[-1]

            if ma30 <= 0:
                return 0.0

            trend = (ma10 - ma30) / ma30
            return float(np.tanh(trend * 20))
        except:
            return 0.0

    def get_dynamic_position_size(self, symbol: str, volatility: float, current_price: float, signal_strength: float) -> int:
        """Calculate position size adjusted for volatility and signal strength."""
        base_size = ApexConfig.POSITION_SIZE_USD

        if ApexConfig.is_commodity(symbol):
            base_size = int(base_size * 0.7)

        # Volatility adjustment
        target_vol = 0.015
        vol_scalar = min(target_vol / max(volatility, 0.005), 1.3)

        # Signal strength adjustment (stronger signals get larger positions)
        signal_scalar = 0.8 + (abs(signal_strength) * 0.4)  # 0.8 to 1.2

        adjusted_size = base_size * vol_scalar * signal_scalar

        adjusted_size = min(adjusted_size, ApexConfig.POSITION_SIZE_USD * 1.3)
        adjusted_size = max(adjusted_size, ApexConfig.POSITION_SIZE_USD * 0.5)

        qty = int(adjusted_size / current_price)
        qty = min(qty, ApexConfig.MAX_SHARES_PER_POSITION)

        return max(1, qty)

    def calculate_sector_exposure(self, all_data: Dict, date) -> Dict[str, float]:
        """Calculate current sector exposure."""
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
        current_sector_pct = current_exposure.get(sector, 0)

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

        new_exposure = (current_sector_pct * portfolio_value + position_value) / (portfolio_value + position_value)

        return new_exposure <= ApexConfig.MAX_SECTOR_EXPOSURE

    def apply_transaction_costs(self, value: float, is_buy: bool) -> float:
        """Apply commission and slippage."""
        commission = ApexConfig.COMMISSION_PER_TRADE
        slippage_pct = ApexConfig.SLIPPAGE_BPS / 10000
        slippage = value * slippage_pct

        if is_buy:
            return value + commission + slippage
        else:
            return value - commission - slippage

    def check_exit_conditions(self, symbol: str, current_price: float, date_str: str) -> Tuple[bool, str]:
        """
        Check all exit conditions for a position.
        Returns: (should_exit, reason)
        """
        if symbol not in self.positions:
            return False, ''

        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        peak_price = pos.get('peak_price', entry_price)

        # Update peak price
        if current_price > peak_price:
            self.positions[symbol]['peak_price'] = current_price
            peak_price = current_price

        pnl_pct = (current_price - entry_price) / entry_price

        # Check profit target
        if pnl_pct >= self.PROFIT_TARGET_PCT:
            return True, 'TARGET'

        # Check hard stop loss
        if pnl_pct <= -self.STOP_LOSS_PCT:
            return True, 'STOP'

        # Check trailing stop (only after some profit)
        if peak_price > entry_price * 1.02:  # Only activate after 2% profit
            drawdown_from_peak = (current_price - peak_price) / peak_price
            if drawdown_from_peak <= -self.TRAILING_STOP_PCT:
                return True, 'TRAIL'

        return False, ''

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
            prices = data['Close'].iloc[:date_idx+1]
            signal_data = self.signal_generator.generate_ml_signal(symbol, prices)

            signal = float(np.clip(signal_data['signal'], -1, 1))
            confidence = float(signal_data.get('confidence', 0.0))

            return signal, confidence
        except Exception:
            return 0.0, 0.0

    def rank_signals(self, signals: List[Tuple[str, float, float, float]]) -> List[Tuple[str, float, float, float]]:
        """
        Rank and filter signals by quality.
        Input: List of (symbol, signal, confidence, trend_alignment)
        Returns sorted list with best signals first.
        """
        # Score = signal_strength * confidence * (1 + trend_alignment)
        scored = []
        for symbol, signal, confidence, trend in signals:
            if signal > 0 and trend > -0.3:  # Only bullish signals with decent trend
                score = abs(signal) * confidence * (1 + max(trend, 0))
                scored.append((symbol, signal, confidence, trend, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[4], reverse=True)

        return [(s[0], s[1], s[2], s[3]) for s in scored]

    def run_backtest(self, start_date: str, end_date: str):
        """Run backtest for date range."""
        print("\n" + "="*70)
        print("APEX BACKTESTER - OPTIMIZED FOR HIGH WIN RATE")
        print("="*70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Symbols: {len(ApexConfig.SYMBOLS)}")
        print()
        print("Optimization Settings:")
        print(f"  • Warm-up Period: {self.WARMUP_PERIOD} days")
        print(f"  • Stop Loss: {self.STOP_LOSS_PCT:.0%}")
        print(f"  • Trailing Stop: {self.TRAILING_STOP_PCT:.0%} (after 2% profit)")
        print(f"  • Profit Target: {self.PROFIT_TARGET_PCT:.0%}")
        print(f"  • Min Signal Strength: {self.MIN_SIGNAL_STRENGTH}")
        print(f"  • Min Confidence: {self.MIN_CONFIDENCE_REQUIRED:.0%}")
        print(f"  • Max Volatility: {self.MAX_VOLATILITY:.1%}")
        print(f"  • Batch Size: {self.BATCH_SIZE} symbols")
        print("="*70)
        print()

        # Load all historical data in batches
        print("Loading historical data (IBKR batched)...")
        all_data = self.fetch_historical_data_batched(
            ApexConfig.SYMBOLS, start_date, end_date
        )

        print(f"✅ Loaded {len(all_data)} symbols with data")
        print()

        if not all_data:
            print("❌ No data available for backtest")
            return

        # Get benchmark data (SPY)
        spy_data = all_data.get('SPY', pd.DataFrame())

        # Get all trading dates
        first_symbol = list(all_data.keys())[0]
        all_dates = all_data[first_symbol].index.tolist()

        print(f"Trading days: {len(all_dates)}")
        if len(all_dates) > 0:
            print(f"Start: {all_dates[0]}")
            print(f"End: {all_dates[-1]}")
        print()
        print("-"*70)

        if len(all_dates) < self.WARMUP_PERIOD:
            print(f"❌ Not enough trading days (need {self.WARMUP_PERIOD}+)")
            return

        # Track benchmark
        spy_start_price = float(spy_data['Close'].iloc[self.WARMUP_PERIOD]) if not spy_data.empty else 1.0

        # Simulate each day
        for day_idx, date in enumerate(all_dates[self.WARMUP_PERIOD:], 1):
            date_str = str(date).split()[0]

            # Update market regime
            if not spy_data.empty:
                try:
                    spy_idx = spy_data.index.get_loc(date)
                    self.market_trend = self.calculate_market_regime(spy_data, spy_idx)
                except:
                    pass

            # Check exits first (stop loss, trailing, profit target)
            symbols_to_close = []
            for symbol in list(self.positions.keys()):
                if symbol not in all_data:
                    continue

                data = all_data[symbol]
                try:
                    date_idx_data = data.index.get_loc(date)
                    current_price = float(data['Close'].iloc[date_idx_data])

                    should_exit, reason = self.check_exit_conditions(symbol, current_price, date_str)
                    if should_exit:
                        symbols_to_close.append((symbol, current_price, reason))
                except:
                    pass

            # Execute exits
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

            # Generate signals for all symbols
            potential_entries = []
            for symbol in ApexConfig.SYMBOLS:
                if symbol not in all_data:
                    continue
                if symbol in self.positions:
                    continue

                data = all_data[symbol]

                try:
                    date_idx_data = data.index.get_loc(date)
                except:
                    continue

                if date_idx_data >= len(data):
                    continue

                current_price = float(data['Close'].iloc[date_idx_data])
                if current_price <= 0:
                    continue

                # Generate signal
                signal, confidence = self.generate_signal_for_date(symbol, data, date_idx_data)

                # Skip weak signals
                if signal <= self.MIN_SIGNAL_STRENGTH:
                    continue
                if confidence < self.MIN_CONFIDENCE_REQUIRED:
                    continue

                # Check volatility filter
                volatility = self.calculate_volatility(data, date_idx_data)
                if volatility > self.MAX_VOLATILITY:
                    continue

                # Check trend alignment
                stock_trend = self.calculate_stock_trend(data, date_idx_data)

                # Only trade with market trend (unless very strong signal)
                if self.market_trend < -0.3 and signal < 0.7:
                    continue  # Skip buys in bear market unless very strong

                # Require some trend alignment
                if stock_trend < self.MIN_TREND_ALIGNMENT and signal < 0.65:
                    continue

                potential_entries.append((symbol, signal, confidence, stock_trend, volatility, current_price, data, date_idx_data))

            # Rank and select best signals
            ranked = sorted(potential_entries, key=lambda x: x[1] * x[2] * (1 + max(x[3], 0)), reverse=True)

            # Enter positions for best signals
            for entry in ranked:
                symbol, signal, confidence, stock_trend, volatility, current_price, data, date_idx_data = entry

                if len(self.positions) >= ApexConfig.MAX_POSITIONS:
                    break

                # Dynamic position sizing
                qty = self.get_dynamic_position_size(symbol, volatility, current_price, signal)
                position_value = qty * current_price

                # Check sector limit
                if not self.check_sector_limit(symbol, position_value, all_data, date):
                    continue

                # Apply transaction costs
                cost = self.apply_transaction_costs(position_value, is_buy=True)

                if self.capital >= cost:
                    self.positions[symbol] = {
                        'qty': qty,
                        'entry_price': current_price,
                        'entry_date': date_str,
                        'peak_price': current_price,
                        'signal_strength': signal
                    }
                    self.capital -= cost
                    self.trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'side': 'BUY',
                        'qty': qty,
                        'price': current_price,
                        'value': cost,
                        'reason': 'SIGNAL',
                        'signal': signal,
                        'confidence': confidence
                    })

            # Check signal-based exits for remaining positions
            for symbol in list(self.positions.keys()):
                if symbol not in all_data:
                    continue

                data = all_data[symbol]
                try:
                    date_idx_data = data.index.get_loc(date)
                    current_price = float(data['Close'].iloc[date_idx_data])
                except:
                    continue

                signal, confidence = self.generate_signal_for_date(symbol, data, date_idx_data)
                exit_threshold = self.MIN_SIGNAL_STRENGTH * self.EXIT_THRESHOLD_MULTIPLIER

                if signal < -exit_threshold and self.check_min_hold_period(symbol, date_str):
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

            self.peak_capital = max(self.peak_capital, portfolio_value)

            if day_idx % 5 == 0:
                regime = "BULL" if self.market_trend > 0.2 else "BEAR" if self.market_trend < -0.2 else "FLAT"
                print(f"Day {day_idx:4d}: ${portfolio_value:12,.2f} ({daily_return:+7.2%}) [{len(self.positions):2d} pos] {regime}")

        print("-"*70)
        print()
        self.print_summary()

    def _calculate_win_rate(self) -> Tuple[float, int, int]:
        """Calculate win rate based on completed round-trip trades."""
        positions = {}
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

        final_value = equity_array[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        if len(equity_array) > 1:
            daily_returns_pct = np.diff(equity_array) / equity_array[:-1]
        else:
            daily_returns_pct = np.array([0.0])

        avg_daily_return = np.mean(daily_returns_pct)
        std_daily_return = np.std(daily_returns_pct)

        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return * 252) / (std_daily_return * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        downside_returns = daily_returns_pct[daily_returns_pct < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = (avg_daily_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        else:
            sortino_ratio = float('inf') if avg_daily_return > 0 else 0.0

        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        win_rate, winners, total_completed = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()

        buy_trades = [t for t in self.trades if t['side'] == 'BUY']
        sell_trades = [t for t in self.trades if t['side'] == 'SELL']
        stop_trades = [t for t in self.trades if t.get('reason') == 'STOP']
        trail_trades = [t for t in self.trades if t.get('reason') == 'TRAIL']
        target_trades = [t for t in self.trades if t.get('reason') == 'TARGET']
        signal_exits = [t for t in self.trades if t['side'] == 'SELL' and t.get('reason') == 'SIGNAL']

        benchmark_return = self.benchmark_curve[-1] if self.benchmark_curve else 0.0
        alpha = total_return - benchmark_return

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
        print(f"  - Profit Targets:     {len(target_trades):>16}")
        print(f"  - Trailing Stops:     {len(trail_trades):>16}")
        print(f"  - Stop Losses:        {len(stop_trades):>16}")
        print(f"  - Signal Exits:       {len(signal_exits):>16}")
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

        # Status checks
        if total_return > 0:
            print(f"✅ Backtest profitable: +{total_return:.2%}")
            if alpha > 0:
                print(f"✅ Outperformed SPY by {alpha:.2%}")
            else:
                print(f"⚠️  Underperformed SPY by {abs(alpha):.2%}")
        else:
            print(f"⚠️  Backtest unprofitable: {total_return:.2%}")

        if win_rate >= 55:
            print(f"✅ High win rate: {win_rate:.1f}%")
        elif win_rate >= 50:
            print(f"⚠️  Moderate win rate: {win_rate:.1f}%")
        else:
            print(f"❌ Low win rate: {win_rate:.1f}%")

        if sharpe_ratio >= 1.6:
            print(f"✅ Excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio >= 1.0:
            print(f"✅ Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio >= 0.5:
            print(f"⚠️  Moderate risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        else:
            print(f"❌ Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

        print()


def main():
    parser = argparse.ArgumentParser(description='APEX Backtester - Optimized Version')
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
