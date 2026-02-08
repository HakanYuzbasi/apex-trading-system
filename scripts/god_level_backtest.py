#!/usr/bin/env python3
"""
scripts/god_level_backtest.py - God Level Backtesting Engine
Features:
- Walk-forward optimization
- Monte Carlo simulation
- Comprehensive risk metrics
- Transaction cost modeling
- Regime-aware performance analysis
"""

import sys
sys.path.insert(0, '/home/user/apex-trading-system')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
from core.logging_config import setup_logging
setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
logger = logging.getLogger(__name__)

# Import local modules
from config import ApexConfig
from data.market_data import MarketDataFetcher

# Try to import god-level modules, fall back to standard
try:
    from models.god_level_signal_generator import GodLevelSignalGenerator, MarketRegime
    from risk.god_level_risk_manager import GodLevelRiskManager, PositionRisk
    GOD_LEVEL_AVAILABLE = True
except ImportError:
    from models.advanced_signal_generator import AdvancedSignalGenerator as GodLevelSignalGenerator
    GOD_LEVEL_AVAILABLE = False
    logger.warning("God level modules not available, using standard modules")


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    direction: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    exit_reason: str
    regime: str
    hold_days: int
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class Position:
    """Active position during backtest."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    direction: str
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    highest_price: float
    lowest_price: float
    regime: str


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    # Returns
    total_return: float
    annualized_return: float
    benchmark_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    avg_hold_days: float

    # Advanced metrics
    expectancy: float
    sqn: float  # System Quality Number
    recovery_factor: float
    ulcer_index: float

    # Regime performance
    regime_performance: Dict[str, Dict]

    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series

    # Trades list
    trades: List[Trade]

    # Monte Carlo results
    monte_carlo: Optional[Dict] = None


class GodLevelBacktester:
    """
    God-level backtesting engine with walk-forward optimization
    and Monte Carlo simulation.
    """

    # Configuration
    WARMUP_DAYS = 60
    WALK_FORWARD_TRAIN_DAYS = 252  # 1 year training
    WALK_FORWARD_TEST_DAYS = 63   # 3 months testing

    # Transaction costs
    COMMISSION_PER_SHARE = 0.005  # $0.005 per share
    MIN_COMMISSION = 1.00         # $1 minimum per trade
    SLIPPAGE_BPS = 5              # 5 basis points slippage

    # Position management
    MAX_POSITIONS = 15
    MIN_POSITION_VALUE = 1000
    MAX_POSITION_PCT = 0.05

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital

        self.signal_generator = GodLevelSignalGenerator()
        self.market_data = MarketDataFetcher()

        if GOD_LEVEL_AVAILABLE:
            self.risk_manager = GodLevelRiskManager(
                initial_capital=initial_capital,
                max_position_pct=self.MAX_POSITION_PCT
            )
        else:
            self.risk_manager = None

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        logger.info(f"God Level Backtester initialized with ${initial_capital:,.0f}")
        logger.info(f"  God Level modules: {GOD_LEVEL_AVAILABLE}")

    def run_backtest(
        self,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        walk_forward: bool = True
    ) -> BacktestResult:
        """
        Run comprehensive backtest with optional walk-forward optimization.
        """
        symbols = symbols or ApexConfig.SYMBOLS[:50]  # Limit for speed

        logger.info("=" * 70)
        logger.info("GOD LEVEL BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Walk-forward: {walk_forward}")

        # Fetch historical data
        logger.info("\nFetching historical data...")
        historical_data = self._fetch_data(symbols)

        if not historical_data:
            logger.error("No data fetched!")
            return None

        logger.info(f"Loaded data for {len(historical_data)} symbols")

        # Get date range
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        if len(all_dates) < self.WARMUP_DAYS + 60:
            logger.error("Insufficient data for backtest")
            return None

        # Run backtest
        if walk_forward:
            result = self._run_walk_forward(historical_data, all_dates)
        else:
            result = self._run_single_period(historical_data, all_dates)

        return result

    def _fetch_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        data = {}

        for i, symbol in enumerate(symbols):
            try:
                df = self.market_data.fetch_historical_data(symbol, days=504)  # 2 years
                if df is not None and len(df) >= 252:
                    data[symbol] = df
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Fetched {i+1}/{len(symbols)} symbols...")
            except Exception as e:
                logger.debug(f"  Failed to fetch {symbol}: {e}")

        return data

    def _run_walk_forward(
        self,
        historical_data: Dict[str, pd.DataFrame],
        all_dates: List
    ) -> BacktestResult:
        """Run walk-forward optimization."""
        logger.info("\nRunning walk-forward optimization...")

        # Reset state
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Walk-forward periods
        total_days = len(all_dates)
        start_idx = self.WARMUP_DAYS

        period_num = 0
        while start_idx + self.WALK_FORWARD_TEST_DAYS < total_days:
            period_num += 1

            # Training period
            train_end_idx = start_idx
            train_start_idx = max(0, train_end_idx - self.WALK_FORWARD_TRAIN_DAYS)
            train_dates = all_dates[train_start_idx:train_end_idx]

            # Test period
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.WALK_FORWARD_TEST_DAYS, total_days)
            test_dates = all_dates[test_start_idx:test_end_idx]

            logger.info(f"\n  Period {period_num}: Train {train_dates[0].strftime('%Y-%m-%d') if hasattr(train_dates[0], 'strftime') else train_dates[0]} -> Test {test_dates[-1].strftime('%Y-%m-%d') if hasattr(test_dates[-1], 'strftime') else test_dates[-1]}")

            # Train models on training period
            train_data = {}
            for symbol, df in historical_data.items():
                mask = df.index.isin(train_dates)
                if mask.sum() >= 60:
                    train_data[symbol] = df[mask]

            if train_data:
                self.signal_generator.train_models(train_data)

            # Test on test period
            self._simulate_period(historical_data, test_dates)

            # Move to next period
            start_idx = test_end_idx

        # Calculate final results
        return self._calculate_results(historical_data)

    def _run_single_period(
        self,
        historical_data: Dict[str, pd.DataFrame],
        all_dates: List
    ) -> BacktestResult:
        """Run single-period backtest (train once, test once)."""
        logger.info("\nRunning single-period backtest...")

        # Reset state
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Train on first half
        mid_point = len(all_dates) // 2
        train_dates = all_dates[:mid_point]
        test_dates = all_dates[mid_point:]

        # Train models
        train_data = {}
        for symbol, df in historical_data.items():
            mask = df.index.isin(train_dates)
            if mask.sum() >= 60:
                train_data[symbol] = df[mask]

        if train_data:
            self.signal_generator.train_models(train_data)

        # Test
        self._simulate_period(historical_data, test_dates)

        return self._calculate_results(historical_data)

    def _simulate_period(self, historical_data: Dict[str, pd.DataFrame], dates: List):
        """Simulate trading over a period."""
        for date in dates:
            # Get prices for this date
            day_data = {}
            for symbol, df in historical_data.items():
                if date in df.index:
                    day_data[symbol] = df.loc[date]

            if not day_data:
                continue

            # Update existing positions
            self._update_positions(day_data, date)

            # Generate new signals
            for symbol, row in day_data.items():
                if symbol in self.positions:
                    continue

                if len(self.positions) >= self.MAX_POSITIONS:
                    continue

                # Get historical prices for signal generation
                df = historical_data[symbol]
                mask = df.index <= date
                if mask.sum() < 60:
                    continue

                prices = df.loc[mask, 'Close']

                # Generate signal
                signal_data = self.signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                regime = signal_data.get('regime', 'neutral')

                # Entry criteria
                if abs(signal) >= 0.40 and confidence >= 0.35:
                    self._enter_position(symbol, row, signal, confidence, regime, date, prices)

            # Record equity
            portfolio_value = self._calculate_portfolio_value(day_data)
            self.equity_curve.append((date, portfolio_value))
            self.peak_capital = max(self.peak_capital, portfolio_value)

    def _update_positions(self, day_data: Dict, date: datetime):
        """Update positions and check exit conditions."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in day_data:
                continue

            row = day_data[symbol]
            current_price = row['Close']
            high = row.get('High', current_price)
            low = row.get('Low', current_price)

            # Update tracking
            position.highest_price = max(position.highest_price, high)
            position.lowest_price = min(position.lowest_price, low)

            # Calculate P&L
            if position.direction == 'long':
                pnl_pct = (current_price / position.entry_price - 1)
            else:
                pnl_pct = (position.entry_price / current_price - 1)

            # Check exit conditions
            exit_signal = False
            exit_reason = ""

            # 1. Stop loss
            if position.direction == 'long' and current_price <= position.stop_loss:
                exit_signal = True
                exit_reason = "Stop loss"
            elif position.direction == 'short' and current_price >= position.stop_loss:
                exit_signal = True
                exit_reason = "Stop loss"

            # 2. Take profit
            if position.direction == 'long' and current_price >= position.take_profit:
                exit_signal = True
                exit_reason = "Take profit"
            elif position.direction == 'short' and current_price <= position.take_profit:
                exit_signal = True
                exit_reason = "Take profit"

            # 3. Trailing stop
            if pnl_pct > 0.02:  # Only after 2% profit
                if position.direction == 'long':
                    new_stop = position.highest_price * (1 - position.trailing_stop_pct)
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                    if current_price <= position.stop_loss:
                        exit_signal = True
                        exit_reason = "Trailing stop"

            # 4. Time-based exit
            hold_days = (date - position.entry_date).days if hasattr(date, 'days') else 1
            if hold_days >= 45:  # Max 45 days
                exit_signal = True
                exit_reason = "Max hold period"

            if exit_signal:
                positions_to_close.append((symbol, current_price, exit_reason, date))

        # Close positions
        for symbol, exit_price, exit_reason, exit_date in positions_to_close:
            self._exit_position(symbol, exit_price, exit_reason, exit_date)

    def _enter_position(
        self,
        symbol: str,
        row: pd.Series,
        signal: float,
        confidence: float,
        regime: str,
        date: datetime,
        prices: pd.Series
    ):
        """Enter a new position."""
        entry_price = row['Close']

        # Apply slippage
        slippage = entry_price * (self.SLIPPAGE_BPS / 10000)
        if signal > 0:
            entry_price += slippage  # Buy higher
        else:
            entry_price -= slippage  # Sell lower

        # Calculate position size
        if self.risk_manager:
            sizing = self.risk_manager.calculate_position_size(
                symbol, entry_price, signal, confidence, prices, regime
            )
            shares = sizing['shares']
            stop_loss = sizing['stop_loss']
            take_profit = sizing['take_profit']
            trailing_stop_pct = sizing['trailing_stop_pct']
        else:
            # Simple sizing
            position_value = min(self.capital * 0.05, self.capital / self.MAX_POSITIONS)
            shares = int(position_value / entry_price)
            atr_pct = 0.02
            stop_loss = entry_price * (1 - 2 * atr_pct) if signal > 0 else entry_price * (1 + 2 * atr_pct)
            take_profit = entry_price * (1 + 3 * atr_pct) if signal > 0 else entry_price * (1 - 3 * atr_pct)
            trailing_stop_pct = 0.03

        if shares <= 0:
            return

        position_value = shares * entry_price

        # Check if we have enough capital
        if position_value > self.capital * 0.95:
            return

        # Apply commission
        commission = max(shares * self.COMMISSION_PER_SHARE, self.MIN_COMMISSION)

        # Create position
        direction = 'long' if signal > 0 else 'short'
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            highest_price=entry_price,
            lowest_price=entry_price,
            regime=regime
        )

        self.positions[symbol] = position
        self.capital -= (position_value + commission)

    def _exit_position(self, symbol: str, exit_price: float, exit_reason: str, exit_date: datetime):
        """Exit a position and record the trade."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Apply slippage
        slippage = exit_price * (self.SLIPPAGE_BPS / 10000)
        if position.direction == 'long':
            exit_price -= slippage  # Sell lower
        else:
            exit_price += slippage  # Buy higher (cover short)

        # Calculate P&L
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * position.shares
            pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:
            pnl = (position.entry_price - exit_price) * position.shares
            pnl_pct = (position.entry_price / exit_price - 1) * 100

        # Apply commission
        commission = max(position.shares * self.COMMISSION_PER_SHARE, self.MIN_COMMISSION)
        pnl -= commission * 2  # Entry + exit commission

        # Calculate hold days
        if hasattr(exit_date, 'days'):
            hold_days = (exit_date - position.entry_date).days
        else:
            hold_days = 1

        # Calculate excursions
        if position.direction == 'long':
            max_favorable = (position.highest_price / position.entry_price - 1) * 100
            max_adverse = (position.lowest_price / position.entry_price - 1) * 100
        else:
            max_favorable = (position.entry_price / position.lowest_price - 1) * 100
            max_adverse = (position.entry_price / position.highest_price - 1) * 100

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            direction=position.direction,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=exit_reason,
            regime=position.regime,
            hold_days=hold_days,
            max_favorable=max_favorable,
            max_adverse=max_adverse
        )
        self.trades.append(trade)

        # Update capital
        self.capital += (exit_price * position.shares)

        # Remove position
        del self.positions[symbol]

        # Update risk manager
        if self.risk_manager:
            self.risk_manager.record_trade(pnl, pnl > 0)

    def _calculate_portfolio_value(self, day_data: Dict) -> float:
        """Calculate total portfolio value."""
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in day_data:
                current_price = day_data[symbol]['Close']
                positions_value += current_price * position.shares

        return self.capital + positions_value

    def _calculate_results(self, historical_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING RESULTS")
        logger.info("=" * 70)

        if not self.equity_curve:
            logger.error("No equity curve data!")
            return None

        # Convert equity curve to series
        dates, values = zip(*self.equity_curve)
        equity_series = pd.Series(values, index=dates)

        # Calculate returns
        daily_returns = equity_series.pct_change().dropna()
        total_return = (equity_series.iloc[-1] / self.initial_capital - 1) * 100

        # Annualized return
        trading_days = len(equity_series)
        annualized_return = ((1 + total_return / 100) ** (252 / trading_days) - 1) * 100 if trading_days > 0 else 0

        # Benchmark return (SPY)
        benchmark_return = self._calculate_benchmark_return(historical_data, dates)

        # Drawdown calculation
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        # Average drawdown
        avg_drawdown = abs(drawdown[drawdown < 0].mean()) * 100 if len(drawdown[drawdown < 0]) > 0 else 0

        # Max drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        for dd in is_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        # Risk metrics
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))

            # Sortino (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else daily_returns.std() * np.sqrt(252)
            sortino_ratio = (daily_returns.mean() * 252) / downside_std if downside_std > 0 else 0

            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0

        # Trade metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean([t.pnl for t in self.trades]) if self.trades else 0

        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        avg_hold_days = np.mean([t.hold_days for t in self.trades]) if self.trades else 0

        # Advanced metrics
        # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * abs(avg_loss)) if total_trades > 0 else 0

        # SQN (System Quality Number) = sqrt(N) * Expectancy / StdDev(Trade P&L)
        if total_trades > 0:
            trade_pnls = [t.pnl for t in self.trades]
            sqn = np.sqrt(total_trades) * np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0
        else:
            sqn = 0

        # Recovery factor = Total Return / Max Drawdown
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

        # Ulcer Index
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean()) * 100

        # Regime performance
        regime_performance = self._calculate_regime_performance()

        logger.info(f"\nTotal Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Total Trades: {total_trades}")

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_hold_days=avg_hold_days,
            expectancy=expectancy,
            sqn=sqn,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            regime_performance=regime_performance,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            trades=self.trades
        )

    def _calculate_benchmark_return(self, historical_data: Dict, dates) -> float:
        """Calculate SPY benchmark return."""
        if 'SPY' not in historical_data:
            return 0.0

        spy_data = historical_data['SPY']
        start_date = dates[0]
        end_date = dates[-1]

        if start_date in spy_data.index and end_date in spy_data.index:
            return (spy_data.loc[end_date, 'Close'] / spy_data.loc[start_date, 'Close'] - 1) * 100

        return 0.0

    def _calculate_regime_performance(self) -> Dict[str, Dict]:
        """Calculate performance by market regime."""
        regime_trades = defaultdict(list)

        for trade in self.trades:
            regime_trades[trade.regime].append(trade)

        regime_perf = {}
        for regime, trades in regime_trades.items():
            wins = [t for t in trades if t.pnl > 0]
            regime_perf[regime] = {
                'trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'avg_pnl': np.mean([t.pnl for t in trades]) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades)
            }

        return regime_perf

    def run_monte_carlo(self, n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation on trade results.
        Shuffles trade order to estimate distribution of outcomes.
        """
        if len(self.trades) < 10:
            logger.warning("Not enough trades for Monte Carlo simulation")
            return None

        logger.info(f"\nRunning Monte Carlo simulation ({n_simulations} iterations)...")

        trade_pnls = [t.pnl for t in self.trades]
        n_trades = len(trade_pnls)

        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Shuffle trades
            shuffled = np.random.permutation(trade_pnls)

            # Calculate equity curve
            equity = [self.initial_capital]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl)

            equity = np.array(equity)

            # Final return
            final_return = (equity[-1] / self.initial_capital - 1) * 100
            final_returns.append(final_return)

            # Max drawdown
            rolling_max = np.maximum.accumulate(equity)
            drawdown = (equity - rolling_max) / rolling_max
            max_dd = abs(drawdown.min()) * 100
            max_drawdowns.append(max_dd)

            # Sharpe (approximate)
            daily_equiv = np.diff(equity) / equity[:-1]
            if daily_equiv.std() > 0:
                sharpe = daily_equiv.mean() / daily_equiv.std() * np.sqrt(252 / n_trades)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        results = {
            'return_mean': np.mean(final_returns),
            'return_std': np.std(final_returns),
            'return_5th': np.percentile(final_returns, 5),
            'return_95th': np.percentile(final_returns, 95),
            'max_dd_mean': np.mean(max_drawdowns),
            'max_dd_95th': np.percentile(max_drawdowns, 95),
            'sharpe_mean': np.mean(sharpe_ratios),
            'sharpe_5th': np.percentile(sharpe_ratios, 5),
            'prob_positive': sum(1 for r in final_returns if r > 0) / n_simulations * 100,
            'prob_beat_benchmark': sum(1 for r in final_returns if r > 10) / n_simulations * 100
        }

        logger.info(f"  Monte Carlo Results:")
        logger.info(f"    Expected Return: {results['return_mean']:.1f}% (5th-95th: {results['return_5th']:.1f}% to {results['return_95th']:.1f}%)")
        logger.info(f"    Expected Max DD: {results['max_dd_mean']:.1f}% (95th percentile: {results['max_dd_95th']:.1f}%)")
        logger.info(f"    Probability of Profit: {results['prob_positive']:.1f}%")

        return results


def print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("GOD LEVEL BACKTEST RESULTS")
    print("=" * 70)

    print("\n📈 RETURNS")
    print("-" * 40)
    print(f"  Total Return:       {result.total_return:>10.2f}%")
    print(f"  Annualized Return:  {result.annualized_return:>10.2f}%")
    print(f"  Benchmark (SPY):    {result.benchmark_return:>10.2f}%")
    print(f"  Alpha:              {result.total_return - result.benchmark_return:>10.2f}%")

    print("\n⚖️  RISK METRICS")
    print("-" * 40)
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:>10.2f}")
    print(f"  Max Drawdown:       {result.max_drawdown:>10.2f}%")
    print(f"  Avg Drawdown:       {result.avg_drawdown:>10.2f}%")
    print(f"  Max DD Duration:    {result.max_drawdown_duration:>10d} days")
    print(f"  Ulcer Index:        {result.ulcer_index:>10.2f}")

    print("\n🎯 TRADE METRICS")
    print("-" * 40)
    print(f"  Total Trades:       {result.total_trades:>10d}")
    print(f"  Win Rate:           {result.win_rate:>10.1f}%")
    print(f"  Profit Factor:      {result.profit_factor:>10.2f}")
    print(f"  Avg Win:            ${result.avg_win:>9.2f}")
    print(f"  Avg Loss:           ${result.avg_loss:>9.2f}")
    print(f"  Avg Trade:          ${result.avg_trade:>9.2f}")
    print(f"  Largest Win:        ${result.largest_win:>9.2f}")
    print(f"  Largest Loss:       ${result.largest_loss:>9.2f}")
    print(f"  Avg Hold Days:      {result.avg_hold_days:>10.1f}")

    print("\n🧠 ADVANCED METRICS")
    print("-" * 40)
    print(f"  Expectancy:         ${result.expectancy:>9.2f}")
    print(f"  SQN:                {result.sqn:>10.2f}")
    print(f"  Recovery Factor:    {result.recovery_factor:>10.2f}")

    if result.regime_performance:
        print("\n📊 REGIME PERFORMANCE")
        print("-" * 40)
        for regime, perf in sorted(result.regime_performance.items()):
            print(f"  {regime:20s} | Trades: {perf['trades']:3d} | Win Rate: {perf['win_rate']:5.1f}% | P&L: ${perf['total_pnl']:,.0f}")

    if result.monte_carlo:
        print("\n🎲 MONTE CARLO SIMULATION")
        print("-" * 40)
        mc = result.monte_carlo
        print(f"  Expected Return:    {mc['return_mean']:>10.1f}%")
        print(f"  Return Range (90%): {mc['return_5th']:>6.1f}% to {mc['return_95th']:.1f}%")
        print(f"  Expected Max DD:    {mc['max_dd_mean']:>10.1f}%")
        print(f"  Prob of Profit:     {mc['prob_positive']:>10.1f}%")

    # Quality assessment
    print("\n" + "=" * 70)
    print("SYSTEM QUALITY ASSESSMENT")
    print("=" * 70)

    score = 0
    assessments = []

    if result.sharpe_ratio >= 2.0:
        score += 25
        assessments.append("Excellent risk-adjusted returns (Sharpe >= 2.0)")
    elif result.sharpe_ratio >= 1.5:
        score += 20
        assessments.append("Great risk-adjusted returns (Sharpe >= 1.5)")
    elif result.sharpe_ratio >= 1.0:
        score += 15
        assessments.append("Good risk-adjusted returns (Sharpe >= 1.0)")

    if result.win_rate >= 60:
        score += 25
        assessments.append("Excellent win rate (>= 60%)")
    elif result.win_rate >= 55:
        score += 20
        assessments.append("Great win rate (>= 55%)")
    elif result.win_rate >= 50:
        score += 15
        assessments.append("Decent win rate (>= 50%)")

    if result.max_drawdown <= 10:
        score += 25
        assessments.append("Excellent drawdown control (<= 10%)")
    elif result.max_drawdown <= 15:
        score += 20
        assessments.append("Good drawdown control (<= 15%)")
    elif result.max_drawdown <= 20:
        score += 15
        assessments.append("Acceptable drawdown (<= 20%)")

    if result.profit_factor >= 2.0:
        score += 25
        assessments.append("Excellent profit factor (>= 2.0)")
    elif result.profit_factor >= 1.5:
        score += 20
        assessments.append("Good profit factor (>= 1.5)")
    elif result.profit_factor >= 1.2:
        score += 15
        assessments.append("Decent profit factor (>= 1.2)")

    for assessment in assessments:
        print(f"  ✓ {assessment}")

    print(f"\n  TOTAL SCORE: {score}/100")

    if score >= 90:
        print("  🏆 GOD LEVEL ACHIEVED!")
    elif score >= 75:
        print("  ⭐ EXCELLENT SYSTEM")
    elif score >= 60:
        print("  👍 GOOD SYSTEM")
    elif score >= 40:
        print("  📈 NEEDS IMPROVEMENT")
    else:
        print("  ⚠️  SIGNIFICANT WORK NEEDED")

    print("=" * 70)


def main():
    """Run god level backtest."""
    print("\n" + "=" * 70)
    print("STARTING GOD LEVEL BACKTEST")
    print("=" * 70)

    # Create backtester
    backtester = GodLevelBacktester(initial_capital=100000)

    # Run backtest with walk-forward optimization
    result = backtester.run_backtest(
        symbols=ApexConfig.SYMBOLS[:40],  # Use 40 symbols for speed
        walk_forward=True
    )

    if result:
        # Run Monte Carlo simulation
        mc_results = backtester.run_monte_carlo(n_simulations=1000)
        result.monte_carlo = mc_results

        # Print results
        print_results(result)

        return result

    return None


if __name__ == "__main__":
    result = main()
