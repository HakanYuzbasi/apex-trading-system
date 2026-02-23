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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
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
from core.symbols import AssetClass, parse_symbol

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
    entry_commission: float
    highest_price: float
    lowest_price: float
    regime: str
    took_partial_profit: bool = False


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
    WALK_FORWARD_TEST_DAYS = 126   # 6 months testing

    # Transaction costs
    COMMISSION_PER_SHARE = 0.005  # $0.005 per share
    MIN_COMMISSION = 1.00         # $1 minimum per trade
    SLIPPAGE_BPS = 5              # 5 basis points slippage

    # Position management
    MAX_POSITIONS = 15
    MIN_POSITION_VALUE = 1000
    MAX_POSITION_PCT = 0.05
    MAX_QUALITY_POSITION_PCT = 0.08
    MAX_GROSS_EXPOSURE = 1.40
    MAX_NEW_ENTRIES_PER_DAY = 6
    HARD_DRAWDOWN_ENTRY_PAUSE = 0.18
    PARTIAL_TAKE_PROFIT_PCT = 0.08
    PARTIAL_EXIT_FRACTION = 0.50
    RUNNER_TRAIL_TIGHTENING = 0.70
    FORCE_FLAT_AT_PERIOD_END = True

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
            self.risk_manager.set_sector_map(ApexConfig.SECTOR_MAP)
        else:
            self.risk_manager = None

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        logger.info(f"God Level Backtester initialized with ${initial_capital:,.0f}")
        logger.info(f"  God Level modules: {GOD_LEVEL_AVAILABLE}")

    @staticmethod
    def _asset_class_for_symbol(symbol: str) -> AssetClass:
        try:
            return parse_symbol(symbol).asset_class
        except ValueError:
            return AssetClass.EQUITY

    def _slippage_bps_for_asset(
        self,
        asset_class: AssetClass,
        prices: Optional[pd.Series] = None
    ) -> float:
        if asset_class == AssetClass.FOREX:
            base_bps = float(ApexConfig.FX_SPREAD_BPS)
            max_bps = 40.0
        elif asset_class == AssetClass.CRYPTO:
            base_bps = float(ApexConfig.CRYPTO_SPREAD_BPS)
            max_bps = 80.0
        else:
            base_bps = float(self.SLIPPAGE_BPS)
            max_bps = 35.0

        if prices is None or len(prices) < 20:
            return base_bps

        returns = prices.pct_change().dropna().tail(20)
        if returns.empty:
            return base_bps
        vol = float(returns.std())
        vol_multiplier = min(2.0, max(0.0, vol * 40.0))
        return float(min(max_bps, base_bps * (1.0 + vol_multiplier)))

    def _commission_for_order(
        self,
        symbol: str,
        shares: int,
        price: float,
        asset_class: Optional[AssetClass] = None
    ) -> float:
        asset = asset_class or self._asset_class_for_symbol(symbol)
        notional = abs(float(shares) * float(price))
        if asset == AssetClass.EQUITY:
            return max(abs(int(shares)) * self.COMMISSION_PER_SHARE, self.MIN_COMMISSION)
        if asset == AssetClass.FOREX:
            return notional * (float(ApexConfig.FX_COMMISSION_BPS) / 10000.0)
        return notional * (float(ApexConfig.CRYPTO_COMMISSION_BPS) / 10000.0)

    def _edge_over_cost_passes(
        self,
        symbol: str,
        asset_class: AssetClass,
        signal: float,
        confidence: float,
        shares: int,
        price: float,
        prices: Optional[pd.Series],
    ) -> bool:
        notional = abs(float(shares) * float(price))
        if notional <= 0:
            return False

        slippage_bps = self._slippage_bps_for_asset(asset_class, prices)
        commission = self._commission_for_order(
            symbol=symbol,
            shares=shares,
            price=price,
            asset_class=asset_class,
        )
        commission_bps = (commission / notional) * 10000.0
        expected_cost_bps = slippage_bps + commission_bps

        if asset_class == AssetClass.FOREX:
            edge_buffer_bps = float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_FX)
        elif asset_class == AssetClass.CRYPTO:
            edge_buffer_bps = float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO)
        else:
            edge_buffer_bps = float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY)

        signal_to_edge_bps = float(getattr(ApexConfig, "EXECUTION_SIGNAL_TO_EDGE_BPS", 80.0))
        expected_edge_bps = abs(float(signal)) * signal_to_edge_bps * (0.4 + 0.6 * float(confidence))
        required_edge_bps = expected_cost_bps + edge_buffer_bps
        return bool(expected_edge_bps + 1e-9 >= required_edge_bps)

    @staticmethod
    def _passes_broker_min_notional(asset_class: AssetClass, notional: float) -> bool:
        if asset_class == AssetClass.FOREX:
            return notional >= float(ApexConfig.IBKR_MIN_FX_NOTIONAL)
        if asset_class == AssetClass.CRYPTO:
            return notional >= float(ApexConfig.IBKR_MIN_CRYPTO_NOTIONAL)
        return True

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
        historical_data = self._fetch_data(symbols, start_date=start_date, end_date=end_date)

        if not historical_data:
            logger.error("No data fetched!")
            return None

        logger.info(f"Loaded data for {len(historical_data)} symbols")

        # Get date range
        all_dates = set()
        for df in historical_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        min_required = self.WALK_FORWARD_TRAIN_DAYS + self.WALK_FORWARD_TEST_DAYS
        if len(all_dates) < min_required:
            logger.error(
                "Insufficient data for backtest: %d days available, need %d",
                len(all_dates), min_required,
            )
            return None

        # Run backtest
        if walk_forward:
            result = self._run_walk_forward(historical_data, all_dates)
        else:
            result = self._run_single_period(historical_data, all_dates)

        return result

    def _fetch_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        data = {}
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None

        for i, symbol in enumerate(symbols):
            try:
                df = self.market_data.fetch_historical_data(symbol, days=504)  # 2 years
                if df is None or df.empty:
                    continue
                if start_ts is not None:
                    df = df[df.index >= start_ts]
                if end_ts is not None:
                    df = df[df.index <= end_ts]
                if len(df) >= 252:
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
        if self.risk_manager:
            self.risk_manager.positions = {}
            self.risk_manager.daily_pnl = []
            self.risk_manager.update_capital(self.initial_capital)

        # Walk-forward periods
        # start_idx = first test-period start.  The training window preceding it
        # is dates[max(0, start_idx - WALK_FORWARD_TRAIN_DAYS) : start_idx].
        # We need enough training bars (>= lookback + forward_window + margin)
        # for the signal generator to produce samples.  Using
        # WALK_FORWARD_TRAIN_DAYS guarantees a full-size first training window.
        total_days = len(all_dates)
        start_idx = max(self.WARMUP_DAYS, self.WALK_FORWARD_TRAIN_DAYS)

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
                if self.risk_manager:
                    self.risk_manager.update_correlation_matrix(train_data)

            # Test on test period
            trades_before = len(self.trades)
            self._simulate_period(historical_data, test_dates)
            trades_after = len(self.trades)
            period_trades = trades_after - trades_before
            logger.info(f"  Period {period_num} produced {period_trades} trades")
            if period_trades < 5:
                logger.warning(f"  ⚠️ Period {period_num} generated fewer than 5 trades, its contribution is noise.")

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
        if self.risk_manager:
            self.risk_manager.positions = {}
            self.risk_manager.daily_pnl = []
            self.risk_manager.update_capital(self.initial_capital)

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
            if self.risk_manager:
                self.risk_manager.update_correlation_matrix(train_data)

        # Test
        self._simulate_period(historical_data, test_dates)

        return self._calculate_results(historical_data)

    def _simulate_period(self, historical_data: Dict[str, pd.DataFrame], dates: List):
        """Simulate trading over a period."""
        last_date: Optional[datetime] = None
        last_day_data: Dict[str, pd.Series] = {}
        for date in dates:
            last_date = date
            # Get prices for this date
            day_data = {}
            for symbol, df in historical_data.items():
                if date in df.index:
                    day_data[symbol] = df.loc[date]
            last_day_data = day_data

            if not day_data:
                continue

            # Update existing positions
            self._update_positions(day_data, date)
            current_portfolio_value = self._calculate_portfolio_value(day_data)
            current_drawdown = (
                (self.peak_capital - current_portfolio_value) / self.peak_capital
                if self.peak_capital > 0 else 0.0
            )

            # Generate new signals
            daily_entries = 0
            for symbol, row in day_data.items():
                if daily_entries >= self.MAX_NEW_ENTRIES_PER_DAY:
                    break
                if symbol in self.positions:
                    continue

                if len(self.positions) >= self.MAX_POSITIONS:
                    continue

                if current_drawdown >= self.HARD_DRAWDOWN_ENTRY_PAUSE:
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
                if not self._passes_directional_filter(prices, signal, regime):
                    continue
                if not self._should_enter_trade(signal, confidence, regime, current_drawdown):
                    continue

                entered = self._enter_position(
                    symbol, row, signal, confidence, regime, date, prices, current_portfolio_value
                )
                if entered:
                    daily_entries += 1

            # Record equity
            portfolio_value = self._calculate_portfolio_value(day_data)
            self.equity_curve.append((date, portfolio_value))
            self.peak_capital = max(self.peak_capital, portfolio_value)
            if self.risk_manager:
                self.risk_manager.update_capital(portfolio_value)

        # Flatten at period boundary so trade metrics reflect realized quality.
        if self.FORCE_FLAT_AT_PERIOD_END and last_date is not None and self.positions:
            self._force_flatten_positions(last_day_data, last_date)
            final_value = self._calculate_portfolio_value(last_day_data)
            self.equity_curve.append((pd.Timestamp(last_date) + pd.Timedelta(seconds=1), final_value))
            self.peak_capital = max(self.peak_capital, final_value)
            if self.risk_manager:
                self.risk_manager.update_capital(final_value)

    def _force_flatten_positions(self, day_data: Dict[str, pd.Series], date: datetime):
        """Close all open positions at the end of a simulation period."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if symbol in day_data:
                exit_price = float(day_data[symbol].get('Close', position.entry_price))
            else:
                exit_price = float(position.entry_price)
            self._exit_position(symbol, exit_price, "Period end flatten", date)

    def _should_enter_trade(
        self,
        signal: float,
        confidence: float,
        regime: str,
        current_drawdown: float
    ) -> bool:
        """Apply adaptive entry thresholds by regime and drawdown state."""
        thresholds = {
            'strong_bull': (0.34, 0.42),
            'bull': (0.37, 0.46),
            'neutral': (0.43, 0.53),
            'bear': (0.45, 0.58),
            'strong_bear': (0.50, 0.60),
            'high_volatility': (0.55, 0.62),
        }
        signal_threshold, confidence_threshold = thresholds.get(regime, (0.50, 0.65))
        if current_drawdown >= 0.08:
            signal_threshold += 0.05
            confidence_threshold += 0.04
        return abs(signal) >= signal_threshold and confidence >= confidence_threshold

    def _passes_directional_filter(self, prices: pd.Series, signal: float, regime: str) -> bool:
        """Avoid trading against persistent trend/regime unless the setup is strong."""
        if len(prices) < 50 or signal == 0:
            return True

        ma20 = prices.iloc[-20:].mean()
        ma50 = prices.iloc[-50:].mean()
        trend_bias = (ma20 / ma50 - 1) if ma50 > 0 else 0.0

        long_allowed = trend_bias >= -0.005
        short_allowed = trend_bias <= 0.005

        if regime == 'strong_bull':
            short_allowed = False
        elif regime == 'bull':
            # In bull regime allow only very strong reversal shorts.
            short_allowed = abs(signal) >= 0.85 and trend_bias < -0.015
        elif regime in {'strong_bear', 'bear'}:
            long_allowed = False

        return long_allowed if signal > 0 else short_allowed

    def _max_open_correlation(self, symbol: str) -> Optional[float]:
        """Return max absolute correlation of candidate vs open positions."""
        if not self.risk_manager or not self.positions:
            return 0.0
        corr = self.risk_manager.correlation_matrix
        if corr is None or symbol not in corr.columns:
            return None
        vals = []
        for existing in self.positions.keys():
            if existing in corr.columns and existing != symbol:
                try:
                    vals.append(abs(float(corr.loc[symbol, existing])))
                except Exception:
                    continue
        return max(vals) if vals else 0.0

    def _position_size_multiplier(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        regime: str
    ) -> float:
        """Boost best setups while shrinking crowded/high-correlation entries."""
        mult = 1.0
        if confidence >= 0.72 and abs(signal) >= 0.62:
            mult += 0.15
        if regime == 'strong_bull' and signal > 0:
            mult += 0.15
        elif regime == 'bull' and signal > 0:
            mult += 0.10

        max_corr = self._max_open_correlation(symbol)
        if max_corr is not None:
            if max_corr <= 0.35:
                mult += 0.15
            elif max_corr >= 0.75:
                mult -= 0.20
        return float(np.clip(mult, 0.75, 1.60))

    def _dynamic_gross_exposure_cap(self, current_drawdown: float) -> float:
        """Scale gross exposure down as drawdown deepens."""
        if current_drawdown <= 0.04:
            return self.MAX_GROSS_EXPOSURE
        if current_drawdown <= 0.08:
            return min(self.MAX_GROSS_EXPOSURE, 1.30)
        if current_drawdown <= 0.12:
            return 1.15
        return 1.00

    def _update_positions(self, day_data: Dict, date: datetime):
        """Update positions and check exit conditions."""
        positions_to_close = []

        for symbol, position in list(self.positions.items()):
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

            take_profit_trigger = (
                (position.direction == 'long' and current_price >= position.take_profit)
                or (position.direction == 'short' and current_price <= position.take_profit)
            )

            # Scale out half near target, keep a trailing runner for trend capture.
            if (
                not position.took_partial_profit
                and position.shares > 1
                and (take_profit_trigger or pnl_pct >= self.PARTIAL_TAKE_PROFIT_PCT)
            ):
                self._execute_partial_exit(
                    symbol=symbol,
                    market_price=current_price,
                    exit_reason="Partial take profit",
                    exit_date=date,
                    fraction=self.PARTIAL_EXIT_FRACTION,
                )
                if symbol not in self.positions:
                    continue
                position = self.positions[symbol]
                position.took_partial_profit = True
                position.trailing_stop_pct = max(
                    0.01, position.trailing_stop_pct * self.RUNNER_TRAIL_TIGHTENING
                )
                if position.direction == 'long':
                    position.stop_loss = max(position.stop_loss, position.entry_price * 1.01)
                else:
                    position.stop_loss = min(position.stop_loss, position.entry_price * 0.99)

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

            # 2. Trailing stop
            if pnl_pct > 0.02:  # Only after 2% profit
                if position.direction == 'long':
                    new_stop = position.highest_price * (1 - position.trailing_stop_pct)
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                    if current_price <= position.stop_loss:
                        exit_signal = True
                        exit_reason = "Trailing stop"
                else:
                    new_stop = position.lowest_price * (1 + position.trailing_stop_pct)
                    if new_stop < position.stop_loss:
                        position.stop_loss = new_stop
                    if current_price >= position.stop_loss:
                        exit_signal = True
                        exit_reason = "Trailing stop"

            # 3. Time-based exit
            hold_days = max((pd.Timestamp(date) - pd.Timestamp(position.entry_date)).days, 0)
            if hold_days >= 90:  # Max 90 days
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
        prices: pd.Series,
        current_equity: float
    ) -> bool:
        """Enter a new position."""
        entry_price = row['Close']
        asset_class = self._asset_class_for_symbol(symbol)

        # Apply slippage
        entry_slippage_bps = self._slippage_bps_for_asset(asset_class, prices)
        slippage = entry_price * (entry_slippage_bps / 10000.0)
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
            stop_loss = entry_price * (1 - 0.05) if signal > 0 else entry_price * (1 + 0.05)
            take_profit = entry_price * (1 + 0.15) if signal > 0 else entry_price * (1 - 0.15)
            trailing_stop_pct = 0.03

        if shares <= 0:
            return False

        dynamic_max_position_pct = self.MAX_POSITION_PCT
        if self.risk_manager:
            size_mult = self._position_size_multiplier(symbol, signal, confidence, regime)
            shares = max(1, int(shares * size_mult))
            dynamic_max_position_pct = min(
                self.MAX_QUALITY_POSITION_PCT,
                self.MAX_POSITION_PCT * size_mult,
            )

        max_shares_by_value = int((self.capital * dynamic_max_position_pct) / entry_price) if entry_price > 0 else 0
        if max_shares_by_value <= 0:
            return False
        shares = min(shares, max_shares_by_value)
        if shares <= 0:
            return False

        position_value = shares * entry_price
        if position_value < self.MIN_POSITION_VALUE:
            return False

        # Exposure guard (gross notional exposure vs equity)
        gross_exposure = self._calculate_gross_exposure(day_data=None)
        current_drawdown = (
            (self.peak_capital - current_equity) / self.peak_capital
            if self.peak_capital > 0 else 0.0
        )
        gross_cap = self._dynamic_gross_exposure_cap(current_drawdown)
        if current_equity > 0 and gross_exposure + position_value > current_equity * gross_cap:
            return False

        # Apply commission
        if not self._passes_broker_min_notional(asset_class, position_value):
            return False

        if not self._edge_over_cost_passes(
            symbol=symbol,
            asset_class=asset_class,
            signal=signal,
            confidence=confidence,
            shares=shares,
            price=entry_price,
            prices=prices,
        ):
            return False

        commission = self._commission_for_order(
            symbol=symbol,
            shares=shares,
            price=entry_price,
            asset_class=asset_class,
        )
        direction = 'long' if signal > 0 else 'short'

        # Cash/margin checks
        if direction == 'long' and (position_value + commission) > self.capital * 0.95:
            return False
        if direction == 'short' and position_value > self.capital * 0.95:
            return False

        if self.risk_manager:
            sector = ApexConfig.get_sector(symbol)
            allowed, _ = self.risk_manager.check_entry_allowed(symbol, sector, position_value)
            if not allowed:
                return False

        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            entry_commission=commission,
            highest_price=entry_price,
            lowest_price=entry_price,
            regime=regime,
            took_partial_profit=False,
        )

        self.positions[symbol] = position
        if direction == 'long':
            self.capital -= (position_value + commission)
        else:
            self.capital += (position_value - commission)

        if self.risk_manager:
            self.risk_manager.add_position(PositionRisk(
                symbol=symbol,
                entry_price=entry_price,
                current_price=entry_price,
                shares=shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop_pct,
                atr=0.0,
                risk_amount=position_value * 0.02,
                position_value=position_value,
                pnl=0.0,
                pnl_percent=0.0,
                days_held=0,
                max_favorable_excursion=0.0,
                max_adverse_excursion=0.0
            ))
        return True

    def _exit_position(self, symbol: str, exit_price: float, exit_reason: str, exit_date: datetime):
        """Exit a position and record the trade."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        asset_class = self._asset_class_for_symbol(symbol)

        # Apply slippage
        exit_slippage_bps = self._slippage_bps_for_asset(asset_class)
        slippage = exit_price * (exit_slippage_bps / 10000.0)
        if position.direction == 'long':
            exit_price -= slippage  # Sell lower
        else:
            exit_price += slippage  # Buy higher (cover short)

        # Calculate P&L
        if position.direction == 'long':
            gross_pnl = (exit_price - position.entry_price) * position.shares
            pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:
            gross_pnl = (position.entry_price - exit_price) * position.shares
            pnl_pct = (position.entry_price / exit_price - 1) * 100

        # Apply commission (round-trip)
        exit_commission = self._commission_for_order(
            symbol=symbol,
            shares=position.shares,
            price=exit_price,
            asset_class=asset_class,
        )
        pnl = gross_pnl - position.entry_commission - exit_commission

        # Calculate hold days
        hold_days = max((pd.Timestamp(exit_date) - pd.Timestamp(position.entry_date)).days, 0)

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
        if position.direction == 'long':
            self.capital += (exit_price * position.shares) - exit_commission
        else:
            self.capital -= (exit_price * position.shares) + exit_commission

        # Remove position
        del self.positions[symbol]

        # Update risk manager
        if self.risk_manager:
            self.risk_manager.remove_position(symbol)
            self.risk_manager.record_trade(pnl, pnl > 0)
            self.risk_manager.update_capital(self.capital)

    def _execute_partial_exit(
        self,
        symbol: str,
        market_price: float,
        exit_reason: str,
        exit_date: datetime,
        fraction: float = 0.5
    ):
        """Close part of a position and keep the remaining runner open."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        asset_class = self._asset_class_for_symbol(symbol)
        close_shares = min(position.shares, max(1, int(round(position.shares * fraction))))
        if close_shares >= position.shares:
            self._exit_position(symbol, market_price, exit_reason, exit_date)
            return

        partial_exit_slippage_bps = self._slippage_bps_for_asset(asset_class)
        slippage = market_price * (partial_exit_slippage_bps / 10000.0)
        exit_price = market_price - slippage if position.direction == 'long' else market_price + slippage

        if position.direction == 'long':
            gross_pnl = (exit_price - position.entry_price) * close_shares
            pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:
            gross_pnl = (position.entry_price - exit_price) * close_shares
            pnl_pct = (position.entry_price / exit_price - 1) * 100

        exit_commission = self._commission_for_order(
            symbol=symbol,
            shares=close_shares,
            price=exit_price,
            asset_class=asset_class,
        )
        entry_commission_alloc = position.entry_commission * (close_shares / max(position.shares, 1))
        pnl = gross_pnl - entry_commission_alloc - exit_commission

        hold_days = max((pd.Timestamp(exit_date) - pd.Timestamp(position.entry_date)).days, 0)
        if position.direction == 'long':
            max_favorable = (position.highest_price / position.entry_price - 1) * 100
            max_adverse = (position.lowest_price / position.entry_price - 1) * 100
        else:
            max_favorable = (position.entry_price / position.lowest_price - 1) * 100
            max_adverse = (position.entry_price / position.highest_price - 1) * 100

        self.trades.append(Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=close_shares,
            direction=position.direction,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=exit_reason,
            regime=position.regime,
            hold_days=hold_days,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
        ))

        if position.direction == 'long':
            self.capital += (exit_price * close_shares) - exit_commission
        else:
            self.capital -= (exit_price * close_shares) + exit_commission

        position.shares -= close_shares
        position.entry_commission = max(0.0, position.entry_commission - entry_commission_alloc)

        if position.shares <= 0:
            del self.positions[symbol]
            if self.risk_manager:
                self.risk_manager.remove_position(symbol)
        elif self.risk_manager and symbol in self.risk_manager.positions:
            rp = self.risk_manager.positions[symbol]
            rp.shares = position.shares
            rp.current_price = exit_price
            rp.position_value = exit_price * position.shares

        if self.risk_manager:
            self.risk_manager.record_trade(pnl, pnl > 0)
            self.risk_manager.update_capital(self.capital)

    def _calculate_gross_exposure(self, day_data: Optional[Dict] = None) -> float:
        """Gross notional open exposure (long + short absolute market value)."""
        gross = 0.0
        for symbol, position in self.positions.items():
            if day_data and symbol in day_data:
                mark = float(day_data[symbol].get('Close', position.entry_price))
            else:
                mark = float(position.entry_price)
            gross += abs(mark * position.shares)
        return gross

    def _calculate_portfolio_value(self, day_data: Dict) -> float:
        """Calculate total portfolio value."""
        positions_value = 0
        for symbol, position in self.positions.items():
            current_price = day_data[symbol]['Close'] if symbol in day_data else position.entry_price
            signed_qty = position.shares if position.direction == 'long' else -position.shares
            positions_value += current_price * signed_qty

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

        period = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
        if len(period) >= 2:
            return (period['Close'].iloc[-1] / period['Close'].iloc[0] - 1) * 100

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
        Uses bootstrap sampling (with replacement) to estimate outcome dispersion.
        """
        if len(self.trades) < 10:
            logger.warning("Not enough trades for Monte Carlo simulation")
            return None

        logger.info(f"\nRunning Monte Carlo simulation ({n_simulations} iterations)...")

        trade_pnls = [t.pnl for t in self.trades]
        n_trades = len(trade_pnls)
        hold_days = np.array([max(t.hold_days, 1) for t in self.trades], dtype=float)
        avg_hold_days = max(float(np.mean(hold_days)), 1.0)
        annualization = np.sqrt(252.0 / avg_hold_days)

        final_returns = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Bootstrap sample trades to create synthetic realizations
            sampled = np.random.choice(trade_pnls, size=n_trades, replace=True)

            # Calculate equity curve
            equity = [self.initial_capital]
            for pnl in sampled:
                equity.append(max(1.0, equity[-1] + pnl))

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
            trade_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
            if trade_returns.std() > 0:
                sharpe = trade_returns.mean() / trade_returns.std() * annualization
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

        logger.info("  Monte Carlo Results:")
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
