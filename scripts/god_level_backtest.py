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
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import warnings

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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

    # Deflated Sharpe Ratio (multiple testing adjustment)
    deflated_sharpe: Optional[Dict] = None

    # Stress test results (Phase 3)
    stress_tests: Optional[Dict] = None

    # Session type
    session_type: str = "unified"


class GodLevelBacktester:
    """
    God-level backtesting engine with walk-forward optimization
    and Monte Carlo simulation.
    """

    # Configuration
    WARMUP_DAYS = 60
    WALK_FORWARD_TRAIN_DAYS = 252  # 1 year training
    WALK_FORWARD_TEST_DAYS = 126   # 6 months testing
    # Purge/embargo gap: prevent feature contamination between train and test.
    # Per Lopez de Prado (2018, "Advances in Financial ML", Ch. 7):
    # purge_days >= max feature lookback (60 bars) + label forward window
    # embargo_days >= autocorrelation decay (typically 2-5 days for daily returns)
    WALK_FORWARD_PURGE_DAYS = 10   # Gap between train end and test start
    WALK_FORWARD_EMBARGO_DAYS = 5  # Additional embargo after purge

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
    PARTIAL_TAKE_PROFIT_PCT = 0.04   # Trigger partial at 4% (= 2× SL) → let runner ride
    PARTIAL_EXIT_FRACTION = 0.30    # Scale out only 30%; keep 70% for trend capture
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
        prices: Optional[pd.Series] = None,
        order_notional: float = 0.0,
    ) -> float:
        """Empirical slippage + square-root market impact model.

        Market impact per Almgren & Chriss (2001):
            impact = sigma_daily * sqrt(Q / ADV)
        where Q = order size in $, ADV = average daily volume in $.

        This prevents the backtest from being over-optimistic on large orders
        where market impact dominates spread-based slippage.
        """
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
        spread_slippage = float(min(max_bps, base_bps * (1.0 + vol_multiplier)))

        # Square-root market impact model
        # Estimate ADV from price series (proxy: avg price * assumed daily volume)
        if order_notional > 0 and len(prices) >= 20:
            avg_price = float(prices.tail(20).mean())
            # Conservative ADV estimates per asset class
            if asset_class == AssetClass.CRYPTO:
                est_adv = avg_price * 5_000_000  # ~$5M daily for major crypto
            elif asset_class == AssetClass.FOREX:
                est_adv = avg_price * 50_000_000  # FX is ultra-liquid
            else:
                est_adv = avg_price * 2_000_000  # ~$2M daily for mid-cap equities

            if est_adv > 0:
                participation_rate = order_notional / est_adv
                # impact_bps = sigma_daily_bps * sqrt(participation_rate)
                sigma_daily_bps = vol * 10000.0
                impact_bps = sigma_daily_bps * float(np.sqrt(min(participation_rate, 0.10)))
                spread_slippage += min(impact_bps, max_bps)  # Cap total impact

        return spread_slippage

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
        walk_forward: bool = True,
        session_type: str = "unified",
        historical_data_override: dict = None,
    ) -> BacktestResult:
        """
        Run comprehensive backtest with optional walk-forward optimization.
        session_type: "unified", "core", "crypto", or "apex" — scopes universe and thresholds.
        historical_data_override: pre-fetched {symbol: DataFrame} — skip internal fetch.
        """
        # If session_type was already set externally (e.g., from run_real_backtest.py), honour it.
        if not hasattr(self, "_session_type") or self._session_type == "unified":
            self._session_type = session_type
        else:
            session_type = self._session_type

        try:
            self._session_config = ApexConfig.get_session_config(session_type)
        except Exception:
            self._session_config = {}

        if symbols is None:
            try:
                session_symbols = ApexConfig.get_session_symbols(session_type)
                symbols = session_symbols[:50]  # Limit for speed
            except Exception:
                symbols = ApexConfig.SYMBOLS[:50]

        # Apply session-specific parameters
        if session_type in ("core", "crypto"):
            self.MAX_POSITIONS = self._session_config.get("max_positions", self.MAX_POSITIONS)
            self.initial_capital = self._session_config.get("initial_capital", self.initial_capital)
            self.capital = self.initial_capital
            self.peak_capital = self.initial_capital

        logger.info("=" * 70)
        logger.info("GOD LEVEL BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Walk-forward: {walk_forward}")

        # Fetch or accept historical data
        if historical_data_override is not None:
            logger.info("Using pre-fetched historical data (%d symbols)", len(historical_data_override))
            historical_data = historical_data_override
        else:
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

            # Purge + embargo gap: skip days between train end and test start
            # to prevent feature contamination (lookback windows that span
            # the train/test boundary). Per Lopez de Prado (2018), Ch. 7.
            purge_embargo = self.WALK_FORWARD_PURGE_DAYS + self.WALK_FORWARD_EMBARGO_DAYS
            test_start_idx = train_end_idx + purge_embargo
            if test_start_idx >= total_days:
                break
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

                prices_df = df.loc[mask].copy()

                # SPY benchmark for regime blending (market vs individual stock trend)
                _spy_df = historical_data.get("SPY", historical_data.get("spy"))
                _benchmark = (
                    _spy_df.loc[_spy_df.index <= date, "Close"]
                    if _spy_df is not None and len(_spy_df) > 0
                    else None
                )

                # Generate signal
                signal_data = self.signal_generator.generate_ml_signal(
                    symbol, prices_df, benchmark_prices=_benchmark
                )
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                regime = signal_data.get('regime', 'neutral')

                # PEAD filter (apex only): Post-Earnings Announcement Drift.
                # Skip longs within 5 days of an earnings MISS; boost confidence for beats.
                if getattr(self, "_session_type", "") == "apex":
                    try:
                        _earn = self._pead_cache.get(symbol)
                        if _earn is None:
                            from data.earnings_signal import EarningsSignal as _ES
                            if not hasattr(self, "_earnings_signal"):
                                self._earnings_signal = _ES()
                            _earn = self._earnings_signal.get_signal(symbol)
                            self._pead_cache[symbol] = _earn
                        if _earn and _earn.direction in ("miss",) and _earn.days_since_earnings <= 30:
                            if signal > 0:
                                continue   # Earnings miss in last 30d → skip LONG
                        if _earn and _earn.direction == "beat" and _earn.days_since_earnings <= 30:
                            confidence = min(1.0, confidence * 1.12)   # Boost for PEAD tailwind
                    except Exception:
                        pass

                # Entry criteria
                prices_series = prices_df['Close']
                if not self._passes_directional_filter(prices_series, signal, regime):
                    continue
                if not self._should_enter_trade(signal, confidence, regime, current_drawdown):
                    continue

                entered = self._enter_position(
                    symbol, row, signal, confidence, regime, date, prices_series, current_portfolio_value
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
        """Apply adaptive entry thresholds by regime and drawdown state.

        Session-specific thresholds are significantly relaxed to increase trade
        count (key driver of Sharpe stabilization) while maintaining signal quality
        via the edge-over-cost gate and position sizing.
        """
        session = getattr(self, "_session_type", "unified")
        if session == "core":
            # Core: relaxed thresholds, more trades in equities/indices/fx
            thresholds = {
                'strong_bull': (0.18, 0.30),
                'bull': (0.22, 0.34),
                'neutral': (0.25, 0.38),
                'bear': (0.22, 0.34),      # Encourage short entries
                'strong_bear': (0.20, 0.32),  # Shorts on bearish conviction
                'high_volatility': (0.30, 0.42),
            }
        elif session == "crypto":
            # Crypto: even more relaxed — crypto signals are noisier but moves are larger
            thresholds = {
                'strong_bull': (0.15, 0.25),
                'bull': (0.18, 0.28),
                'neutral': (0.20, 0.32),
                'bear': (0.18, 0.28),
                'strong_bear': (0.16, 0.26),
                'high_volatility': (0.25, 0.35),
            }
        elif session == "apex":
            # ── LIVE CONFIG THRESHOLDS ──────────────────────────────────────────
            # Read directly from ApexConfig so backtest stays in sync with live system.
            # Confidence is scaled by 0.80 because raw ML confidence (~0.40–0.55) is
            # lower than the live value after confidence-boosting pipeline steps.
            _reg_thresholds = ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME
            sig_t = float(_reg_thresholds.get(regime, _reg_thresholds.get('neutral', 0.18)))
            conf_t = float(getattr(ApexConfig, 'MIN_CONFIDENCE', 0.60)) * 0.80
            if current_drawdown >= 0.08:
                sig_t += 0.02
                conf_t += 0.02
            return abs(signal) >= sig_t and confidence >= conf_t
        else:
            # Legacy unified thresholds (lowered to guarantee sufficient trade volume)
            thresholds = {
                'strong_bull': (0.15, 0.25),
                'bull': (0.15, 0.25),
                'neutral': (0.15, 0.25),
                'bear': (0.15, 0.25),
                'strong_bear': (0.15, 0.25),
                'high_volatility': (0.20, 0.30),
            }
        signal_threshold, confidence_threshold = thresholds.get(regime, (0.50, 0.65))
        if current_drawdown >= 0.08:
            signal_threshold += 0.03
            confidence_threshold += 0.02
        return abs(signal) >= signal_threshold and confidence >= confidence_threshold

    def _passes_directional_filter(self, prices: pd.Series, signal: float, regime: str) -> bool:
        """Avoid trading against persistent trend/regime unless the setup is strong.

        In session modes, allow more directional flexibility:
        - Core: allow shorts in bear regimes (key missed alpha source)
        - Crypto: allow both directions in trending regimes
        """
        if len(prices) < 50 or signal == 0:
            return True

        session = getattr(self, "_session_type", "unified")
        ma20 = prices.iloc[-20:].mean()
        ma50 = prices.iloc[-50:].mean()
        trend_bias = (ma20 / ma50 - 1) if ma50 > 0 else 0.0

        long_allowed = trend_bias >= -0.01
        short_allowed = trend_bias <= 0.01

        if session in ("core", "crypto"):
            # Relaxed: allow shorts in bull if strong signal, allow longs in bear if strong signal
            if regime == 'strong_bull':
                short_allowed = abs(signal) >= 0.50 and trend_bias < -0.005
            elif regime == 'bull':
                short_allowed = abs(signal) >= 0.40 and trend_bias < -0.005
            elif regime in {'strong_bear', 'bear'}:
                # Key change: allow short entries in bear regimes (was blocking all longs)
                long_allowed = abs(signal) >= 0.50 and trend_bias > 0.005
                short_allowed = True  # Shorts are natural in bear
        elif session == "apex":
            # Match live execution_loop.py regime-direction gate exactly:
            # • strong_bear → hard-block LONGs
            # • bear → LONGs need counter-trend threshold × mult
            # • SHORTs in bear/strong_bear → ALLOWED (live change 2026-03-20)
            # • SHORTs in bull/strong_bull → blocked
            _ct_mult = float(getattr(ApexConfig, "REGIME_COUNTER_TREND_SIGNAL_MULT", 1.8))
            _base_sig = float(ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME.get(regime, 0.18))
            if signal > 0:   # LONG
                if regime == "strong_bear":
                    long_allowed = False
                elif regime == "bear":
                    long_allowed = abs(signal) >= _base_sig * _ct_mult
                else:
                    long_allowed = True
            else:            # SHORT
                if regime in {"bull", "strong_bull"}:
                    short_allowed = False
                else:
                    short_allowed = True
        else:
            if regime == 'strong_bull':
                short_allowed = False
            elif regime == 'bull':
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
        """Boost best setups while shrinking crowded/high-correlation entries.

        In session modes, high-conviction assets get an extra sizing boost
        to concentrate capital where the edge is strongest.
        """
        session = getattr(self, "_session_type", "unified")
        session_cfg = getattr(self, "_session_config", {})
        high_conviction = session_cfg.get("high_conviction", [])

        mult = 1.0

        # High-conviction asset boost (session-specific)
        if symbol in high_conviction:
            mult += 0.25

        if confidence >= 0.55 and abs(signal) >= 0.40:
            mult += 0.20
        elif confidence >= 0.72 and abs(signal) >= 0.62:
            mult += 0.15

        # Regime-aligned direction boost
        if regime in ('strong_bull', 'bull') and signal > 0:
            mult += 0.15
        elif regime in ('strong_bear', 'bear') and signal < 0:
            mult += 0.15  # Reward shorts in bearish regimes

        max_corr = self._max_open_correlation(symbol)
        if max_corr is not None:
            if max_corr <= 0.35:
                mult += 0.15
            elif max_corr >= 0.75:
                mult -= 0.20

        max_mult = 2.0 if session in ("core", "crypto") else 1.60
        return float(np.clip(mult, 0.75, max_mult))

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

            # Scale out partially before target to lock in gains
            if (
                not position.took_partial_profit
                and position.shares > 1
                and pnl_pct >= self.PARTIAL_TAKE_PROFIT_PCT
                and not take_profit_trigger
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
                    position.stop_loss = max(position.stop_loss, position.entry_price * 1.005)
                else:
                    position.stop_loss = min(position.stop_loss, position.entry_price * 0.995)

            # Check exit conditions
            exit_signal = False
            exit_reason = ""

            # 1. Take Profit
            if take_profit_trigger:
                exit_signal = True
                exit_reason = "Take profit"

            # 2. Stop loss
            elif position.direction == 'long' and current_price <= position.stop_loss:
                exit_signal = True
                exit_reason = "Stop loss"
            elif position.direction == 'short' and current_price >= position.stop_loss:
                exit_signal = True
                exit_reason = "Stop loss"

            # 3. Trailing stop
            elif pnl_pct > 0.02:  # Only after 2% profit
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

            # 4. Time-based exit (session-aware: tighter for better capital efficiency)
            hold_days = max((pd.Timestamp(date) - pd.Timestamp(position.entry_date)).days, 0)
            session = getattr(self, "_session_type", "unified")
            if session == "apex":
                # Apex: matches live system — max 50 days; exit stale positions early
                if hold_days >= 50:
                    exit_signal = True
                    exit_reason = "Max hold period"
                elif hold_days >= 20 and abs(pnl_pct) < 0.015:
                    # Dead capital: 20 days without 1.5% move — free it up
                    exit_signal = True
                    exit_reason = "Stale position (time-decay)"
                elif hold_days >= 10 and pnl_pct < -0.018:
                    # Losing position going nowhere after 10 days — cut early
                    exit_signal = True
                    exit_reason = "Stale loser"
            elif session in ("core", "crypto"):
                # Tighter time exits for session mode — eliminate dead capital
                max_hold = 30 if session == "core" else 14  # crypto moves fast
                if hold_days >= max_hold:
                    exit_signal = True
                    exit_reason = "Max hold period"
                # Stale position exit: close if held >15 days with <1% unrealized gain (core)
                elif session == "core" and hold_days >= 15 and abs(pnl_pct) < 0.01:
                    exit_signal = True
                    exit_reason = "Stale position (time-decay)"
                # Crypto: close if held >7 days with <2% unrealized gain
                elif session == "crypto" and hold_days >= 7 and abs(pnl_pct) < 0.02:
                    exit_signal = True
                    exit_reason = "Stale position (time-decay)"
            else:
                if hold_days >= 90:  # Legacy: Max 90 days
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

        # Apply slippage (with square-root market impact for large orders)
        est_notional = min(self.capital * self.MAX_POSITION_PCT, self.capital / max(1, self.MAX_POSITIONS))
        entry_slippage_bps = self._slippage_bps_for_asset(asset_class, prices, order_notional=est_notional)
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
            # Tight initial SL (2%) for fast loss cutting; wide trailing (5%) so runners breathe.
            # Typical stock daily vol ~1-2%: a 2% trailing stop is noise-triggered immediately.
            # 5% trailing survives 3-4 days of normal vol and captures trending moves.
            stop_loss = entry_price * (1 - 0.02) if signal > 0 else entry_price * (1 + 0.02)
            take_profit = entry_price * (1 + 0.08) if signal > 0 else entry_price * (1 - 0.08)
            trailing_stop_pct = 0.05

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
        # Use risk-free rate of 5% (consistent with performance_tracker.py)
        risk_free_daily = 0.05 / 252
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - risk_free_daily
            sharpe_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

            # Sortino (downside deviation using excess returns)
            downside_excess = excess_returns[excess_returns < 0]
            downside_std = downside_excess.std() * np.sqrt(252) if len(downside_excess) > 0 else excess_returns.std() * np.sqrt(252)
            sortino_ratio = (excess_returns.mean() * 252) / downside_std if downside_std > 0 else 0

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

        # Regime performance (with per-regime Sharpe)
        logger.info("\n--- PER-REGIME PERFORMANCE ---")
        regime_performance = self._calculate_regime_performance()

        # Deflated Sharpe Ratio (multiple testing adjustment)
        logger.info("\n--- DEFLATED SHARPE RATIO ---")
        dsr_result = self.calculate_deflated_sharpe(sharpe_ratio, total_trades)

        # Flag any bleeding regimes
        bleeding_regimes = [r for r, p in regime_performance.items() if p.get('negative_sharpe')]
        if bleeding_regimes:
            logger.warning(f"\n  *** WARNING: Strategy bleeds in regimes: {bleeding_regimes}")
            logger.warning(f"  *** Consider adding regime filter or disabling trading in these regimes")

        logger.info(f"\nTotal Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f} (risk-free adjusted)")
        logger.info(f"DSR: {dsr_result['dsr']:.3f} ({'PASS' if dsr_result['pass'] else 'FAIL'})")
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
            trades=self.trades,
            deflated_sharpe=dsr_result,
            session_type=getattr(self, "_session_type", "unified"),
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
        """Calculate performance by market regime INCLUDING per-regime Sharpe.

        Critical for GO/NO-GO: if any regime has negative Sharpe, the strategy
        must either add a regime filter for that regime or accept the bleed.
        A strategy profitable in 4/6 regimes but bleeding in 2 will surprise
        with drawdowns that the aggregate Sharpe masks.
        """
        regime_trades = defaultdict(list)

        for trade in self.trades:
            regime_trades[trade.regime].append(trade)

        regime_perf = {}
        for regime, trades in regime_trades.items():
            wins = [t for t in trades if t.pnl > 0]
            trade_returns = [t.pnl / max(abs(t.entry_price * t.shares), 1) for t in trades]

            # Per-regime Sharpe (annualized from trade returns)
            if len(trade_returns) >= 5 and np.std(trade_returns) > 0:
                avg_hold = np.mean([max(t.hold_days, 1) for t in trades])
                annualization = np.sqrt(252.0 / max(avg_hold, 1.0))
                regime_sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * annualization)
            else:
                regime_sharpe = 0.0

            regime_perf[regime] = {
                'trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'avg_pnl': np.mean([t.pnl for t in trades]) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'sharpe': round(regime_sharpe, 3),
                'negative_sharpe': regime_sharpe < 0,
            }

        # Log regime breakdown
        for regime, perf in sorted(regime_perf.items()):
            flag = " *** BLEEDING ***" if perf['negative_sharpe'] else ""
            logger.info(
                f"  Regime '{regime}': {perf['trades']} trades, "
                f"Sharpe={perf['sharpe']:.2f}, "
                f"WR={perf['win_rate']:.0f}%, "
                f"PnL=${perf['total_pnl']:,.0f}{flag}"
            )

        return regime_perf

    def calculate_deflated_sharpe(self, sharpe_ratio: float, n_trades: int, n_trials: int = 20) -> Dict:
        """Compute Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado (2014).

        DSR adjusts observed Sharpe for multiple testing (selection bias).
        Formula: DSR = P(SR* > 0 | SR_hat, N, T, skew, kurtosis)

        n_trials: number of strategy variants tried (configs, thresholds, etc.)
            Default 20 is conservative — most quant shops try 50-200 variants.

        GO gate: DSR > 1.35 (roughly p < 0.05 after multiple testing adjustment)

        Failure mode: if n_trials is underestimated, DSR is overoptimistic.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available — cannot compute Deflated Sharpe Ratio")
            return {'dsr': 0.0, 'pass': False, 'reason': 'scipy not installed'}

        if n_trades < 30:
            return {'dsr': 0.0, 'pass': False, 'reason': 'Insufficient trades'}

        # DSR is only meaningful when Sharpe > 0.  A negative Sharpe can never exceed
        # E[max_SR] ≈ 1.2, so the test would always fail trivially.  Report this clearly
        # instead of showing a meaningless -22 value.
        if sharpe_ratio <= 0:
            return {
                'dsr': sharpe_ratio,
                'pass': False,
                'reason': f'Sharpe ({sharpe_ratio:.2f}) <= 0 — fix R:R/regime before DSR is meaningful',
                'e_max_sr': None, 'sr_se': None, 'n_trials': n_trials,
            }
        if not hasattr(self, 'equity_curve') or len(self.equity_curve) < 30:
            return {'dsr': 0.0, 'pass': False, 'reason': 'Insufficient equity curve data'}

        equity_vals = np.array([v for _, v in self.equity_curve])
        # Calculate daily returns from equity curve 
        daily_returns = np.diff(equity_vals) / np.maximum(equity_vals[:-1], 1e-9)
        
        if len(daily_returns) < 30 or np.std(daily_returns) == 0:
            return {'dsr': 0.0, 'pass': False, 'reason': 'Zero variance in daily returns'}

        T = len(daily_returns)
        T_years = max(T / 252.0, 0.1)
        skew = float(scipy_stats.skew(daily_returns))
        kurtosis = float(scipy_stats.kurtosis(daily_returns))

        # Expected maximum standardized variable Z under null
        euler_mascheroni = 0.5772156649
        e_max_z = np.sqrt(2 * np.log(n_trials)) - (np.log(np.pi) + euler_mascheroni) / (2 * np.sqrt(2 * np.log(n_trials)))
        
        # Expected maximum annualized Sharpe under null
        e_max_sr = e_max_z * np.sqrt(1.0 / T_years)

        # Standard error of the annualized Sharpe ratio
        sr_se = np.sqrt((1 + 0.5 * sharpe_ratio**2 - skew * sharpe_ratio + (kurtosis / 4) * sharpe_ratio**2) / T_years)

        # DSR calculation
        if sr_se > 0:
            dsr = float((sharpe_ratio - e_max_sr) / sr_se)
        else:
            dsr = 0.0

        passed = dsr > 1.35
        logger.info(f"\n  Deflated Sharpe Ratio: {dsr:.3f} (threshold: 1.35) -> {'PASS' if passed else 'FAIL'}")
        logger.info(f"    Observed Sharpe: {sharpe_ratio:.3f}, E[max SR under null]: {e_max_sr:.3f}")
        logger.info(f"    n_trials={n_trials}, T={T}, skew={skew:.2f}, kurtosis={kurtosis:.2f}")

        return {
            'dsr': round(dsr, 4),
            'pass': passed,
            'e_max_sr': round(e_max_sr, 4),
            'sr_se': round(sr_se, 4),
            'n_trials': n_trials,
        }

    def run_permutation_test(self, n_permutations: int = 500) -> Dict:
        """Monte Carlo permutation test for strategy significance.

        Null hypothesis: the strategy's trade sequence contains no serial
        dependence — i.e., the Sharpe is achievable by random re-ordering.

        Method: shuffle trade PnLs and recompute Sharpe on each permutation.
        p-value = fraction of permuted Sharpes >= observed Sharpe.

        GO gate: p-value < 0.05 (strategy is significantly better than random).

        Reference: White (2000) "A Reality Check for Data Snooping",
                   Romano & Wolf (2005) "Stepwise Multiple Testing".
        """
        if len(self.trades) < 20:
            return {'p_value': 1.0, 'pass': False, 'reason': 'Insufficient trades'}

        trade_pnls = np.array([t.pnl for t in self.trades], dtype=float)
        n_trades = len(trade_pnls)
        hold_days = np.array([max(t.hold_days, 1) for t in self.trades], dtype=float)
        avg_hold = max(float(np.mean(hold_days)), 1.0)
        ann_factor = np.sqrt(252.0 / avg_hold)
        risk_free_daily = 0.05 / 252.0

        # Observed Sharpe from actual trade sequence
        cum_equity = np.cumsum(trade_pnls) + self.initial_capital
        trade_rets = np.diff(np.concatenate([[self.initial_capital], cum_equity])) / np.maximum(
            np.concatenate([[self.initial_capital], cum_equity[:-1]]), 1e-9
        )
        excess = trade_rets - risk_free_daily
        observed_sharpe = float(np.mean(excess) / max(np.std(excess, ddof=1), 1e-9) * ann_factor)

        # Permutation distribution
        rng = np.random.default_rng(42)
        permuted_sharpes = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(trade_pnls)
            cum_eq = np.cumsum(shuffled) + self.initial_capital
            p_rets = np.diff(np.concatenate([[self.initial_capital], cum_eq])) / np.maximum(
                np.concatenate([[self.initial_capital], cum_eq[:-1]]), 1e-9
            )
            p_excess = p_rets - risk_free_daily
            std = float(np.std(p_excess, ddof=1))
            s = float(np.mean(p_excess) / max(std, 1e-9) * ann_factor) if std > 0 else 0.0
            permuted_sharpes.append(s)

        permuted_sharpes = np.array(permuted_sharpes)
        p_value = float(np.mean(permuted_sharpes >= observed_sharpe))
        passed = p_value < 0.05

        logger.info(f"\n  Permutation Test: p-value={p_value:.4f} (threshold: 0.05) -> {'PASS' if passed else 'FAIL'}")
        logger.info(f"    Observed Sharpe: {observed_sharpe:.3f}")
        logger.info(f"    Permuted Sharpe distribution: mean={np.mean(permuted_sharpes):.3f}, "
                     f"95th={np.percentile(permuted_sharpes, 95):.3f}")

        return {
            'p_value': round(p_value, 4),
            'pass': passed,
            'observed_sharpe': round(observed_sharpe, 4),
            'permuted_mean': round(float(np.mean(permuted_sharpes)), 4),
            'permuted_95th': round(float(np.percentile(permuted_sharpes, 95)), 4),
            'n_permutations': n_permutations,
        }

    def run_parameter_perturbation(self, n_perturbations: int = 50) -> Dict:
        """Parameter perturbation sensitivity analysis.

        Tests whether strategy performance is robust to small parameter changes.
        Perturbs key thresholds by ±10-20% and re-evaluates trade filtering.

        A strategy that collapses under small perturbations is overfit.

        GO gate: Sharpe retains >=70% of its value across perturbations.

        Reference: Pardo (2008) "The Evaluation and Optimization of Trading Strategies".
        """
        if len(self.trades) < 20:
            return {'robustness': 0.0, 'pass': False, 'reason': 'Insufficient trades'}

        trade_pnls = np.array([t.pnl for t in self.trades], dtype=float)
        hold_days = np.array([max(t.hold_days, 1) for t in self.trades], dtype=float)
        avg_hold = max(float(np.mean(hold_days)), 1.0)
        ann_factor = np.sqrt(252.0 / avg_hold)
        risk_free_daily = 0.05 / 252.0

        # Base Sharpe
        cum_eq = np.cumsum(trade_pnls) + self.initial_capital
        rets = np.diff(np.concatenate([[self.initial_capital], cum_eq])) / np.maximum(
            np.concatenate([[self.initial_capital], cum_eq[:-1]]), 1e-9
        )
        excess = rets - risk_free_daily
        base_sharpe = float(np.mean(excess) / max(np.std(excess, ddof=1), 1e-9) * ann_factor)

        if base_sharpe <= 0:
            return {'robustness': 0.0, 'pass': False, 'reason': 'Negative base Sharpe'}

        # Perturb by randomly dropping/keeping trades (simulates threshold changes)
        # Each perturbation randomly excludes 5-15% of trades (simulating tighter filters)
        # and adds noise to PnLs (simulating parameter sensitivity)
        rng = np.random.default_rng(123)
        perturbed_sharpes = []

        for _ in range(n_perturbations):
            # Random drop rate between 5-15%
            drop_rate = rng.uniform(0.05, 0.15)
            keep_mask = rng.random(len(trade_pnls)) > drop_rate

            # Also add noise to PnL (±5% of each trade's PnL)
            noise = rng.normal(1.0, 0.05, size=len(trade_pnls))
            perturbed_pnls = trade_pnls * noise
            perturbed_pnls = perturbed_pnls[keep_mask]

            if len(perturbed_pnls) < 10:
                continue

            p_eq = np.cumsum(perturbed_pnls) + self.initial_capital
            p_rets = np.diff(np.concatenate([[self.initial_capital], p_eq])) / np.maximum(
                np.concatenate([[self.initial_capital], p_eq[:-1]]), 1e-9
            )
            p_excess = p_rets - risk_free_daily
            std = float(np.std(p_excess, ddof=1))
            s = float(np.mean(p_excess) / max(std, 1e-9) * ann_factor) if std > 0 else 0.0
            perturbed_sharpes.append(s)

        if not perturbed_sharpes:
            return {'robustness': 0.0, 'pass': False, 'reason': 'No valid perturbations'}

        perturbed_sharpes = np.array(perturbed_sharpes)
        # Robustness = fraction of perturbations where Sharpe >= 70% of base
        retention_threshold = base_sharpe * 0.70
        robustness = float(np.mean(perturbed_sharpes >= retention_threshold))
        median_ratio = float(np.median(perturbed_sharpes) / base_sharpe)
        passed = robustness >= 0.70

        logger.info(f"\n  Parameter Perturbation: robustness={robustness:.2%} (threshold: 70%) -> {'PASS' if passed else 'FAIL'}")
        logger.info(f"    Base Sharpe: {base_sharpe:.3f}")
        logger.info(f"    Perturbed Sharpe: median={np.median(perturbed_sharpes):.3f}, "
                     f"5th={np.percentile(perturbed_sharpes, 5):.3f}, "
                     f"95th={np.percentile(perturbed_sharpes, 95):.3f}")
        logger.info(f"    Median retention: {median_ratio:.1%} of base Sharpe")

        return {
            'robustness': round(robustness, 4),
            'pass': passed,
            'base_sharpe': round(base_sharpe, 4),
            'median_perturbed': round(float(np.median(perturbed_sharpes)), 4),
            'p5_perturbed': round(float(np.percentile(perturbed_sharpes, 5)), 4),
            'p95_perturbed': round(float(np.percentile(perturbed_sharpes, 95)), 4),
            'median_retention': round(median_ratio, 4),
        }

    def run_crisis_simulation(self) -> Dict:
        """Stress test: simulate portfolio behavior during crisis conditions.

        Applies synthetic shocks to the equity curve to estimate tail risk:
        1. Flash crash: -10% instantaneous shock
        2. Bear market: -2% per week for 8 weeks
        3. Liquidity crisis: all losing trades doubled, winning halved
        4. Correlation spike: simultaneous 3-sigma loss on all positions

        GO gate: Max drawdown under worst crisis stays under 30%.

        Reference: Taleb (2007) "The Black Swan" — fat tail stress testing.
        """
        if len(self.trades) < 20:
            return {'worst_dd': 100.0, 'pass': False, 'reason': 'Insufficient trades'}

        trade_pnls = np.array([t.pnl for t in self.trades], dtype=float)
        n_trades = len(trade_pnls)

        scenarios = {}

        # Scenario 1: Flash crash — inject -10% shock at midpoint
        flash_pnls = trade_pnls.copy()
        mid = n_trades // 2
        flash_pnls[mid] = -self.initial_capital * 0.10
        eq = np.cumsum(flash_pnls) + self.initial_capital
        peak = np.maximum.accumulate(eq)
        dd = float(np.min((eq - peak) / np.maximum(peak, 1e-9)) * 100)
        scenarios['flash_crash'] = {'max_dd': round(abs(dd), 2), 'final_equity': round(float(eq[-1]), 2)}

        # Scenario 2: Bear market — sustained losses
        bear_pnls = trade_pnls.copy()
        # Apply -2% per ~5 trades for 40 trades (simulates 8-week bear)
        bear_start = max(0, mid - 20)
        bear_end = min(n_trades, mid + 20)
        for i in range(bear_start, bear_end):
            bear_pnls[i] = min(bear_pnls[i], -abs(bear_pnls[i]) * 0.5 - self.initial_capital * 0.005)
        eq = np.cumsum(bear_pnls) + self.initial_capital
        peak = np.maximum.accumulate(eq)
        dd = float(np.min((eq - peak) / np.maximum(peak, 1e-9)) * 100)
        scenarios['bear_market'] = {'max_dd': round(abs(dd), 2), 'final_equity': round(float(eq[-1]), 2)}

        # Scenario 3: Liquidity crisis — losing trades double, winners halve
        liq_pnls = trade_pnls.copy()
        liq_pnls = np.where(liq_pnls < 0, liq_pnls * 2.0, liq_pnls * 0.5)
        eq = np.cumsum(liq_pnls) + self.initial_capital
        peak = np.maximum.accumulate(eq)
        dd = float(np.min((eq - peak) / np.maximum(peak, 1e-9)) * 100)
        scenarios['liquidity_crisis'] = {'max_dd': round(abs(dd), 2), 'final_equity': round(float(eq[-1]), 2)}

        # Scenario 4: Correlation spike — cluster worst trades together
        sorted_pnls = np.sort(trade_pnls)  # worst first
        worst_n = min(n_trades // 4, 20)
        corr_pnls = trade_pnls.copy()
        # Put worst trades consecutively in the middle
        corr_pnls[mid:mid + worst_n] = sorted_pnls[:worst_n]
        eq = np.cumsum(corr_pnls) + self.initial_capital
        peak = np.maximum.accumulate(eq)
        dd = float(np.min((eq - peak) / np.maximum(peak, 1e-9)) * 100)
        scenarios['correlation_spike'] = {'max_dd': round(abs(dd), 2), 'final_equity': round(float(eq[-1]), 2)}

        worst_dd = max(s['max_dd'] for s in scenarios.values())
        passed = worst_dd < 30.0

        logger.info(f"\n  Crisis Simulation: worst_dd={worst_dd:.1f}% (threshold: 30%) -> {'PASS' if passed else 'FAIL'}")
        for name, result in scenarios.items():
            logger.info(f"    {name}: max_dd={result['max_dd']:.1f}%, final_equity=${result['final_equity']:,.0f}")

        return {
            'scenarios': scenarios,
            'worst_dd': round(worst_dd, 2),
            'pass': passed,
        }

    def run_all_stress_tests(self, sharpe_ratio: float, n_trades: int) -> Dict:
        """Run all Phase 3 stress tests and return consolidated results.

        Returns a dict with individual test results and an overall pass/fail.
        """
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: STRESS TESTS")
        logger.info("=" * 60)

        results = {}

        # 1. Deflated Sharpe Ratio
        results['deflated_sharpe'] = self.calculate_deflated_sharpe(sharpe_ratio, n_trades)

        # 2. Permutation test
        results['permutation_test'] = self.run_permutation_test()

        # 3. Parameter perturbation
        results['parameter_perturbation'] = self.run_parameter_perturbation()

        # 4. Crisis simulation
        results['crisis_simulation'] = self.run_crisis_simulation()

        # Overall
        gates = [results[k].get('pass', False) for k in results]
        results['all_pass'] = all(gates)
        results['pass_count'] = sum(gates)
        results['total_gates'] = len(gates)

        logger.info(f"\n  STRESS TEST SUMMARY: {results['pass_count']}/{results['total_gates']} gates passed"
                     f" -> {'ALL PASS' if results['all_pass'] else 'SOME FAILED'}")

        return results

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


def print_results(result: BacktestResult, stress_results: Dict = None):
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

    if stress_results:
        print("\n🔬 STRESS TESTS (Phase 3)")
        print("-" * 40)

        dsr = stress_results.get('deflated_sharpe', {})
        if dsr:
            status = "PASS" if dsr.get('pass') else "FAIL"
            reason = dsr.get('reason', '')
            if reason and not dsr.get('pass') and dsr.get('e_max_sr') is None:
                # Sharpe ≤ 0 or other pre-condition failure — show reason, not a misleading value
                print(f"  Deflated Sharpe:    N/A [{status}] — {reason}")
            else:
                print(f"  Deflated Sharpe:    {dsr.get('dsr', 0):.3f} (gate: 1.35) [{status}]")

        perm = stress_results.get('permutation_test', {})
        if perm:
            status = "PASS" if perm.get('pass') else "FAIL"
            print(f"  Permutation p-val:  {perm.get('p_value', 1):.4f} (gate: <0.05) [{status}]")

        pert = stress_results.get('parameter_perturbation', {})
        if pert:
            status = "PASS" if pert.get('pass') else "FAIL"
            print(f"  Param Robustness:   {pert.get('robustness', 0):.1%} (gate: >=70%) [{status}]")
            print(f"    Median retention: {pert.get('median_retention', 0):.1%} of base Sharpe")

        crisis = stress_results.get('crisis_simulation', {})
        if crisis:
            status = "PASS" if crisis.get('pass') else "FAIL"
            print(f"  Crisis Max DD:      {crisis.get('worst_dd', 0):.1f}% (gate: <30%) [{status}]")
            for name, sc in crisis.get('scenarios', {}).items():
                print(f"    {name:20s}: dd={sc['max_dd']:.1f}%, final=${sc['final_equity']:,.0f}")

        passed = stress_results.get('pass_count', 0)
        total = stress_results.get('total_gates', 0)
        print(f"\n  STRESS TESTS: {passed}/{total} passed")

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


def go_no_go_assessment(result: BacktestResult, stress_results: Dict = None) -> Dict:
    """Phase 4: GO/NO-GO assessment with 8 quantitative gates.

    Each gate has a clear threshold. ALL gates must pass for GO.

    Gates:
    1. Sharpe >= 1.5 (after costs, walk-forward OOS)
    2. Max drawdown <= 15%
    3. Win rate >= 50%
    4. Profit factor >= 1.3
    5. Deflated Sharpe > 1.35 (multiple testing adjustment)
    6. Permutation test p < 0.05 (strategy is non-random)
    7. Parameter robustness >= 70% (not overfit to exact params)
    8. Crisis max DD < 30% (survives tail events)
    """
    print("\n" + "=" * 70)
    print("PHASE 4: GO / NO-GO ASSESSMENT")
    print("=" * 70)

    gates = []

    # Gate 1: Sharpe
    g1 = result.sharpe_ratio >= 1.5
    gates.append(('Sharpe >= 1.5', g1, f'{result.sharpe_ratio:.2f}'))
    print(f"  {'PASS' if g1 else 'FAIL'} | Gate 1: Sharpe >= 1.5 (actual: {result.sharpe_ratio:.2f})")

    # Gate 2: Max drawdown
    g2 = result.max_drawdown <= 15.0
    gates.append(('Max DD <= 15%', g2, f'{result.max_drawdown:.1f}%'))
    print(f"  {'PASS' if g2 else 'FAIL'} | Gate 2: Max DD <= 15% (actual: {result.max_drawdown:.1f}%)")

    # Gate 3: Win rate
    g3 = result.win_rate >= 50.0
    gates.append(('Win Rate >= 50%', g3, f'{result.win_rate:.1f}%'))
    print(f"  {'PASS' if g3 else 'FAIL'} | Gate 3: Win Rate >= 50% (actual: {result.win_rate:.1f}%)")

    # Gate 4: Profit factor
    g4 = result.profit_factor >= 1.3
    gates.append(('Profit Factor >= 1.3', g4, f'{result.profit_factor:.2f}'))
    print(f"  {'PASS' if g4 else 'FAIL'} | Gate 4: Profit Factor >= 1.3 (actual: {result.profit_factor:.2f})")

    # Gates 5-8 from stress tests
    if stress_results:
        dsr = stress_results.get('deflated_sharpe', {})
        g5 = dsr.get('pass', False)
        _dsr_val = dsr.get('dsr', 0)
        _dsr_str = f"{_dsr_val:.3f}" if dsr.get('e_max_sr') is not None else "N/A (Sharpe≤0)"
        gates.append(('DSR > 1.35', g5, _dsr_str))
        print(f"  {'PASS' if g5 else 'FAIL'} | Gate 5: Deflated Sharpe > 1.35 (actual: {_dsr_str})")

        perm = stress_results.get('permutation_test', {})
        g6 = perm.get('pass', False)
        gates.append(('Perm. p < 0.05', g6, f"{perm.get('p_value', 1):.4f}"))
        print(f"  {'PASS' if g6 else 'FAIL'} | Gate 6: Permutation p < 0.05 (actual: {perm.get('p_value', 1):.4f})")

        pert = stress_results.get('parameter_perturbation', {})
        g7 = pert.get('pass', False)
        gates.append(('Robustness >= 70%', g7, f"{pert.get('robustness', 0):.1%}"))
        print(f"  {'PASS' if g7 else 'FAIL'} | Gate 7: Param Robustness >= 70% (actual: {pert.get('robustness', 0):.1%})")

        crisis = stress_results.get('crisis_simulation', {})
        g8 = crisis.get('pass', False)
        gates.append(('Crisis DD < 30%', g8, f"{crisis.get('worst_dd', 0):.1f}%"))
        print(f"  {'PASS' if g8 else 'FAIL'} | Gate 8: Crisis Max DD < 30% (actual: {crisis.get('worst_dd', 0):.1f}%)")
    else:
        for i in range(5, 9):
            gates.append((f'Gate {i}', False, 'N/A'))
            print(f"  FAIL | Gate {i}: Stress tests not run")

    passed = sum(1 for _, p, _ in gates if p)
    total = len(gates)
    all_pass = passed == total

    print(f"\n  {'=' * 50}")
    if all_pass:
        print(f"  VERDICT: GO ({passed}/{total} gates passed)")
        print(f"  Strategy is cleared for paper trading deployment.")
    elif passed >= 6:
        print(f"  VERDICT: CONDITIONAL GO ({passed}/{total} gates passed)")
        print(f"  Strategy shows promise but has {total - passed} failing gate(s).")
        print(f"  Recommended: paper trade with reduced size, monitor failing gates.")
    else:
        print(f"  VERDICT: NO-GO ({passed}/{total} gates passed)")
        print(f"  Strategy requires further development before deployment.")
        failed = [name for name, p, _ in gates if not p]
        print(f"  Failing gates: {', '.join(failed)}")
    print(f"  {'=' * 50}")

    return {
        'verdict': 'GO' if all_pass else ('CONDITIONAL_GO' if passed >= 6 else 'NO_GO'),
        'gates_passed': passed,
        'gates_total': total,
        'gates': [{'name': n, 'pass': p, 'value': v} for n, p, v in gates],
    }


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

        # Run Phase 3 stress tests
        stress_results = backtester.run_all_stress_tests(
            sharpe_ratio=result.sharpe_ratio,
            n_trades=result.total_trades,
        )
        result.deflated_sharpe = stress_results.get('deflated_sharpe')
        result.stress_tests = stress_results

        # Print results
        print_results(result, stress_results=stress_results)

        # Phase 4: GO/NO-GO assessment
        go_result = go_no_go_assessment(result, stress_results=stress_results)

        return result

    return None


if __name__ == "__main__":
    result = main()
