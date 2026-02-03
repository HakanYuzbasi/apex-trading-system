"""
main.py - APEX Trading System
PRODUCTION VERSION - All critical fixes implemented
- Position sync after every trade
- 60-second cooldown protection
- Short position support
- Max shares limit
- Transaction cost tracking
- Parallel processing
- Comprehensive error handling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
import numpy as np
import pandas as pd
from pathlib import Path
import json

from execution.ibkr_connector import IBKRConnector
from models.advanced_signal_generator import AdvancedSignalGenerator
from risk.risk_manager import RiskManager
from portfolio.portfolio_optimizer import PortfolioOptimizer
from data.market_data import MarketDataFetcher
from monitoring.performance_tracker import PerformanceTracker
from monitoring.live_monitor import LiveMonitor
from config import ApexConfig

# Institutional-grade components
from models.institutional_signal_generator import (
    UltimateSignalGenerator as InstitutionalSignalGenerator,
    SignalOutput
)
from risk.institutional_risk_manager import (
    InstitutionalRiskManager,
    RiskConfig,
    SizingResult
)
from monitoring.institutional_metrics import (
    InstitutionalMetrics,
    print_performance_report
)
from risk.god_level_risk_manager import GodLevelRiskManager
from portfolio.correlation_manager import CorrelationManager
from execution.advanced_order_executor import AdvancedOrderExecutor
from models.god_level_signal_generator import GodLevelSignalGenerator, MarketRegime
from models.enhanced_signal_filter import EnhancedSignalFilter, create_enhanced_filter
from execution.options_trader import OptionsTrader, OptionType, OptionStrategy

# ═══════════════════════════════════════════════════════════════════════════════
# SOTA IMPROVEMENTS - Phase 2 & 3 Modules
# ═══════════════════════════════════════════════════════════════════════════════
from risk.vix_regime_manager import VIXRegimeManager, VIXRegime
from models.cross_sectional_momentum import CrossSectionalMomentum
from data.sentiment_analyzer import SentimentAnalyzer, VolumePriceSentiment
from execution.arrival_price_benchmark import ArrivalPriceBenchmark
from monitoring.health_dashboard import HealthDashboard, HealthStatus
from monitoring.data_quality import DataQualityMonitor
from risk.dynamic_exit_manager import DynamicExitManager, get_exit_manager, ExitUrgency

logging.basicConfig(
    level=getattr(logging, ApexConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ApexConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress IBKR noise
logging.getLogger('ib_insync.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ib_insync.client').setLevel(logging.WARNING)


class ApexTradingSystem:
    """
    Main trading system with comprehensive risk controls.
    
    Features:
    - Long/Short trading
    - Position sync with IBKR
    - 60-second trade cooldown
    - Max shares limit
    - Transaction cost tracking
    - Sector exposure limits
    """
    
    def __init__(self):
        self.print_banner()
        logger.info("=" * 80)
        logger.info(f"🚀 {ApexConfig.SYSTEM_NAME} V{ApexConfig.VERSION}")
        logger.info("=" * 80)
        
        # Initialize components
        if ApexConfig.LIVE_TRADING:
            mode = "PAPER" if ApexConfig.IBKR_PORT == 7497 else "LIVE"
            logger.info(f"📊 Mode: {mode} TRADING")
            self.ibkr = IBKRConnector(
                host=ApexConfig.IBKR_HOST,
                port=ApexConfig.IBKR_PORT,
                client_id=ApexConfig.IBKR_CLIENT_ID
            )
            # ✅ Phase 2.1: Initialize advanced executor for TWAP/VWAP
            if ApexConfig.USE_ADVANCED_EXECUTION:
                self.advanced_executor = AdvancedOrderExecutor(self.ibkr)
                logger.info("📊 Advanced execution (TWAP/VWAP) enabled")
        else:
            logger.info("📊 Mode: SIMULATION")
            self.ibkr = None
        
        # Initialize modules
        self.signal_generator = AdvancedSignalGenerator()
        self.risk_manager = RiskManager(
            max_daily_loss=ApexConfig.MAX_DAILY_LOSS,
            max_drawdown=ApexConfig.MAX_DRAWDOWN
        )
        self.portfolio_optimizer = PortfolioOptimizer()
        self.market_data = MarketDataFetcher()
        self.performance_tracker = PerformanceTracker()
        self.live_monitor = LiveMonitor()

        # Institutional-grade components
        self.inst_signal_generator = InstitutionalSignalGenerator(
            lookback=60,
            n_cv_splits=5,
            purge_gap=5,
            embargo_gap=2,
            random_seed=42
        )
        self.inst_risk_manager = InstitutionalRiskManager(
            config=RiskConfig(
                max_position_pct=0.05,
                max_sector_pct=0.30,
                target_portfolio_vol=0.12,
                max_daily_loss_pct=ApexConfig.MAX_DAILY_LOSS,
                max_drawdown_pct=ApexConfig.MAX_DRAWDOWN
            )
        )
        self.inst_metrics = InstitutionalMetrics(risk_free_rate=0.02)

        # God-level risk manager for ATR-based stops and Kelly sizing
        self.god_risk_manager = GodLevelRiskManager(
            initial_capital=ApexConfig.INITIAL_CAPITAL,
            max_position_pct=0.05,
            max_portfolio_risk=0.02,
            max_correlation=ApexConfig.MAX_CORRELATION,
            max_sector_exposure=ApexConfig.MAX_SECTOR_EXPOSURE,
            max_drawdown_limit=ApexConfig.MAX_DRAWDOWN
        )
        self.god_risk_manager.set_sector_map(ApexConfig.SECTOR_MAP)

        # Correlation manager for diversification monitoring
        self.correlation_manager = CorrelationManager()

        # Advanced order executor for TWAP/VWAP on large orders
        self.advanced_executor: Optional[AdvancedOrderExecutor] = None  # Initialized after IBKR connection

        # ✅ Options trader for hedging and income generation
        self.options_trader: Optional[OptionsTrader] = None  # Initialized after IBKR connection
        self.options_positions: Dict[str, dict] = {}  # Track options positions

        # ✅ Phase 3.2: GodLevel signal generator for regime detection
        self.god_signal_generator = GodLevelSignalGenerator()
        self._current_regime: str = 'neutral'  # Cache current market regime

        # ✅ ENHANCED: Signal quality filter for higher-quality trades
        self.signal_filter = create_enhanced_filter()
        self._current_vix: Optional[float] = None  # Cache VIX for signal filtering

        # ✅ DYNAMIC: Exit manager for adaptive exit thresholds
        self.exit_manager = get_exit_manager()
        self.position_entry_signals: Dict[str, float] = {}  # Track entry signal strength

        # ═══════════════════════════════════════════════════════════════════════
        # SOTA IMPROVEMENTS - Phase 2 & 3 Components
        # ═══════════════════════════════════════════════════════════════════════
        
        # VIX-based adaptive risk management
        self.vix_manager = VIXRegimeManager(cache_minutes=5)
        self._vix_risk_multiplier: float = 1.0
        
        # Cross-sectional momentum for universe ranking
        self.cs_momentum = CrossSectionalMomentum(
            lookback_months=12,
            skip_months=1,
            volatility_adjust=True
        )
        
        # News sentiment analyzer (free Yahoo Finance)
        self.sentiment_analyzer = SentimentAnalyzer(cache_minutes=30)
        
        # Arrival price benchmarking for execution quality
        self.arrival_benchmark = ArrivalPriceBenchmark(max_history=1000)
        
        # Health monitoring dashboard
        self.health_dashboard = HealthDashboard(data_dir=str(ApexConfig.DATA_DIR))
        
        # Data quality monitoring
        self.data_quality_monitor = DataQualityMonitor(
            stale_threshold_minutes=30,
            min_history_days=60
        )
        
        logger.info("✅ SOTA modules initialized (VIX, Momentum, Sentiment, Health)")

        # Feature flag for institutional mode
        self.use_institutional = True  # Toggle to enable/disable institutional components
        
        # State
        self.capital = ApexConfig.INITIAL_CAPITAL
        self.positions: Dict[str, int] = {}  # symbol -> quantity (positive=long, negative=short)
        self.is_running = False
        self._cached_ibkr_positions: Optional[Dict[str, int]] = None  # Cycle-level cache
        
        # Cache
        self.price_cache: Dict[str, float] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.position_entry_prices: Dict[str, float] = {}
        self.position_entry_times: Dict[str, datetime] = {}
        self.position_peak_prices: Dict[str, float] = {}  # For trailing stops

        # ATR-based dynamic stop levels per position
        self.position_stops: Dict[str, Dict] = {}  # symbol -> {stop_loss, take_profit, trailing_stop_pct, atr}
        
        # ✅ NEW: Protection mechanisms
        self.pending_orders: set = set()
        self.last_trade_time: Dict[str, datetime] = {}  # 60-second cooldown
        self.sector_exposure: Dict[str, float] = {}  # Track sector concentration
        self.total_commissions: float = 0.0  # Track transaction costs

        # ✅ Failed exit retry tracking
        self.failed_exits: Dict[str, Dict] = {}  # symbol -> {reason, attempts, last_attempt}

        # ✅ CRITICAL: Semaphore to prevent race condition in parallel processing
        # This ensures only a limited number of entry trades can execute concurrently
        self._entry_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent entries
        self._position_lock = asyncio.Lock()  # Lock for position count checks

        # ✅ Phase 1.4: Graduated circuit breaker risk multiplier
        self._risk_multiplier: float = 1.0  # 1.0 = full size, 0.5 = half size during WARNING
        
        logger.info(f"💰 Capital: ${self.capital:,.2f}")
        logger.info(f"📈 Universe: {ApexConfig.UNIVERSE_MODE} ({len(ApexConfig.SYMBOLS)} symbols)")
        logger.info(f"📊 Max Positions: {ApexConfig.MAX_POSITIONS}")
        logger.info(f"💵 Position Size: ${ApexConfig.POSITION_SIZE_USD:,}")
        logger.info(f"🛡️  Max Shares/Position: {ApexConfig.MAX_SHARES_PER_POSITION}")
        logger.info(f"⏱️  Trade Cooldown: {ApexConfig.TRADE_COOLDOWN_SECONDS}s")
        logger.info(f"📱 Dashboard: Enabled")
        logger.info("✅ All modules initialized!")
        logger.info("=" * 80)

    @property
    def position_count(self) -> int:
        """Get current number of active positions (derived from positions dict)."""
        return len([qty for qty in self.positions.values() if qty != 0])

    def _get_est_hour(self) -> float:
        """Get current hour in Eastern Time (handles DST properly)."""
        if PYTZ_AVAILABLE:
            eastern = pytz.timezone('America/New_York')
            now_est = datetime.now(pytz.UTC).astimezone(eastern)
            return now_est.hour + now_est.minute / 60.0
        else:
            # Fallback: approximate EST (UTC-5)
            now = datetime.utcnow()
            est_hour = now.hour - 5 + now.minute / 60.0
            if est_hour < 0:
                est_hour += 24
            return est_hour

    def is_trading_hours(self, symbol: str, est_hour: float) -> bool:
        """
        Check if it's valid trading hours for a given symbol.

        Commodity ETFs (GLD, SLV, PALL, USO, UNG) can trade extended hours
        if COMMODITY_ETF_USE_EXTENDED is enabled.

        Args:
            symbol: Stock ticker
            est_hour: Current hour in EST

        Returns:
            True if within trading hours for this asset type
        """
        # Check if commodity/ETF with extended hours enabled
        if ApexConfig.is_commodity(symbol) and ApexConfig.COMMODITY_ETF_USE_EXTENDED:
            return ApexConfig.EXTENDED_HOURS_START <= est_hour <= ApexConfig.EXTENDED_HOURS_END

        # Regular stock hours
        return ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END

    def _write_heartbeat(self):
        """
        Write heartbeat file for watchdog monitoring.

        The watchdog process monitors this file to detect if the
        trading system is hung or crashed.
        """
        try:
            heartbeat_file = ApexConfig.DATA_DIR / 'heartbeat.json'
            data = {
                'timestamp': datetime.now().isoformat(),
                'position_count': self.position_count,
                'capital': self.capital,
                'is_trading': True,
                'cycle_count': getattr(self, '_cycle_count', 0)
            }
            with open(heartbeat_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Error writing heartbeat: {e}")

    def print_banner(self):
        print("""
╔═══════════════════════════════════════════════════════════════╗
║     █████╗ ██████╗ ███████╗██╗  ██╗                           ║
║    ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝                           ║
║    ███████║██████╔╝█████╗   ╚███╔╝                            ║
║    ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗                            ║
║    ██║  ██║██║     ███████╗██╔╝ ██╗                           ║
║    ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝                           ║
║    ALGORITHMIC PORTFOLIO EXECUTION                            ║
║    PRODUCTION VERSION - ALL FIXES APPLIED                     ║
╚═══════════════════════════════════════════════════════════════╝
        """)
    
    async def initialize(self):
        """Initialize connections and load data."""
        logger.info("🔄 Initializing system...")
        
        # Connect to IBKR
        if self.ibkr:
            await self.ibkr.connect()
            self.capital = await self.ibkr.get_portfolio_value()
            self.risk_manager.set_starting_capital(self.capital)
            logger.info(f"💰 IBKR Account: ${self.capital:,.2f}")
            
            # Load existing positions from IBKR
            self.positions = await self.ibkr.get_all_positions()

            if self.positions:
                logger.info(f"📊 Loaded {self.position_count} existing positions:")
                for symbol, qty in self.positions.items():
                    if qty != 0:
                        pos_type = "LONG" if qty > 0 else "SHORT"
                        # Get current price as entry price for loaded positions
                        try:
                            price = await self.ibkr.get_market_price(symbol)
                            if price and price > 0:
                                self.position_entry_prices[symbol] = price
                                self.position_entry_times[symbol] = datetime.now()
                                self.position_peak_prices[symbol] = price
                                self.position_entry_signals[symbol] = 0.5 if qty > 0 else -0.5  # Assume moderate signal for synced positions
                                self.price_cache[symbol] = price
                                logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) @ ${price:.2f}")
                            else:
                                logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type})")
                        except:
                            logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type})")
            
            # Load pending orders
            await self.refresh_pending_orders()

            # Initialize options trader
            if ApexConfig.OPTIONS_ENABLED:
                self.options_trader = OptionsTrader(
                    ibkr_connector=self.ibkr,
                    risk_free_rate=ApexConfig.OPTIONS_RISK_FREE_RATE,
                    default_vol=0.25
                )
                # Load existing options positions
                self.options_positions = await self.ibkr.get_option_positions()
                if self.options_positions:
                    logger.info(f"🎯 Loaded {len(self.options_positions)} existing option positions")
                logger.info("✅ Options trading enabled")

        # Pre-load historical data
        logger.info("📥 Loading historical data for ML training...")
        loaded = 0
        for i, symbol in enumerate(ApexConfig.SYMBOLS, 1):
            if i % 10 == 0:
                logger.info(f"   Loaded {i}/{len(ApexConfig.SYMBOLS)} symbols...")
            try:
                data = self.market_data.fetch_historical_data(symbol, days=400)
                if not data.empty:
                    self.historical_data[symbol] = data
                    loaded += 1
            except Exception as e:
                logger.debug(f"   Failed to load {symbol}: {e}")
        
        logger.info(f"✅ Loaded data for {loaded} symbols")

        # Initialize ATR-based stops for existing positions (now that we have historical data)
        if self.positions and self.ibkr:
            logger.info("")
            logger.info("🎯 Initializing ATR-based stops for existing positions...")
            for symbol, qty in self.positions.items():
                if qty != 0 and symbol in self.historical_data:
                    try:
                        prices = self.historical_data[symbol]['Close']
                        entry_price = self.position_entry_prices.get(symbol, prices.iloc[-1])
                        current_price = prices.iloc[-1]

                        # Calculate stops using the god-level risk manager
                        stops = self.god_risk_manager.calculate_stops_for_existing_position(
                            symbol=symbol,
                            entry_price=entry_price,
                            current_price=current_price,
                            position_qty=qty,
                            prices=prices,
                            regime=self._current_regime or 'neutral'
                        )

                        self.position_stops[symbol] = stops

                        # Update peak price if current price is better than entry
                        if qty > 0 and current_price > entry_price:
                            self.position_peak_prices[symbol] = current_price
                        elif qty < 0 and current_price < entry_price:
                            self.position_peak_prices[symbol] = current_price

                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not set stops for {symbol}: {e}")
            logger.info("✅ Position stops initialized")

        # Train ML models
        logger.info("")
        logger.info("🧠 Training advanced ML models...")
        logger.info("   This may take 30-60 seconds...")
        try:
            self.signal_generator.train_models(self.historical_data)
            logger.info("✅ ML models trained and ready!")
        except Exception as e:
            logger.warning(f"⚠️  ML training failed: {e}")
            logger.warning("   Falling back to technical analysis only")

        # Train institutional signal generator
        if self.use_institutional:
            logger.info("")
            logger.info("🏛️  Training INSTITUTIONAL ML models...")
            logger.info("   Purged time-series cross-validation enabled")
            try:
                training_results = self.inst_signal_generator.train(
                    self.historical_data,
                    target_horizon=5,
                    min_samples_per_regime=200
                )
                if training_results:
                    logger.info("✅ Institutional ML models trained!")
                    for regime_name, metrics_list in training_results.items():
                        if metrics_list:
                            avg_mse = np.mean([m.val_mse for m in metrics_list])
                            avg_acc = np.mean([m.directional_accuracy for m in metrics_list])
                            logger.info(f"   {regime_name}: avg_val_mse={avg_mse:.6f}, avg_dir_acc={avg_acc:.1%}")
                else:
                    logger.warning("⚠️  Institutional training returned no results")
                    self.use_institutional = False
            except Exception as e:
                logger.warning(f"⚠️  Institutional training failed: {e}")
                logger.warning("   Falling back to standard signal generator")
                self.use_institutional = False

            # Initialize institutional risk manager with capital
            self.inst_risk_manager.initialize(self.capital)

        # ✅ Phase 1.3: Initialize correlation manager with returns data
        if ApexConfig.USE_CORRELATION_MANAGER:
            logger.info("📊 Initializing correlation manager...")
            for symbol, data in self.historical_data.items():
                if 'Close' in data.columns and len(data) >= 60:
                    returns = data['Close'].pct_change().dropna()
                    self.correlation_manager.update_returns(symbol, returns)
            logger.info(f"✅ Correlation manager initialized with {len(self.correlation_manager.returns_history)} symbols")

            # Update god-level risk manager correlation matrix
            self.god_risk_manager.update_correlation_matrix(self.historical_data)

        # Initialize dashboard
        self._export_initial_state()
        logger.info("")
    
    async def refresh_pending_orders(self):
        """Refresh the set of symbols with pending orders."""
        if not self.ibkr:
            return
        
        try:
            self.pending_orders.clear()
            open_trades = self.ibkr.ib.openTrades()
            
            for trade in open_trades:
                symbol = trade.contract.symbol
                status = trade.orderStatus.status
                
                if status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'ApiPending']:
                    self.pending_orders.add(symbol)
                    logger.debug(f"   Pending: {symbol} (status={status})")
            
            if self.pending_orders:
                logger.info(f"⏳ Found {len(self.pending_orders)} pending orders")
        
        except Exception as e:
            logger.error(f"Error refreshing pending orders: {e}")
    
    async def sync_positions_with_ibkr(self):
        """✅ Force sync positions with IBKR's actual positions."""
        if not self.ibkr:
            return
        
        try:
            actual_positions = await self.ibkr.get_all_positions()
            
            # Check for mismatches
            mismatches = []
            for symbol in set(list(self.positions.keys()) + list(actual_positions.keys())):
                local_qty = self.positions.get(symbol, 0)
                ibkr_qty = actual_positions.get(symbol, 0)
                
                if local_qty != ibkr_qty:
                    mismatches.append(f"{symbol}: Local={local_qty}, IBKR={ibkr_qty}")
            
            if mismatches:
                logger.warning(f"⚠️ Position mismatches detected:")
                for mismatch in mismatches:
                    logger.warning(f"   {mismatch}")
                logger.warning(f"   → Syncing to IBKR values")
            
            # Replace our tracking with IBKR truth
            self.positions = actual_positions.copy()

            logger.debug(f"✅ Position sync: {self.position_count} active positions")
        
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate current exposure by sector."""
        exposure = {}
        total_value = 0.0
        
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            
            sector = ApexConfig.get_sector(symbol)
            price = self.price_cache.get(symbol, 0)
            
            if price > 0:
                value = abs(qty * price)  # Absolute value for both long/short
                total_value += value
                exposure[sector] = exposure.get(sector, 0) + value
        
        # Convert to percentages
        if total_value > 0:
            for sector in exposure:
                exposure[sector] = exposure[sector] / total_value
        
        return exposure
    
    async def check_sector_limit(self, symbol: str) -> bool:
        """Check if adding this symbol would exceed sector limits."""
        sector = ApexConfig.get_sector(symbol)
        current_exposure = self.calculate_sector_exposure()
        
        # Check if adding this position would breach limit
        if current_exposure.get(sector, 0) >= ApexConfig.MAX_SECTOR_EXPOSURE:
            logger.warning(f"⚠️ {symbol}: Sector limit reached ({sector}: {current_exposure[sector]*100:.1f}%)")
            return False
        
        return True
    
    async def process_symbol(self, symbol: str):
        """Process symbol with all protections including circuit breaker."""

        # Check 0: Circuit breaker check (use institutional for graduated response)
        if self.use_institutional:
            can_trade, reason, risk_mult = self.inst_risk_manager.can_trade()
            self._risk_multiplier = risk_mult  # Store for position sizing
            if not can_trade:
                logger.debug(f"🛑 {symbol}: Trading halted - {reason}")
                return
            if risk_mult < 1.0:
                logger.debug(f"⚠️ {symbol}: Reduced risk mode ({risk_mult:.0%} position size)")
        else:
            can_trade, reason = self.risk_manager.can_trade()
            self._risk_multiplier = 1.0
            if not can_trade:
                logger.debug(f"🛑 {symbol}: Trading halted - {reason}")
                return

        # Check 1: Cooldown protection
        last_trade = self.last_trade_time.get(symbol, datetime(2000, 1, 1))
        seconds_since = (datetime.now() - last_trade).total_seconds()

        if seconds_since < ApexConfig.TRADE_COOLDOWN_SECONDS:
            logger.debug(f"⏸️  {symbol}: Cooldown ({int(ApexConfig.TRADE_COOLDOWN_SECONDS - seconds_since)}s left)")
            return

        # Check 2: Skip if order pending
        if symbol in self.pending_orders:
            logger.debug(f"⏳ {symbol}: Order pending")
            return
        
        # Get data
        if symbol not in self.historical_data:
            return
        
        try:
            # Use cached positions (refreshed at cycle start) to avoid race conditions
            if self.ibkr:
                # Use cycle-level cached positions if available
                if self._cached_ibkr_positions is not None:
                    current_pos = self._cached_ibkr_positions.get(symbol, 0)
                else:
                    current_pos = self.positions.get(symbol, 0)

                # Get current price
                price = await self.ibkr.get_market_price(symbol)
                if not price or price == 0:
                    logger.debug(f"⚠️ {symbol}: No price available")
                    return

                self.price_cache[symbol] = price
            else:
                current_pos = self.positions.get(symbol, 0)
                price = float(self.historical_data[symbol]['Close'].iloc[-1])
                self.price_cache[symbol] = price
            
            # Generate signal (use institutional or standard)
            prices = self.historical_data[symbol]['Close']

            # SOTA: Get Cross-Sectional Momentum
            cs_data = self.cs_momentum.get_signal(symbol, self.historical_data)
            cs_signal = cs_data.get('signal', 0)
            
            # SOTA: Get News Sentiment
            sent_result = self.sentiment_analyzer.analyze(symbol)
            sent_signal = sent_result.sentiment_score
            sent_conf = sent_result.confidence

            if self.use_institutional:
                # Institutional signal generator with full metadata
                inst_signal: SignalOutput = self.inst_signal_generator.generate_signal(symbol, prices)
                signal = inst_signal.signal
                confidence = inst_signal.confidence
                
                # Blend in SOTA signals
                # Momentum weight: 0.2, Sentiment weight: 0.1
                combined_signal = signal * 0.7 + cs_signal * 0.2 + sent_signal * 0.1
                
                # Confidence adjustment based on component agreement
                if np.sign(signal) == np.sign(cs_signal) == np.sign(sent_signal):
                    confidence = min(1.0, confidence * 1.2)
                
                signal = combined_signal

                # Log component breakdown for quant transparency
                if abs(signal) >= 0.30:
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Breakdown: Tech={inst_signal.signal:.2f} Mom={cs_signal:.2f}({cs_data['rank_percentile']:.0%}) Sent={sent_signal:.2f}")
                    logger.debug(f"   Components: mom={inst_signal.momentum_signal:.2f} rev={inst_signal.mean_reversion_signal:.2f} "
                                f"trend={inst_signal.trend_signal:.2f} vol={inst_signal.volatility_signal:.2f}")
            else:
                # Fallback to standard signal generator
                signal_data = self.signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data['signal']
                confidence = signal_data['confidence']

                # Blend SOTA signals
                signal = signal * 0.7 + cs_signal * 0.2 + sent_signal * 0.1

                # LOG SIGNAL STRENGTH (Quant transparency)
                if abs(signal) >= 0.30:
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Breakdown: ML={signal_data['signal']:.2f} Mom={cs_signal:.2f} Sent={sent_signal:.2f}")

            # ═══════════════════════════════════════════════════════════════
            # ENHANCED SIGNAL FILTERING - Quality gate before entry
            # ═══════════════════════════════════════════════════════════════
            if current_pos == 0 and abs(signal) >= 0.30:  # Only filter potential entries
                # Get volume data if available
                volume = self.historical_data[symbol].get('Volume') if symbol in self.historical_data else None

                # Get cross-sectional rank
                cs_rank = cs_data.get('rank_percentile', 0.5)

                # Build component signals dict for agreement calculation
                if self.use_institutional:
                    component_signals = {
                        'momentum': inst_signal.momentum_signal,
                        'mean_reversion': inst_signal.mean_reversion_signal,
                        'trend': inst_signal.trend_signal,
                        'volatility': inst_signal.volatility_signal,
                        'ml': inst_signal.ml_prediction,
                        'cs_momentum': cs_signal,
                        'sentiment': sent_signal
                    }
                else:
                    component_signals = {
                        'ml': signal_data['signal'],
                        'cs_momentum': cs_signal,
                        'sentiment': sent_signal
                    }

                # Apply enhanced signal filter
                filter_result = self.signal_filter.filter_signal(
                    symbol=symbol,
                    raw_signal=signal,
                    confidence=confidence,
                    component_signals=component_signals,
                    prices=prices,
                    volume=volume,
                    vix_level=self._current_vix,
                    cross_sectional_rank=cs_rank
                )

                # Update signal and confidence based on filter
                if not filter_result['passed']:
                    logger.info(f"🚫 {symbol}: Signal filtered out - {', '.join(filter_result['rejection_reasons'][:2])}")
                    return

                # Use filtered values
                signal = filter_result['filtered_signal']
                confidence = filter_result['filtered_confidence']

                # Log adjustments if any
                if filter_result['adjustments']:
                    logger.debug(f"   Filter adjustments: {', '.join(filter_result['adjustments'][:3])}")
            
            # ═══════════════════════════════════════════════════════════════
            # DYNAMIC EXIT LOGIC - Adapts to market conditions
            # ═══════════════════════════════════════════════════════════════

            if current_pos != 0:  # ✅ Handles both long (pos) and short (neg)
                # ✅ FIX: Skip if already failed too many times - let retry_failed_exits handle exclusively
                if symbol in self.failed_exits and self.failed_exits[symbol].get('attempts', 0) >= 5:
                    logger.debug(f"⏭️ {symbol}: Skipping exit in process_symbol - max attempts reached, requires manual intervention")
                    return

                entry_price = self.position_entry_prices.get(symbol, price)
                entry_time = self.position_entry_times.get(symbol, datetime.now())
                entry_signal = self.position_entry_signals.get(symbol, signal)  # Signal at entry
                side = 'LONG' if current_pos > 0 else 'SHORT'

                # Calculate P&L (works for both long/short)
                if current_pos > 0:  # LONG
                    pnl = (price - entry_price) * current_pos
                    pnl_pct = (price / entry_price - 1) * 100
                else:  # SHORT
                    pnl = (entry_price - price) * abs(current_pos)
                    pnl_pct = (entry_price / price - 1) * 100

                holding_days = (datetime.now() - entry_time).days

                # Get ATR from position stops if available
                pos_stops = self.position_stops.get(symbol, {})
                atr = pos_stops.get('atr', None)

                # Get peak price for trailing stop
                peak_price = self.position_peak_prices.get(symbol, price)

                # Update peak price tracking
                if current_pos > 0 and price > peak_price:
                    self.position_peak_prices[symbol] = price
                    peak_price = price
                elif current_pos < 0 and price < peak_price:
                    self.position_peak_prices[symbol] = price
                    peak_price = price

                # ✅ DYNAMIC EXIT DECISION using exit manager
                should_exit, exit_reason, urgency = self.exit_manager.should_exit(
                    symbol=symbol,
                    entry_price=entry_price,
                    current_price=price,
                    side=side,
                    entry_signal=entry_signal,
                    current_signal=signal,
                    confidence=confidence,
                    regime=self._current_regime,
                    vix_level=self._current_vix,
                    atr=atr,
                    entry_time=entry_time,
                    peak_price=peak_price
                )

                # Log position status periodically
                if holding_days >= 1 and not should_exit:
                    status = self.exit_manager.get_position_status(
                        symbol, entry_price, price, side, entry_signal, signal,
                        confidence, self._current_regime, self._current_vix, atr, entry_time
                    )
                    if status['urgency'] in ['moderate', 'high']:
                        logger.info(f"⚠️ {symbol}: {status['status']} (urgency: {status['urgency']})")
                        logger.debug(f"   Dynamic levels: SL={status['stop_pct']*100:.1f}%, TP={status['target_pct']*100:.1f}%, "
                                    f"max_hold={status['max_hold_days']}d, signal_exit={status['signal_exit_threshold']:.2f}")
                
                if should_exit:
                    pos_type = "LONG" if current_pos > 0 else "SHORT"
                    logger.info(f"🚪 EXIT {symbol} ({pos_type}): {exit_reason}")
                    logger.info(f"   Quantity: {abs(current_pos)}")
                    logger.info(f"   Entry: ${entry_price:.2f} → Current: ${price:.2f}")
                    logger.info(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    logger.info(f"   Holding: {holding_days}d")
                    
                    if self.ibkr:
                        self.pending_orders.add(symbol)
                        
                        # Determine order side
                        order_side = 'SELL' if current_pos > 0 else 'BUY'
                        
                        trade = await self.ibkr.execute_order(
                            symbol=symbol,
                            side=order_side,
                            quantity=abs(current_pos),
                            confidence=abs(signal) if signal != 0 else 0.8
                        )
                        
                        if trade:
                            # ✅ CRITICAL: Force sync after trade
                            await self.sync_positions_with_ibkr()
                            
                            # Track commission
                            commission = ApexConfig.COMMISSION_PER_TRADE
                            self.total_commissions += commission
                            
                            # Clean up tracking
                            if symbol in self.position_entry_prices:
                                del self.position_entry_prices[symbol]
                            if symbol in self.position_entry_times:
                                del self.position_entry_times[symbol]
                            if symbol in self.position_peak_prices:
                                del self.position_peak_prices[symbol]
                            if symbol in self.position_entry_signals:
                                del self.position_entry_signals[symbol]
                            if symbol in self.position_stops:
                                del self.position_stops[symbol]

                            logger.info(f"   ✅ Position closed (commission: ${commission:.2f})")

                            self.live_monitor.log_trade(symbol, order_side, abs(current_pos), price, pnl - commission)
                            self.performance_tracker.record_trade(symbol, order_side, abs(current_pos), price, commission)

                            # Record trade result for circuit breaker
                            self.risk_manager.record_trade_result(pnl - commission)

                            # Update cooldown
                            self.last_trade_time[symbol] = datetime.now()

                            self.pending_orders.discard(symbol)
                            # Clear from failed exits on success
                            if symbol in self.failed_exits:
                                del self.failed_exits[symbol]
                        else:
                            # ✅ Track failed exit for retry
                            self.pending_orders.discard(symbol)
                            attempts = self.failed_exits.get(symbol, {}).get('attempts', 0) + 1
                            self.failed_exits[symbol] = {
                                'reason': exit_reason,
                                'attempts': attempts,
                                'last_attempt': datetime.now(),
                                'quantity': abs(current_pos),
                                'side': order_side
                            }
                            logger.warning(f"   ⚠️ Exit order failed for {symbol} (attempt {attempts})")
                            # Don't apply normal cooldown for failed exits - allow faster retry (30s)
                            if attempts <= 3:
                                self.last_trade_time[symbol] = datetime.now() - timedelta(seconds=ApexConfig.TRADE_COOLDOWN_SECONDS - 30)
                    else:
                        # Simulation mode - close position
                        order_side = 'SELL' if current_pos > 0 else 'BUY'
                        if symbol in self.positions:
                            del self.positions[symbol]
                        if symbol in self.position_entry_prices:
                            del self.position_entry_prices[symbol]
                        if symbol in self.position_entry_times:
                            del self.position_entry_times[symbol]
                        if symbol in self.position_peak_prices:
                            del self.position_peak_prices[symbol]
                        if symbol in self.position_entry_signals:
                            del self.position_entry_signals[symbol]
                        if symbol in self.position_stops:
                            del self.position_stops[symbol]

                        self.live_monitor.log_trade(symbol, order_side, abs(current_pos), price, pnl)
                        self.performance_tracker.record_trade(symbol, order_side, abs(current_pos), price, 0)

                        # Record trade result for circuit breaker
                        self.risk_manager.record_trade_result(pnl)

                        self.last_trade_time[symbol] = datetime.now()

                    return
                
                else:
                    # Update trailing stop peak
                    if symbol not in self.position_peak_prices:
                        self.position_peak_prices[symbol] = price
                    elif current_pos > 0 and price > self.position_peak_prices[symbol]:
                        self.position_peak_prices[symbol] = price
                    elif current_pos < 0 and price < self.position_peak_prices[symbol]:
                        self.position_peak_prices[symbol] = price
                    
                    logger.debug(f"💼 HOLD {symbol}: signal={signal:.3f}, P&L={pnl_pct:+.1f}%")
                    return
            
            # ═══════════════════════════════════════════════════════════════
            # ENTRY LOGIC - Only if no position
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"🔍 {symbol}: Entry evaluation - signal={signal:+.3f} conf={confidence:.3f}")

            # ✅ Phase 3.2: Detect market regime and use regime-adjusted thresholds
            try:
                regime_enum = self.god_signal_generator.detect_market_regime(prices)
                self._current_regime = regime_enum.value  # Convert enum to string
            except Exception:
                self._current_regime = 'neutral'

            # ✅ Phase 3.1: Get regime-adjusted signal threshold
            signal_threshold = ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME.get(
                self._current_regime, ApexConfig.MIN_SIGNAL_THRESHOLD
            )

            if abs(signal) < signal_threshold:
                logger.info(f"⏭️ {symbol}: Signal {signal:.3f} below regime threshold {signal_threshold} ({self._current_regime})")
                return

            # ✅ Phase 1.2: Check VaR limit before new entries
            if self.use_institutional and len(self.positions) > 0:
                portfolio_risk = self.inst_risk_manager.calculate_portfolio_risk(
                    self.positions,
                    self.price_cache,
                    self.historical_data
                )
                max_var = self.capital * ApexConfig.MAX_PORTFOLIO_VAR
                if portfolio_risk.var_95 > max_var:
                    logger.warning(f"⚠️ {symbol}: VaR limit exceeded (${portfolio_risk.var_95:,.0f} > ${max_var:,.0f}) - blocking entry")
                    return

            # ✅ Phase 1.3: Check portfolio correlation before new entries
            if ApexConfig.USE_CORRELATION_MANAGER and len(self.positions) > 1:
                existing_symbols = [s for s, qty in self.positions.items() if qty != 0]
                avg_corr = self.correlation_manager.get_average_correlation(symbol, existing_symbols)
                if avg_corr > ApexConfig.MAX_PORTFOLIO_CORRELATION:
                    logger.warning(f"⚠️ {symbol}: Correlation too high ({avg_corr:.2f} > {ApexConfig.MAX_PORTFOLIO_CORRELATION}) - blocking entry")
                    return

            # ✅ CRITICAL: Use lock to check position count atomically
            # This prevents race condition where multiple parallel tasks pass the check
            async with self._position_lock:
                if self.position_count >= ApexConfig.MAX_POSITIONS:
                    logger.info(f"⚠️ {symbol}: Max positions reached ({self.position_count}/{ApexConfig.MAX_POSITIONS})")
                    return

                # ✅ Check sector limits (inside lock)
                if not await self.check_sector_limit(symbol):
                    return

                # Reserve the position slot (prevents race condition)
                # We'll update with actual quantity after trade or remove if failed
                self.positions[symbol] = 1 if signal > 0 else -1  # Placeholder

            # ✅ Use semaphore to limit concurrent entry attempts
            trade_success = False
            async with self._entry_semaphore:
                try:
                    # SOTA: Check data quality before entry
                    dq_issues = self.data_quality_monitor.run_all_checks(symbol, prices=prices)
                    if any(i.severity in ['error', 'critical'] for i in dq_issues):
                        logger.warning(f"🛑 {symbol}: Data quality issues block entry: {[i.message for i in dq_issues]}")
                        async with self._position_lock:
                            if symbol in self.positions:
                                del self.positions[symbol]
                        return

                    # ✅ Calculate position size with institutional risk manager
                    if self.use_institutional:
                        sector = ApexConfig.get_sector(symbol)
                        sizing: SizingResult = self.inst_risk_manager.calculate_position_size(
                            symbol=symbol,
                            price=price,
                            signal_strength=signal,
                            signal_confidence=confidence,
                            current_positions=self.positions,
                            price_cache=self.price_cache,
                            sector=sector,
                            historical_prices=prices
                        )

                        shares = sizing.target_shares

                        # ✅ Apply POSITION_SIZE_USD as additional cap
                        max_shares_by_value = int(ApexConfig.POSITION_SIZE_USD / price)
                        shares = min(shares, max_shares_by_value)
                        shares = min(shares, ApexConfig.MAX_SHARES_PER_POSITION)  # Cap max shares
                        
                        # SOTA: Apply VIX-based risk multiplier
                        shares = int(shares * self._vix_risk_multiplier)

                        # ✅ Phase 1.4: Apply graduated circuit breaker risk multiplier
                        if self._risk_multiplier < 1.0:
                            shares = int(shares * self._risk_multiplier)
                            logger.info(f"   ⚠️ Risk reduced: {self._risk_multiplier:.0%} (VIX: {self._vix_risk_multiplier:.2f}) → {shares} shares")

                        if sizing.constraints:
                            logger.debug(f"   Size constraints: {', '.join(sizing.constraints)}")

                        logger.info(f"🔢 {symbol}: Sizing result - shares={shares} (inst={sizing.target_shares}, max_val={max_shares_by_value}, vix_mult={self._vix_risk_multiplier:.2f}, risk_mult={self._risk_multiplier:.2f})")
                        if sizing.constraints:
                            logger.info(f"   📋 Constraints: {', '.join(sizing.constraints)}")

                        if shares < 1:
                            if sizing.constraints:
                                logger.info(f"⚠️ {symbol}: Position blocked by {sizing.constraints}")
                            else:
                                logger.info(f"⚠️ {symbol}: Price too high or risk too high (${price:.2f})")
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return
                    else:
                        # Fallback: standard position sizing
                        shares = int(ApexConfig.POSITION_SIZE_USD / price)
                        shares = min(shares, ApexConfig.MAX_SHARES_PER_POSITION)
                        # SOTA: Apply VIX multiplier
                        shares = int(shares * self._vix_risk_multiplier)

                        if shares < 1:
                            logger.debug(f"⚠️ {symbol}: Price too high (${price:.2f})")
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                    # Determine side (long or short)
                    side = 'BUY' if signal > 0 else 'SELL'

                    logger.info(f"📈 {side} {shares} {symbol} @ ${price:.2f} (${shares*price:,.0f})")
                    logger.info(f"   Signal: {signal:+.3f} | Confidence: {confidence:.3f}")
                    if self.use_institutional:
                        logger.debug(f"   Vol-adjusted: ${sizing.vol_adjusted_size:,.0f} | Corr penalty: {sizing.correlation_penalty:.2f}")

                    if self.ibkr:
                        self.pending_orders.add(symbol)
                        
                        # SOTA: Record arrival price
                        arrival_price = price
                        start_time = datetime.now()

                        # ✅ Phase 2.1: Use TWAP/VWAP for large orders
                        order_value = shares * price
                        use_advanced = (
                            ApexConfig.USE_ADVANCED_EXECUTION and
                            self.advanced_executor is not None and
                            order_value >= ApexConfig.LARGE_ORDER_THRESHOLD
                        )

                        if use_advanced:
                            # Use TWAP for large orders to reduce market impact
                            logger.info(f"   📊 Using TWAP execution (order value: ${order_value:,.0f})")
                            trade = await self.advanced_executor.execute_twap_order(
                                symbol=symbol,
                                side=side,
                                total_quantity=shares,
                                time_horizon_minutes=30,  # Execute over 30 minutes
                                slice_interval_seconds=60  # Execute every minute
                            )
                        else:
                            trade = await self.ibkr.execute_order(
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                confidence=confidence
                            )
                        
                        # SOTA: Record execution benchmark if we have a trade
                        if trade:
                            fill_price = trade.get('price', 0) if isinstance(trade, dict) else trade.orderStatus.avgFillPrice
                            if fill_price > 0:
                                self.arrival_benchmark.record_execution(
                                    symbol=symbol,
                                    side=side,
                                    quantity=shares,
                                    arrival_price=arrival_price,
                                    fill_price=fill_price,
                                    decision_time=start_time
                                )
                                shortfall = self.arrival_benchmark.executions[-1].implementation_shortfall_bps
                                logger.info(f"   📊 Execution Shortfall: {shortfall:.1f} bps")

                        if trade:
                            trade_success = True
                            # ✅ CRITICAL: Force sync after trade
                            await self.sync_positions_with_ibkr()

                            # Track commission
                            commission = ApexConfig.COMMISSION_PER_TRADE
                            self.total_commissions += commission

                            self.position_entry_prices[symbol] = price
                            self.position_entry_times[symbol] = datetime.now()
                            self.position_peak_prices[symbol] = price
                            self.position_entry_signals[symbol] = signal  # Track entry signal for dynamic exits

                            # ✅ Calculate ATR-based dynamic stops using GodLevelRiskManager
                            if ApexConfig.USE_ATR_STOPS:
                                god_sizing = self.god_risk_manager.calculate_position_size(
                                    symbol=symbol,
                                    entry_price=price,
                                    signal_strength=signal,
                                    confidence=confidence,
                                    prices=prices,
                                    regime=self._current_regime  # ✅ Phase 3.2: Use detected regime
                                )
                                self.position_stops[symbol] = {
                                    'stop_loss': god_sizing['stop_loss'],
                                    'take_profit': god_sizing['take_profit'],
                                    'trailing_stop_pct': god_sizing['trailing_stop_pct'],
                                    'atr': god_sizing['atr']
                                }
                                logger.info(f"   🎯 ATR Stops: SL=${god_sizing['stop_loss']:.2f} TP=${god_sizing['take_profit']:.2f} Trail={god_sizing['trailing_stop_pct']*100:.1f}%")
                            else:
                                # Fallback to fixed percentage stops
                                self.position_stops[symbol] = {
                                    'stop_loss': price * 0.95 if signal > 0 else price * 1.05,
                                    'take_profit': price * 1.15 if signal > 0 else price * 0.85,
                                    'trailing_stop_pct': 0.03,
                                    'atr': 0
                                }

                            logger.info(f"   ✅ Order placed (commission: ${commission:.2f})")

                            self.live_monitor.log_trade(symbol, side, shares, price, -commission)
                            self.performance_tracker.record_trade(symbol, side, shares, price, commission)

                            # ✅ Update cooldown
                            self.last_trade_time[symbol] = datetime.now()

                            self.pending_orders.discard(symbol)
                        else:
                            self.pending_orders.discard(symbol)
                    else:
                        # Simulation mode - open new position
                        trade_success = True
                        qty = shares if side == 'BUY' else -shares
                        self.positions[symbol] = qty
                        self.position_entry_prices[symbol] = price
                        self.position_entry_times[symbol] = datetime.now()
                        self.position_peak_prices[symbol] = price
                        self.position_entry_signals[symbol] = signal  # Track entry signal for dynamic exits

                        # ✅ Calculate ATR-based dynamic stops using GodLevelRiskManager
                        if ApexConfig.USE_ATR_STOPS:
                            god_sizing = self.god_risk_manager.calculate_position_size(
                                symbol=symbol,
                                entry_price=price,
                                signal_strength=signal,
                                confidence=confidence,
                                prices=prices,
                                regime=self._current_regime  # ✅ Phase 3.2: Use detected regime
                            )
                            self.position_stops[symbol] = {
                                'stop_loss': god_sizing['stop_loss'],
                                'take_profit': god_sizing['take_profit'],
                                'trailing_stop_pct': god_sizing['trailing_stop_pct'],
                                'atr': god_sizing['atr']
                            }
                            logger.info(f"   🎯 ATR Stops: SL=${god_sizing['stop_loss']:.2f} TP=${god_sizing['take_profit']:.2f} Trail={god_sizing['trailing_stop_pct']*100:.1f}%")
                        else:
                            self.position_stops[symbol] = {
                                'stop_loss': price * 0.95 if signal > 0 else price * 1.05,
                                'take_profit': price * 1.15 if signal > 0 else price * 0.85,
                                'trailing_stop_pct': 0.03,
                                'atr': 0
                            }

                        self.live_monitor.log_trade(symbol, side, shares, price, 0)
                        self.performance_tracker.record_trade(symbol, side, shares, price, 0)
                        self.last_trade_time[symbol] = datetime.now()

                finally:
                    # ✅ CRITICAL: Clean up placeholder if trade failed
                    if not trade_success and symbol in self.positions:
                        if self.positions.get(symbol) in [1, -1]:  # Was a placeholder
                            del self.positions[symbol]
                            logger.debug(f"   ⚠️ {symbol}: Removed position placeholder (trade failed)")
        
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    async def retry_failed_exits(self):
        """
        Retry exit orders that previously failed.
        This ensures positions get closed even if IBKR had temporary issues.
        """
        if not self.failed_exits or not self.ibkr:
            return

        now = datetime.now()
        symbols_to_retry = []

        for symbol, info in list(self.failed_exits.items()):
            # Skip if too many attempts (max 5)
            if info['attempts'] >= 5:
                logger.error(f"❌ {symbol}: Exit failed after 5 attempts - manual intervention required!")
                continue

            # Retry after 30 seconds
            seconds_since_attempt = (now - info['last_attempt']).total_seconds()
            if seconds_since_attempt >= 30:
                symbols_to_retry.append((symbol, info))

        for symbol, info in symbols_to_retry:
            # Check if position still exists
            current_pos = self.positions.get(symbol, 0)
            if current_pos == 0:
                # Position already closed (maybe manually)
                del self.failed_exits[symbol]
                logger.info(f"✅ {symbol}: Failed exit cleared - position no longer exists")
                continue

            logger.info(f"🔄 Retrying exit for {symbol} (attempt {info['attempts'] + 1})")

            try:
                self.pending_orders.add(symbol)
                order_side = 'SELL' if current_pos > 0 else 'BUY'

                trade = await self.ibkr.execute_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=abs(current_pos),
                    confidence=0.9  # High confidence for exits
                )

                if trade:
                    await self.sync_positions_with_ibkr()
                    logger.info(f"   ✅ {symbol}: Exit retry successful!")

                    # Clean up tracking
                    for tracking_dict in [self.position_entry_prices, self.position_entry_times,
                                          self.position_peak_prices, self.position_stops, self.failed_exits]:
                        if symbol in tracking_dict:
                            del tracking_dict[symbol]

                    self.pending_orders.discard(symbol)
                else:
                    # Increment attempt counter
                    self.failed_exits[symbol]['attempts'] += 1
                    self.failed_exits[symbol]['last_attempt'] = now
                    self.pending_orders.discard(symbol)
                    logger.warning(f"   ⚠️ {symbol}: Exit retry failed (attempt {self.failed_exits[symbol]['attempts']})")

            except Exception as e:
                logger.error(f"   ❌ {symbol}: Exit retry error: {e}")
                self.pending_orders.discard(symbol)
                self.failed_exits[symbol]['attempts'] += 1
                self.failed_exits[symbol]['last_attempt'] = now

    async def process_symbols_parallel(self, symbols: List[str]):
        """
        Process symbols in batches to respect IBKR's 100 market data line limit.

        IBKR limits simultaneous market data subscriptions to 100.
        We process in batches of 50 with cleanup between batches.
        """
        BATCH_SIZE = 50  # Max symbols per batch (IBKR limit is 100, we use 50 for safety)

        total = len(symbols)
        num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        if num_batches > 1:
            logger.debug(f"📊 Processing {total} symbols in {num_batches} batches ({BATCH_SIZE} max per batch)")

        for batch_num in range(num_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total)
            batch_symbols = symbols[start_idx:end_idx]

            # Process batch in parallel
            tasks = [self.process_symbol(symbol) for symbol in batch_symbols]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay between batches to allow cleanup of market data subscriptions
            if batch_num < num_batches - 1:
                await asyncio.sleep(0.5)

    async def check_and_execute_rebalance(self, est_hour: float):
        """
        Check if rebalancing is needed and execute if conditions are met.

        Args:
            est_hour: Current hour in EST
        """
        if not ApexConfig.REBALANCE_ENABLED:
            return

        # Check if it's a good time to rebalance
        if not self.portfolio_optimizer.should_rebalance_now(est_hour):
            return

        # Check circuit breaker
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"🛑 Rebalancing skipped - {reason}")
            return

        try:
            # Get current portfolio value
            if self.ibkr:
                total_value = await self.ibkr.get_portfolio_value()
            else:
                total_value = self.capital
                for symbol, qty in self.positions.items():
                    price = self.price_cache.get(symbol, 0)
                    if price and qty:
                        total_value += abs(qty) * price

            # Check if rebalancing is needed
            needs_rebal, reason = self.portfolio_optimizer.needs_rebalance(
                self.positions,
                self.price_cache
            )

            if not needs_rebal:
                logger.debug(f"📊 Rebalance check: {reason}")
                return

            logger.info(f"📊 Rebalancing triggered: {reason}")

            # Calculate current weights before rebalance
            before_weights = self.portfolio_optimizer.calculate_current_weights(
                self.positions, self.price_cache
            )

            # Calculate rebalance trades
            trades = self.portfolio_optimizer.calculate_rebalance_trades(
                self.positions,
                self.price_cache,
                total_value
            )

            if not trades:
                logger.info("📊 No rebalancing trades needed")
                return

            logger.info(f"📊 Executing {len(trades)} rebalance trades...")

            # Execute trades
            for symbol, trade_qty in trades.items():
                if trade_qty == 0:
                    continue

                side = 'BUY' if trade_qty > 0 else 'SELL'
                qty = abs(trade_qty)

                logger.info(f"   {side} {qty} {symbol}")

                if self.ibkr:
                    result = await self.ibkr.execute_order(
                        symbol, side, qty,
                        confidence=0.7,
                        force_market=True  # Use market orders for rebalancing
                    )

                    if result and result.get('status') == 'FILLED':
                        self.total_commissions += ApexConfig.COMMISSION_PER_TRADE
                else:
                    # Simulation mode
                    price = self.price_cache.get(symbol, 0)
                    if price > 0:
                        if trade_qty > 0:
                            self.positions[symbol] = self.positions.get(symbol, 0) + qty
                        else:
                            self.positions[symbol] = self.positions.get(symbol, 0) - qty

            # Calculate weights after rebalance
            after_weights = self.portfolio_optimizer.calculate_current_weights(
                self.positions, self.price_cache
            )

            # Record rebalance
            self.portfolio_optimizer.record_rebalance(trades, before_weights, after_weights)

            logger.info("✅ Rebalancing complete")

        except Exception as e:
            logger.error(f"❌ Rebalancing error: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    async def manage_options(self):
        """
        Manage options positions:
        - Auto-hedge large stock positions with protective puts
        - Sell covered calls on eligible long positions
        - Monitor and roll expiring options
        """
        if not self.options_trader or not ApexConfig.OPTIONS_ENABLED:
            return

        try:
            logger.debug("🎯 Checking options opportunities...")

            # Get current stock positions value
            for symbol, qty in self.positions.items():
                if qty <= 0:  # Only hedge long positions
                    continue

                price = self.price_cache.get(symbol, 0)
                if price <= 0:
                    continue

                position_value = qty * price

                # Auto-hedge large positions with protective puts
                if ApexConfig.OPTIONS_AUTO_HEDGE and position_value >= ApexConfig.OPTIONS_HEDGE_THRESHOLD:
                    # Check if already hedged
                    hedge_key = f"{symbol}_hedge"
                    if hedge_key not in self.options_positions:
                        logger.info(f"🛡️ Auto-hedging {symbol}: ${position_value:,.2f} position")

                        result = await self.options_trader.buy_protective_put(
                            symbol=symbol,
                            shares=qty,
                            delta=ApexConfig.OPTIONS_HEDGE_DELTA,
                            days_to_expiry=ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY
                        )

                        if result:
                            self.options_positions[hedge_key] = result
                            logger.info(f"   ✅ Protective put purchased: {result.get('contract', {}).get('strike')} strike")

                # Sell covered calls on eligible positions
                if ApexConfig.OPTIONS_COVERED_CALLS_ENABLED and qty >= ApexConfig.OPTIONS_MIN_SHARES_FOR_COVERED_CALL:
                    cc_key = f"{symbol}_cc"
                    if cc_key not in self.options_positions:
                        logger.info(f"💰 Selling covered call on {symbol}: {qty} shares")

                        result = await self.options_trader.sell_covered_call(
                            symbol=symbol,
                            shares=qty,
                            delta=ApexConfig.OPTIONS_COVERED_CALL_DELTA,
                            days_to_expiry=ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY
                        )

                        if result:
                            self.options_positions[cc_key] = result
                            logger.info(f"   ✅ Covered call sold: ${result.get('premium', 0):,.2f} premium")

            # Check for expiring options (within 7 days)
            await self._check_expiring_options()

            # Log portfolio Greeks
            if self.options_positions:
                greeks = self.options_trader.get_portfolio_greeks()
                logger.debug(f"📊 Portfolio Greeks - Delta: {greeks['delta']:.1f}, Theta: ${greeks['theta']:.2f}/day")

        except Exception as e:
            logger.error(f"❌ Options management error: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    async def _check_expiring_options(self):
        """Check and handle options expiring soon."""
        if not self.options_trader:
            return

        try:
            # Refresh options positions from IBKR
            if self.ibkr:
                ibkr_options = await self.ibkr.get_option_positions()

                for pos in ibkr_options.values():
                    expiry_str = pos.get('expiry', '')
                    if expiry_str:
                        try:
                            expiry_date = datetime.strptime(expiry_str, '%Y%m%d')
                            days_to_expiry = (expiry_date - datetime.now()).days

                            if days_to_expiry <= 7:
                                symbol = pos.get('symbol')
                                strike = pos.get('strike')
                                right = pos.get('right')
                                qty = pos.get('quantity', 0)

                                logger.warning(f"⚠️ Option expiring soon: {symbol} {expiry_str} ${strike} {'CALL' if right == 'C' else 'PUT'} ({days_to_expiry} days)")

                                # For protective puts near expiration, consider rolling
                                if right == 'P' and qty > 0 and days_to_expiry <= 3:
                                    logger.info(f"   🔄 Consider rolling protective put for {symbol}")

                        except ValueError:
                            pass

        except Exception as e:
            logger.debug(f"Error checking expiring options: {e}")

    async def execute_option_trade(
        self,
        symbol: str,
        strategy: OptionStrategy,
        contracts: int = 1,
        **kwargs
    ) -> Optional[dict]:
        """
        Execute an options trade based on strategy.

        Args:
            symbol: Underlying symbol
            strategy: Options strategy to execute
            contracts: Number of contracts
            **kwargs: Strategy-specific parameters

        Returns:
            Trade result or None
        """
        if not self.options_trader:
            logger.error("Options trader not initialized")
            return None

        try:
            if strategy == OptionStrategy.PROTECTIVE_PUT:
                shares = kwargs.get('shares', contracts * 100)
                return await self.options_trader.buy_protective_put(
                    symbol=symbol,
                    shares=shares,
                    delta=kwargs.get('delta', ApexConfig.OPTIONS_HEDGE_DELTA),
                    days_to_expiry=kwargs.get('days', ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY)
                )

            elif strategy == OptionStrategy.COVERED_CALL:
                shares = kwargs.get('shares', contracts * 100)
                return await self.options_trader.sell_covered_call(
                    symbol=symbol,
                    shares=shares,
                    delta=kwargs.get('delta', ApexConfig.OPTIONS_COVERED_CALL_DELTA),
                    days_to_expiry=kwargs.get('days', ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY)
                )

            elif strategy == OptionStrategy.STRADDLE:
                return await self.options_trader.buy_straddle(
                    symbol=symbol,
                    contracts=contracts,
                    days_to_expiry=kwargs.get('days', ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY)
                )

            elif strategy == OptionStrategy.LONG_CALL:
                chain = await self.options_trader.get_options_chain(
                    symbol,
                    min_days=ApexConfig.OPTIONS_MIN_DAYS_TO_EXPIRY,
                    max_days=ApexConfig.OPTIONS_MAX_DAYS_TO_EXPIRY
                )
                if chain:
                    call = self.options_trader.select_option_by_delta(
                        chain,
                        target_delta=kwargs.get('delta', 0.50),
                        option_type=OptionType.CALL
                    )
                    if call:
                        expiry_str = call.expiry.strftime('%Y%m%d')
                        return await self.ibkr.execute_option_order(
                            symbol=symbol,
                            expiry=expiry_str,
                            strike=call.strike,
                            right='C',
                            side='BUY',
                            quantity=contracts
                        )

            elif strategy == OptionStrategy.LONG_PUT:
                chain = await self.options_trader.get_options_chain(
                    symbol,
                    min_days=ApexConfig.OPTIONS_MIN_DAYS_TO_EXPIRY,
                    max_days=ApexConfig.OPTIONS_MAX_DAYS_TO_EXPIRY
                )
                if chain:
                    put = self.options_trader.select_option_by_delta(
                        chain,
                        target_delta=kwargs.get('delta', -0.50),
                        option_type=OptionType.PUT
                    )
                    if put:
                        expiry_str = put.expiry.strftime('%Y%m%d')
                        return await self.ibkr.execute_option_order(
                            symbol=symbol,
                            expiry=expiry_str,
                            strike=put.strike,
                            right='P',
                            side='BUY',
                            quantity=contracts
                        )

            else:
                logger.warning(f"Strategy {strategy} not yet implemented")
                return None

        except Exception as e:
            logger.error(f"❌ Option trade error: {e}")
            return None

    async def check_risk(self):
        """Check risk limits and update metrics."""
        try:
            if self.ibkr:
                current_value = await self.ibkr.get_portfolio_value()
            else:
                current_value = self.capital
                for symbol, qty in self.positions.items():
                    price = self.price_cache.get(symbol, 0)
                    if price and qty:
                        if qty > 0:  # Long
                            current_value += float(qty) * float(price)
                        else:  # Short (qty is negative)
                            current_value += float(qty) * float(price)

            loss_check = self.risk_manager.check_daily_loss(current_value)
            dd_check = self.risk_manager.check_drawdown(current_value)
            self.performance_tracker.record_equity(current_value)

            # Update institutional components
            if self.use_institutional:
                self.inst_risk_manager.update_capital(current_value)
                self.inst_metrics.record_equity(datetime.now(), current_value)

                # Calculate portfolio risk with institutional risk manager
                portfolio_risk = self.inst_risk_manager.calculate_portfolio_risk(
                    self.positions,
                    self.price_cache,
                    self.historical_data
                )

            try:
                sharpe = self.performance_tracker.get_sharpe_ratio()
            except:
                sharpe = 0.0

            try:
                win_rate = self.performance_tracker.get_win_rate()
            except:
                win_rate = 0.0

            total_trades = len(self.performance_tracker.trades)

            # Calculate sector exposure
            sector_exp = self.calculate_sector_exposure()

            logger.info("")
            logger.info("═" * 80)
            logger.info(f"💼 Portfolio: ${current_value:,.2f}")
            logger.info(f"📊 Daily P&L: ${loss_check['daily_pnl']:+,.2f} ({loss_check['daily_return']*100:+.2f}%)")
            logger.info(f"📉 Drawdown: {dd_check['drawdown']*100:.2f}%")
            logger.info(f"📦 Positions: {self.position_count}/{ApexConfig.MAX_POSITIONS}")
            logger.info(f"⏳ Pending: {len(self.pending_orders)}")
            logger.info(f"💸 Total Commissions: ${self.total_commissions:,.2f}")
            logger.info(f"📈 Sharpe: {sharpe:.2f} | Win Rate: {win_rate*100:.1f}% | Trades: {total_trades}")

            # Institutional risk metrics
            if self.use_institutional:
                logger.info(f"🏛️  INSTITUTIONAL RISK:")
                logger.info(f"   Portfolio Vol: {portfolio_risk.portfolio_volatility:.1%} | VaR(95%): ${portfolio_risk.var_95:,.0f}")
                logger.info(f"   Risk Level: {portfolio_risk.risk_level.value.upper()} | Risk Mult: {portfolio_risk.risk_multiplier:.2f}")
                logger.info(f"   Gross Exp: ${portfolio_risk.gross_exposure:,.0f} | Net Exp: ${portfolio_risk.net_exposure:,.0f}")
                logger.info(f"   Concentration (HHI): {portfolio_risk.herfindahl_index:.3f}")

            if sector_exp:
                logger.info(f"🏢 Sector Exposure:")
                for sector, pct in sorted(sector_exp.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"   {sector}: {pct*100:.1f}%")
            
            if self.positions:
                position_list = []
                for symbol, qty in self.positions.items():
                    if qty != 0:
                        position_list.append((symbol, qty))

                sorted_positions = sorted(position_list, key=lambda x: abs(x[1]), reverse=True)

                logger.info(f"📊 Active Positions ({len(sorted_positions)}):")
                for symbol, qty in sorted_positions:  # Show ALL positions, not just top 5
                    try:
                        pos_type = "LONG" if qty > 0 else "SHORT"
                        price = self.price_cache.get(symbol, 0)
                        
                        if price:
                            value = abs(qty) * price
                            entry = self.position_entry_prices.get(symbol, price)
                            
                            if qty > 0:
                                pnl_pct = (price / entry - 1) * 100
                            else:
                                pnl_pct = (entry / price - 1) * 100
                            
                            logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) ${price:.2f} P&L:{pnl_pct:+.1f}%")
                    except Exception as e:
                        logger.debug(f"Error displaying {symbol}: {e}")
            
            logger.info("═" * 80)
            logger.info("")
            
            # Export to dashboard
            self.export_dashboard_state()
            self.export_trades_history()
            self.export_equity_curve()
            
            if loss_check.get('breached', False):
                logger.error("🚨 DAILY LOSS LIMIT BREACHED!")
            
            if dd_check.get('breached', False):
                logger.error("🚨 MAX DRAWDOWN BREACHED!")
                await self.close_all_positions()
        
        except Exception as e:
            logger.error(f"❌ Risk check error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    async def close_all_positions(self):
        """Emergency: close all positions."""
        logger.warning("⚠️  EMERGENCY: Closing all positions...")
        
        for symbol, qty in list(self.positions.items()):
            if qty != 0:
                if self.ibkr:
                    price = await self.ibkr.get_market_price(symbol)
                    side = 'SELL' if qty > 0 else 'BUY'
                    
                    await self.ibkr.execute_order(symbol, side, abs(qty), confidence=1.0, force_market=True)
                    
                    entry_price = self.position_entry_prices.get(symbol, price)
                    
                    if qty > 0:
                        pnl = (price - entry_price) * qty
                    else:
                        pnl = (entry_price - price) * abs(qty)
                    
                    self.live_monitor.log_trade(symbol, side, abs(qty), price, pnl)
                
                logger.info(f"   ✅ Closed {symbol}: {abs(qty)} shares")
        
        self.positions = {}
        self.position_entry_prices = {}
        self.position_entry_times = {}
        self.position_peak_prices = {}
        self.pending_orders.clear()
        logger.warning("✅ All positions closed")
    
    async def refresh_data(self):
        """Refresh market data periodically."""
        try:
            logger.info("🔄 Refreshing market data...")
            updated = 0
            
            for symbol in list(self.historical_data.keys())[:50]:  # Limit refresh
                try:
                    data = self.market_data.fetch_historical_data(symbol, days=100)
                    if not data.empty:
                        self.historical_data[symbol] = data
                        updated += 1
                except:
                    pass
            
            logger.info(f"✅ Refreshed {updated} symbols")
            
            if updated > 30:
                logger.info("🧠 Re-training ML models...")
                self.signal_generator.train_models(self.historical_data)
                logger.info("✅ ML models updated")
        
        except Exception as e:
            logger.error(f"❌ Data refresh error: {e}")
    
    def get_current_signals(self) -> dict:
        """Get current signals for dashboard."""
        signals = {}
        
        for symbol in list(self.historical_data.keys())[:50]:
            try:
                price_data = self.historical_data[symbol]
                signal_data = self.signal_generator.generate_ml_signal(symbol, price_data)

                signals[symbol] = {
                    'signal': signal_data['signal'],
                    'confidence': signal_data['confidence'],
                    'direction': 'STRONG BUY' if signal_data['signal'] > 0.60 else
                                'BUY' if signal_data['signal'] > 0.40 else
                                'WEAK BUY' if signal_data['signal'] > 0.20 else
                                'NEUTRAL' if signal_data['signal'] > -0.20 else
                                'WEAK SELL' if signal_data['signal'] > -0.40 else
                                'SELL' if signal_data['signal'] > -0.60 else
                                'STRONG SELL',
                    'strength_pct': abs(signal_data['signal']) * 100,
                    'timestamp': datetime.now().isoformat()
                }
            except:
                pass
        
        return signals
    
    def export_dashboard_state(self):
        """Export current state for dashboard."""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            if self.ibkr:
                try:
                    # Get NetLiquidation (total account equity) from IBKR
                    account_values = self.ibkr.ib.accountValues()
                    current_value = self.capital  # fallback
                    for av in account_values:
                        if av.tag == 'NetLiquidation' and av.currency == 'USD':
                            current_value = float(av.value)
                            break
                except:
                    current_value = self.capital
            else:
                current_value = self.capital
            
            current_signals = self.get_current_signals()
            
            state = {
                'timestamp': datetime.now().isoformat(),
                'capital': float(current_value),
                'starting_capital': float(self.risk_manager.starting_capital),
                'positions': {},
                'signals': current_signals,
                'daily_pnl': float(current_value - self.risk_manager.day_start_capital) if self.risk_manager.day_start_capital > 0 else 0.0,
                'total_pnl': float(current_value - self.risk_manager.starting_capital),
                'total_commissions': float(self.total_commissions),
                'max_drawdown': float(self.performance_tracker.get_max_drawdown()),
                'sharpe_ratio': float(self.performance_tracker.get_sharpe_ratio()),
                'win_rate': float(self.performance_tracker.get_win_rate()),
                'total_trades': len(self.performance_tracker.trades),
                'open_positions': self.position_count,
                'sector_exposure': self.calculate_sector_exposure()
            }
            
            for symbol, qty in self.positions.items():
                if qty == 0:
                    continue
                
                try:
                    price = self.price_cache.get(symbol, 0)
                    if price == 0:
                        continue
                    
                    avg_price = self.position_entry_prices.get(symbol, price)
                    entry_time = self.position_entry_times.get(symbol, datetime.now())
                    
                    if qty > 0:  # Long
                        pnl = (price - avg_price) * qty
                        pnl_pct = (price / avg_price - 1) * 100
                    else:  # Short
                        pnl = (avg_price - price) * abs(qty)
                        pnl_pct = (avg_price / price - 1) * 100
                    
                    state['positions'][symbol] = {
                        'qty': int(qty),
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'avg_price': float(avg_price),
                        'current_price': float(price),
                        'pnl': float(pnl),
                        'pnl_pct': float(pnl_pct),
                        'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
                        'current_signal': current_signals.get(symbol, {}).get('signal', 0),
                        'signal_direction': current_signals.get(symbol, {}).get('direction', 'UNKNOWN')
                    }
                except Exception as e:
                    logger.debug(f"Error adding position {symbol}: {e}")
            
            state_file = data_dir / "trading_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"📊 Dashboard state exported")
        
        except Exception as e:
            logger.error(f"❌ Dashboard export error: {e}")
    
    def export_trades_history(self):
        """Export trades to CSV."""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            if len(self.performance_tracker.trades) == 0:
                return
            
            trades_data = []
            for trade in self.performance_tracker.trades:
                trades_data.append({
                    'timestamp': trade['timestamp'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'commission': trade.get('commission', 0),
                    'pnl': trade.get('pnl', 0)
                })
            
            df = pd.DataFrame(trades_data)
            trades_file = data_dir / "trades.csv"
            df.to_csv(trades_file, index=False)
            logger.debug(f"📊 Trades exported ({len(trades_data)} trades)")
        
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
    
    def export_equity_curve(self):
        """Export equity curve."""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            if len(self.performance_tracker.equity_curve) == 0:
                return
            
            equity_data = []
            peak = float(self.risk_manager.starting_capital)
            
            for timestamp, value in self.performance_tracker.equity_curve:
                try:
                    value = float(value)
                except:
                    continue
                
                peak = max(peak, value)
                drawdown = (value - peak) / peak if peak > 0 else 0
                
                equity_data.append({
                    'timestamp': timestamp,
                    'equity': value,
                    'drawdown': drawdown
                })
            
            if len(equity_data) == 0:
                return
            
            df = pd.DataFrame(equity_data)
            equity_file = data_dir / "equity_curve.csv"
            df.to_csv(equity_file, index=False)
            logger.debug(f"📊 Equity curve exported ({len(equity_data)} points)")
        
        except Exception as e:
            logger.error(f"Error exporting equity curve: {e}")
    
    def _export_initial_state(self):
        """Export initial state."""
        try:
            state = {
                'capital': self.capital,
                'starting_capital': self.capital,
                'positions': {},
                'daily_pnl': 0.0,
                'total_pnl': 0.0,
                'total_commissions': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'open_positions': self.position_count
            }
            self.live_monitor.update_state(state)
            self.export_dashboard_state()
            logger.info("📱 Dashboard initialized")
        except Exception as e:
            logger.warning(f"⚠️  Dashboard export failed: {e}")
    
    async def run(self):
        """Run the complete system."""
        try:
            await self.initialize()
            
            logger.info("▶️  Starting trading loop...")
            logger.info(f"   Interval: {ApexConfig.CHECK_INTERVAL_SECONDS}s")
            logger.info(f"   Hours: {ApexConfig.TRADING_HOURS_START:.1f} - {ApexConfig.TRADING_HOURS_END:.1f} EST")
            logger.info(f"   🛡️  Protection: {ApexConfig.TRADE_COOLDOWN_SECONDS}s cooldown")
            logger.info(f"   🚀 Parallel processing enabled")
            logger.info(f"   📱 Dashboard: streamlit run dashboard/streamlit_app.py")
            logger.info("")
            
            self.is_running = True
            cycle = 0
            last_data_refresh = datetime.now()
            
            while self.is_running:
                try:
                    cycle += 1
                    self._cycle_count = cycle
                    now = datetime.now()

                    # Write heartbeat for watchdog monitoring
                    self._write_heartbeat()

                    # Get EST hour using proper timezone handling
                    est_hour = self._get_est_hour()

                    # Cache positions at start of each cycle (avoids race conditions)
                    if self.ibkr:
                        self._cached_ibkr_positions = await self.ibkr.get_all_positions()
                        self.positions = self._cached_ibkr_positions.copy()

                    # Refresh pending orders
                    if self.ibkr:
                        await self.refresh_pending_orders()

                    # Refresh data hourly
                    if (now - last_data_refresh).total_seconds() > 3600:
                        await self.refresh_data()
                        last_data_refresh = now

                    if ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END:
                        logger.info(f"⏰ Cycle #{cycle}: {now.strftime('%Y-%m-%d %H:%M:%S')} (EST: {est_hour:.1f}h)")
                        logger.info("─" * 80)
                        
                        # ═══════════════════════════════════════════════════════
                        # SOTA: Update Market State & Health
                        # ═══════════════════════════════════════════════════════
                        
                        # Check VIX Regime
                        vix_state = self.vix_manager.get_current_state()
                        self._vix_risk_multiplier = vix_state.risk_multiplier
                        self._current_vix = vix_state.current_vix  # For signal filtering

                        # Log regime change if significant
                        if hasattr(self, '_last_regime') and self._last_regime != vix_state.regime:
                            logger.warning(f"🚨 REGIME CHANGE: {self._last_regime} -> {vix_state.regime.value} (VIX: {vix_state.current_vix:.1f})")
                        self._last_regime = vix_state.regime
                        
                        if vix_state.regime != VIXRegime.NORMAL and cycle % 10 == 0:
                            logger.info(f"🌪️ Market Regime: {vix_state.regime.value.upper()} (Risk Multiplier: {self._vix_risk_multiplier:.2f})")
                        
                        # Feed VIX data to risk managers
                        if self.use_institutional and hasattr(self.inst_risk_manager, 'set_market_volatility'):
                            self.inst_risk_manager.set_market_volatility(vix_state.current_vix / 100.0)
                        
                        # Update Health Dashboard
                        health_checks = self.health_dashboard.run_all_checks(
                            current_capital=self.capital,
                            peak_capital=ApexConfig.INITIAL_CAPITAL,  # Approximate
                            positions=self.positions
                        )
                        health_status = self.health_dashboard.get_overall_status()
                        if health_status != HealthStatus.HEALTHY and cycle % 5 == 0:
                            logger.warning(f"🏥 System Health: {health_status.value.upper()}")
                        
                        # ═══════════════════════════════════════════════════════

                        # ✅ Retry any failed exits first (critical for risk management)
                        await self.retry_failed_exits()

                        # Check circuit breaker status
                        can_trade, cb_reason = self.risk_manager.can_trade()
                        if not can_trade:
                            logger.warning(f"🛑 Trading halted: {cb_reason}")
                        else:
                            # SOTA: Update universe momentum ranking
                            if cycle % 20 == 0 or cycle == 1:
                                self.cs_momentum.calculate_universe_momentum(self.historical_data)
                                tops = self.cs_momentum.get_top_momentum_stocks(self.historical_data, n=5)
                                if tops and cycle % 60 == 0:
                                    logger.info(f"🚀 Top Momentum: {', '.join([f'{s}({v:.2f})' for s,v in tops])}")

                            # Process symbols in parallel
                            await self.process_symbols_parallel(ApexConfig.SYMBOLS)

                            # Check for rebalancing (near market close)
                            await self.check_and_execute_rebalance(est_hour)

                            # Manage options (hedging, covered calls, expiring positions)
                            if ApexConfig.OPTIONS_ENABLED:
                                await self.manage_options()

                        # Sync positions after processing (captures any trades)
                        if self.ibkr:
                            await self.sync_positions_with_ibkr()

                        await self.check_risk()
                        logger.info("")
                    else:
                        if cycle % 10 == 0:
                            logger.info(f"🌙 Outside trading hours (EST: {est_hour:.1f}h)")
                            logger.info(f"   Market hours: {ApexConfig.TRADING_HOURS_START:.1f} - {ApexConfig.TRADING_HOURS_END:.1f} EST")

                    # Clear cycle cache
                    self._cached_ibkr_positions = None

                    await asyncio.sleep(ApexConfig.CHECK_INTERVAL_SECONDS)
                
                except KeyboardInterrupt:
                    logger.info("\n⏹️  Stopping...")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"❌ Loop error: {e}", exc_info=True)
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Shutting down gracefully...")
        finally:
            if self.ibkr:
                self.ibkr.disconnect()
            self.performance_tracker.print_summary()

            # Print institutional performance report
            if self.use_institutional:
                try:
                    report = self.inst_metrics.generate_report()
                    print_performance_report(report)
                except Exception as e:
                    logger.warning(f"⚠️  Could not generate institutional report: {e}")

            logger.info("=" * 80)
            logger.info("✅ APEX System stopped")
            logger.info(f"💸 Total Commissions Paid: ${self.total_commissions:,.2f}")
            logger.info("=" * 80)


async def main():
    """Main entry point."""
    system = ApexTradingSystem()
    await system.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
