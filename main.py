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
    InstitutionalSignalGenerator,
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
        
        # ✅ NEW: Protection mechanisms
        self.pending_orders: set = set()
        self.last_trade_time: Dict[str, datetime] = {}  # 60-second cooldown
        self.sector_exposure: Dict[str, float] = {}  # Track sector concentration
        self.total_commissions: float = 0.0  # Track transaction costs
        
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
                                self.price_cache[symbol] = price
                                logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) @ ${price:.2f}")
                            else:
                                logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type})")
                        except:
                            logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type})")
            
            # Load pending orders
            await self.refresh_pending_orders()
        
        # Pre-load historical data
        logger.info("📥 Loading historical data for ML training...")
        loaded = 0
        for i, symbol in enumerate(ApexConfig.SYMBOLS, 1):
            if i % 10 == 0:
                logger.info(f"   Loaded {i}/{len(ApexConfig.SYMBOLS)} symbols...")
            try:
                data = self.market_data.fetch_historical_data(symbol, days=252)
                if not data.empty:
                    self.historical_data[symbol] = data
                    loaded += 1
            except Exception as e:
                logger.debug(f"   Failed to load {symbol}: {e}")
        
        logger.info(f"✅ Loaded data for {loaded} symbols")
        
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
                    min_samples=500
                )
                if training_results:
                    logger.info("✅ Institutional ML models trained!")
                    for name, metrics in training_results.items():
                        logger.info(f"   {name}: val_mse={metrics.val_mse:.6f}, dir_acc={metrics.directional_accuracy:.1%}")
                else:
                    logger.warning("⚠️  Institutional training returned no results")
                    self.use_institutional = False
            except Exception as e:
                logger.warning(f"⚠️  Institutional training failed: {e}")
                logger.warning("   Falling back to standard signal generator")
                self.use_institutional = False

            # Initialize institutional risk manager with capital
            self.inst_risk_manager.initialize(self.capital)
        
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

        # Check 0: Circuit breaker check
        can_trade, reason = self.risk_manager.can_trade()
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

            if self.use_institutional:
                # Institutional signal generator with full metadata
                inst_signal: SignalOutput = self.inst_signal_generator.generate_signal(symbol, prices)
                signal = inst_signal.signal
                confidence = inst_signal.confidence

                # Log component breakdown for quant transparency
                if abs(signal) >= 0.30:
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Components: mom={inst_signal.momentum_signal:.2f} rev={inst_signal.mean_reversion_signal:.2f} "
                                f"trend={inst_signal.trend_signal:.2f} vol={inst_signal.volatility_signal:.2f}")
                    logger.debug(f"   ML: pred={inst_signal.ml_prediction:.4f} std={inst_signal.ml_std:.4f}")
            else:
                # Fallback to standard signal generator
                signal_data = self.signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data['signal']
                confidence = signal_data['confidence']

                # LOG SIGNAL STRENGTH (Quant transparency)
                if abs(signal) >= 0.30:
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
            
            # ═══════════════════════════════════════════════════════════════
            # EXIT LOGIC - Works for BOTH long and short
            # ═══════════════════════════════════════════════════════════════
            
            if current_pos != 0:  # ✅ Handles both long (pos) and short (neg)
                entry_price = self.position_entry_prices.get(symbol, price)
                entry_time = self.position_entry_times.get(symbol, datetime.now())
                
                # Calculate P&L (works for both long/short)
                if current_pos > 0:  # LONG
                    pnl = (price - entry_price) * current_pos
                    pnl_pct = (price / entry_price - 1) * 100
                else:  # SHORT
                    pnl = (entry_price - price) * abs(current_pos)
                    pnl_pct = (entry_price / price - 1) * 100
                
                holding_days = (datetime.now() - entry_time).days
                holding_hours = (datetime.now() - entry_time).total_seconds() / 3600
                
                should_exit = False
                exit_reason = ""
                
                # Exit conditions (work for both long/short)
                if pnl_pct < -5:
                    should_exit = True
                    exit_reason = f"Stop loss ({pnl_pct:+.1f}%)"
                
                elif pnl_pct > 15:
                    should_exit = True
                    exit_reason = f"Take profit ({pnl_pct:+.1f}%)"
                
                elif current_pos > 0 and signal < -0.40 and confidence > 0.50:  # LONG + strong bearish
                    should_exit = True
                    exit_reason = f"Strong bearish signal ({signal:.3f}, conf={confidence:.2f})"

                elif current_pos < 0 and signal > 0.40 and confidence > 0.50:  # SHORT + strong bullish
                    should_exit = True
                    exit_reason = f"Strong bullish signal ({signal:.3f}, conf={confidence:.2f})"

                # Time-based decay: lower threshold for longer holds
                elif holding_days > 10 and current_pos > 0 and signal < -0.25:
                    should_exit = True
                    exit_reason = f"Bearish after {holding_days}d ({signal:.3f})"

                elif holding_days > 10 and current_pos < 0 and signal > 0.25:
                    should_exit = True
                    exit_reason = f"Bullish after {holding_days}d ({signal:.3f})"
                
                elif holding_days > 30:
                    should_exit = True
                    exit_reason = f"Max holding ({holding_days}d)"
                
                elif pnl_pct > 5:  # ✅ Trailing stop
                    peak_price = self.position_peak_prices.get(symbol, price)
                    
                    if current_pos > 0:  # LONG
                        if price > peak_price:
                            self.position_peak_prices[symbol] = price
                        else:
                            drawdown = (price / peak_price - 1) * 100
                            if drawdown < -3:
                                should_exit = True
                                exit_reason = f"Trailing stop ({drawdown:.1f}% from peak)"
                    
                    else:  # SHORT
                        if price < peak_price:
                            self.position_peak_prices[symbol] = price
                        else:
                            drawdown = (peak_price / price - 1) * 100
                            if drawdown < -3:
                                should_exit = True
                                exit_reason = f"Trailing stop ({drawdown:.1f}% from peak)"
                
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
                            
                            logger.info(f"   ✅ Position closed (commission: ${commission:.2f})")

                            self.live_monitor.log_trade(symbol, order_side, abs(current_pos), price, pnl - commission)
                            self.performance_tracker.record_trade(symbol, order_side, abs(current_pos), price, commission)

                            # Record trade result for circuit breaker
                            self.risk_manager.record_trade_result(pnl - commission)

                            # Update cooldown
                            self.last_trade_time[symbol] = datetime.now()
                            
                            self.pending_orders.discard(symbol)
                        else:
                            self.pending_orders.discard(symbol)
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
            
            if abs(signal) < ApexConfig.MIN_SIGNAL_THRESHOLD:
                return
            
            if self.position_count >= ApexConfig.MAX_POSITIONS:
                return
            
            # ✅ Check sector limits
            if not await self.check_sector_limit(symbol):
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
                shares = min(shares, ApexConfig.MAX_SHARES_PER_POSITION)  # Cap max shares

                if sizing.constraints:
                    logger.debug(f"   Size constraints: {', '.join(sizing.constraints)}")

                if shares < 1:
                    if sizing.constraints:
                        logger.debug(f"⚠️ {symbol}: Position blocked by {sizing.constraints}")
                    else:
                        logger.debug(f"⚠️ {symbol}: Price too high (${price:.2f})")
                    return
            else:
                # Fallback: standard position sizing
                shares = int(ApexConfig.POSITION_SIZE_USD / price)
                shares = min(shares, ApexConfig.MAX_SHARES_PER_POSITION)

                if shares < 1:
                    logger.debug(f"⚠️ {symbol}: Price too high (${price:.2f})")
                    return

            # Determine side (long or short)
            side = 'BUY' if signal > 0 else 'SELL'

            logger.info(f"📈 {side} {shares} {symbol} @ ${price:.2f} (${shares*price:,.0f})")
            logger.info(f"   Signal: {signal:+.3f} | Confidence: {confidence:.3f}")
            if self.use_institutional:
                logger.debug(f"   Vol-adjusted: ${sizing.vol_adjusted_size:,.0f} | Corr penalty: {sizing.correlation_penalty:.2f}")
            
            if self.ibkr:
                self.pending_orders.add(symbol)
                
                trade = await self.ibkr.execute_order(
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    confidence=confidence
                )
                
                if trade:
                    # ✅ CRITICAL: Force sync after trade
                    await self.sync_positions_with_ibkr()
                    
                    # Track commission
                    commission = ApexConfig.COMMISSION_PER_TRADE
                    self.total_commissions += commission
                    
                    self.position_entry_prices[symbol] = price
                    self.position_entry_times[symbol] = datetime.now()
                    self.position_peak_prices[symbol] = price
                    
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
                qty = shares if side == 'BUY' else -shares
                self.positions[symbol] = qty
                self.position_entry_prices[symbol] = price
                self.position_entry_times[symbol] = datetime.now()
                self.position_peak_prices[symbol] = price

                self.live_monitor.log_trade(symbol, side, shares, price, 0)
                self.performance_tracker.record_trade(symbol, side, shares, price, 0)
                self.last_trade_time[symbol] = datetime.now()
        
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    async def process_symbols_parallel(self, symbols: List[str]):
        """Process multiple symbols in parallel for speed."""
        tasks = [self.process_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

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
                
                logger.info(f"📊 Active Positions:")
                for symbol, qty in sorted_positions[:5]:
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
                prices = self.historical_data[symbol]['Close']
                signal_data = self.signal_generator.generate_ml_signal(symbol, prices)
                
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
                    positions = self.ibkr.ib.portfolio()
                    current_value = sum([p.marketValue for p in positions])
                    if current_value == 0:
                        current_value = self.capital
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
                'daily_pnl': 0.0,
                'total_pnl': float(current_value - self.risk_manager.starting_capital),
                'total_commissions': float(self.total_commissions),
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
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
                    now = datetime.now()

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

                    # Check trading hours
                    if ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END:
                        logger.info(f"⏰ Cycle #{cycle}: {now.strftime('%Y-%m-%d %H:%M:%S')} (EST: {est_hour:.1f}h)")
                        logger.info("─" * 80)

                        # Check circuit breaker status
                        can_trade, cb_reason = self.risk_manager.can_trade()
                        if not can_trade:
                            logger.warning(f"🛑 Trading halted: {cb_reason}")
                        else:
                            # Process symbols in parallel
                            await self.process_symbols_parallel(ApexConfig.SYMBOLS)

                            # Check for rebalancing (near market close)
                            await self.check_and_execute_rebalance(est_hour)

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
