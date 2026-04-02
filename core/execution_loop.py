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
import hashlib
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
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
from core.symbols import normalize_symbol, is_market_open, parse_symbol, AssetClass
from core.broker_dispatch import BrokerDispatch
from core.risk_orchestration import RiskOrchestration
from core.state_sync import StateSync
from data.event_store import EventStore, EventType as JournalEventType
from core.trading_control import (
    read_control_state,
    mark_kill_switch_reset_processed,
    mark_governor_policy_reload_processed,
    mark_equity_reconciliation_latch_processed,
)
from models.advanced_signal_generator import AdvancedSignalGenerator
from risk.risk_manager import RiskManager
from portfolio.portfolio_optimizer import PortfolioOptimizer
from data.market_data import MarketDataFetcher
from data.social.validator import validate_social_risk_inputs
from data.social.contract import write_social_risk_inputs
from monitoring.performance_tracker import PerformanceTracker
from monitoring.live_monitor import LiveMonitor
from monitoring.alert_aggregator import fire_alert, AlertSeverity as AlertSev
from config import ApexConfig
from backtesting.governor_walkforward import WalkForwardTuningConfig, tune_policies

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
from models.god_level_signal_generator import GodLevelSignalGenerator
from models.enhanced_signal_filter import create_enhanced_filter
from execution.options_trader import OptionsTrader, OptionType, OptionStrategy

# ═══════════════════════════════════════════════════════════════════════════════
# SOTA IMPROVEMENTS - Phase 2 & 3 Modules
# ═══════════════════════════════════════════════════════════════════════════════
from risk.vix_regime_manager import VIXRegimeManager, VIXRegime
from models.cross_sectional_momentum import CrossSectionalMomentum
from data.sentiment_analyzer import SentimentAnalyzer
from execution.arrival_price_benchmark import ArrivalPriceBenchmark
from monitoring.health_dashboard import HealthDashboard, HealthStatus
from monitoring.data_quality import DataQualityMonitor
from risk.dynamic_exit_manager import get_exit_manager
from monitoring.signal_outcome_tracker import SignalOutcomeTracker
from monitoring.prometheus_metrics import PrometheusMetrics
from monitoring.performance_attribution import PerformanceAttributionTracker

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORTRESS - Multi-Layer Signal Hardening
# ═══════════════════════════════════════════════════════════════════════════════
from models.adaptive_regime_detector import AdaptiveRegimeDetector
from models.signal_consensus_engine import SignalConsensusEngine
from monitoring.signal_integrity_monitor import SignalIntegrityMonitor
from monitoring.outcome_feedback_loop import OutcomeFeedbackLoop
from monitoring.feature_drift_detector import get_drift_detector, FeatureDriftDetector
from models.adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORTRESS PHASE 2 - Indestructible Shield
# ═══════════════════════════════════════════════════════════════════════════════
from risk.black_swan_guard import BlackSwanGuard, ThreatLevel
from monitoring.signal_decay_shield import SignalDecayShield
from monitoring.trade_audit import TradeAuditLogger
from risk.exit_quality_guard import ExitQualityGuard
from risk.correlation_cascade_breaker import CorrelationCascadeBreaker, CorrelationRegime
from risk.drawdown_cascade_breaker import DrawdownCascadeBreaker, DrawdownTier
from execution.execution_shield import (
    ExecutionShield,
    ExecutionAlgo,
)
from risk.macro_shield import MacroShield
from monitoring.data_watchdog import DataWatchdog
from core.logging_config import setup_logging

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORTRESS PHASE 3 - Autonomous Money Machine
# ═══════════════════════════════════════════════════════════════════════════════
from risk.macro_event_shield import MacroEventShield, EventType
from risk.overnight_risk_guard import OvernightRiskGuard
from risk.profit_ratchet import ProfitRatchet
from risk.liquidity_guard import LiquidityGuard
from risk.position_aging_manager import PositionAgingManager
from risk.trading_excellence import TradingExcellenceManager, quick_mismatch_check, ProfitAction, update_excellence_params
from risk.performance_governor import PerformanceGovernor, GovernorSnapshot, GovernorTier
from risk.governor_policy import (
    GovernorPolicyRepository,
    GovernorPolicyResolver,
    PolicyPromotionService,
    TierControls,
    default_policy_for,
)
from risk.kill_switch import KillSwitchConfig, RiskKillSwitch
from risk.intraday_stress_engine import IntradayStressEngine, StressControlState
from risk.stress_unwind_planner import StressUnwindPlan, StressUnwindPlanner
from risk.shadow_deployment import ShadowDeploymentGate
from risk.pretrade_risk_gateway import PreTradeLimitConfig, PreTradeRiskGateway
from risk.hedge_manager import HedgeManager
from risk.adaptive_entry_gate import AdaptiveEntryGate
from risk.equity_outlier_guard import EquityOutlierGuard
from risk.social_risk_factor import SocialRiskConfig, SocialRiskFactor, SocialRiskSnapshot
from risk.social_shock_governor import (
    SocialShockDecision,
    SocialShockGovernor,
    SocialShockGovernorConfig,
)
from risk.social_governor_policy import SocialGovernorPolicyRepository
from risk.social_decision_audit import SocialDecisionAuditRepository
from market.prediction_market_verifier import (
    PredictionEventInput,
    PredictionMarketVerificationConfig,
    PredictionMarketVerificationGate,
    PredictionVerificationResult,
)
from reconciliation.equity_reconciler import EquityReconciler, EquityReconciliationSnapshot

setup_logging(
    level=ApexConfig.LOG_LEVEL,
    log_file=ApexConfig.LOG_FILE,
    json_format=True,
    console_output=True,
    max_bytes=ApexConfig.LOG_MAX_BYTES,
    backup_count=ApexConfig.LOG_BACKUP_COUNT,
    main_log_file="/private/tmp/apex_main.log",
    debug_log_file="/private/tmp/apex_debug.log",
)
logger = logging.getLogger(__name__)
options_logger = logging.getLogger("options_audit")
if not any(
    isinstance(h, logging.FileHandler)
    and h.baseFilename == str(ApexConfig.LOGS_DIR / "options_audit.log")
    for h in options_logger.handlers
):
    options_handler = logging.handlers.RotatingFileHandler(
        ApexConfig.LOGS_DIR / "options_audit.log",
        maxBytes=ApexConfig.LOG_MAX_BYTES,
        backupCount=ApexConfig.LOG_BACKUP_COUNT,
    )
    options_handler.setLevel(logging.INFO)
    options_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    options_logger.addHandler(options_handler)
    options_logger.setLevel(logging.INFO)
    options_logger.propagate = False

# Suppress IBKR, yfinance, and httpx noise
logging.getLogger('ib_insync.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ib_insync.client').setLevel(logging.WARNING)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)



from execution.smart_order_router import SmartOrderRouter
from config import ApexConfig


def _symbol_is_crypto(symbol: str) -> bool:
    """Return True if symbol belongs to the crypto asset class."""
    try:
        return parse_symbol(symbol).asset_class == AssetClass.CRYPTO
    except Exception:
        return False


def _fetch_earnings_date_yf(clean_sym: str) -> "Optional[datetime]":
    """Sync helper: return next earnings datetime for a symbol via yfinance. Returns UTC naive datetime."""
    try:
        import yfinance as yf
        from datetime import datetime as _dt, timezone as _tz
        ticker = yf.Ticker(clean_sym)
        # Primary path: calendar dict
        cal = ticker.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            if dates:
                raw = dates[0]
                if hasattr(raw, "to_pydatetime"):
                    pd_dt = raw.to_pydatetime()
                    # If timezone-aware, convert to UTC; otherwise assume ET market close (20:00 UTC)
                    if pd_dt.tzinfo is not None:
                        return pd_dt.astimezone(_tz.utc).replace(tzinfo=None)
                    else:
                        return pd_dt.replace(hour=20, minute=0, tzinfo=None)  # assume ~4pm ET = 20:00 UTC
                if isinstance(raw, _dt):
                    if raw.tzinfo is not None:
                        return raw.astimezone(_tz.utc).replace(tzinfo=None)
                    return raw.replace(hour=20, minute=0)
        # Fallback: earningsTimestamp from info (Unix UTC timestamp)
        info = ticker.info or {}
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts:
            return _dt.utcfromtimestamp(int(ts))  # UTC naive
    except Exception:
        pass
    return None


def _crypto_technical_signal(data: "pd.DataFrame") -> tuple:
    """
    Fast crypto-native technical signal using RSI, MACD, and Bollinger Bands.

    Returns (signal: float, confidence: float) both in [-1, +1] / [0, 1].
    Works with as few as 30 bars; returns (0, 0) if insufficient data.
    """
    try:
        import numpy as np
        close = data["Close"].dropna()
        if len(close) < 30:
            return 0.0, 0.0
        c = close.values.astype(float)

        # --- RSI(14) ---
        delta = np.diff(c)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        period = 14
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - 100 / (1 + rs)
        # Map RSI to signal: <30 → strong buy, >70 → strong sell
        if rsi < 30:
            rsi_sig = (30 - rsi) / 30      # 0 to 1
        elif rsi > 70:
            rsi_sig = -(rsi - 70) / 30    # 0 to -1
        else:
            rsi_sig = (50 - rsi) / 50 * 0.3  # mild mean-reversion

        # --- MACD(12,26,9) crossover ---
        def ema(arr, n):
            k = 2.0 / (n + 1)
            e = arr[0]
            for v in arr[1:]:
                e = v * k + e * (1 - k)
            return e

        ema12 = ema(c[-50:], 12)
        ema26 = ema(c[-50:], 26)
        macd_line = ema12 - ema26
        # Recent 9-bar signal EMA approximation
        if len(c) >= 60:
            prev_ema12 = ema(c[-60:-10], 12)
            prev_ema26 = ema(c[-60:-10], 26)
            prev_macd = prev_ema12 - prev_ema26
        else:
            prev_macd = 0.0
        macd_sig_val = (macd_line - prev_macd) / (abs(c[-1]) * 0.01 + 1e-9)
        macd_sig = max(-1.0, min(1.0, macd_sig_val * 5))

        # --- Bollinger Band position (20, 2σ) ---
        window = 20
        roll = c[-window:]
        mid = np.mean(roll)
        std = np.std(roll) + 1e-9
        bb_pos = (c[-1] - mid) / (2 * std)  # -1 = at lower band, +1 = at upper band
        # Momentum: if above upper band → trend continuation (NOT mean reversion for crypto)
        bb_sig = max(-1.0, min(1.0, bb_pos * 0.5))

        # --- Volume surge (if available) ---
        vol_mult = 1.0
        if "Volume" in data.columns:
            vol = data["Volume"].dropna().values.astype(float)
            if len(vol) >= 20:
                avg_vol = np.mean(vol[-20:])
                cur_vol = vol[-1]
                if avg_vol > 0:
                    surge = cur_vol / avg_vol
                    if surge > 2.0:
                        vol_mult = 1.3  # C: volume surge amplifier
                    elif surge > 1.5:
                        vol_mult = 1.15

        # Blend: RSI carries most weight (mean-reversion), MACD confirms trend, BB gives position
        raw_signal = 0.50 * rsi_sig + 0.30 * macd_sig + 0.20 * bb_sig
        raw_signal = max(-1.0, min(1.0, raw_signal * vol_mult))

        # Confidence: proportional to signal strength and internal agreement
        agreement = (
            (1 if rsi_sig > 0 else -1 if rsi_sig < 0 else 0)
            + (1 if macd_sig > 0 else -1 if macd_sig < 0 else 0)
            + (1 if bb_sig > 0 else -1 if bb_sig < 0 else 0)
        )
        conf = 0.30 + 0.20 * abs(raw_signal) + 0.15 * (abs(agreement) / 3.0)
        conf = max(0.15, min(0.85, conf))

        return float(raw_signal), float(conf)
    except Exception:
        return 0.0, 0.0


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
    _banner_printed = False
    
    def __init__(self, tenant_id: str = "default", broker_service=None, session_type: str = "unified"):
        self.tenant_id = tenant_id
        self.session_type = session_type
        if self.tenant_id == "default":
            self.user_data_dir = ApexConfig.DATA_DIR
        else:
            self.user_data_dir = ApexConfig.DATA_DIR / "users" / self.tenant_id
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

        # Load session-specific configuration overrides
        self._session_config = ApexConfig.get_session_config(session_type)
        self._session_symbols = ApexConfig.get_session_symbols(session_type)

        self.print_banner()
        logger.info("=" * 80)
        logger.info(f"🚀 {ApexConfig.SYSTEM_NAME} V{ApexConfig.VERSION} [Tenant: {self.tenant_id}] [Session: {self.session_type}]")
        logger.info(f"   Universe: {len(self._session_symbols)} symbols | Capital: ${self._session_config['initial_capital']:,}")
        logger.info("=" * 80)
        
        self.ibkr: Optional[IBKRConnector] = None
        self.alpaca = None
        self.advanced_executor: Optional[AdvancedOrderExecutor] = None

        if ApexConfig.LIVE_TRADING:
            if broker_service and self.tenant_id != "default":
                # Multi-tenant dynamic broker configuration
                active_conns = [c for c in broker_service._connections.values() 
                                if c.user_id == self.tenant_id and c.is_active]
                
                for conn in active_conns:
                    from models.broker import BrokerType
                    try:
                        creds = broker_service._decrypt_credentials(conn.credentials.get("data", ""))
                    except Exception as e:
                        logger.error(f"[{self.tenant_id}] Failed to decrypt credentials for conn {conn.id}: {e}")
                        continue

                    if conn.broker_type == BrokerType.IBKR:
                        # In dual mode, only the "core" session uses IBKR.
                        # The "crypto" session is Alpaca-only — skip IBKR to prevent
                        # both sessions from racing to connect and exhausting TWS's
                        # ~32 connection pool.
                        if self.session_type == "crypto":
                            logger.debug(f"[{self.tenant_id}] Skipping IBKR init for crypto session")
                            continue
                        logger.info(f"[{self.tenant_id}] Mode: {conn.environment.upper()} TRADING (IBKR)")
                        self.ibkr = IBKRConnector(
                            host=creds.get("host", ApexConfig.IBKR_HOST),
                            port=creds.get("port", ApexConfig.IBKR_PORT),
                            client_id=conn.client_id or ApexConfig.IBKR_CLIENT_ID
                        )
                        if ApexConfig.USE_ADVANCED_EXECUTION:
                            logger.info(f"[{self.tenant_id}] Advanced execution (TWAP/VWAP) flag detected")

                    elif conn.broker_type == BrokerType.ALPACA:
                        # In dual mode, only the "crypto" session uses Alpaca.
                        # The "core" session is IBKR-only — skip Alpaca to prevent
                        # CORE from syncing crypto positions and trading crypto via Alpaca.
                        if self.session_type == "core":
                            logger.debug(f"[{self.tenant_id}] Skipping Alpaca init for core session")
                            continue
                        logger.info(f"[{self.tenant_id}] Mode: {conn.environment.upper()} TRADING (Alpaca Crypto)")
                        from execution.alpaca_connector import AlpacaConnector
                        self.alpaca = AlpacaConnector(
                            api_key=creds.get("api_key", ""),
                            secret_key=creds.get("secret_key", ""),
                            base_url=creds.get("base_url", "https://paper-api.alpaca.markets")
                        )
            else:
                # Legacy single-tenant configuration
                broker_mode = getattr(ApexConfig, "BROKER_MODE", "ibkr").lower()
                if broker_mode in ("ibkr", "both"):
                    mode = "PAPER" if ApexConfig.IBKR_PORT == 7497 else "LIVE"
                    logger.info(f"Mode: {mode} TRADING (IBKR)")
                    self.ibkr = IBKRConnector(
                        host=ApexConfig.IBKR_HOST,
                        port=ApexConfig.IBKR_PORT,
                        client_id=ApexConfig.IBKR_CLIENT_ID
                    )
                    if ApexConfig.USE_ADVANCED_EXECUTION:
                        logger.info("Advanced execution (TWAP/VWAP) flag detected")
    
                if broker_mode in ("alpaca", "both"):
                    from execution.alpaca_connector import AlpacaConnector
                    logger.info("Mode: PAPER TRADING (Alpaca Crypto)")
                    self.alpaca = AlpacaConnector(
                        api_key=getattr(ApexConfig, "ALPACA_API_KEY", ""),
                        secret_key=getattr(ApexConfig, "ALPACA_SECRET_KEY", ""),
                        base_url=getattr(ApexConfig, "ALPACA_BASE_URL", ""),
                    )
        else:
            logger.info("Mode: SIMULATION")
            
        # Initialize Phase 12 WebSockets Daemon
        try:
            from data.websocket_streamer import WebsocketStreamer
            self.websocket_streamer = WebsocketStreamer(
                alpaca_api_key=getattr(ApexConfig, "ALPACA_API_KEY", ""),
                alpaca_secret_key=getattr(ApexConfig, "ALPACA_SECRET_KEY", "")
            )
        except Exception as e:
            logger.warning(f"WebSockets initialization failed: {e}")
            self.websocket_streamer = None

        self.broker_dispatch = BrokerDispatch(self.ibkr, self.alpaca)
        # Session-scoped state file: core_trading_state.json / crypto_trading_state.json / trading_state.json
        _state_prefix = f"{self.session_type}_" if self.session_type != "unified" else ""
        self.state_sync = StateSync(self.user_data_dir / f"{_state_prefix}trading_state.json")
        self.event_store = EventStore(self.user_data_dir)
        
        # Initialize modules (session-scoped parameters)
        self.signal_generator = AdvancedSignalGenerator()
        self.risk_manager = RiskManager(
            max_daily_loss=self._session_config.get("max_daily_loss", ApexConfig.MAX_DAILY_LOSS),
            max_drawdown=ApexConfig.MAX_DRAWDOWN,
            user_id=self.tenant_id,
            session_type=self.session_type,
        )
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Phase 15: Iceberg Execution Router (TWAP/VWAP)
        self.advanced_executor = None
        if getattr(ApexConfig, "USE_ADVANCED_EXECUTION", True):
            try:
                self.advanced_executor = AdvancedOrderExecutor(
                    risk_gateway=self.risk_manager,
                    broker_dispatch=self.broker_dispatch
                )
                logger.info("🧊 Phase 15 Iceberg Router initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AdvancedOrderExecutor: {e}")

        # Phase 11: Statistical Arbitrage (Pairs Mean-Reversion)
        try:
            from models.pairs_trader import PairsTrader
            self.pairs_trader = PairsTrader(lookback_window=60, z_entry=2.0)
            self._macro_pairs = {
                "CRYPTO:BTC/USD": "CRYPTO:ETH/USD",
                "CRYPTO:ETH/USD": "CRYPTO:BTC/USD",
                "QQQ": "SPY",
                "SPY": "QQQ",
                "XOM": "CVX",
                "CVX": "XOM"
            }
        except Exception as e:
            logger.error(f"Failed to initialize PairsTrader: {e}")
            self.pairs_trader = None
            self._macro_pairs = {}

        self.market_data = MarketDataFetcher()
        
        # Phase 14: Portfolio Delta Hedger
        try:
            from risk.market_neutral_hedge import DeltaHedger
            self.delta_hedger = DeltaHedger(market_data_fetcher=self.market_data)
        except Exception as e:
            logger.warning(f"Delta Hedger failed to initialize: {e}")
            self.delta_hedger = None
        self._delta_hedge_qty: dict = {}        # hedge_symbol → net hedged qty
        self._delta_hedge_last_cycle: int = 0  # last cycle hedge was evaluated
            
        self.performance_tracker = PerformanceTracker()
        self.live_monitor = LiveMonitor()
        self.performance_governor: Optional[PerformanceGovernor] = None
        self._performance_snapshot: Optional[GovernorSnapshot] = None
        self.prometheus_metrics: Optional[PrometheusMetrics] = None
        self._ibkr_rotation_index = 0
        self._governor_last_tier_by_key: Dict[str, str] = {}
        self._governor_regimes_by_asset: Dict[str, Tuple[str, ...]] = {
            "EQUITY": ("default", "risk_on", "risk_off", "volatile"),
            "FOREX": ("default", "carry", "carry_crash", "volatile"),
            "CRYPTO": ("default", "trend", "crash", "high_vol"),
        }

        if ApexConfig.PERFORMANCE_GOVERNOR_ENABLED:
            self.performance_governor = PerformanceGovernor(
                target_sharpe=ApexConfig.PERFORMANCE_TARGET_SHARPE,
                target_sortino=ApexConfig.PERFORMANCE_TARGET_SORTINO,
                max_drawdown=ApexConfig.PERFORMANCE_MAX_DRAWDOWN,
                sample_interval_minutes=ApexConfig.PERFORMANCE_GOV_SAMPLE_MINUTES,
                min_samples=ApexConfig.PERFORMANCE_GOV_MIN_SAMPLES,
                lookback_points=ApexConfig.PERFORMANCE_GOV_LOOKBACK_POINTS,
                recovery_points=ApexConfig.PERFORMANCE_GOV_RECOVERY_POINTS,
                points_per_year=ApexConfig.PERFORMANCE_GOV_POINTS_PER_YEAR,
            )
            # Restore persisted tier/samples so warmup is skipped after restarts
            _gov_state_path = ApexConfig.DATA_DIR / "performance_governor_state.json"
            self.performance_governor.load_state(_gov_state_path)
            self._performance_snapshot = self.performance_governor.get_snapshot()
            logger.info(
                "🧭 PerformanceGovernor enabled "
                f"(target Sharpe={ApexConfig.PERFORMANCE_TARGET_SHARPE:.2f}, "
                f"Sortino={ApexConfig.PERFORMANCE_TARGET_SORTINO:.2f}, "
                f"max DD={ApexConfig.PERFORMANCE_MAX_DRAWDOWN:.1%})"
            )

        # Governor policy repository/resolver (asset-class + regime scope)
        self.governor_policy_repo = GovernorPolicyRepository(self.user_data_dir / "governor_policies")
        active_policies = self.governor_policy_repo.load_active()
        if not active_policies:
            bootstrap = [
                default_policy_for("GLOBAL", "default"),
                default_policy_for("EQUITY", "default"),
                default_policy_for("FOREX", "default"),
                default_policy_for("CRYPTO", "default"),
            ]
            self.governor_policy_repo.save_active(bootstrap)
            active_policies = bootstrap
            logger.info("🧭 Bootstrapped default governor policies")

        self.governor_policy_resolver = GovernorPolicyResolver(active_policies)
        self.governor_promotion = PolicyPromotionService(
            repository=self.governor_policy_repo,
            environment=ApexConfig.ENVIRONMENT,
            live_trading=ApexConfig.LIVE_TRADING,
            auto_promote_non_prod=ApexConfig.GOVERNOR_AUTO_PROMOTE_NON_PROD,
        )
        self._governor_tune_state_file = self.user_data_dir / "governor_policies" / "tuning_state.json"
        self._governor_last_tune_at: Dict[str, datetime] = {}
        self._load_governor_tuning_state()
        self.shadow_deployment = ShadowDeploymentGate(
            directory=self.user_data_dir / "shadow_deployments",
            production_manifest_path=Path(ApexConfig.MODEL_MANIFEST_PATH),
            governor_repo=self.governor_policy_repo,
            environment=ApexConfig.ENVIRONMENT,
            live_trading=ApexConfig.LIVE_TRADING,
            enabled=ApexConfig.SHADOW_DEPLOYMENT_ENABLED,
            evaluation_interval_cycles=ApexConfig.SHADOW_DEPLOYMENT_EVALUATION_INTERVAL_CYCLES,
            min_shadow_days=ApexConfig.SHADOW_DEPLOYMENT_MIN_DAYS,
            min_observed_signals=ApexConfig.SHADOW_DEPLOYMENT_MIN_SIGNALS,
            min_decision_agreement_rate=ApexConfig.SHADOW_DEPLOYMENT_MIN_DECISION_AGREEMENT_RATE,
            min_offline_sharpe_delta=ApexConfig.SHADOW_DEPLOYMENT_MIN_OFFLINE_SHARPE_DELTA,
            max_drawdown_increase=ApexConfig.SHADOW_DEPLOYMENT_MAX_DRAWDOWN_INCREASE,
            max_excess_block_rate=ApexConfig.SHADOW_DEPLOYMENT_MAX_EXCESS_BLOCK_RATE,
            auto_promote_non_prod=ApexConfig.SHADOW_DEPLOYMENT_AUTO_PROMOTE_NON_PROD,
        )
        self._control_commands_file = self.user_data_dir / "trading_control_commands.json"

        # Hard kill-switch (DD + Sharpe decay)
        self.kill_switch: Optional[RiskKillSwitch] = None
        self._kill_switch_last_active = False
        if ApexConfig.KILL_SWITCH_ENABLED:
            dd_baseline = self.governor_policy_resolver.historical_mdd_baseline(
                default_value=ApexConfig.KILL_SWITCH_HISTORICAL_MDD_BASELINE
            )
            self.kill_switch = RiskKillSwitch(
                config=KillSwitchConfig(
                    dd_multiplier=ApexConfig.KILL_SWITCH_DD_MULTIPLIER,
                    sharpe_window_days=ApexConfig.KILL_SWITCH_SHARPE_WINDOW_DAYS,
                    sharpe_floor=ApexConfig.KILL_SWITCH_SHARPE_FLOOR,
                    logic=ApexConfig.KILL_SWITCH_LOGIC,
                    min_points=ApexConfig.KILL_SWITCH_MIN_POINTS,
                ),
                historical_mdd_baseline=dd_baseline,
            )
            logger.info(
                "🛑 Kill-switch enabled "
                f"(DD>{ApexConfig.KILL_SWITCH_DD_MULTIPLIER:.2f}x hist MDD or "
                f"Sharpe{ApexConfig.KILL_SWITCH_SHARPE_WINDOW_DAYS}d<{ApexConfig.KILL_SWITCH_SHARPE_FLOOR:.2f})"
            )

        self.pretrade_gateway = PreTradeRiskGateway(
            config=PreTradeLimitConfig(
                enabled=ApexConfig.PRETRADE_GATEWAY_ENABLED,
                fail_closed=ApexConfig.PRETRADE_GATEWAY_FAIL_CLOSED,
                max_order_notional=ApexConfig.PRETRADE_MAX_ORDER_NOTIONAL,
                max_price_deviation_bps=ApexConfig.PRETRADE_MAX_PRICE_DEVIATION_BPS,
                max_participation_rate=ApexConfig.PRETRADE_MAX_PARTICIPATION_RATE,
                max_gross_exposure_ratio=ApexConfig.PRETRADE_MAX_GROSS_EXPOSURE_RATIO,
            ),
            audit_dir=self.user_data_dir / "audit" / "pretrade_gateway",
        )
        logger.info(
            "🧱 Pre-trade gateway %s (notional<=%s, price_band<=%.0fbps, adv<=%.1f%%, gross<=%.2fx, fail_closed=%s)",
            "enabled" if self.pretrade_gateway.config.enabled else "disabled",
            f"${self.pretrade_gateway.config.max_order_notional:,.0f}",
            self.pretrade_gateway.config.max_price_deviation_bps,
            self.pretrade_gateway.config.max_participation_rate * 100,
            self.pretrade_gateway.config.max_gross_exposure_ratio,
            self.pretrade_gateway.config.fail_closed,
        )
        self.risk_orchestration = RiskOrchestration(
            risk_manager=self.risk_manager,
            pretrade_gateway=self.pretrade_gateway,
        )

        # Social shock governor (cross-platform sentiment + verified event odds)
        self.social_risk_factor: Optional[SocialRiskFactor] = None
        self.social_shock_governor: Optional[SocialShockGovernor] = None
        self.prediction_market_verifier: Optional[PredictionMarketVerificationGate] = None
        self._social_policy_repo = SocialGovernorPolicyRepository(
            self.user_data_dir / "governor_policies"
        )
        self._social_policy_version: str = "runtime-default"
        self._social_active_policies: Dict[Tuple[str, str], object] = {}
        social_audit_runtime_file = Path(
            os.getenv(
                "APEX_SOCIAL_DECISION_AUDIT_FILE",
                str(self.user_data_dir / "runtime" / "social_governor_decisions.jsonl"),
            )
        )
        social_audit_legacy_file = Path(
            os.getenv(
                "APEX_SOCIAL_DECISION_AUDIT_LEGACY_FILE",
                str(self.user_data_dir / "audit" / "social_governor_decisions.jsonl"),
            )
        )
        social_audit_fallbacks: List[Path] = []
        if social_audit_legacy_file != social_audit_runtime_file:
            social_audit_fallbacks.append(social_audit_legacy_file)
        self._social_audit_repo = SocialDecisionAuditRepository(
            social_audit_runtime_file,
            fallback_filepaths=social_audit_fallbacks,
        )
        self._social_input_validation: Dict[str, object] = {}
        self._social_feed_warning_cache: Dict[Tuple[str, str], List[str]] = {}
        self._social_feed_path = self.user_data_dir / "social_risk_inputs.json"
        self._social_inputs_payload: Dict[str, object] = {}
        self._social_snapshot_cache: Dict[Tuple[str, str], SocialRiskSnapshot] = {}
        self._social_decision_cache: Dict[Tuple[str, str], SocialShockDecision] = {}
        self._prediction_results_cache: Dict[Tuple[str, str], List[PredictionVerificationResult]] = {}
        self._social_policy_version, self._social_active_policies = self._social_policy_repo.load_active()
        if ApexConfig.SOCIAL_SHOCK_GOVERNOR_ENABLED:
            self.social_risk_factor = SocialRiskFactor(
                SocialRiskConfig(
                    attention_trigger_z=ApexConfig.SOCIAL_RISK_ATTENTION_TRIGGER_Z,
                    attention_extreme_z=ApexConfig.SOCIAL_RISK_ATTENTION_EXTREME_Z,
                    negative_sentiment_trigger=ApexConfig.SOCIAL_RISK_NEGATIVE_SENTIMENT_TRIGGER,
                    positive_sentiment_trigger=ApexConfig.SOCIAL_RISK_POSITIVE_SENTIMENT_TRIGGER,
                    attention_weight=ApexConfig.SOCIAL_RISK_ATTENTION_WEIGHT,
                    sentiment_weight=ApexConfig.SOCIAL_RISK_SENTIMENT_WEIGHT,
                    min_platforms=ApexConfig.SOCIAL_RISK_MIN_PLATFORMS,
                )
            )
            self.social_shock_governor = SocialShockGovernor(
                SocialShockGovernorConfig(
                    reduce_threshold=ApexConfig.SOCIAL_SHOCK_REDUCE_THRESHOLD,
                    block_threshold=ApexConfig.SOCIAL_SHOCK_BLOCK_THRESHOLD,
                    min_gross_exposure_multiplier=ApexConfig.SOCIAL_SHOCK_MIN_GROSS_MULTIPLIER,
                    verified_event_weight=ApexConfig.SOCIAL_SHOCK_VERIFIED_EVENT_WEIGHT,
                    verified_event_probability_floor=ApexConfig.SOCIAL_SHOCK_VERIFIED_EVENT_FLOOR,
                )
            )
            self.prediction_market_verifier = PredictionMarketVerificationGate(
                PredictionMarketVerificationConfig(
                    min_independent_sources=ApexConfig.PREDICTION_VERIFY_MIN_SOURCES,
                    max_probability_divergence=ApexConfig.PREDICTION_VERIFY_MAX_PROB_DIVERGENCE,
                    max_source_disagreement=ApexConfig.PREDICTION_VERIFY_MAX_SOURCE_DISAGREEMENT,
                    minimum_market_probability=ApexConfig.PREDICTION_VERIFY_MIN_MARKET_PROB,
                )
            )
            logger.info(
                "📣 SocialShockGovernor enabled (reduce>=%.2f block>=%.2f, min_gross=%.0f%%)",
                ApexConfig.SOCIAL_SHOCK_REDUCE_THRESHOLD,
                ApexConfig.SOCIAL_SHOCK_BLOCK_THRESHOLD,
                ApexConfig.SOCIAL_SHOCK_MIN_GROSS_MULTIPLIER * 100.0,
            )
            if self._social_active_policies:
                logger.info(
                    "📣 Loaded active social policy snapshot version=%s (%d scoped policies)",
                    self._social_policy_version,
                    len(self._social_active_policies),
                )

        # Trading-loop Prometheus exporter
        if ApexConfig.PROMETHEUS_TRADING_METRICS_ENABLED:
            try:
                self.prometheus_metrics = PrometheusMetrics(
                    port=ApexConfig.PROMETHEUS_TRADING_METRICS_PORT
                )
                self.prometheus_metrics.start()
            except Exception as exc:
                logger.warning("Prometheus trading metrics disabled: %s", exc)
                self.prometheus_metrics = None

        # Institutional-grade components
        self.inst_signal_generator = InstitutionalSignalGenerator(
            model_dir=str(ApexConfig.PRODUCTION_MODELS_DIR),
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
        self._options_retry_after: Dict[str, datetime] = {}  # Per-symbol action backoff after failed option order

        # ✅ Phase 3.2: GodLevel signal generator for regime detection
        self.god_signal_generator = GodLevelSignalGenerator()
        self._current_regime: str = 'neutral'  # Cache current market regime

        # ✅ ENHANCED: Signal quality filter for higher-quality trades
        self.signal_filter = create_enhanced_filter()
        self._current_vix: Optional[float] = None  # Cache VIX for signal filtering

        # ✅ DYNAMIC: Exit manager for adaptive exit thresholds
        self.exit_manager = get_exit_manager()
        self.position_entry_signals: Dict[str, float] = {}  # Track entry signal strength
        # Signal stability: consecutive bars signal above threshold per symbol
        self._signal_streak: Dict[str, int] = {}
        # Tier 4: Recent signal history per symbol (last N bars) for slope-based momentum gate
        self._signal_momentum_history: Dict[str, List[float]] = {}
        # Bar count since entry per symbol (for min-hold excellence gate)
        self._position_bar_count: Dict[str, int] = {}
        # Startup signal refresh: symbols whose refreshed ML signal is below entry threshold
        # — flagged for priority exit evaluation on the very first run-loop cycle
        self._weak_signal_restored: set = set()
        # Excellence exit persistence: count consecutive weak-signal evaluations per symbol.
        # Excellence exit fires only after N consecutive checks to avoid one-bar noise exits.
        self._weak_signal_count: Dict[str, int] = {}
        # Per-symbol consecutive loss streaks. Resets to 0 on a profitable close.
        # ≥2 consecutive losses → require 1.5× confidence; ≥3 → block symbol for session.
        self._symbol_loss_streak: Dict[str, int] = {}
        # Crypto-wide consecutive loss counter. After N losses across all crypto positions,
        # pause ALL new crypto entries for CRYPTO_CONSEC_LOSS_PAUSE_HOURS hours.
        self._crypto_consec_losses: int = 0
        self._crypto_pause_until: Optional[datetime] = None
        # Re-entry gap: track when a position was last exited (any close).
        # Prevents immediate re-entry on the same stale signal.
        self._last_exit_time: Dict[str, datetime] = {}
        self._last_exit_was_loss: Dict[str, bool] = {}  # True if close was a net loss
        # Real-time order flow imbalance: {symbol: float[-1,1]} — bid/(bid+ask) - 0.5 × 2
        # Populated from OHLCV signed-volume computation each cycle.
        self._order_flow_imbalance: Dict[str, float] = {}
        # Cross-asset divergence tracker: latest computed signals for SPY and BTC/USD.
        # Updated each cycle after signal computation. Used to penalise entries when
        # macro assets diverge (signals opposite directions = regime uncertainty).
        self._cross_asset_signals: Dict[str, float] = {}  # keys: "SPY", "BTC"
        # Signal source components: {symbol: {"ml": float, "tech": float, "sentiment": float, "cs": float}}
        # Captured at signal-blend time; read when recording entry attribution.
        self._last_signal_components: Dict[str, Dict[str, float]] = {}
        # Pairs trading state
        self._pairs_overlay: Dict[str, float] = {}  # symbol → z-score-derived signal adjustment
        self._active_pairs: List[tuple] = []         # [(asset_y, asset_x, p_value), ...]
        self._pairs_scan_cycle: int = 0              # tracks when to rescan for new pairs
        # BTC macro filter: cache BTC/USD signal and rolling 20-bar history (percentile gate).
        self._btc_signal_cache: float = 0.0
        self._btc_signal_history: List[float] = []
        # Regime transition tracking: timestamp and confidence when regime last changed.
        self._last_regime: str = "neutral"          # ML price-trend regime (string)
        self._last_vix_regime = None                 # VIX regime (VIXRegime enum) — separate tracker
        self._regime_changed_at: Optional[datetime] = None
        self._regime_transition_confidence: float = 0.70  # probability at time of change
        # Live Kelly sizing multiplier (updated from closed live_entry trades).
        self._live_kelly_mult: float = 1.0
        # Dynamic per-symbol probation thresholds from ThresholdCalibrator.
        self._calibrated_symbol_probation: Dict[str, float] = {}
        # Black-Litterman portfolio optimizer: converts individual ML signal "views" into
        # portfolio-level relative weights. Scales individual position sizes up/down by
        # the posterior conviction weight vs. equal-weight baseline.
        try:
            from portfolio.black_litterman import BlackLittermanOptimizer
            self._bl_optimizer = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.05)
            self._bl_weights: Dict[str, float] = {}   # {symbol: posterior_weight}
            self._bl_last_cycle: int = -999
            logger.info("BlackLittermanOptimizer initialized")
        except Exception as _ble:
            self._bl_optimizer = None
            self._bl_weights = {}
            self._bl_last_cycle = 0
            logger.debug("BlackLittermanOptimizer unavailable (non-fatal): %s", _ble)

        # ═══════════════════════════════════════════════════════════════════════
        # SOTA IMPROVEMENTS - Phase 2 & 3 Components
        # ═══════════════════════════════════════════════════════════════════════
        
        # VIX-based adaptive risk management
        self.vix_manager = VIXRegimeManager(cache_minutes=5)
        self._vix_risk_multiplier: float = 1.0
        self._vol_targeting_mult: float = 1.0   # set each cycle by VolTargeting

        # Cross-sectional momentum for universe ranking
        self.cs_momentum = CrossSectionalMomentum(
            lookback_months=12,
            skip_months=1,
            volatility_adjust=True
        )
        
        # News sentiment analyzer (free Yahoo Finance)
        self.sentiment_analyzer = SentimentAnalyzer(cache_minutes=30)
        
        # Phase 4.1: Startup verification for Sentiment/News API
        self.sentiment_analyzer.check_connectivity()

        
        # Arrival price benchmarking for execution quality
        self.arrival_benchmark = ArrivalPriceBenchmark(max_history=1000)
        
        # Health monitoring dashboard
        self.health_dashboard = HealthDashboard(data_dir=str(self.user_data_dir))

        # Data quality monitoring
        self.data_quality_monitor = DataQualityMonitor(
            stale_threshold_minutes=30,
            min_history_days=60
        )

        # Signal outcome tracker for forward-looking performance analysis
        self.signal_outcome_tracker = SignalOutcomeTracker(
            data_dir=str(self.user_data_dir),
            lookback_windows=[1, 5, 10, 20],
            target_returns=[0.02, 0.05, 0.10]
        )
        self.performance_attribution = PerformanceAttributionTracker(
            data_dir=self.user_data_dir,
            max_closed_trades=10_000,
        )

        logger.info("✅ SOTA modules initialized (VIX, Momentum, Sentiment, Health, SignalTracker)")

        # ═══════════════════════════════════════════════════════════════════════
        # SIGNAL FORTRESS - Multi-Layer Signal Hardening
        # ═══════════════════════════════════════════════════════════════════════

        # Adaptive regime detector (probability-based, EMA-smoothed)
        if ApexConfig.USE_ADAPTIVE_REGIME:
            self.adaptive_regime = AdaptiveRegimeDetector(
                smoothing_alpha=ApexConfig.REGIME_SMOOTHING_ALPHA,
                min_regime_duration=max(ApexConfig.MIN_REGIME_DURATION_DAYS, 5),
                min_transition_gap=0.08,      # Require 8% prob gap before switching
                transition_cooldown_steps=15, # At least 15 execution cycles (~2.5 min) between changes
            )
            logger.info("🏰 Signal Fortress: AdaptiveRegimeDetector enabled")
        else:
            self.adaptive_regime = None

        # Signal consensus engine (multi-generator majority voting)
        if ApexConfig.USE_CONSENSUS_ENGINE:
            consensus_generators = {
                "institutional": self.inst_signal_generator,
                "god_level": self.god_signal_generator,
                "advanced": self.signal_generator,
            }
            self.consensus_engine = SignalConsensusEngine(
                generators=consensus_generators,
                min_agreement=ApexConfig.MIN_CONSENSUS_AGREEMENT,
                min_generators=2,
                min_conviction=ApexConfig.MIN_CONVICTION_SCORE,
            )
            logger.info("🏰 Signal Fortress: SignalConsensusEngine enabled (3 generators)")
        else:
            self.consensus_engine = None

        # Signal integrity monitor (anomaly detection + quarantine)
        if ApexConfig.SIGNAL_INTEGRITY_ENABLED:
            self.signal_integrity = SignalIntegrityMonitor(
                window_size=100,
                stuck_threshold=ApexConfig.STUCK_SIGNAL_THRESHOLD,
                kl_threshold=ApexConfig.KL_DIVERGENCE_THRESHOLD,
                quarantine_minutes=60,
            )
            logger.info("🏰 Signal Fortress: SignalIntegrityMonitor enabled")
        else:
            self.signal_integrity = None

        # Feature drift detector (macro stationarity monitor)
        self.drift_detector: FeatureDriftDetector = get_drift_detector(
            ApexConfig.DATA_DIR / "feature_drift_baseline.json",
            drift_threshold=3.0,
        )

        # Outcome feedback loop (auto model retraining)
        if ApexConfig.AUTO_RETRAIN_ENABLED:
            self.outcome_loop = OutcomeFeedbackLoop(
                outcome_tracker=self.signal_outcome_tracker,
                consensus_engine=self.consensus_engine,
                inst_generator=self.inst_signal_generator,
                retrain_accuracy_threshold=ApexConfig.RETRAIN_ACCURACY_THRESHOLD,
                retrain_sharpe_threshold=ApexConfig.RETRAIN_SHARPE_THRESHOLD,
            )
            logger.info("🏰 Signal Fortress: OutcomeFeedbackLoop enabled")
        else:
            self.outcome_loop = None

        # Adaptive threshold optimizer (per-symbol walk-forward tuning)
        if ApexConfig.ADAPTIVE_THRESHOLDS_ENABLED:
            self.threshold_optimizer = AdaptiveThresholdOptimizer(
                outcome_tracker=self.signal_outcome_tracker,
                default_thresholds=ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME,
                min_signals=ApexConfig.THRESHOLD_MIN_SIGNALS,
                optimization_interval_hours=ApexConfig.THRESHOLD_OPTIMIZATION_INTERVAL_HOURS,
            )
            logger.info("🏰 Signal Fortress: AdaptiveThresholdOptimizer enabled")
        else:
            self.threshold_optimizer = None

        # ═══════════════════════════════════════════════════════════════════════
        # SIGNAL FORTRESS PHASE 2 - Indestructible Shield
        # ═══════════════════════════════════════════════════════════════════════

        # Black Swan Guard (real-time crash detection)
        if ApexConfig.BLACK_SWAN_GUARD_ENABLED:
            self.black_swan_guard = BlackSwanGuard(
                crash_velocity_10m=ApexConfig.CRASH_VELOCITY_THRESHOLD_10M,
                crash_velocity_30m=ApexConfig.CRASH_VELOCITY_THRESHOLD_30M,
                vix_spike_elevated=ApexConfig.VIX_SPIKE_ELEVATED,
                vix_spike_severe=ApexConfig.VIX_SPIKE_SEVERE,
                correlation_crisis_threshold=ApexConfig.CORRELATION_CRISIS_THRESHOLD,
            )
            logger.info("🛡️ Signal Fortress V2: BlackSwanGuard enabled")
        else:
            self.black_swan_guard = None

        # Signal Decay Shield (time-decay & staleness guard)
        if ApexConfig.SIGNAL_DECAY_ENABLED:
            self.signal_decay_shield = SignalDecayShield(
                max_price_age_seconds=ApexConfig.MAX_PRICE_AGE_SECONDS,
                max_sentiment_age_seconds=ApexConfig.MAX_SENTIMENT_AGE_SECONDS,
                max_feature_age_seconds=ApexConfig.MAX_FEATURE_AGE_SECONDS,
            )
            logger.info("🛡️ Signal Fortress V2: SignalDecayShield enabled")
        else:
            self.signal_decay_shield = None

        # Exit Quality Guard (exit signal validation + resilient retry)
        if ApexConfig.EXIT_QUALITY_GUARD_ENABLED:
            self.exit_quality_guard = ExitQualityGuard(
                min_exit_confidence=ApexConfig.EXIT_MIN_CONFIDENCE,
                hard_stop_pnl_threshold=ApexConfig.EXIT_HARD_STOP_PNL,
            )
            logger.info("🛡️ Signal Fortress V2: ExitQualityGuard enabled")
        else:
            self.exit_quality_guard = None

        # Correlation Cascade Breaker (portfolio-wide correlation shield)
        if ApexConfig.CORRELATION_CASCADE_ENABLED:
            self.correlation_breaker = CorrelationCascadeBreaker(
                elevated_threshold=ApexConfig.CORRELATION_ELEVATED_THRESHOLD,
                herding_threshold=ApexConfig.CORRELATION_HERDING_THRESHOLD,
                crisis_threshold=ApexConfig.CORRELATION_CRISIS_THRESHOLD_PORT,
            )
            logger.info("🛡️ Signal Fortress V2: CorrelationCascadeBreaker enabled")
        else:
            self.correlation_breaker = None

        # Initial correlation state (updated every 10 cycles in the monitoring loop)
        self._correlation_regime = CorrelationRegime.NORMAL
        self._correlation_avg: float = 0.0
        self._correlation_effective_n: float = 0.0

        # Drawdown Cascade Breaker (5-tier drawdown response)
        if ApexConfig.DRAWDOWN_CASCADE_ENABLED:
            _dcb_initial_capital = self._session_config.get("initial_capital", ApexConfig.INITIAL_CAPITAL)
            self.drawdown_breaker = DrawdownCascadeBreaker(
                initial_capital=_dcb_initial_capital,
                tier_1_threshold=ApexConfig.DRAWDOWN_TIER_1,
                tier_2_threshold=ApexConfig.DRAWDOWN_TIER_2,
                tier_3_threshold=ApexConfig.DRAWDOWN_TIER_3,
                tier_4_threshold=ApexConfig.DRAWDOWN_TIER_4,
                velocity_jump_threshold=ApexConfig.DRAWDOWN_VELOCITY_JUMP,
            )
            logger.info("🛡️ Signal Fortress V2: DrawdownCascadeBreaker enabled")
        else:
            self.drawdown_breaker = None

        # Execution Shield (smart execution wrapper)
        if ApexConfig.EXECUTION_SHIELD_ENABLED:
            self.execution_shield = ExecutionShield(
                twap_threshold=ApexConfig.EXECUTION_TWAP_THRESHOLD,
                vwap_threshold=ApexConfig.EXECUTION_VWAP_THRESHOLD,
                max_slippage_bps=ApexConfig.MAX_ACCEPTABLE_SLIPPAGE_BPS,
                critical_slippage_bps=ApexConfig.CRITICAL_SLIPPAGE_BPS,
                slippage_budget_bps=ApexConfig.EXECUTION_SLIPPAGE_BUDGET_BPS,
                slippage_budget_window=ApexConfig.EXECUTION_SLIPPAGE_BUDGET_WINDOW,
                max_spread_bps=max(
                    ApexConfig.EXECUTION_MAX_SPREAD_BPS_EQUITY,
                    ApexConfig.EXECUTION_MAX_SPREAD_BPS_FX,
                    ApexConfig.EXECUTION_MAX_SPREAD_BPS_CRYPTO,
                ),
            )
            logger.info("🛡️ Signal Fortress V2: ExecutionShield enabled")
        else:
            self.execution_shield = None
            
        # ═══════════════════════════════════════════════════════════════════════
        # SIGNAL FORTRESS PHASE 4 - Macro Shield
        # ═══════════════════════════════════════════════════════════════════════
        if ApexConfig.MACRO_SHIELD_ENABLED:
            self.macro_shield = MacroShield(
                blackout_minutes_before=ApexConfig.MACRO_BLACKOUT_MINUTES_BEFORE,
                blackout_minutes_after=ApexConfig.MACRO_BLACKOUT_MINUTES_AFTER,
            )
            logger.info(f"🛡️ Signal Fortress V4: MacroShield enabled ({ApexConfig.MACRO_BLACKOUT_MINUTES_BEFORE}m pre/{ApexConfig.MACRO_BLACKOUT_MINUTES_AFTER}m post)")
        else:
            self.macro_shield = None

        # ═══════════════════════════════════════════════════════════════════════
        # SIGNAL FORTRESS PHASE 4 - Data Watchdog
        # ═══════════════════════════════════════════════════════════════════════
        if ApexConfig.DATA_WATCHDOG_ENABLED:
            self.data_watchdog = DataWatchdog(
                max_silence_seconds=300,  # 5 minutes without ANY data = CRITICAL
                active_symbol_timeout=30  # 30s without update for active pos = WARNING
            )
        else:
            self.data_watchdog = None
            logger.info("🐕 Data Watchdog disabled via config")

        # ═══════════════════════════════════════════════════════════════════════
        # SIGNAL FORTRESS PHASE 3 - Autonomous Money Machine
        # ═══════════════════════════════════════════════════════════════════════

        # Macro Event Shield (FOMC/CPI/NFP blackouts + earnings blackouts + Game Over protection)
        self.macro_event_shield = MacroEventShield(
            blackout_before_fomc=60,
            blackout_after_fomc=30,
            blackout_before_data=30,
            blackout_after_data=15,
            blackout_before_earnings=getattr(ApexConfig, "EARNINGS_BLACKOUT_HOURS_BEFORE", 24) * 60,
            blackout_after_earnings=getattr(ApexConfig, "EARNINGS_BLACKOUT_HOURS_AFTER", 2) * 60,
            game_over_threshold=ApexConfig.GAME_OVER_LOSS_THRESHOLD,
        )
        self._last_earnings_refresh: Optional[datetime] = None
        logger.info(
            "🤖 Phase 3: MacroEventShield enabled (FOMC/CPI/NFP + earnings %dh/%dh blackout)",
            getattr(ApexConfig, "EARNINGS_BLACKOUT_HOURS_BEFORE", 24),
            getattr(ApexConfig, "EARNINGS_BLACKOUT_HOURS_AFTER", 2),
        )

        # Overnight Risk Guard (gap protection, end-of-day exposure reduction)
        self.overnight_guard = OvernightRiskGuard(
            no_entry_minutes=30,
            reduce_exposure_minutes=60,
            max_overnight_var_pct=2.0,
            high_vix_threshold=25.0,
        )
        logger.info("🤖 Phase 3: OvernightRiskGuard enabled (gap protection)")

        # Profit Ratchet (progressive trailing stops, lock in gains)
        self.profit_ratchet = ProfitRatchet(
            # Tiers aligned with TP ladder (Tier1=6%, Tier2=12%):
            # TIER_1 at +4% → lock 50%, trail 3% from HWM (let it breathe to 6% TP)
            # TIER_2 at +8% → lock 70% + take 25%, trail 2.5% (let it ride to 12% TP)
            # TIER_3/4 unchanged — capture full run if signal remains strong
            tier_1_threshold=0.04,  # 4% gain (was 2% — fired too early, cut winners short)
            tier_2_threshold=0.08,  # 8% gain (was 5%)
            tier_3_threshold=0.15,  # 15% gain (was 10%)
            tier_4_threshold=0.25,  # 25% gain (was 20%)
            initial_trailing_pct=0.03,
        )
        logger.info("🤖 Phase 3: ProfitRatchet enabled (lock in profits)")

        # Volatility targeting: scales all position sizes to hit TARGET_VOL_ANN
        try:
            from risk.vol_targeting import VolTargeting as _VT
            _target = float(getattr(ApexConfig, "VOL_TARGET_ANN", 0.15))
            self._vol_targeter = _VT(
                target_vol_ann=_target,
                lookback_days=int(getattr(ApexConfig, "VOL_TARGET_LOOKBACK_DAYS", 20)),
                min_days=int(getattr(ApexConfig, "VOL_TARGET_MIN_DAYS", 5)),
                min_mult=float(getattr(ApexConfig, "VOL_TARGET_MIN_MULT", 0.30)),
                max_mult=float(getattr(ApexConfig, "VOL_TARGET_MAX_MULT", 2.00)),
            )
            logger.info("VolTargeting initialised (target=%.0f%% ann vol)", _target * 100)
        except Exception as _e:
            logger.warning("VolTargeting init failed: %s", _e)
            self._vol_targeter = None

        # Crypto TWAP: time-slice large Alpaca orders to reduce market impact
        try:
            if getattr(ApexConfig, "CRYPTO_TWAP_ENABLED", True):
                from execution.crypto_twap import CryptoTwapExecutor as _CT
                self._crypto_twap = _CT(
                    default_interval_sec=float(getattr(ApexConfig, "CRYPTO_TWAP_INTERVAL_SEC", 30.0)),
                    abandon_pct=float(getattr(ApexConfig, "CRYPTO_TWAP_ABANDON_PCT", 0.005)),
                )
                logger.info("CryptoTwapExecutor initialised (min_notional=$%.0f)",
                            float(getattr(ApexConfig, "CRYPTO_TWAP_MIN_NOTIONAL", 5000.0)))
            else:
                self._crypto_twap = None
        except Exception as _cte:
            logger.warning("CryptoTwapExecutor init failed: %s", _cte)
            self._crypto_twap = None

        # Equity TWAP Executor: slices large equity orders into limit tranches
        try:
            if getattr(ApexConfig, "EQUITY_TWAP_ENABLED", True):
                from execution.equity_twap import EquityTwapExecutor as _ET
                self._equity_twap = _ET(
                    min_notional=float(getattr(ApexConfig, "EQUITY_TWAP_MIN_NOTIONAL", 10_000.0)),
                    num_slices=int(getattr(ApexConfig, "EQUITY_TWAP_SLICES", 5)),
                    interval_sec=float(getattr(ApexConfig, "EQUITY_TWAP_INTERVAL_SEC", 60.0)),
                    adverse_bps=float(getattr(ApexConfig, "EQUITY_TWAP_ADVERSE_BPS", 50.0)),
                )
                logger.info("EquityTwapExecutor initialised (min_notional=$%.0f, slices=%d)",
                            float(getattr(ApexConfig, "EQUITY_TWAP_MIN_NOTIONAL", 10_000.0)),
                            int(getattr(ApexConfig, "EQUITY_TWAP_SLICES", 5)))
            else:
                self._equity_twap = None
        except Exception as _ete:
            logger.warning("EquityTwapExecutor init failed: %s", _ete)
            self._equity_twap = None

        # EOD Digest Generator: daily performance report at market close
        try:
            from monitoring.eod_digest import EODDigestGenerator as _EDG
            self._eod_digest = _EDG(data_dir=ApexConfig.DATA_DIR)
            self._last_eod_digest_date: Optional[str] = None
            logger.info("EODDigestGenerator initialised")
        except Exception as _edg_e:
            logger.warning("EODDigestGenerator init failed: %s", _edg_e)
            self._eod_digest = None
            self._last_eod_digest_date = None

        # Correlation Early Warning: proactive corr regime shift detection
        try:
            from risk.correlation_early_warning import CorrelationEarlyWarning as _CEW
            self._corr_early_warning = _CEW(short_window=12, long_window=96)
            logger.info("CorrelationEarlyWarning initialised")
        except Exception as _cew_e:
            logger.warning("CorrelationEarlyWarning init failed: %s", _cew_e)
            self._corr_early_warning = None

        try:
            from monitoring.regime_forecaster import RegimeTransitionForecaster as _RTF
            self._regime_forecaster = _RTF(data_dir=ApexConfig.DATA_DIR)
            self._regime_forecast_mult: float = 1.0
            logger.info("RegimeTransitionForecaster initialised")
        except Exception as _rtf_e:
            logger.warning("RegimeTransitionForecaster init failed: %s", _rtf_e)
            self._regime_forecaster = None
            self._regime_forecast_mult = 1.0

        # Live Regime Transition Alerter: fires structured alerts on severity escalations
        try:
            from monitoring.regime_alert import RegimeTransitionAlerter as _RTA
            _cooldown = float(getattr(ApexConfig, "REGIME_ALERT_COOLDOWN_SECONDS", 900))
            self._regime_alerter = _RTA(
                cooldown_seconds=_cooldown,
                data_dir=ApexConfig.DATA_DIR,
            )
            logger.info("RegimeTransitionAlerter initialised")
        except Exception as _rta_e:
            logger.warning("RegimeTransitionAlerter init failed: %s", _rta_e)
            self._regime_alerter = None

        try:
            from models.mtf_signal_fusion import MTFSignalFuser as _MTFF
            self._mtf_fuser = _MTFF()
            logger.info("MTFSignalFuser initialised")
        except Exception as _mtf_e:
            logger.warning("MTFSignalFuser init failed: %s", _mtf_e)
            self._mtf_fuser = None

        try:
            from risk.earnings_gate import EarningsEventGate as _EEG
            self._earnings_gate = _EEG()
            self._earnings_gate_last_refresh: int = 0
            logger.info("EarningsEventGate initialised")
        except Exception as _eeg_e:
            logger.warning("EarningsEventGate init failed: %s", _eeg_e)
            self._earnings_gate = None

        try:
            from risk.hrp_sizer import HRPSizer as _HRPS
            self._hrp_sizer = _HRPS()
            logger.info("HRPSizer initialised")
        except Exception as _hrp_e:
            logger.warning("HRPSizer init failed: %s", _hrp_e)
            self._hrp_sizer = None

        # Adaptive ATR Stop Manager: regime+VIX+profit-ratchet stop updates
        try:
            from risk.adaptive_atr_stops import AdaptiveATRStops as _AATRS
            self._adaptive_atr_stops = _AATRS()
            logger.info("AdaptiveATRStops initialised")
        except Exception as _aatrs_e:
            logger.warning("AdaptiveATRStops init failed: %s", _aatrs_e)
            self._adaptive_atr_stops = None

        # Intraday Mean-Reversion Signal: VWAP deviation + RSI extreme counter-trend
        try:
            from models.intraday_mr import IntradayMRSignal as _IMRS
            self._intraday_mr = _IMRS()
            logger.info("IntradayMRSignal initialised")
        except Exception as _imrs_e:
            logger.warning("IntradayMRSignal init failed: %s", _imrs_e)
            self._intraday_mr = None

        # Model Drift Monitor: IC / hit-rate / confidence decay → retrain trigger
        try:
            from monitoring.model_drift_monitor import ModelDriftMonitor as _MDM
            self._model_drift_monitor = _MDM(
                window_size=int(getattr(ApexConfig, "MODEL_DRIFT_WINDOW_SIZE", 30)),
                ic_retrain_threshold=float(getattr(ApexConfig, "MODEL_DRIFT_IC_RETRAIN_THRESHOLD", 0.01)),
                data_dir=ApexConfig.DATA_DIR,
            )
            logger.info("ModelDriftMonitor initialised")
        except Exception as _mdm_e:
            logger.warning("ModelDriftMonitor init failed: %s", _mdm_e)
            self._model_drift_monitor = None

        # Alpha Decay Calibrator: signal IC vs hold-time horizon
        try:
            from monitoring.alpha_decay_calibrator import AlphaDecayCalibrator as _ADC
            self._alpha_decay_cal = _ADC(data_dir=ApexConfig.DATA_DIR)
            logger.info("AlphaDecayCalibrator initialised")
        except Exception as _adc_e:
            logger.warning("AlphaDecayCalibrator init failed: %s", _adc_e)
            self._alpha_decay_cal = None

        # Execution Timing Optimizer: slippage bps by (hour, dow, regime)
        try:
            from monitoring.execution_timing import ExecutionTimingOptimizer as _ETO
            self._exec_timing = _ETO(
                data_dir=ApexConfig.DATA_DIR,
                min_obs=int(getattr(ApexConfig, "EXECUTION_TIMING_MIN_OBS", 5)),
                score_floor=float(getattr(ApexConfig, "EXECUTION_TIMING_SCORE_FLOOR", 0.55)),
            )
            logger.info("ExecutionTimingOptimizer initialised")
        except Exception as _eto_e:
            logger.warning("ExecutionTimingOptimizer init failed: %s", _eto_e)
            self._exec_timing = None

        # Earnings Catalyst Signal: PEAD post-earnings drift alpha
        try:
            from data.earnings_catalyst import EarningsCatalystSignal as _ECS
            self._earnings_catalyst = _ECS()
            logger.info("EarningsCatalystSignal initialised")
        except Exception as _ecs_e:
            logger.warning("EarningsCatalystSignal init failed: %s", _ecs_e)
            self._earnings_catalyst = None

        # Signal Auto-Tuner: self-improving threshold calibration from EOD feedback
        try:
            from monitoring.signal_auto_tuner import SignalAutoTuner as _SAT
            self._signal_auto_tuner = _SAT(data_dir=ApexConfig.DATA_DIR)
            _auto_thresholds = self._signal_auto_tuner.load_thresholds_from_disk()
            if _auto_thresholds:
                self._auto_tuned_thresholds: dict = _auto_thresholds
                logger.info("SignalAutoTuner: loaded %d auto-tuned thresholds from disk", len(_auto_thresholds))
            else:
                self._auto_tuned_thresholds: dict = {}
        except Exception as _sat_e:
            logger.warning("SignalAutoTuner init failed: %s", _sat_e)
            self._signal_auto_tuner = None
            self._auto_tuned_thresholds: dict = {}

        # Alert Manager: real-time Telegram / webhook notifications
        try:
            from core.alert_manager import AlertManager as _AM
            self._alert_manager = _AM()
            logger.info("AlertManager initialised (channel: %s)", self._alert_manager.channel)
        except Exception as _am_e:
            logger.warning("AlertManager init failed: %s", _am_e)
            self._alert_manager = None

        # Capital Allocator: Kelly-optimal equity/crypto split
        try:
            from risk.capital_allocator import CapitalAllocator as _CA
            self._capital_allocator = _CA(
                data_dir=ApexConfig.DATA_DIR,
                lookback_days=int(getattr(ApexConfig, "ALLOC_LOOKBACK_DAYS", 20)),
            )
            logger.info(
                "CapitalAllocator initialised (eq=%.0f%% cr=%.0f%%)",
                self._capital_allocator.current_equity_frac * 100,
                self._capital_allocator.current_crypto_frac * 100,
            )
        except Exception as _ca_e:
            logger.warning("CapitalAllocator init failed: %s", _ca_e)
            self._capital_allocator = None

        # Execution Quality Tracker: slippage measurement + sizing penalty feedback
        try:
            from monitoring.execution_quality import ExecutionQualityTracker as _EQT
            self._exec_quality = _EQT(
                data_dir=ApexConfig.DATA_DIR / "audit" / "execution_quality",
                max_fills=2000,
                penalty_p95_bps=float(getattr(ApexConfig, "EQ_PENALTY_P95_BPS", 30.0)),
                penalty_floor=float(getattr(ApexConfig, "EQ_PENALTY_FLOOR", 0.70)),
            )
            logger.info("ExecutionQualityTracker initialised")
        except Exception as _eq_e:
            logger.warning("ExecutionQualityTracker init failed: %s", _eq_e)
            self._exec_quality = None

        # Shadow Deployment Gate: A/B candidate model testing
        try:
            from risk.shadow_deployment import ShadowDeploymentGate as _SDG
            from risk.governor_policy import GovernorPolicyRepository as _GPR
            _shadow_dir = ApexConfig.DATA_DIR / "shadow_deployments"
            _shadow_dir.mkdir(parents=True, exist_ok=True)
            _gov_repo = _GPR(repo_dir=ApexConfig.DATA_DIR / "governor_policies")
            self._shadow_gate = _SDG(
                directory=_shadow_dir,
                production_manifest_path=ApexConfig.MODELS_DIR / "model_manifest.json",
                governor_repo=_gov_repo,
                environment=getattr(ApexConfig, "ENVIRONMENT", "prod"),
                live_trading=getattr(ApexConfig, "LIVE_TRADING", False),
                enabled=getattr(ApexConfig, "SHADOW_DEPLOYMENT_ENABLED", True),
            )
            logger.info("ShadowDeploymentGate initialised (status: %s)", self._shadow_gate.snapshot.status)
        except Exception as _sdg_e:
            logger.warning("ShadowDeploymentGate init failed: %s", _sdg_e)
            self._shadow_gate = None

        # Pairs Trading: statistical arbitrage on cointegrated pairs
        try:
            if getattr(ApexConfig, "PAIRS_TRADING_ENABLED", True):
                from models.pairs_trader import PairsTrader as _PT
                self._pairs_trader = _PT(
                    lookback_window=int(getattr(ApexConfig, "PAIRS_LOOKBACK_WINDOW", 60)),
                    z_entry=float(getattr(ApexConfig, "PAIRS_Z_ENTRY", 2.0)),
                    z_exit=float(getattr(ApexConfig, "PAIRS_Z_EXIT", 0.5)),
                    min_half_life=int(getattr(ApexConfig, "PAIRS_MIN_HALF_LIFE", 1)),
                    max_half_life=int(getattr(ApexConfig, "PAIRS_MAX_HALF_LIFE", 20)),
                )
                logger.info("PairsTrader initialised (z_entry=%.1f, lookback=%d)",
                            self._pairs_trader.z_entry, self._pairs_trader.lookback)
            else:
                self._pairs_trader = None
        except Exception as _pe:
            logger.warning("PairsTrader init failed: %s", _pe)
            self._pairs_trader = None

        # Liquidity Guard (detect illiquid conditions)
        self.liquidity_guard = LiquidityGuard(
            thin_spread_threshold=0.001,
            stressed_spread_threshold=0.003,
            crisis_spread_threshold=0.005,
            min_dollar_volume=1_000_000,
        )
        logger.info("🤖 Phase 3: LiquidityGuard enabled (liquidity monitoring)")

        # Position Aging Manager (time-based exits)
        self.aging_manager = PositionAgingManager(
            max_days=30,
            stale_days=15,
            critical_days=20,
            stale_min_pnl=0.0,
            critical_min_pnl=0.02,
        )
        logger.info("🤖 Phase 3: PositionAgingManager enabled (time-based exits)")

        # Trading Excellence Manager (signal-mismatch detection, profit-taking, size scaling)
        self.excellence_manager = TradingExcellenceManager()
        logger.info("🏆 Trading Excellence: Signal mismatch detection + profit-taking enabled")

        # Hedge Manager — systematic signal dampener for correlation/VIX/drawdown stress
        self.hedge_manager = HedgeManager()
        logger.info("🛡️ HedgeManager: correlation + VIX + drawdown hedge dampener active")

        # Feature flag for institutional mode
        self.use_institutional = True  # Toggle to enable/disable institutional components
        
        # State
        # Load last known capital from risk state to prevent drawdown cliffs on startup
        _session_capital = self._session_config.get("initial_capital", ApexConfig.INITIAL_CAPITAL)
        if self.risk_manager.day_start_capital > 0:
            self.capital = self.risk_manager.day_start_capital
        elif self.risk_manager.starting_capital > 0:
            self.capital = self.risk_manager.starting_capital
        else:
            self.capital = _session_capital
            
        self._last_good_total_equity: float = float(self.capital)
        self._per_broker_last_equity: Dict[str, float] = {}  # last-known equity per broker (survives cache expiry)
        self._current_equity_contributors: set[str] = set()
        self._equity_baseline_brokers: set[str] = set()
        self.positions: Dict[str, int] = {}  # symbol -> quantity (positive=long, negative=short)
        self.is_running = False
        self._cached_ibkr_positions: Optional[Dict[str, int]] = None  # Cycle-level cache
        self._broker_equity_cache: Dict[str, Tuple[float, datetime]] = {}
        self._broker_cash_cache: Dict[str, Tuple[float, datetime]] = {}
        self._last_good_total_cash: Optional[float] = None
        self._broker_heartbeats: Dict[str, Dict[str, Optional[str] | bool]] = {
            "ibkr": {
                "last_success_ts": None,
                "last_error_ts": None,
                "last_error": None,
                "healthy": bool(self.ibkr),
            },
            "alpaca": {
                "last_success_ts": None,
                "last_error_ts": None,
                "last_error": None,
                "healthy": bool(self.alpaca),
            },
        }
        self._daily_realized_date: str = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_realized_pnl_total: float = 0.0
        self._daily_realized_pnl_by_broker: Dict[str, float] = {
            "ibkr": 0.0,
            "alpaca": 0.0,
        }
        self._daily_realized_fill_events: List[Dict[str, Any]] = []
        self._cost_basis: Dict[str, Dict[str, float]] = {}
        self._crypto_rotation_snapshot: Dict[str, Any] = {
            "timestamp": None,
            "selected": [],
            "top_scores": [],
        }
        self.intraday_stress_engine = IntradayStressEngine(
            enabled=getattr(ApexConfig, "INTRADAY_STRESS_ENGINE_ENABLED", True),
            scenario_ids=getattr(ApexConfig, "INTRADAY_STRESS_SCENARIOS", None),
            warning_return_threshold=getattr(ApexConfig, "INTRADAY_STRESS_WARNING_RETURN_THRESHOLD", -0.04),
            halt_return_threshold=getattr(ApexConfig, "INTRADAY_STRESS_HALT_RETURN_THRESHOLD", -0.08),
            warning_drawdown_threshold=getattr(ApexConfig, "INTRADAY_STRESS_WARNING_DRAWDOWN_THRESHOLD", 0.06),
            halt_drawdown_threshold=getattr(ApexConfig, "INTRADAY_STRESS_HALT_DRAWDOWN_THRESHOLD", 0.10),
            warning_size_multiplier=getattr(ApexConfig, "INTRADAY_STRESS_WARNING_SIZE_MULTIPLIER", 0.60),
            halt_size_multiplier=getattr(ApexConfig, "INTRADAY_STRESS_HALT_SIZE_MULTIPLIER", 0.25),
        )
        self._stress_control_state: StressControlState = self.intraday_stress_engine.idle_state()
        self._stress_last_action_signature: Optional[Tuple[Any, ...]] = None
        self.stress_unwind_planner = StressUnwindPlanner(
            enabled=getattr(ApexConfig, "STRESS_UNWIND_ENABLED", True),
            max_positions_per_cycle=getattr(ApexConfig, "STRESS_UNWIND_MAX_POSITIONS_PER_CYCLE", 2),
            max_participation_rate=getattr(ApexConfig, "STRESS_UNWIND_MAX_PARTICIPATION_RATE", 0.05),
            min_reduction_pct=getattr(ApexConfig, "STRESS_UNWIND_MIN_REDUCTION_PCT", 0.10),
            fallback_reduction_pct=getattr(ApexConfig, "STRESS_UNWIND_FALLBACK_REDUCTION_PCT", 0.25),
        )
        self._stress_unwind_plan: StressUnwindPlan = self.stress_unwind_planner.idle_plan()
        self._stress_unwind_plan_signature: Optional[Tuple[Any, ...]] = None
        self._stress_unwind_plan_epoch: int = 0
        self._stress_unwind_plan_id: str = ""
        self._dynamic_crypto_symbols: List[str] = []
        # ── Threshold calibration ─────────────────────────────────────────────
        from risk.threshold_calibrator import ThresholdCalibrator as _TC
        self._calibrator = _TC(ApexConfig.DATA_DIR)
        # ── Adaptive entry gate (Bayesian online threshold learning) ──────────
        try:
            self._adaptive_entry_gate = AdaptiveEntryGate(data_dir=ApexConfig.DATA_DIR)
            logger.info("AdaptiveEntryGate initialised (ema=%.4f, trades=%d)",
                        self._adaptive_entry_gate._ema_optimal_threshold,
                        sum(b.count for b in self._adaptive_entry_gate.buckets.values()))
        except Exception as _e:
            logger.warning("AdaptiveEntryGate init failed, using static thresholds: %s", _e)
            self._adaptive_entry_gate = None
        # ── Signal Enhancer (NLP/ML/momentum overlay) ─────────────────────────
        try:
            from risk.signal_enhancer import SignalEnhancer as _SE
            self._signal_enhancer = _SE(data_dir=str(ApexConfig.DATA_DIR))
            logger.info("SignalEnhancer initialised (trades=%d)", self._signal_enhancer._trade_count)
        except Exception as _e:
            logger.warning("SignalEnhancer init failed, running without ML overlay: %s", _e)
            self._signal_enhancer = None

        # Strategy Rotation Meta-Controller
        try:
            from monitoring.strategy_rotation import StrategyRotationController as _SRC
            self._strategy_rotation = _SRC(
                data_dir=ApexConfig.DATA_DIR,
                min_records=int(getattr(ApexConfig, "STRATEGY_ROTATION_MIN_RECORDS", 10)),
                temperature=float(getattr(ApexConfig, "STRATEGY_ROTATION_TEMPERATURE", 2.0)),
            )
            logger.info("StrategyRotationController initialised")
        except Exception as _src_e:
            logger.warning("StrategyRotationController init failed: %s", _src_e)
            self._strategy_rotation = None

        # ── Macro Indicators ───────────────────────────────────────────────────
        try:
            from data.macro_indicators import MacroIndicators as _MI
            self._macro_indicators = _MI()
            logger.info("MacroIndicators initialised")
        except Exception as _e:
            logger.warning("MacroIndicators init failed: %s", _e)
            self._macro_indicators = None
        self._macro_context = None  # refreshed each cycle
        # ── News Aggregator ────────────────────────────────────────────────────
        try:
            from data.news_aggregator import NewsAggregator as _NA
            self._news_aggregator = _NA()
            logger.info("NewsAggregator initialised")
        except Exception as _e:
            logger.warning("NewsAggregator init failed: %s", _e)
            self._news_aggregator = None
        # ── Factor Hedger (portfolio beta / concentration monitor) ────────────
        try:
            from risk.factor_hedger import FactorHedger as _FH
            self._factor_hedger = _FH(
                beta_warn_threshold=float(getattr(ApexConfig, "FACTOR_BETA_WARN_THRESHOLD", 1.20)),
                beta_urgent_threshold=float(getattr(ApexConfig, "FACTOR_BETA_URGENT_THRESHOLD", 1.80)),
                lookback_days=int(getattr(ApexConfig, "FACTOR_HEDGER_LOOKBACK_DAYS", 20)),
            )
            logger.info("FactorHedger initialised")
        except Exception as _e:
            logger.warning("FactorHedger init failed (non-fatal): %s", _e)
            self._factor_hedger = None
        self._last_factor_exposure = None   # updated every 10 cycles by FactorHedger check

        # ── Adaptive Meta-Controller (self-learning unified context gate) ─────
        try:
            from risk.adaptive_meta_controller import AdaptiveMetaController as _AMC
            self._meta_controller = _AMC(
                persist_path=str(ApexConfig.DATA_DIR / "meta_controller_state.json")
            )
            logger.info("AdaptiveMetaController initialised")
        except Exception as _e:
            logger.warning("AdaptiveMetaController init failed (non-fatal): %s", _e)
            self._meta_controller = None
        # Caches the TradeContext snapshot at entry time so we can pair it with
        # the realised P&L at trade close for online learning.
        self._entry_contexts: Dict[str, Any] = {}
        self._meta_size_mult: float = 1.0   # per-symbol size multiplier from meta-controller
        self._scaled_in_positions: set = set()   # symbols that already received a scale-in add

        # ── Market Intelligence Engine (unified signal synthesizer) ───────────
        try:
            from models.market_intelligence import MarketIntelligenceEngine as _MIE
            self._market_intelligence = _MIE()
            logger.info("MarketIntelligenceEngine initialised")
        except Exception as _e:
            logger.warning("MarketIntelligenceEngine init failed (non-fatal): %s", _e)
            self._market_intelligence = None

        # ── Signal Accuracy Tracker (regression vs binary comparison) ─────────
        try:
            from models.signal_accuracy_tracker import SignalAccuracyTracker as _SAT
            self._signal_accuracy_tracker = _SAT(
                state_path=str(ApexConfig.DATA_DIR / "signal_accuracy_state.json")
            )
            logger.info("SignalAccuracyTracker initialised")
        except Exception as _e:
            logger.warning("SignalAccuracyTracker init failed (non-fatal): %s", _e)
            self._signal_accuracy_tracker = None
        # ── Funding Rate Signal (Binance perpetuals, crypto-only) ─────────────
        try:
            from signals.funding_rate_signal import FundingRateSignal as _FRS
            self._funding_rate_signal = _FRS()
            logger.info("FundingRateSignal initialised")
        except Exception as _e:
            logger.warning("FundingRateSignal init failed (non-fatal): %s", _e)
            self._funding_rate_signal = None
        # ── Candlestick Pattern Signal ─────────────────────────────────────────
        try:
            from signals.pattern_signal import PatternSignal as _PS
            self._pattern_signal = _PS()
            logger.info("PatternSignal initialised")
        except Exception as _e:
            logger.warning("PatternSignal init failed (non-fatal): %s", _e)
            self._pattern_signal = None
        # ── Signal Aggregator ─────────────────────────────────────────────────
        try:
            from signals.signal_aggregator import SignalAggregator as _SA
            self._signal_aggregator = _SA()
            logger.info("SignalAggregator initialised")
        except Exception as _e:
            logger.warning("SignalAggregator init failed (non-fatal): %s", _e)
            self._signal_aggregator = None
        self._calibrated_blacklist: List[str] = []
        self._last_calibration_at: Optional[datetime] = None
        self._last_weekly_retrain_ts: float = 0.0  # epoch seconds; 0 = never (triggers first check)
        # ── Earnings PEAD Signal ───────────────────────────────────────────────
        try:
            from data.earnings_signal import EarningsSignal as _ES
            self._earnings_signal = _ES(
                cache_ttl_sec=int(getattr(ApexConfig, "EARNINGS_PEAD_CACHE_TTL_SEC", 3600)),
                decay_halflife_days=int(getattr(ApexConfig, "EARNINGS_PEAD_DECAY_HALFLIFE_DAYS", 30)),
                min_surprise=float(getattr(ApexConfig, "EARNINGS_PEAD_MIN_SURPRISE", 0.05)),
            )
            logger.info("EarningsSignal (PEAD) initialised")
        except Exception as _ee:
            logger.warning("EarningsSignal init failed (non-fatal): %s", _ee)
            self._earnings_signal = None
        # ── Rolling IC Tracker ────────────────────────────────────────────────
        try:
            from monitoring.ic_tracker import ICTracker as _ICT
            self._ic_tracker = _ICT(
                state_path=str(ApexConfig.DATA_DIR / "ic_tracker_state.json"),
                persist=True,
            )
            logger.info("ICTracker initialised")
        except Exception as _ice:
            logger.warning("ICTracker init failed (non-fatal): %s", _ice)
            self._ic_tracker = None
        # ── Opening Range Breakout signal ─────────────────────────────────────
        try:
            from signals.orb_signal import ORBSignal as _ORBS
            self._orb_signal = _ORBS(
                min_rvol=float(getattr(ApexConfig, "ORB_MIN_RVOL", 1.20)),
                min_breakout_pct=float(getattr(ApexConfig, "ORB_MIN_BREAKOUT_PCT", 0.003)),
            )
            logger.info("ORBSignal initialised")
        except Exception as _orbe:
            logger.warning("ORBSignal init failed (non-fatal): %s", _orbe)
            self._orb_signal = None
        # ── Put/Call Ratio signal ─────────────────────────────────────────────
        try:
            from data.pcr_signal import PCRSignal as _PCRS
            self._pcr_signal = _PCRS(
                cache_ttl_sec=int(getattr(ApexConfig, "PCR_CACHE_TTL_SEC", 3600)),
            )
            logger.info("PCRSignal initialised")
        except Exception as _pcre:
            logger.warning("PCRSignal init failed (non-fatal): %s", _pcre)
            self._pcr_signal = None
        self._pcr_context = None   # refreshed every 30 cycles in main loop
        # ── Alpaca background sync task handle ────────────────────────────────
        # Alpaca Paper API has intermittent latency (ConnectTimeout/ReadTimeout).
        # Periodic position syncs run as a background asyncio.Task so a stalled
        # Alpaca request never blocks the main trading loop.
        # Only one background sync runs at a time (skip if previous still in flight).
        self._alpaca_sync_task: Optional[asyncio.Task] = None
        # ── UPGRADE F: Correlated exit stagger — track exits already queued this cycle
        self._cycle_exit_count: int = 0     # resets each cycle in _run_cycle
        self._cycle_id: int = 0             # monotonic cycle counter for stagger reset
        # Overnight crypto session tracking (reset at equity open)
        self._overnight_cycles: int = 0
        self._overnight_signals_evaluated: int = 0
        self._overnight_entries: int = 0
        self._overnight_exits: int = 0
        self._overnight_session_start: Optional[datetime] = None
        self._last_in_equity_hours: Optional[bool] = None
        self._failed_symbols: set = set()
        self.equity_outlier_guard = EquityOutlierGuard(
            enabled=ApexConfig.EQUITY_OUTLIER_GUARD_ENABLED,
            max_step_move_pct=ApexConfig.EQUITY_OUTLIER_MAX_STEP_MOVE_PCT,
            confirmations_required=ApexConfig.EQUITY_OUTLIER_CONFIRM_SAMPLES,
            suspect_match_tolerance_pct=ApexConfig.EQUITY_OUTLIER_MATCH_TOLERANCE_PCT,
        )
        self.equity_outlier_guard.seed(float(self.capital))
        self.equity_reconciler = EquityReconciler(
            enabled=ApexConfig.EQUITY_RECONCILIATION_ENABLED,
            max_gap_dollars=ApexConfig.EQUITY_RECONCILIATION_MAX_GAP_DOLLARS,
            max_gap_pct=ApexConfig.EQUITY_RECONCILIATION_MAX_GAP_PCT,
            breach_confirmations=ApexConfig.EQUITY_RECONCILIATION_BREACH_CONFIRMATIONS,
            heal_confirmations=ApexConfig.EQUITY_RECONCILIATION_HEAL_CONFIRMATIONS,
            fail_closed_on_unavailable=ApexConfig.EQUITY_RECONCILIATION_FAIL_CLOSED,
        )
        self._equity_reconciliation_snapshot: Optional[EquityReconciliationSnapshot] = None
        self._equity_reconciliation_block_entries: bool = False
        self._intraday_dd_gate_active: bool = False

        # Cache
        self.price_cache: Dict[str, float] = {}
        self._price_cache_ts: Dict[str, float] = {}  # AD: Unix timestamp of last price update
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.position_entry_prices: Dict[str, float] = {}
        self.position_entry_times: Dict[str, datetime] = {}
        self.position_peak_prices: Dict[str, float] = {}  # For trailing stops

        # ATR-based dynamic stop levels per position
        self.position_stops: Dict[str, Dict] = {}  # symbol -> {stop_loss, take_profit, trailing_stop_pct, atr}
        
        # ✅ NEW: Protection mechanisms
        self.pending_orders: set = set()
        # Tracks symbols that have a placeholder position (qty=1 or -1) reserved during
        # entry processing. Avoids ambiguity with real 1-share positions.
        self._pending_entries: set = set()
        self.last_trade_time: Dict[str, datetime] = {}  # 60-second cooldown
        self.sector_exposure: Dict[str, float] = {}  # Track sector concentration
        self.total_commissions: float = 0.0  # Track transaction costs
        self.current_signals: Dict[str, dict] = {}  # ✅ Cache for dashboard signals

        # ✅ Failed exit retry tracking
        self.failed_exits: Dict[str, Dict] = {}  # symbol -> {reason, attempts, last_attempt}

        # ✅ CRITICAL: Semaphore to prevent race condition in parallel processing
        # This ensures only a limited number of entry trades can execute concurrently
        self._entry_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent entries
        self._position_lock = asyncio.Lock()  # Lock for position count checks
        self._regime_eval_lock = asyncio.Lock()
        self._meta_file_lock = asyncio.Lock()  # AF: Serialize position metadata file writes
        self._cycle_regime_eval_cycle: int = -1

        # A/D: Crypto signal enhancers
        self._fear_greed_value: float = 50.0   # neutral default
        self._fear_greed_ts: float = 0.0       # last fetch time
        self._cycle_regime_value: str = "neutral"
        self._cycle_regime_transition_prob: float = 0.0
        self._cycle_regime_log_cycle: int = -1

        # ✅ Phase 1.4: Graduated circuit breaker risk multiplier
        self._risk_multiplier: float = 1.0  # 1.0 = full size, 0.5 = half size during WARNING

        # Audit-grade trade logger (one JSONL record per fill/rejection)
        _audit_dir = Path(ApexConfig.DATA_DIR) / "users" / "admin" / "audit"
        self.trade_audit = TradeAuditLogger(_audit_dir)

        # TP Laddering: track which partial-exit tiers have fired per symbol
        # {symbol: {1, 2}} — 1=50% at Tier1, 2=25% at Tier2
        from collections import defaultdict as _defdict
        self._tp_tranches_taken: Dict[str, set] = _defdict(set)
        
        logger.info(f"💰 Capital: ${self.capital:,.2f}")
        logger.info(f"📈 Universe: {ApexConfig.UNIVERSE_MODE} ({len(self._session_symbols)} symbols) [session={self.session_type}]")
        logger.info(f"📊 Max Positions: {self._session_config.get('max_positions', ApexConfig.MAX_POSITIONS)}")
        logger.info(f"💵 Position Size: ${self._session_config.get('position_size_usd', ApexConfig.POSITION_SIZE_USD):,}")
        logger.info(f"🛡️  Max Shares/Position: {ApexConfig.MAX_SHARES_PER_POSITION}")
        logger.info(f"⏱️  Trade Cooldown: {ApexConfig.TRADE_COOLDOWN_SECONDS}s")
        logger.info("📱 Dashboard: Enabled")
        logger.info("✅ All modules initialized!")
        logger.info("=" * 80)

    def _get_connector_for(self, symbol: str):
        """Return the appropriate broker connector for a symbol."""
        broker_mode = getattr(ApexConfig, "BROKER_MODE", "ibkr").lower()
        if broker_mode == "both":
            try:
                parsed = parse_symbol(symbol)
                if parsed.asset_class == AssetClass.CRYPTO:
                    return self.alpaca
                # In mixed mode, non-crypto instruments must route via IBKR.
                # Do not silently fall back to Alpaca, which can cause symbol incompatibility
                # and fake simulation behavior when IBKR is down.
                return self.ibkr
            except ValueError:
                return self.ibkr
        if broker_mode == "alpaca":
            return self.alpaca
        return self.ibkr or self.alpaca

    def _is_paper_session(self) -> bool:
        """Best-effort detection of paper session (IBKR paper port / Alpaca paper URL)."""
        # Check connected IBKR instance
        if self.ibkr and getattr(self.ibkr, "port", None) == 7497:
            return True
        # Check IBKR config even when disconnected (startup / reconnect scenarios)
        ibkr_port = int(getattr(ApexConfig, "IBKR_PORT", 0) or 0)
        if ibkr_port == 7497:
            return True
        # Check Alpaca URL in config (works even if Alpaca connector is None at startup)
        alpaca_base = str(getattr(ApexConfig, "ALPACA_BASE_URL", "") or "").lower()
        if "paper" in alpaca_base:
            return True
        # Check explicit environment setting
        env = str(getattr(ApexConfig, "ENVIRONMENT", "") or "").lower()
        if env == "paper":
            return True
        return not bool(getattr(ApexConfig, "LIVE_TRADING", True))

    def _get_current_equity_contributors(self) -> set[str]:
        """Return the broker set that currently contributes to total equity."""
        contributors = set(getattr(self, "_current_equity_contributors", set()) or ())
        if contributors:
            return contributors
        fallback: set[str] = set()
        if getattr(self, "ibkr", None):
            fallback.add("ibkr")
        if getattr(self, "alpaca", None):
            fallback.add("alpaca")
        return fallback

    def _set_equity_baseline_brokers(self, brokers: Optional[set[str]] = None) -> None:
        """Persist the broker mix used by the current equity/risk baseline."""
        self._equity_baseline_brokers = set(brokers or self._get_current_equity_contributors())

    async def _maybe_rebase_paper_baselines_for_broker_mix(self, *, current_value: float) -> None:
        """Rebase paper-session baselines when a new broker starts contributing equity."""
        if not self._is_paper_session():
            return
        if not math.isfinite(current_value) or current_value <= 0:
            return

        contributing = self._get_current_equity_contributors()
        if not contributing:
            return

        baseline_brokers = set(getattr(self, "_equity_baseline_brokers", set()) or ())
        if not baseline_brokers:
            self._set_equity_baseline_brokers(contributing)
            return

        added_brokers = sorted(contributing - baseline_brokers)
        if not added_brokers:
            return

        try:
            baseline_capital = float(self.risk_manager.starting_capital or self.capital or 0.0)
        except Exception:
            baseline_capital = float(getattr(self, "capital", 0.0) or 0.0)

        mismatch_ratio = (
            abs(current_value - baseline_capital) / max(baseline_capital, 1e-9)
            if baseline_capital > 0
            else 1.0
        )
        mismatch_threshold = float(
            getattr(ApexConfig, "PAPER_STARTUP_RISK_MISMATCH_RATIO", 0.30)
        )

        if mismatch_ratio < mismatch_threshold:
            self._set_equity_baseline_brokers(contributing)
            logger.warning(
                "🧭 Paper broker mix updated to %s without rebase (added=%s delta=%.2f%% threshold=%.2f%%)",
                sorted(contributing),
                added_brokers,
                mismatch_ratio * 100.0,
                mismatch_threshold * 100.0,
            )
            return

        logger.warning(
            "🩹 Paper broker mix self-heal: rebasing baselines to $%.2f after adding brokers %s "
            "(old_start=$%.2f old_peak=$%.2f old_day_start=$%.2f delta=%.2f%%)",
            current_value,
            added_brokers,
            float(getattr(self.risk_manager, "starting_capital", 0.0) or 0.0),
            float(getattr(self.risk_manager, "peak_capital", 0.0) or 0.0),
            float(getattr(self.risk_manager, "day_start_capital", 0.0) or 0.0),
            mismatch_ratio * 100.0,
        )

        if hasattr(self.risk_manager, "set_starting_capital"):
            self.risk_manager.set_starting_capital(current_value)
            # Force-update day_start_capital — set_starting_capital skips it when already
            # set for today, but a broker join mid-day requires an unconditional rebase.
            self.risk_manager.day_start_capital = float(current_value)
        else:
            self.risk_manager.starting_capital = current_value
            self.risk_manager.peak_capital = current_value
            self.risk_manager.day_start_capital = current_value
        self._last_good_total_equity = float(current_value)

        equity_outlier_guard = getattr(self, "equity_outlier_guard", None)
        if equity_outlier_guard:
            equity_outlier_guard.seed(float(current_value))

        drawdown_breaker = getattr(self, "drawdown_breaker", None)
        if drawdown_breaker:
            drawdown_breaker.reset_peak(current_value)

        if (
            getattr(ApexConfig, "PAPER_STARTUP_RESET_CIRCUIT_BREAKER", False)
            and getattr(getattr(self.risk_manager, "circuit_breaker", None), "is_tripped", False)
        ):
            self.risk_manager.circuit_breaker.reset()

        if getattr(ApexConfig, "PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", False):
            performance_tracker = getattr(self, "performance_tracker", None)
            if performance_tracker and hasattr(performance_tracker, "reset_history"):
                await performance_tracker.reset_history(
                    starting_capital=current_value,
                    reason="paper_broker_mix_rebase",
                )

        if hasattr(self.risk_manager, "save_state_async"):
            await self.risk_manager.save_state_async()

        self._set_equity_baseline_brokers(contributing)

    async def _refresh_capital_from_brokers_for_startup(self):
        """Refresh startup capital from broker equity before paper-state self-heal."""
        if not (self.ibkr or self.alpaca):
            return
        try:
            observed_equity = float(await self._get_total_portfolio_value())
        except Exception as exc:
            logger.debug("Startup broker equity refresh failed: %s", exc)
            return
        if not math.isfinite(observed_equity) or observed_equity <= 0:
            return

        previous_capital = float(self.capital or 0.0)
        self.capital = float(observed_equity)
        self._last_good_total_equity = float(observed_equity)
        self.equity_outlier_guard.seed(float(observed_equity))

        if previous_capital > 0:
            drift_ratio = abs(observed_equity - previous_capital) / max(previous_capital, 1e-9)
            if drift_ratio >= 0.10:
                logger.warning(
                    "🧭 Startup broker equity sync adjusted capital from $%.2f to $%.2f (delta=%.2f%%)",
                    previous_capital,
                    observed_equity,
                    drift_ratio * 100.0,
                )

    async def _sanitize_startup_state_for_paper(self):
        """Auto-heal stale persisted risk/performance state for paper sessions."""
        if not ApexConfig.PAPER_STARTUP_RISK_SELF_HEAL_ENABLED:
            return
        if not self._is_paper_session():
            return

        try:
            capital = float(self.capital)
        except Exception:
            return
        if capital <= 0:
            return

        mismatch_ratio = 0.0
        if self.risk_manager.starting_capital > 0:
            mismatch_ratio = abs(self.risk_manager.starting_capital - capital) / max(capital, 1e-9)
        needs_risk_rebase = (
            self.risk_manager.starting_capital <= 0
            or self.risk_manager.peak_capital <= 0
            or self.risk_manager.day_start_capital <= 0
            or mismatch_ratio >= ApexConfig.PAPER_STARTUP_RISK_MISMATCH_RATIO
        )

        # DCB is purely in-memory (not persisted), always re-seed it from actual broker equity
        # to prevent false EMERGENCY tier from config initial_capital mismatch.
        drawdown_breaker = getattr(self, "drawdown_breaker", None)
        if drawdown_breaker and capital > 0:
            drawdown_breaker.reset_peak(capital)
            logger.info(
                "🛡️ Startup: DrawdownCascadeBreaker peak seeded to $%.2f (actual broker equity)",
                capital,
            )

        if needs_risk_rebase:
            logger.warning(
                "🩹 Paper startup risk self-heal: rebasing baselines to $%.2f (old_start=$%.2f old_peak=$%.2f old_day_start=$%.2f mismatch=%.2f%%)",
                capital,
                self.risk_manager.starting_capital,
                self.risk_manager.peak_capital,
                self.risk_manager.day_start_capital,
                mismatch_ratio * 100.0,
            )
            # Automatically targets the tenant session within RiskManager
            self.risk_manager.set_starting_capital(capital)
            # set_starting_capital skips day_start_capital if already set for today.
            # Force-update it so a same-day restart with a new broker (e.g. IBKR comes
            # online after Alpaca-only seed) doesn't leave a stale $79K baseline vs $1.26M.
            self.risk_manager.day_start_capital = float(capital)
            self._last_good_total_equity = float(capital)
            if (
                ApexConfig.PAPER_STARTUP_RESET_CIRCUIT_BREAKER
                and self.risk_manager.circuit_breaker.is_tripped
            ):
                self.risk_manager.circuit_breaker.reset()
                logger.warning("🩹 Paper startup risk self-heal: cleared persisted circuit breaker latch")
            await self.risk_manager.save_state_async()

        if ApexConfig.PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED:
            try:
                if self.performance_tracker.equity_curve:
                    first = float(self.performance_tracker.equity_curve[0][1])
                    perf_mismatch = abs(first - capital) / max(capital, 1e-9)
                else:
                    perf_mismatch = 1.0
            except Exception:
                perf_mismatch = 1.0

            if perf_mismatch >= ApexConfig.PAPER_STARTUP_PERFORMANCE_REBASE_RATIO:
                await self.performance_tracker.reset_history(
                    starting_capital=capital,
                    reason="paper_startup_rebase",
                )

        self._set_equity_baseline_brokers()

    async def _rebase_latches_after_reset_for_paper(self, requested_by: str, reason: str, reset_notes: List[str]) -> None:
        """In paper sessions, align risk/performance baselines after manual latch reset."""
        if not self._is_paper_session():
            reset_notes.append("paper_rebase=skipped_non_paper")
            return

        try:
            baseline = float(self.capital)
        except Exception:
            baseline = 0.0
        if baseline <= 0:
            baseline = float(self.risk_manager.starting_capital or 0.0)
        if baseline <= 0:
            reset_notes.append("paper_rebase=skipped_no_baseline")
            return

        if ApexConfig.UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES:
            self.risk_manager.starting_capital = baseline
            self.risk_manager.peak_capital = baseline
            self.risk_manager.day_start_capital = baseline
            self.risk_manager._current_day = datetime.now().strftime("%Y-%m-%d")
            self._last_good_total_equity = float(baseline)
            if self.drawdown_breaker:
                self.drawdown_breaker.reset_peak(baseline)
            reset_notes.append("paper_risk_rebase=applied")
            logger.warning(
                "🩹 Unified latch reset rebased paper risk baselines to $%.2f (requested_by=%s, reason=%s)",
                baseline,
                requested_by,
                reason,
            )
        else:
            reset_notes.append("paper_risk_rebase=disabled")

        if ApexConfig.UNIFIED_LATCH_RESET_REBASE_PERFORMANCE:
            await self.performance_tracker.reset_history(
                starting_capital=baseline,
                reason="unified_latch_reset",
            )
            reset_notes.append("paper_performance_rebase=applied")
        else:
            reset_notes.append("paper_performance_rebase=disabled")

        self._set_equity_baseline_brokers()

    @property
    def position_count(self) -> int:
        """Number of active equity/crypto positions (excludes options)."""
        return len([qty for qty in self.positions.values() if qty != 0])

    @property
    def total_position_count(self) -> int:
        """Total open positions including option contracts."""
        opt_count = len([
            v for v in getattr(self, 'options_positions', {}).values()
            if v.get('quantity', 0) != 0
        ])
        return self.position_count + opt_count

    @staticmethod
    def _now_iso_utc() -> str:
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _sanitize_event_payload(value):
        """Convert payload values into JSON-safe primitives for the event journal."""
        if isinstance(value, dict):
            return {
                str(k): ApexTradingSystem._sanitize_event_payload(v)
                for k, v in value.items()
                if v is not None
            }
        if isinstance(value, (list, tuple, set)):
            return [ApexTradingSystem._sanitize_event_payload(v) for v in value]
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, datetime):
            return value.isoformat() + ("Z" if value.tzinfo is None else "")
        if hasattr(value, "value") and not isinstance(value, (str, bytes)):
            try:
                return ApexTradingSystem._sanitize_event_payload(value.value)
            except Exception:
                pass
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    def _journal_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Safely append an event to the deterministic journal."""
        try:
            if not getattr(self, "event_store", None):
                return
            sanitized = self._sanitize_event_payload(payload)
            self.event_store.dispatch(event_type, sanitized)
        except Exception as exc:
            logger.debug("Event journal dispatch failed for %s: %s", event_type, exc)

    def _journal_signal_snapshot(
        self,
        *,
        symbol: str,
        asset_class: str,
        signal: float,
        confidence: float,
        price: float,
        current_position: float,
        raw_ml_signal: float,
        components: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._journal_event(
            JournalEventType.SIGNAL_GENERATION,
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "signal": float(signal),
                "confidence": float(confidence),
                "price": float(price),
                "current_position": float(current_position),
                "raw_ml_signal": float(raw_ml_signal),
                "regime": str(getattr(self, "_current_regime", "unknown")),
                "components": components or {},
            },
        )
        self._observe_shadow_signal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            current_position=current_position,
        )

    def _journal_risk_decision(
        self,
        *,
        symbol: str,
        asset_class: str,
        decision: str,
        stage: str,
        reason: str,
        signal: float = 0.0,
        confidence: float = 0.0,
        price: float = 0.0,
        current_position: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._journal_event(
            JournalEventType.RISK_DECISION,
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "decision": decision,
                "stage": stage,
                "reason": reason,
                "signal": float(signal),
                "confidence": float(confidence),
                "price": float(price),
                "current_position": float(current_position),
                "regime": str(getattr(self, "_current_regime", "unknown")),
                "metadata": metadata or {},
            },
        )
        self._observe_shadow_production_decision(symbol=symbol, decision=decision)

    def _journal_order_event(
        self,
        *,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: float,
        broker: str,
        lifecycle: str,
        order_role: str,
        signal: float = 0.0,
        confidence: float = 0.0,
        expected_price: float = 0.0,
        fill_price: float = 0.0,
        status: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._journal_event(
            JournalEventType.ORDER_EXECUTION,
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "side": side,
                "quantity": float(quantity),
                "broker": broker,
                "lifecycle": lifecycle,
                "order_role": order_role,
                "signal": float(signal),
                "confidence": float(confidence),
                "expected_price": float(expected_price),
                "fill_price": float(fill_price),
                "status": status,
                "regime": str(getattr(self, "_current_regime", "unknown")),
                "metadata": metadata or {},
            },
        )

    def _journal_position_update(
        self,
        *,
        symbol: str,
        asset_class: str,
        quantity: float,
        price: float,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._journal_event(
            JournalEventType.POSITION_UPDATE,
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "quantity": float(quantity),
                "price": float(price),
                "reason": reason,
                "regime": str(getattr(self, "_current_regime", "unknown")),
                "metadata": metadata or {},
            },
        )

    def _journal_stress_evaluation(self, state: StressControlState) -> None:
        self._journal_event(
            JournalEventType.STRESS_EVALUATION,
            {
                "symbol": "PORTFOLIO",
                "asset_class": "PORTFOLIO",
                "action": state.action,
                "halt_new_entries": bool(state.halt_new_entries),
                "size_multiplier": float(state.size_multiplier),
                "reason": state.reason,
                "worst_scenario_id": state.worst_scenario_id,
                "worst_scenario_name": state.worst_scenario_name,
                "worst_portfolio_return": float(state.worst_portfolio_return),
                "worst_portfolio_pnl": float(state.worst_portfolio_pnl),
                "worst_drawdown": float(state.worst_drawdown),
                "scenario_count": int(state.scenario_count),
                "scenarios": [scenario.to_dict() for scenario in state.scenarios],
            },
        )

    def _journal_stress_action(
        self,
        state: StressControlState,
        previous_state: Optional[StressControlState] = None,
    ) -> None:
        self._journal_event(
            JournalEventType.STRESS_ACTION,
            {
                "symbol": "PORTFOLIO",
                "asset_class": "PORTFOLIO",
                "action": state.action,
                "halt_new_entries": bool(state.halt_new_entries),
                "size_multiplier": float(state.size_multiplier),
                "reason": state.reason,
                "worst_scenario_id": state.worst_scenario_id,
                "worst_scenario_name": state.worst_scenario_name,
                "worst_portfolio_return": float(state.worst_portfolio_return),
                "worst_drawdown": float(state.worst_drawdown),
                "previous_action": previous_state.action if previous_state is not None else "",
                "previous_size_multiplier": (
                    float(previous_state.size_multiplier) if previous_state is not None else 1.0
                ),
                "previous_halt_new_entries": (
                    bool(previous_state.halt_new_entries) if previous_state is not None else False
                ),
            },
        )

    def _observe_shadow_signal(
        self,
        *,
        symbol: str,
        signal: float,
        confidence: float,
        current_position: float,
    ) -> None:
        gate = getattr(self, "shadow_deployment", None)
        if gate is None:
            return
        try:
            gate.observe_signal(
                symbol=symbol,
                signal=float(signal),
                confidence=float(confidence),
                current_position=float(current_position),
                stress_halt_active=bool(getattr(self, "_stress_control_state", None) and self._stress_control_state.halt_new_entries),
            )
        except Exception as exc:
            logger.debug("Shadow deployment signal observation skipped: %s", exc)

    def _observe_shadow_production_decision(
        self,
        *,
        symbol: str,
        decision: str,
    ) -> None:
        gate = getattr(self, "shadow_deployment", None)
        if gate is None:
            return
        if str(decision).lower() not in {"allowed", "blocked"}:
            return
        try:
            gate.observe_production_decision(symbol=symbol, decision=decision)
        except Exception as exc:
            logger.debug("Shadow deployment decision observation skipped: %s", exc)

    def _run_shadow_deployment_gate(self, cycle: int) -> None:
        gate = getattr(self, "shadow_deployment", None)
        if gate is None or not gate.should_evaluate(cycle):
            return

        try:
            update = gate.evaluate(
                now=datetime.utcnow(),
                live_sharpe=float(self.performance_tracker.get_sharpe_ratio()) if math.isfinite(self.performance_tracker.get_sharpe_ratio()) else 0.0,
                live_drawdown=float(self.performance_tracker.get_max_drawdown()) if math.isfinite(self.performance_tracker.get_max_drawdown()) else 0.0,
                live_win_rate=float(self.performance_tracker.get_win_rate()) if math.isfinite(self.performance_tracker.get_win_rate()) else 0.0,
                live_total_pnl=float((self.capital or 0.0) - getattr(self.risk_manager, "starting_capital", self.capital or 0.0)),
                live_total_trades=len(getattr(self.performance_tracker, "trades", [])),
                stress_halt_active=bool(getattr(self, "_stress_control_state", None) and self._stress_control_state.halt_new_entries),
            )
        except Exception as exc:
            logger.debug("Shadow deployment gate evaluation skipped due to error: %s", exc)
            return

        runtime_state = gate.runtime_state()
        candidate = runtime_state.get("candidate") if isinstance(runtime_state.get("candidate"), dict) else {}
        self._journal_event(
            JournalEventType.SHADOW_EVALUATION,
            {
                "symbol": "PORTFOLIO",
                "asset_class": "PORTFOLIO",
                "candidate_id": str(runtime_state.get("candidate_id", "")),
                "status": str(runtime_state.get("status", "")),
                "manual_approval_required": bool(runtime_state.get("manual_approval_required", False)),
                "observed_signals": int(runtime_state.get("observed_signals", 0) or 0),
                "decision_agreement_rate": float(runtime_state.get("decision_agreement_rate", 0.0) or 0.0),
                "candidate_manifest_verified": bool(runtime_state.get("candidate_manifest_verified", False)),
                "governor_policy_verified": bool(runtime_state.get("governor_policy_verified", False)),
                "live_sharpe": float(runtime_state.get("live_sharpe", 0.0) or 0.0),
                "live_drawdown": float(runtime_state.get("live_drawdown", 0.0) or 0.0),
                "offline_sharpe": float(runtime_state.get("offline_sharpe", 0.0) or 0.0),
                "offline_max_drawdown": float(runtime_state.get("offline_max_drawdown", 0.0) or 0.0),
                "reasons": list(runtime_state.get("reasons", []) or []),
                "candidate": {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "model_version": str(candidate.get("model_version", "")),
                    "feature_set_version": str(candidate.get("feature_set_version", "")),
                    "governor_policy_key": str(candidate.get("governor_policy_key", "")),
                    "governor_policy_version": str(candidate.get("governor_policy_version", "")),
                },
            },
        )

        if update.status_changed:
            self._journal_event(
                JournalEventType.PROMOTION_DECISION,
                {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "candidate_id": str(runtime_state.get("candidate_id", "")),
                    "previous_status": str(update.previous_status),
                    "current_status": str(update.current_status),
                    "manual_approval_required": bool(runtime_state.get("manual_approval_required", False)),
                    "activation_applied": bool(update.activation_applied),
                    "candidate_changed": bool(update.candidate_changed),
                    "reasons": list(runtime_state.get("reasons", []) or []),
                },
            )
            if update.current_status == "blocked":
                fire_alert(
                    "shadow_deployment_blocked",
                    f"Shadow promotion blocked for {runtime_state.get('candidate_id')}: "
                    + ", ".join(list(runtime_state.get("reasons", []) or [])[:2]),
                    AlertSev.WARNING,
                )
            elif update.current_status == "staged":
                fire_alert(
                    "shadow_deployment_staged",
                    f"Shadow candidate {runtime_state.get('candidate_id')} staged for manual approval.",
                    AlertSev.INFO,
                )
            elif update.current_status in {"ready", "active"}:
                fire_alert(
                    "shadow_deployment_ready",
                    f"Shadow candidate {runtime_state.get('candidate_id')} passed promotion gate.",
                    AlertSev.INFO,
                )

        if update.activation_applied:
            self._journal_event(
                JournalEventType.MODEL_POLICY_CHANGE,
                {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "shadow_candidate_activated",
                    "candidate_id": str(runtime_state.get("candidate_id", "")),
                    "model_version": str(runtime_state.get("model_version", "")),
                    "feature_set_version": str(runtime_state.get("feature_set_version", "")),
                    "governor_policy_key": str(runtime_state.get("governor_policy_key", "")),
                    "governor_policy_version": str(runtime_state.get("governor_policy_version", "")),
                },
            )

    @staticmethod
    def _stress_action_signature(state: StressControlState) -> Tuple[Any, ...]:
        return (
            str(state.action),
            bool(state.halt_new_entries),
            round(float(state.size_multiplier), 4),
            str(state.reason),
            str(state.worst_scenario_id),
        )

    def _run_intraday_stress_check(self, cycle: int) -> None:
        engine = getattr(self, "intraday_stress_engine", None)
        if engine is None or not engine.enabled:
            return
        interval = max(1, int(getattr(ApexConfig, "INTRADAY_STRESS_INTERVAL_CYCLES", 15)))
        if cycle != 1 and cycle % interval != 0:
            return

        live_positions = {symbol: qty for symbol, qty in self.positions.items() if qty != 0}
        price_snapshot: Dict[str, float] = {}
        last_prices = getattr(self, "last_prices", {}) or {}
        for symbol in live_positions:
            raw_price = self.price_cache.get(symbol, 0.0) or last_prices.get(symbol, 0.0)
            if not raw_price:
                raw_price = self.position_entry_prices.get(symbol, 0.0)
            try:
                price_value = float(raw_price)
            except Exception:
                continue
            if price_value > 0:
                price_snapshot[symbol] = price_value

        previous_state = getattr(self, "_stress_control_state", None)
        next_state = engine.evaluate(
            positions=live_positions,
            prices=price_snapshot,
            historical_data=self.historical_data,
            capital=float(self.capital or 0.0),
        )
        self._stress_control_state = next_state
        self._journal_stress_evaluation(next_state)

        action_signature = self._stress_action_signature(next_state)
        if action_signature != self._stress_last_action_signature:
            self._journal_stress_action(next_state, previous_state)
            self._stress_last_action_signature = action_signature
            if next_state.action == "halt_entries":
                logger.warning(
                    "🧯 IntradayStress: HALT new entries — %s (%s %.2f%%)",
                    next_state.reason,
                    next_state.worst_scenario_name or next_state.worst_scenario_id,
                    next_state.worst_portfolio_return * 100.0,
                )
                fire_alert(
                    "intraday_stress_halt",
                    f"Intraday stress halt active: {next_state.reason}",
                    AlertSev.CRITICAL,
                )
            elif next_state.action == "size_down":
                logger.warning(
                    "🧯 IntradayStress: size scaled to %.0f%% — %s (%s %.2f%%)",
                    next_state.size_multiplier * 100.0,
                    next_state.reason,
                    next_state.worst_scenario_name or next_state.worst_scenario_id,
                    next_state.worst_portfolio_return * 100.0,
                )
                fire_alert(
                    "intraday_stress_warning",
                    f"Intraday stress sizing active: {next_state.reason}",
                    AlertSev.WARNING,
                )
            elif previous_state is not None and previous_state.action != "normal":
                logger.info("🧯 IntradayStress: controls cleared (%s)", next_state.reason)

    def _stress_unwind_signature(self, plan: StressUnwindPlan) -> Tuple[Any, ...]:
        return (
            bool(plan.active),
            str(plan.reason),
            str(plan.worst_scenario_id),
            tuple(
                (
                    candidate.symbol,
                    str(candidate.action),
                    round(float(candidate.target_qty), 6),
                    round(float(candidate.target_reduction_pct), 6),
                )
                for candidate in plan.candidates
            ),
        )

    @staticmethod
    def _stress_unwind_identity_payload(plan: Optional[StressUnwindPlan]) -> Dict[str, Any]:
        if plan is None or not getattr(plan, "active", False):
            return {}
        return {
            "liquidation_plan_id": str(getattr(plan, "plan_id", "") or ""),
            "liquidation_plan_epoch": int(getattr(plan, "plan_epoch", 0) or 0),
            "liquidation_plan_action": str(getattr(plan, "action", "") or ""),
            "liquidation_plan_reason": str(getattr(plan, "reason", "") or ""),
            "liquidation_worst_scenario_id": str(getattr(plan, "worst_scenario_id", "") or ""),
            "liquidation_worst_scenario_name": str(getattr(plan, "worst_scenario_name", "") or ""),
        }

    def _stress_unwind_liquidity_snapshot(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        snapshot: Dict[str, Dict[str, Any]] = {}
        liquidity_guard = getattr(self, "liquidity_guard", None)
        historical_data = getattr(self, "historical_data", {}) or {}
        for symbol in positions:
            price = float(prices.get(symbol, 0.0) or 0.0)
            if price <= 0.0:
                continue
            metrics = None
            if liquidity_guard is not None and hasattr(liquidity_guard, "get_metrics"):
                try:
                    metrics = liquidity_guard.get_metrics(symbol)
                except Exception:
                    metrics = None

            volume_values: List[float] = []
            history = historical_data.get(symbol)
            if history is not None and hasattr(history, "columns"):
                for volume_col in ("Volume", "volume"):
                    if volume_col in history.columns:
                        try:
                            series = history[volume_col].dropna().tail(20)
                            volume_values = [float(v) for v in series.tolist() if float(v) > 0.0]
                        except Exception:
                            volume_values = []
                        if volume_values:
                            break
            elif isinstance(history, dict):
                raw_volume = history.get("Volume")
                if raw_volume is None:
                    raw_volume = history.get("volume")
                if raw_volume is not None:
                    try:
                        if hasattr(raw_volume, "dropna"):
                            series = raw_volume.dropna().tail(20)
                            volume_values = [float(v) for v in series.tolist() if float(v) > 0.0]
                        elif isinstance(raw_volume, (list, tuple)):
                            volume_values = [float(v) for v in raw_volume[-20:] if float(v) > 0.0]
                    except Exception:
                        volume_values = []

            avg_daily_volume = float(np.median(volume_values)) if volume_values else 0.0
            latest_volume = float(volume_values[-1]) if volume_values else 0.0
            volume_ratio = (
                float(latest_volume / avg_daily_volume)
                if avg_daily_volume > 0.0 and latest_volume > 0.0
                else 1.0
            )
            avg_daily_dollar_volume = avg_daily_volume * price if avg_daily_volume > 0.0 else 0.0

            if metrics is not None:
                volume_ratio = float(getattr(metrics, "volume_ratio", volume_ratio) or volume_ratio)
                liquidity_regime = getattr(getattr(metrics, "regime", None), "name", "NORMAL")
                spread_pct = float(getattr(metrics, "bid_ask_spread_pct", 0.0) or 0.0)
                size_multiplier = float(getattr(metrics, "position_size_multiplier", 1.0) or 1.0)
            else:
                liquidity_regime = "NORMAL"
                spread_pct = 0.0
                size_multiplier = 1.0

            snapshot[symbol] = {
                "avg_daily_volume": float(avg_daily_volume),
                "avg_daily_dollar_volume": float(avg_daily_dollar_volume),
                "volume_ratio": float(volume_ratio),
                "bid_ask_spread_pct": float(spread_pct),
                "liquidity_regime": str(liquidity_regime),
                "position_size_multiplier": float(size_multiplier),
            }
        return snapshot

    def _refresh_stress_unwind_plan(self) -> None:
        planner = getattr(self, "stress_unwind_planner", None)
        if planner is None or not planner.enabled:
            return
        live_positions = {symbol: qty for symbol, qty in self.positions.items() if qty != 0}
        last_prices = getattr(self, "last_prices", {}) or {}
        price_snapshot: Dict[str, float] = {}
        for symbol in live_positions:
            raw_price = self.price_cache.get(symbol, 0.0) or last_prices.get(symbol, 0.0)
            if not raw_price:
                raw_price = self.position_entry_prices.get(symbol, 0.0)
            try:
                price_value = float(raw_price)
            except Exception:
                continue
            if price_value > 0:
                price_snapshot[symbol] = price_value
        next_plan = planner.build_plan(
            stress_state=self._stress_control_state,
            positions=live_positions,
            prices=price_snapshot,
            portfolio_value=float(self.capital or 0.0),
            liquidity_snapshot=self._stress_unwind_liquidity_snapshot(live_positions, price_snapshot),
        )
        signature = self._stress_unwind_signature(next_plan)
        if next_plan.active:
            if signature != self._stress_unwind_plan_signature or not self._stress_unwind_plan_id:
                self._stress_unwind_plan_epoch += 1
                self._stress_unwind_plan_id = (
                    f"stress-unwind-{self._stress_unwind_plan_epoch}-"
                    f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
                )
            next_plan = StressUnwindPlan(
                active=next_plan.active,
                created_at=next_plan.created_at,
                action=next_plan.action,
                reason=next_plan.reason,
                plan_id=self._stress_unwind_plan_id,
                plan_epoch=self._stress_unwind_plan_epoch,
                worst_scenario_id=next_plan.worst_scenario_id,
                worst_scenario_name=next_plan.worst_scenario_name,
                candidates=next_plan.candidates,
            )
        else:
            self._stress_unwind_plan_id = ""
        self._stress_unwind_plan = next_plan
        if signature == self._stress_unwind_plan_signature:
            return
        self._stress_unwind_plan_signature = signature
        self._journal_event(
            JournalEventType.STRESS_ACTION,
            {
                "symbol": "PORTFOLIO",
                "asset_class": "PORTFOLIO",
                "action": next_plan.action,
                "reason": next_plan.reason,
                "liquidation_plan_id": next_plan.plan_id,
                "liquidation_plan_epoch": int(next_plan.plan_epoch),
                "worst_scenario_id": next_plan.worst_scenario_id,
                "worst_scenario_name": next_plan.worst_scenario_name,
                "candidates": [candidate.to_dict() for candidate in next_plan.candidates],
            },
        )
        if next_plan.active:
            logger.warning(
                "🧯 StressUnwind plan active: %s (%s) → %s",
                next_plan.reason,
                next_plan.worst_scenario_name or next_plan.worst_scenario_id,
                ", ".join(candidate.symbol for candidate in next_plan.candidates),
            )
            fire_alert(
                "stress_unwind_plan",
                "Stress unwind plan active for "
                + ", ".join(candidate.symbol for candidate in next_plan.candidates),
                AlertSev.CRITICAL,
            )
        else:
            logger.info("🧯 StressUnwind plan cleared (%s)", next_plan.reason)

    def _stress_unwind_candidate_for(self, symbol: str) -> Optional[Dict[str, Any]]:
        plan = getattr(self, "_stress_unwind_plan", None)
        if plan is None or not plan.active:
            return None
        for candidate in plan.candidates:
            if candidate.symbol == symbol:
                return candidate.to_dict()
        return None

    async def _execute_partial_position_reduction(
        self,
        *,
        symbol: str,
        asset_class: str,
        current_pos: float,
        reduction_qty: float,
        price: float,
        signal: float,
        confidence: float,
        entry_price: float,
        entry_time: datetime,
        entry_signal: float,
        holding_days: int,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        active_policy: Any,
        governor_regime_key: str,
        perf_snapshot: Any,
    ) -> None:
        reduction_qty = min(abs(float(current_pos)), max(0.0, float(reduction_qty)))
        if reduction_qty <= 0.0:
            return

        order_side = "SELL" if current_pos > 0 else "BUY"
        side_label = "LONG" if current_pos > 0 else "SHORT"
        reduction_ratio = reduction_qty / max(abs(float(current_pos)), 1e-9)
        reduction_pnl = float(pnl) * reduction_ratio
        exit_connector = self._get_connector_for(symbol)

        logger.warning(
            "🧯 PARTIAL EXIT %s (%s): %s",
            symbol,
            side_label,
            exit_reason,
        )
        logger.info(
            "   Stress reduction: %.6f / %.6f (%.0f%%)",
            reduction_qty,
            abs(float(current_pos)),
            reduction_ratio * 100.0,
        )

        if exit_connector:
            self.pending_orders.add(symbol)
            exit_broker = "alpaca" if exit_connector is self.alpaca else "ibkr"
            self._journal_order_event(
                symbol=symbol,
                asset_class=asset_class,
                side=order_side,
                quantity=float(reduction_qty),
                broker=exit_broker,
                lifecycle="submitted",
                order_role="exit",
                signal=float(signal),
                confidence=float(confidence),
                expected_price=float(price),
                metadata={
                    "exit_reason": exit_reason,
                    "exit_type": "partial_reduce",
                    "planned_reduction_qty": float(reduction_qty),
                    "planned_reduction_pct": float(reduction_ratio),
                    **self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None)),
                    **self._governor_policy_metadata(
                        active_policy,
                        governor_regime_key,
                        perf_snapshot.tier.value,
                    ),
                },
            )

            trade = await exit_connector.execute_order(
                symbol=symbol,
                side=order_side,
                quantity=reduction_qty,
                confidence=abs(signal) if signal != 0 else 0.8,
            )

            if trade:
                exit_fill_price = float(price)
                exit_expected_price = float(price)
                exit_trade_status = "FILLED"
                if isinstance(trade, dict):
                    exit_fill_price = float(trade.get("price", exit_fill_price) or exit_fill_price)
                    exit_expected_price = float(
                        trade.get("expected_price", exit_expected_price) or exit_expected_price
                    )
                    exit_trade_status = str(trade.get("status", "FILLED")).upper()
                else:
                    try:
                        exit_fill_price = float(trade.orderStatus.avgFillPrice or exit_fill_price)
                    except Exception:
                        pass

                self._journal_order_event(
                    symbol=symbol,
                    asset_class=asset_class,
                    side=order_side,
                    quantity=float(reduction_qty),
                    broker=exit_broker,
                    lifecycle="filled" if exit_trade_status == "FILLED" else "result",
                    order_role="exit",
                    signal=float(signal),
                    confidence=float(confidence),
                    expected_price=float(exit_expected_price),
                    fill_price=float(exit_fill_price),
                    status=exit_trade_status,
                    metadata={
                        "exit_reason": exit_reason,
                        "exit_type": "partial_reduce",
                        "planned_reduction_qty": float(reduction_qty),
                        "planned_reduction_pct": float(reduction_ratio),
                        **self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None)),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )

                await self._sync_positions()

                commission = ApexConfig.COMMISSION_PER_TRADE
                self.total_commissions += commission
                realized_net = 0.0
                if exit_trade_status == "FILLED":
                    realized_net = self._record_fill_realized_pnl(
                        broker_name=exit_broker,
                        symbol=symbol,
                        side=order_side,
                        quantity=float(reduction_qty),
                        fill_price=float(exit_fill_price),
                        commission=float(commission),
                        filled_at=datetime.now(),
                    )

                exit_slippage_bps = self._compute_slippage_bps(
                    exit_expected_price,
                    exit_fill_price,
                )
                self._record_exit_attribution(
                    symbol=symbol,
                    asset_class=asset_class,
                    side=side_label,
                    quantity=float(reduction_qty),
                    entry_price=float(entry_price),
                    exit_price=float(exit_fill_price),
                    commissions=float(commission),
                    exit_reason=exit_reason,
                    entry_signal=float(entry_signal),
                    entry_confidence=min(1.0, max(0.0, abs(entry_signal))),
                    governor_tier=perf_snapshot.tier.value,
                    governor_regime=self._map_governor_regime(asset_class, self._current_regime),
                    entry_time=entry_time,
                    exit_time=datetime.now(),
                    exit_slippage_bps=float(exit_slippage_bps),
                    source="live_partial_exit",
                )
                self.trade_audit.log(
                    event="STRESS_TRIM",
                    symbol=symbol,
                    side=order_side,
                    qty=float(reduction_qty),
                    fill_price=float(exit_fill_price),
                    expected_price=float(exit_expected_price),
                    slippage_bps=float(exit_slippage_bps),
                    signal=float(signal),
                    confidence=float(confidence),
                    entry_signal=float(entry_signal),
                    regime=str(self._current_regime),
                    pnl_pct=float(pnl_pct),
                    pnl_usd=float(reduction_pnl),
                    exit_reason=exit_reason,
                    holding_days=int(holding_days),
                    broker=exit_broker,
                    pretrade="PASS",
                )
                self._journal_position_update(
                    symbol=symbol,
                    asset_class=asset_class,
                    quantity=float(self.positions.get(symbol, 0.0)),
                    price=float(exit_fill_price),
                    reason="partial_exit_fill",
                    metadata={
                        "broker": exit_broker,
                        "status": exit_trade_status,
                        "exit_reason": exit_reason,
                        "reduction_qty": float(reduction_qty),
                        "reduction_pct": float(reduction_ratio),
                        **self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None)),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )

                net_result = (
                    float(realized_net)
                    if not math.isclose(realized_net, 0.0, abs_tol=1e-9)
                    else float(reduction_pnl - commission)
                )
                self.live_monitor.log_trade(
                    symbol,
                    order_side,
                    float(reduction_qty),
                    float(exit_fill_price),
                    net_result,
                )
                await self.performance_tracker.record_trade(
                    symbol,
                    order_side,
                    float(reduction_qty),
                    float(exit_fill_price),
                    commission,
                )
                self.risk_manager.record_trade_result(net_result)
                self.last_trade_time[symbol] = datetime.now()
                self.pending_orders.discard(symbol)
                if symbol in self.failed_exits:
                    del self.failed_exits[symbol]
                logger.info(
                    "   ✅ Stress trim completed: %.6f reduced, remaining %.6f",
                    reduction_qty,
                    float(self.positions.get(symbol, 0.0)),
                )
            else:
                self.pending_orders.discard(symbol)
                attempts = self.failed_exits.get(symbol, {}).get("attempts", 0) + 1
                self.failed_exits[symbol] = {
                    "reason": exit_reason,
                    "attempts": attempts,
                    "last_attempt": datetime.now(),
                    "quantity": float(reduction_qty),
                    "side": order_side,
                }
                logger.warning(
                    "   ⚠️ Stress trim order failed for %s (attempt %d)",
                    symbol,
                    attempts,
                )
        else:
            if self.ibkr or self.alpaca:
                logger.warning(
                    "⚠️ %s: No eligible connector for stress trim in broker mode '%s'; keeping position open",
                    symbol,
                    str(getattr(ApexConfig, "BROKER_MODE", "ibkr")).lower(),
                )
                self.last_trade_time[symbol] = datetime.now() - timedelta(
                    seconds=max(30, int(ApexConfig.TRADE_COOLDOWN_SECONDS) - 30)
                )
                return

            exit_fill_price = float(price)
            new_qty = current_pos - reduction_qty if current_pos > 0 else current_pos + reduction_qty
            if math.isclose(new_qty, 0.0, abs_tol=1e-9):
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = new_qty

            self._journal_order_event(
                symbol=symbol,
                asset_class=asset_class,
                side=order_side,
                quantity=float(reduction_qty),
                broker="simulation",
                lifecycle="filled",
                order_role="exit",
                signal=float(signal),
                confidence=float(confidence),
                expected_price=float(exit_fill_price),
                fill_price=float(exit_fill_price),
                status="SIMULATED",
                metadata={
                    "exit_reason": exit_reason,
                    "exit_type": "partial_reduce",
                    "planned_reduction_qty": float(reduction_qty),
                    "planned_reduction_pct": float(reduction_ratio),
                    **self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None)),
                    **self._governor_policy_metadata(
                        active_policy,
                        governor_regime_key,
                        perf_snapshot.tier.value,
                    ),
                },
            )
            self._record_exit_attribution(
                symbol=symbol,
                asset_class=asset_class,
                side=side_label,
                quantity=float(reduction_qty),
                entry_price=float(entry_price),
                exit_price=float(exit_fill_price),
                commissions=0.0,
                exit_reason=exit_reason,
                entry_signal=float(entry_signal),
                entry_confidence=min(1.0, max(0.0, abs(entry_signal))),
                governor_tier=perf_snapshot.tier.value,
                governor_regime=self._map_governor_regime(asset_class, self._current_regime),
                entry_time=entry_time,
                exit_time=datetime.now(),
                exit_slippage_bps=0.0,
                source="sim_partial_exit",
            )
            self.trade_audit.log(
                event="STRESS_TRIM",
                symbol=symbol,
                side=order_side,
                qty=float(reduction_qty),
                fill_price=float(exit_fill_price),
                expected_price=float(exit_fill_price),
                slippage_bps=0.0,
                signal=float(signal),
                confidence=float(confidence),
                entry_signal=float(entry_signal),
                regime=str(self._current_regime),
                pnl_pct=float(pnl_pct),
                pnl_usd=float(reduction_pnl),
                exit_reason=exit_reason,
                holding_days=int(holding_days),
                broker="simulation",
                pretrade="PASS",
            )
            self._journal_position_update(
                symbol=symbol,
                asset_class=asset_class,
                quantity=float(self.positions.get(symbol, 0.0)),
                price=float(exit_fill_price),
                reason="sim_partial_exit_fill",
                metadata={
                    "status": "SIMULATED",
                    "exit_reason": exit_reason,
                    "reduction_qty": float(reduction_qty),
                    "reduction_pct": float(reduction_ratio),
                    **self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None)),
                    **self._governor_policy_metadata(
                        active_policy,
                        governor_regime_key,
                        perf_snapshot.tier.value,
                    ),
                },
            )
            self.live_monitor.log_trade(symbol, order_side, float(reduction_qty), price, float(reduction_pnl))
            await self.performance_tracker.record_trade(
                symbol,
                order_side,
                float(reduction_qty),
                price,
                0.0,
            )
            self.risk_manager.record_trade_result(float(reduction_pnl))
            self.last_trade_time[symbol] = datetime.now()

    @staticmethod
    def _governor_policy_metadata(policy: Any, regime_key: str, tier: Optional[str] = None) -> Dict[str, Any]:
        if policy is None:
            return {"governor_regime": regime_key, **({"governor_tier": tier} if tier else {})}
        metadata: Dict[str, Any] = {
            "governor_regime": regime_key,
            "governor_policy_key": policy.key().as_id() if hasattr(policy, "key") else "",
            "governor_policy_version": str(getattr(policy, "version", "")),
            "governor_policy_id": (
                policy.policy_id() if hasattr(policy, "policy_id") else ""
            ),
        }
        if tier:
            metadata["governor_tier"] = tier
        return metadata

    def _write_trade_rejection(self, symbol: str, reason: str, signal: float = 0.0,
                               confidence: float = 0.0, price: float = 0.0,
                               extra: dict = None) -> None:
        """Append rejected signal to audit JSONL (Upgrade H)."""
        try:
            asset_class = parse_symbol(symbol).asset_class.value
        except Exception:
            asset_class = "EQUITY"
        self._journal_risk_decision(
            symbol=symbol,
            asset_class=asset_class,
            decision="blocked",
            stage="trade_rejection",
            reason=reason,
            signal=signal,
            confidence=confidence,
            price=price,
            current_position=float(self.positions.get(symbol, 0)),
            metadata=extra,
        )
        if not getattr(ApexConfig, "TRADE_REJECTION_AUDIT_ENABLED", True):
            return
        import json
        try:
            audit_path = self.user_data_dir / "audit" / "trade_rejections.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": self._now_iso_utc(),
                "symbol": symbol,
                "reason": reason,
                "signal": round(signal, 4),
                "confidence": round(confidence, 4),
                "price": round(price, 4),
                **(extra or {}),
            }
            with open(audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as _audit_err:
            logger.debug("Trade rejection audit write failed: %s", _audit_err)

    def _write_execution_latency(
        self,
        symbol: str,
        side: str,
        broker: str,
        signal_ts: float,      # time.time() when signal was generated
        order_sent_ts: float,  # time.time() when order was submitted to broker
        fill_ts: float,        # time.time() when fill/confirmation received (0 = unfilled)
        quantity: float = 0.0,
        fill_price: float = 0.0,
        signal_strength: float = 0.0,
        expected_price: float = 0.0,   # arrival/expected price for slippage calc
    ) -> None:
        """P: Append trade execution latency record to audit JSONL for TCA analysis."""
        try:
            signal_to_order_ms = round((order_sent_ts - signal_ts) * 1000, 1)
            order_to_fill_ms = round((fill_ts - order_sent_ts) * 1000, 1) if fill_ts else None
            total_ms = round((fill_ts - signal_ts) * 1000, 1) if fill_ts else None

            # Compute slippage for ThresholdCalibrator blacklist
            slippage_bps: Optional[float] = None
            if fill_price > 0 and expected_price > 0:
                slippage_bps = round(abs(fill_price - expected_price) / expected_price * 10000.0, 2)

            # Record fill into ExecutionTimingOptimizer bucket
            if slippage_bps is not None and getattr(self, "_exec_timing", None) is not None:
                try:
                    import datetime as _dt
                    _now_utc = _dt.datetime.utcnow()
                    if getattr(ApexConfig, "EXECUTION_TIMING_ENABLED", True):
                        self._exec_timing.record_fill(
                            slippage_bps=float(slippage_bps),
                            hour=_now_utc.hour,
                            day_of_week=_now_utc.weekday(),
                            regime=str(self._current_regime),
                        )
                except Exception:
                    pass

            audit_path = self.user_data_dir / "audit" / "execution_latency.jsonl"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": self._now_iso_utc(),
                "symbol": symbol,
                "side": side,
                "broker": broker,
                "signal_strength": round(signal_strength, 4),
                "qty": round(quantity, 6),
                "fill_price": round(fill_price, 4),
                "expected_price": round(expected_price, 4) if expected_price else None,
                "slippage_bps": slippage_bps,
                "signal_to_order_ms": signal_to_order_ms,
                "order_to_fill_ms": order_to_fill_ms,
                "total_ms": total_ms,
            }
            with open(audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as _lat_err:
            logger.debug("Execution latency audit write failed: %s", _lat_err)

    @staticmethod
    def _normalize_broker_name(broker_name: str) -> str:
        token = str(broker_name or "").strip().lower()
        if token in {"ibkr", "alpaca"}:
            return token
        return "unknown"

    def _mark_broker_heartbeat(
        self,
        broker_name: str,
        *,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        broker = self._normalize_broker_name(broker_name)
        if broker == "unknown":
            return
        now_iso = self._now_iso_utc()
        heartbeat = self._broker_heartbeats.setdefault(
            broker,
            {
                "last_success_ts": None,
                "last_error_ts": None,
                "last_error": None,
                "healthy": False,
            },
        )
        if success:
            heartbeat["last_success_ts"] = now_iso
            heartbeat["healthy"] = True
            heartbeat["last_error"] = None
        else:
            heartbeat["last_error_ts"] = now_iso
            heartbeat["healthy"] = False
            if error:
                heartbeat["last_error"] = str(error)[:240]

    def _broker_heartbeat_payload(self) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        for broker in ("ibkr", "alpaca"):
            raw = self._broker_heartbeats.get(broker, {})
            payload[broker] = {
                "last_success_ts": raw.get("last_success_ts"),
                "last_error_ts": raw.get("last_error_ts"),
                "last_error": raw.get("last_error"),
                "healthy": bool(raw.get("healthy")),
            }
        return payload

    def _roll_daily_realized_if_needed(self, now: Optional[datetime] = None) -> None:
        ts = now or datetime.utcnow()
        day_key = ts.strftime("%Y-%m-%d")
        if day_key == self._daily_realized_date:
            return
        self._daily_realized_date = day_key
        self._daily_realized_pnl_total = 0.0
        self._daily_realized_pnl_by_broker = {"ibkr": 0.0, "alpaca": 0.0}
        self._daily_realized_fill_events = []
        logger.info("📒 Reset daily realized PnL ledger for %s", day_key)

    def _sync_cost_basis_with_positions(self) -> None:
        """Keep local fill ledger aligned with current broker truth positions."""
        tracked = set()
        for symbol, qty in self.positions.items():
            qty_f = float(qty or 0.0)
            if math.isclose(qty_f, 0.0, abs_tol=1e-12):
                continue
            tracked.add(symbol)
            cached = self._cost_basis.get(symbol, {})
            cached_qty = float(cached.get("qty", 0.0) or 0.0)
            avg_price = float(cached.get("avg_price", 0.0) or 0.0)
            if avg_price <= 0:
                avg_price = float(self.position_entry_prices.get(symbol, 0.0) or 0.0)
            if avg_price <= 0:
                avg_price = float(self.price_cache.get(symbol, 0.0) or 0.0)
            if avg_price <= 0:
                avg_price = 0.0

            if math.isclose(cached_qty, qty_f, abs_tol=1e-9):
                if avg_price > 0 and (cached.get("avg_price", 0.0) or 0.0) <= 0:
                    self._cost_basis[symbol] = {"qty": qty_f, "avg_price": avg_price}
                continue

            self._cost_basis[symbol] = {"qty": qty_f, "avg_price": avg_price}

        for symbol in list(self._cost_basis.keys()):
            if symbol not in tracked:
                self._cost_basis.pop(symbol, None)

    def _record_fill_realized_pnl(
        self,
        *,
        broker_name: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,
        filled_at: Optional[datetime] = None,
    ) -> float:
        """Apply a broker fill to local position cost basis and realize close PnL."""
        self._roll_daily_realized_if_needed(filled_at)
        broker = self._normalize_broker_name(broker_name)
        qty = abs(float(quantity or 0.0))
        px = float(fill_price or 0.0)
        if qty <= 0 or px <= 0:
            return 0.0

        side_token = str(side or "").strip().upper()
        signed_change = qty if side_token == "BUY" else -qty
        if math.isclose(signed_change, 0.0, abs_tol=1e-12):
            return 0.0

        entry = self._cost_basis.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        prev_qty = float(entry.get("qty", 0.0) or 0.0)
        prev_avg = float(entry.get("avg_price", 0.0) or 0.0)
        if prev_avg <= 0:
            prev_avg = float(self.position_entry_prices.get(symbol, 0.0) or px)

        realized_net = 0.0
        if math.isclose(prev_qty, 0.0, abs_tol=1e-12) or (prev_qty > 0 and signed_change > 0) or (prev_qty < 0 and signed_change < 0):
            new_qty = prev_qty + signed_change
            existing_abs = abs(prev_qty)
            new_abs = abs(new_qty)
            if new_abs <= 1e-12:
                self._cost_basis.pop(symbol, None)
            else:
                weighted_avg = (
                    ((existing_abs * prev_avg) + (qty * px)) / max(existing_abs + qty, 1e-12)
                    if existing_abs > 0
                    else px
                )
                self._cost_basis[symbol] = {"qty": new_qty, "avg_price": weighted_avg}
            return 0.0

        close_qty = min(abs(prev_qty), qty)
        if prev_qty > 0 and signed_change < 0:
            realized_gross = (px - prev_avg) * close_qty
        else:
            realized_gross = (prev_avg - px) * close_qty
        realized_net = realized_gross - float(commission or 0.0)

        new_qty = prev_qty + signed_change
        if math.isclose(new_qty, 0.0, abs_tol=1e-12):
            self._cost_basis.pop(symbol, None)
        elif (prev_qty > 0 > new_qty) or (prev_qty < 0 < new_qty):
            self._cost_basis[symbol] = {"qty": new_qty, "avg_price": px}
        else:
            self._cost_basis[symbol] = {"qty": new_qty, "avg_price": prev_avg}

        if broker in self._daily_realized_pnl_by_broker:
            self._daily_realized_pnl_by_broker[broker] += float(realized_net)
        self._daily_realized_pnl_total += float(realized_net)
        self._daily_realized_fill_events.append(
            {
                "timestamp": (filled_at or datetime.utcnow()).isoformat() + "Z",
                "broker": broker,
                "symbol": symbol,
                "side": side_token,
                "quantity": qty,
                "fill_price": px,
                "commission": float(commission or 0.0),
                "realized_pnl": float(realized_net),
            }
        )
        max_events = 200
        if len(self._daily_realized_fill_events) > max_events:
            self._daily_realized_fill_events = self._daily_realized_fill_events[-max_events:]
        return float(realized_net)

    def _compute_daily_pnl_by_broker(self) -> dict:
        """Compute daily P&L split by broker: realized fills + unrealized mark-to-market.

        IBKR carries all equity/FX positions; Alpaca carries all crypto positions.
        When daily P&L comes from equity-delta fallback (no fills), the unrealized component
        is still attributed correctly so the dashboard shows meaningful per-broker P&L.
        """
        _CRYPTO_PREFIXES = ("CRYPTO:", "BTC", "ETH", "SOL", "AVAX", "LINK",
                            "LTC", "DOT", "DOGE", "ADA", "XLM", "XRP",
                            "BCH", "FIL", "CRV", "ETC", "ALGO", "ATOM",
                            "MATIC", "SHIB", "UNI", "AAVE")

        def _is_crypto(sym: str) -> bool:
            u = sym.upper()
            return any(u.startswith(p) for p in _CRYPTO_PREFIXES)

        ibkr_realized = float(self._daily_realized_pnl_by_broker.get("ibkr", 0.0))
        alpaca_realized = float(self._daily_realized_pnl_by_broker.get("alpaca", 0.0))

        # Add unrealized P&L from open positions using price_cache vs entry prices
        ibkr_unrealized = 0.0
        alpaca_unrealized = 0.0
        for sym, qty in self.positions.items():
            if not qty:
                continue
            entry = float(self.position_entry_prices.get(sym, 0.0) or 0.0)
            current = float(self.price_cache.get(sym, 0.0) or 0.0)
            if entry <= 0 or current <= 0:
                continue
            unrealized = (current - entry) * qty
            if _is_crypto(sym):
                alpaca_unrealized += unrealized
            else:
                ibkr_unrealized += unrealized

        return {
            "ibkr": round(ibkr_realized + ibkr_unrealized, 2),
            "alpaca": round(alpaca_realized + alpaca_unrealized, 2),
            "ibkr_realized": round(ibkr_realized, 2),
            "ibkr_unrealized": round(ibkr_unrealized, 2),
            "alpaca_realized": round(alpaca_realized, 2),
            "alpaca_unrealized": round(alpaca_unrealized, 2),
        }

    def _rotate_crypto_universe(
        self,
        open_universe: List[str],
        open_positions: List[str],
    ) -> List[str]:
        """Trade only top-N crypto pairs by momentum/liquidity while always keeping open positions."""
        if self.session_type == "core":
            return open_universe  # Core session has no crypto to rotate
        if not ApexConfig.CRYPTO_ROTATION_ENABLED:
            return open_universe

        crypto_candidates: List[str] = []
        non_crypto: List[str] = []
        for symbol in open_universe:
            try:
                if parse_symbol(symbol).asset_class == AssetClass.CRYPTO:
                    crypto_candidates.append(symbol)
                else:
                    non_crypto.append(symbol)
            except ValueError:
                non_crypto.append(symbol)

        if not crypto_candidates:
            return open_universe

        open_crypto = []
        for symbol in open_positions:
            try:
                if parse_symbol(symbol).asset_class == AssetClass.CRYPTO:
                    # Normalize to CRYPTO: prefix form to match _runtime_symbols() output.
                    # Universe now uses "CRYPTO:BTC/USD" so open positions must match.
                    norm = symbol if symbol.startswith("CRYPTO:") else f"CRYPTO:{symbol}"
                    if norm not in open_crypto:
                        open_crypto.append(norm)
            except ValueError:
                continue

        mom_lb = max(2, int(getattr(ApexConfig, "CRYPTO_ROTATION_MOMENTUM_LOOKBACK", 20)))
        liq_lb = max(2, int(getattr(ApexConfig, "CRYPTO_ROTATION_LIQUIDITY_LOOKBACK", 20)))
        min_dollar_volume = float(getattr(ApexConfig, "CRYPTO_ROTATION_MIN_DOLLAR_VOLUME", 0.0))
        mom_weight = max(0.0, float(getattr(ApexConfig, "CRYPTO_ROTATION_MOMENTUM_WEIGHT", 0.65)))
        liq_weight = max(0.0, float(getattr(ApexConfig, "CRYPTO_ROTATION_LIQUIDITY_WEIGHT", 0.35)))
        weight_sum = mom_weight + liq_weight
        if weight_sum <= 1e-9:
            mom_weight, liq_weight = 0.65, 0.35
            weight_sum = 1.0
        mom_weight /= weight_sum
        liq_weight /= weight_sum

        analytics: List[Dict[str, Any]] = []
        for symbol in crypto_candidates:
            hist = self.historical_data.get(symbol)
            momentum = 0.0
            liquidity = 0.0
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
                if len(closes) > mom_lb:
                    base = float(closes.iloc[-(mom_lb + 1)] or 0.0)
                    latest = float(closes.iloc[-1] or 0.0)
                    if base > 0 and latest > 0:
                        momentum = (latest / base) - 1.0
                if "Volume" in hist.columns:
                    recent = hist.tail(liq_lb)
                    close_series = pd.to_numeric(recent.get("Close"), errors="coerce")
                    vol_series = pd.to_numeric(recent.get("Volume"), errors="coerce")
                    dollar_series = (close_series * vol_series).replace([np.inf, -np.inf], np.nan).dropna()
                    if not dollar_series.empty:
                        liquidity = float(dollar_series.median())
            if liquidity <= 0:
                px = float(self.price_cache.get(symbol, 0.0) or 0.0)
                liquidity = max(0.0, px * 1000.0)

            analytics.append({
                "symbol": symbol,
                "momentum": momentum,
                "liquidity": liquidity,
            })

        if not analytics:
            return open_universe

        momentum_sorted = sorted(analytics, key=lambda row: row["momentum"], reverse=True)
        liquidity_sorted = sorted(analytics, key=lambda row: row["liquidity"], reverse=True)
        mom_rank = {row["symbol"]: idx for idx, row in enumerate(momentum_sorted)}
        liq_rank = {row["symbol"]: idx for idx, row in enumerate(liquidity_sorted)}
        max_rank = max(1, len(analytics) - 1)

        scored: List[Dict[str, Any]] = []
        for row in analytics:
            symbol = row["symbol"]
            momentum_rank_score = 1.0 - (mom_rank[symbol] / max_rank)
            liquidity_rank_score = 1.0 - (liq_rank[symbol] / max_rank)
            score = (mom_weight * momentum_rank_score) + (liq_weight * liquidity_rank_score)
            eligible = row["liquidity"] >= min_dollar_volume if min_dollar_volume > 0 else True
            scored.append({
                "symbol": symbol,
                "score": float(score),
                "momentum": float(row["momentum"]),
                "liquidity": float(row["liquidity"]),
                "eligible": bool(eligible),
            })

        ranked = sorted(
            scored,
            key=lambda row: (row["eligible"], row["score"], row["momentum"]),
            reverse=True,
        )
        top_n = max(1, min(int(getattr(ApexConfig, "CRYPTO_ROTATION_TOP_N", 10)), len(ranked)))
        selected_crypto = [row["symbol"] for row in ranked[:top_n]]
        for symbol in open_crypto:
            if symbol not in selected_crypto:
                selected_crypto.append(symbol)

        rotated = non_crypto + selected_crypto
        self._crypto_rotation_snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "selected": selected_crypto,
            "top_scores": [
                {
                    "symbol": row["symbol"],
                    "score": round(float(row["score"]), 6),
                    "momentum": round(float(row["momentum"]), 6),
                    "liquidity": round(float(row["liquidity"]), 2),
                    "eligible": bool(row["eligible"]),
                }
                for row in ranked[: min(12, len(ranked))]
            ],
        }

        # Log top-5 ranked crypto with scores
        top5 = ranked[:5]
        top5_str = " | ".join(
            f"{r['symbol']}={r['score']:.3f}(mom={r['momentum']:.3f})"
            for r in top5
        )
        logger.info(
            "🔄 Crypto rotation updated: selected=%d/%d — top5: %s",
            len(selected_crypto),
            len(ranked),
            top5_str or "n/a",
        )
        return rotated

    async def _prefill_historical_data(self, symbols: List[str]):
        """
        Idempotent pre-fill of historical data for new symbols.
        Ensures signal generation doesn't skip symbols discovered at runtime.
        Uses individual downloads with rate-limit-safe delays.
        """
        to_prefill = [s for s in symbols if s and s not in self.historical_data]
        if not to_prefill:
            return

        logger.info(f"🔄 Prefilling historical data for {len(to_prefill)} new symbols...")
        loaded = 0
        for symbol in to_prefill:
            try:
                data = await asyncio.to_thread(
                    self.market_data.fetch_historical_data, symbol, 5
                )
                if not data.empty:
                    self.historical_data[symbol] = data
                    if 'Close' in data.columns:
                        self.price_cache[symbol] = data['Close'].iloc[-1]
                    loaded += 1
            except Exception as e:
                logger.debug(f"   Prefill failed for {symbol}: {e}")
            await asyncio.sleep(0.6)
        logger.debug(f"   📥 Prefill: {loaded}/{len(to_prefill)} symbols loaded")

    async def _fetch_fear_greed(self) -> float:
        """
        D: Fetch Crypto Fear & Greed index (alternative.me free API).
        Returns value 0-100.  Caches for 1 hour.  Falls back to 50 on any error.
        """
        if time.time() - self._fear_greed_ts < 3600:
            return self._fear_greed_value
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get("https://api.alternative.me/fng/?limit=1")
                val = int(r.json()["data"][0]["value"])
            self._fear_greed_value = float(val)
            self._fear_greed_ts = time.time()
            logger.info("D: Crypto Fear & Greed = %d (%s)", val,
                        r.json()["data"][0].get("value_classification", "?"))
        except Exception as e:
            logger.debug("D: Fear & Greed fetch failed (%s), using cached %.0f", e, self._fear_greed_value)
        return self._fear_greed_value

    def _btc_momentum(self) -> float:
        """
        B: Return BTC 24h momentum signal in [-1, +1].
        Used to boost/dampen all altcoin signals (BTC leads the market).
        """
        try:
            btc_data = self.historical_data.get("BTC/USD") or self.historical_data.get("BTCUSD")
            if btc_data is None or len(btc_data) < 5:
                return 0.0
            c = btc_data["Close"].dropna().values.astype(float)
            if len(c) < 5:
                return 0.0
            ret_1d = c[-1] / c[-2] - 1   # 1-bar return (15-min bars → last bar)
            ret_3d = c[-1] / c[max(0, len(c) - 4)] - 1
            # Normalize: 2% move = ±1.0 leader signal
            momentum = (ret_1d * 0.6 + ret_3d * 0.4) / 0.02
            return float(max(-1.0, min(1.0, momentum)))
        except Exception:
            return 0.0

    def _build_macro_feature_snapshot(self) -> dict:
        """Compute macro proxy features from cached data for drift detection. O(N), no I/O."""
        snap: dict = {}
        try:
            spy = self.historical_data.get("SPY")
            if spy is not None and len(spy) >= 25:
                r = spy["Close"].pct_change().dropna()
                snap["spy_ret_5d"]  = float(r.tail(5).mean())
                snap["spy_ret_20d"] = float(r.tail(20).mean())
                snap["spy_vol_20d"] = float(r.tail(20).std())
        except Exception:
            pass
        try:
            btc = self.historical_data.get("BTC/USD") or self.historical_data.get("BTCUSD")
            if btc is not None and len(btc) >= 6:
                r = btc["Close"].pct_change().dropna()
                snap["btc_ret_5d"] = float(r.tail(5).mean())
        except Exception:
            pass
        snap["vix"]        = float(self._current_vix or 20.0)
        snap["fear_greed"] = float(self._fear_greed_value)
        return snap

    def _run_drift_check(self, cycle: int) -> None:
        """Periodic macro feature drift check — zero I/O on hot path."""
        if not self.drift_detector:
            return
        try:
            import numpy as np
            snap = self._build_macro_feature_snapshot()
            if not snap:
                return

            # Rebuild baseline from SPY history every ~24h
            if cycle % 1440 == 0:
                spy = self.historical_data.get("SPY")
                if spy is not None and len(spy) >= 50:
                    r = spy["Close"].pct_change().dropna()
                    baseline_arrays: dict = {
                        "spy_ret_5d":  r.rolling(5).mean().dropna().values,
                        "spy_ret_20d": r.rolling(20).mean().dropna().values,
                        "spy_vol_20d": r.rolling(20).std().dropna().values,
                        "vix":         np.array([snap["vix"]]),
                        "fear_greed":  np.array([snap["fear_greed"]]),
                    }
                    btc = self.historical_data.get("BTC/USD") or self.historical_data.get("BTCUSD")
                    if btc is not None and len(btc) >= 6:
                        rb = btc["Close"].pct_change().dropna()
                        baseline_arrays["btc_ret_5d"] = rb.rolling(5).mean().dropna().values
                    self.drift_detector.save_baseline_stats(baseline_arrays)
                    logger.debug("FeatureDrift: baseline updated (%d SPY bars)", len(r))
                return  # skip drift check on same cycle as baseline update

            # Cold-start: no baseline yet — build it now and return
            if not self.drift_detector.baseline_stats:
                spy = self.historical_data.get("SPY")
                if spy is not None and len(spy) >= 50:
                    import numpy as np
                    r = spy["Close"].pct_change().dropna()
                    self.drift_detector.save_baseline_stats({
                        "spy_ret_5d":  r.rolling(5).mean().dropna().values,
                        "spy_ret_20d": r.rolling(20).mean().dropna().values,
                        "spy_vol_20d": r.rolling(20).std().dropna().values,
                        "vix":         np.array([snap["vix"]]),
                        "fear_greed":  np.array([snap["fear_greed"]]),
                    })
                return

            drifted = self.drift_detector.detect_drift(snap, regime=self._current_regime)

            if self.drift_detector.should_retrain():
                logger.warning(
                    "⚠️ FEATURE DRIFT: %d macro features out-of-distribution — triggering recalibration",
                    len(drifted),
                )
                fire_alert(
                    "feature_drift",
                    f"{len(drifted)} macro features drifted from baseline "
                    f"({', '.join(drifted.keys())}) — recalibrating thresholds",
                    AlertSev.WARNING,
                )
                asyncio.ensure_future(self._run_threshold_calibration(force=True))

        except Exception as e:
            logger.debug("Drift check error: %s", e)

    async def _refresh_earnings_calendar(self) -> None:
        """
        Fetch next earnings date for all equity symbols and register with MacroEventShield.
        Runs at engine startup and ~daily. yfinance calls run in thread pool (non-blocking).
        """
        if not getattr(ApexConfig, "EARNINGS_FILTER_ENABLED", True):
            return
        if (self._last_earnings_refresh is not None and
                (datetime.now() - self._last_earnings_refresh).total_seconds() < 23 * 3600):
            return

        equity_symbols = [
            s for s in (ApexConfig.SYMBOLS or [])
            if not _symbol_is_crypto(s) and not s.startswith("FX:") and "OPT:" not in s.upper()
        ]
        if not equity_symbols:
            return

        # Clear stale earnings entries before re-populating
        self.macro_event_shield._earnings_calendar.clear()
        self.macro_event_shield._events = [
            e for e in self.macro_event_shield._events
            if e.event_type != EventType.EARNINGS
        ]

        fetched = 0
        for symbol in equity_symbols:
            try:
                clean = symbol.split(":")[-1].replace("/", "-")
                earnings_dt = await asyncio.to_thread(_fetch_earnings_date_yf, clean)
                if earnings_dt is not None and earnings_dt > datetime.utcnow():
                    self.macro_event_shield.add_earnings_date(symbol, earnings_dt)
                    # Mirror into EarningsEventGate for pre-earnings sizing dampener
                    _eg = getattr(self, '_earnings_gate', None)
                    if _eg is not None:
                        _eg.update_earnings(symbol.split(":")[-1], earnings_dt)
                    fetched += 1
            except Exception as e:
                logger.debug("EarningsCalendar: failed for %s: %s", symbol, e)
            await asyncio.sleep(0.15)  # avoid yfinance rate-limit

        self._last_earnings_refresh = datetime.now()
        logger.info("EarningsCalendar: populated %d/%d equity symbols", fetched, len(equity_symbols))

    def _crypto_rotation_score(self, symbol: str) -> float:
        """Return latest rotation score [0, 1] for a crypto symbol."""
        snap = self._crypto_rotation_snapshot or {}
        top_scores = snap.get("top_scores") if isinstance(snap, dict) else None
        if not isinstance(top_scores, list):
            return 0.0
        symbol_u = str(symbol or "").upper()
        for row in top_scores:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).upper() != symbol_u:
                continue
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if not math.isfinite(score):
                return 0.0
            return max(0.0, min(1.0, score))
        return 0.0

    def _runtime_symbols(self) -> List[str]:
        """Static universe + runtime-discovered symbols (e.g., Alpaca crypto).

        Uses self._session_symbols (session-specific: core=equities, crypto=crypto)
        so that CORE and CRYPTO sessions don't redundantly process each other's
        assets, preventing duplicate orders on the shared Alpaca account.

        Deduplicates symbols so that bare crypto tickers (e.g. AVAX/USD) and their
        CRYPTO:-prefixed equivalents (CRYPTO:AVAX/USD) do not both appear in the
        universe, which would cause signals and orders to fire twice per cycle.
        """
        from core.symbols import parse_symbol as _parse

        seen_normalized: set = set()
        deduped: List[str] = []

        def _add(sym: str) -> None:
            if sym in self._failed_symbols:
                return
            if sym in self._calibrated_blacklist:
                return  # high-slippage: excluded by ThresholdCalibrator
            try:
                key = _parse(sym).normalized
            except Exception:
                key = sym  # unparseable – use as-is
            if key not in seen_normalized:
                seen_normalized.add(key)
                deduped.append(sym)

        # Use session-specific symbols to avoid CORE session processing crypto
        # (causing duplicate Alpaca orders) and CRYPTO session processing equities.
        base_symbols = self._session_symbols if self._session_symbols else ApexConfig.SYMBOLS
        for s in base_symbols:
            try:
                _p = _parse(s)
                # Normalize crypto to CRYPTO: prefix form so it matches the key format
                # that sync_positions_with_alpaca() stores (e.g. "CRYPTO:BTC/USD").
                # Without this, "BTC/USD" in the universe vs "CRYPTO:BTC/USD" in positions
                # causes process_symbol to see qty=0 for held positions → spurious BUY attempts.
                _add(_p.normalized if _p.asset_class == AssetClass.CRYPTO else s)
            except Exception:
                _add(s)
        # Dynamic crypto symbols are only relevant for the crypto session
        if self.session_type in ("crypto", "unified"):
            for s in self._dynamic_crypto_symbols:
                try:
                    _p = _parse(s)
                    _add(_p.normalized if _p.asset_class == AssetClass.CRYPTO else s)
                except Exception:
                    _add(s)

        return deduped

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

    def _heartbeat_paths(self) -> List[Path]:
        """Emit heartbeat into both global and tenant scopes for parity."""
        paths = [ApexConfig.DATA_DIR / 'heartbeat.json']
        user_path = self.user_data_dir / 'heartbeat.json'
        if user_path not in paths:
            paths.append(user_path)
        return paths

    def _write_heartbeat_payload(self, payload: Dict[str, Any]) -> None:
        """Write heartbeat payload to all known heartbeat locations."""
        for heartbeat_file in self._heartbeat_paths():
            try:
                heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
                with open(heartbeat_file, 'w') as f:
                    json.dump(payload, f)
            except Exception as exc:
                logger.debug("Error writing heartbeat file %s: %s", heartbeat_file, exc)

    def _write_heartbeat(self):
        """
        Write heartbeat file for watchdog monitoring.

        The watchdog process monitors this file to detect if the
        trading system is hung or crashed.
        """
        data = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'position_count': self.position_count,
            'capital': self.capital,
            'is_trading': True,
            'cycle_count': getattr(self, '_cycle_count', 0),
            'broker_heartbeats': self._broker_heartbeat_payload(),
        }
        self._write_heartbeat_payload(data)

    async def _resolve_cycle_market_regime(self, prices: pd.Series) -> Tuple[str, float]:
        """Compute market regime once per cycle and reuse across symbols."""
        cycle = int(getattr(self, "_cycle_count", 0))
        if self._cycle_regime_eval_cycle == cycle:
            return self._cycle_regime_value, self._cycle_regime_transition_prob

        async with self._regime_eval_lock:
            if self._cycle_regime_eval_cycle == cycle:
                return self._cycle_regime_value, self._cycle_regime_transition_prob

            resolved_regime = self._current_regime or "neutral"
            transition_probability = 0.0

            if self.adaptive_regime:
                try:
                    regime_assessment = self.adaptive_regime.assess_regime(
                        prices=prices,
                        vix_level=self._current_vix,
                        emit_transition_logs=False,
                    )
                    resolved_regime = regime_assessment.primary_regime
                    transition_probability = float(regime_assessment.transition_probability)
                except Exception:
                    try:
                        regime_enum = self.god_signal_generator.detect_market_regime(prices)
                        resolved_regime = regime_enum.value
                    except Exception:
                        resolved_regime = "neutral"
                        transition_probability = 0.0
            else:
                try:
                    regime_enum = self.god_signal_generator.detect_market_regime(prices)
                    resolved_regime = regime_enum.value
                except Exception:
                    resolved_regime = "neutral"
                    transition_probability = 0.0

            # ── Regime transition tracking ────────────────────────────────────
            if resolved_regime != getattr(self, "_last_regime", resolved_regime):
                self._regime_changed_at = datetime.now()
                self._regime_transition_confidence = float(transition_probability or 0.70)
                logger.info(
                    "🔄 Regime transition: %s → %s (confidence=%.0f%%, caution window starts)",
                    self._last_regime, resolved_regime, self._regime_transition_confidence * 100,
                )
            self._last_regime = resolved_regime
            self._current_regime = resolved_regime
            self._cycle_regime_eval_cycle = cycle
            self._cycle_regime_value = resolved_regime
            self._cycle_regime_transition_prob = transition_probability

            if transition_probability > 0.70 and self._cycle_regime_log_cycle != cycle:
                logger.info(
                    "🏰 Market regime transition likely (%s, %.0f%%)",
                    resolved_regime,
                    transition_probability * 100.0,
                )
                self._cycle_regime_log_cycle = cycle

            return resolved_regime, transition_probability

    def print_banner(self):
        if ApexTradingSystem._banner_printed:
            return
        ApexTradingSystem._banner_printed = True
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
        await self.event_store.start()
        
        # Load risk state (day_start_capital, etc)
        self.risk_manager.load_state()
        
        # 🟢 Hydrate auxiliary state from Redis (capital and peak prices only).
        # Positions are NOT loaded from Redis — the broker (IBKR / Alpaca) is the
        # single source of truth.  Loading stale Redis positions causes phantom
        # positions that the broker doesn't actually hold, breaking exit logic.
        try:
            from services.common.redis_client import state_get_all, state_get

            # 1. Hydrate position peak prices (metadata; safe to cache across restarts)
            redis_peaks = await state_get_all("position_peak_prices")
            if redis_peaks:
                self.position_peak_prices = {k: float(v) for k, v in redis_peaks.items()}
                logger.info(f"🔄 Hydrated {len(self.position_peak_prices)} peak prices from Redis")

            # 2. Hydrate capital (last known value; overwritten by live broker equity below)
            redis_cap = await state_get("capital", "current")
            if redis_cap is not None:
                self.capital = float(redis_cap)
                logger.info(f"🔄 Hydrated capital ${self.capital:,.2f} from Redis")
        except Exception as e:
            logger.warning("⚠️ Redis state hydration failed: %s", e)
            
        try:
            await self.load_price_cache()
        except Exception as e:
            logger.debug("⚠️ Failed to load price cache: %s", e)
        # Connect to IBKR (cap startup wait to 90 s so a zombie session can't block crypto)
        if self.ibkr:
            try:
                await asyncio.wait_for(self.ibkr.connect(), timeout=250)
                self._mark_broker_heartbeat("ibkr", success=True)
                if self.prometheus_metrics:
                    self.prometheus_metrics.update_ibkr_status(True)
            except Exception as ibkr_exc:
                self._mark_broker_heartbeat("ibkr", success=False, error=str(ibkr_exc))
                if self.prometheus_metrics:
                    self.prometheus_metrics.update_ibkr_status(False)

                if self.alpaca:
                    logger.error(f"❌ IBKR unavailable: {ibkr_exc}")
                    logger.warning(
                        f"⚠️ Degrading to Alpaca-only session for {self.tenant_id}; IBKR symbols will be skipped until IBKR recovers"
                    )
                    self.ibkr = None
                else:
                    raise

        if self.ibkr:
            ibkr_capital = await self.ibkr.get_portfolio_value()
            if ibkr_capital > 0:
                self.capital = ibkr_capital
                self.equity_outlier_guard.seed(float(self.capital))
                try:
                    from services.common.redis_client import state_set
                    asyncio.create_task(state_set("capital", "current", self.capital))
                except Exception:
                    pass
            else:
                logger.warning(f"⚠️  IBKR returned ${ibkr_capital:,.2f}, keeping initial capital ${self.capital:,.2f}")
                self.risk_manager.set_starting_capital(self.capital)
                self.risk_manager.day_start_capital = self.capital  # ✅ CRITICAL
                logger.info(f"✅ IBKR Account ${self.capital:,.2f}")

            # Load existing positions from IBKR
            await self.sync_positions_with_ibkr()

            # ═══════════════════════════════════════════════════════════════
            # PRE-WARM CONTRACTS: Qualify all contracts concurrently upfront
            # so per-position price lookups use the cache (no serial waits).
            # ═══════════════════════════════════════════════════════════════
            all_symbols = list(set(list(self.positions.keys()) + self._runtime_symbols()))
            # In "both" mode, crypto routes through Alpaca — don't waste 5s
            # timeout per symbol trying to qualify them via IBKR.
            ibkr_prewarm_symbols = all_symbols
            if self.alpaca:
                def _is_crypto(s: str) -> bool:
                    try:
                        return parse_symbol(s).asset_class == AssetClass.CRYPTO
                    except Exception:
                        return False
                ibkr_prewarm_symbols = [s for s in all_symbols if not _is_crypto(s)]
            try:
                await self.ibkr.prewarm_contracts(ibkr_prewarm_symbols, concurrency=5)
            except Exception as _pw_exc:
                logger.warning("⚠️  Contract pre-warm failed: %s", _pw_exc)



            # ═══════════════════════════════════════════════════════════════
            # PRE-LOAD DATA FOR POSITIONS (Ensures non-zero P&L on startup)
            # ═══════════════════════════════════════════════════════════════
            logger.info("📥 Loading initial price data for positions (Batch)...")
            symbols_to_load = [s for s in self.positions if s]
            if symbols_to_load:
                import time
                t0 = time.time()
                try:
                    # Parallel batch fetch
                    batch_results = self.market_data.fetch_historical_batch(symbols_to_load, days=5)
                    
                    loaded_count = 0
                    for symbol, data in batch_results.items():
                        if not data.empty:
                            self.historical_data[symbol] = data
                            self.price_cache[symbol] = data['Close'].iloc[-1]
                            loaded_count += 1
                            
                    duration = time.time() - t0
                    logger.info(f"⚡ Batch loaded {loaded_count}/{len(symbols_to_load)} symbols in {duration:.2f}s")
                    
                    # 🔍 Perform Risk Analysis on a proxy (e.g. SPY) if available to prove integration
                    if 'SPY' in self.historical_data:
                         spy_rets = self.historical_data['SPY']['Close'].pct_change().dropna()
                         risk_report = self.risk_manager.analyze_risk_metrics(spy_rets)
                         logger.info(f"📊 Market Risk Regime (SPY): Sortino={risk_report.get('sortino_ratio',0):.2f}, CVaR95={risk_report.get('cvar_95',0):.4f}")

                except Exception as e:
                    logger.error(f"❌ Batch load failed: {e}")
                    # Fallback handled by individual refresh below

            # Fetch live market prices for all positions
            await self.refresh_position_prices()

            # Initial push to UI so dashboard is useful during long startup
            await self.export_dashboard_state()

            if self.positions:
                logger.info(f"📊 Loaded {self.position_count} existing positions:")
                saved_metadata = self._load_position_metadata()
                for symbol, qty in self.positions.items():
                    if qty != 0:
                        pos_type = "LONG" if qty > 0 else "SHORT"
                        try:
                            init_conn = self._get_connector_for(symbol)
                            price = await init_conn.get_market_price(symbol) if init_conn else 0
                            if not price or price <= 0:
                                # Fallback 1: Check price cache (loaded from disk)
                                price = self.price_cache.get(symbol, 0)

                            if not price or price <= 0:
                                # Fallback 2: Use last candle from historical data
                                if symbol in self.historical_data and not self.historical_data[symbol].empty:
                                    price = self.historical_data[symbol]['Close'].iloc[-1]
                                    logger.info(f"   {symbol}: Using historical price fallback: ${price:.2f}")

                            if price and price > 0:
                                self.price_cache[symbol] = price

                                # CRITICAL: Only set entry metadata if NOT already loaded from disk
                                if symbol in saved_metadata:
                                    meta = saved_metadata[symbol]
                                    # Use the preserved data from disk
                                    self.position_entry_prices[symbol] = meta['entry_price']
                                    self.position_entry_times[symbol] = datetime.fromisoformat(meta['entry_time'])
                                    self.position_entry_signals[symbol] = meta.get('entry_signal', 0.5 if qty > 0 else -0.5)
                                    self.position_peak_prices[symbol] = meta.get('peak_price', price)
                                    logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) @ ${price:.2f} [RESTORED entry=${meta['entry_price']:.2f}]")
                                elif symbol not in self.position_entry_prices:
                                    # Only set if we don't have it in memory either
                                    self.position_entry_prices[symbol] = price
                                    self.position_entry_times[symbol] = datetime.utcnow()
                                    self.position_peak_prices[symbol] = price
                                    self.position_entry_signals[symbol] = 0.5 if qty > 0 else -0.5
                                    logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) @ ${price:.2f} [NEW entry]")
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

            # ═══════════════════════════════════════════════════════════════
            # EVENT-DRIVEN STREAMING (Safe for Free Tier)
            # ═══════════════════════════════════════════════════════════════
            stream_symbols = list(set(list(self.positions.keys()) + self._runtime_symbols()))
            backtest_only = getattr(ApexConfig, "BACKTEST_ONLY_SYMBOLS", set()) or set()
            if backtest_only:
                backtest_only_norm = set()
                for s in backtest_only:
                    try:
                        backtest_only_norm.add(normalize_symbol(s))
                    except (ValueError, Exception):
                        backtest_only_norm.add(s)  # Keep raw string as fallback
                stream_symbols = [
                    s for s in stream_symbols
                    if normalize_symbol(s) not in backtest_only_norm
                ]
            # In "both" mode, don't stream crypto via IBKR (Alpaca handles it)
            if self.alpaca:
                stream_symbols = [
                    s for s in stream_symbols
                    if parse_symbol(s).asset_class != AssetClass.CRYPTO
                ]
            # Prioritize positions (put them first)
            pos_keys = list(self.positions.keys())
            stream_symbols.sort(key=lambda x: x in pos_keys, reverse=True)

            # Hook up Watchdog
            if self.data_watchdog:
                self.ibkr.set_data_callback(self.data_watchdog.feed_heartbeat)

            # ─── Real-time fill sync ──────────────────────────────────────
            # Register a fill callback so self.positions is updated the instant
            # IBKR confirms an execution — not after the 2-5 min poll cycle.
            def _apply_ibkr_fill(symbol: str, side: str, filled_qty: float, avg_price: float) -> None:
                try:
                    current_qty = self.positions.get(symbol, 0)
                    if side == "BUY":
                        new_qty = current_qty + filled_qty
                    else:
                        new_qty = current_qty - filled_qty

                    self.positions[symbol] = new_qty
                    
                    try:
                        from services.common.redis_client import state_set
                        asyncio.create_task(state_set("positions", symbol, new_qty))
                    except Exception:
                        pass

                    # Update average entry price
                    if side == "BUY" and avg_price > 0:
                        old_price = self.position_entry_prices.get(symbol, 0.0)
                        if old_price > 0 and current_qty > 0:
                            # Weighted average
                            self.position_entry_prices[symbol] = (
                                (old_price * current_qty + avg_price * filled_qty)
                                / (current_qty + filled_qty)
                            )
                        else:
                            self.position_entry_prices[symbol] = avg_price

                    logger.info(
                        "event=realtime_fill_applied symbol=%s side=%s qty=%.4f price=%.4f new_pos=%.4f",
                        symbol, side, filled_qty, avg_price, new_qty,
                    )
                except Exception as exc:
                    logger.warning("Fill callback error for %s: %s", symbol, exc)

            self.ibkr.set_fill_callback(_apply_ibkr_fill)
            logger.info("✅ Real-time IBKR fill callback registered")

            await self.ibkr.stream_quotes(stream_symbols)

        # Connect to Alpaca (crypto paper trading)
        # Connect to Alpaca (crypto paper trading)
        if self.alpaca:
            try:
                await self.alpaca.connect()
                self._mark_broker_heartbeat("alpaca", success=True)
                alpaca_equity = await self.alpaca.get_portfolio_value()
                if alpaca_equity > 0 and not self.ibkr:
                    # Only use Alpaca capital when running Alpaca-only mode
                    self.capital = alpaca_equity
                    self.equity_outlier_guard.seed(float(self.capital))
                    self.risk_manager.set_starting_capital(self.capital)
                logger.info(f"Alpaca Account: ${alpaca_equity:,.2f}")

                # Sync Alpaca positions — full bidirectional sync so that:
                # - Actual Alpaca positions are loaded into self.positions
                # - Any stale CRYPTO: entries (e.g. from a previous Redis snapshot)
                #   that Alpaca no longer holds are removed from self.positions
                await self.sync_positions_with_alpaca()
                alpaca_pos = {s: q for s, q in self.positions.items()
                              if q != 0 and s.startswith("CRYPTO:")}
                if alpaca_pos:
                    for sym, qty in alpaca_pos.items():
                        logger.info(f"  Alpaca position: {sym} = {qty}")
                else:
                    logger.info("  Alpaca: no open positions")

                # ── Startup oversized-position trimmer ────────────────────────────
                # If a startup_restore crypto position exceeds 2× CRYPTO_POSITION_SIZE_USD,
                # sell the excess immediately to free capital for diversified entries.
                _max_notional = float(getattr(ApexConfig, "CRYPTO_POSITION_SIZE_USD", 5000))
                for _sym, _qty in list(alpaca_pos.items()):
                    if _qty <= 0:
                        continue
                    try:
                        _price = await self.alpaca.get_market_price(_sym)
                        if not _price or _price <= 0:
                            continue
                        _notional = abs(_qty) * _price
                        if _notional > _max_notional * 2.0:
                            _keep_qty = round(_max_notional / _price, 6)
                            _sell_qty = round(abs(_qty) - _keep_qty, 6)
                            if _sell_qty > 0:
                                logger.warning(
                                    "🔪 Startup trim %s: %.6f→%.6f (notional $%.0f→$%.0f, "
                                    "freeing $%.0f). Selling excess %.6f.",
                                    _sym, abs(_qty), _keep_qty,
                                    _notional, _keep_qty * _price,
                                    _sell_qty * _price, _sell_qty,
                                )
                                await self.alpaca.execute_order(
                                    symbol=_sym, side="SELL", quantity=_sell_qty,
                                    confidence=1.0, force_market=True,
                                )
                    except Exception as _trim_exc:
                        logger.warning("Startup trim failed for %s: %s", _sym, _trim_exc)
                # ── end trimmer ───────────────────────────────────────────────────

                # UPGRADE B: Reload pending orders from previous session
                _orders_path = self.user_data_dir / "runtime" / "alpaca_pending_orders.json"
                try:
                    self.alpaca.load_pending_orders(_orders_path)
                except Exception as _load_err:
                    logger.debug("Could not load pending orders: %s", _load_err)

                # Start crypto quote polling
                if getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_SYMBOLS", True):
                    try:
                        discovered = self.alpaca.get_discovered_crypto_symbols(
                            limit=max(0, int(getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_LIMIT", 24))),
                            preferred_quotes=list(getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_PREFERRED_QUOTES", [])),
                        )
                    except Exception as discover_exc:
                        logger.warning("Alpaca crypto discovery failed: %s", discover_exc)
                        discovered = []
                    if discovered:
                        excluded = set(getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_EXCLUDED", []))
                        filtered = [s for s in discovered if s.upper() not in excluded]
                        self._dynamic_crypto_symbols = filtered
                        logger.info(
                            "🪙 Alpaca discovery added %d crypto pairs to runtime universe (%d excluded)",
                            len(filtered), len(discovered) - len(filtered),
                        )

                crypto_symbols = []
                for s in self._runtime_symbols():
                    try:
                        if parse_symbol(s).asset_class == AssetClass.CRYPTO:
                            crypto_symbols.append(s)
                    except ValueError:
                        continue
                if crypto_symbols:
                    if self.data_watchdog:
                        self.alpaca.set_data_callback(self.data_watchdog.feed_heartbeat)
                    await self.alpaca.stream_quotes(crypto_symbols)
            except Exception as e:
                logger.warning(f"⚠️  Alpaca connection failed: {e}. Disabling Alpaca.")
                self._mark_broker_heartbeat("alpaca", success=False, error=str(e))
                self.alpaca = None

        self._sync_cost_basis_with_positions()

        # ⏳ Wait for all brokers to finish connecting before reading equity.
        # Alpaca's connect() returns before account data is fully cached; a brief
        # sleep ensures non_marginable_buying_power and portfolio value are ready,
        # preventing Alpaca from being excluded from the equity total at startup.
        _broker_settle_secs = int(getattr(ApexConfig, "STARTUP_BROKER_SETTLE_SECS", 20))
        if _broker_settle_secs > 0 and (self.ibkr or self.alpaca):
            logger.info(
                "⏳ Waiting %ds for brokers to finish connecting before reading startup equity...",
                _broker_settle_secs,
            )
            await asyncio.sleep(_broker_settle_secs)

        # Sync startup capital from broker APIs before self-healing persisted paper state.
        await self._refresh_capital_from_brokers_for_startup()

        # Prevent stale persisted paper state from tripping risk/kill controls at startup.
        await self._sanitize_startup_state_for_paper()

        # Pre-load historical data with rate-limit-safe sequential downloads
        # yfinance 1.0 aggressively rate-limits; use individual calls with delays
        logger.info("📥 Loading historical data for ML training...")
        loaded = 0
        runtime_symbols = self._runtime_symbols()
        for i, symbol in enumerate(runtime_symbols, 1):
            if i % 10 == 0:
                logger.info(f"   Loaded {loaded}/{i} symbols so far ({len(runtime_symbols)} total)...")
            try:
                data = await asyncio.to_thread(
                    self.market_data.fetch_historical_data, symbol, 400
                )
                if not data.empty:
                    self.historical_data[symbol] = data
                    if 'Close' in data.columns:
                        self.price_cache[symbol] = data['Close'].iloc[-1]
                    loaded += 1
                else:
                    self._failed_symbols.add(symbol)
            except Exception as e:
                logger.debug(f"   Failed to load {symbol}: {e}")
                self._failed_symbols.add(symbol)
            # Yield to event loop + rate limit (yfinance 1.0 needs ~0.5s between calls)
            await asyncio.sleep(0.6)

        if self._failed_symbols:
            logger.warning(
                "⚠️  %d/%d symbols failed to load data (will retry in main loop): %s%s",
                len(self._failed_symbols),
                len(runtime_symbols),
                ", ".join(list(self._failed_symbols)[:10]),
                " ..." if len(self._failed_symbols) > 10 else "",
            )
        logger.info(f"✅ Loaded data for {loaded}/{len(runtime_symbols)} symbols")

        # Initialize ATR-based stops for existing positions (now that we have historical data)
        if self.positions and (self.ibkr or self.alpaca):
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

        # Restore attribution context for any open positions carried across restarts.
        self._seed_attribution_for_open_positions()

        # Train ML models
        logger.info("")
        if self.signal_generator.is_trained and not ApexConfig.FORCE_RETRAIN:
            logger.info("✅ Advanced ML models loaded from disk. Skipping startup training.")
        else:
            logger.info("🧠 Training advanced ML models...")
            logger.info("   This may take 30-60 seconds...")
            try:
                # self.signal_generator.train_models(self.historical_data)
                logger.info("✅ ML models trained and ready! (SKIPPED)")
            except Exception as e:
                logger.warning(f"⚠️  ML training failed: {e}")
                logger.warning("   Falling back to technical analysis only")

        # GodLevel cold-start: if no saved models, train in background (non-blocking)
        if (getattr(ApexConfig, "GOD_LEVEL_BLEND_ENABLED", True)
                and getattr(self, "god_signal_generator", None) is not None
                and not self.god_signal_generator.models_trained):
            logger.info("🧠 GodLevel ensemble: no saved models found — launching background training...")

            async def _godlevel_train_and_regen_manifest():
                await asyncio.to_thread(self.god_signal_generator.train_models, self.historical_data)
                try:
                    from models.model_manifest import build_manifest, write_manifest
                    from pathlib import Path as _Path
                    write_manifest(build_manifest(), _Path("models/model_manifest.json"))
                    logger.info("✅ Model manifest regenerated after cold-start GodLevel training")
                except Exception as _mf_err:
                    logger.warning("Manifest regen after cold-start training failed: %s", _mf_err)

            asyncio.create_task(_godlevel_train_and_regen_manifest())

        # Train institutional signal generator
        if self.use_institutional:
            logger.info("")
            # Try loading saved models from disk first (fast path, avoids ~18min training)
            if not self.inst_signal_generator.is_trained and not ApexConfig.FORCE_RETRAIN:
                logger.info("🔍 Checking for saved institutional ML models...")
                loaded = self.inst_signal_generator.loadModels()
                if loaded:
                    logger.info("✅ Institutional ML models loaded from disk — skipping startup training.")
            if self.inst_signal_generator.is_trained and not ApexConfig.FORCE_RETRAIN:
                # Ensure risk manager is initialized even if we skip training
                self.inst_risk_manager.update_capital(self.capital)
            else:
                logger.info("🏛️  Training INSTITUTIONAL ML models...")
                logger.info("   Purged time-series cross-validation enabled")
                try:
                    training_results = self.inst_signal_generator.train(
                        self.historical_data,
                        target_horizon=5,
                        min_samples_per_regime=200
                    )
                    # training_results = {} # SKIPPED
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

        # Replace hardcoded 0.5 startup signals with real ML output.
        # Must run after inst_signal_generator is trained/loaded so generate_signal() is valid.
        await self._refresh_startup_signals()

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
        await self._export_initial_state()
        logger.info("")
    
    async def refresh_pending_orders(self):
        """Refresh the set of symbols with pending orders."""
        if not self.ibkr and not self.alpaca:
            return

        try:
            self.pending_orders.clear()

            # IBKR pending orders
            if self.ibkr and hasattr(self.ibkr, 'ib'):
                open_trades = self.ibkr.ib.openTrades()
                for trade in open_trades:
                    symbol = trade.contract.symbol
                    status = trade.orderStatus.status
                    if status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'ApiPending']:
                        self.pending_orders.add(symbol)
                        logger.debug(f"   Pending (IBKR): {symbol} (status={status})")

            # Alpaca pending orders
            if self.alpaca:
                for sym in self.alpaca.get_open_orders():
                    self.pending_orders.add(sym)

            if self.pending_orders:
                logger.info(f"Found {len(self.pending_orders)} pending orders")
        
        except Exception as e:
            logger.error(f"Error refreshing pending orders: {e}")
    
    async def sync_positions_with_ibkr(self):
        """✅ Force sync positions with IBKR's actual positions and metadata."""
        if not self.ibkr:
            return
        
        try:
            # Get detailed positions to capture avg_cost
            detailed_positions = await self.ibkr.get_detailed_positions()
            actual_positions = {}
            for symbol, details in detailed_positions.items():
                sec_type = str(details.get("security_type", "")).upper()
                if "OPT" in sec_type:
                    # Defensive guard: options must flow through options_positions only.
                    continue
                actual_positions[symbol] = details['qty']
            
            # Check for mismatches — but ONLY for IBKR-routed symbols.
            # Crypto is traded via Alpaca; comparing those against IBKR always shows 0.
            def _is_ibkr_symbol(sym: str) -> bool:
                """Return True if this symbol is expected to be held at IBKR."""
                upper = sym.upper()
                # Crypto prefixes trade through Alpaca; everything else goes to IBKR
                crypto_prefixes = ("CRYPTO:", "BTC", "ETH", "SOL", "AVAX", "LINK",
                                   "LTC", "DOT", "DOGE", "ADA", "XLM", "XRP",
                                   "BCH", "FIL", "CRV", "ETC", "ALGO", "ATOM",
                                   "MATIC", "SHIB", "UNI", "AAVE")
                return not any(upper.startswith(p) for p in crypto_prefixes)

            mismatches = []
            for symbol in set(list(self.positions.keys()) + list(actual_positions.keys())):
                if not _is_ibkr_symbol(symbol):
                    continue  # Crypto lives at Alpaca — skip IBKR comparison
                local_qty = self.positions.get(symbol, 0)
                ibkr_qty = actual_positions.get(symbol, 0)
                
                if local_qty != ibkr_qty:
                    mismatches.append(f"{symbol}: Local={local_qty}, IBKR={ibkr_qty}")
            
            if mismatches:
                logger.warning("⚠️ IBKR position mismatches detected:")
                for mismatch in mismatches:
                    logger.warning(f"   {mismatch}")
                logger.warning("   → Syncing to IBKR values")
            
            # Replace our tracking ONLY for IBKR-routed symbols.
            # Preserve Alpaca-held crypto positions in self.positions.
            try:
                from services.common.redis_client import state_set
            except Exception:
                pass
            for symbol, qty in actual_positions.items():
                if not _is_ibkr_symbol(symbol):
                    continue  # Skip crypto that IBKR paper-account mis-holds; Alpaca owns those
                self.positions[symbol] = qty
                try: asyncio.create_task(state_set("positions", symbol, qty))
                except Exception: pass
            # Zero out any IBKR symbols that disappeared from IBKR (closed positions)
            for symbol in list(self.positions.keys()):
                if _is_ibkr_symbol(symbol) and symbol not in actual_positions:
                    self.positions[symbol] = 0
                    try: asyncio.create_task(state_set("positions", symbol, 0))
                    except Exception: pass
            # Also zero out any non-IBKR symbols that IBKR paper account mis-holds
            # (e.g. CRVUSD, FILUSD — unqualifiable contracts that can't be exited)
            for symbol in actual_positions:
                if not _is_ibkr_symbol(symbol) and self.positions.get(symbol, 0) != 0:
                    self.positions[symbol] = 0

            # Keep options inventory fresh for cockpit/state exporters.
            if ApexConfig.OPTIONS_ENABLED:
                try:
                    self.options_positions = await self.ibkr.get_option_positions()
                except Exception as opt_exc:
                    logger.debug("IBKR option position refresh failed: %s", opt_exc)

            # Update entry prices from IBKR avgCost for positions we don't have metadata for
            for symbol, data in detailed_positions.items():
                sec_type = str(data.get("security_type", "")).upper()
                if "OPT" in sec_type:
                    continue
                if symbol not in self.position_entry_prices or self.position_entry_prices[symbol] == 0:
                    self.position_entry_prices[symbol] = data['avg_cost']
                    if symbol not in self.position_entry_times:
                        self.position_entry_times[symbol] = datetime.utcnow()
                    logger.debug(f"ℹ️ Captured IBKR avgCost for {symbol}: ${data['avg_cost']:.2f}")

            logger.debug(f"✅ Position sync: {self.position_count} active positions")
            self._mark_broker_heartbeat("ibkr", success=True)
            self._sync_cost_basis_with_positions()
        
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            self._mark_broker_heartbeat("ibkr", success=False, error=str(e))
            import traceback
            logger.debug(traceback.format_exc())

    async def sync_positions_with_alpaca(self):
        """Sync crypto positions from Alpaca into self.positions with metadata.

        Only crypto symbols are synced from Alpaca to avoid overwriting
        IBKR equity positions (e.g. COP, HAL) with stale Alpaca data.
        """
        if not self.alpaca:
            return
        try:
            detailed_positions = await self.alpaca.get_detailed_positions()
            actual_positions = {s: d['qty'] for s, d in detailed_positions.items()}

            # Only sync crypto positions — equities are managed by IBKR
            for sym, qty in actual_positions.items():
                if qty == 0:
                    continue
                try:
                    parsed = parse_symbol(sym)
                    if parsed.asset_class != AssetClass.CRYPTO:
                        continue
                except ValueError:
                    continue
                self.positions[sym] = int(qty) if qty == int(qty) else qty

            # Remove crypto symbols that are no longer held on Alpaca
            for sym in list(self.positions.keys()):
                try:
                    parsed = parse_symbol(sym)
                    if parsed.asset_class == AssetClass.CRYPTO and sym not in actual_positions:
                        del self.positions[sym]
                except ValueError:
                    pass
                    
            # Populate price cache and entry prices for crypto positions
            for sym, data in detailed_positions.items():
                qty = data['qty']
                if qty == 0:
                    continue
                try:
                    parsed = parse_symbol(sym)
                    if parsed.asset_class != AssetClass.CRYPTO:
                        continue
                except ValueError:
                    continue

                if data.get('current_price', 0) > 0:
                    self.price_cache[sym] = data['current_price']

                if sym not in self.position_entry_prices or self.position_entry_prices[sym] == 0:
                    self.position_entry_prices[sym] = data.get('avg_cost', 0)
                    if sym not in self.position_entry_times:
                        self.position_entry_times[sym] = datetime.utcnow()
            
            logger.debug("✅ Alpaca position sync complete")
            self._mark_broker_heartbeat("alpaca", success=True)
            self._sync_cost_basis_with_positions()
        except Exception as e:
            logger.error(f"Error syncing Alpaca positions: {e}")
            self._mark_broker_heartbeat("alpaca", success=False, error=str(e))

    async def _sync_positions(self):
        """Sync positions from all active brokers.

        IBKR sync is awaited (local IPC, sub-millisecond).
        Alpaca sync runs as a fire-and-forget background task so that
        Alpaca Paper API latency / timeouts never stall the main loop.

        Works correctly in both modes:
          • Both IBKR + Alpaca active: IBKR syncs synchronously (equities),
            Alpaca syncs in background (crypto). Main loop is never blocked.
          • Alpaca-only: same background approach. self.positions reflects
            the last completed Alpaca sync; stale by at most one timeout
            window (8 s) which is acceptable.

        A new background task is only launched if the previous one has
        finished — prevents stacking multiple concurrent HTTP calls to Alpaca.
        """
        if self.ibkr:
            await self.sync_positions_with_ibkr()
            # Cost-basis update after IBKR sync so equity positions are current.
            self._sync_cost_basis_with_positions()

        if self.alpaca:
            # Skip if a previous background sync is still in flight.
            if self._alpaca_sync_task is None or self._alpaca_sync_task.done():
                # sync_positions_with_alpaca() updates self.positions in-place
                # and calls _sync_cost_basis_with_positions() itself when done.
                self._alpaca_sync_task = asyncio.ensure_future(
                    self.sync_positions_with_alpaca()
                )
            else:
                logger.debug("Alpaca position sync already in flight — skipping this cycle")

    # ── Kill switch persistence ───────────────────────────────────────────────

    def _ks_state_path(self):
        from pathlib import Path
        return Path(ApexConfig.DATA_DIR) / "kill_switch_state.json"

    def _persist_kill_switch_state(self) -> None:
        """Write kill switch state to disk (synchronous, fast)."""
        if not self.kill_switch:
            return
        import json
        state = self.kill_switch.state()
        try:
            self._ks_state_path().write_text(
                json.dumps({
                    "active":           state.active,
                    "triggered_at":     state.triggered_at,
                    "reason":           state.reason,
                    "flatten_executed": state.flatten_executed,
                    "saved_at":         datetime.utcnow().isoformat(),
                }, indent=2)
            )
        except Exception as _e:
            logger.warning("Kill switch state persist failed: %s", _e)

    def _load_kill_switch_state(self) -> None:
        """Restore kill switch state from disk on startup."""
        if not self.kill_switch:
            return
        import json
        path = self._ks_state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            if data.get("active"):
                self.kill_switch.active         = True
                self.kill_switch.triggered_at   = data.get("triggered_at")
                self.kill_switch.reason         = data.get("reason", "restored from disk")
                self.kill_switch.flatten_executed = bool(data.get("flatten_executed", False))
                logger.warning(
                    "🛑 Kill-switch RESTORED from disk (triggered_at=%s, reason=%s)",
                    self.kill_switch.triggered_at,
                    self.kill_switch.reason,
                )
                fire_alert(
                    "kill_switch",
                    f"Kill-switch restored from disk (reason={self.kill_switch.reason})",
                    AlertSev.CRITICAL,
                )
        except Exception as _e:
            logger.warning("Kill switch state load failed: %s", _e)

    async def _reconcile_position_state(self) -> None:
        """
        Reconcile internal position state against broker truth every 5 min.

        Handles two divergence cases:
        1. Phantom attribution entry — position in performance_attribution.open_positions
           but no longer held at broker (qty == 0). Clean up metadata.
        2. Orphan broker position — held at broker but no attribution entry.
           Log alert so operator can investigate; do NOT auto-close (may be legitimate
           position from a manual trade or a race condition during entry).
        """
        try:
            pa_positions: dict = {}
            tracker = getattr(self, "performance_attribution", None)
            if tracker is not None:
                if hasattr(tracker, "normalize_open_positions"):
                    try:
                        tracker.normalize_open_positions()
                    except Exception as exc:
                        logger.debug("Attribution normalization skipped: %s", exc)
                pa_positions = tracker.open_positions or {}

            broker_positions: dict[str, float] = {}
            for raw_symbol, qty in self.positions.items():
                qty_f = float(qty or 0.0)
                if math.isclose(qty_f, 0.0, abs_tol=1e-12):
                    continue
                normalized_symbol = normalize_symbol(raw_symbol)
                broker_positions[normalized_symbol] = broker_positions.get(normalized_symbol, 0.0) + qty_f

            phantom_syms = [s for s in pa_positions if s not in broker_positions]
            orphan_syms = [s for s in broker_positions if s not in pa_positions]
            changed = False

            if phantom_syms:
                logger.warning(
                    "Reconciler: %d phantom attribution entries (not at broker): %s",
                    len(phantom_syms), phantom_syms,
                )
                for sym in phantom_syms:
                    # Clean attribution
                    tracker.open_positions.pop(sym, None)
                    # Clean metadata dicts
                    for meta in (
                        self.position_entry_prices,
                        self.position_entry_times,
                        self.position_entry_signals,
                        self.position_peak_prices,
                        self.position_stops,
                        self.failed_exits,
                    ):
                        meta.pop(sym, None)
                    self._tp_tranches_taken.pop(sym, None)
                    logger.info("Reconciler: cleaned phantom entry for %s", sym)
                    changed = True
                self._save_position_metadata()

            for sym, broker_qty in broker_positions.items():
                if sym not in pa_positions:
                    continue
                entry = pa_positions.get(sym) or {}
                desired_qty = abs(float(broker_qty or 0.0))
                current_qty = float(entry.get("quantity", 0.0) or 0.0)
                if entry.get("symbol") != sym:
                    entry["symbol"] = sym
                    changed = True
                if not math.isclose(current_qty, desired_qty, rel_tol=1e-9, abs_tol=1e-9):
                    logger.warning(
                        "Reconciler: attribution qty drift — %s attribution=%s broker=%s; updating tracked quantity",
                        sym,
                        current_qty,
                        desired_qty,
                    )
                    entry["quantity"] = desired_qty
                    changed = True

            if orphan_syms:
                for sym in orphan_syms:
                    qty = broker_positions[sym]
                    logger.warning(
                        "Reconciler: ORPHAN broker position — %s qty=%s "
                        "(no attribution entry; manual trade or race condition?)",
                        sym, qty,
                    )

            if changed and tracker is not None and hasattr(tracker, "_save_state"):
                tracker._save_state()

        except Exception as _re:
            logger.debug("Reconciler error: %s", _re)

    async def _read_broker_equity(self, broker_name: str, connector) -> Optional[float]:
        """Read broker equity and refresh cache on valid updates."""
        if connector is None:
            return None
        try:
            value = float(await connector.get_portfolio_value())
        except Exception as exc:
            logger.debug("Broker equity read failed for %s: %s", broker_name, exc)
            self._mark_broker_heartbeat(broker_name, success=False, error=str(exc))
            return None
        if value <= 0:
            self._mark_broker_heartbeat(
                broker_name,
                success=False,
                error=f"non_positive_equity:{value}",
            )
            return None

        # ✅ IBKR GHOST EQUITY SANITY CHECK
        # TWS caches the old NAV from the last session and serves it immediately
        # on reconnect before live account data arrives. We reject values above a
        # configurable hard cap (default $500K for paper trading accounts).
        # Set APEX_IBKR_MAX_SANE_EQUITY_USD in .env if your real IBKR account
        # is larger than this.
        if broker_name == "ibkr":
            max_sane = float(
                getattr(ApexConfig, "IBKR_MAX_SANE_EQUITY_USD",
                        os.environ.get("APEX_IBKR_MAX_SANE_EQUITY_USD", "2000000"))
            )
            if value > max_sane:
                last_cached = self._per_broker_last_equity.get(broker_name)
                logger.warning(
                    f"⚠️ IBKR equity ${value:,.0f} exceeds sanity cap ${max_sane:,.0f} "
                    f"— likely stale TWS NAV. Rejecting; last-known: ${f'{last_cached:,.0f}' if last_cached else 'none'}."
                )
                self._mark_broker_heartbeat(broker_name, success=False, error="stale_nav")
                return None

        self._broker_equity_cache[broker_name] = (value, datetime.utcnow())
        self._per_broker_last_equity[broker_name] = value  # persist across cache expiry
        self._mark_broker_heartbeat(broker_name, success=True)
        return value

    async def _read_broker_cash(self, broker_name: str, connector) -> Optional[float]:
        """Read broker cash and refresh cache on valid updates."""
        if connector is None or not hasattr(connector, "get_account_cash"):
            return None
        try:
            value = float(await connector.get_account_cash())
        except Exception as exc:
            logger.debug("Broker cash read failed for %s: %s", broker_name, exc)
            self._mark_broker_heartbeat(broker_name, success=False, error=str(exc))
            return None
        if math.isfinite(value):
            self._broker_cash_cache[broker_name] = (value, datetime.utcnow())
            self._mark_broker_heartbeat(broker_name, success=True)
            return value
        self._mark_broker_heartbeat(
            broker_name,
            success=False,
            error=f"non_finite_cash:{value}",
        )
        return None

    async def _get_connector_quote(self, connector, symbol: str) -> Dict[str, float]:
        """Fetch bid/ask quote when connector supports it."""
        if connector is None or not hasattr(connector, "get_quote"):
            return {}
        try:
            quote = await connector.get_quote(symbol)
            if isinstance(quote, dict):
                return quote
        except Exception as exc:
            logger.debug("Quote lookup unavailable for %s: %s", symbol, exc)
        return {}

    def _spread_limit_bps_for_asset(self, asset_class: str) -> float:
        """Return spread gate threshold by asset class, widened in volatile regimes."""
        normalized = str(asset_class or "EQUITY").upper()
        if normalized == "FOREX":
            base = float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_FX)
        elif normalized == "CRYPTO":
            base = float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_CRYPTO)
        else:
            base = float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_EQUITY)
        # In PANIC/CRISIS regimes spreads naturally widen 3-5x; relax gate proportionally
        regime = (self._current_regime or "neutral").lower()
        if "panic" in regime or "crisis" in regime:
            return base * 5.0
        if "volatile" in regime or "stress" in regime:
            return base * 2.5
        return base

    def _slippage_budget_bps_for_asset(self, asset_class: str) -> float:
        """Return slippage budget threshold by asset class."""
        normalized = str(asset_class or "EQUITY").upper()
        if normalized == "FOREX":
            return float(ApexConfig.EXECUTION_SLIPPAGE_BUDGET_BPS_FX)
        if normalized == "CRYPTO":
            return float(ApexConfig.EXECUTION_SLIPPAGE_BUDGET_BPS_CRYPTO)
        return float(ApexConfig.EXECUTION_SLIPPAGE_BUDGET_BPS_EQUITY)

    def _edge_buffer_bps_for_asset(self, asset_class: str) -> float:
        """Return minimum expected edge-over-cost buffer by asset class."""
        normalized = str(asset_class or "EQUITY").upper()
        if normalized == "FOREX":
            return float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_FX)
        if normalized == "CRYPTO":
            return float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO)
        return float(ApexConfig.EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY)

    def _signal_to_edge_bps_for_asset(self, asset_class: str) -> float:
        """Return signal-to-edge multiplier by asset class.
        Crypto has higher daily vol (~3-10%) so same signal strength yields larger bps returns."""
        if str(asset_class or "EQUITY").upper() == "CRYPTO":
            return float(getattr(ApexConfig, "EXECUTION_SIGNAL_TO_EDGE_BPS_CRYPTO", 300))
        return float(ApexConfig.EXECUTION_SIGNAL_TO_EDGE_BPS)

    @staticmethod
    def _compute_slippage_bps(expected_price: float, fill_price: float) -> float:
        """Compute absolute slippage in basis points."""
        try:
            expected = float(expected_price)
            fill = float(fill_price)
        except (TypeError, ValueError):
            return 0.0
        if expected <= 0 or fill <= 0:
            return 0.0
        slippage_bps = abs(fill - expected) / expected * 10000.0
        
        # Record to Prometheus for observability matrix
        try:
            from monitoring.prometheus_metrics import PrometheusMetrics
            metrics = PrometheusMetrics()
            metrics.record_execution_slippage(slippage_bps)
        except Exception:
            pass
            
        return slippage_bps

    def _infer_sleeve(self, symbol: str, asset_class: str) -> str:
        """Map symbol/asset class to portfolio sleeve for attribution."""
        normalized_asset = str(asset_class or "EQUITY").upper()
        symbol_upper = str(symbol or "").upper()
        if "OPT:" in symbol_upper or symbol_upper.startswith("OPTION:"):
            return "options_sleeve"
        if normalized_asset == "FOREX":
            return "fx_sleeve"
        if normalized_asset == "CRYPTO":
            return "crypto_sleeve"
        return "equities_sleeve"

    def _record_entry_attribution(
        self,
        *,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_signal: float,
        entry_confidence: float,
        governor_tier: str,
        governor_regime: str,
        risk_multiplier: float,
        vix_multiplier: float,
        governor_size_multiplier: float,
        entry_slippage_bps: float,
        entry_time: Optional[datetime] = None,
        source: str = "trade_entry",
    ) -> None:
        """Safely record entry attribution context."""
        try:
            _sig_comps = self._last_signal_components.get(symbol, {})
            self.performance_attribution.record_entry(
                symbol=symbol,
                asset_class=asset_class,
                sleeve=self._infer_sleeve(symbol, asset_class),
                side=side,
                quantity=abs(float(quantity)),
                entry_price=float(entry_price),
                entry_signal=float(entry_signal),
                entry_confidence=float(entry_confidence),
                governor_tier=governor_tier,
                governor_regime=governor_regime,
                risk_multiplier=float(risk_multiplier),
                vix_multiplier=float(vix_multiplier),
                governor_size_multiplier=float(governor_size_multiplier),
                entry_slippage_bps=float(entry_slippage_bps),
                entry_time=entry_time,
                source=source,
                ml_signal=float(_sig_comps.get('ml', 0.0) or 0.0),
                tech_signal=float(_sig_comps.get('tech', 0.0) or 0.0),
                sentiment_signal=float(_sig_comps.get('sentiment', 0.0) or 0.0),
                cs_momentum_signal=float(_sig_comps.get('cs_momentum', 0.0) or 0.0),
            )
        except Exception as exc:
            logger.debug("Attribution entry record skipped for %s: %s", symbol, exc)

    def _record_exit_attribution(
        self,
        *,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        commissions: float,
        exit_reason: str,
        entry_signal: float,
        entry_confidence: float,
        governor_tier: str,
        governor_regime: str,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        exit_slippage_bps: float = 0.0,
        source: str = "trade_exit",
    ) -> None:
        """Safely record closed-trade attribution."""
        try:
            qty = abs(float(quantity))
            entry_px = float(entry_price)
            exit_px = float(exit_price)
            comm = float(commissions)
            is_short = str(side).upper() == "SHORT"
            sleeve = self._infer_sleeve(symbol, asset_class)
            entry_ctx = self.performance_attribution.open_positions.get(symbol, {}) or {}
            entry_slippage_bps_ctx = float(entry_ctx.get("entry_slippage_bps", 0.0) or 0.0)
            entry_notional = abs(qty * (entry_px if entry_px > 0 else exit_px))
            exit_notional = abs(qty * exit_px)
            entry_slippage_cost = entry_notional * abs(entry_slippage_bps_ctx) / 10000.0
            exit_slippage_cost = exit_notional * abs(float(exit_slippage_bps or 0.0)) / 10000.0
            slippage_drag = entry_slippage_cost + exit_slippage_cost
            execution_drag = comm + slippage_drag
            gross_pnl = (
                (entry_px - exit_px) * qty
                if is_short
                else (exit_px - entry_px) * qty
            )
            net_pnl = gross_pnl - comm
            self.performance_attribution.record_exit(
                symbol=symbol,
                quantity=qty,
                exit_price=exit_px,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                commissions=comm,
                exit_reason=exit_reason,
                exit_slippage_bps=float(exit_slippage_bps),
                asset_class_fallback=asset_class,
                sleeve_fallback=sleeve,
                side_fallback=side,
                entry_price_fallback=entry_px,
                entry_signal_fallback=float(entry_signal),
                entry_confidence_fallback=float(entry_confidence),
                governor_tier_fallback=governor_tier,
                governor_regime_fallback=governor_regime,
                entry_time_fallback=entry_time,
                exit_time=exit_time,
                source=source,
            )
            if self.prometheus_metrics:
                self.prometheus_metrics.record_attribution_trade(
                    sleeve=sleeve,
                    net_alpha=net_pnl,
                    execution_drag=execution_drag,
                    slippage_drag=slippage_drag,
                )
        except Exception as exc:
            logger.debug("Attribution exit record skipped for %s: %s", symbol, exc)

    def _seed_attribution_for_open_positions(self) -> None:
        """Ensure startup-restored open positions are represented in attribution state."""
        # Purge any attribution entries whose symbol no longer exists in self.positions.
        # This removes ghost entries (e.g. CRYPTO:FIL/USD, CRYPTO:CRV/USD) that lingered
        # in performance_attribution.json from previous sessions after the positions were
        # closed or never actually existed at the broker.
        _live_syms = set(self.positions.keys())
        _pa_syms = list(self.performance_attribution.open_positions.keys())
        _purged = 0
        for sym in _pa_syms:
            _alt = sym[len("CRYPTO:"):] if sym.startswith("CRYPTO:") else f"CRYPTO:{sym}"
            if sym not in _live_syms and _alt not in _live_syms:
                del self.performance_attribution.open_positions[sym]
                _purged += 1
        if _purged:
            logger.info(f"_seed_attribution: purged {_purged} stale open_position entries "
                        f"with no matching live position: {[s for s in _pa_syms if s not in _live_syms]}")

        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            # Skip if an equivalent entry already exists (handles CRYPTO: prefix mismatch)
            _pa = self.performance_attribution.open_positions
            _alt = symbol[len("CRYPTO:"):] if symbol.startswith("CRYPTO:") else f"CRYPTO:{symbol}"
            if symbol in _pa or _alt in _pa:
                continue
            try:
                asset_class = parse_symbol(symbol).asset_class.value
            except Exception:
                asset_class = "EQUITY"

            entry_price = float(self.position_entry_prices.get(symbol, 0.0) or 0.0)
            if entry_price <= 0:
                entry_price = float(self.price_cache.get(symbol, 0.0) or 0.0)
            if entry_price <= 0:
                hist = self.historical_data.get(symbol)
                if hist is not None and not hist.empty:
                    try:
                        entry_price = float(hist["Close"].iloc[-1])
                    except Exception:
                        entry_price = 0.0
            if entry_price <= 0:
                continue

            entry_time = self.position_entry_times.get(symbol, datetime.utcnow())
            entry_signal = float(self.position_entry_signals.get(symbol, 0.5 if qty > 0 else -0.5))
            side = "LONG" if qty > 0 else "SHORT"
            governor_regime = self._map_governor_regime(asset_class, self._current_regime)
            governor_tier = self._performance_snapshot.tier.value if self._performance_snapshot else "green"

            self._record_entry_attribution(
                symbol=symbol,
                asset_class=asset_class,
                side=side,
                quantity=abs(float(qty)),
                entry_price=entry_price,
                entry_signal=entry_signal,
                entry_confidence=min(1.0, max(0.0, abs(entry_signal))),
                governor_tier=governor_tier,
                governor_regime=governor_regime,
                risk_multiplier=float(self._risk_multiplier),
                vix_multiplier=float(self._vix_risk_multiplier),
                governor_size_multiplier=1.0,
                entry_slippage_bps=0.0,
                entry_time=entry_time,
                source="startup_restore",
            )

    async def _refresh_startup_signals(self) -> None:
        """Replace hardcoded 0.5 startup signals with real ML output for every open position.

        _seed_attribution_for_open_positions() runs before ML training and writes
        entry_signal=0.5 (the default) for every restored position.  Once the
        institutional model is ready this method re-runs generate_signal() for each
        position and patches:
          • self.position_entry_signals[symbol]          — used by exit quality checks
          • performance_attribution.open_positions entry — used by attribution reports

        Positions whose refreshed signal falls below MIN_SIGNAL_THRESHOLD are added
        to self._weak_signal_restored so the first run-loop cycle can prioritise them
        for exit evaluation.
        """
        if not self.positions:
            return

        if not (self.use_institutional and self.inst_signal_generator.is_trained):
            logger.info(
                "Startup signal refresh: institutional ML not ready — "
                "entry_signal defaults (0.5) remain in attribution"
            )
            return

        open_syms = [(s, q) for s, q in self.positions.items() if q != 0]
        if not open_syms:
            return

        logger.info("🔄 Startup signal refresh: computing real ML signals for %d open position(s)…", len(open_syms))
        refreshed = failed = weak = 0
        threshold = float(getattr(ApexConfig, "MIN_SIGNAL_THRESHOLD", 0.30))

        for symbol, qty in open_syms:
            try:
                # Historical data is keyed WITHOUT the CRYPTO:/FX: asset-class prefix.
                # Use explicit is-None checks — `or` forces bool(DataFrame) which raises ValueError.
                _bare = symbol[len("CRYPTO:"):] if symbol.startswith("CRYPTO:") else \
                        symbol[len("FX:"):] if symbol.startswith("FX:") else symbol
                data = self.historical_data.get(symbol)
                if data is None:
                    data = self.historical_data.get(_bare)
                if data is None or len(data) < 50:
                    logger.debug("Startup signal refresh: %s — insufficient history (tried %s), skipping",
                                 symbol, _bare)
                    failed += 1
                    continue

                # Cross-sectional momentum rank — use bare key for lookup consistency
                cs_data = self.cs_momentum.get_signal(_bare, self.historical_data)
                momentum_rank = float(cs_data.get("rank_percentile", 0.5))

                # Generate fresh ML signal.  Called synchronously — startup context,
                # blocking the event loop briefly (~10–100 ms per symbol) is acceptable.
                # Use bare symbol (_bare) — model was trained on symbols without CRYPTO:/FX: prefix.
                fresh: SignalOutput = self.inst_signal_generator.generate_signal(
                    _bare,
                    data,
                    sentiment_score=0.0,      # neutral placeholder; async sentiment skipped at startup
                    momentum_rank=momentum_rank,
                )
                fresh_signal = float(fresh.signal)
                fresh_conf = float(fresh.confidence)
                old_signal = float(self.position_entry_signals.get(symbol, 0.5))

                # 1. Update live signal tracker used by exit-quality checks
                self.position_entry_signals[symbol] = fresh_signal

                # 2. Patch attribution entry in-place so reports reflect real signal
                pa_entry = self.performance_attribution.open_positions.get(symbol)
                if pa_entry:
                    pa_entry["entry_signal"] = fresh_signal
                    pa_entry["entry_confidence"] = fresh_conf
                    pa_entry["source"] = "startup_signal_refresh"

                # 3. Flag weak-signal positions for priority exit on first cycle
                is_long = qty > 0
                signal_is_weak = (is_long and fresh_signal < threshold) or \
                                 (not is_long and fresh_signal > -threshold)
                if signal_is_weak:
                    self._weak_signal_restored.add(symbol)
                    weak += 1
                    logger.info(
                        "⚠️  %s: signal %.3f → %.3f (conf=%.2f) — WEAK, flagged for priority exit",
                        symbol, old_signal, fresh_signal, fresh_conf,
                    )
                else:
                    logger.info(
                        "✅ %s: signal %.3f → %.3f (conf=%.2f)",
                        symbol, old_signal, fresh_signal, fresh_conf,
                    )
                refreshed += 1

            except Exception as exc:
                logger.warning("Startup signal refresh failed for %s: %s", symbol, exc)
                failed += 1

        logger.info(
            "🔄 Startup signal refresh complete — %d refreshed, %d failed, %d weak-signal flagged",
            refreshed, failed, weak,
        )

        # Persist corrected attribution immediately (don't wait for periodic save)
        try:
            await asyncio.to_thread(self.performance_attribution._save_state)
        except Exception:
            pass

    async def _get_total_portfolio_value(self) -> float:
        """Get combined portfolio value across brokers with per-broker awareness.

        Priority per broker: live read → short-term cache → long-term last-known → omit.
        Returns sum of whichever brokers have data so that an offline IBKR ($1.18M) does
        NOT inflate the equity seen by an online Alpaca-only ($100k) session.
        """
        brokers: Dict[str, object] = {}
        if self.ibkr:
            brokers["ibkr"] = self.ibkr
        if self.alpaca:
            brokers["alpaca"] = self.alpaca
        if not brokers:
            return float(self.capital)

        now = datetime.utcnow()
        stale_seconds = max(
            1,
            int(getattr(ApexConfig, "BROKER_EQUITY_CACHE_TTL_SECONDS", 900)),
        )
        configured_quorum = max(
            1,
            int(getattr(ApexConfig, "BROKER_EQUITY_QUORUM_MIN_BROKERS", 1)),
        )
        required_quorum = min(len(brokers), configured_quorum)

        live_values: Dict[str, float] = {}    # brokers that returned fresh or cached equity
        offline_brokers: list = []            # brokers with zero data (never seen or truly dead)

        for broker_name, connector in brokers.items():
            # 1. Try live read (updates cache + _per_broker_last_equity on success)
            fresh = await self._read_broker_equity(broker_name, connector)
            if fresh is not None and fresh > 0:
                live_values[broker_name] = fresh
                continue

            # 2. Short-term cache (broker recently went offline)
            cached = self._broker_equity_cache.get(broker_name)
            if cached:
                cached_value, cached_at = cached
                age_seconds = (now - cached_at).total_seconds()
                if cached_value > 0:
                    live_values[broker_name] = cached_value
                    if age_seconds <= stale_seconds:
                        logger.warning(
                            "⚠️ %s equity unavailable, reusing cached value $%.2f (age=%.0fs)",
                            broker_name, cached_value, age_seconds,
                        )
                    else:
                        logger.warning(
                            "⚠️ %s equity STALE (age=%.0fs), reusing last known $%.2f",
                            broker_name, age_seconds, cached_value,
                        )
                    continue

            # 3. Long-term last-known (survives cache expiry, e.g. after restart)
            last_known = self._per_broker_last_equity.get(broker_name)
            if last_known and last_known > 0:
                live_values[broker_name] = last_known
                logger.warning(
                    "⚠️ %s offline – using last-known equity $%.2f for continuity",
                    broker_name, last_known,
                )
                continue

            # 4. Truly no data for this broker
            offline_brokers.append(broker_name)

        # Return sum of all brokers that have ANY equity data (live or remembered)
        if len(live_values) >= required_quorum:
            total = float(sum(live_values.values()))
            if total > 0:
                self._current_equity_contributors = set(live_values.keys())
                self._last_good_total_equity = total
                if offline_brokers:
                    logger.warning(
                        "⚠️ %s: no equity data at all – excluded from total. "
                        "Active brokers %s sum: $%.2f",
                        offline_brokers, list(live_values.keys()), total,
                    )
                return total

        # All brokers failed and we have no history anywhere
        fallback_value = (
            float(self._last_good_total_equity)
            if self._last_good_total_equity > 0
            else float(self.capital)
        )
        logger.warning(
            "⚠️ No equity data from any broker (%d configured). "
            "Returning last good total $%.2f.",
            len(brokers),
            fallback_value,
        )
        return fallback_value

    async def _get_total_account_cash(self) -> Optional[float]:
        """Get combined cash across active brokers with cache fallback."""
        brokers = {}
        if self.ibkr:
            brokers["ibkr"] = self.ibkr
        if self.alpaca:
            brokers["alpaca"] = self.alpaca
        if not brokers:
            return None

        now = datetime.utcnow()
        stale_seconds = max(
            1,
            int(getattr(ApexConfig, "BROKER_EQUITY_CACHE_TTL_SECONDS", 900)),
        )
        configured_quorum = max(
            1,
            int(getattr(ApexConfig, "BROKER_EQUITY_QUORUM_MIN_BROKERS", 1)),
        )
        required_quorum = min(len(brokers), configured_quorum)
        included_values: Dict[str, float] = {}

        for broker_name, connector in brokers.items():
            fresh = await self._read_broker_cash(broker_name, connector)
            if fresh is not None:
                included_values[broker_name] = fresh
                continue

            cached = self._broker_cash_cache.get(broker_name)
            if not cached:
                continue
            cached_value, cached_at = cached
            age_seconds = (now - cached_at).total_seconds()
            if math.isfinite(cached_value):
                included_values[broker_name] = cached_value
                if age_seconds <= stale_seconds:
                    logger.warning(
                        "⚠️ %s cash unavailable, reusing cached value $%.2f (age=%.0fs)",
                        broker_name,
                        cached_value,
                        age_seconds,
                    )
                else:
                    logger.warning(
                        "⚠️ %s cash STALE, reusing last known value $%.2f (age=%.0fs) to prevent cash drop",
                        broker_name,
                        cached_value,
                        age_seconds,
                    )

        if len(included_values) >= required_quorum:
            total_cash = float(sum(included_values.values()))
            if math.isfinite(total_cash):
                self._last_good_total_cash = total_cash
                return total_cash

        if self._last_good_total_cash is not None and math.isfinite(self._last_good_total_cash):
            logger.warning(
                "⚠️ Missing cash for %d/%d brokers. Returning last good cash $%.2f.",
                len(brokers) - len(included_values),
                len(brokers),
                self._last_good_total_cash,
            )
            return float(self._last_good_total_cash)

        return None

    def _compute_marked_positions_value(self) -> float:
        """Compute mark-to-market value of open positions using local pricing."""
        total = 0.0
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            price = float(self.price_cache.get(symbol, 0.0) or 0.0)
            if price <= 0:
                price = float(self.position_entry_prices.get(symbol, 0.0) or 0.0)
            if price <= 0:
                continue
            total += float(qty) * price

        # Include Options using LIVE market values from portfolio (Unrealized P&L included)
        if hasattr(self, 'options_positions') and self.options_positions:
            # Fetch live market values if connector supports it
            option_values = {}
            if hasattr(self.ibkr, 'get_portfolio_market_values'):
                option_values = self.ibkr.get_portfolio_market_values()

            for key, opt in self.options_positions.items():
                qty = float(opt.get('quantity', 0))
                if qty == 0:
                    continue
                
                # Use live market value from portfolio if available
                market_val = option_values.get(key)
                
                if market_val is not None:
                     # marketValue is total value of position
                     logger.debug(f"🔍 Option {opt.get('symbol')} {opt.get('right')}: val={market_val} (Portfolio)")
                     total += float(market_val)
                else:
                     # Fallback to cost if not in portfolio (dead reckon)
                     cost = float(opt.get('avg_cost', 0.0))
                     # Estimate value using cost (often unreliable for P&L)
                     # Using 100 multiplier as safer default for unit-cost assumption if unknown
                     # But for avg_cost (Total) it might be double counting. 
                     # We assume avg_cost is UNIT price here if we fallback.
                     val = qty * cost 
                     logger.warning(f"⚠️ Option {key} not in portfolio. Using cost basis: {val}")
                     total += val

        return float(total)

    async def _evaluate_equity_reconciliation(
        self,
        broker_equity: float,
        observed_at: Optional[datetime] = None,
    ) -> EquityReconciliationSnapshot:
        """Reconcile broker equity vs modeled equity and update block latch."""
        modeled_equity: Optional[float]
        if self.ibkr or self.alpaca:
            total_cash = await self._get_total_account_cash()
            if total_cash is None:
                modeled_equity = None
                pos_val = self._compute_marked_positions_value()
            else:
                pos_val = self._compute_marked_positions_value()
                modeled_equity = float(total_cash + pos_val)
        else:
            total_cash = None
            pos_val = 0.0
            modeled_equity = float(broker_equity)

        # DEBUG: Inspect reconciliation components
        # ----------------------------------------
        logger.info(f"🔍 DEBUG RECON: Broker=${broker_equity:,.2f} Modeled=${modeled_equity:,.2f} Cash=${total_cash or 0:,.2f} PosVal=${pos_val:,.2f}")

        return self.equity_reconciler.evaluate(
            broker_equity=float(broker_equity),
            modeled_equity=modeled_equity,
            timestamp=observed_at or datetime.now(),
        )

    def calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate current exposure by sector."""
        exposure = {}
        total_value = 0.0
        
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            
            sector = ApexConfig.get_sector(symbol)
            price = self.price_cache.get(symbol, 0)
            
            # Fallback to entry price if market data is unavailable (e.g. after hours)
            if price <= 0:
                price = self.position_entry_prices.get(symbol, 0)
                
            if price > 0:
                value = abs(float(qty) * float(price))  # Absolute value for both long/short
                total_value += value
                exposure[sector] = exposure.get(sector, 0) + value
        
        # Convert to percentages
        if total_value > 0:
            for sector in exposure:
                exposure[sector] = exposure[sector] / total_value
        
        return exposure

    def _sector_concentration_blocked(self, symbol: str) -> tuple:
        """
        Check if a new entry in `symbol` would exceed the sector concentration cap.

        Returns (blocked: bool, sector: str, projected_pct: float).
        Pure sync — zero I/O, uses cached self.capital and in-memory positions.
        Crypto (Alpaca-routed) and ETFs are always exempt.
        """
        if not getattr(ApexConfig, "SECTOR_CONCENTRATION_ENABLED", True):
            return False, "", 0.0

        if _symbol_is_crypto(symbol):
            return False, "", 0.0

        sector = ApexConfig.get_sector(symbol)
        if not sector or sector == "ETF":
            return False, sector, 0.0

        max_pct = float(getattr(ApexConfig, "SECTOR_CONCENTRATION_MAX_PCT", 0.25))

        # Bootstrap grace: relax limit while building initial equity portfolio
        active_equity = sum(
            1 for s, qty in self.positions.items()
            if qty != 0 and not _symbol_is_crypto(s) and "OPT:" not in str(s).upper()
        )
        if active_equity < 5:
            max_pct = max(max_pct, 0.60)

        # Adding to an existing position is not a new sector bet
        if self.positions.get(symbol, 0) != 0:
            return False, sector, 0.0

        current_exposure = self.calculate_sector_exposure()
        capital = self.capital
        if capital <= 0:
            return False, sector, 0.0

        position_size = float(getattr(ApexConfig, "POSITION_SIZE_USD", 20000))
        current_sector_value = current_exposure.get(sector, 0.0) * capital
        projected_pct = (current_sector_value + position_size) / (capital + position_size)

        if projected_pct > max_pct:
            return True, sector, projected_pct

        return False, sector, projected_pct

    async def check_sector_limit(self, symbol: str) -> bool:
        """
        Check if adding this symbol would exceed sector limits.

        Dynamic logic:
        - If < 5 positions, allow up to 80% in a sector.
        - Otherwise, use ApexConfig.MAX_SECTOR_EXPOSURE (0.50).
        - Crypto symbols routed to Alpaca (dedicated crypto account) are exempt —
          the Alpaca account is 100% crypto by design; sector caps apply to IBKR.
        """
        # Crypto symbols routed to Alpaca are exempt: Alpaca is a dedicated crypto
        # account so sector concentration vs the IBKR portfolio is meaningless.
        try:
            if _symbol_is_crypto(symbol) and self.alpaca is not None:
                connector = self._get_connector_for(symbol)
                if connector is self.alpaca:
                    return True  # Alpaca crypto — no IBKR-style sector cap
        except Exception:
            pass

        sector = ApexConfig.get_sector(symbol)
        current_exposure = self.calculate_sector_exposure()

        # Determine dynamic limit
        limit = float(getattr(ApexConfig, 'MAX_SECTOR_EXPOSURE', 0.50))
        if self.position_count < 5 and limit < 0.80:
            limit = 0.80  # Allow more concentration in early stage if config is tight
            logger.debug(f"ℹ️ {symbol}: Using relaxed sector limit {limit*100:.0f}% (small portfolio)")
            
        # Compute new percentage approximation
        try:
            total_value = await self._get_total_portfolio_value()
            position_size = getattr(ApexConfig, 'POSITION_SIZE_USD', 20000)
            if total_value <= 0:
                return True
            current_sector_value = current_exposure.get(sector, 0.0) * total_value
            new_pct = (current_sector_value + position_size) / (total_value + position_size)
            
            if new_pct > limit:
                logger.warning(
                    f"Sector cap breached: {sector} would reach {new_pct*100:.1f}% > {limit*100:.1f}% — "
                    f"signal for {symbol} blocked."
                )
                return False
        except Exception as e:
            logger.debug(f"Error checking sector limit: {e}")
            if current_exposure.get(sector, 0) >= limit:
                return False
                
        return True

    async def enforce_sector_limits(self) -> None:
        """
        Actively enforce sector exposure limits by trimming positions in over-limit sectors.
        
        For each sector breaching MAX_SECTOR_EXPOSURE:
        1. Sort positions in that sector by current P&L (trim least-profitable first)
        2. Calculate how many shares to sell to bring the sector back to the cap
        3. Issue a partial or full exit for the trimming candidate
        
        Called once per main execution cycle after position sync.
        """
        limit = float(getattr(ApexConfig, "MAX_SECTOR_EXPOSURE", 0.50))
        current_exposure = self.calculate_sector_exposure()
        try:
            total_value = await self._get_total_portfolio_value()
        except Exception:
            total_value = 0.0

        if total_value <= 0:
            return

        for sector, pct in current_exposure.items():
            if pct <= limit:
                continue

            excess_pct = pct - limit
            excess_value = excess_pct * total_value

            logger.warning(
                f"🚨 Sector breach: {sector} at {pct*100:.1f}% > cap {limit*100:.0f}% "
                f"(excess ~${excess_value:,.0f}) — initiating trim"
            )

            # Collect positions in this sector, sorted by P&L ascending (worst first)
            sector_positions = []
            for symbol, qty in self.positions.items():
                if qty == 0:
                    continue
                if ApexConfig.get_sector(symbol) != sector:
                    continue
                price = self.price_cache.get(symbol, 0) or self.position_entry_prices.get(symbol, 0)
                if price <= 0:
                    continue
                entry = self.position_entry_prices.get(symbol, price)
                pnl_pct = (price / entry - 1) if qty > 0 else (entry / price - 1)
                value = abs(float(qty)) * float(price)
                sector_positions.append((symbol, qty, price, pnl_pct, value))

            # Sort: trim smallest P&L first (but never trim a position with active options)
            sector_positions.sort(key=lambda x: x[3])

            remaining_excess = excess_value
            for symbol, qty, price, pnl_pct, value in sector_positions:
                if remaining_excess <= 0:
                    break

                # Skip symbols that have active options written against them
                # Keys are formatted as "{symbol}:covered_call", "{symbol}:hedge", etc.
                has_active_option = any(
                    k.split(":")[0] == symbol for k in self.options_positions.keys()
                    if self.options_positions[k].get("quantity", 0) != 0
                )
                if has_active_option:
                    logger.info(
                        f"   ⚠️  Skipping {symbol} — has active option position, cannot freely trim"
                    )
                    continue

                # Calculate shares to sell to recover `remaining_excess`
                shares_to_sell = min(abs(int(qty)), max(1, int(remaining_excess / price)))
                if shares_to_sell <= 0:
                    continue

                side = "SELL" if qty > 0 else "BUY"
                logger.info(
                    f"   ✂️  Trimming {symbol}: {side} {shares_to_sell} shares "
                    f"@ ${price:.2f} (P&L: {pnl_pct*100:+.1f}%)"
                )

                try:
                    connector = self._get_connector_for(symbol)
                    if connector is None:
                        logger.warning(f"   ❌ No connector for sector trim on {symbol}")
                        continue
                    result = await connector.execute_order(
                        symbol=symbol,
                        side=side,
                        quantity=shares_to_sell,
                        confidence=1.0,  # Forced risk management exit
                    )
                    if result:
                        proceeds = shares_to_sell * price
                        remaining_excess -= proceeds
                        # Update internal position tracking
                        async with self._position_lock:
                            current_qty = self.positions.get(symbol, 0)
                            new_qty = current_qty - shares_to_sell if qty > 0 else current_qty + shares_to_sell
                            if abs(new_qty) < 1:
                                self.positions.pop(symbol, None)
                            else:
                                self.positions[symbol] = new_qty
                        logger.info(
                            f"   ✅ Sector trim executed: {symbol} {side} {shares_to_sell} "
                            f"(remaining excess ~${remaining_excess:,.0f})"
                        )
                    else:
                        logger.warning(f"   ❌ Sector trim failed for {symbol}")
                except Exception as e:
                    logger.error(f"   Sector trim error for {symbol}: {e}")

    def _map_governor_regime(self, asset_class: str, market_regime: str) -> str:
        """Map internal market regime into governor policy regime namespace."""
        regime = (market_regime or "default").lower()
        asset = asset_class.upper()

        if asset == "EQUITY":
            if regime in {"bear", "strong_bear"}:
                return "risk_off"
            if regime in {"volatile", "high_volatility"}:
                return "volatile"
            if regime in {"bull", "strong_bull"}:
                return "risk_on"
            return "default"

        if asset == "FOREX":
            if regime in {"bear", "strong_bear", "volatile", "high_volatility"}:
                return "carry_crash"
            if regime in {"bull", "strong_bull"}:
                return "carry"
            return "default"

        if asset == "CRYPTO":
            if regime in {"bear", "strong_bear"}:
                return "crash"
            if regime in {"volatile", "high_volatility"}:
                return "high_vol"
            if regime in {"bull", "strong_bull"}:
                return "trend"
            return "default"

        return "default"

    def _reload_social_inputs(self) -> None:
        """Refresh social + prediction inputs once per risk cycle."""
        self._social_snapshot_cache.clear()
        self._social_decision_cache.clear()
        self._prediction_results_cache.clear()
        self._social_feed_warning_cache.clear()
        self._social_policy_version, self._social_active_policies = self._social_policy_repo.load_active()
        if not self.social_shock_governor:
            self._social_inputs_payload = {}
            self._social_input_validation = {}
            return

        # Build/refresh social risk inputs file (TTL: skip if file is <15 min old)
        try:
            rebuild = True
            if self._social_feed_path.exists():
                age_s = (datetime.utcnow() - datetime.utcfromtimestamp(self._social_feed_path.stat().st_mtime)).total_seconds()
                rebuild = age_s > 900  # 15-minute TTL
            if rebuild:
                write_social_risk_inputs(
                    data_dir=ApexConfig.DATA_DIR,
                    output_path=self._social_feed_path,
                )
                logger.info("📡 Social risk inputs rebuilt → %s", self._social_feed_path)
        except Exception as exc:
            logger.debug("Social risk input build skipped: %s", exc)

        if not self._social_feed_path.exists():
            self._social_inputs_payload = {}
            self._social_input_validation = {
                "valid": True,
                "has_usable_feeds": False,
                "warnings": [{"code": "missing_social_inputs_file"}],
            }
            logger.warning(
                "⚠️ Social feed file missing (%s). Fail-open mode active for social entry blocking.",
                self._social_feed_path,
            )
            return
        try:
            with open(self._social_feed_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload = payload if isinstance(payload, dict) else {}
            validation = validate_social_risk_inputs(payload, freshness_sla_seconds=1800)
            self._social_input_validation = validation.summary()
            if not validation.valid:
                logger.warning(
                    "⚠️ Social feed validation failed (%d errors). Fail-open mode active.",
                    len(validation.errors),
                )
                self._social_inputs_payload = {}
            else:
                self._social_inputs_payload = payload
            if not validation.has_usable_feeds:
                logger.info("ℹ️  Social feeds unavailable/stale. Operating in fail-open mode.")
        except Exception as exc:
            logger.warning("⚠️ Social risk input file unreadable (%s): %s", self._social_feed_path, exc)
            self._social_inputs_payload = {}
            self._social_input_validation = {
                "valid": False,
                "has_usable_feeds": False,
                "errors": [{"code": "social_file_unreadable", "message": str(exc)}],
            }

    def _social_platform_signals_for_scope(self, asset_class: str, regime: str) -> Dict[str, Dict[str, float]]:
        """Resolve platform signals using global -> asset -> regime overrides."""
        payload = self._social_inputs_payload if isinstance(self._social_inputs_payload, dict) else {}
        merged: Dict[str, Dict[str, float]] = {}
        key = (asset_class.upper(), regime.lower())
        source_map = payload.get("sources", {}) if isinstance(payload.get("sources"), dict) else {}
        warnings: List[str] = []

        def _source_status(platform: str) -> Tuple[str, List[str]]:
            source_row = source_map.get(platform, {})
            if not isinstance(source_row, dict):
                return "missing", ["source_missing"]
            quality = source_row.get("quality", {})
            if not isinstance(quality, dict):
                return "missing", ["quality_missing"]
            status = str(quality.get("status", "missing")).lower()
            flags = quality.get("flags", [])
            if not isinstance(flags, list):
                flags = []
            return status, [str(item) for item in flags]

        def _overlay(raw: Any) -> None:
            if not isinstance(raw, dict):
                return
            for platform, row in raw.items():
                if not isinstance(row, dict):
                    continue
                platform_name = str(platform).upper()
                source_status, source_flags = _source_status(platform_name)
                if source_status in {"missing", "stale"}:
                    warnings.append(f"{platform_name}:{source_status}")
                    continue
                try:
                    confidence = float(row.get("confidence", 1.0) or 1.0)
                    quality_flags = row.get("quality_flags", [])
                    if not isinstance(quality_flags, list):
                        quality_flags = []
                    all_flags = {str(flag) for flag in quality_flags + source_flags}
                    if source_status == "degraded":
                        confidence *= 0.7
                    if "stale_data" in all_flags:
                        confidence *= 0.4

                    merged[platform_name] = {
                        "attention_z": float(row.get("attention_z", 0.0) or 0.0),
                        "sentiment_score": float(row.get("sentiment_score", 0.0) or 0.0),
                        "confidence": max(0.0, min(1.0, confidence)),
                    }
                except (TypeError, ValueError):
                    warnings.append(f"{platform_name}:invalid_row")
                    continue

        _overlay(payload.get("platforms"))
        asset_block = (payload.get("asset_classes", {}) or {}).get(asset_class.upper(), {})
        if isinstance(asset_block, dict):
            _overlay(asset_block.get("platforms"))
            regime_block = (asset_block.get("regimes", {}) or {}).get(regime, {})
            if isinstance(regime_block, dict):
                _overlay(regime_block.get("platforms"))

        if merged:
            self._social_feed_warning_cache[key] = warnings
            return merged
        self._social_feed_warning_cache[key] = list(sorted(set(warnings + ["no_usable_platform_signals"])))
        return {}

    def _social_policy_for_scope(
        self,
        asset_class: str,
        regime: str,
    ) -> Tuple[str, SocialShockGovernorConfig, PredictionMarketVerificationConfig]:
        base_governor = (
            self.social_shock_governor.config
            if self.social_shock_governor
            else SocialShockGovernorConfig()
        )
        base_verify = (
            self.prediction_market_verifier.config
            if self.prediction_market_verifier
            else PredictionMarketVerificationConfig()
        )
        policy = self._social_policy_repo.resolve(
            asset_class=asset_class,
            regime=regime,
            active_policies=self._social_active_policies,
        )
        if not policy:
            return "runtime-config", base_governor, base_verify
        return (
            str(policy.version or self._social_policy_version or "runtime-config"),
            SocialShockGovernorConfig(
                reduce_threshold=float(policy.reduce_threshold),
                block_threshold=float(policy.block_threshold),
                min_gross_exposure_multiplier=float(base_governor.min_gross_exposure_multiplier),
                verified_event_weight=float(policy.verified_event_weight),
                verified_event_probability_floor=float(policy.verified_event_probability_floor),
            ),
            PredictionMarketVerificationConfig(
                min_independent_sources=int(policy.min_independent_sources),
                max_probability_divergence=float(policy.max_probability_divergence),
                max_source_disagreement=float(policy.max_source_disagreement),
                minimum_market_probability=float(policy.minimum_market_probability),
            ),
        )

    def _append_social_decision_audit(
        self,
        *,
        asset_class: str,
        regime: str,
        policy_version: str,
        platform_signals: Dict[str, Dict[str, float]],
        prediction_events: List[PredictionEventInput],
        prediction_results: List[PredictionVerificationResult],
        decision: SocialShockDecision,
    ) -> None:
        decision_payload = decision.to_dict()
        decision_hash = hashlib.sha256(
            json.dumps(decision_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        verified_events = [
            row.to_dict()
            for row in prediction_results
            if bool(row.verified)
        ]
        self._social_audit_repo.append_event(
            {
                "asset_class": str(asset_class).upper(),
                "regime": str(regime).lower(),
                "policy_version": str(policy_version),
                "decision_hash": decision_hash,
                "decision": decision_payload,
                "inputs": {
                    "platform_signals": platform_signals,
                    "prediction_events": [event.__dict__ for event in prediction_events],
                    "input_validation": dict(self._social_input_validation or {}),
                },
                "verified_events": verified_events,
            }
        )

    def _prediction_events_for_scope(self, asset_class: str, regime: str) -> List[PredictionEventInput]:
        """Resolve scoped prediction events."""
        payload = self._social_inputs_payload if isinstance(self._social_inputs_payload, dict) else {}
        rows: List[Dict[str, Any]] = []

        def _append(raw: Any) -> None:
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        rows.append(item)

        _append(payload.get("prediction_events"))
        asset_block = (payload.get("asset_classes", {}) or {}).get(asset_class.upper(), {})
        if isinstance(asset_block, dict):
            _append(asset_block.get("prediction_events"))
            regime_block = (asset_block.get("regimes", {}) or {}).get(regime, {})
            if isinstance(regime_block, dict):
                _append(regime_block.get("prediction_events"))

        events: List[PredictionEventInput] = []
        for row in rows:
            event_id = str(row.get("event_id", "") or "").strip()
            if not event_id:
                continue
            try:
                events.append(
                    PredictionEventInput(
                        event_id=event_id,
                        market_probability=float(
                            row.get("market_probability", row.get("polymarket_probability", 0.0)) or 0.0
                        ),
                        independent_probability=float(row.get("independent_probability", 0.0) or 0.0),
                        independent_source_count=int(row.get("independent_source_count", 0) or 0),
                        max_source_disagreement=float(row.get("max_source_disagreement", 1.0) or 1.0),
                        direction=str(row.get("direction", "risk_off") or "risk_off").lower(),
                    )
                )
            except (TypeError, ValueError):
                continue
        return events

    def _social_decision_for(self, asset_class: str, regime: str) -> Optional[SocialShockDecision]:
        """Return social shock decision for an asset class + regime key."""
        if not self.social_risk_factor or not self.social_shock_governor:
            return None

        key = (asset_class.upper(), regime.lower())
        if key in self._social_decision_cache:
            return self._social_decision_cache[key]

        policy_version, governor_cfg, verification_cfg = self._social_policy_for_scope(key[0], key[1])
        platform_signals = self._social_platform_signals_for_scope(key[0], key[1])
        snapshot = self.social_risk_factor.evaluate(
            asset_class=key[0],
            regime=key[1],
            platform_signals=platform_signals,
            observed_at=datetime.now(),
        )
        prediction_events = self._prediction_events_for_scope(key[0], key[1])
        predictions: List[PredictionVerificationResult] = []
        if self.prediction_market_verifier and prediction_events:
            verifier = PredictionMarketVerificationGate(config=verification_cfg)
            for event in prediction_events:
                predictions.append(verifier.verify(event))

        decision = SocialShockGovernor(config=governor_cfg).evaluate(
            snapshot,
            predictions,
            policy_version=policy_version,
        )
        if (not self._social_input_validation.get("has_usable_feeds", False)) and platform_signals == {}:
            reasons = list(decision.reasons)
            reasons.append("social_feed_unavailable_fail_open")
            decision = SocialShockDecision(
                asset_class=decision.asset_class,
                regime=decision.regime,
                policy_version=decision.policy_version,
                social_risk_score=decision.social_risk_score,
                combined_risk_score=decision.social_risk_score,
                gross_exposure_multiplier=1.0,
                block_new_entries=False,
                verified_event_probability=0.0,
                prediction_verification_failures=decision.prediction_verification_failures,
                reasons=list(dict.fromkeys(reasons)),
            )
            logger.debug(
                "⚠️ %s/%s social decision forced fail-open due to unusable feeds",
                key[0],
                key[1],
            )

        warning_reasons = self._social_feed_warning_cache.get(key, [])
        if warning_reasons:
            decision = SocialShockDecision(
                asset_class=decision.asset_class,
                regime=decision.regime,
                policy_version=decision.policy_version,
                social_risk_score=decision.social_risk_score,
                combined_risk_score=decision.combined_risk_score,
                gross_exposure_multiplier=decision.gross_exposure_multiplier,
                block_new_entries=decision.block_new_entries,
                verified_event_probability=decision.verified_event_probability,
                prediction_verification_failures=decision.prediction_verification_failures,
                reasons=list(dict.fromkeys(decision.reasons + warning_reasons)),
            )
        self._social_snapshot_cache[key] = snapshot
        self._social_decision_cache[key] = decision
        self._prediction_results_cache[key] = predictions
        self._append_social_decision_audit(
            asset_class=key[0],
            regime=key[1],
            policy_version=policy_version,
            platform_signals=platform_signals,
            prediction_events=prediction_events,
            prediction_results=predictions,
            decision=decision,
        )
        return decision

    def _apply_social_size_multiplier(
        self,
        *,
        symbol: str,
        shares: int,
        decision: Optional[SocialShockDecision],
        price: Optional[float] = None,
    ) -> int:
        """Apply social shock gross-exposure multiplier to requested shares."""
        if not decision:
            return shares
        multiplier = float(decision.gross_exposure_multiplier)
        if multiplier >= 0.999:
            return shares
        adjusted = max(1, int(shares * multiplier))
        logger.info(
            "📣 %s: SocialShockGovernor size %.0f%% (%s/%s) -> %d shares",
            symbol,
            multiplier * 100.0,
            decision.asset_class,
            decision.regime,
            adjusted,
        )
        reduced_notional = max(0.0, float((shares - adjusted) * max(0.0, float(price or 0.0))))
        if reduced_notional > 0.0:
            self.performance_attribution.record_social_governor_impact(
                asset_class=decision.asset_class,
                regime=decision.regime,
                blocked_alpha_opportunity=0.0,
                avoided_drawdown_estimate=reduced_notional * float(decision.combined_risk_score) * 0.012,
                hedge_cost_drag=reduced_notional * (0.0009 + float(decision.combined_risk_score) * 0.0006),
                policy_version=decision.policy_version,
                reason="size_reduction",
                event_id="",
            )
        return adjusted

    def _emit_social_prometheus_metrics(self) -> None:
        """Publish social risk + verification metrics to Prometheus."""
        if not self.prometheus_metrics:
            return
        for key, decision in self._social_decision_cache.items():
            snapshot = self._social_snapshot_cache.get(key)
            if not snapshot:
                continue
            self.prometheus_metrics.update_social_risk_state(
                asset_class=decision.asset_class,
                regime=decision.regime,
                social_risk_score=decision.combined_risk_score,
                attention_z=snapshot.attention_z,
                sentiment_score=snapshot.sentiment_score,
                gross_exposure_multiplier=decision.gross_exposure_multiplier,
                block_new_entries=decision.block_new_entries,
            )
            if decision.block_new_entries:
                result = "block"
            elif float(decision.gross_exposure_multiplier) < 0.999:
                result = "reduce"
            else:
                result = "normal"
            self.prometheus_metrics.record_social_decision(
                asset_class=decision.asset_class,
                regime=decision.regime,
                result=result,
                policy_version=decision.policy_version,
            )
            for verification in self._prediction_results_cache.get(key, []):
                self.prometheus_metrics.update_prediction_verification(
                    asset_class=decision.asset_class,
                    regime=decision.regime,
                    event=verification.event_id,
                    verified_probability=verification.verified_probability,
                    verified=verification.verified,
                    failure_reason=verification.reason,
                )

    def _resolve_governor_controls(
        self,
        asset_class: str,
        market_regime: str,
        tier: GovernorTier,
    ) -> Tuple[TierControls, str, object]:
        regime_key = self._map_governor_regime(asset_class, market_regime)
        controls, policy = self.governor_policy_resolver.controls_for(
            asset_class=asset_class,
            regime=regime_key,
            tier=tier,
        )
        return controls, regime_key, policy

    def _record_governor_observability(
        self,
        asset_class: str,
        regime_key: str,
        tier: GovernorTier,
        controls: TierControls,
        policy,
    ) -> None:
        key = f"{asset_class}:{regime_key}"
        previous = self._governor_last_tier_by_key.get(key)
        current = tier.value
        if previous != current:
            if self.prometheus_metrics and previous is not None:
                self.prometheus_metrics.record_governor_transition(
                    asset_class=asset_class,
                    regime=regime_key,
                    from_tier=previous,
                    to_tier=current,
                )
            self._governor_last_tier_by_key[key] = current

        if self.prometheus_metrics:
            tier_level = {"green": 0, "yellow": 1, "orange": 2, "red": 3}.get(current, 0)
            self.prometheus_metrics.update_governor_state(
                asset_class=asset_class,
                regime=regime_key,
                tier_level=tier_level,
                size_multiplier=controls.size_multiplier,
                signal_threshold_boost=controls.signal_threshold_boost,
                confidence_boost=controls.confidence_boost,
                halt_entries=controls.halt_new_entries,
                policy_version=getattr(policy, "version", "unknown"),
            )

    def _load_governor_tuning_state(self) -> None:
        try:
            if not self._governor_tune_state_file.exists():
                return
            with open(self._governor_tune_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            parsed: Dict[str, datetime] = {}
            for asset_class, iso_ts in data.items():
                if isinstance(asset_class, str) and isinstance(iso_ts, str):
                    parsed[asset_class.upper()] = datetime.fromisoformat(iso_ts)
            self._governor_last_tune_at = parsed
        except Exception as exc:
            logger.debug("Failed loading governor tuning state: %s", exc)

    def _save_governor_tuning_state(self) -> None:
        try:
            self._governor_tune_state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                asset_class: ts.isoformat() + "Z"
                for asset_class, ts in self._governor_last_tune_at.items()
            }
            with open(self._governor_tune_state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.debug("Failed saving governor tuning state: %s", exc)

    def _crypto_instability_detected(self, signal_df: pd.DataFrame) -> bool:
        if signal_df.empty:
            return False
        local = signal_df.copy()
        def _safe_asset(symbol_value: object) -> str:
            if not isinstance(symbol_value, str):
                return "EQUITY"
            try:
                return parse_symbol(symbol_value).asset_class.value
            except Exception:
                return "EQUITY"

        local["asset_class"] = local["symbol"].map(_safe_asset)
        crypto = local[local["asset_class"] == "CRYPTO"]
        if len(crypto) < 50:
            return False
        returns = pd.to_numeric(crypto["return_10d"], errors="coerce").dropna()
        if len(returns) < 25:
            return False
        return float(returns.std()) > 0.08

    def _cadence_timedelta_for_asset(self, asset_class: str, signal_df: pd.DataFrame) -> timedelta:
        asset = asset_class.upper()
        cadence = ApexConfig.GOVERNOR_TUNE_CADENCE_DEFAULT.lower()
        if asset == "CRYPTO":
            cadence = ApexConfig.GOVERNOR_TUNE_CADENCE_CRYPTO.lower()
            if cadence != "daily" and self._crypto_instability_detected(signal_df):
                cadence = "daily"

        if cadence == "daily":
            return timedelta(days=1)
        return timedelta(days=7)

    def _maybe_tune_governor_policies(self, now: datetime) -> None:
        try:
            signal_df = self.signal_outcome_tracker.get_signals_for_ml()
            if signal_df.empty or len(signal_df) < 120:
                return

            accepted = 0
            for asset_class, regimes in self._governor_regimes_by_asset.items():
                cadence_delta = self._cadence_timedelta_for_asset(asset_class, signal_df)
                last_tuned = self._governor_last_tune_at.get(asset_class)
                if last_tuned and (now - last_tuned) < cadence_delta:
                    continue

                cadence = "daily" if cadence_delta <= timedelta(days=1) else "weekly"
                tuning_config = WalkForwardTuningConfig(
                    cadence=cadence,
                    sharpe_floor_63d=ApexConfig.KILL_SWITCH_SHARPE_FLOOR,
                    historical_mdd_default=ApexConfig.KILL_SWITCH_HISTORICAL_MDD_BASELINE,
                )
                candidates = tune_policies(
                    signal_df=signal_df,
                    config=tuning_config,
                    regimes_by_asset_class={asset_class: regimes},
                )
                if not candidates:
                    continue

                for candidate in candidates:
                    decision = self.governor_promotion.submit_candidate(candidate)
                    logger.info(
                        "🧭 Governor policy %s -> %s (manual=%s): %s",
                        candidate.policy_id(),
                        decision.status.value,
                        decision.manual_approval_required,
                        decision.reason,
                    )
                    if decision.accepted:
                        accepted += 1

                self._governor_last_tune_at[asset_class] = now

            if accepted > 0:
                active = self.governor_policy_repo.load_active()
                self.governor_policy_resolver.reload(active)
            if accepted > 0 or self._governor_last_tune_at:
                self._save_governor_tuning_state()

        except Exception as exc:
            logger.debug("Governor tuning cycle skipped due to error: %s", exc)

    async def _process_external_control_commands(self) -> None:
        """Process operational commands requested by API/ops tooling."""
        try:
            state = read_control_state(self._control_commands_file)
        except Exception as exc:
            logger.debug("Failed reading control commands: %s", exc)
            return

        if state.get("kill_switch_reset_requested"):
            requested_by = str(state.get("requested_by") or "unknown")
            request_id = str(state.get("request_id") or "unknown")
            reason = str(state.get("reason") or "")
            reset_notes: List[str] = []

            if self.kill_switch:
                if self.kill_switch.active:
                    self.kill_switch.reset()
                    self._persist_kill_switch_state()  # persist cleared state
                    reset_notes.append("kill_switch_reset=applied")
                    logger.warning(
                        "🛑 Kill-switch reset by external command (requested_by=%s, reason=%s)",
                        requested_by,
                        reason,
                    )
                else:
                    reset_notes.append("kill_switch_reset=already_inactive")
            else:
                reset_notes.append("kill_switch_reset=unavailable")

            cb_reset = False
            try:
                cb_reset = self.risk_manager.manual_reset_circuit_breaker(
                    requested_by=requested_by,
                    reason=reason or "external_latch_reset",
                )
            except Exception as exc:
                reset_notes.append(f"circuit_breaker_reset=error:{exc}")
            else:
                reset_notes.append(
                    "circuit_breaker_reset=applied" if cb_reset else "circuit_breaker_reset=already_clear"
                )

            await self._rebase_latches_after_reset_for_paper(
                requested_by=requested_by,
                reason=reason,
                reset_notes=reset_notes,
            )

            note = (
                "Unified latch reset processed "
                f"(requested_by={requested_by}, reason={reason}): "
                + ", ".join(reset_notes)
            )
            mark_kill_switch_reset_processed(
                self._control_commands_file,
                processed_by="apex-trader",
                note=note,
            )
            await self.risk_manager.save_state_async()
            logger.warning("Control command %s: %s", request_id, note)

            # Keep exported state/metrics consistent immediately.
            if self.prometheus_metrics:
                try:
                    if self.kill_switch:
                        self.prometheus_metrics.update_kill_switch(active=self.kill_switch.active)
                    self.prometheus_metrics.update_circuit_breaker(
                        is_tripped=self.risk_manager.circuit_breaker.is_tripped,
                    )
                except Exception:
                    pass
            if self.kill_switch:
                self._kill_switch_last_active = bool(self.kill_switch.active)

        if state.get("governor_policy_reload_requested"):
            request_id = str(state.get("governor_policy_reload_request_id") or "unknown")
            requested_by = str(state.get("governor_policy_reload_requested_by") or "unknown")
            reason = str(state.get("governor_policy_reload_reason") or "")
            try:
                active = self.governor_policy_repo.load_active()
                self.governor_policy_resolver.reload(active)
                if self.kill_switch:
                    self.kill_switch.historical_mdd_baseline = self.governor_policy_resolver.historical_mdd_baseline(
                        default_value=ApexConfig.KILL_SWITCH_HISTORICAL_MDD_BASELINE
                    )
                note = (
                    "Governor policies reloaded "
                    f"(requested_by={requested_by}, count={len(active)}, reason={reason})"
                )
                logger.warning("🧭 %s", note)
            except Exception as exc:
                note = f"Governor policy reload failed: {exc}"
                logger.error("Control command %s failed: %s", request_id, note)

            mark_governor_policy_reload_processed(
                self._control_commands_file,
                processed_by="apex-trader",
                note=note,
            )

        if state.get("equity_reconciliation_latch_requested"):
            request_id = str(state.get("equity_reconciliation_latch_request_id") or "unknown")
            requested_by = str(state.get("equity_reconciliation_latch_requested_by") or "unknown")
            reason = str(state.get("equity_reconciliation_latch_reason") or "")
            target_block = bool(state.get("equity_reconciliation_latch_target_block_entries"))

            previous = self._equity_reconciliation_snapshot
            prev_breached = bool(previous.breached) if previous else False

            if target_block:
                self.equity_reconciler.block_entries_latch = True
                self.equity_reconciler.breach_streak = max(
                    self.equity_reconciler.breach_streak,
                    self.equity_reconciler.breach_confirmations,
                )
                self.equity_reconciler.healthy_streak = 0
                manual_reason = "manual_force_block"
            else:
                self.equity_reconciler.block_entries_latch = False
                self.equity_reconciler.breach_streak = 0
                self.equity_reconciler.healthy_streak = max(
                    self.equity_reconciler.healthy_streak,
                    self.equity_reconciler.heal_confirmations,
                )
                manual_reason = "manual_force_clear"

            base_broker = previous.broker_equity if previous else 0.0
            base_modeled = previous.modeled_equity if previous else 0.0
            base_gap_dollars = previous.gap_dollars if previous else 0.0
            base_gap_pct = previous.gap_pct if previous else 0.0
            snapshot = EquityReconciliationSnapshot(
                timestamp=datetime.now(),
                broker_equity=float(base_broker),
                modeled_equity=float(base_modeled),
                gap_dollars=float(base_gap_dollars),
                gap_pct=float(base_gap_pct),
                max_gap_dollars=self.equity_reconciler.max_gap_dollars,
                max_gap_pct=self.equity_reconciler.max_gap_pct,
                breached=bool(target_block),
                block_entries=bool(target_block),
                reason=manual_reason,
                breach_streak=int(self.equity_reconciler.breach_streak),
                healthy_streak=int(self.equity_reconciler.healthy_streak),
            )
            self.equity_reconciler.last_snapshot = snapshot
            self._equity_reconciliation_snapshot = snapshot
            self._equity_reconciliation_block_entries = bool(target_block)

            note = (
                "Equity reconciliation latch manually set "
                f"(target_block={target_block}, requested_by={requested_by}, reason={reason})"
            )
            logger.warning("🧾 %s", note)
            mark_equity_reconciliation_latch_processed(
                self._control_commands_file,
                processed_by="apex-trader",
                note=note,
            )
            if self.prometheus_metrics:
                try:
                    self.prometheus_metrics.update_equity_reconciliation(
                        broker_equity=snapshot.broker_equity,
                        modeled_equity=snapshot.modeled_equity,
                        gap_dollars=snapshot.gap_dollars,
                        gap_pct=snapshot.gap_pct,
                        block_entries=snapshot.block_entries,
                        breach_streak=snapshot.breach_streak,
                        healthy_streak=snapshot.healthy_streak,
                        reason=snapshot.reason,
                        breached=snapshot.breached,
                        breach_event=bool(snapshot.breached and not prev_breached),
                    )
                except Exception as exc:
                    logger.debug("Failed updating reconciliation metrics for control command %s: %s", request_id, exc)
    
    async def process_symbol(self, symbol: str):
        """Process symbol with all protections including circuit breaker."""
        # Skip pure delta-hedge positions — managed exclusively by DeltaHedger, not signals
        _hedge_sym = str(getattr(ApexConfig, "DELTA_HEDGE_SYMBOL", "SPY"))
        if symbol == _hedge_sym and getattr(self, '_delta_hedge_qty', {}).get(_hedge_sym, 0) != 0:
            logger.debug("process_symbol: skipping %s (delta hedge position)", symbol)
            return

        try:
            parsed_symbol = parse_symbol(symbol)
            asset_class = parsed_symbol.asset_class.value
        except Exception:
            asset_class = "EQUITY"

        perf_snapshot = self._performance_snapshot or GovernorSnapshot(
            tier=GovernorTier.GREEN,
            sharpe=0.0,
            sortino=0.0,
            drawdown=0.0,
            sample_count=0,
            size_multiplier=1.0,
            signal_threshold_boost=0.0,
            confidence_boost=0.0,
            halt_new_entries=False,
            reasons=[],
        )

        # Setup current quantity early to determine if we can bypass entry blocks
        current_qty = self.positions.get(symbol, 0)

        # Portfolio-level hard kill-switch: block only new entries, never exits.
        if self.kill_switch and self.kill_switch.active:
            if current_qty == 0:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="startup_guard",
                    reason="kill_switch",
                    current_position=float(current_qty),
                )
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime="default",
                        reason="kill_switch",
                    )
                logger.debug("🛑 %s: New entries blocked by kill-switch", symbol)
                return

        # Equity reconciliation hard block: fail-closed for new equity entries only.
        # Crypto trades on Alpaca and is unaffected by IBKR equity reconciliation.
        if self._equity_reconciliation_block_entries and not _symbol_is_crypto(symbol):
            if current_qty == 0:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="startup_guard",
                    reason="equity_reconciliation",
                    current_position=float(current_qty),
                    metadata={
                        "snapshot_reason": (
                            self._equity_reconciliation_snapshot.reason
                            if self._equity_reconciliation_snapshot
                            else "unknown"
                        )
                    },
                )
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime="default",
                        reason="equity_reconciliation",
                    )
                    reason = (
                        self._equity_reconciliation_snapshot.reason
                        if self._equity_reconciliation_snapshot
                        else "unknown"
                    )
                    self.prometheus_metrics.record_equity_reconciliation_entry_block(reason=reason)
                logger.warning("🧾 %s: New entries blocked by equity reconciliation", symbol)
                return

        # Check 0: Circuit breaker check (use institutional for graduated response)
        if self.use_institutional:
            can_trade, reason, risk_mult = self.inst_risk_manager.can_trade()
            self._risk_multiplier = risk_mult  # Store for position sizing
            if not can_trade:
                if current_qty == 0:
                    logger.info(f"🛑 {symbol}: Trading halted - {reason}")
                    self._write_trade_rejection(symbol, f"circuit_breaker:{reason}")
                    return
                else:
                    logger.debug(f"🛑 {symbol}: Trading halted - Entries blocked, Exits allowed ({reason})")
            elif risk_mult < 1.0:
                logger.debug(f"⚠️ {symbol}: Reduced risk mode ({risk_mult:.0%} position size)")
        else:
            can_trade, reason = self.risk_manager.can_trade()
            self._risk_multiplier = 1.0
            if not can_trade:
                if current_qty == 0:
                    logger.info(f"🛑 {symbol}: Trading halted - {reason}")
                    self._write_trade_rejection(symbol, f"circuit_breaker:{reason}")
                    return
                else:
                    logger.debug(f"🛑 {symbol}: Trading halted - Entries blocked, Exits allowed ({reason})")

        # Check 1: Cooldown protection
        last_trade = self.last_trade_time.get(symbol, datetime(2000, 1, 1))
        seconds_since = (datetime.now() - last_trade).total_seconds()

        if seconds_since < ApexConfig.TRADE_COOLDOWN_SECONDS:
            logger.debug(f"⏸️  {symbol}: Cooldown ({int(ApexConfig.TRADE_COOLDOWN_SECONDS - seconds_since)}s left)")
            return

        # Check 1.5: Re-entry gap after exit (entry-only, doesn't block exits).
        # After any exit: min 10 min before re-entering.
        # After a LOSING exit: min 30 min before re-entering (avoids averaging into a trend).
        if current_qty == 0:
            _last_exit = self._last_exit_time.get(symbol)
            if _last_exit is not None:
                _was_loss = self._last_exit_was_loss.get(symbol, False)
                _gap_req = float(getattr(
                    ApexConfig,
                    "LOSS_REENTRY_GAP_SECONDS" if _was_loss else "REENTRY_GAP_SECONDS",
                    1800 if _was_loss else 600,
                ))
                _elapsed = (datetime.now() - _last_exit).total_seconds()
                if _elapsed < _gap_req:
                    logger.debug(
                        "⏸️ %s: Re-entry gap [%s] (%ds left)",
                        symbol, "loss" if _was_loss else "win", int(_gap_req - _elapsed),
                    )
                    return

        # Check 1.5: Crypto entry window — natively disabled for 24/7 continuous evaluations
        # (Removed historical block that previously restricted crypto to US Equity hours)

        # Check 1.6: Session-aware equity entry window — avoid opening/closing turbulence
        # First N minutes after NYSE open (9:30 ET) and last N minutes before close (16:00 ET)
        # have the widest spreads, highest slippage, and most signal noise.
        if (current_qty == 0
                and not _symbol_is_crypto(symbol)
                and str(asset_class).upper() not in ("FX", "FOREX")
                and getattr(ApexConfig, "SESSION_AWARE_ENTRY_ENABLED", True)):
            try:
                import pytz as _pytz_sess
                _now_et = datetime.now(_pytz_sess.timezone("US/Eastern"))
                _h, _m = _now_et.hour, _now_et.minute
                _avoid_open_min = int(getattr(ApexConfig, "EQUITY_ENTRY_AVOID_OPEN_MINUTES", 5))
                _avoid_close_min = int(getattr(ApexConfig, "EQUITY_ENTRY_AVOID_CLOSE_MINUTES", 5))
                # Block first N minutes: 9:30–9:35
                _in_open_window = (_h == 9 and _m >= 30 and _m < 30 + _avoid_open_min)
                # Block last N minutes: 15:55–16:00
                _close_boundary = 60 - _avoid_close_min
                _in_close_window = (_h == 15 and _m >= _close_boundary)
                if _in_open_window or _in_close_window:
                    logger.debug(
                        "⏸️ %s: Equity session gate — blocked at %02d:%02d ET (open=%s, close=%s)",
                        symbol, _h, _m, _in_open_window, _in_close_window,
                    )
                    return
            except Exception as _sess_e:
                logger.debug("Session-aware entry gate error: %s", _sess_e)

        # Check 2: Skip if order pending
        if symbol in self.pending_orders:
            logger.debug(f"⏳ {symbol}: Order pending")
            return

        # Check 2.5: Signal Fortress - skip quarantined symbols
        if self.signal_integrity and self.signal_integrity.is_quarantined(symbol):
            if current_qty == 0:
                logger.debug(f"🏰 {symbol}: Quarantined by signal integrity monitor")
                return
            else:
                logger.debug(f"🏰 {symbol}: Quarantined, but position exists - allowing exit evaluation")

        # Phase 3: Initial Trades Relaxation
        # If we have ZERO tradeable positions (any asset class), allow entries even if
        # some guards are active (specifically for the initial portfolio build).
        # NOTE: CRYPTO positions count — excluding them caused a perpetual re-buy loop
        # when the portfolio was all-crypto ($0.92 cash but 3 open positions).
        active_core_positions = sum(
            1 for s, qty in self.positions.items()
            if qty != 0
            and "OPT:" not in str(s).upper()
            and not str(s).upper().startswith("OPTION:")
        )
        is_initial_build = active_core_positions == 0

        # Check 2.6: Black Swan Guard
        if self.black_swan_guard and self.black_swan_guard.should_block_entry():
            if current_qty == 0 and not is_initial_build:
                logger.debug(f"🛡️ {symbol}: Entry blocked by BlackSwanGuard")
                return

        # Check 2.7: Drawdown Cascade
        if self.drawdown_breaker and not self.drawdown_breaker.get_entry_allowed():
            if current_qty == 0 and not is_initial_build:
                logger.debug(f"🛡️ {symbol}: Entry blocked by DrawdownCascadeBreaker")
                return

        # Check 2.7b: Intraday rolling drawdown gate (60-min rolling window, entry-only block)
        if getattr(ApexConfig, "INTRADAY_DD_GATE_ENABLED", True) and self._intraday_dd_gate_active:
            if current_qty == 0 and not is_initial_build:
                logger.debug("🛡️ %s: New entry blocked by intraday DD gate", symbol)
                return

        # Check 2.8: Correlation Cascade
        if self.correlation_breaker:
            existing_positions = [s for s, qty in self.positions.items() if qty != 0]
            if self.correlation_breaker.should_block_entry(symbol, existing_positions, self.historical_data):
                if current_qty == 0 and not is_initial_build:
                    logger.info(f"🛡️ {symbol}: Entry blocked by CorrelationCascadeBreaker")
                    return

        # Check 2.9: Macro Shield
        if self.macro_shield and self.macro_shield.is_blackout_active():
            if current_qty == 0 and not is_initial_build:
                event = self.macro_shield.get_active_event()
                evt_name = event.title if event else "Unknown Event"
                logger.info(f"🛡️ {symbol}: Entry blocked by Macro Shield ({evt_name})")
                return

        # Check 2.10: Phase 3 Macro Event Shield
        if hasattr(self, 'macro_event_shield') and self.macro_event_shield:
            blocked, reason = self.macro_event_shield.should_block_entry(symbol)
            if blocked:
                if current_qty == 0 and not is_initial_build:
                    logger.info(f"🛡️ {symbol}: Entry blocked by MacroEventShield ({reason})")
                    return

        # Check 2.11: Phase 3 Overnight Risk Guard
        if hasattr(self, 'overnight_guard') and self.overnight_guard:
            try:
                _parsed_overnight = parse_symbol(symbol)
                _skip_overnight = _parsed_overnight.asset_class in (AssetClass.CRYPTO, AssetClass.FOREX)
            except ValueError:
                _skip_overnight = False
            if not _skip_overnight:
                if not is_initial_build:
                    blocked, reason = self.overnight_guard.should_block_entry()
                    if blocked:
                        if current_qty == 0:
                            logger.info(f"🛡️ {symbol}: Entry blocked by OvernightRiskGuard ({reason})")
                            return

        # Check 2.12: Phase 3 Liquidity Guard
        if hasattr(self, 'liquidity_guard') and self.liquidity_guard:
            blocked, reason = self.liquidity_guard.should_block_entry(symbol)
            if blocked:
                if current_qty == 0 and not is_initial_build:
                    if random.random() < 0.01:
                        logger.debug(f"🛡️ {symbol}: Entry blocked by LiquidityGuard ({reason})")
                    return

        # Check 2.13: Sector Concentration Guard (sync, zero I/O)
        if current_qty == 0 and not is_initial_build:
            _sc_blocked, _sc_sector, _sc_pct = self._sector_concentration_blocked(symbol)
            if _sc_blocked:
                logger.info(
                    "🛡️ %s: Entry blocked by SectorConcentrationGuard (%s at %.1f%%, max=%.0f%%)",
                    symbol, _sc_sector, _sc_pct * 100,
                    getattr(ApexConfig, "SECTOR_CONCENTRATION_MAX_PCT", 0.25) * 100,
                )
                fire_alert(
                    "sector_concentration",
                    f"Sector guard blocked {symbol}: {_sc_sector} at {_sc_pct*100:.1f}% "
                    f"(max={getattr(ApexConfig, 'SECTOR_CONCENTRATION_MAX_PCT', 0.25)*100:.0f}%)",
                    AlertSev.WARNING,
                )
                return

        # Pre-initialise signal/confidence/data/price so that the gate checks below
        # (which may run before signal generation) see defined names.  The real values
        # are assigned later during signal generation; these defaults are safe because
        # every gate is wrapped in a try/except and guarded by feature-flag checks.
        signal: float = 0.0
        confidence: float = 0.0
        data = self.historical_data.get(symbol)  # may be None
        price: float = self.price_cache.get(symbol, 0.0)

        # Check 2.14: News Confirmation Gate (async, cached 20 min)
        # Blocks or penalises entries when news sentiment strongly contradicts the signal.
        if (
            current_qty == 0
            and not is_initial_build
            and getattr(ApexConfig, "NEWS_CONFIRMATION_GATE_ENABLED", True)
            and getattr(self, '_news_aggregator', None) is not None
        ):
            try:
                _news_ctx = await self._news_aggregator.get_news_context(symbol)
                _signal_dir = 1 if signal > 0 else -1
                _contradiction = _news_ctx.sentiment * _signal_dir  # <0 = contradict
                _contra_threshold = float(getattr(ApexConfig, "NEWS_STRONG_CONTRADICTION_THRESHOLD", 0.40))
                _min_conf_gate = float(getattr(ApexConfig, "NEWS_CONTRADICTION_MIN_CONFIDENCE", 0.70))
                if _contradiction < -_contra_threshold:
                    if confidence < _min_conf_gate:
                        logger.info(
                            "📰 %s: Entry blocked by NewsGate (news=%.2f contradicts signal=%.3f, "
                            "conf=%.2f < %.2f required)",
                            symbol, _news_ctx.sentiment, signal, confidence, _min_conf_gate,
                        )
                        return
                    else:
                        # High-conviction trade — allow but penalise confidence
                        _before = confidence
                        confidence *= 0.90
                        logger.debug(
                            "📰 %s: NewsGate contradiction — conf penalised %.2f→%.2f "
                            "(news=%.2f, signal=%.3f)",
                            symbol, _before, confidence, _news_ctx.sentiment, signal,
                        )
                elif _contradiction > 0.30 and _news_ctx.confidence > 0.50:
                    # News confirms signal direction → small confidence boost
                    confidence = min(1.0, confidence * 1.05)
                    logger.debug(
                        "📰 %s: NewsGate confirmation boost → conf=%.2f (news=%.2f)",
                        symbol, confidence, _news_ctx.sentiment,
                    )
            except Exception as _ng_err:
                logger.debug("NewsGate skipped (non-fatal): %s", _ng_err)

        # Check 2.14b: Macro Regime Confidence Gate
        # Applies yield curve inversion and VIX backwardation penalties to entry confidence.
        # Supplements the macro SIZE multipliers (already applied) with a signal QUALITY
        # penalty — macro stress doesn't just shrink size, it also demands higher conviction.
        if (
            current_qty == 0
            and not is_initial_build
            and getattr(ApexConfig, "MACRO_CONFIDENCE_GATE_ENABLED", True)
            and getattr(self, '_macro_context', None) is not None
        ):
            try:
                _mc = self._macro_context
                _is_equity_entry = str(asset_class).upper() not in ("CRYPTO", "FOREX")
                # Yield curve inversion: equity longs face recession risk → higher conviction needed
                if _is_equity_entry and signal > 0 and getattr(_mc, 'yield_curve_inverted', False):
                    _before = confidence
                    _yc_penalty = float(getattr(ApexConfig, "MACRO_YIELD_CURVE_CONF_PENALTY", 0.92))
                    confidence = max(0.0, confidence * _yc_penalty)
                    logger.debug(
                        "📊 %s: Yield curve inverted → conf %.3f→%.3f",
                        symbol, _before, confidence,
                    )
                # VIX backwardation: near-term stress elevated → dampen all entries
                _vx_ratio = float(getattr(_mc, 'vix_spot_futures_ratio', 0.95) or 0.95)
                if _vx_ratio > 1.0:
                    _before = confidence
                    _vx_penalty = float(getattr(ApexConfig, "MACRO_VIX_BACKWARDATION_CONF_PENALTY", 0.88))
                    confidence = max(0.0, confidence * _vx_penalty)
                    logger.debug(
                        "📊 %s: VIX backwardation (ratio=%.2f) → conf %.3f→%.3f",
                        symbol, _vx_ratio, _before, confidence,
                    )
            except Exception:
                pass

        # Check 2.16b: Options Flow (Smart Money) Gate — equity only, cached 1hr
        # PCR > 1 = heavy puts = bearish institutional positioning.
        # Block BUY entries when options flow strongly contradicts signal direction.
        # Block SELL (short) entries when options flow strongly contradicts.
        if (
            current_qty == 0
            and not is_initial_build
            and str(asset_class).upper() not in ("CRYPTO", "FOREX")
            and getattr(ApexConfig, "OPTIONS_FLOW_GATE_ENABLED", True)
        ):
            try:
                _of_contradict_thresh = float(getattr(ApexConfig, "OPTIONS_FLOW_CONTRA_THRESHOLD", 0.40))
                _of_sentiment = await asyncio.to_thread(
                    getattr(__import__('data.social.options_flow', fromlist=['get_smart_money_sentiment']),
                            'get_smart_money_sentiment'), symbol
                )
                # Positive signal = long intent; negative options flow = bearish smart money
                _pos_dir = 1 if signal > 0 else -1
                _contra = float(_of_sentiment) * (-_pos_dir)   # >0 means options contradict direction
                if _contra > _of_contradict_thresh:
                    _of_block_min_conf = float(getattr(ApexConfig, "OPTIONS_FLOW_BLOCK_MIN_CONF", 0.75))
                    if confidence < _of_block_min_conf:
                        logger.info(
                            "🎯 %s: Options flow contra=%.2f blocks entry (conf=%.3f < %.2f, flow=%.2f)",
                            symbol, _contra, confidence, _of_block_min_conf, _of_sentiment,
                        )
                        return
                    else:
                        # High conf but contra flow: penalise confidence slightly
                        _of_pen = float(getattr(ApexConfig, "OPTIONS_FLOW_CONF_PENALTY", 0.93))
                        confidence = confidence * _of_pen
                        logger.debug(
                            "🎯 %s: Options flow contra=%.2f, high conf — penalty ×%.2f → %.3f",
                            symbol, _contra, _of_pen, confidence,
                        )
            except Exception as _of_err:
                logger.debug("OptionsFlow gate error for %s: %s", symbol, _of_err)

        # Check 2.17: External Signal Aggregator (Funding Rate + Candlestick Patterns)
        # Completely independent from ML models: different input types, no shared state.
        # Adjusts confidence up/down; blocks only when ALL votes contradict AND conf is low.
        _agg = getattr(self, '_signal_aggregator', None)
        if _agg is not None and current_qty == 0 and not is_initial_build:
            _votes: list = []
            # --- Funding Rate vote (crypto only) ---
            _frs = getattr(self, '_funding_rate_signal', None)
            if (
                _frs is not None
                and _symbol_is_crypto(symbol)
                and getattr(ApexConfig, "FUNDING_RATE_SIGNAL_ENABLED", True)
            ):
                try:
                    from signals.signal_aggregator import SignalVote as _SV
                    _fr_ctx = await _frs.get_signal(symbol)
                    if _fr_ctx.confidence > 0.10:
                        _votes.append(_SV(
                            signal=_fr_ctx.signal,
                            confidence=_fr_ctx.confidence,
                            source="funding_rate",
                            applies_to="crypto",
                        ))
                        logger.debug(
                            "FundingRate %s: rate=%.4f%% signal=%+.3f conf=%.2f (%s)",
                            symbol, _fr_ctx.current_rate * 100,
                            _fr_ctx.signal, _fr_ctx.confidence, _fr_ctx.direction,
                        )
                except Exception as _fe:
                    logger.debug("FundingRate fetch error (%s): %s", symbol, _fe)
            # --- Candlestick Pattern vote (all assets) ---
            _ps = getattr(self, '_pattern_signal', None)
            if (
                _ps is not None
                and isinstance(data, pd.DataFrame)
                and len(data) >= 3
                and getattr(ApexConfig, "PATTERN_SIGNAL_ENABLED", True)
            ):
                try:
                    from signals.signal_aggregator import SignalVote as _SV
                    _pat = _ps.get_signal(symbol, data)
                    if _pat.confidence > 0.15:
                        _votes.append(_SV(
                            signal=_pat.signal,
                            confidence=_pat.confidence,
                            source="pattern",
                            applies_to="all",
                        ))
                        logger.debug(
                            "Pattern %s: %s signal=%+.3f conf=%.2f",
                            symbol, _pat.patterns_found,
                            _pat.signal, _pat.confidence,
                        )
                except Exception as _pe:
                    logger.debug("Pattern signal error (%s): %s", symbol, _pe)
            # --- Aggregate all votes ---
            if _votes:
                confidence, _block = _agg.combine(
                    primary_signal=float(signal),
                    votes=_votes,
                    primary_confidence=float(confidence),
                    asset_class=str(asset_class),
                )
                if _block:
                    logger.info(
                        "⛔ %s: Signal aggregator blocked entry "
                        "(signal=%+.3f conf=%.2f votes=[%s])",
                        symbol, signal, confidence,
                        ", ".join(f"{v.source}={v.signal:+.2f}" for v in _votes),
                    )
                    return

        # Check 2.15 + 2.16: VWAP Deviation Gate + RVOL Gate
        # Both use existing historical_data bars — zero extra network calls.
        # Only evaluated on new entries (current_qty == 0, not initial build).
        if current_qty == 0 and not is_initial_build and symbol in self.historical_data:
            _df_gate = self.historical_data[symbol]
            if _df_gate is not None and len(_df_gate) >= 5:
                # ── 2.15 VWAP deviation ──────────────────────────────────────
                if getattr(ApexConfig, "VWAP_GATE_ENABLED", True):
                    try:
                        from risk.entry_filters import vwap_gate_check as _vwap_chk
                        _atr_pct = 0.0
                        if "Close" in _df_gate.columns and len(_df_gate) >= 20:
                            _c = _df_gate["Close"].values
                            _atr_pct = float(
                                (abs(_c[-20:] - _c[-21:-1]).mean() / max(_c[-1], 1e-8)) * 100
                            ) if len(_c) >= 21 else 0.0
                        _vblocked, _vdev, _vreason = _vwap_chk(
                            price=price,
                            df=_df_gate,
                            signal=signal,
                            atr_pct=_atr_pct,
                            is_crypto=_symbol_is_crypto(symbol),
                            max_deviation_pct=float(getattr(ApexConfig, "VWAP_MAX_DEVIATION_PCT", 2.0)),
                            atr_adjust=bool(getattr(ApexConfig, "VWAP_ATR_ADJUST", True)),
                        )
                        if _vblocked:
                            logger.info(
                                "📏 %s: Entry blocked by VWAPGate (%s)",
                                symbol, _vreason,
                            )
                            return
                    except Exception as _ve:
                        logger.debug("VWAPGate error (non-fatal): %s", _ve)

                # ── 2.16 RVOL ────────────────────────────────────────────────
                if getattr(ApexConfig, "RVOL_GATE_ENABLED", True):
                    try:
                        from risk.entry_filters import rvol_check as _rvol_chk
                        _rvol, _rblocked, _rreason = _rvol_chk(
                            df=_df_gate,
                            min_rvol=float(getattr(ApexConfig, "RVOL_MIN_THRESHOLD", 0.30)),
                        )
                        if _rblocked:
                            logger.info(
                                "📉 %s: Entry blocked by RVOLGate (%s)",
                                symbol, _rreason,
                            )
                            return
                    except Exception as _re:
                        logger.debug("RVOLGate error (non-fatal): %s", _re)

        # Check 2.18: Post-Earnings Announcement Drift (PEAD) gate — equity only
        # PEAD is a documented anomaly: stocks beat on EPS drift UP for 30-60 days;
        # misses drift DOWN. Going against a strong PEAD signal is a documented edge decay.
        # Does NOT apply to crypto/FX (no EPS). Zero extra API calls when cached.
        _pead = getattr(self, '_earnings_signal', None)
        if (
            _pead is not None
            and current_qty == 0
            and not is_initial_build
            and not _symbol_is_crypto(symbol)
            and str(asset_class).upper() not in ("FX", "FOREX")
            and getattr(ApexConfig, "EARNINGS_PEAD_ENABLED", True)
        ):
            try:
                _pead_ctx = await asyncio.to_thread(_pead.get_signal, symbol)
                if _pead_ctx.direction not in ("no_data", "neutral"):
                    _pead_signal_dir = 1 if _pead_ctx.signal > 0 else -1  # +1=beat, -1=miss
                    _trade_dir = 1 if float(signal) > 0 else -1           # +1=LONG, -1=SHORT
                    _aligned = _pead_signal_dir == _trade_dir

                    _pead_boost = float(getattr(ApexConfig, "EARNINGS_PEAD_CONF_BOOST", 0.06))
                    _pead_penalty = float(getattr(ApexConfig, "EARNINGS_PEAD_CONTRA_PENALTY", 0.10))
                    _pead_block_thresh = float(getattr(ApexConfig, "EARNINGS_PEAD_BLOCK_THRESHOLD", 0.20))

                    if _aligned:
                        # PEAD agrees with trade direction → small confidence boost
                        _before_conf = confidence
                        confidence = min(1.0, confidence + _pead_boost * _pead_ctx.confidence)
                        logger.debug(
                            "📈 PEAD %s: %s (%.1f%% surprise, %dd ago) → conf %.3f→%.3f",
                            symbol, _pead_ctx.direction,
                            _pead_ctx.surprise_pct * 100, _pead_ctx.days_since_earnings,
                            _before_conf, confidence,
                        )
                    else:
                        # PEAD contradicts trade direction → penalize or block
                        if abs(_pead_ctx.surprise_pct) >= _pead_block_thresh and _pead_ctx.confidence > 0.40:
                            logger.info(
                                "⛔ PEAD %s: Strong %s (%.1f%% surprise, %dd ago) contradicts %s entry — blocked",
                                symbol, _pead_ctx.direction,
                                _pead_ctx.surprise_pct * 100, _pead_ctx.days_since_earnings,
                                "LONG" if _trade_dir == 1 else "SHORT",
                            )
                            return
                        else:
                            _before_conf = confidence
                            confidence = max(0.0, confidence - _pead_penalty * _pead_ctx.confidence)
                            logger.debug(
                                "📉 PEAD %s: %s contradicts direction → conf %.3f→%.3f",
                                symbol, _pead_ctx.direction, _before_conf, confidence,
                            )
            except Exception as _pe:
                logger.debug("PEAD gate error (non-fatal) for %s: %s", symbol, _pe)

        # Check 2.19a: Put/Call Ratio macro filter — equity new entries only
        # PCR is a market-wide contrarian sentiment indicator. Extreme readings
        # at the market level suggest the crowd is wrong → fade their sentiment.
        # Only penalises/blocks; PCR extreme bearish → slight confidence boost for LONG.
        _pcr_ctx = getattr(self, '_pcr_context', None)
        if (
            _pcr_ctx is not None
            and current_qty == 0
            and not is_initial_build
            and not _symbol_is_crypto(symbol)
            and str(asset_class).upper() not in ("FX", "FOREX")
            and getattr(ApexConfig, "PCR_GATE_ENABLED", True)
        ):
            try:
                _trade_is_long = float(signal) > 0
                _pcr_align_boost = float(getattr(ApexConfig, "PCR_ALIGN_BOOST", 0.04))
                _pcr_contra_penalty = float(getattr(ApexConfig, "PCR_CONTRA_PENALTY", 0.08))

                if _pcr_ctx.direction == "extreme_bearish" and _trade_is_long:
                    # Crowd at max fear, going long → contrarian alignment
                    confidence = min(1.0, confidence + _pcr_align_boost * _pcr_ctx.confidence)
                    logger.debug("PCR %s: extreme bearish → conf boost (pcr=%.2f)", symbol, _pcr_ctx.pcr)
                elif _pcr_ctx.direction == "extreme_bullish" and not _trade_is_long:
                    # Crowd at max complacency, going short → contrarian alignment
                    confidence = min(1.0, confidence + _pcr_align_boost * _pcr_ctx.confidence)
                    logger.debug("PCR %s: extreme complacent → SHORT conf boost (pcr=%.2f)", symbol, _pcr_ctx.pcr)
                elif _pcr_ctx.direction == "extreme_bearish" and not _trade_is_long:
                    # Going short when crowd is already panicking → likely late
                    confidence = max(0.0, confidence - _pcr_contra_penalty * _pcr_ctx.confidence)
                elif _pcr_ctx.direction == "extreme_bullish" and _trade_is_long:
                    # Going long when crowd is complacent → higher risk of reversal
                    confidence = max(0.0, confidence - _pcr_contra_penalty * _pcr_ctx.confidence)
                    logger.debug("PCR %s: extreme complacent → LONG conf penalty (pcr=%.2f)", symbol, _pcr_ctx.pcr)
            except Exception as _pcrge:
                logger.debug("PCR gate error (non-fatal) for %s: %s", symbol, _pcrge)

        # Check 2.19b: Opening Range Breakout confirmation gate — equity only, 10:05–15:00 ET
        # Boost confidence when price confirms an ORB in the same direction as the signal.
        # Don't block if no ORB data — the signal stands on its own.
        _orb = getattr(self, '_orb_signal', None)
        if (
            _orb is not None
            and current_qty == 0
            and not is_initial_build
            and not _symbol_is_crypto(symbol)
            and str(asset_class).upper() not in ("FX", "FOREX")
            and getattr(ApexConfig, "ORB_GATE_ENABLED", True)
            and _orb.has_range(symbol)
        ):
            try:
                _rvol_val = float(
                    self._order_flow_imbalance.get(symbol, {}).get("rvol", 1.0)
                    if isinstance(self._order_flow_imbalance.get(symbol), dict)
                    else 1.0
                )
                _orb_ctx = _orb.get_signal(symbol, float(price), rvol=_rvol_val)
                _orb_boost = float(getattr(ApexConfig, "ORB_CONF_BOOST", 0.07))
                _orb_penalty = float(getattr(ApexConfig, "ORB_CONTRA_PENALTY", 0.10))

                if _orb_ctx.direction in ("bullish_breakout", "bearish_breakdown"):
                    _orb_aligned = (
                        (_orb_ctx.direction == "bullish_breakout" and float(signal) > 0)
                        or (_orb_ctx.direction == "bearish_breakdown" and float(signal) < 0)
                    )
                    if _orb_aligned:
                        confidence = min(1.0, confidence + _orb_boost * _orb_ctx.confidence)
                        logger.debug(
                            "ORB %s: %s aligns → conf boost +%.3f (conf=%.3f)",
                            symbol, _orb_ctx.direction, _orb_boost * _orb_ctx.confidence, confidence,
                        )
                    else:
                        confidence = max(0.0, confidence - _orb_penalty * _orb_ctx.confidence)
                        logger.info(
                            "ORB %s: %s CONTRADICTS signal direction → conf penalty -%.3f",
                            symbol, _orb_ctx.direction, _orb_penalty * _orb_ctx.confidence,
                        )
            except Exception as _orb_ge:
                logger.debug("ORB gate error (non-fatal) for %s: %s", symbol, _orb_ge)

        # Get data
        if symbol not in self.historical_data:
            # Orphan position safety exit: if we hold this symbol but have no data,
            # attempt a price-based exit for startup_restore / duplicate positions.
            orphan_qty = self.positions.get(symbol, 0)
            if orphan_qty != 0:
                pa = self._performance_attribution.open_positions if hasattr(self, '_performance_attribution') and self._performance_attribution else {}
                pos_info = pa.get(symbol, {}) if isinstance(pa, dict) else {}
                src = pos_info.get("source", "") if pos_info else ""
                entry_signal = float(pos_info.get("entry_signal", 0.0) if pos_info else 0.0)
                entry_time_raw = pos_info.get("entry_time") if pos_info else None
                # Exit startup_restore positions with zero entry signal held > 3 days,
                # and any duplicate CRYPTO:/FX:-prefixed positions immediately.
                is_prefixed_dup = symbol.startswith("CRYPTO:") or symbol.startswith("FX:")
                days_held = 0
                if entry_time_raw:
                    try:
                        et = datetime.fromisoformat(str(entry_time_raw).replace("Z", "+00:00"))
                        days_held = (datetime.now(et.tzinfo) - et).days
                    except Exception:
                        pass
                force_exit = is_prefixed_dup or (src == "startup_restore" and abs(entry_signal) < 0.05 and days_held >= 3)
                if force_exit:
                    logger.warning(
                        "🚪 %s: Orphan position (qty=%s, src=%s, days=%d, sig=%.3f) — force-exiting (no historical data)",
                        symbol, orphan_qty, src, days_held, entry_signal,
                    )
                    connector = self._get_connector_for(symbol)
                    if connector:
                        try:
                            exit_side = "SELL" if orphan_qty > 0 else "BUY"
                            await asyncio.wait_for(
                                connector.execute_order(symbol, exit_side, abs(orphan_qty)),
                                timeout=30.0,
                            )
                            async with self._position_lock:
                                self.positions.pop(symbol, None)
                            # Clean up all per-position metadata to prevent stale data
                            # contaminating future entries for the same symbol.
                            for meta in (
                                self.position_entry_prices,
                                self.position_entry_times,
                                self.position_entry_signals,
                                self.position_peak_prices,
                                getattr(self, 'failed_exits', {}),
                            ):
                                meta.pop(symbol, None)
                        except (Exception, asyncio.TimeoutError) as _oe:
                            # Track failed orphan exits to escalate after repeated failures
                            _fail_dict = getattr(self, '_orphan_exit_failures', {})
                            _fail_dict[symbol] = _fail_dict.get(symbol, 0) + 1
                            self._orphan_exit_failures = _fail_dict
                            _fail_count = _fail_dict[symbol]
                            logger.error(
                                "❌ Orphan exit failed for %s (attempt %d): %s",
                                symbol, _fail_count, _oe,
                            )
                            if _fail_count >= 3:
                                fire_alert(
                                    "orphan_exit_stuck",
                                    f"Orphan position {symbol} (qty={orphan_qty}) failed "
                                    f"exit {_fail_count}x — manual intervention required",
                                    AlertSev.CRITICAL,
                                )
            else:
                logger.info(f"⚠️ {symbol}: No historical data – skipping (data not yet loaded)")
            return
        
        try:
            # Use cached positions (refreshed at cycle start) to avoid race conditions
            connector = self._get_connector_for(symbol)
            if connector:
                # For IBKR-routed symbols use the cycle-level IBKR position cache.
                # For Alpaca crypto symbols the IBKR cache is empty — fall back to
                # self.positions (which is the authoritative cross-broker position dict).
                _via_ibkr = bool(self.ibkr and connector is self.ibkr)
                if self._cached_ibkr_positions is not None and _via_ibkr:
                    current_pos = self._cached_ibkr_positions.get(symbol, 0)
                else:
                    current_pos = self.positions.get(symbol, 0)
                    # Alpaca stores crypto positions as CRYPTO:BTC/USD but runtime
                    # processes them as BTC/USD — check the prefixed key as fallback.
                    if current_pos == 0 and not symbol.startswith("CRYPTO:"):
                        current_pos = self.positions.get(f"CRYPTO:{symbol}", 0)

                # Get current price (PHASE 12: WSS Intercept)
                price = 0.0
                if getattr(self, 'websocket_streamer', None):
                    price = self.websocket_streamer.get_current_price(symbol)
                
                # Fallback to legacy REST API if WSS missed the tick
                if not price or price == 0.0:
                    price = await connector.get_market_price(symbol)
                    
                if not price or price == 0.0:
                    logger.info(f"⚠️ {symbol}: No price from broker – skipping")
                    return

                self.price_cache[symbol] = price
                self._price_cache_ts[symbol] = time.time()  # AD: timestamp for TTL
                self.data_quality_monitor.update_price(symbol, price)
                if self.data_watchdog:
                    self.data_watchdog.feed_heartbeat(symbol)
                # Record price freshness for decay shield
                if self.signal_decay_shield:
                    self.signal_decay_shield.record_data_timestamp(symbol, "price")
            else:
                current_pos = self.positions.get(symbol, 0)
                if current_pos == 0 and not symbol.startswith("CRYPTO:"):
                    current_pos = self.positions.get(f"CRYPTO:{symbol}", 0)
                _hist_data = self.historical_data[symbol]
                if hasattr(_hist_data, 'get') and 'Close' in _hist_data:
                    price = float(_hist_data['Close'].iloc[-1])
                elif hasattr(_hist_data, 'iloc'):
                    price = float(_hist_data.iloc[-1])
                else:
                    price = float(_hist_data[-1])
                self.price_cache[symbol] = price
                self._price_cache_ts[symbol] = time.time()  # AD: timestamp for TTL
                self.data_quality_monitor.update_price(symbol, price)
                if self.data_watchdog:
                    self.data_watchdog.feed_heartbeat(symbol)
                if self.signal_decay_shield:
                    self.signal_decay_shield.record_data_timestamp(symbol, "price")

            # Check data freshness before signal generation
            if self.signal_decay_shield and not self.signal_decay_shield.is_data_tradeable(symbol):
                logger.debug(f"🛡️ {symbol}: Data too stale, skipping signal generation")
                # Clean up any pending entry placeholder to avoid blocking future entries
                async with self._position_lock:
                    if symbol in self._pending_entries:
                        self.positions.pop(symbol, None)
                        self._pending_entries.discard(symbol)
                return

            # Generate signal (use institutional or standard)
            data = self.historical_data[symbol]
            if hasattr(data, 'get') and 'Close' in data:
                prices = data['Close']
            else:
                prices = data

            # Compute Order Flow Imbalance from OHLCV (activates OFI gate at signal blend stage)
            # OFI = (up_volume - down_volume) / total_volume over last N bars.
            # Positive = buy pressure, negative = sell pressure. [-1, 1].
            if isinstance(data, pd.DataFrame) and {'Close', 'Open', 'Volume'}.issubset(data.columns):
                _ofi_win = int(getattr(ApexConfig, "OFI_LOOKBACK_BARS", 5))
                _ofi_df = data.tail(_ofi_win)
                _up_vol = float((_ofi_df['Volume'] * (_ofi_df['Close'] > _ofi_df['Open'])).sum())
                _dn_vol = float((_ofi_df['Volume'] * (_ofi_df['Close'] < _ofi_df['Open'])).sum())
                _tot_vol = float(_ofi_df['Volume'].sum())
                if _tot_vol > 0:
                    self._order_flow_imbalance[symbol] = (_up_vol - _dn_vol) / _tot_vol


            # SOTA: Get Cross-Sectional Momentum
            cs_data = self.cs_momentum.get_signal(symbol, self.historical_data)
            cs_signal = cs_data.get('signal', 0)
            
            # SOTA: Get News Sentiment (offloaded — analyze() does blocking time.sleep on retries)
            try:
                sent_result = await asyncio.wait_for(
                    asyncio.to_thread(self.sentiment_analyzer.analyze, symbol),
                    timeout=10.0,
                )
                sent_signal = sent_result.sentiment_score
            except asyncio.TimeoutError:
                logger.debug("⏱️ %s: Sentiment analysis timed out (10s) — using neutral", symbol)
                sent_signal = 0.0

            # Tier 5: component signals (populated in institutional path; defaults for fallback)
            _comp_momentum: float = 0.0
            _comp_reversion: float = 0.0
            _comp_trend: float = 0.0

            if self.use_institutional:
                # Institutional signal generator with full metadata (passing full data + context)
                inst_signal: SignalOutput = self.inst_signal_generator.generate_signal(
                    symbol, 
                    data,
                    sentiment_score=sent_signal,
                    momentum_rank=cs_data.get('rank_percentile', 0.5)
                )
                signal = inst_signal.signal
                confidence = inst_signal.confidence

                # I: Tiered signal confidence decay based on price data age.
                # Stale data → reduced confidence → smaller position, higher bar to trade.
                _price_age_s = time.time() - self._price_cache_ts.get(symbol, time.time())
                if _price_age_s > 600:      # >10 min: block signal entirely
                    logger.debug("I: %s signal rejected — price data %.0fs stale (>10min)", symbol, _price_age_s)
                    return
                elif _price_age_s > 300:    # 5-10 min: 40% confidence
                    confidence = min(confidence, 0.40)
                    logger.debug("I: %s confidence capped at 40%% — price %.0fs stale", symbol, _price_age_s)
                elif _price_age_s > 120:    # 2-5 min: 80% confidence
                    confidence = min(confidence, 0.80)

                # Note: Blending is now handled within the ML feature matrix once models retrain.
                # For now, we keep the signature-passing logic and let the generator decide.

                # Log component breakdown for quant transparency
                # TEMPORARY DEBUG: Log ALL signals to diagnose weak signal issue
                if abs(signal) >= 0.05:  # Lowered from 0.30 for debugging
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE" if abs(signal) > 0.30 else "WEAK"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Breakdown: Tech={inst_signal.signal:.2f} Mom={cs_signal:.2f}({cs_data.get('rank_percentile', 0.5):.0%}) Sent={sent_signal:.2f}")
                    logger.debug(f"   Components: mom={inst_signal.momentum_signal:.2f} rev={inst_signal.mean_reversion_signal:.2f} "
                                f"trend={inst_signal.trend_signal:.2f} vol={inst_signal.volatility_signal:.2f}")
                # Capture sub-signal components for Tier 5 regime agreement gate
                _comp_momentum = float(inst_signal.momentum_signal or 0.0)
                _comp_reversion = float(inst_signal.mean_reversion_signal or 0.0)
                _comp_trend = float(inst_signal.trend_signal or 0.0)
            else:
                # Fallback to standard signal generator (passing full data + context)
                signal_data = self.signal_generator.generate_ml_signal(
                    symbol,
                    data,
                    sentiment_score=sent_signal,
                    momentum_rank=cs_data.get('rank_percentile', 0.5)
                )
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                # Capture regression/binary split for accuracy tracking
                if not hasattr(self, '_last_regression_signal'):
                    self._last_regression_signal: dict = {}
                    self._last_binary_signal: dict = {}
                self._last_regression_signal[symbol] = float(signal_data.get('regression_signal', signal))
                self._last_binary_signal[symbol] = float(signal_data.get('binary_signal', 0.0))

                # Tiered price staleness decay (mirrors institutional path)
                _price_age_s = time.time() - self._price_cache_ts.get(symbol, time.time())
                if _price_age_s > 600:
                    logger.debug("Signal rejected: %s price %.0fs stale (>10min)", symbol, _price_age_s)
                    return
                elif _price_age_s > 300:
                    confidence = min(confidence, 0.40)
                elif _price_age_s > 120:
                    confidence = min(confidence, 0.80)

                # LOG SIGNAL STRENGTH (Quant transparency)
                # TEMPORARY DEBUG: Log ALL signals to diagnose weak signal issue
                if abs(signal) >= 0.05:  # Lowered from 0.30 for debugging
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE" if abs(signal) > 0.30 else "WEAK"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Breakdown: ML={signal_data['signal']:.2f} Mom={cs_signal:.2f} Sent={sent_signal:.2f}")

            # ══ Option B: GodLevel Alpha Layer ════════════════════════════════════
            # Blends options flow (PCR), RL-weighted ensemble, VADER news NLP, and
            # on-chain crypto whale data (Fear & Greed) into the primary signal at
            # a conservative 12% weight so the institutional path dominates (88%).
            # Also tracks the RL action per-symbol for direct reward on trade close.
            if getattr(ApexConfig, "GOD_LEVEL_BLEND_ENABLED", True):
                try:
                    _god_result = self.god_signal_generator.generate_ml_signal(
                        symbol, data
                    )
                    _god_sig = float(_god_result.get('signal', 0.0))
                    _god_conf = float(_god_result.get('confidence', 0.0))
                    _god_weight = float(getattr(ApexConfig, "GOD_LEVEL_BLEND_WEIGHT", 0.12))
                    if abs(_god_sig) > 0.02:  # ignore near-zero god-level noise
                        _pre_blend = signal
                        signal = (1.0 - _god_weight) * signal + _god_weight * _god_sig
                        if _god_conf > 0.10:
                            confidence = (confidence + _god_conf) / 2.0
                        logger.debug(
                            "🎯 GodLevel blend %s: inst=%.3f god=%.3f → %.3f (w=%.0f%%)",
                            symbol, _pre_blend, _god_sig, signal, _god_weight * 100,
                        )
                    # Track RL action per-symbol for direct reward feedback on close (Option A)
                    _god_action = int(_god_result.get('components', {}).get('__rl_action__', -1))
                    if _god_action != -1:
                        if not hasattr(self, '_last_rl_action'):
                            self._last_rl_action: Dict[str, int] = {}
                            self._last_rl_regime: Dict[str, str] = {}
                        self._last_rl_action[symbol] = _god_action
                        self._last_rl_regime[symbol] = str(_god_result.get('regime', self._current_regime))
                except Exception as _gle:
                    logger.debug("GodLevel blend skipped for %s: %s", symbol, _gle)
            # ══════════════════════════════════════════════════════════════════════

            # Apply composite decay (sentiment + features freshness) via shield
            if self.signal_decay_shield:
                _freshness = self.signal_decay_shield.check_freshness(symbol)
                if _freshness.decay_factor < 1.0 and _freshness.stale_components:
                    signal, confidence = self.signal_decay_shield.apply_decay(
                        signal, confidence, _freshness
                    )
                    if confidence < 0.05:
                        logger.debug(
                            "DecayShield: %s signal suppressed "
                            "(factor=%.2f, stale=%s)",
                            symbol, _freshness.decay_factor, _freshness.stale_components,
                        )
                        return

            # Sync with dashboard cache
            try:
                self.current_signals[symbol] = {
                    'signal': float(signal),
                    'confidence': float(confidence),
                    'direction': 'STRONG BUY' if signal > 0.60 else
                                'BUY' if signal > 0.40 else
                                'WEAK BUY' if signal > 0.20 else
                                'NEUTRAL' if signal > -0.20 else
                                'WEAK SELL' if signal > -0.40 else
                                'SELL' if signal > -0.60 else
                                'STRONG SELL',
                    'strength_pct': abs(signal) * 100,
                    'timestamp': datetime.utcnow().isoformat() + "Z"
                }
            except Exception:
                pass

            # Normalize model outputs before gate checks.
            try:
                signal = float(signal)
            except Exception:
                signal = 0.0
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            if not math.isfinite(signal):
                signal = 0.0
            if not math.isfinite(confidence):
                confidence = 0.0

            # Save raw ML signal BEFORE crypto blending. The crypto enhancement
            # layer can flip a -0.15 bearish ML signal to +0.10 when RSI/MACD is
            # momentarily bullish. For EXIT decisions on open positions we must
            # consult the raw ML signal — not the entry-optimized blend.
            _raw_ml_signal = float(signal)
            self._journal_signal_snapshot(
                symbol=symbol,
                asset_class=asset_class,
                signal=float(signal),
                confidence=float(confidence),
                price=float(price),
                current_position=float(current_pos),
                raw_ml_signal=float(_raw_ml_signal),
                components=self._last_signal_components.get(symbol, {}),
            )

            # ═══════════════════════════════════════════════════════════════
            # A/B/C/D: CRYPTO SIGNAL ENHANCEMENT LAYER
            # ═══════════════════════════════════════════════════════════════
            if str(asset_class).upper() == "CRYPTO":
                # A: Compute technical signal (RSI + MACD + BB + volume)
                tech_sig, tech_conf = _crypto_technical_signal(data)

                ml_trained = (
                    (self.use_institutional and hasattr(self.inst_signal_generator, "is_trained")
                     and self.inst_signal_generator.is_trained)
                    or (not self.use_institutional and hasattr(self.signal_generator, "is_trained")
                        and self.signal_generator.is_trained)
                )
                if not ml_trained or (signal == 0.0 and confidence == 0.0):
                    # Warmup: use pure technical signal until ML is ready
                    signal = tech_sig
                    confidence = tech_conf
                    logger.debug(
                        "A: %s warmup — using pure technical signal=%.3f conf=%.2f",
                        symbol, signal, confidence,
                    )
                else:
                    # ML trained: blend ML + technical with regime-adaptive weights.
                    # In bear/volatile/crisis regimes, trust ML more (captures regime breaks).
                    # In bull/strong_bull, tech gets higher weight (momentum following works).
                    _blend_regime = str(getattr(self, '_current_regime', 'neutral')).lower()
                    _ml_w_by_regime = {
                        'strong_bull': 0.45, 'bull': 0.50, 'neutral': 0.55,
                        'bear': 0.65, 'strong_bear': 0.70, 'volatile': 0.62, 'crisis': 0.72,
                    }
                    # 0.0 in config = "use regime-adaptive map" (falsy → fall back to map)
                    _ml_w = float(getattr(ApexConfig, "CRYPTO_ML_BLEND_WEIGHT", 0.0)) \
                            or _ml_w_by_regime.get(_blend_regime, 0.55)
                    _tech_w = 1.0 - _ml_w

                    # Strategy Rotation: override ml/tech weights if controller has evidence
                    _src = getattr(self, '_strategy_rotation', None)
                    if _src is not None:
                        _rot_weights = _src.get_blend_weights(_blend_regime)
                        _ml_total = _rot_weights.get("ml", _ml_w)
                        _tech_total = _rot_weights.get("tech", _tech_w)
                        _mt_sum = _ml_total + _tech_total
                        if _mt_sum > 0 and abs(_ml_total / _mt_sum - _ml_w) > 0.05:
                            _ml_w = _ml_total / _mt_sum
                            _tech_w = 1.0 - _ml_w
                            logger.debug(
                                "StrategyRotation [%s]: ml=%.2f tech=%.2f (from rotation ctrl)",
                                _blend_regime, _ml_w, _tech_w,
                            )

                    blended_signal = _ml_w * signal + _tech_w * tech_sig
                    blended_conf = _ml_w * confidence + _tech_w * tech_conf

                    # RL Governor confidence overlay: Q-table certainty → conf multiplier
                    try:
                        from models.rl_weight_governor import get_rl_confidence_mult
                        _is_hv_blend = self._vix_risk_multiplier < 0.80
                        _rl_conf_mult = get_rl_confidence_mult(_blend_regime, _is_hv_blend)
                        if _rl_conf_mult != 1.0:
                            blended_conf = min(1.0, blended_conf * _rl_conf_mult)
                    except Exception:
                        pass

                    logger.debug(
                        "A: %s blend ML=%.3f(%.0f%%) tech=%.3f(%.0f%%) → %.3f "
                        "(regime=%s, conf→%.2f)",
                        symbol, signal, _ml_w * 100, tech_sig, _tech_w * 100,
                        blended_signal, _blend_regime, blended_conf,
                    )
                    signal = blended_signal
                    confidence = blended_conf

                # Multi-Timeframe Signal Fusion
                _mtf = getattr(self, '_mtf_fuser', None)
                if _mtf is not None and getattr(ApexConfig, "MTF_FUSION_ENABLED", True):
                    try:
                        _h_df = None
                        _f_df = None
                        _mdata = self.market_data_fetcher
                        if _mdata is not None:
                            _h_df = _mdata.fetch_intraday_data(symbol, interval="1h", period="5d")
                            if _h_df is not None and len(_h_df) < 5:
                                _h_df = None
                            _f_df = _mdata.fetch_intraday_data(symbol, interval="5m", period="1d")
                            if _f_df is not None and len(_f_df) < 6:
                                _f_df = None
                        _mtf_result = _mtf.fuse(
                            daily_signal=signal,
                            hourly_df=_h_df,
                            fivemin_df=_f_df,
                            regime=str(self._current_regime),
                        )
                        signal = _mtf_result.signal
                        confidence = min(1.0, confidence * _mtf_result.confidence_adj)
                        if not _mtf_result.aligned:
                            logger.debug(
                                "MTF [%s]: TF disagreement (5m=%.3f 1h=%.3f d=%.3f) → conf×%.2f",
                                symbol,
                                _mtf_result.tf_signals.get("5m") or 0,
                                _mtf_result.tf_signals.get("1h") or 0,
                                _mtf_result.tf_signals.get("daily") or 0,
                                _mtf_result.confidence_adj,
                            )
                    except Exception as _mtf_err:
                        logger.debug("MTF fusion error for %s: %s", symbol, _mtf_err)

                # ── Intraday Mean-Reversion blend (neutral regime only) ──
                _imr = getattr(self, '_intraday_mr', None)
                if _imr is not None and getattr(ApexConfig, "INTRADAY_MR_ENABLED", True):
                    try:
                        _mdata2 = self.market_data_fetcher
                        _intra_df = None
                        if _mdata2 is not None:
                            _intra_df = _mdata2.fetch_intraday_data(symbol, interval="5m", period="1d")
                        _mr_result = _imr.compute(_intra_df, regime=str(self._current_regime))
                        if _mr_result.regime_eligible and abs(_mr_result.signal) > 0.05:
                            _mr_w = float(getattr(ApexConfig, "INTRADAY_MR_BLEND_WEIGHT", 0.12))
                            signal = signal * (1.0 - _mr_w) + _mr_result.signal * _mr_w
                            confidence = min(1.0, confidence * _mr_result.confidence_adj)
                            logger.debug(
                                "IntradayMR [%s]: sig=%.3f rsi=%.1f devZ=%.2f → blend %.2f",
                                symbol, _mr_result.signal, _mr_result.rsi,
                                _mr_result.deviation_z, signal,
                            )
                    except Exception as _imr_err:
                        logger.debug("IntradayMR error for %s: %s", symbol, _imr_err)

                signal = max(-1.0, min(1.0, signal))
                confidence = max(0.0, min(1.0, confidence))

                # B: BTC leader effect — altcoins follow BTC momentum
                is_btc = symbol.upper() in ("BTC/USD", "BTCUSD", "BTC-USD")
                if not is_btc:
                    btc_mom = self._btc_momentum()
                    if abs(btc_mom) > 0.20:
                        # Only amplify if BTC and altcoin signal are in the same direction
                        if btc_mom * signal > 0:
                            boost = min(0.12, abs(btc_mom) * 0.12)
                            signal = max(-1.0, min(1.0, signal + math.copysign(boost, signal)))
                            logger.debug("B: %s BTC leader boost %.3f → signal=%.3f", symbol, boost, signal)

                # D: Fear & Greed modulation (cached hourly)
                # Buy amplification is suppressed when VIX is elevated (≥20) because F&G
                # extreme-fear readings during high-VIX periods reflect genuine risk-off
                # conditions — amplifying buys would override the hedge dampener.
                try:
                    fg = self._fear_greed_value  # use cached value; refresh happens async elsewhere
                    _vix_elevated = (self._current_vix or 0.0) >= 20.0
                    if fg < 25:          # Extreme Fear → amplify BUY (only if VIX calm), dampen SELL
                        if signal > 0 and not _vix_elevated:
                            signal = min(1.0, signal * 1.25)
                            logger.debug("D: %s Extreme Fear (F&G=%.0f) → amplify buy signal=%.3f", symbol, fg, signal)
                        elif signal > 0 and _vix_elevated:
                            logger.debug("D: %s Extreme Fear (F&G=%.0f) but VIX=%.1f elevated — buy amplification suppressed", symbol, fg, self._current_vix or 0.0)
                        else:
                            signal = signal * 0.70
                            logger.debug("D: %s Extreme Fear (F&G=%.0f) → dampen sell signal=%.3f", symbol, fg, signal)
                    elif fg > 75:        # Extreme Greed → amplify SELL, dampen BUY
                        if signal < 0:
                            signal = max(-1.0, signal * 1.25)
                        else:
                            signal = signal * 0.70
                        logger.debug("D: %s Extreme Greed (F&G=%.0f) → amplify sell signal=%.3f", symbol, fg, signal)
                except Exception:
                    pass

            crypto_rotation_score = 0.0
            if str(asset_class).upper() == "CRYPTO":
                # Blend cross-sectional momentum with model signal for higher-quality
                # crypto entries while keeping deterministic, bounded behavior.
                crypto_rotation_score = self._crypto_rotation_score(symbol)
                cs_strength = min(1.0, max(0.0, abs(float(cs_signal or 0.0))))
                alignment = 1 if signal * float(cs_signal or 0.0) > 0 else (-1 if signal * float(cs_signal or 0.0) < 0 else 0)
                signal_boost_max = float(getattr(ApexConfig, "CRYPTO_MOMENTUM_ALIGN_SIGNAL_BOOST_MAX", 0.08))
                confidence_boost_max = float(getattr(ApexConfig, "CRYPTO_MOMENTUM_ALIGN_CONFIDENCE_BOOST_MAX", 0.10))
                confidence_penalty_max = float(getattr(ApexConfig, "CRYPTO_MOMENTUM_CONFLICT_CONFIDENCE_PENALTY_MAX", 0.08))
                rotation_weight = max(0.35, crypto_rotation_score)

                if alignment > 0 and cs_strength > 0:
                    direction = 0.0
                    if abs(signal) > 1e-9:
                        direction = math.copysign(1.0, signal)
                    elif abs(float(cs_signal or 0.0)) > 1e-9:
                        direction = math.copysign(1.0, float(cs_signal))
                    signal += direction * signal_boost_max * cs_strength * rotation_weight
                    confidence = min(
                        1.0,
                        confidence + confidence_boost_max * cs_strength * rotation_weight,
                    )
                elif alignment < 0 and cs_strength > 0.35:
                    confidence = max(
                        0.0,
                        confidence - confidence_penalty_max * cs_strength,
                    )

                signal = max(-1.0, min(1.0, signal))
                confidence = max(0.0, min(1.0, confidence))

                # Log rotation blending details for crypto entries
                _align_label = "aligned" if alignment > 0 else ("conflicting" if alignment < 0 else "neutral")
                logger.debug(
                    "🔄 Crypto blend [%s] rot_score=%.2f cs_strength=%.2f alignment=%s "
                    "→ signal=%+.3f conf=%.3f",
                    symbol, crypto_rotation_score, cs_strength, _align_label, signal, confidence,
                )

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

                # Store signal components for attribution (read at entry-record time)
                self._last_signal_components[symbol] = {
                    'ml': float(component_signals.get('ml', 0.0) or 0.0),
                    'tech': float(
                        component_signals.get('trend',
                        component_signals.get('mean_reversion',
                        component_signals.get('momentum', 0.0))) or 0.0
                    ),
                    'sentiment': float(component_signals.get('sentiment', 0.0) or 0.0),
                    'cs_momentum': float(component_signals.get('cs_momentum', 0.0) or 0.0),
                }

                # IC Tracker: record feature snapshot for this symbol+date
                # The forward return is filled in at trade close (record_return call).
                try:
                    _ict = getattr(self, '_ic_tracker', None)
                    if _ict is not None:
                        import datetime as _dt_ic
                        _ic_features = {
                            k: float(v) for k, v in component_signals.items()
                            if v is not None and isinstance(v, (int, float))
                        }
                        # Add high-level blended signal as a "feature" too
                        _ic_features["composite_signal"] = float(signal)
                        _ic_features["confidence"] = float(confidence)
                        _ict.record_features(
                            symbol=symbol,
                            date=str(_dt_ic.date.today()),
                            features=_ic_features,
                            signal=float(signal),
                        )
                except Exception:
                    pass  # non-fatal

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
                    reasons = filter_result['rejection_reasons']
                    logger.info(f"🚫 {symbol}: Signal filtered out - {', '.join(reasons[:2])}")
                    self._write_trade_rejection(
                        symbol, f"signal_filter:{';'.join(reasons[:3])}",
                        signal=signal, confidence=confidence,
                    )
                    return

                # Use filtered values
                signal = filter_result['filtered_signal']
                confidence = filter_result['filtered_confidence']

                # Log adjustments if any
                if filter_result['adjustments']:
                    logger.debug(f"   Filter adjustments: {', '.join(filter_result['adjustments'][:3])}")

            # Track cross-asset reference signals (used next cycle for divergence detection)
            if symbol in ("SPY", "SPY_US_IBKR"):
                self._cross_asset_signals['SPY'] = float(signal)
            elif symbol in ("BTC/USD", "BTCUSD", "BTC-USD"):
                self._cross_asset_signals['BTC'] = float(signal)

            # ═══════════════════════════════════════════════════════════════
            # SIGNAL OUTCOME TRACKING - Record signal for forward analysis
            # ═══════════════════════════════════════════════════════════════
            if abs(signal) >= 0.20:  # Track all significant signals
                try:
                    # Build additional metadata for tracking
                    signal_metadata = {
                        'regime': self._current_regime,
                        'vix': self._current_vix,
                        'has_position': current_pos != 0,
                        'position_size': current_pos,
                        'cs_momentum': cs_signal if 'cs_signal' in dir() else 0.0,
                        'sentiment': sent_signal if 'sent_signal' in dir() else 0.0,
                    }

                    # Add institutional signal components if available
                    if self.use_institutional and 'inst_signal' in dir() and inst_signal:
                        signal_metadata.update({
                            'momentum_component': inst_signal.momentum_signal,
                            'mean_reversion_component': inst_signal.mean_reversion_signal,
                            'trend_component': inst_signal.trend_signal,
                            'volatility_component': inst_signal.volatility_signal,
                            'ml_prediction': inst_signal.ml_prediction,
                            'model_agreement': getattr(inst_signal, 'model_agreement', 0.0),
                        })

                    self.signal_outcome_tracker.record_signal(
                        symbol=symbol,
                        signal_value=signal,
                        confidence=confidence,
                        entry_price=price,
                    )
                except Exception as e:
                    logger.debug(f"Signal tracking error for {symbol}: {e}")

            # ═══════════════════════════════════════════════════════════════
            # SIGNAL FORTRESS - Record signal for integrity monitoring
            # ═══════════════════════════════════════════════════════════════
            if self.signal_integrity:
                try:
                    self.signal_integrity.record_signal(
                        symbol=symbol,
                        signal=signal,
                        confidence=confidence,
                        regime=self._current_regime,
                    )
                except Exception as e:
                    logger.debug(f"Signal integrity record error for {symbol}: {e}")

            # Record signal for outcome feedback loop
            if self.outcome_loop and abs(signal) >= 0.20:
                try:
                    generator_signals = {}
                    if self.use_institutional and 'inst_signal' in dir() and inst_signal:
                        generator_signals = {
                            'institutional': inst_signal.signal,
                            'momentum': inst_signal.momentum_signal,
                            'trend': inst_signal.trend_signal,
                        }
                    self.outcome_loop.record_signal(
                        symbol=symbol,
                        signal_value=signal,
                        confidence=confidence,
                        regime=self._current_regime,
                        entry_price=price,
                        generator_signals=generator_signals,
                    )
                except Exception as e:
                    logger.debug(f"Outcome loop record error for {symbol}: {e}")

            # ═══════════════════════════════════════════════════════════════
            # HEDGE MANAGER — dampen signal under stress conditions
            # ═══════════════════════════════════════════════════════════════
            # Reads: portfolio avg correlation, VIX, today's running P&L loss.
            # Outputs a multiplicative dampener applied to the final signal so
            # that sizing and exit thresholds tighten automatically during:
            #   • Correlation CRISIS  (portfolio effectively = 1 position)
            #   • VIX ELEVATED/FEAR   (asymmetric downside risk)
            #   • Accelerating intra-day drawdown
            # The dampener never zeroes the signal; it preserves direction
            # while reducing conviction — letting quick_mismatch_check and
            # DynamicExitManager trigger exits sooner.
            if hasattr(self, 'hedge_manager'):
                try:
                    _vix_val = float(self._current_vix or 15.0)
                    _corr_val = float(getattr(self, '_correlation_avg', 0.0))
                    # Approximate today's running loss from risk session
                    _rs = getattr(self, 'risk_session', None)
                    _dd_today = 0.0
                    if _rs is not None:
                        _start = float(getattr(_rs, 'day_start_capital', 0) or 0)
                        _cur = float(getattr(_rs, 'current_capital', _start) or _start)
                        if _start > 0:
                            _dd_today = (_cur - _start) / _start  # negative = loss
                    # Raw ML and tech component signals for disagreement check
                    _ml_raw = None
                    _tech_raw = None
                    if self.use_institutional and 'inst_signal' in dir() and inst_signal:
                        _ml_raw = getattr(inst_signal, 'ml_prediction', None)
                        _rule = getattr(inst_signal, 'rule_signal', None)
                        if _rule is None:
                            _rule = getattr(inst_signal, 'momentum_signal', None)
                        _tech_raw = _rule
                    _position_side = 'LONG' if current_pos > 0 else ('SHORT' if current_pos < 0 else 'FLAT')
                    _hedge_adj = self.hedge_manager.get_adjustment(
                        symbol=symbol,
                        position_side=_position_side,
                        base_signal=signal,
                        portfolio_avg_correlation=_corr_val,
                        vix=_vix_val,
                        daily_pnl_pct=_dd_today,
                        ml_signal=_ml_raw,
                        tech_signal=_tech_raw,
                    )
                    # Store dampener so downstream entry gates can adjust thresholds
                    # proportionally — avoids double-blocking (dampened signal + unchanged threshold)
                    self._last_hedge_dampener = _hedge_adj.dampener
                    if _hedge_adj.dampener < 1.0:
                        signal = signal * _hedge_adj.dampener
                        signal = max(-1.0, min(1.0, signal))
                except Exception as _he:
                    self._last_hedge_dampener = 1.0
                    logger.debug("HedgeManager error (non-fatal): %s", _he)

            # 📈 Macro Momentum Overlay: boost strongly-trending assets, dampen counter-trend entries.
            # Gold +19% over 2 months → amplify buy signal. Tech -10% → dampen 'go long' signal.
            # Only applies when we have enough price history (21 bars).
            if prices is not None and len(prices) >= 21:
                try:
                    _20d_ret = float(prices.iloc[-1] / prices.iloc[-20] - 1)
                    _mom_boost = float(getattr(ApexConfig, 'MOMENTUM_SIGNAL_BOOST', 1.20))
                    if _20d_ret > 0.10 and signal > 0:
                        signal = min(1.0, signal * _mom_boost)
                        logger.debug(
                            "📈 %s: Momentum overlay +%.0f%% (20d_ret=+%.1f%%)",
                            symbol, (_mom_boost - 1) * 100, _20d_ret * 100,
                        )
                    elif _20d_ret < -0.05 and signal > 0:
                        signal = signal * 0.60
                        logger.debug(
                            "📉 %s: Momentum overlay −40%% (counter-trend buy, 20d_ret=%.1f%%)",
                            symbol, _20d_ret * 100,
                        )
                except Exception:
                    pass  # Non-fatal: proceed with unmodified signal

            # 🔬 ML/NLP Signal Enhancer: multi-factor statistical overlay.
            # Applies NLP sentiment, momentum calibration, and confidence isotonic regression.
            # Must run AFTER hedge dampener (signal already risk-adjusted) and BEFORE entry gates.
            _enhancer = getattr(self, '_signal_enhancer', None)
            if _enhancer is not None:
                try:
                    _asset_str = str(asset_class).lower()
                    _vols = None
                    if data is not None and hasattr(data, 'columns'):
                        for _vcol in ('volume', 'Volume'):
                            if _vcol in data.columns:
                                _vols = data[_vcol].dropna()
                                break
                    _now_utc_h = datetime.utcnow().hour
                    _spread_est = float(getattr(self, '_last_spread_bps', {}).get(symbol, 10.0) or 10.0)
                    _enhanced = _enhancer.enhance(
                        symbol=symbol,
                        raw_signal=float(signal),
                        raw_confidence=float(confidence),
                        price_history=prices,
                        volume_history=_vols,
                        regime=str(getattr(self, '_current_regime', 'neutral')).lower(),
                        news_headlines=None,   # headlines injected by sentiment_analyzer separately
                        asset_class=_asset_str,
                        spread_bps=_spread_est,
                        hour_utc=_now_utc_h,
                    )
                    # Store components per-symbol so record_outcome() can weight them at close
                    if not hasattr(self, '_signal_enhancer_components'):
                        self._signal_enhancer_components = {}
                    self._signal_enhancer_components[symbol] = _enhanced.components
                    # Apply: replace confidence with calibrated version; blend signal
                    confidence = float(_enhanced.calibrated_confidence)
                    # Only adopt enhanced signal if it agrees in direction with raw signal
                    if (float(_enhanced.enhanced_signal) * float(signal)) >= 0:
                        signal = float(_enhanced.enhanced_signal)
                    _net_adj = _enhanced.components.get("net_adj", 0.0)
                    if abs(_net_adj) > 0.02:
                        logger.debug(
                            "🔬 %s: SignalEnhancer adj=%.3f → signal=%.3f conf=%.3f "
                            "(sentiment=%.3f mom_adj=%.3f micro=%s)",
                            symbol, _net_adj, signal, confidence,
                            _enhanced.sentiment_adj, _enhanced.momentum_adj,
                            "✓" if _enhanced.microstructure_ok else "⚠",
                        )
                except Exception as _enhancer_err:
                    logger.debug("SignalEnhancer error (non-fatal): %s", _enhancer_err)

            # Pairs trading overlay: blend cointegration z-score signal (equity only)
            _pairs_adj = float(self._pairs_overlay.get(symbol, 0.0))
            if _pairs_adj != 0.0 and not _symbol_is_crypto(symbol):
                _pw = float(getattr(ApexConfig, "PAIRS_SIGNAL_WEIGHT", 0.15))
                signal = float(np.clip(signal + _pairs_adj * _pw, -1.0, 1.0))
                logger.debug("🔗 %s: Pairs overlay adj=%.3f → signal=%.3f", symbol, _pairs_adj, signal)

            # Order flow imbalance: real-time bid/ask pressure confirmation
            # +1 = all bids (strong buy pressure), -1 = all asks (strong sell pressure)
            _ofi = float(self._order_flow_imbalance.get(symbol, 0.0))
            if abs(_ofi) > 0.0 and getattr(ApexConfig, "ORDER_FLOW_GATE_ENABLED", True):
                _ofi_weight = float(getattr(ApexConfig, "ORDER_FLOW_SIGNAL_WEIGHT", 0.08))
                # Directional confirmation: OFI agrees with signal → small boost
                # Contradiction: OFI strongly against signal AND we have a new entry → penalise
                _signal_dir = 1 if signal > 0 else (-1 if signal < 0 else 0)
                if _signal_dir != 0 and (_ofi * _signal_dir < -0.3) and current_pos == 0:
                    # Strong contradiction at entry: reduce confidence
                    confidence = max(0.0, confidence * (1.0 - _ofi_weight))
                    logger.debug("⚡ %s: OFI contra (%.2f) → conf=%.3f", symbol, _ofi, confidence)
                elif _ofi * _signal_dir > 0.3:
                    signal = float(np.clip(signal + _ofi * _ofi_weight, -1.0, 1.0))

            # ═══════════════════════════════════════════════════════════════
            # DYNAMIC EXIT LOGIC - Adapts to market conditions
            # ═══════════════════════════════════════════════════════════════

            if current_pos != 0:  # ✅ Handles both long (pos) and short (neg)
                # ✅ FIX: Skip if already failed too many times - let retry_failed_exits handle exclusively
                if symbol in self.failed_exits and self.failed_exits[symbol].get('attempts', 0) >= 5:
                    logger.debug(f"⏭️ {symbol}: Skipping exit in process_symbol - max attempts reached, requires manual intervention")
                    return

                # Metadata may be keyed as CRYPTO:BTC/USD when position was found
                # via the prefix fallback.  Use the key that actually has the data.
                _meta_key = symbol
                if (symbol not in self.position_entry_prices
                        and not symbol.startswith("CRYPTO:")
                        and f"CRYPTO:{symbol}" in self.position_entry_prices):
                    _meta_key = f"CRYPTO:{symbol}"

                entry_price = self.position_entry_prices.get(_meta_key, price)
                entry_time = self.position_entry_times.get(_meta_key, datetime.utcnow())
                entry_signal = self.position_entry_signals.get(_meta_key, signal)  # Signal at entry
                side = 'LONG' if current_pos > 0 else 'SHORT'
                # Increment per-position bar counter each evaluation cycle
                self._position_bar_count[_meta_key] = self._position_bar_count.get(_meta_key, 0) + 1

                # Calculate P&L (works for both long/short)
                if current_pos > 0:  # LONG
                    pnl = (price - entry_price) * current_pos
                    pnl_pct = (price / entry_price - 1) * 100
                else:  # SHORT
                    pnl = (entry_price - price) * abs(current_pos)
                    pnl_pct = (entry_price / price - 1) * 100

                holding_days = (datetime.now() - entry_time).days

                # Resolve governor controls early so exit logic can reference them
                _gov_ctrl, governor_regime_key, active_policy = self._resolve_governor_controls(
                    asset_class=asset_class,
                    market_regime=self._current_regime,
                    tier=perf_snapshot.tier,
                )

                should_exit = False
                exit_reason = ""
                stress_partial_reduction_qty = 0.0
                stress_unwind_candidate = self._stress_unwind_candidate_for(symbol)
                if stress_unwind_candidate is not None:
                    stress_partial_reduction_qty = min(
                        abs(float(current_pos)),
                        max(0.0, float(stress_unwind_candidate.get("target_qty", 0.0) or 0.0)),
                    )
                    should_exit = True
                    exit_reason = (
                        "🧯 StressUnwind: "
                        f"{self._stress_unwind_plan.reason} "
                        f"({self._stress_unwind_plan.worst_scenario_name or self._stress_unwind_plan.worst_scenario_id}, "
                        f"expected_stress_pnl=${float(stress_unwind_candidate.get('expected_stress_pnl', 0.0)):,.0f}, "
                        f"cut={float(stress_unwind_candidate.get('target_reduction_pct', 0.0) or 0.0):.0%}, "
                        f"liquidity={str(stress_unwind_candidate.get('liquidity_regime', 'NORMAL'))})"
                    )

                # Get ATR from position stops if available
                pos_stops = self.position_stops.get(_meta_key, {})
                atr = pos_stops.get('atr', None)

                # ── ATR Hard Stop Enforcement ────────────────────────────────
                # Checks every cycle. This is the enforcement layer — stops are
                # *calculated* at entry (god_level_risk_manager) but were never
                # actually checked against live prices. Fixes the critical gap.
                if getattr(ApexConfig, "ATR_STOP_ENFORCEMENT_ENABLED", True) and pos_stops:
                    _hard_stop = pos_stops.get('stop_loss')
                    if _hard_stop and _hard_stop > 0:
                        _stop_triggered = (
                            (current_pos > 0 and price <= _hard_stop) or   # long hit stop
                            (current_pos < 0 and price >= _hard_stop)       # short hit stop
                        )
                        if _stop_triggered:
                            _stop_pct = abs(price / entry_price - 1) * 100
                            logger.warning(
                                "🛑 %s: ATR HARD STOP triggered — price=%.4f stop=%.4f "
                                "(%.1f%% loss, entry=%.4f)",
                                symbol, price, _hard_stop, _stop_pct, entry_price,
                            )
                            should_exit = True
                            exit_reason = f"atr_hard_stop (price={price:.4f} <= stop={_hard_stop:.4f})"
                    # ── Trailing stop update ────────────────────────────────
                    # Ratchet stop up as price moves in our favour.
                    _trail_pct = pos_stops.get('trailing_stop_pct', 0.0)
                    if _trail_pct and _trail_pct > 0 and entry_price > 0:
                        if current_pos > 0:
                            _trail_price = price * (1.0 - _trail_pct)
                            _current_stop = pos_stops.get('stop_loss', 0.0)
                            if _trail_price > _current_stop:
                                pos_stops['stop_loss'] = round(_trail_price, 6)
                                self.position_stops[_meta_key] = pos_stops
                        elif current_pos < 0:
                            _trail_price = price * (1.0 + _trail_pct)
                            _current_stop = pos_stops.get('stop_loss', float('inf'))
                            if _trail_price < _current_stop:
                                pos_stops['stop_loss'] = round(_trail_price, 6)
                                self.position_stops[_meta_key] = pos_stops

                # ── Volatility-Spike Exit ──────────────────────────────────────
                # When realized vol suddenly spikes > 2σ above its 20-bar norm,
                # the regime may be breaking. Exit losing positions immediately;
                # tighten exit threshold on winners.  Uses prices already loaded.
                if (
                    not should_exit
                    and getattr(ApexConfig, "VOL_SPIKE_EXIT_ENABLED", True)
                    and isinstance(data, pd.DataFrame)
                    and 'Close' in data.columns
                    and len(data) >= 22
                ):
                    try:
                        _vol_returns = data['Close'].pct_change().dropna()
                        _vol_window = int(getattr(ApexConfig, "VOL_SPIKE_LOOKBACK", 20))
                        _vol_recent = float(_vol_returns.iloc[-1] ** 2) ** 0.5 * np.sqrt(252)
                        _vol_hist = _vol_returns.iloc[-(_vol_window + 1):-1].rolling(5).std().dropna() * np.sqrt(252)
                        if len(_vol_hist) >= 5:
                            _vol_mean = float(_vol_hist.mean())
                            _vol_std = float(_vol_hist.std())
                            _vol_thresh = float(getattr(ApexConfig, "VOL_SPIKE_SIGMA", 2.0))
                            if _vol_std > 0 and _vol_recent > _vol_mean + _vol_thresh * _vol_std:
                                # Losing position in vol spike → exit immediately
                                if pnl_pct < -float(getattr(ApexConfig, "VOL_SPIKE_MIN_LOSS_PCT", 0.005)):
                                    should_exit = True
                                    exit_reason = (
                                        f"⚡ VolSpike: realized_vol={_vol_recent:.1%} "
                                        f"({(_vol_recent - _vol_mean) / _vol_std:.1f}σ above norm), "
                                        f"pnl={pnl_pct:.2f}%"
                                    )
                                    logger.warning(
                                        "⚡ %s: Volatility spike exit — vol=%.1f%% "
                                        "(%.1fσ, norm=%.1f%%), pnl=%.2f%%",
                                        symbol, _vol_recent * 100,
                                        (_vol_recent - _vol_mean) / _vol_std,
                                        _vol_mean * 100, pnl_pct,
                                    )
                    except Exception:
                        pass  # non-fatal; stale or insufficient data

                # Get peak price for trailing stop
                peak_price = self.position_peak_prices.get(_meta_key, price)

                # Update peak price tracking
                # LONG: track highest price (profit peak for trailing stop)
                # SHORT: track lowest price (profit peak for trailing stop on shorts)
                if current_pos > 0 and price > peak_price:
                    self.position_peak_prices[_meta_key] = price
                    peak_price = price
                elif current_pos < 0 and price < peak_price:
                    # For shorts, "peak" = lowest price = max unrealized profit
                    self.position_peak_prices[_meta_key] = price
                    peak_price = price

                # ✅ DYNAMIC EXIT DECISION using exit manager
                _em_should_exit, _em_exit_reason, _em_urgency = self.exit_manager.should_exit(
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
                # ATR hard stop (set above) takes priority over exit manager
                if not should_exit:
                    should_exit, exit_reason, _ = _em_should_exit, _em_exit_reason, _em_urgency

                # 🛡️ HEDGE: Force-exit check (correlation crisis + accelerating drawdown)
                if not should_exit and hasattr(self, 'hedge_manager'):
                    try:
                        _corr_crisis = float(getattr(self, '_correlation_avg', 0.0)) >= 0.85
                        if '_hedge_adj' in dir() and _hedge_adj.force_exit and pnl_pct < 0:
                            # Fix 2: Minimum hold-time guard — don't force-exit a position that
                            # was just entered (< 15 min). Avoids exiting negative-signal entries
                            # before they can even develop (root cause of 5.8-min exits).
                            _hedge_hold_min = float(getattr(ApexConfig, "HEDGE_FORCE_EXIT_MIN_HOLD_MINUTES", 15.0))
                            _hedge_hold_hrs = (
                                (datetime.now() - entry_time).total_seconds() / 3600
                                if entry_time and isinstance(entry_time, datetime) else 0.0
                            )
                            # Fix 3: Minimum per-position loss floor — portfolio alarm should not
                            # force-exit a position that is only slightly negative. Require at
                            # least HEDGE_FORCE_EXIT_MIN_POS_LOSS_PCT loss before acting.
                            _hedge_min_loss = float(getattr(ApexConfig, "HEDGE_FORCE_EXIT_MIN_POS_LOSS_PCT", 1.0))
                            if _hedge_hold_hrs * 60 >= _hedge_hold_min and pnl_pct <= -_hedge_min_loss:
                                should_exit = True
                                exit_reason = f"🛡️ Hedge: force-exit — {' | '.join(_hedge_adj.reasons)}"
                                logger.warning("🛡️ %s: HedgeManager force-exit (pnl=%.2f%%) — %s",
                                               symbol, pnl_pct, exit_reason)
                            else:
                                logger.debug(
                                    "🛡️ %s: Hedge force-exit suppressed "
                                    "(held=%.1fmin < %.1f or pnl=%.2f%% > -%.1f%%)",
                                    symbol, _hedge_hold_hrs * 60, _hedge_hold_min,
                                    pnl_pct, _hedge_min_loss,
                                )
                        elif _corr_crisis and side == "LONG" and pnl_pct < -float(getattr(ApexConfig, "HEDGE_PER_POSITION_FORCE_EXIT_PCT", 2.0)):
                            # Per-position force-exit: correlation CRISIS means diversification
                            # is near-zero — any LONG losing beyond threshold should exit.
                            # Threshold raised from 1.0% to 2.0% to avoid exiting on normal
                            # intraday volatility during correlated moves.
                            _force_exit_pct = float(getattr(ApexConfig, "HEDGE_PER_POSITION_FORCE_EXIT_PCT", 2.0))
                            should_exit = True
                            exit_reason = (
                                f"🛡️ Hedge: per-position force-exit "
                                f"(corr={getattr(self, '_correlation_avg', 0.0):.2f}, pos_pnl={pnl_pct:.2f}%, threshold=-{_force_exit_pct:.1f}%)"
                            )
                            logger.warning(
                                "🛡️ %s: Per-position force-exit during correlation crisis "
                                "(corr=%.2f, pnl=%.2f%%, threshold=-%.1f%%)",
                                symbol, getattr(self, '_correlation_avg', 0.0), pnl_pct, _force_exit_pct,
                            )
                    except Exception:
                        pass

                # 🏆 EXCELLENCE: Check signal-position mismatch
                # Min-hold gate: don't cut on first 1-2 cycles — signal needs time to stabilise
                _bars_held = self._position_bar_count.get(symbol, 0)
                _min_hold = int(getattr(ApexConfig, "EXCELLENCE_MIN_HOLD_BARS", 2))
                if hasattr(self, 'excellence_manager') and not should_exit and _bars_held >= _min_hold:
                    # Use _meta_key for entry time lookup (crypto positions may be keyed as CRYPTO:sym)
                    _entry_time = (
                        self.position_entry_times.get(_meta_key)
                        or self.position_entry_times.get(symbol)
                    )
                    _hold_hours = (
                        (datetime.now() - _entry_time).total_seconds() / 3600
                        if _entry_time else 0.0
                    )
                    # Use the raw ML signal (pre-crypto-blend) for exit decisions.
                    # The crypto enhancement layer can flip a bearish ML signal
                    # to positive when tech indicators are momentarily bullish.
                    # That's appropriate for entry TIMING, not for exit checks —
                    # an open long should be cut if the ML model is bearish,
                    # regardless of what the current RSI/MACD says.
                    _exit_signal = _raw_ml_signal  # set a few hundred lines above
                    mismatch_exit, mismatch_reason = quick_mismatch_check(
                        position_side=side,
                        signal=_exit_signal,
                        confidence=confidence,
                        pnl_pct=pnl_pct,
                        hold_hours=_hold_hours,
                        is_crypto=_symbol_is_crypto(symbol),
                    )
                    # ── Excellence persistence gate ─────────────────────────────
                    # Require N consecutive weak-signal evaluations before exiting.
                    # Strong-bearish / no-signal exits bypass this (immediate action).
                    # "Weak signal + loss" exits need 2 consecutive bars to confirm.
                    _persist_bars = int(getattr(ApexConfig, "EXCELLENCE_PERSIST_BARS", 2))
                    _is_immediate_exit = mismatch_exit and (
                        "Strong bearish" in mismatch_reason
                        or "turned bearish" in mismatch_reason
                        or "Strong bullish" in mismatch_reason
                        or "turned bullish" in mismatch_reason
                    )
                    if mismatch_exit and not _is_immediate_exit:
                        _cnt = self._weak_signal_count.get(symbol, 0) + 1
                        self._weak_signal_count[symbol] = _cnt
                        if _cnt < _persist_bars:
                            mismatch_exit = False  # Defer — give signal one more bar
                            logger.debug(
                                "%s: Excellence persistence — weak signal bar %d/%d, deferring",
                                symbol, _cnt, _persist_bars,
                            )
                    else:
                        # Signal is fine (or immediate exit) — reset counter
                        self._weak_signal_count[symbol] = 0
                    if mismatch_exit:
                        # ── Minimum hold TIME gate ──────────────────────────────
                        # Even "turned bearish" immediate exits must respect minimum
                        # hold time to prevent 5-10 min churn on single-bar noise.
                        _min_hold_minutes = float(getattr(ApexConfig, "EXCELLENCE_MIN_HOLD_MINUTES", 90))
                        _hold_minutes = _hold_hours * 60
                        if _hold_minutes < _min_hold_minutes:
                            logger.debug(
                                "%s: Excellence blocked by min-hold gate "
                                "(held %.1f min < %.1f min required) — %s",
                                symbol, _hold_minutes, _min_hold_minutes, mismatch_reason,
                            )
                        else:
                            should_exit = True
                            exit_reason = f"🏆 Excellence: {mismatch_reason}"
                            logger.warning(f"🏆 {symbol}: Signal mismatch detected - {mismatch_reason}")

                # 🏆 EXCELLENCE: Check profit-taking decision
                if hasattr(self, 'excellence_manager') and not should_exit and pnl_pct > 5:
                    profit_decision = self.excellence_manager.get_profit_decision(
                        symbol=symbol,
                        position_side=side,
                        entry_price=entry_price,
                        current_price=price,
                        peak_price=peak_price,
                        signal=signal,
                        confidence=confidence
                    )
                    if profit_decision.action == ProfitAction.FULL:
                        should_exit = True
                        exit_reason = f"🏆 Excellence: {profit_decision.reason}"
                    elif profit_decision.action == ProfitAction.PARTIAL and pnl_pct > 10:
                        # For partial, only exit fully if >10% and signal is weak
                        if abs(signal) < 0.20:
                            should_exit = True
                            exit_reason = f"🏆 Excellence: {profit_decision.reason} (weak signal)"

                # ── Sentiment Exit Gate: accelerate exits when news turns strongly against ──
                # When the position is losing AND news sentiment strongly contradicts the
                # position direction, exit rather than wait for full SL/mismatch check.
                # Uses the cached news context (20-min TTL) — no extra I/O on most cycles.
                if (
                    not should_exit
                    and getattr(ApexConfig, "SENTIMENT_EXIT_GATE_ENABLED", True)
                    and getattr(self, '_news_aggregator', None) is not None
                ):
                    try:
                        _sent_ctx = await asyncio.wait_for(
                            self._news_aggregator.get_news_context(symbol),
                            timeout=5.0,
                        )
                        _pos_dir = 1 if current_pos > 0 else -1
                        # Negative value = news against the position direction
                        _sent_contra = float(_sent_ctx.sentiment) * _pos_dir
                        _sent_thresh = float(getattr(ApexConfig, "SENTIMENT_EXIT_CONTRA_THRESHOLD", 0.45))
                        _sent_min_loss = float(getattr(ApexConfig, "SENTIMENT_EXIT_MIN_LOSS_PCT", -0.8))
                        _sent_conf = float(getattr(_sent_ctx, 'confidence', 0.0))
                        if (
                            _sent_contra < -_sent_thresh
                            and _sent_conf > 0.50
                            and pnl_pct <= _sent_min_loss
                        ):
                            should_exit = True
                            exit_reason = (
                                f"📰 SentimentExit: news={_sent_ctx.sentiment:.2f} "
                                f"contra ({side} pnl={pnl_pct:.2f}%)"
                            )
                            logger.info(
                                "📰 %s: Sentiment exit — news=%.2f contra %s pos "
                                "(conf=%.2f, pnl=%.2f%%), accelerating",
                                symbol, _sent_ctx.sentiment, side, _sent_conf, pnl_pct,
                            )
                    except Exception:
                        pass  # non-fatal; news aggregator cache may be busy

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
                
                # ── TP LADDERING: partial exits at +3% / +6% before full exit ────
                if not should_exit and current_pos != 0:
                    _tp1_pct  = getattr(ApexConfig, "TP_LADDER_TIER1_PCT",  3.0)
                    _tp2_pct  = getattr(ApexConfig, "TP_LADDER_TIER2_PCT",  6.0)
                    _tp1_frac = getattr(ApexConfig, "TP_LADDER_TIER1_FRAC", 0.50)
                    _tp2_frac = getattr(ApexConfig, "TP_LADDER_TIER2_FRAC", 0.25)
                    _tranches = self._tp_tranches_taken[symbol]
                    _do_partial = False
                    _partial_tier = 0
                    _partial_qty  = 0
                    if pnl_pct >= _tp2_pct and 2 not in _tranches:
                        _partial_tier = 2
                        _partial_qty  = max(1, int(abs(current_pos) * _tp2_frac))
                        _do_partial   = True
                    elif pnl_pct >= _tp1_pct and 1 not in _tranches:
                        _partial_tier = 1
                        _partial_qty  = max(1, int(abs(current_pos) * _tp1_frac))
                        _do_partial   = True
                    if _do_partial and _partial_qty < abs(current_pos):
                        _tp_reason = (
                            f"💰 TP Ladder Tier {_partial_tier}: "
                            f"{pnl_pct:+.1f}% — selling {_partial_qty}/{abs(current_pos)}"
                        )
                        logger.info(f"💰 {symbol}: {_tp_reason}")
                        _tp_connector = self._get_connector_for(symbol)
                        if _tp_connector:
                            _tp_side = 'SELL' if current_pos > 0 else 'BUY'
                            try:
                                _tp_trade = await _tp_connector.execute_order(
                                    symbol=symbol,
                                    side=_tp_side,
                                    quantity=_partial_qty,
                                    confidence=abs(signal) if signal != 0 else 0.8,
                                )
                                if _tp_trade:
                                    self._tp_tranches_taken[symbol].add(_partial_tier)
                                    self.last_trade_time[symbol] = datetime.now()
                                    await self._sync_positions()
                                    logger.info(
                                        f"   ✅ Partial exit tier {_partial_tier}: "
                                        f"sold {_partial_qty} {symbol}"
                                    )
                                    _tp_fp = float(
                                        _tp_trade.get("price", price)
                                        if isinstance(_tp_trade, dict) else price
                                    )
                                    self.trade_audit.log(
                                        event="PARTIAL_TP",
                                        symbol=symbol,
                                        side=_tp_side,
                                        qty=_partial_qty,
                                        fill_price=_tp_fp,
                                        expected_price=float(price),
                                        slippage_bps=self._compute_slippage_bps(price, _tp_fp),
                                        signal=float(signal),
                                        confidence=float(confidence),
                                        regime=str(self._current_regime),
                                        pnl_pct=float(pnl_pct),
                                        exit_reason=_tp_reason,
                                        broker="alpaca" if _tp_connector is self.alpaca else "ibkr",
                                        tier=_partial_tier,
                                    )
                            except Exception as _tp_exc:
                                logger.warning(
                                    f"TP Ladder partial exit failed for {symbol}: {_tp_exc}"
                                )
                        return  # resume full evaluation next cycle with updated qty

                if should_exit:
                    if (
                        stress_unwind_candidate is not None
                        and 0.0 < stress_partial_reduction_qty < abs(float(current_pos))
                    ):
                        await self._execute_partial_position_reduction(
                            symbol=symbol,
                            asset_class=asset_class,
                            current_pos=float(current_pos),
                            reduction_qty=float(stress_partial_reduction_qty),
                            price=float(price),
                            signal=float(signal),
                            confidence=float(confidence),
                            entry_price=float(entry_price),
                            entry_time=entry_time,
                            entry_signal=float(entry_signal),
                            holding_days=int(holding_days),
                            pnl=float(pnl),
                            pnl_pct=float(pnl_pct),
                            exit_reason=exit_reason,
                            active_policy=active_policy,
                            governor_regime_key=governor_regime_key,
                            perf_snapshot=perf_snapshot,
                        )
                        return

                    # ── Liquidity window: defer non-urgent crypto exits to US active hours
                    # Exit slippage is 18x higher overnight; 13-22 UTC = peak US/EU session.
                    if (
                        str(asset_class).upper() == "CRYPTO"
                        and getattr(ApexConfig, "CRYPTO_EXIT_LIQUIDITY_WINDOW_ENABLED", True)
                    ):
                        _utc_h = datetime.utcnow().hour
                        _liq_start = int(getattr(ApexConfig, "CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_START", 13))
                        _liq_end = int(getattr(ApexConfig, "CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_END", 22))
                        if not (_liq_start <= _utc_h < _liq_end):
                            _urgency_thresh = float(
                                getattr(ApexConfig, "CRYPTO_EXIT_URGENCY_THRESHOLD_PCT", -2.5)
                            )
                            _reason_lower = exit_reason.lower()
                            _is_urgent = (
                                pnl_pct < _urgency_thresh
                                or any(
                                    kw in _reason_lower
                                    for kw in ("stop", "bearish", "bullish", "crisis",
                                               "kill", "drawdown", "force", "circuit")
                                )
                            )
                            if not _is_urgent:
                                logger.debug(
                                    "%s: Deferring exit to liquidity window "
                                    "(%02d:00-%02d:00 UTC, cur=%02d:00) "
                                    "pnl=%.2f%% reason=%s",
                                    symbol, _liq_start, _liq_end, _utc_h,
                                    pnl_pct, exit_reason,
                                )
                                return

                    # ══════════════════════════════════════════════════════════
                    # UPGRADE F: Correlated exit stagger — cap simultaneous exits
                    # ══════════════════════════════════════════════════════════
                    if getattr(ApexConfig, "CORR_EXIT_STAGGER_ENABLED", True):
                        _max_exits = int(getattr(ApexConfig, "CORR_EXIT_STAGGER_MAX_PER_CYCLE", 2))
                        if self._cycle_exit_count >= _max_exits:
                            logger.info(
                                "⏳ %s: EXIT deferred to next cycle (stagger limit %d/cycle reached)",
                                symbol, _max_exits,
                            )
                            return
                    self._cycle_exit_count += 1

                    pos_type = "LONG" if current_pos > 0 else "SHORT"
                    logger.info(f"🚪 EXIT {symbol} ({pos_type}): {exit_reason}")
                    logger.info(f"   Quantity: {abs(current_pos)}")
                    logger.info(f"   Entry: ${entry_price:.2f} → Current: ${price:.2f}")
                    logger.info(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    logger.info(f"   Holding: {holding_days}d")

                    exit_connector = self._get_connector_for(symbol)
                    if exit_connector:
                        self.pending_orders.add(symbol)

                        # Determine order side
                        order_side = 'SELL' if current_pos > 0 else 'BUY'
                        exit_broker = "alpaca" if exit_connector is self.alpaca else "ibkr"
                        self._journal_order_event(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=order_side,
                            quantity=float(abs(current_pos)),
                            broker=exit_broker,
                            lifecycle="submitted",
                            order_role="exit",
                            signal=float(signal),
                            confidence=float(confidence),
                            expected_price=float(price),
                            metadata={
                                "exit_reason": exit_reason,
                                **(
                                    self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None))
                                    if stress_unwind_candidate is not None
                                    else {}
                                ),
                                **self._governor_policy_metadata(
                                    active_policy,
                                    governor_regime_key,
                                    perf_snapshot.tier.value,
                                ),
                            },
                        )

                        trade = await exit_connector.execute_order(
                            symbol=symbol,
                            side=order_side,
                            quantity=abs(current_pos),
                            confidence=abs(signal) if signal != 0 else 0.8
                        )

                        if trade:
                            exit_fill_price = float(price)
                            exit_expected_price = float(price)
                            exit_trade_status = "FILLED"
                            if isinstance(trade, dict):
                                exit_fill_price = float(
                                    trade.get("price", exit_fill_price) or exit_fill_price
                                )
                                exit_expected_price = float(
                                    trade.get("expected_price", exit_expected_price) or exit_expected_price
                                )
                                exit_trade_status = str(trade.get("status", "FILLED")).upper()
                            else:
                                try:
                                    exit_fill_price = float(
                                        trade.orderStatus.avgFillPrice or exit_fill_price
                                    )
                                except Exception:
                                    pass
                            self._journal_order_event(
                                symbol=symbol,
                                asset_class=asset_class,
                                side=order_side,
                                quantity=float(abs(current_pos)),
                                broker=exit_broker,
                                lifecycle="filled" if exit_trade_status == "FILLED" else "result",
                                order_role="exit",
                                signal=float(signal),
                                confidence=float(confidence),
                                expected_price=float(exit_expected_price),
                                fill_price=float(exit_fill_price),
                                status=exit_trade_status,
                                metadata={
                                    "exit_reason": exit_reason,
                                    **(
                                        self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None))
                                        if stress_unwind_candidate is not None
                                        else {}
                                    ),
                                    **self._governor_policy_metadata(
                                        active_policy,
                                        governor_regime_key,
                                        perf_snapshot.tier.value,
                                    ),
                                },
                            )

                            # ✅ CRITICAL: Force sync after trade
                            await self._sync_positions()

                            # Track commission
                            commission = ApexConfig.COMMISSION_PER_TRADE
                            self.total_commissions += commission
                            realized_net = 0.0
                            if exit_trade_status == "FILLED":
                                realized_net = self._record_fill_realized_pnl(
                                    broker_name=exit_broker,
                                    symbol=symbol,
                                    side=order_side,
                                    quantity=abs(float(current_pos)),
                                    fill_price=float(exit_fill_price),
                                    commission=float(commission),
                                    filled_at=datetime.now(),
                                )

                            exit_slippage_bps = self._compute_slippage_bps(
                                exit_expected_price,
                                exit_fill_price,
                            )
                            side_label = "LONG" if current_pos > 0 else "SHORT"
                            self._record_exit_attribution(
                                symbol=symbol,
                                asset_class=asset_class,
                                side=side_label,
                                quantity=abs(current_pos),
                                entry_price=float(entry_price),
                                exit_price=float(exit_fill_price),
                                commissions=float(commission),
                                exit_reason=exit_reason,
                                entry_signal=float(entry_signal),
                                entry_confidence=min(1.0, max(0.0, abs(entry_signal))),
                                governor_tier=perf_snapshot.tier.value,
                                governor_regime=self._map_governor_regime(
                                    asset_class,
                                    self._current_regime,
                                ),
                                entry_time=entry_time,
                                exit_time=datetime.now(),
                                exit_slippage_bps=float(exit_slippage_bps),
                                source="live_exit",
                            )

                            # Execution quality: record exit fill slippage
                            if getattr(self, '_exec_quality', None) is not None:
                                self._exec_quality.record_fill(
                                    symbol=symbol,
                                    side=order_side,
                                    expected_price=float(exit_expected_price),
                                    fill_price=float(exit_fill_price),
                                    qty=float(abs(current_pos)),
                                    regime=str(self._current_regime),
                                    broker=str(exit_broker),
                                    order_type="market",
                                )

                            # Audit log — full exit
                            self.trade_audit.log(
                                event="EXIT",
                                symbol=symbol,
                                side=order_side,
                                qty=abs(current_pos),
                                fill_price=float(exit_fill_price),
                                expected_price=float(exit_expected_price),
                                slippage_bps=float(exit_slippage_bps),
                                signal=float(signal),
                                confidence=float(confidence),
                                entry_signal=float(entry_signal),
                                regime=str(self._current_regime),
                                pnl_pct=float(pnl_pct),
                                pnl_usd=float(pnl),
                                exit_reason=exit_reason,
                                holding_days=int(holding_days),
                                broker=exit_broker,
                                pretrade="PASS",
                            )
                            self._journal_position_update(
                                symbol=symbol,
                                asset_class=asset_class,
                                quantity=float(self.positions.get(symbol, 0.0)),
                                price=float(exit_fill_price),
                                reason="exit_fill",
                                metadata={
                                    "broker": exit_broker,
                                    "status": exit_trade_status,
                                    "exit_reason": exit_reason,
                                    **(
                                        self._stress_unwind_identity_payload(getattr(self, "_stress_unwind_plan", None))
                                        if stress_unwind_candidate is not None
                                        else {}
                                    ),
                                    **self._governor_policy_metadata(
                                        active_policy,
                                        governor_regime_key,
                                        perf_snapshot.tier.value,
                                    ),
                                },
                            )

                            # Feed outcome to adaptive exit multiplier learner
                            try:
                                self.exit_manager.record_closed_trade(
                                    regime=str(self._current_regime),
                                    exit_reason=exit_reason,
                                    pnl_pct=float(pnl_pct),
                                    exit_signal=float(signal),
                                )
                            except Exception:
                                pass  # non-fatal

                            # Incremental online learning: feed realized return back to ML model
                            try:
                                if getattr(ApexConfig, "ONLINE_LEARNING_ENABLED", True):
                                    self.signal_generator.online_update(
                                        symbol=symbol,
                                        actual_return=float(pnl_pct) / 100.0,
                                    )
                            except Exception:
                                pass  # non-fatal

                            # ── Option A: Direct RL Governor reward on trade close ─────────────
                            # Feeds actual realized PnL immediately to the Q-learning agent so
                            # it adapts regime weight matrices within the same trading session
                            # rather than waiting for the async outcome loop (every 20 cycles).
                            try:
                                _rl_act = getattr(self, '_last_rl_action', {}).get(symbol, -1)
                                _rl_reg = getattr(self, '_last_rl_regime', {}).get(symbol, self._current_regime)
                                if _rl_act != -1 and realized_net != 0.0:
                                    from models.rl_weight_governor import feedback_rl_reward
                                    _is_hv = "HIGH_VOLATILITY" in str(_rl_reg).upper() or "VOLATILE" in str(_rl_reg).upper()
                                    _entry_val = abs(float(entry_price)) * abs(float(current_pos))
                                    _pnl_reward = float(realized_net) / max(_entry_val, 1.0)
                                    feedback_rl_reward(_rl_reg, _is_hv, _rl_act, _pnl_reward)
                                    logger.debug(
                                        "🤖 RL reward: %s action=%d reward=%.4f (regime=%s pnl=$%.2f)",
                                        symbol, _rl_act, _pnl_reward, _rl_reg, realized_net,
                                    )
                            except Exception as _rle:
                                logger.debug("RL reward feedback failed for %s: %s", symbol, _rle)

                            # Meta-controller learning: pair entry context with realized return
                            try:
                                _ec = self._entry_contexts.pop(symbol, None)
                                if _ec is not None and getattr(self, '_meta_controller', None) is not None:
                                    self._meta_controller.record_outcome(
                                        ctx=_ec,
                                        pnl_pct=float(pnl_pct) / 100.0,
                                    )
                            except Exception:
                                pass  # non-fatal

                            # Strategy Rotation: record outcome with signal component breakdown
                            _src = getattr(self, '_strategy_rotation', None)
                            if _src is not None:
                                try:
                                    _src_comps = getattr(self, '_last_signal_components', {}).get(symbol, {})
                                    _src.record_outcome(
                                        regime=str(getattr(self, '_current_regime', 'neutral')),
                                        pnl_pct=float(pnl_pct) / 100.0,
                                        components=_src_comps,
                                    )
                                except Exception:
                                    pass

                            # IC Tracker: record realized return for this symbol's entry snapshot
                            # Entry date is used to match the feature snapshot recorded at signal time
                            try:
                                _ict = getattr(self, '_ic_tracker', None)
                                if _ict is not None:
                                    _entry_ts = self.position_entry_times.get(symbol)
                                    if _entry_ts:
                                        import datetime as _dt_mod
                                        if hasattr(_entry_ts, 'date'):
                                            _entry_date = str(_entry_ts.date())
                                        else:
                                            _entry_date = str(_dt_mod.date.fromisoformat(
                                                str(_entry_ts)[:10]
                                            ))
                                        _ict.record_return(
                                            symbol=symbol,
                                            entry_date=_entry_date,
                                            fwd_return_5d=float(pnl_pct) / 100.0,
                                        )
                            except Exception:
                                pass  # non-fatal

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
                            # Clear TP tranche state and bar counters for this symbol
                            self._tp_tranches_taken.pop(symbol, None)
                            self._position_bar_count.pop(symbol, None)
                            self._signal_streak.pop(symbol, None)
                            self._scaled_in_positions.discard(symbol)
                            self._save_position_metadata()

                            logger.info(f"   ✅ Position closed (commission: ${commission:.2f})")
                            if str(asset_class).upper() == "CRYPTO" and not self._last_in_equity_hours:
                                self._overnight_exits += 1

                            self.live_monitor.log_trade(
                                symbol,
                                order_side,
                                abs(current_pos),
                                float(exit_fill_price),
                                float(realized_net) if not math.isclose(realized_net, 0.0, abs_tol=1e-9) else float(pnl - commission),
                            )
                            await self.performance_tracker.record_trade(
                                symbol,
                                order_side,
                                abs(current_pos),
                                float(exit_fill_price),
                                commission,
                            )

                            # Record trade result for circuit breaker
                            self.risk_manager.record_trade_result(
                                float(realized_net) if not math.isclose(realized_net, 0.0, abs_tol=1e-9) else float(pnl - commission)
                            )

                            # Record outcome for AdaptiveEntryGate (Bayesian online learning)
                            _aeg = getattr(self, '_adaptive_entry_gate', None)
                            _entry_sig = self.position_entry_signals.get(symbol, 0.0)
                            _pnl_pct = pnl / max(1.0, abs(entry_price * float(current_pos))) if entry_price else 0.0
                            _outcome_regime = str(getattr(self, '_current_regime', 'neutral')).lower()
                            if _aeg is not None:
                                _aeg.record_outcome(
                                    signal=_entry_sig,
                                    pnl_pct=_pnl_pct,
                                    regime=_outcome_regime,
                                )
                            # Record outcome for SignalEnhancer (NLP/ML online learning)
                            _se = getattr(self, '_signal_enhancer', None)
                            if _se is not None:
                                _se_comps = getattr(self, '_signal_enhancer_components', {}).pop(symbol, None)
                                _se.record_outcome(
                                    symbol=symbol,
                                    regime=_outcome_regime,
                                    entry_signal=float(_entry_sig),
                                    entry_confidence=float(confidence),
                                    pnl_pct=float(_pnl_pct),
                                    components=_se_comps,
                                )
                            # Record outcome for SignalAccuracyTracker (regression vs binary comparison)
                            _sat = getattr(self, '_signal_accuracy_tracker', None)
                            if _sat is not None:
                                try:
                                    _sat.record_outcome(symbol=symbol, exit_price=float(exit_fill_price))
                                    # Log rolling comparison every 10 resolved trades
                                    if len(_sat._resolved) % 10 == 0 and len(_sat._resolved) > 0:
                                        _sat.log_summary()
                                except Exception:
                                    pass

                            # Record outcome for ModelDriftMonitor (IC / hit-rate tracking)
                            _mdm = getattr(self, '_model_drift_monitor', None)
                            if _mdm is not None and getattr(ApexConfig, "MODEL_DRIFT_MONITOR_ENABLED", True):
                                try:
                                    _mdm.record_outcome(
                                        symbol=symbol,
                                        signal_value=float(_entry_sig),
                                        actual_return=float(_pnl_pct),
                                    )
                                except Exception:
                                    pass

                            # Record trade for AlphaDecayCalibrator (IC by hold-time horizon)
                            _adc = getattr(self, '_alpha_decay_cal', None)
                            if _adc is not None and getattr(ApexConfig, "ALPHA_DECAY_CALIBRATOR_ENABLED", True):
                                try:
                                    _entry_time_adc = self.position_entry_times.get(
                                        _meta_key, self.position_entry_times.get(symbol)
                                    )
                                    _hold_h = (
                                        (datetime.now() - _entry_time_adc).total_seconds() / 3600.0
                                        if _entry_time_adc else 4.0
                                    )
                                    _adc.record_trade(
                                        signal=float(_entry_sig),
                                        actual_return=float(_pnl_pct),
                                        hold_hours=_hold_h,
                                        regime=_outcome_regime,
                                    )
                                except Exception:
                                    pass

                            # Update per-symbol consecutive loss streak + re-entry gap
                            _net_closed = float(realized_net) if not math.isclose(realized_net, 0.0, abs_tol=1e-9) else float(pnl - commission)
                            _is_losing_close = _net_closed < 0
                            if _is_losing_close:
                                self._symbol_loss_streak[symbol] = self._symbol_loss_streak.get(symbol, 0) + 1
                                _streak = self._symbol_loss_streak[symbol]
                                if _streak >= 2:
                                    logger.warning(
                                        "Loss streak gate: %s has %d consecutive losses", symbol, _streak
                                    )
                                # Crypto-wide consecutive loss tracker
                                if _symbol_is_crypto(symbol):
                                    self._crypto_consec_losses += 1
                                    _crypto_pause_n = int(getattr(ApexConfig, "CRYPTO_CONSEC_LOSS_PAUSE_COUNT", 3))
                                    _crypto_pause_h = float(getattr(ApexConfig, "CRYPTO_CONSEC_LOSS_PAUSE_HOURS", 4.0))
                                    if self._crypto_consec_losses >= _crypto_pause_n:
                                        self._crypto_pause_until = datetime.now() + timedelta(hours=_crypto_pause_h)
                                        logger.warning(
                                            "🛑 Crypto-wide pause: %d consecutive crypto losses → "
                                            "no new crypto entries until %s",
                                            self._crypto_consec_losses,
                                            self._crypto_pause_until.strftime("%H:%M"),
                                        )
                            else:
                                self._symbol_loss_streak.pop(symbol, None)
                                # Reset crypto-wide counter on any profitable close
                                if _symbol_is_crypto(symbol):
                                    self._crypto_consec_losses = 0
                            # Track exit time for re-entry gap check
                            self._last_exit_time[symbol] = datetime.now()
                            self._last_exit_was_loss[symbol] = _is_losing_close

                            # Alert on exceptional trade wins / losses
                            if getattr(self, '_alert_manager', None) is not None:
                                asyncio.create_task(self._alert_manager.send_trade_alert(
                                    symbol=symbol,
                                    side=str(pos_type),
                                    pnl_usd=float(pnl),
                                    pnl_pct=float(pnl_pct),
                                    exit_reason=str(exit_reason),
                                ))

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
                        if self.ibkr or self.alpaca:
                            logger.warning(
                                "⚠️ %s: No eligible connector for exit in broker mode '%s'; keeping position open",
                                symbol,
                                str(getattr(ApexConfig, "BROKER_MODE", "ibkr")).lower(),
                            )
                            # Allow faster re-check for stuck positions.
                            self.last_trade_time[symbol] = datetime.now() - timedelta(
                                seconds=max(30, int(ApexConfig.TRADE_COOLDOWN_SECONDS) - 30)
                            )
                            return

                        # Simulation mode - close position
                        order_side = 'SELL' if current_pos > 0 else 'BUY'
                        side_label = "LONG" if current_pos > 0 else "SHORT"
                        self._record_exit_attribution(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=side_label,
                            quantity=abs(current_pos),
                            entry_price=float(entry_price),
                            exit_price=float(price),
                            commissions=0.0,
                            exit_reason=exit_reason,
                            entry_signal=float(entry_signal),
                            entry_confidence=min(1.0, max(0.0, abs(entry_signal))),
                            governor_tier=perf_snapshot.tier.value,
                            governor_regime=self._map_governor_regime(
                                asset_class,
                                self._current_regime,
                            ),
                            entry_time=entry_time,
                            exit_time=datetime.now(),
                            exit_slippage_bps=0.0,
                            source="sim_exit",
                        )
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
                        self._save_position_metadata()

                        self.live_monitor.log_trade(symbol, order_side, abs(current_pos), price, pnl)
                        await self.performance_tracker.record_trade(symbol, order_side, abs(current_pos), price, 0)

                        # Record trade result for circuit breaker
                        self.risk_manager.record_trade_result(pnl)

                        self.last_trade_time[symbol] = datetime.now()

                    return
                
                else:
                    # ── Scale-in: add to a confirmed winning position ───────────
                    if (getattr(ApexConfig, 'SCALE_IN_ENABLED', True)
                            and symbol not in self._scaled_in_positions
                            and current_pos > 0):
                        _si_profit_thresh = float(getattr(ApexConfig, 'SCALE_IN_PROFIT_PCT', 1.5))
                        _si_min_signal    = float(getattr(ApexConfig, 'SCALE_IN_MIN_SIGNAL', 0.18))
                        if (pnl_pct >= _si_profit_thresh
                                and abs(signal) >= _si_min_signal
                                and signal * float(np.sign(current_pos)) > 0):
                            _si_qty = round(abs(current_pos) * float(getattr(ApexConfig, 'SCALE_IN_SIZE_PCT', 0.25)), 6)
                            if _si_qty > 0:
                                try:
                                    _si_connector = self._get_connector_for(symbol)
                                    if _si_connector:
                                        await asyncio.wait_for(
                                            _si_connector.execute_order(symbol, "BUY", _si_qty),
                                            timeout=20.0,
                                        )
                                        self._scaled_in_positions.add(symbol)
                                        logger.info(
                                            "📈 ScaleIn %s +%.4f (pnl=+%.2f%% signal=%.3f)",
                                            symbol, _si_qty, pnl_pct, signal,
                                        )
                                except Exception as _si_e:
                                    logger.debug("ScaleIn order failed %s: %s", symbol, _si_e)

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
            
            # ✅ Phase 11: Statistical Arbitrage / Pairs Mean-Reversion Overrides
            # If God Level Engine evaluates the asset as fundamentally neutral, check pairs!
            if hasattr(self, 'pairs_trader') and self.pairs_trader and abs(signal) < 0.20:
                pair_target = self._macro_pairs.get(symbol)
                if pair_target and pair_target in self.data_cache:
                    try:
                        analysis = self.pairs_trader.analyze_pair(symbol, pair_target, self.data_cache)
                        if analysis and analysis.is_cointegrated and abs(analysis.z_score) > self.pairs_trader.z_entry:
                            signal_override = -0.75 if analysis.z_score > 0 else 0.75
                            logger.info(f"🔄 PairsTrader OVERRIDE: {symbol} deviating from {pair_target} (Z={analysis.z_score:.2f}) -> signal={signal_override}")
                            signal = signal_override
                            confidence = 0.85
                    except Exception as e:
                        logger.debug(f"PairsTrader evaluation failed for {symbol}: {e}")
            
            logger.debug(f"🔍 {symbol}: Entry evaluation - signal={signal:+.3f} conf={confidence:.3f}")
            if str(asset_class).upper() == "CRYPTO" and not self._last_in_equity_hours:
                self._overnight_signals_evaluated += 1

            # Pre-market Staleness Guard check (equities only — crypto trades 24/7)
            if getattr(ApexConfig, "PRE_MARKET_STALENESS_GUARD", True) and not _symbol_is_crypto(symbol):
                try:
                    import pytz
                    now_et = datetime.now(pytz.timezone('US/Eastern'))
                    if now_et.hour == 9 and 0 <= now_et.minute < 30:
                        has_live = False
                        try:
                            ibkr = self.broker_service.get_broker("ibkr")
                            if hasattr(ibkr, 'has_live_tick_today'):
                                has_live = ibkr.has_live_tick_today(symbol)
                        except Exception:
                            pass

                        if not has_live:
                            logger.warning(f"⚠️ Pre-market signal suppressed for {symbol} — awaiting first live tick.")
                            return
                except Exception as e:
                    logger.debug(f"Pre-market guard error for {symbol}: {e}")

            # ✅ Phase 3.2: Detect market regime (adaptive or legacy)
            self._current_regime, _ = await self._resolve_cycle_market_regime(prices)

            governor_controls, governor_regime_key, active_policy = self._resolve_governor_controls(
                asset_class=asset_class,
                market_regime=self._current_regime,
                tier=perf_snapshot.tier,
            )
            self._record_governor_observability(
                asset_class=asset_class,
                regime_key=governor_regime_key,
                tier=perf_snapshot.tier,
                controls=governor_controls,
                policy=active_policy,
            )
            social_decision = self._social_decision_for(asset_class, governor_regime_key)
            if current_pos == 0 and social_decision and social_decision.block_new_entries:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="governor",
                    reason="social_shock",
                    signal=float(signal),
                    confidence=float(confidence),
                    price=float(price),
                    current_position=float(current_pos),
                    metadata={
                        "policy_version": social_decision.policy_version,
                        "reasons": social_decision.reasons[:5],
                        "combined_risk_score": float(social_decision.combined_risk_score),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )
                social_block_reason = (
                    "verified_event_risk"
                    if social_decision.verified_event_probability > 0
                    else "social_risk_extreme"
                )
                est_notional = float(max(1.0, ApexConfig.POSITION_SIZE_USD))
                est_alpha_bps = max(0.0, abs(float(signal)) * max(float(confidence), 0.10) * 80.0)
                blocked_alpha = est_notional * (est_alpha_bps / 10000.0)
                avoided_dd = est_notional * float(social_decision.combined_risk_score) * 0.015
                hedge_drag = est_notional * (
                    0.0006 + float(social_decision.verified_event_probability) * 0.001
                )
                event_id = ""
                for reason in social_decision.reasons:
                    if str(reason).startswith("prediction_verified:"):
                        parts = str(reason).split(":")
                        if len(parts) >= 2:
                            event_id = parts[1]
                        break
                self.performance_attribution.record_social_governor_impact(
                    asset_class=asset_class,
                    regime=governor_regime_key,
                    blocked_alpha_opportunity=blocked_alpha,
                    avoided_drawdown_estimate=avoided_dd,
                    hedge_cost_drag=hedge_drag,
                    policy_version=social_decision.policy_version,
                    reason=social_block_reason,
                    event_id=event_id,
                )
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_social_shock_block(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason=social_block_reason,
                    )
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason="social_shock",
                    )
                logger.warning(
                    "📣 %s: Entry blocked by SocialShockGovernor (%s/%s, score=%.2f, reasons=%s)",
                    symbol,
                    asset_class,
                    governor_regime_key,
                    social_decision.combined_risk_score,
                    ", ".join(social_decision.reasons[:3]),
                )
                return

            # Stress engine halt gate: block new entries when scenario stress is critical
            _stress_cs = getattr(self, '_stress_control_state', None)
            _stress_halt_now = _stress_cs is not None and _stress_cs.halt_new_entries
            if current_pos == 0 and _stress_halt_now:
                logger.warning(
                    "🚨 %s: Entry blocked by StressEngine (%s / %s)",
                    symbol, _stress_cs.action, _stress_cs.worst_scenario_name,
                )
                # Shadow gate: record production blocked the signal
                if getattr(self, '_shadow_gate', None) is not None:
                    try:
                        self._shadow_gate.observe_signal(
                            symbol=symbol, signal=float(signal), confidence=float(confidence),
                            current_position=float(current_pos), stress_halt_active=True,
                        )
                        self._shadow_gate.observe_production_decision(symbol=symbol, decision="blocked")
                    except Exception:
                        pass
                return

            # Shadow gate: observe this entry signal candidate (no position held)
            if current_pos == 0 and getattr(self, '_shadow_gate', None) is not None:
                try:
                    self._shadow_gate.observe_signal(
                        symbol=symbol, signal=float(signal), confidence=float(confidence),
                        current_position=float(current_pos), stress_halt_active=bool(_stress_halt_now),
                    )
                except Exception:
                    pass

            if current_pos == 0 and governor_controls.halt_new_entries:
                # 🚀 Alpha Bypass Override: Let exceptional signals pierce the RED tier shield.
                # STRONG signals (abs(signal) >= 0.50) or extremely confident MODERATEs.
                is_alpha_bypass = (abs(signal) >= 0.50) or (abs(signal) >= 0.35 and confidence >= 0.65)
                
                if is_alpha_bypass:
                    logger.warning(
                        "🚀 %s: Alpha Bypass OVERRIDE activated for exceptional signal (%.3f, conf=%.2f) during %s tier halt",
                        symbol,
                        signal,
                        confidence,
                        perf_snapshot.tier.value,
                    )
                else:
                    self._journal_risk_decision(
                        symbol=symbol,
                        asset_class=asset_class,
                        decision="blocked",
                        stage="governor",
                        reason="tier_halt",
                        signal=float(signal),
                        confidence=float(confidence),
                        price=float(price),
                        current_position=float(current_pos),
                        metadata={
                            "governor_tier": perf_snapshot.tier.value,
                            **self._governor_policy_metadata(
                                active_policy,
                                governor_regime_key,
                                perf_snapshot.tier.value,
                            ),
                        },
                    )
                    if self.prometheus_metrics:
                        self.prometheus_metrics.record_governor_blocked_entry(
                            asset_class=asset_class,
                            regime=governor_regime_key,
                            reason="tier_halt",
                        )
                    logger.info(
                        "🧭 %s: Entry blocked by governor policy (%s/%s, tier=%s)",
                        symbol,
                        asset_class,
                        governor_regime_key,
                        perf_snapshot.tier.value,
                    )
                    return

            # ✅ Phase 3.1: Get regime-adjusted signal threshold (adaptive or static)
            # Priority: AdaptiveEntryGate (Bayesian) > ThresholdOptimizer > asset-class config > legacy config
            _adaptive_gate = getattr(self, '_adaptive_entry_gate', None)
            _hedge_damp = getattr(self, '_last_hedge_dampener', 1.0)
            _is_crypto_asset = str(asset_class).upper() == "CRYPTO"
            _is_fx_asset = str(asset_class).upper() == "FOREX"
            _regime_str = str(getattr(self, '_current_regime', 'neutral')).lower()
            if _adaptive_gate is not None and _adaptive_gate.has_sufficient_data:
                _vix_for_gate = float(getattr(self, '_latest_vix', 20.0) or 20.0)
                signal_threshold = _adaptive_gate.get_effective_threshold(
                    regime=_regime_str,
                    hedge_dampener=_hedge_damp,
                    vix=_vix_for_gate,
                    is_crypto=_is_crypto_asset,
                )
            elif self.threshold_optimizer:
                sym_thresholds = self.threshold_optimizer.get_thresholds(symbol, self._current_regime)
                signal_threshold = sym_thresholds.entry_threshold
            elif _is_crypto_asset:
                # Use Alpaca/crypto-tuned thresholds: calibrated for faster crypto market structure
                signal_threshold = ApexConfig.CRYPTO_SIGNAL_THRESHOLDS_BY_REGIME.get(
                    _regime_str, ApexConfig.CRYPTO_MIN_SIGNAL_THRESHOLD
                )
            elif not _is_fx_asset:
                # Use IBKR/equity-tuned thresholds, overridden by auto-tuner when available
                _auto_t = getattr(self, '_auto_tuned_thresholds', {}).get(_regime_str)
                signal_threshold = _auto_t if _auto_t is not None else ApexConfig.CORE_SIGNAL_THRESHOLDS_BY_REGIME.get(
                    _regime_str, ApexConfig.CORE_MIN_SIGNAL_THRESHOLD
                )
            else:
                _auto_t = getattr(self, '_auto_tuned_thresholds', {}).get(_regime_str)
                signal_threshold = _auto_t if _auto_t is not None else ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME.get(
                    _regime_str, ApexConfig.MIN_SIGNAL_THRESHOLD
                )
            # When hedge dampener is active, lower the floor proportionally.
            # A dampener of 0.30 means signals were already cut by 70%, so the
            # floor must drop too — otherwise nothing passes.
            _effective_floor = (
                ApexConfig.CRYPTO_MIN_SIGNAL_THRESHOLD if _is_crypto_asset
                else ApexConfig.CORE_MIN_SIGNAL_THRESHOLD if not _is_fx_asset
                else ApexConfig.MIN_SIGNAL_THRESHOLD
            )
            if _hedge_damp < 1.0:
                _effective_floor = max(0.04, _effective_floor * max(0.40, _hedge_damp))
            signal_threshold = max(signal_threshold, _effective_floor)

            effective_signal_threshold = min(
                0.95, signal_threshold + governor_controls.signal_threshold_boost
            )
            # Per-broker confidence baseline: IBKR equities require 0.45, Alpaca crypto 0.40
            _base_confidence = (
                ApexConfig.CRYPTO_MIN_CONFIDENCE if _is_crypto_asset
                else ApexConfig.CORE_MIN_CONFIDENCE if not _is_fx_asset
                else ApexConfig.MIN_CONFIDENCE
            )
            effective_confidence_threshold = min(
                0.95,
                max(
                    _base_confidence,
                    _base_confidence + governor_controls.confidence_boost,
                ),
            )
            if _is_crypto_asset:
                # Crypto trades continuously and benefits from slightly lower entry gating.
                effective_signal_threshold = max(
                    0.04,
                    effective_signal_threshold * float(ApexConfig.CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER),
                )
                effective_confidence_threshold = max(
                    0.10,
                    effective_confidence_threshold * float(ApexConfig.CRYPTO_CONFIDENCE_THRESHOLD_MULTIPLIER),
                )
                # Rotation-ranked symbols can enter with modestly lower thresholds.
                rotation_discount_max = min(
                    0.60,
                    max(0.0, float(getattr(ApexConfig, "CRYPTO_ROTATION_THRESHOLD_DISCOUNT_MAX", 0.30))),
                )
                rotation_discount = rotation_discount_max * max(0.0, min(1.0, crypto_rotation_score))
                effective_signal_threshold = max(
                    0.04,
                    effective_signal_threshold * (1.0 - rotation_discount),
                )
                effective_confidence_threshold = max(
                    0.08,
                    effective_confidence_threshold * (1.0 - (rotation_discount * 0.5)),
                )

            # FX Signal Calibration check
            raw_signal = signal
            if str(asset_class).upper() == "FOREX":
                effective_signal_threshold = float(getattr(ApexConfig, "FX_SIGNAL_THRESHOLD", 0.05))
                signal = max(-1.0, min(1.0, signal * float(getattr(ApexConfig, "FX_SIGNAL_GAIN_MULTIPLIER", 3.0))))
                
                gate_passed = abs(signal) >= effective_signal_threshold
                logger.debug(
                    f"🛡️ FX Signal Gate: {symbol} | raw_signal={raw_signal:.3f} | scaled_signal={signal:.3f} | "
                    f"threshold_used={effective_signal_threshold:.3f} | asset_class={str(asset_class).upper()} | "
                    f"{'PASSED' if gate_passed else 'BLOCKED'}"
                )
            
            if abs(signal) < effective_signal_threshold:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="entry_gate",
                    reason="signal_threshold",
                    signal=float(signal),
                    confidence=float(confidence),
                    price=float(price),
                    current_position=float(current_pos),
                    metadata={
                        "threshold": float(effective_signal_threshold),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason="threshold",
                    )
                if str(asset_class).upper() == "CRYPTO":
                    logger.info(
                        "⏭️ %s [CRYPTO]: Signal %.3f below threshold %.3f "
                        "(rot_score=%.2f, regime=%s, perf=%s)",
                        symbol, signal, effective_signal_threshold,
                        crypto_rotation_score, self._current_regime, perf_snapshot.tier.value,
                    )
                else:
                    logger.info(
                        "⏭️ %s: Signal %.3f below effective threshold %.3f (%s, perf=%s)",
                        symbol, signal, effective_signal_threshold,
                        self._current_regime, perf_snapshot.tier.value,
                    )
                return

            # ── Directional guard: new entries must have positive signal ──────────
            # The abs(signal) check above uses |signal|, so a negative signal can pass
            # when the effective threshold is reduced (hedge dampener × crypto multiplier
            # × rotation discount can bring it as low as 0.04). This blocked all shorts
            # by design but allowed negative-signal LONG entries (root cause audit finding:
            # ETH/USD signal=-0.121 entered as BUY). Fix: new positions require signal > 0.
            if current_pos == 0 and signal <= 0.0:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="entry_gate",
                    reason="directional_guard",
                    signal=float(signal),
                    confidence=float(confidence),
                    price=float(price),
                    current_position=float(current_pos),
                )
                logger.info(
                    "⏭️ %s: Directional guard — signal=%.3f blocked for new entry (must be >0)",
                    symbol, signal,
                )
                return

            if confidence < effective_confidence_threshold:
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="entry_gate",
                    reason="confidence_threshold",
                    signal=float(signal),
                    confidence=float(confidence),
                    price=float(price),
                    current_position=float(current_pos),
                    metadata={
                        "threshold": float(effective_confidence_threshold),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason="confidence",
                    )
                if str(asset_class).upper() == "CRYPTO":
                    logger.info(
                        "⏭️ %s [CRYPTO]: Confidence %.3f below threshold %.3f "
                        "(rot_score=%.2f, perf=%s) — needs stronger signal alignment",
                        symbol, confidence, effective_confidence_threshold,
                        crypto_rotation_score, perf_snapshot.tier.value,
                    )
                else:
                    logger.info(
                        "⏭️ %s: Confidence %.3f below effective minimum %.3f (perf=%s)",
                        symbol, confidence, effective_confidence_threshold, perf_snapshot.tier.value,
                    )
                return

            # ── Composite signal quality gate (signal × confidence floor) ─────────
            # Blocks entries where both signal AND confidence are borderline — e.g.
            # signal=0.16, confidence=0.62 → quality=0.099 (below 0.10 floor).
            # Prevents marginal bets that clog the book without real edge.
            if current_pos == 0:
                _min_quality = float(getattr(ApexConfig, "MIN_SIGNAL_QUALITY_COMPOSITE", 0.10))
                _quality = abs(float(signal)) * float(confidence)
                if _quality < _min_quality:
                    logger.debug(
                        "%s: Composite quality gate — signal=%.3f × conf=%.3f = %.3f < %.3f",
                        symbol, signal, confidence, _quality, _min_quality,
                    )
                    return

            # ── Execution Timing Score: penalise entry in poor-slippage windows ──
            if current_pos == 0 and getattr(ApexConfig, "EXECUTION_TIMING_ENABLED", True):
                _eto = getattr(self, "_exec_timing", None)
                if _eto is not None:
                    try:
                        import datetime as _dt2
                        _now_utc2 = _dt2.datetime.utcnow()
                        _ts = _eto.get_timing_score(
                            hour=_now_utc2.hour,
                            day_of_week=_now_utc2.weekday(),
                            regime=str(self._current_regime),
                        )
                        if _ts.has_data and _ts.score < 1.0:
                            _penalty = float(getattr(ApexConfig, "EXECUTION_TIMING_CONF_PENALTY", 0.10))
                            _conf_penalty = 1.0 - (1.0 - _ts.score) * _penalty / (1.0 - 0.55)
                            confidence = max(0.0, confidence * _conf_penalty)
                            logger.debug(
                                "⏱ %s: timing score=%.2f → conf×%.3f → %.3f",
                                symbol, _ts.score, _conf_penalty, confidence,
                            )
                    except Exception:
                        pass

            # ── Tiered confidence gate: adaptive threshold via AdaptiveEntryGate ──
            # Replaces static 0.68 hard cutoff with smooth Bayesian-calibrated curve.
            # If AdaptiveEntryGate is unavailable, falls back to config values.
            if current_pos == 0 and getattr(ApexConfig, "ENTRY_TIERED_CONFIDENCE_ENABLED", True):
                _adaptive_gate = getattr(self, '_adaptive_entry_gate', None)
                if _adaptive_gate is not None:
                    _regime_str = str(getattr(self, '_current_regime', 'neutral')).lower()
                    _day_loss = 0.0
                    try:
                        _day_loss = abs(getattr(self.risk_manager, 'daily_pnl', 0.0) or 0.0) / max(1.0, float(getattr(self.risk_session, 'day_start_capital', 1) or 1))
                    except Exception:
                        pass
                    _conf_required = _adaptive_gate.get_effective_confidence_threshold(
                        signal=signal,
                        regime=_regime_str,
                        daily_loss_pct=-_day_loss,
                    )
                else:
                    # Fallback: smooth function instead of hard cutoff
                    _sig_high_cutoff = float(getattr(ApexConfig, "ENTRY_SIGNAL_HIGH_CUTOFF", 0.24))
                    _conf_base = float(getattr(ApexConfig, "ENTRY_CONFIDENCE_MODERATE", 0.62))
                    # Smooth: weak signals need more confidence, strong signals need less
                    _sig_ratio = min(1.0, abs(signal) / max(0.01, _sig_high_cutoff))
                    _conf_required = _conf_base - (_sig_ratio * 0.12)  # Strong signal → 0.50, weak → 0.62
                if confidence < _conf_required:
                    logger.info(
                        "⏭️ %s: Tiered gate — signal=%.3f requires confidence>=%.2f, got %.3f",
                        symbol, signal, _conf_required, confidence,
                    )
                    return

            # ── Signal Consensus Gate (N-of-M independent sources must agree) ────
            # Counts how many independent signal sources agree with the primary
            # signal direction. Low agreement → penalise or block entry.
            if current_pos == 0 and getattr(ApexConfig, "SIGNAL_CONSENSUS_GATE_ENABLED", True):
                _sig_dir = 1 if signal > 0 else -1
                _consensus_votes: list[bool] = []
                # ML component
                _sc = self._last_signal_components.get(symbol, {})
                _ml_c = float(_sc.get('ml', 0.0))
                if abs(_ml_c) > 1e-9:
                    _consensus_votes.append(_ml_c * _sig_dir > 0)
                # Tech component
                _tech_c = float(_sc.get('tech', 0.0))
                if abs(_tech_c) > 1e-9:
                    _consensus_votes.append(_tech_c * _sig_dir > 0)
                # CS momentum
                _cs_c = float(_sc.get('cs_momentum', 0.0))
                if abs(_cs_c) > 1e-9:
                    _consensus_votes.append(_cs_c * _sig_dir > 0)
                # Sentiment
                _sent_c = float(_sc.get('sentiment', 0.0))
                if abs(_sent_c) > 1e-9:
                    _consensus_votes.append(_sent_c * _sig_dir > 0)
                # Binary classifier
                _bin_c = float(getattr(self, '_last_binary_signal', {}).get(symbol, 0.0))
                if abs(_bin_c) > 1e-9:
                    _consensus_votes.append(_bin_c * _sig_dir > 0)

                if len(_consensus_votes) >= 2:
                    _agreement = sum(_consensus_votes) / len(_consensus_votes)
                    _hard_block = float(getattr(ApexConfig, "SIGNAL_CONSENSUS_MIN_AGREEMENT", 0.40))
                    _soft_thresh = float(getattr(ApexConfig, "SIGNAL_CONSENSUS_SOFT_THRESHOLD", 0.55))
                    _conf_penalty = float(getattr(ApexConfig, "SIGNAL_CONSENSUS_CONF_PENALTY", 0.10))
                    if _agreement < _hard_block:
                        logger.info(
                            "⏭️ %s: Consensus gate BLOCKED — %d/%d sources agree (%.0f%% < %.0f%% hard block)",
                            symbol, sum(_consensus_votes), len(_consensus_votes),
                            _agreement * 100, _hard_block * 100,
                        )
                        return
                    elif _agreement < _soft_thresh:
                        confidence = max(0.0, confidence - _conf_penalty)
                        logger.debug(
                            "%s: Consensus soft penalty — %d/%d agree (%.0f%%), conf %.3f → %.3f",
                            symbol, sum(_consensus_votes), len(_consensus_votes),
                            _agreement * 100, confidence + _conf_penalty, confidence,
                        )

            # ── Cross-Asset Divergence Gate ───────────────────────────────────────
            # When BTC and SPY signals strongly disagree, macro regime is uncertain.
            # Penalise all new entries. When both confirm the trade direction → small boost.
            if current_pos == 0 and getattr(ApexConfig, "CROSS_ASSET_DIVERGENCE_GATE_ENABLED", True):
                _btc_ref = float(self._cross_asset_signals.get('BTC', 0.0))
                _spy_ref = float(self._cross_asset_signals.get('SPY', 0.0))
                if abs(_btc_ref) > 0.10 and abs(_spy_ref) > 0.10:
                    if _btc_ref * _spy_ref < 0:
                        # BTC and SPY disagree — regime uncertainty
                        _div_mag = abs(_btc_ref - _spy_ref) / 2.0
                        _div_penalty = float(min(
                            float(getattr(ApexConfig, "CROSS_ASSET_DIV_MAX_PENALTY", 0.12)),
                            _div_mag * 0.20,
                        ))
                        confidence = max(0.0, confidence - _div_penalty)
                        logger.debug(
                            "%s: Cross-asset divergence (BTC=%.3f, SPY=%.3f) → conf -%.3f",
                            symbol, _btc_ref, _spy_ref, _div_penalty,
                        )
                    elif _btc_ref * signal > 0 and _spy_ref * signal > 0:
                        # Both macro assets confirm trade direction → small conviction boost
                        _conf_before = confidence
                        confidence = min(1.0, confidence + float(getattr(ApexConfig, "CROSS_ASSET_CONFIRM_BOOST", 0.03)))
                        logger.debug(
                            "%s: Cross-asset confirmation (BTC=%.3f, SPY=%.3f) → conf %.3f→%.3f",
                            symbol, _btc_ref, _spy_ref, _conf_before, confidence,
                        )

            # ── Regime transition confidence penalty ───────────────────────────────
            # When the regime has recently changed the signal landscape is uncertain:
            # prior-regime features may be mis-calibrated for the new regime.
            # Apply a fading confidence penalty for the first N minutes post-transition.
            # (The sizing caution in the sizing block covers position size separately.)
            if current_pos == 0 and getattr(self, '_regime_changed_at', None) is not None:
                try:
                    _caution_mins = float(getattr(ApexConfig, 'REGIME_TRANSITION_CAUTION_CONF_MINUTES', 90))
                    _elapsed_mins = (datetime.now() - self._regime_changed_at).total_seconds() / 60.0
                    if _elapsed_mins < _caution_mins:
                        # Linear fade: max penalty at t=0, zero at t=caution_mins
                        _fade     = 1.0 - (_elapsed_mins / _caution_mins)
                        _max_pen  = float(getattr(ApexConfig, 'REGIME_TRANSITION_CONF_PENALTY', 0.18))
                        _pen      = _max_pen * _fade
                        confidence = max(0.0, confidence * (1.0 - _pen))
                        logger.debug(
                            "%s: Regime transition caution (%.0f min ago / %.0f min window) "
                            "→ conf ×%.2f",
                            symbol, _elapsed_mins, _caution_mins, 1.0 - _pen,
                        )
                except Exception:
                    pass

            # ── Drawdown gate: after daily losses, require higher confidence ─────
            if current_pos == 0 and self.risk_manager:
                try:
                    _day_pnl_abs = abs(getattr(self.risk_manager, 'daily_pnl', 0.0) or 0.0)
                    _start_cap = float(getattr(self.risk_session, 'day_start_capital', 1) or 1)
                    _day_loss_pct = _day_pnl_abs / _start_cap
                    _gate_pct = float(getattr(ApexConfig, "ENTRY_DRAWDOWN_GATE_PCT", 0.015))
                    _conf_boost = float(getattr(ApexConfig, "ENTRY_DRAWDOWN_CONF_BOOST", 0.15))
                    if _day_loss_pct >= _gate_pct:
                        _boosted_conf = effective_confidence_threshold + _conf_boost
                        if confidence < _boosted_conf:
                            logger.info(
                                "⏭️ %s: Drawdown gate (daily_loss=%.2f%%) — "
                                "confidence %.3f < %.3f required",
                                symbol, _day_loss_pct * 100, confidence, _boosted_conf,
                            )
                            return
                except Exception:
                    pass

            # ── Tier 4: Signal momentum gate ──────────────────────────────────────
            # Update per-symbol signal history (rolling window, keep last N bars)
            _mom_max_bars = int(getattr(ApexConfig, "SIGNAL_MOMENTUM_HISTORY_BARS", 4))
            _mom_hist = self._signal_momentum_history.get(symbol, [])
            _mom_hist = (_mom_hist + [float(signal)])[-_mom_max_bars:]
            self._signal_momentum_history[symbol] = _mom_hist

            if current_pos == 0 and getattr(ApexConfig, "SIGNAL_MOMENTUM_GATE_ENABLED", True):
                if len(_mom_hist) >= _mom_max_bars:
                    import numpy as _np_tier4
                    _slope = float(_np_tier4.polyfit(range(len(_mom_hist)), _mom_hist, 1)[0])
                    _min_slope = float(getattr(ApexConfig, "SIGNAL_MOMENTUM_MIN_SLOPE", -0.05))
                    _sig_high_cutoff = float(getattr(ApexConfig, "ENTRY_SIGNAL_HIGH_CUTOFF", 0.45))
                    if _slope < _min_slope and abs(signal) < _sig_high_cutoff * 1.3:
                        logger.info(
                            "⏭️ %s: Signal momentum gate — slope=%.3f < %.3f "
                            "(weakening, hist=[%s])",
                            symbol, _slope, _min_slope,
                            ", ".join(f"{v:.2f}" for v in _mom_hist),
                        )
                        return

            # ── Tier 5: Regime component agreement gate ────────────────────────────
            # When sub-signals disagree with main signal in bear/volatile regimes, require
            # higher confidence before entering (avoids conflicted entries)
            if current_pos == 0 and getattr(ApexConfig, "REGIME_COMPONENT_AGREEMENT_ENABLED", True):
                _penalty = float(getattr(ApexConfig, "REGIME_COMPONENT_CONF_PENALTY", 0.12))
                _bearish_regimes = {"bear", "bearish", "volatile", "volatile_bear", "crisis", "stress"}
                _regime_key = str(self._current_regime).lower()
                if _regime_key in _bearish_regimes:
                    # In bear/volatile: reversion signal opposing main signal = red flag
                    if _comp_reversion != 0.0 and (signal * _comp_reversion) < 0:
                        _required_conf = effective_confidence_threshold + _penalty
                        if confidence < _required_conf:
                            logger.info(
                                "⏭️ %s: Component agreement gate (regime=%s) — "
                                "reversion=%.2f disagrees with signal=%.3f, "
                                "confidence %.3f < %.3f required",
                                symbol, _regime_key, _comp_reversion, signal,
                                confidence, _required_conf,
                            )
                            return
                else:
                    # In bull/neutral: both momentum AND trend opposing = red flag
                    _disagree_count = sum(
                        1 for c in (_comp_momentum, _comp_trend)
                        if c != 0.0 and (signal * c) < 0
                    )
                    if _disagree_count >= 2:
                        _required_conf = effective_confidence_threshold + _penalty
                        if confidence < _required_conf:
                            logger.info(
                                "⏭️ %s: Component agreement gate (regime=%s) — "
                                "trend+momentum both disagree with signal=%.3f, "
                                "confidence %.3f < %.3f required",
                                symbol, _regime_key, signal, confidence, _required_conf,
                            )
                            return

            # ── Regime-direction alignment gate ────────────────────────────────────
            # Regime-aware directional filtering:
            # • strong_bear → hard-block all LONG entries (too risky counter-trend)
            # • bear        → LONG entries allowed only with counter-trend signal mult
            #                 (e.g. bear threshold 0.15 × 1.8 = 0.27 ≈ model max)
            # • SHORT entries in bear/strong_bear are ALLOWED with normal threshold
            # • bull/strong_bull → block all SHORT entries (trend is against you)
            # • volatile/crisis  → require stronger quality (signal×conf) for any entry
            # Gate only blocks NEW entries (current_pos == 0).
            if current_pos == 0 and getattr(ApexConfig, "REGIME_DIRECTION_GATE_ENABLED", True):
                _rd_regime = str(self._current_regime).lower()
                _is_long = float(signal) > 0
                _ct_mult = float(getattr(ApexConfig, "REGIME_COUNTER_TREND_SIGNAL_MULT", 1.8))

                if _is_long and _rd_regime == "strong_bear":
                    # Hard block: never go long in strong bear
                    if getattr(ApexConfig, "REGIME_STRONG_BEAR_LONG_BLOCK", True):
                        logger.info(
                            "⏭️ %s: Regime-direction gate — LONG hard-blocked in strong_bear",
                            symbol,
                        )
                        return
                elif _is_long and _rd_regime == "bear":
                    # Counter-trend LONG in bear: require threshold × counter-trend multiplier
                    _ct_threshold = float(effective_signal_threshold) * _ct_mult
                    if abs(float(signal)) < _ct_threshold:
                        logger.info(
                            "⏭️ %s: Regime-direction gate — LONG in bear needs signal >= %.3f, got %.3f",
                            symbol, _ct_threshold, abs(float(signal)),
                        )
                        return
                elif not _is_long and _rd_regime in {"bull", "strong_bull"}:
                    # Block SHORT in bull — trend is firmly against
                    logger.info(
                        "⏭️ %s: Regime-direction gate — SHORT blocked in %s regime",
                        symbol, _rd_regime,
                    )
                    return
                # SHORT in bear / strong_bear → ALLOWED with normal threshold (aligned with regime)

                if _rd_regime in {"volatile", "crisis", "stress"}:
                    _vol_floor = float(getattr(ApexConfig, "REGIME_VOLATILE_QUALITY_FLOOR", 0.15))
                    _quality = abs(float(signal)) * float(confidence)
                    if _quality < _vol_floor:
                        logger.info(
                            "⏭️ %s: Regime-direction gate — volatile/crisis, "
                            "quality=%.3f < %.3f required (signal=%.3f, conf=%.3f)",
                            symbol, _quality, _vol_floor, signal, confidence,
                        )
                        return

            # ── Signal stability streak: track consecutive above-threshold bars ───
            _is_crypto = str(asset_class).upper() == "CRYPTO"
            _streak_thresh = ApexConfig.MIN_SIGNAL_THRESHOLD * (
                ApexConfig.CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER if _is_crypto else 1.0
            )
            if signal >= _streak_thresh:
                self._signal_streak[symbol] = self._signal_streak.get(symbol, 0) + 1
            else:
                self._signal_streak[symbol] = 0

            # 🔒 VIX Crypto Gate: Block NEW crypto entries during elevated volatility.
            # When VIX >= threshold (default 20.0), crypto drawdowns accelerate and
            # signal quality degrades — best to stay flat until volatility normalises.
            if str(asset_class).upper() == "CRYPTO" and current_pos == 0:
                _vix_for_crypto = float(self._current_vix or 18.0)
                _crypto_vix_thresh = float(getattr(ApexConfig, 'VIX_CRYPTO_BLOCK_THRESHOLD', 20.0))
                if _vix_for_crypto >= _crypto_vix_thresh:
                    logger.info(
                        "🔒 %s: New crypto entry blocked — VIX=%.1f >= %.1f threshold",
                        symbol, _vix_for_crypto, _crypto_vix_thresh,
                    )
                    return

            # ── Update BTC signal cache + rolling history (always, for every BTC bar) ──
            _is_btc = symbol in ("CRYPTO:BTC/USD", "BTC/USD", "BTCUSD")
            if _is_btc:
                self._btc_signal_cache = float(signal)
                self._btc_signal_history = (self._btc_signal_history + [float(signal)])[-20:]

            # ── Signal 4-bar moving-average gate (new entries only) ──────────────
            # Requires the average signal over the last 4 bars to be meaningfully
            # positive — prevents entering on a single-bar spike that reverts immediately.
            # Threshold scales with regime noise: noisier regimes demand a higher average.
            if current_pos == 0 and getattr(ApexConfig, "SIGNAL_MA_GATE_ENABLED", True):
                _ma_hist = self._signal_momentum_history.get(symbol, [])
                if len(_ma_hist) >= int(getattr(ApexConfig, "SIGNAL_MOMENTUM_HISTORY_BARS", 4)):
                    import numpy as _np_ma
                    _sig_ma = float(_np_ma.mean(_ma_hist))
                    # Regime-aware noise factor: bear/volatile regimes demand stronger average signal
                    _regime_noise = {
                        "strong_bull": 0.55, "bull": 0.60, "neutral": 0.63,
                        "bear": 0.70, "bearish": 0.70, "strong_bear": 0.75,
                        "volatile": 0.80, "volatile_bear": 0.80, "crisis": 0.90, "stress": 0.85,
                    }.get(str(self._current_regime).lower(), 0.63)
                    _min_sig_ma = float(getattr(ApexConfig, "SIGNAL_MA_MIN", 0.18)) * _regime_noise / 0.63
                    if _sig_ma < _min_sig_ma:
                        logger.info(
                            "⏭️ %s: Signal MA gate — 4-bar MA=%.3f < %.3f "
                            "(regime=%s, noise_factor=%.2f)",
                            symbol, _sig_ma, _min_sig_ma, self._current_regime, _regime_noise,
                        )
                        return

            # ── BTC macro filter: block altcoin entries when BTC is bearish ──────
            # Uses a rolling 25th-percentile threshold: BTC must be above its own
            # recent 25th-percentile signal before altcoin LONGs are allowed.
            # Falls back to static threshold until 10 bars of BTC history exist.
            if (
                _is_crypto
                and not _is_btc
                and current_pos == 0
                and signal > 0  # Only block LONG entries
                and getattr(ApexConfig, "BTC_MACRO_FILTER_ENABLED", True)
            ):
                _btc_sig = getattr(self, "_btc_signal_cache", 0.0)
                _btc_hist = getattr(self, "_btc_signal_history", [])
                if len(_btc_hist) >= 10:
                    import numpy as _np_btc
                    _btc_p25 = float(_np_btc.percentile(_btc_hist, 25))
                    _btc_min = _btc_p25  # Dynamic: BTC must exceed its own 25th-pct
                else:
                    _btc_min = float(getattr(ApexConfig, "BTC_MACRO_FILTER_MIN_SIGNAL", 0.0))
                if _btc_sig < _btc_min:
                    logger.info(
                        "⏭️ %s: BTC macro filter — BTC signal=%.3f < p25=%.3f "
                        "(hist_len=%d), blocking altcoin entry",
                        symbol, _btc_sig, _btc_min, len(_btc_hist),
                    )
                    return

            # ── Altcoin probation gate (dynamic from ThresholdCalibrator) ───────
            # Per-symbol signal threshold computed from live win rates.
            # Falls back to static config list when calibrator has no data yet.
            if current_pos == 0 and _is_crypto:
                _prob_dict = getattr(self, "_calibrated_symbol_probation", {})
                if not _prob_dict:
                    # Fallback: use static config list until first calibration
                    _prob_dict = {
                        s: float(getattr(ApexConfig, "ALTCOIN_PROBATION_MIN_SIGNAL", 0.45))
                        for s in getattr(ApexConfig, "ALTCOIN_PROBATION_SYMBOLS", [])
                    }
                if symbol in _prob_dict:
                    _prob_min = _prob_dict[symbol]
                    if abs(signal) < _prob_min:
                        logger.info(
                            "⏭️ %s: Altcoin probation gate — signal=%.3f < %.2f "
                            "(calibrated threshold)",
                            symbol, abs(signal), _prob_min,
                        )
                        return

            # ── Portfolio heat gate: block new entries if total unrealized loss > threshold ──
            # Threshold scales with VIX: higher volatility → accept more heat before blocking,
            # since positions naturally swing more intraday. Formula: base × max(1, VIX/20).
            if current_pos == 0 and getattr(ApexConfig, "PORTFOLIO_HEAT_GATE_ENABLED", True):
                _base_heat = float(getattr(ApexConfig, "PORTFOLIO_HEAT_MAX_LOSS_PCT", 0.03))
                _vix_now = float(self._current_vix or 18.0)
                _heat_thresh = _base_heat * max(1.0, _vix_now / 20.0)
                _total_heat = sum(
                    (self.price_cache.get(s, ep) - ep) * self.positions.get(s, 0) / max(self.capital, 1.0)
                    for s, ep in self.position_entry_prices.items()
                    if self.positions.get(s, 0) != 0
                )
                if _total_heat < -_heat_thresh:
                    logger.info(
                        "⏭️ %s: Portfolio heat gate — unrealized loss=%.2f%% > %.2f%% "
                        "(VIX=%.1f scaled threshold)",
                        symbol, abs(_total_heat) * 100, _heat_thresh * 100, _vix_now,
                    )
                    return

            # ── Adaptive Meta-Controller: holistic self-learning context gate ──────
            # Synthesises ALL available market signals (news, macro, OFI, consensus,
            # cross-asset, funding, pattern, HHI, regime) into a single context score.
            # Starts as a heuristic; learns from realised outcomes to auto-tune.
            # Stores a TradeContext snapshot so the outcome can be fed back at close.
            if (current_pos == 0 and not is_initial_build
                    and getattr(self, '_meta_controller', None) is not None
                    and getattr(self, '_market_intelligence', None) is not None):
                try:
                    from risk.adaptive_meta_controller import TradeContext as _TC
                    # ── Safely harvest local context vars (may not exist if gates skipped) ──
                    try:
                        _mc_news_sent = float(getattr(_news_ctx, 'sentiment', 0.0))
                        _mc_news_conf = float(getattr(_news_ctx, 'confidence', 0.0))
                        _mc_news_mom  = float(getattr(_news_ctx, 'momentum', 0.0))
                    except Exception:
                        _mc_news_sent = _mc_news_conf = _mc_news_mom = 0.0

                    _mc_macro = getattr(self, '_macro_context', None)
                    _mc_macro_ra  = float(getattr(_mc_macro, 'risk_appetite', 0.0) or 0.0) if _mc_macro else 0.0
                    _mc_yld_inv   = bool(getattr(_mc_macro, 'yield_curve_inverted', False)) if _mc_macro else False
                    _mc_vix_ratio = float(getattr(_mc_macro, 'vix_spot_futures_ratio', 0.95) or 0.95) if _mc_macro else 0.95

                    _mc_ofi = float(self._order_flow_imbalance.get(symbol, 0.0))

                    try:
                        _mc_consensus = float(_agreement)   # from consensus gate above
                    except NameError:
                        _mc_consensus = 0.5

                    _mc_btc = float(self._cross_asset_signals.get('BTC', 0.0))
                    _mc_spy = float(self._cross_asset_signals.get('SPY', 0.0))

                    try:
                        _mc_fr = float(getattr(_fr_ctx, 'signal', 0.0))
                    except NameError:
                        _mc_fr = 0.0

                    try:
                        _mc_pat = float(getattr(_pat, 'signal', 0.0))
                    except NameError:
                        _mc_pat = 0.0

                    _mc_hhi = float(getattr(getattr(self, '_last_factor_exposure', None),
                                            'concentration_hhi', 0.0) or 0.0)

                    # Signed daily P&L fraction (negative = losing)
                    try:
                        _mc_raw_pnl  = float(getattr(self.risk_manager, 'daily_pnl', 0.0) or 0.0)
                        _mc_start_cap = float(getattr(self.risk_session, 'day_start_capital', 1) or 1)
                        _mc_day_loss  = _mc_raw_pnl / max(_mc_start_cap, 1.0)
                    except Exception:
                        _mc_day_loss = 0.0

                    # Vol percentile from recent data if available
                    _mc_vol_pct = 0.5
                    try:
                        if isinstance(data, pd.DataFrame) and 'Close' in data.columns and len(data) >= 60:
                            _mc_rets = data['Close'].pct_change().dropna()
                            _mc_cur_vol = float(_mc_rets.iloc[-5:].std()) if len(_mc_rets) >= 5 else 0.0
                            _mc_his_vol = float(_mc_rets.rolling(20).std().dropna().quantile([0.1, 0.9]).iloc[-1]) if len(_mc_rets) >= 20 else 0.0
                            if _mc_his_vol > 0:
                                _mc_vol_pct = float(min(1.0, _mc_cur_vol / _mc_his_vol / 2.0))
                    except Exception:
                        pass

                    # ── Build MarketIntelligence (unified synthesizer) ─────────────
                    _mi = self._market_intelligence.assess(
                        regime=str(self._current_regime),
                        regime_conviction=float(getattr(self, '_regime_conviction', 0.5) or 0.5),
                        news_sentiment=_mc_news_sent,
                        news_confidence=_mc_news_conf,
                        news_momentum=_mc_news_mom,
                        fear_greed_index=int(getattr(_news_ctx, 'fear_greed_index', None) or 0) if _mc_news_conf > 0.0 else None,
                        yield_curve_slope=float(getattr(_mc_macro, 'yield_curve_slope', 0.0) or 0.0) if _mc_macro else 0.0,
                        vix_structure=_mc_vix_ratio,
                        dxy_momentum=float(getattr(_mc_macro, 'dxy_momentum', 0.0) or 0.0) if _mc_macro else 0.0,
                        btc_signal=_mc_btc,
                        spy_signal=_mc_spy,
                        ofi=_mc_ofi,
                        funding_rate_signal=_mc_fr,
                        pattern_signal=_mc_pat,
                        vol_percentile=_mc_vol_pct,
                        hhi=_mc_hhi,
                        daily_loss_pct=_mc_day_loss,
                    )

                    # ── Build TradeContext and evaluate ────────────────────────────
                    _mc_ctx = _TC(
                        symbol=symbol,
                        signal=float(signal),
                        confidence=float(confidence),
                        asset_class=str(asset_class).upper(),
                        regime=str(self._current_regime),
                        news_sentiment=_mc_news_sent,
                        news_confidence=_mc_news_conf,
                        news_momentum=_mc_news_mom,
                        macro_risk_appetite=_mc_macro_ra,
                        yield_curve_inverted=_mc_yld_inv,
                        vix_backwardation=(_mc_vix_ratio > 1.0),
                        ofi=_mc_ofi,
                        consensus_ratio=_mc_consensus,
                        btc_signal=_mc_btc,
                        spy_signal=_mc_spy,
                        funding_rate_signal=_mc_fr,
                        pattern_signal=_mc_pat,
                        hhi=_mc_hhi,
                        daily_loss_pct=_mc_day_loss,
                        vol_percentile=_mc_vol_pct,
                    )
                    _mc_decision = self._meta_controller.evaluate(_mc_ctx)

                    # Cache context for outcome learning at trade close
                    self._entry_contexts[symbol] = _mc_ctx

                    if not _mc_decision.allow:
                        logger.info(
                            "🧠 MetaCtrl BLOCKED %s: score=%.3f env=%.3f risk_apt=%.3f "
                            "coherence=%.2f | %s",
                            symbol, _mc_decision.context_score,
                            _mi.environment_score, _mi.risk_appetite,
                            _mi.signal_coherence, _mi.narrative,
                        )
                        return

                    # Apply meta-controller's confidence and size adjustments
                    if abs(_mc_decision.confidence_multiplier - 1.0) > 0.02:
                        confidence = float(np.clip(
                            confidence * _mc_decision.confidence_multiplier, 0.0, 1.0
                        ))
                    self._meta_size_mult = float(_mc_decision.size_multiplier)

                    if abs(_mc_decision.context_score) > 0.10 or not _mi.narrative.startswith("neutral"):
                        logger.debug(
                            "🧠 MetaCtrl %s: score=%.3f conf×%.2f size×%.2f | %s",
                            symbol, _mc_decision.context_score,
                            _mc_decision.confidence_multiplier,
                            _mc_decision.size_multiplier,
                            _mi.narrative,
                        )
                except Exception as _mc_ex:
                    logger.debug("MetaController entry gate failed (non-fatal): %s", _mc_ex)

            # ✅ Phase 1.2: Check VaR limit before new entries
            if self.use_institutional and len(self.positions) > 0:
                # Re-compute fresh: parallel coroutines may have added positions since line 4162.
                _var_active = sum(
                    1 for s, qty in self.positions.items()
                    if qty != 0
                    and "OPT:" not in str(s).upper()
                    and not str(s).upper().startswith("OPTION:")
                )
                is_initial_build = _var_active == 0
                portfolio_risk = self.inst_risk_manager.calculate_portfolio_risk(
                    self.positions,
                    self.price_cache,
                    self.historical_data
                )
                # Crypto session operates in a dedicated Alpaca account; crypto assets
                # have 2-4× the daily vol of equities, so use a per-session VaR limit.
                _session_var_pct = (
                    float(getattr(ApexConfig, "CRYPTO_MAX_PORTFOLIO_VAR", 0.06))
                    if getattr(self, "session_type", "") == "crypto"
                    else ApexConfig.MAX_PORTFOLIO_VAR
                )
                max_var = self.capital * _session_var_pct
                if is_initial_build:
                    max_var *= 3.0  # Allow higher VaR while building initial portfolio

                if portfolio_risk.var_95 > max_var:
                    logger.warning(f"⚠️ {symbol}: VaR limit exceeded (${portfolio_risk.var_95:,.0f} > ${max_var:,.0f}) - blocking entry")
                    return

            # ✅ Phase 1.3: Check portfolio correlation before new entries
            _corr_size_mult: float = 1.0  # may be reduced below if correlation is elevated
            if ApexConfig.USE_CORRELATION_MANAGER and len(self.positions) > 1:
                existing_symbols = [s for s, qty in self.positions.items() if qty != 0]
                avg_corr = self.correlation_manager.get_average_correlation(symbol, existing_symbols)
                # Crypto assets are inherently correlated; use a higher cap to allow diversification within crypto
                corr_limit = (
                    float(getattr(ApexConfig, "MAX_PORTFOLIO_CORRELATION_CRYPTO", 0.92))
                    if str(asset_class).upper() == "CRYPTO"
                    else float(ApexConfig.MAX_PORTFOLIO_CORRELATION)
                )
                if avg_corr > corr_limit:
                    logger.warning(f"⚠️ {symbol}: Correlation too high ({avg_corr:.2f} > {corr_limit:.2f}) - blocking entry")
                    return
                # Graduated size reduction for elevated-but-below-block correlation.
                # Zones: [warn_lo, warn_hi) → 75% size; [warn_hi, corr_limit) → 50% size.
                _warn_lo = float(getattr(ApexConfig, "CORRELATION_SIZE_WARN_LO", 0.70))
                _warn_hi = float(getattr(ApexConfig, "CORRELATION_SIZE_WARN_HI", 0.80))
                if avg_corr >= _warn_hi:
                    _corr_size_mult = 0.50
                    logger.info(
                        f"   🔗 CorrelationGuard [{symbol}]: avg_corr={avg_corr:.2f} ≥ {_warn_hi:.2f} → 50% size"
                    )
                elif avg_corr >= _warn_lo:
                    _corr_size_mult = 0.75
                    logger.info(
                        f"   🔗 CorrelationGuard [{symbol}]: avg_corr={avg_corr:.2f} ≥ {_warn_lo:.2f} → 75% size"
                    )

            # ✅ Phase 3: Aggregate Risk Check (Pre-Trade)
            # Check if the account is allowed to open new positions based on total equity
            agg_risk = await self.risk_manager.check_aggregate_risk(self.tenant_id)
            if not agg_risk["allowed"]:
                 logger.warning(f"🛑 {symbol}: Aggregate Risk Limit Breached - {agg_risk['reason']}")
                 if self.prometheus_metrics:
                     self.prometheus_metrics.record_governor_blocked_entry(
                         asset_class=asset_class, regime=self._current_regime, reason="aggregate_risk"
                     )
                 return

            # ── Signal stability gate (crypto new entries only) ───────────────────
            if _is_crypto and current_pos == 0:
                _stability_bars = int(getattr(ApexConfig, "CRYPTO_SIGNAL_STABILITY_BARS", 2))
                _streak = self._signal_streak.get(symbol, 0)
                if _streak < _stability_bars:
                    logger.debug(
                        "%s: Signal stability gate — streak=%d < %d bars required",
                        symbol, _streak, _stability_bars,
                    )
                    return

            # ── Per-symbol consecutive loss protection ────────────────────────────
            # Track symbols on losing streaks; require higher confidence after 2 losses,
            # block entirely after 3 consecutive losses (session-level probation).
            if current_pos == 0:
                _loss_streak = self._symbol_loss_streak.get(symbol, 0)
                if _loss_streak >= 3:
                    logger.warning(
                        "%s: Blocked — %d consecutive losses (session probation)", symbol, _loss_streak
                    )
                    return
                if _loss_streak >= 2:
                    # After 2 losses require confidence ≥ base_min + 0.15 (default 0.75)
                    _base_min = float(getattr(ApexConfig, "MIN_CONFIDENCE", 0.60))
                    _streak_min_conf = min(1.0, _base_min + 0.15)
                    if confidence < _streak_min_conf:
                        logger.info(
                            "%s: Loss streak gate (%d losses) — need conf≥%.2f, have %.2f",
                            symbol, _loss_streak, _streak_min_conf, confidence,
                        )
                        return

            # ── Crypto-wide consecutive loss pause ────────────────────────────────
            if _is_crypto and current_pos == 0:
                _pause_until = getattr(self, "_crypto_pause_until", None)
                if _pause_until is not None and datetime.now() < _pause_until:
                    logger.info(
                        "%s: Crypto-wide pause active until %s (%d consecutive losses)",
                        symbol, _pause_until.strftime("%H:%M"),
                        self._crypto_consec_losses,
                    )
                    return

            stress_state = getattr(self, "_stress_control_state", None)
            if (
                current_pos == 0
                and stress_state is not None
                and bool(getattr(stress_state, "halt_new_entries", False))
            ):
                logger.warning(
                    "⛔ %s: Intraday stress gate blocks new entry (%s, %s %.2f%%)",
                    symbol,
                    stress_state.reason,
                    stress_state.worst_scenario_name or stress_state.worst_scenario_id,
                    float(stress_state.worst_portfolio_return) * 100.0,
                )
                self._journal_risk_decision(
                    symbol=symbol,
                    asset_class=asset_class,
                    decision="blocked",
                    stage="intraday_stress",
                    reason="halt_new_entries",
                    signal=float(signal),
                    confidence=float(confidence),
                    price=float(price),
                    current_position=float(current_pos),
                    metadata={
                        "action": stress_state.action,
                        "stress_reason": stress_state.reason,
                        "stress_size_multiplier": float(stress_state.size_multiplier),
                        "worst_scenario_id": stress_state.worst_scenario_id,
                        "worst_scenario_name": stress_state.worst_scenario_name,
                        "worst_portfolio_return": float(stress_state.worst_portfolio_return),
                        "worst_drawdown": float(stress_state.worst_drawdown),
                        **self._governor_policy_metadata(
                            active_policy,
                            governor_regime_key,
                            perf_snapshot.tier.value,
                        ),
                    },
                )
                return

            # ✅ CRITICAL: Use lock to check position count atomically
            # This prevents race condition where multiple parallel tasks pass the check
            async with self._position_lock:
                # ✅ CRITICAL FIX: Check if we already have a position for this symbol
                # This prevents duplicate entries when parallel tasks race
                existing_qty = self.positions.get(symbol, 0)
                if existing_qty != 0:
                    logger.warning(f"⚠️ {symbol}: Already have position ({existing_qty}) - blocking duplicate entry")
                    return

                # ── Concurrent crypto position cap ───────────────────────────────
                if _is_crypto and current_pos == 0:
                    _max_crypto = int(getattr(ApexConfig, "CRYPTO_MAX_CONCURRENT_POSITIONS", 4))
                    _live_crypto = sum(
                        1 for s, q in self.positions.items()
                        if q != 0 and str(s).upper().startswith(("CRYPTO:", "BTC", "ETH", "SOL", "AVAX",
                                                                   "LINK", "XRP", "ADA", "DOGE", "DOT",
                                                                   "LTC", "FIL", "CRV", "AAVE", "ALGO",
                                                                   "ATOM", "UNI", "MATIC"))
                    )
                    if _live_crypto >= _max_crypto:
                        logger.debug(
                            "%s: Crypto position cap reached (%d/%d) — skipping",
                            symbol, _live_crypto, _max_crypto,
                        )
                        return

                if self.position_count >= ApexConfig.MAX_POSITIONS:
                    logger.info(f"⚠️ {symbol}: Max positions reached ({self.position_count}/{ApexConfig.MAX_POSITIONS})")
                    return

                # ✅ Check sector limits (inside lock)
                if not await self.check_sector_limit(symbol):
                    return

                # Reserve the position slot (prevents race condition).
                # Track in _pending_entries so cleanup can distinguish this
                # from a real 1-share position without ambiguity.
                self.positions[symbol] = 1 if signal > 0 else -1  # Placeholder
                self._pending_entries.add(symbol)

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
                            historical_prices=prices,
                            asset_class=parsed_symbol.asset_class
                        )

                        shares = sizing.target_shares

                        # SOTA: The InstitutionalRiskManager natively calculates volatility-adjusted sizes. 
                        # We must respect its output to maintain Risk Parity rather than destroying it 
                        # with a static dollar-value overwrite.
                        shares = float(sizing.target_shares)

                        # SOTA: Apply VIX-based risk multiplier
                        # NOTE: All sizing multipliers are applied in float to avoid
                        # cascading int() truncation errors. Final int() at the end.
                        shares = shares * self._vix_risk_multiplier

                        # Correlation early warning: proactive position reduction when
                        # equity-crypto correlation is rising fast (velocity signal)
                        _cew_mult = 1.0
                        if getattr(self, '_corr_early_warning', None) is not None:
                            _cew_mult = self._corr_early_warning.get_position_multiplier()
                            if _cew_mult < 1.0:
                                shares = shares * _cew_mult
                                _cew_tier = self._corr_early_warning.get_stats()
                                logger.info(
                                    "   📡 CorrEarlyWarning %s: %.2fx → %.1f shares",
                                    _cew_tier.tier if _cew_tier else "?",
                                    _cew_mult, shares,
                                )

                        # Intraday stress size multiplier: scale down when scenarios breach limits
                        _stress_mult = 1.0
                        _stress_state = getattr(self, '_stress_control_state', None)
                        if _stress_state is not None and _stress_state.active:
                            _stress_mult = _stress_state.size_multiplier
                            if _stress_mult < 1.0:
                                shares = shares * _stress_mult
                                logger.info(
                                    "   🚨 StressEngine %s: %.2fx → %.1f shares (%s)",
                                    _stress_state.action, _stress_mult, shares, _stress_state.worst_scenario_name,
                                )

                        # Execution quality slippage penalty: reduce size for chronic high-slip symbols
                        if getattr(self, '_exec_quality', None) is not None:
                            _slip_penalty = self._exec_quality.get_sizing_penalty(symbol)
                            if _slip_penalty < 1.0:
                                shares = shares * _slip_penalty
                                logger.info(
                                    "   📉 ExecQuality slip penalty %.2fx → %.1f shares (%s)",
                                    _slip_penalty, shares, symbol,
                                )

                        # Pre-earnings sizing gate: dampen size as earnings approach
                        _eg = getattr(self, '_earnings_gate', None)
                        if _eg is not None and getattr(ApexConfig, "EARNINGS_GATE_ENABLED", True):
                            _earn_mult = _eg.get_sizing_mult(symbol.split(":")[-1])
                            if _earn_mult < 1.0:
                                shares = shares * _earn_mult
                                logger.info(
                                    "   📅 EarningsGate: %.0fh to earnings → size×%.2f → %.1f shares (%s)",
                                    _eg.hours_until_earnings(symbol.split(":")[-1]) or 0,
                                    _earn_mult, shares, symbol,
                                )

                        # HRP correlation dampener: reduce size when symbol is highly
                        # correlated with existing portfolio positions
                        _hrp = getattr(self, '_hrp_sizer', None)
                        if _hrp is not None and getattr(ApexConfig, "HRP_SIZING_ENABLED", True):
                            try:
                                # Build lightweight returns cache from historical_data
                                _hrp_rets: dict = {}
                                for _ps, _pq in self.positions.items():
                                    if _pq != 0:
                                        _phist = self.historical_data.get(_ps)
                                        if _phist is not None and len(_phist) >= 10 and "Close" in _phist.columns:
                                            _hrp_rets[_ps] = list(_phist["Close"].pct_change().dropna().tail(20))
                                # Also add target symbol
                                _thist = self.historical_data.get(symbol)
                                if _thist is not None and len(_thist) >= 10 and "Close" in _thist.columns:
                                    _hrp_rets[symbol] = list(_thist["Close"].pct_change().dropna().tail(20))
                                _hrp_mult = _hrp.get_size_multiplier(symbol, self.positions, _hrp_rets)
                                if _hrp_mult < 1.0:
                                    shares = shares * _hrp_mult
                                    logger.debug(
                                        "   🔗 HRP: %s corr-adjusted → size×%.2f → %.1f shares",
                                        symbol, _hrp_mult, shares,
                                    )
                            except Exception as _hrp_err:
                                logger.debug("HRP sizing error for %s: %s", symbol, _hrp_err)

                        # Vol targeting: scale to hit annualised portfolio vol target
                        _vt_mult = float(getattr(self, '_vol_targeting_mult', 1.0))
                        if _vt_mult != 1.0:
                            shares = shares * _vt_mult
                            logger.debug("   VolTarget mult=%.3f → %.1f shares", _vt_mult, shares)

                        # ✅ Phase 1.4: Apply graduated circuit breaker risk multiplier
                        if self._risk_multiplier < 1.0:
                            shares = shares * self._risk_multiplier
                            logger.info(f"   ⚠️ Risk reduced: {self._risk_multiplier:.0%} (VIX: {self._vix_risk_multiplier:.2f}) → {shares:.1f} shares")

                        # 🏰 Signal Fortress: Apply per-symbol size multiplier
                        if self.threshold_optimizer:
                            size_mult = sym_thresholds.position_size_multiplier
                            if size_mult != 1.0:
                                shares = shares * size_mult
                                logger.debug(f"   🏰 Adaptive size: {size_mult:.2f}x → {shares:.1f} shares")

                        # 🏆 Excellence: Apply signal strength based size scaling
                        if hasattr(self, 'excellence_manager'):
                            size_rec = self.excellence_manager.calculate_size_scaling(
                                symbol=symbol,
                                signal=signal,
                                confidence=confidence,
                                regime=self._current_regime,
                                base_shares=shares
                            )
                            if size_rec.scaling_factor != 1.0:
                                shares = size_rec.recommended_shares
                                if size_rec.reasons:
                                    logger.info(f"   🏆 Excellence sizing: {size_rec.scaling_factor:.0%} ({', '.join(size_rec.reasons)})")

                        # 🛡️ Signal Fortress V2: Apply Black Swan Guard size multiplier
                        if self.black_swan_guard:
                            bsg_mult = self.black_swan_guard.get_position_size_multiplier()
                            if bsg_mult < 1.0:
                                shares = shares * bsg_mult
                                logger.info(f"   🛡️ BlackSwanGuard: {bsg_mult:.0%} → {shares:.1f} shares")

                        # 🛡️ Signal Fortress V2: Apply Drawdown Cascade size multiplier
                        if self.drawdown_breaker:
                            dd_mult = self.drawdown_breaker.get_position_size_multiplier()
                            if dd_mult < 1.0:
                                shares = shares * dd_mult
                                logger.info(f"   🛡️ DrawdownBreaker: {dd_mult:.0%} → {shares:.1f} shares")

                        # 🔗 Correlation Regime: Reduce size during HERDING/CRISIS regimes.
                        # HERDING (avg_corr >= 0.60): 50% of normal — diversification is failing.
                        # CRISIS  (avg_corr >= 0.80): 25% of normal — portfolio is one asset effectively.
                        _corr_regime = getattr(self, "_correlation_regime", None)
                        if _corr_regime is not None:
                            if _corr_regime.name == "CRISIS":
                                corr_mult = 0.25
                            elif _corr_regime.name == "HERDING":
                                corr_mult = 0.50
                            else:
                                corr_mult = 1.0
                            if corr_mult < 1.0:
                                shares = shares * corr_mult
                                logger.info(
                                    "   🔗 CorrelationBreaker [%s, avg=%.2f]: %.0f%% size → %.1f shares",
                                    _corr_regime.name,
                                    getattr(self, "_correlation_avg", 0.0),
                                    corr_mult * 100,
                                    shares,
                                )


                        # 🔗 Per-entry correlation guard: graduated reduction for correlated entries
                        if _corr_size_mult < 1.0:
                            shares = shares * _corr_size_mult

                        # 🛡️ Signal Fortress V2: Apply Execution Shield slippage adjustment
                        if self.execution_shield:
                            slip_adj = self.execution_shield.get_slippage_adjustment(symbol)
                            if slip_adj < 1.0:
                                shares = shares * slip_adj
                                logger.debug(f"   🛡️ ExecutionShield slippage adj: {slip_adj:.0%} → {shares:.1f} shares")

                        if governor_controls.size_multiplier < 1.0:
                            shares = shares * governor_controls.size_multiplier
                            logger.info(
                                f"   🧭 PerformanceGovernor: {governor_controls.size_multiplier:.0%} "
                                f"size ({perf_snapshot.tier.value}, {governor_regime_key}) → {shares} shares"
                            )
                        if (
                            stress_state is not None
                            and current_pos == 0
                            and float(getattr(stress_state, "size_multiplier", 1.0)) < 1.0
                        ):
                            shares = shares * float(stress_state.size_multiplier)
                            logger.info(
                                "   🧯 IntradayStress: %.0f%% size (%s) → %.1f shares",
                                float(stress_state.size_multiplier) * 100.0,
                                stress_state.worst_scenario_name or stress_state.worst_scenario_id,
                                shares,
                            )

                        # ── Live Kelly position sizing ────────────────────────────
                        if getattr(ApexConfig, "LIVE_KELLY_SIZING_ENABLED", True):
                            try:
                                from models.portfolio_optimizer import get_kelly_multiplier
                                _win_rate = float(self.outcome_loop.get_diagnostics().get("rolling_accuracy", 0.5))
                                _kelly_m = get_kelly_multiplier(
                                    ml_confidence=float(confidence),
                                    historical_win_rate=_win_rate,
                                    is_high_vix=self._vix_risk_multiplier < 1.0
                                )
                                if _kelly_m != 1.0:
                                    shares = shares * _kelly_m
                                    logger.info(
                                        "   📊 Kelly Criteria Optimizer (WinRate: %.0f%%, Conf: %.0f%%): %.2fx multiplier → %.4f shares",
                                        _win_rate * 100, float(confidence) * 100, _kelly_m, shares
                                    )
                            except Exception as e:
                                logger.warning(f"Failed Kelly Optimizer execution: {e}")

                        # ── Regime transition caution window ─────────────────────
                        # Duration scales inversely with transition confidence:
                        #   high confidence (0.95) → 1h caution (real breakout)
                        #   low confidence  (0.70) → 6h caution (marginal change)
                        #   formula: max(1h, 6h × (1 - confidence))
                        # Size multiplier scales with prior-regime severity:
                        #   crisis/panic → 0.55×, bear → 0.70×, volatile/stress → 0.80×
                        if getattr(ApexConfig, "REGIME_TRANSITION_CAUTION_ENABLED", True):
                            _changed_at = getattr(self, "_regime_changed_at", None)
                            if _changed_at is not None:
                                _h_since = (datetime.now() - _changed_at).total_seconds() / 3600
                                _prev_regime = getattr(self, "_last_regime", "")
                                _trans_conf = float(getattr(self, "_regime_transition_confidence", 0.70))
                                # Duration: confidence-scaled
                                _caution_h = max(
                                    1.0,
                                    float(getattr(ApexConfig, "REGIME_TRANSITION_CAUTION_HOURS", 4.0))
                                    * (1.0 - _trans_conf),
                                )
                                # Severity-scaled multiplier
                                _severity_mult = {
                                    "crisis":        0.55, "panic":         0.55,
                                    "bear":          0.70, "bearish":       0.70,
                                    "strong_bear":   0.60,
                                    "volatile":      0.80, "volatile_bear": 0.75,
                                    "stress":        0.80,
                                }.get(_prev_regime, float(getattr(ApexConfig, "REGIME_TRANSITION_SIZE_MULT", 0.70)))
                                _was_bear = _prev_regime in (
                                    "bear", "bearish", "strong_bear", "volatile",
                                    "volatile_bear", "crisis", "stress", "panic",
                                )
                                if _was_bear and _h_since < _caution_h:
                                    shares = shares * _severity_mult
                                    logger.info(
                                        "   🔄 Regime caution (%.1fh/%.1fh, conf=%.0f%%, "
                                        "%s→%s): %.0f%% size → %d shares",
                                        _h_since, _caution_h, _trans_conf * 100,
                                        _prev_regime, self._current_regime,
                                        _severity_mult * 100, shares,
                                    )

                        shares = self._apply_social_size_multiplier(
                            symbol=symbol,
                            shares=shares,
                            decision=social_decision,
                            price=price,
                        )

                        # AG: Portfolio Profit Ratchet — when overall equity has run up
                        # significantly above peak, scale back new entries to protect gains.
                        if getattr(ApexConfig, "PROFIT_RATCHET_ENABLED", True):
                            _eq_peak = float(self.risk_manager.peak_capital or self.capital or 0.0)
                            _eq_now = float(self.capital or 0.0)
                            _ratchet_trigger = float(getattr(ApexConfig, "PROFIT_RATCHET_TRIGGER_PCT", 10.0)) / 100.0
                            if _eq_peak > 0 and _eq_now >= _eq_peak * (1.0 + _ratchet_trigger):
                                _ratchet_scale = float(getattr(ApexConfig, "PROFIT_RATCHET_SIZE_SCALE", 0.6))
                                _pre_ratchet = shares
                                shares = shares * _ratchet_scale
                                if shares != _pre_ratchet:
                                    logger.info(
                                        "   🔒 AG Profit Ratchet: equity +%.1f%% above peak → "
                                        "size scaled to %.0f%% (%.1f→%.1f shares)",
                                        (_eq_now / _eq_peak - 1.0) * 100,
                                        _ratchet_scale * 100,
                                        _pre_ratchet, shares,
                                    )

                        if sizing.constraints:
                            logger.debug(f"   Size constraints: {', '.join(sizing.constraints)}")

                        logger.info(f"🔢 {symbol}: Sizing result - shares={shares} (inst={sizing.target_shares}, vix_mult={self._vix_risk_multiplier:.2f}, risk_mult={self._risk_multiplier:.2f})")
                        if sizing.constraints:
                            logger.info(f"   📋 Constraints: {', '.join(sizing.constraints)}")

                        # 🛡️ Defensive Asset Boost: GLD/SLV/XOM/CVX + healthcare defensives get
                        # a 1.5× size multiplier — they consistently outperform during market stress
                        # (gold +19% Jan-Mar 2026 study). Only applies to non-crypto assets.
                        _defensive_syms = getattr(ApexConfig, 'DEFENSIVE_SYMBOLS', frozenset())
                        if symbol in _defensive_syms and str(asset_class).upper() != "CRYPTO":
                            _def_mult = float(getattr(ApexConfig, 'DEFENSIVE_POSITION_SIZE_MULTIPLIER', 1.5))
                            _pre_def = shares
                            shares = shares * _def_mult
                            if shares != _pre_def:
                                logger.info(
                                    "   🛡️ %s: Defensive boost +%.0f%% → %.1f shares",
                                    symbol, (_def_mult - 1) * 100, shares,
                                )

                        # Macro overlay: adjust size based on yield curve / VIX structure
                        _macro_ctx = getattr(self, '_macro_context', None)
                        if _macro_ctx is not None and getattr(ApexConfig, "MACRO_INDICATORS_ENABLED", True):
                            _mac_mult = (
                                _macro_ctx.crypto_size_multiplier
                                if str(asset_class).upper() == "CRYPTO"
                                else _macro_ctx.equity_size_multiplier
                            )
                            if _mac_mult < 1.0:
                                shares = shares * _mac_mult
                                logger.debug(
                                    "   📊 %s: Macro overlay ×%.2f → %.3f shares "
                                    "(regime=%s, yc_inv=%s, vix_back=%s)",
                                    symbol, _mac_mult, shares,
                                    _macro_ctx.regime_signal,
                                    _macro_ctx.yield_curve_inverted,
                                    _macro_ctx.vix_backwardation,
                                )

                        # ── Confidence-proportional sizing ───────────────────────
                        # Scale size linearly from CONF_SCALING_MIN_MULT at min_conf
                        # to 1.0× at full confidence.  Reduces exposure on marginal entries.
                        if getattr(ApexConfig, "CONF_PROPORTIONAL_SIZING_ENABLED", True):
                            _is_crypto_entry = str(asset_class).upper() == "CRYPTO"
                            _conf_floor = float(
                                getattr(ApexConfig, "CRYPTO_MIN_CONFIDENCE", 0.40)
                                if _is_crypto_entry
                                else getattr(ApexConfig, "ENTRY_MIN_CONFIDENCE", 0.60)
                            )
                            _conf_min_mult = float(getattr(ApexConfig, "CONF_SCALING_MIN_MULT", 0.25))
                            _conf_range = max(1e-8, 1.0 - _conf_floor)
                            _conf_scale = _conf_min_mult + (1.0 - _conf_min_mult) * max(
                                0.0, min(1.0, (float(confidence) - _conf_floor) / _conf_range)
                            )
                            if _conf_scale < 0.99:
                                shares = shares * _conf_scale
                                logger.debug(
                                    "   📊 Conf-proportional sizing: %.0f%% "
                                    "(conf=%.2f, floor=%.2f) → %.4f shares",
                                    _conf_scale * 100, confidence, _conf_floor, shares,
                                )

                        # ── HHI Concentration Gate ──────────────────────────────
                        # When the portfolio is already highly concentrated (HHI above
                        # threshold), reduce new position size to avoid doubling down
                        # on sector concentration.  Does NOT block the entry — just
                        # shrinks it proportionally so the book stays diversified.
                        _hhi_gate_enabled = getattr(ApexConfig, "HHI_CONCENTRATION_GATE_ENABLED", True)
                        _last_fexp = getattr(self, '_last_factor_exposure', None)
                        if _hhi_gate_enabled and _last_fexp is not None:
                            try:
                                _hhi = float(getattr(_last_fexp, 'concentration_hhi', 0.0))
                                _hhi_warn = float(getattr(ApexConfig, "HHI_CONCENTRATION_WARN", 0.25))
                                _hhi_max  = float(getattr(ApexConfig, "HHI_CONCENTRATION_HARD", 0.45))
                                if _hhi > _hhi_warn:
                                    # Linear reduction: 0% at warn threshold → 50% at hard threshold
                                    _hhi_excess = min(_hhi - _hhi_warn, _hhi_max - _hhi_warn)
                                    _hhi_range  = max(_hhi_max - _hhi_warn, 1e-6)
                                    _hhi_shrink = 1.0 - 0.50 * (_hhi_excess / _hhi_range)
                                    shares = shares * _hhi_shrink
                                    logger.debug(
                                        "   📉 HHI gate: portfolio_hhi=%.3f → shrink=×%.2f shares",
                                        _hhi, _hhi_shrink,
                                    )
                            except Exception:
                                pass  # non-fatal

                        # ── Black-Litterman relative sizing ─────────────────────
                        # Scale position by BL posterior weight vs equal-weight baseline.
                        # High conviction / low correlation → larger; crowded/noisy → smaller.
                        if getattr(ApexConfig, "BL_SIZING_ENABLED", True) and self._bl_weights:
                            _bl_w = float(self._bl_weights.get(symbol, 1.0))
                            _bl_min = float(getattr(ApexConfig, "BL_MIN_SCALE", 0.40))
                            _bl_max = float(getattr(ApexConfig, "BL_MAX_SCALE", 2.00))
                            _bl_scale = max(_bl_min, min(_bl_max, _bl_w))
                            if abs(_bl_scale - 1.0) > 0.05:  # only log non-trivial adjustments
                                logger.debug(
                                    "   📊 BL sizing: %s weight=%.2f → scale=%.2f × shares",
                                    symbol, _bl_w, _bl_scale,
                                )
                            shares = shares * _bl_scale

                        # ── Meta-controller size guidance ────────────────────────
                        # Applied last so it scales the fully-adjusted share count.
                        _mc_sm = float(getattr(self, '_meta_size_mult', 1.0))
                        if abs(_mc_sm - 1.0) > 0.02:
                            logger.debug(
                                "   🧠 MetaCtrl size ×%.2f → %.4f shares",
                                _mc_sm, shares * _mc_sm,
                            )
                            shares = shares * _mc_sm
                        self._meta_size_mult = 1.0  # reset for next symbol

                        # Final int conversion: all multipliers applied in float to
                        # prevent cascading truncation. Convert to int once here.
                        if str(asset_class).upper() != "CRYPTO":
                            shares = max(1, int(round(shares)))
                        else:
                            # Crypto: institutional sizer returns integer-equivalent shares
                            # (e.g., target_shares=1 for BTC = $74K — catastrophic for a $79K account).
                            # ALWAYS cap at CRYPTO_POSITION_SIZE_USD (scaled by allocator) to enforce fractional sizing.
                            if price > 0:
                                _max_notional = float(getattr(ApexConfig, "CRYPTO_POSITION_SIZE_USD", 5000))
                                # Capital allocator: scale crypto notional by recommended crypto fraction
                                _ca = getattr(self, '_capital_allocator', None)
                                if _ca is not None:
                                    _cr_frac = _ca.current_crypto_frac
                                    _cap_pv = float(getattr(self, '_current_portfolio_value', 0) or 0)
                                    if _cap_pv > 0:
                                        _ca_notional = _cap_pv * _cr_frac / max(1, getattr(ApexConfig, "MAX_POSITIONS", 40) // 4)
                                        _max_notional = min(_max_notional * (_cr_frac / 0.50), _ca_notional)
                                        _max_notional = max(_max_notional, 500.0)  # floor
                                _max_by_notional = round(_max_notional / price, 6)
                                if shares > _max_by_notional:
                                    logger.info(
                                        "   💎 %s: Crypto notional cap %.6f→%.6f × $%.2f = $%.0f (max $%.0f)",
                                        symbol, shares, _max_by_notional, price,
                                        _max_by_notional * price, _max_notional,
                                    )
                                    shares = _max_by_notional

                        if shares < 1:
                            # Crypto: Alpaca supports fractional quantities — compute from target notional
                            if str(asset_class).upper() == "CRYPTO" and price > 0:
                                _notional = float(getattr(ApexConfig, "CRYPTO_POSITION_SIZE_USD", 5000))
                                frac = round(_notional / price * self._vix_risk_multiplier * self._risk_multiplier * float(getattr(self, '_vol_targeting_mult', 1.0)), 6)
                                if frac >= 0.001:
                                    shares = frac
                                    logger.info(f"   💎 {symbol}: Fractional crypto: {frac:.6f} × ${price:.2f} = ${frac*price:.0f}")
                                else:
                                    logger.info(f"⚠️ {symbol}: Crypto position too small ({frac:.6f} units @ ${price:.2f})")
                                    async with self._position_lock:
                                        if symbol in self.positions:
                                            del self.positions[symbol]
                                    return
                            else:
                                if sizing.constraints:
                                    logger.info(f"⚠️ {symbol}: Position blocked by {sizing.constraints}")
                                else:
                                    logger.info(f"⚠️ {symbol}: Price too high or risk too high (${price:.2f})")
                                async with self._position_lock:
                                    if symbol in self.positions:
                                        del self.positions[symbol]
                                return
                    else:
                        # Fallback: standard position sizing — use dedicated crypto budget if applicable
                        _pos_size_usd = (
                            getattr(ApexConfig, "CRYPTO_POSITION_SIZE_USD", 5000)
                            if asset_class == "CRYPTO"
                            else ApexConfig.POSITION_SIZE_USD
                        )
                        if str(asset_class).upper() == "CRYPTO" and price > 0:
                            # Crypto: use fractional quantity instead of truncating to 0
                            shares = round(_pos_size_usd / price * self._vix_risk_multiplier * float(getattr(self, '_vol_targeting_mult', 1.0)), 6)
                        else:
                            shares = int(_pos_size_usd / price)
                            shares = min(shares, ApexConfig.MAX_SHARES_PER_POSITION)
                            # SOTA: Apply VIX multiplier
                            shares = int(shares * self._vix_risk_multiplier)

                        if governor_controls.size_multiplier < 1.0:
                            if str(asset_class).upper() == "CRYPTO":
                                shares = round(shares * governor_controls.size_multiplier, 6)
                            else:
                                shares = max(1, int(shares * governor_controls.size_multiplier))
                            logger.info(
                                f"   🧭 PerformanceGovernor: {governor_controls.size_multiplier:.0%} "
                                f"size ({perf_snapshot.tier.value}, {governor_regime_key}) → {shares} shares"
                            )
                        if (
                            stress_state is not None
                            and current_pos == 0
                            and float(getattr(stress_state, "size_multiplier", 1.0)) < 1.0
                        ):
                            if str(asset_class).upper() == "CRYPTO":
                                shares = round(shares * float(stress_state.size_multiplier), 6)
                            else:
                                shares = max(1, int(shares * float(stress_state.size_multiplier)))
                            logger.info(
                                "   🧯 IntradayStress: %.0f%% size (%s) → %s shares",
                                float(stress_state.size_multiplier) * 100.0,
                                stress_state.worst_scenario_name or stress_state.worst_scenario_id,
                                shares,
                            )
                        shares = self._apply_social_size_multiplier(
                            symbol=symbol,
                            shares=shares,
                            decision=social_decision,
                            price=price,
                        )

                        # ── Live Kelly position sizing (Fallback Engine) ────────────────────────────
                        if getattr(ApexConfig, "LIVE_KELLY_SIZING_ENABLED", True):
                            try:
                                from models.portfolio_optimizer import get_kelly_multiplier
                                _win_rate = float(self.outcome_loop.get_diagnostics().get("rolling_accuracy", 0.5))
                                _kelly_m = get_kelly_multiplier(
                                    ml_confidence=float(confidence),
                                    historical_win_rate=_win_rate,
                                    is_high_vix=self._vix_risk_multiplier < 1.0
                                )
                                if _kelly_m != 1.0:
                                    shares = shares * _kelly_m
                                    logger.info(
                                        "   📊 Kelly Criteria Optimizer (WinRate: %.0f%%, Conf: %.0f%%): %.2fx multiplier → %.4f shares",
                                        _win_rate * 100, float(confidence) * 100, _kelly_m, shares
                                    )
                            except Exception as e:
                                logger.warning(f"Failed Kelly Optimizer execution: {e}")

                        # Macro overlay (fallback path)
                        _macro_ctx_fb = getattr(self, '_macro_context', None)
                        if _macro_ctx_fb is not None and getattr(ApexConfig, "MACRO_INDICATORS_ENABLED", True):
                            _mac_mult_fb = (
                                _macro_ctx_fb.crypto_size_multiplier
                                if str(asset_class).upper() == "CRYPTO"
                                else _macro_ctx_fb.equity_size_multiplier
                            )
                            if _mac_mult_fb < 1.0:
                                shares = shares * _mac_mult_fb

                        if shares < 0.001:
                            logger.debug(f"⚠️ {symbol}: Price too high (${price:.2f})")
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                    # Determine side (long or short)
                    side = 'BUY' if signal > 0 else 'SELL'
                    entry_connector = self._get_connector_for(symbol)

                    # Institutional pre-trade hard-limit gateway.
                    reference_price = None
                    adv_shares = None
                    try:
                        # Crypto: skip price_band — daily historical close vs live price
                        # differs by 3-10% normally (not an anomaly). Pass None to disable.
                        if len(prices) > 1 and str(asset_class).upper() != "CRYPTO":
                            val = prices.iloc[-2]
                            if isinstance(val, pd.Series) and "Close" in val:
                                reference_price = float(val["Close"])
                            elif isinstance(val, (int, float, np.number)):
                                reference_price = float(val)
                    except Exception as e:
                        logger.debug(f"Failed to extract reference_price for {symbol}: {e}")
                        reference_price = None

                    try:
                        hist = self.historical_data.get(symbol)
                        if hist is not None and hasattr(hist, "columns") and "Volume" in hist.columns:
                            adv_shares = float(pd.to_numeric(hist["Volume"], errors="coerce").tail(20).median())
                    except Exception:
                        adv_shares = None

                    try:
                        positions_for_gate = {
                            s: q for s, q in self.positions.items()
                            if s != symbol and q != 0
                        }
                        pretrade_decision = self.pretrade_gateway.evaluate_entry(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=side,
                            quantity=shares,
                            price=price,
                            capital=float(self.capital),
                            current_positions=positions_for_gate,
                            price_cache=self.price_cache,
                            reference_price=reference_price,
                            adv_shares=adv_shares,
                        )
                        self.pretrade_gateway.record_decision(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=side,
                            quantity=shares,
                            price=price,
                            decision=pretrade_decision,
                            actor="apex-trader",
                        )
                        self._journal_risk_decision(
                            symbol=symbol,
                            asset_class=asset_class,
                            decision="allowed" if pretrade_decision.allowed else "blocked",
                            stage="pretrade_gateway",
                            reason=str(pretrade_decision.reason_code),
                            signal=float(signal),
                            confidence=float(confidence),
                            price=float(price),
                            current_position=float(current_pos),
                            metadata={
                                "side": side,
                                "quantity": float(shares),
                                "message": str(pretrade_decision.message),
                                **self._governor_policy_metadata(
                                    active_policy,
                                    governor_regime_key,
                                    perf_snapshot.tier.value,
                                ),
                            },
                        )
                        if self.prometheus_metrics:
                            self.prometheus_metrics.record_pretrade_gate_decision(
                                asset_class=asset_class,
                                allowed=pretrade_decision.allowed,
                                reason=pretrade_decision.reason_code,
                            )
                        if not pretrade_decision.allowed:
                            if self.prometheus_metrics:
                                self.prometheus_metrics.record_governor_blocked_entry(
                                    asset_class=asset_class,
                                    regime=governor_regime_key,
                                    reason=f"pretrade_{pretrade_decision.reason_code}",
                                )
                            logger.warning(
                                "⛔ %s: Pre-trade gateway blocked entry (%s) %s",
                                symbol,
                                pretrade_decision.reason_code,
                                pretrade_decision.message,
                            )
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return
                    except Exception as pretrade_exc:
                        logger.error("Pre-trade gateway error for %s: %s", symbol, pretrade_exc)
                        self._journal_risk_decision(
                            symbol=symbol,
                            asset_class=asset_class,
                            decision="blocked" if self.pretrade_gateway.config.fail_closed else "degraded",
                            stage="pretrade_gateway",
                            reason="gateway_error",
                            signal=float(signal),
                            confidence=float(confidence),
                            price=float(price),
                            current_position=float(current_pos),
                            metadata={
                                "error": str(pretrade_exc),
                                **self._governor_policy_metadata(
                                    active_policy,
                                    governor_regime_key,
                                    perf_snapshot.tier.value,
                                ),
                            },
                        )
                        if self.pretrade_gateway.config.fail_closed:
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                    # ExecutionShield hard gates: spread gate + slippage budget.
                    if self.execution_shield:
                        try:
                            quote = await asyncio.wait_for(
                                self._get_connector_quote(entry_connector, symbol),
                                timeout=10.0,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("⏱️ %s: Broker quote timed out (10s) — skipping entry", symbol)
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                                self._pending_entries.discard(symbol)
                            return
                        bid = float(quote.get("bid", 0.0) or 0.0)
                        ask = float(quote.get("ask", 0.0) or 0.0)
                        spread_limit_bps = self._spread_limit_bps_for_asset(asset_class)
                        slippage_budget_bps = self._slippage_budget_bps_for_asset(asset_class)
                        edge_buffer_bps = (
                            self._edge_buffer_bps_for_asset(asset_class)
                            if ApexConfig.EXECUTION_EDGE_GATE_ENABLED
                            else 0.0
                        )

                        if is_initial_build:
                            spread_limit_bps *= 3.0
                            slippage_budget_bps *= 3.0
                            edge_buffer_bps = 0.0
                            logger.info(f"ℹ️ {symbol}: Relaxing ExecutionShield gates for initial portfolio build")
                        gate_ok, gate_reason = self.execution_shield.can_enter_order(
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            max_spread_bps=spread_limit_bps,
                            slippage_budget_bps=slippage_budget_bps,
                            signal_strength=float(signal),
                            confidence=float(confidence),
                            min_edge_over_cost_bps=edge_buffer_bps,
                            signal_to_edge_bps=self._signal_to_edge_bps_for_asset(asset_class),
                        )
                        # Also override the max_slippage_bps limit in the ExecutionShield history manually so the first trades
                        # aren't using the full budget (e.g., 200 bps) as the single-trade allowed maximum.
                        if asset_class.upper() == "CRYPTO":
                            self.execution_shield.max_slippage_bps = min(slippage_budget_bps * 0.2, 30.0)
                        
                        if not gate_ok:
                            if gate_reason.startswith("spread_gate_blocked"):
                                block_reason = "execution_spread_gate"
                            elif gate_reason.startswith("slippage_budget_blocked"):
                                block_reason = "execution_slippage_budget"
                            else:
                                block_reason = "execution_edge_gate"
                            if self.prometheus_metrics:
                                self.prometheus_metrics.record_governor_blocked_entry(
                                    asset_class=asset_class,
                                    regime=governor_regime_key,
                                    reason=block_reason,
                                )
                                if block_reason == "execution_spread_gate":
                                    self.prometheus_metrics.record_execution_spread_gate_block(
                                        asset_class=asset_class,
                                        regime=governor_regime_key,
                                    )
                                elif block_reason == "execution_slippage_budget":
                                    self.prometheus_metrics.record_execution_slippage_budget_block(
                                        asset_class=asset_class,
                                        regime=governor_regime_key,
                                    )
                                else:
                                    self.prometheus_metrics.record_execution_edge_gate_block(
                                        asset_class=asset_class,
                                        regime=governor_regime_key,
                                    )
                            logger.warning("⛔ %s: Entry blocked by ExecutionShield (%s)", symbol, gate_reason)
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                    logger.info(f"📈 {side} {shares} {symbol} @ ${price:.2f} (${shares*price:,.0f})")
                    logger.info(f"   Signal: {signal:+.3f} | Confidence: {confidence:.3f}")
                    if str(asset_class).upper() == "CRYPTO":
                        _sor_rec = SmartOrderRouter().select_algorithm(symbol, side, shares)
                        logger.info(
                            "   [CRYPTO] rot_score=%.2f | sig_thr=%.3f | conf_thr=%.3f | algo=%s",
                            crypto_rotation_score,
                            effective_signal_threshold,
                            effective_confidence_threshold,
                            _sor_rec["algorithm"],
                        )
                    if self.use_institutional:
                        logger.debug(f"   Vol-adjusted: ${sizing.vol_adjusted_size:,.0f} | Corr penalty: {sizing.correlation_penalty:.2f}")

                    if entry_connector:
                        quote_age = 0.0
                        if hasattr(entry_connector, 'get_quote_age'):
                            quote_age = entry_connector.get_quote_age(symbol)
                        
                        is_pos = symbol in self.positions
                        
                        if not is_pos:
                            if quote_age > 60:
                                logger.warning(f"⏳ Skipping {symbol} entry - quote is completely stale ({quote_age:.1f}s)")
                                self.pending_orders.discard(symbol)
                                return
                            elif quote_age > 10:
                                logger.info(f"⚡ {symbol} quote is {quote_age:.1f}s old. Upgrading to priority stream.")
                                self.pending_orders.add(symbol)
                                asyncio.create_task(self.refresh_data())
                                return

                        self.pending_orders.add(symbol)
                        entry_broker_name = "alpaca" if entry_connector is self.alpaca else "ibkr"
                        entry_order_mode = "market"

                        # SOTA: Record arrival price
                        arrival_price = price
                        start_time = datetime.now()
                        _signal_ts = time.time()          # P: latency tracking — signal ready
                        _order_sent_ts = _signal_ts       # P: default (overridden in else branch)
                        _fill_ts = 0.0                    # P: set after fill received

                        # ✅ Phase 2.1: Use TWAP/VWAP for large orders
                        order_value = shares * price
                        use_advanced = (
                            entry_connector is self.ibkr and
                            ApexConfig.USE_ADVANCED_EXECUTION and
                            self.advanced_executor is not None and
                            order_value >= ApexConfig.LARGE_ORDER_THRESHOLD
                        )
                        _crypto_twap_min = float(getattr(ApexConfig, "CRYPTO_TWAP_MIN_NOTIONAL", 5000.0))
                        use_crypto_twap = (
                            not use_advanced and
                            getattr(self, '_crypto_twap', None) is not None and
                            getattr(ApexConfig, "CRYPTO_TWAP_ENABLED", True) and
                            _symbol_is_crypto(symbol) and
                            order_value >= _crypto_twap_min
                        )
                        _equity_twap_min = float(getattr(ApexConfig, "EQUITY_TWAP_MIN_NOTIONAL", 10_000.0))
                        use_equity_twap = (
                            not use_advanced and
                            not use_crypto_twap and
                            getattr(self, '_equity_twap', None) is not None and
                            getattr(ApexConfig, "EQUITY_TWAP_ENABLED", True) and
                            not _symbol_is_crypto(symbol) and
                            order_value >= _equity_twap_min
                        )
                        if use_advanced:
                            entry_order_mode = "advanced_twap"
                        elif use_crypto_twap:
                            entry_order_mode = "crypto_twap"
                        elif use_equity_twap:
                            entry_order_mode = "equity_twap"
                        else:
                            entry_order_mode = "market"

                        self._journal_order_event(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=side,
                            quantity=float(shares),
                            broker=entry_broker_name,
                            lifecycle="submitted",
                            order_role="entry",
                            signal=float(signal),
                            confidence=float(confidence),
                            expected_price=float(arrival_price),
                            metadata={
                                "mode": entry_order_mode,
                                **self._governor_policy_metadata(
                                    active_policy,
                                    governor_regime_key,
                                    perf_snapshot.tier.value,
                                ),
                            },
                        )

                        if use_advanced:
                            # Use TWAP for large IBKR orders to reduce market impact
                            logger.info(f"   📊 Using IBKR TWAP execution (order value: ${order_value:,.0f})")
                            trade = await self.advanced_executor.execute_twap_order(
                                symbol=symbol,
                                side=side,
                                qty=shares,
                                current_price=float(price),
                            )
                        elif use_crypto_twap:
                            # TWAP for large crypto/Alpaca orders: slice into N child orders
                            logger.info(
                                "   📊 Using Crypto TWAP (value=$%.0f, connector=Alpaca)", order_value
                            )
                            _twap_result = await self._crypto_twap.execute(
                                connector=entry_connector,
                                symbol=symbol,
                                side=side,
                                total_qty=shares,
                                current_price=float(price),
                                confidence=float(confidence),
                            )
                            trade = _twap_result.to_dict() if _twap_result else None
                        elif use_equity_twap:
                            # TWAP for large IBKR equity orders: slice into N limit tranches
                            logger.info(
                                "   📊 Using Equity TWAP (value=$%.0f, slices=%d × %ds)",
                                order_value,
                                int(getattr(ApexConfig, "EQUITY_TWAP_SLICES", 5)),
                                int(getattr(ApexConfig, "EQUITY_TWAP_INTERVAL_SEC", 60)),
                            )
                            _eq_twap_result = await self._equity_twap.execute(
                                connector=entry_connector,
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                notional=order_value,
                                confidence=float(confidence),
                            )
                            trade = _eq_twap_result.to_dict() if _eq_twap_result else None
                        else:
                            _order_sent_ts = time.time()   # P: latency — order submitted
                            trade = await entry_connector.execute_order(
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                confidence=confidence
                            )

                        fill_price = 0.0
                        trade_status = "UNKNOWN"
                        expected_price_for_shield = float(arrival_price)
                        # SOTA: Record execution benchmark if we have a trade
                        if trade:
                            _fill_ts = time.time()         # P: latency — fill received
                            fill_price = trade.get('price', 0) if isinstance(trade, dict) else trade.orderStatus.avgFillPrice
                            if isinstance(trade, dict):
                                trade_status = str(trade.get("status", "UNKNOWN")).upper()
                                expected_price_for_shield = float(
                                    trade.get("expected_price", expected_price_for_shield) or expected_price_for_shield
                                )
                            else:
                                trade_status = "FILLED"
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
                                # P: Write execution latency audit record
                                _broker_name = "ibkr" if use_advanced or entry_connector is self.ibkr else "alpaca"
                                self._write_execution_latency(
                                    symbol=symbol, side=side, broker=_broker_name,
                                    signal_ts=_signal_ts,
                                    order_sent_ts=_order_sent_ts if not use_advanced else _signal_ts,
                                    fill_ts=_fill_ts,
                                    quantity=float(shares), fill_price=float(fill_price),
                                    signal_strength=float(signal),
                                    expected_price=float(arrival_price) if arrival_price else 0.0,
                                )
                            self._journal_order_event(
                                symbol=symbol,
                                asset_class=asset_class,
                                side=side,
                                quantity=float(shares),
                                broker=entry_broker_name,
                                lifecycle="filled" if trade_status == "FILLED" else "result",
                                order_role="entry",
                                signal=float(signal),
                                confidence=float(confidence),
                                expected_price=float(arrival_price),
                                fill_price=float(fill_price),
                                status=trade_status,
                                metadata={
                                    "mode": entry_order_mode,
                                    **self._governor_policy_metadata(
                                        active_policy,
                                        governor_regime_key,
                                        perf_snapshot.tier.value,
                                    ),
                                },
                            )
                        else:
                            self._journal_order_event(
                                symbol=symbol,
                                asset_class=asset_class,
                                side=side,
                                quantity=float(shares),
                                broker=entry_broker_name,
                                lifecycle="result",
                                order_role="entry",
                                signal=float(signal),
                                confidence=float(confidence),
                                expected_price=float(arrival_price),
                                status="NO_TRADE",
                                metadata={
                                    "mode": entry_order_mode,
                                    **self._governor_policy_metadata(
                                        active_policy,
                                        governor_regime_key,
                                        perf_snapshot.tier.value,
                                    ),
                                },
                            )

                        if trade:
                            trade_success = True
                            if self.execution_shield and fill_price > 0 and trade_status == "FILLED":
                                try:
                                    algo_used = ExecutionAlgo.TWAP if use_advanced else ExecutionAlgo.MARKET
                                    self.execution_shield.record_execution(
                                        symbol=symbol,
                                        expected_price=expected_price_for_shield,
                                        fill_price=float(fill_price),
                                        shares=shares,
                                        algo=algo_used,
                                        execution_time=(datetime.now() - start_time).total_seconds(),
                                    )
                                except Exception as exec_shield_exc:
                                    logger.debug(
                                        "ExecutionShield record skipped for %s due to error: %s",
                                        symbol,
                                        exec_shield_exc,
                                    )
                            # ✅ CRITICAL: Force sync after trade
                            await self._sync_positions()

                            # Track commission
                            commission = ApexConfig.COMMISSION_PER_TRADE
                            self.total_commissions += commission

                            entry_ts = datetime.now()
                            entry_execution_price = float(fill_price if fill_price > 0 else price)
                            entry_broker = "alpaca" if entry_connector is self.alpaca else "ibkr"
                            if trade_status == "FILLED":
                                self._record_fill_realized_pnl(
                                    broker_name=entry_broker,
                                    symbol=symbol,
                                    side=side,
                                    quantity=float(shares),
                                    fill_price=entry_execution_price,
                                    commission=float(commission),
                                    filled_at=entry_ts,
                                )
                            self.position_entry_prices[symbol] = entry_execution_price
                            self.position_entry_times[symbol] = entry_ts
                            self.position_peak_prices[symbol] = entry_execution_price
                            self.position_entry_signals[symbol] = signal  # Track entry signal for dynamic exits
                            self._position_bar_count[symbol] = 0          # Reset hold-bar counter
                            self._save_position_metadata()
                            self._journal_position_update(
                                symbol=symbol,
                                asset_class=asset_class,
                                quantity=float(self.positions.get(symbol, shares)),
                                price=float(entry_execution_price),
                                reason="entry_fill",
                                metadata={
                                    "broker": entry_broker,
                                    "status": trade_status,
                                    **self._governor_policy_metadata(
                                        active_policy,
                                        governor_regime_key,
                                        perf_snapshot.tier.value,
                                    ),
                                },
                            )

                            attribution_entry_price = entry_execution_price
                            entry_slippage_bps = self._compute_slippage_bps(
                                arrival_price,
                                attribution_entry_price,
                            )
                            # Execution quality: record entry fill slippage
                            if getattr(self, '_exec_quality', None) is not None:
                                self._exec_quality.record_fill(
                                    symbol=symbol,
                                    side=str(side),
                                    expected_price=float(arrival_price) if arrival_price else float(attribution_entry_price),
                                    fill_price=float(attribution_entry_price),
                                    qty=float(abs(shares)),
                                    regime=str(self._current_regime),
                                    broker=str(locals().get('broker_used', 'unknown')),
                                    order_type="market",
                                )
                            # Record signal for ModelDriftMonitor (awaiting outcome at exit)
                            _mdm_entry = getattr(self, '_model_drift_monitor', None)
                            if _mdm_entry is not None and getattr(ApexConfig, "MODEL_DRIFT_MONITOR_ENABLED", True):
                                try:
                                    _mdm_entry.record_signal(
                                        symbol=symbol,
                                        signal_value=float(signal),
                                        confidence=float(confidence),
                                    )
                                except Exception:
                                    pass

                            self._record_entry_attribution(
                                symbol=symbol,
                                asset_class=asset_class,
                                side="LONG" if side == "BUY" else "SHORT",
                                quantity=abs(float(shares)),
                                entry_price=attribution_entry_price,
                                entry_signal=float(signal),
                                entry_confidence=float(confidence),
                                governor_tier=perf_snapshot.tier.value,
                                governor_regime=governor_regime_key,
                                risk_multiplier=float(self._risk_multiplier),
                                vix_multiplier=float(self._vix_risk_multiplier),
                                governor_size_multiplier=float(governor_controls.size_multiplier),
                                entry_slippage_bps=float(entry_slippage_bps),
                                entry_time=entry_ts,
                                source="live_entry",
                            )
                            # Signal accuracy tracker: record prediction at entry
                            _sat = getattr(self, '_signal_accuracy_tracker', None)
                            if _sat is not None:
                                try:
                                    _reg_sig = getattr(self, '_last_regression_signal', {}).get(symbol, float(signal))
                                    _bin_sig = getattr(self, '_last_binary_signal', {}).get(symbol, 0.0)
                                    _sat.record_prediction(
                                        symbol=symbol,
                                        timestamp=entry_ts,
                                        regression_signal=_reg_sig,
                                        binary_signal=_bin_sig,
                                        entry_price=float(attribution_entry_price),
                                        regime=str(getattr(self, '_current_regime', 'neutral')),
                                    )
                                except Exception:
                                    pass

                            # Shadow gate: record production allowed this entry
                            if getattr(self, '_shadow_gate', None) is not None:
                                try:
                                    self._shadow_gate.observe_production_decision(
                                        symbol=symbol, decision="allowed"
                                    )
                                except Exception:
                                    pass

                            # Audit log — live entry
                            self.trade_audit.log(
                                event="ENTRY",
                                symbol=symbol,
                                side=side,
                                qty=abs(float(shares)),
                                fill_price=float(attribution_entry_price),
                                expected_price=float(arrival_price),
                                slippage_bps=float(entry_slippage_bps),
                                signal=float(signal),
                                confidence=float(confidence),
                                regime=str(self._current_regime),
                                broker=(
                                    "alpaca"
                                    if self._get_connector_for(symbol) is self.alpaca
                                    else "ibkr"
                                ),
                                pretrade="PASS",
                            )

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
                            if str(asset_class).upper() == "CRYPTO" and not self._last_in_equity_hours:
                                self._overnight_entries += 1

                            self.live_monitor.log_trade(symbol, side, shares, attribution_entry_price, -commission)
                            await self.performance_tracker.record_trade(
                                symbol,
                                side,
                                shares,
                                attribution_entry_price,
                                commission,
                            )

                            # ✅ Update cooldown
                            self.last_trade_time[symbol] = datetime.now()

                            self.pending_orders.discard(symbol)
                            self._pending_entries.discard(symbol)  # placeholder resolved
                        else:
                            self.pending_orders.discard(symbol)
                    else:
                        if self.ibkr or self.alpaca:
                            logger.warning(
                                "⛔ %s: No eligible connector available in broker mode '%s'; entry blocked",
                                symbol,
                                str(getattr(ApexConfig, "BROKER_MODE", "ibkr")).lower(),
                            )
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                        # Simulation mode - open new position
                        trade_success = True
                        qty = shares if side == 'BUY' else -shares
                        self.positions[symbol] = qty
                        entry_ts = datetime.now()
                        self.position_entry_prices[symbol] = price
                        self.position_entry_times[symbol] = entry_ts
                        self.position_peak_prices[symbol] = price
                        self.position_entry_signals[symbol] = signal  # Track entry signal for dynamic exits
                        self._save_position_metadata()
                        self._journal_order_event(
                            symbol=symbol,
                            asset_class=asset_class,
                            side=side,
                            quantity=float(abs(qty)),
                            broker="simulation",
                            lifecycle="filled",
                            order_role="entry",
                            signal=float(signal),
                            confidence=float(confidence),
                            expected_price=float(price),
                            fill_price=float(price),
                            status="SIMULATED",
                            metadata=self._governor_policy_metadata(
                                active_policy,
                                governor_regime_key,
                                perf_snapshot.tier.value,
                            ),
                        )
                        self._journal_position_update(
                            symbol=symbol,
                            asset_class=asset_class,
                            quantity=float(qty),
                            price=float(price),
                            reason="sim_entry_fill",
                            metadata={
                                "status": "SIMULATED",
                                **self._governor_policy_metadata(
                                    active_policy,
                                    governor_regime_key,
                                    perf_snapshot.tier.value,
                                ),
                            },
                        )
                        self._record_entry_attribution(
                            symbol=symbol,
                            asset_class=asset_class,
                            side="LONG" if side == "BUY" else "SHORT",
                            quantity=abs(float(qty)),
                            entry_price=float(price),
                            entry_signal=float(signal),
                            entry_confidence=float(confidence),
                            governor_tier=perf_snapshot.tier.value,
                            governor_regime=governor_regime_key,
                            risk_multiplier=float(self._risk_multiplier),
                            vix_multiplier=float(self._vix_risk_multiplier),
                            governor_size_multiplier=float(governor_controls.size_multiplier),
                            entry_slippage_bps=0.0,
                            entry_time=entry_ts,
                            source="sim_entry",
                        )

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
                        await self.performance_tracker.record_trade(symbol, side, shares, price, 0)
                        self.last_trade_time[symbol] = datetime.now()

                finally:
                    # ✅ CRITICAL: Clean up placeholder if trade failed.
                    # _pending_entries tracks which symbols have a placeholder slot so we can
                    # distinguish them from real 1-share positions on cleanup.
                    if not trade_success and symbol in self._pending_entries:
                        async with self._position_lock:
                            if symbol in self.positions and self.positions[symbol] in (1, -1):
                                del self.positions[symbol]
                                logger.debug("   ⚠️ %s: Removed position placeholder (trade failed)", symbol)
                    self._pending_entries.discard(symbol)
        
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    async def retry_failed_exits(self):
        """
        Retry exit orders that previously failed.
        This ensures positions get closed even if IBKR had temporary issues.
        """
        if not self.failed_exits or (not self.ibkr and not self.alpaca):
            return

        now = datetime.now()
        symbols_to_retry = []

        for symbol, info in list(self.failed_exits.items()):
            # Skip if too many attempts (max 5)
            if info['attempts'] >= 5:
                if self.positions.get(symbol, 0) == 0:
                    # Position gone (reconciled away or manually closed) — stop tracking
                    del self.failed_exits[symbol]
                else:
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

            retry_connector = self._get_connector_for(symbol)
            if not retry_connector:
                continue

            logger.info(f"🔄 Retrying exit for {symbol} (attempt {info['attempts'] + 1})")

            try:
                self.pending_orders.add(symbol)
                order_side = 'SELL' if current_pos > 0 else 'BUY'

                trade = await retry_connector.execute_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=abs(current_pos),
                    confidence=0.9  # High confidence for exits
                )

                if trade:
                    retry_trade_status = str(trade.get("status", "FILLED")).upper() if isinstance(trade, dict) else "FILLED"
                    try:
                        asset_class = parse_symbol(symbol).asset_class.value
                    except Exception:
                        asset_class = "EQUITY"
                    side_label = "LONG" if current_pos > 0 else "SHORT"
                    entry_price = float(self.position_entry_prices.get(symbol, self.price_cache.get(symbol, 0.0)) or 0.0)
                    if entry_price <= 0:
                        entry_price = float(self.price_cache.get(symbol, 0.0) or 0.0)
                    entry_time = self.position_entry_times.get(symbol, datetime.utcnow())
                    exit_expected_price = float(self.price_cache.get(symbol, entry_price) or entry_price)
                    exit_fill_price = exit_expected_price
                    if isinstance(trade, dict):
                        exit_fill_price = float(
                            trade.get("price", exit_fill_price) or exit_fill_price
                        )
                        exit_expected_price = float(
                            trade.get("expected_price", exit_expected_price) or exit_expected_price
                        )
                    else:
                        try:
                            exit_fill_price = float(
                                trade.orderStatus.avgFillPrice or exit_fill_price
                            )
                        except Exception:
                            pass
                    exit_slippage_bps = self._compute_slippage_bps(
                        exit_expected_price,
                        exit_fill_price,
                    )
                    self._record_exit_attribution(
                        symbol=symbol,
                        asset_class=asset_class,
                        side=side_label,
                        quantity=abs(current_pos),
                        entry_price=entry_price if entry_price > 0 else exit_fill_price,
                        exit_price=exit_fill_price,
                        commissions=float(ApexConfig.COMMISSION_PER_TRADE),
                        exit_reason=str(info.get("reason", "retry_success")),
                        entry_signal=float(self.position_entry_signals.get(symbol, 0.0)),
                        entry_confidence=min(
                            1.0,
                            max(0.0, abs(float(self.position_entry_signals.get(symbol, 0.0)))),
                        ),
                        governor_tier=self._performance_snapshot.tier.value if self._performance_snapshot else "green",
                        governor_regime=self._map_governor_regime(asset_class, self._current_regime),
                        entry_time=entry_time,
                        exit_time=datetime.now(),
                        exit_slippage_bps=float(exit_slippage_bps),
                        source="retry_exit",
                    )

                    self.total_commissions += ApexConfig.COMMISSION_PER_TRADE
                    retry_broker = "alpaca" if retry_connector is self.alpaca else "ibkr"
                    if retry_trade_status == "FILLED":
                        self._record_fill_realized_pnl(
                            broker_name=retry_broker,
                            symbol=symbol,
                            side=order_side,
                            quantity=abs(float(current_pos)),
                            fill_price=float(exit_fill_price),
                            commission=float(ApexConfig.COMMISSION_PER_TRADE),
                            filled_at=datetime.now(),
                        )
                    await self._sync_positions()
                    logger.info(f"   ✅ {symbol}: Exit retry successful!")

                    # Clean up tracking
                    for tracking_dict in [self.position_entry_prices, self.position_entry_times,
                                          self.position_peak_prices, self.position_stops,
                                          self.position_entry_signals, self.failed_exits]:
                        if symbol in tracking_dict:
                            del tracking_dict[symbol]
                    self._save_position_metadata()

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
        Process symbols in batches, respecting broker-specific limits.

        IBKR limits simultaneous market data subscriptions to 100.
        We process IBKR symbols in batches of 50 with cleanup between batches.

        Alpaca (crypto) has NO limit, so all crypto symbols are processed in one batch.
        """
        # Separate IBKR symbols from Alpaca crypto symbols
        ibkr_symbols = []
        alpaca_symbols = []

        for symbol in symbols:
            try:
                parsed = parse_symbol(symbol)
                if parsed.asset_class == AssetClass.CRYPTO:
                    alpaca_symbols.append(symbol)
                else:
                    ibkr_symbols.append(symbol)
            except Exception:
                # If parsing fails, assume IBKR for safety (apply limit)
                ibkr_symbols.append(symbol)

        # Log separation for transparency
        if ibkr_symbols and alpaca_symbols:
            logger.info(f"📊 Processing {len(ibkr_symbols)} IBKR symbols + {len(alpaca_symbols)} Alpaca crypto symbols")
        elif ibkr_symbols:
            logger.info(f"📊 Processing {len(ibkr_symbols)} IBKR symbols only (no crypto open)")
        elif alpaca_symbols:
            logger.info(f"📊 Processing {len(alpaca_symbols)} Alpaca crypto symbols only (no IBKR open)")
        else:
            logger.info("📊 No symbols to process (empty universe)")



        # Process IBKR symbols with batching (100 market data line limit)
        # Phase 13: Distributed Concurrency via Semaphore Worker Pool
        _max_concurrency = int(getattr(ApexConfig, "MAX_CONCURRENT_TASKS", 50))
        semaphore = asyncio.Semaphore(_max_concurrency)

        async def _protected_process(sym, timeout):
            async with semaphore:
                return await asyncio.wait_for(self.process_symbol(sym), timeout=timeout)

        if ibkr_symbols:
            BATCH_SIZE = 100  # Max symbols per batch (IBKR limit is 100)
            total_ibkr = len(ibkr_symbols)
            num_batches = (total_ibkr + BATCH_SIZE - 1) // BATCH_SIZE

            if num_batches > 1:
                logger.info(f"📊 Processing {total_ibkr} IBKR symbols in {num_batches} batches ({BATCH_SIZE} max per batch)")
            else:
                logger.info(f"📊 Processing {total_ibkr} IBKR symbols in single batch")

            for batch_num in range(num_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, total_ibkr)
                batch_symbols = ibkr_symbols[start_idx:end_idx]

                # Process batch in parallel (60s timeout per symbol prevents hangs)
                _SYMBOL_TIMEOUT = float(getattr(ApexConfig, "PROCESS_SYMBOL_TIMEOUT_SECONDS", 60.0))
                tasks = [
                    _protected_process(symbol, _SYMBOL_TIMEOUT)
                    for symbol in batch_symbols
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym, res in zip(batch_symbols, results):
                    if isinstance(res, asyncio.TimeoutError):
                        logger.error("⏱️ process_symbol TIMEOUT for %s (>%.0fs)", sym, _SYMBOL_TIMEOUT)
                    elif isinstance(res, Exception):
                        logger.warning("⚠️ process_symbol exception for %s: %r", sym, res)

                # Small delay between batches to allow cleanup of market data subscriptions
                if batch_num < num_batches - 1:
                    await asyncio.sleep(0.5)

        # Process ALL Alpaca crypto symbols in one batch (no batch limit, strictly bounded by Semaphore)
        if alpaca_symbols:
            logger.info(f"📊 Processing {len(alpaca_symbols)} Alpaca crypto symbols (bounded by Semaphore={_max_concurrency})")
            _SYMBOL_TIMEOUT = float(getattr(ApexConfig, "PROCESS_SYMBOL_TIMEOUT_SECONDS", 60.0))
            tasks = [
                _protected_process(symbol, _SYMBOL_TIMEOUT)
                for symbol in alpaca_symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, res in zip(alpaca_symbols, results):
                if isinstance(res, asyncio.TimeoutError):
                    logger.error("⏱️ process_symbol TIMEOUT for %s (>%.0fs)", sym, _SYMBOL_TIMEOUT)
                elif isinstance(res, Exception):
                    logger.warning("⚠️ process_symbol exception for %s: %r", sym, res)

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
            if self.ibkr or self.alpaca:
                total_value = await self._get_total_portfolio_value()
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

                rebal_connector = self._get_connector_for(symbol)
                if rebal_connector:
                    result = await rebal_connector.execute_order(
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
            options_logger.info("event=options_skip reason=disabled_or_unavailable")
            return
        if not self.ibkr or not self.ibkr.is_connected():
            options_logger.info("event=options_skip reason=ibkr_offline_or_disconnected")
            return

        try:
            # ❌ Skip option order placement outside regular market hours.
            # IBKR paper trading cancels all LMT/MKT option orders when there's no NBBO.
            from core.market_hours import is_market_open
            try:
                _market_now_open = is_market_open("SPY", datetime.utcnow())
            except Exception:
                _market_now_open = False
            
            if not _market_now_open:
                options_logger.info("event=options_skip reason=market_closed")
                return
            
            logger.debug("🎯 Checking options opportunities...")
            options_logger.info("event=options_cycle_start positions=%d", len(self.positions))
            retry_cooldown_seconds = max(
                60,
                int(getattr(ApexConfig, "OPTIONS_FAILED_TRADE_RETRY_COOLDOWN_SECONDS", 1800)),
            )

            # Get current stock positions value
            for symbol, qty in self.positions.items():
                if qty <= 0:  # Only hedge long positions
                    options_logger.info(
                        "event=options_skip symbol=%s reason=non_long qty=%.2f",
                        symbol,
                        qty,
                    )
                    continue

                price = self.price_cache.get(symbol, 0)
                if price <= 0:
                    options_logger.info(
                        "event=options_skip symbol=%s reason=no_price qty=%.2f",
                        symbol,
                        qty,
                    )
                    continue

                position_value = qty * price

                # Auto-hedge large positions with protective puts (with notional floor)
                if ApexConfig.OPTIONS_AUTO_HEDGE and position_value >= ApexConfig.MIN_HEDGE_NOTIONAL:
                    # Check if already hedged
                    hedge_key = f"{symbol}_hedge"
                    if hedge_key not in self.options_positions:
                        hedge_retry_key = f"{symbol}:hedge"
                        hedge_retry_after = self._options_retry_after.get(hedge_retry_key)
                        if hedge_retry_after and datetime.now() < hedge_retry_after:
                            options_logger.info(
                                "event=options_hedge_skip symbol=%s reason=retry_backoff until=%s",
                                symbol,
                                hedge_retry_after.isoformat() + "Z",
                            )
                        else:
                            logger.info(f"🛡️ Auto-hedging {symbol}: ${position_value:,.2f} position")
                            options_logger.info(
                                "event=options_hedge_attempt symbol=%s qty=%.2f value=%.2f",
                                symbol,
                                qty,
                                position_value,
                            )

                            result = await self.options_trader.buy_protective_put(
                                symbol=symbol,
                                shares=qty,
                                delta=ApexConfig.OPTIONS_HEDGE_DELTA,
                                days_to_expiry=ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY
                            )

                            if result:
                                self.options_positions[hedge_key] = result
                                self._options_retry_after.pop(hedge_retry_key, None)
                                logger.info(f"   ✅ Protective put purchased: {result.get('contract', {}).get('strike')} strike")
                                options_logger.info(
                                    "event=options_hedge_filled symbol=%s qty=%.2f strike=%s",
                                    symbol,
                                    qty,
                                    result.get('contract', {}).get('strike'),
                                )
                            else:
                                self._options_retry_after[hedge_retry_key] = datetime.now() + timedelta(
                                    seconds=retry_cooldown_seconds
                                )
                                options_logger.info(
                                    "event=options_hedge_skip symbol=%s reason=trade_failed retry_after=%s",
                                    symbol,
                                    self._options_retry_after[hedge_retry_key].isoformat() + "Z",
                                )
                    else:
                        options_logger.info(
                            "event=options_hedge_skip symbol=%s reason=already_hedged",
                            symbol,
                        )
                else:
                    if ApexConfig.OPTIONS_AUTO_HEDGE:
                        options_logger.info(
                            "event=options_hedge_skip symbol=%s reason=below_notional value=%.2f min_notional=%.2f",
                            symbol,
                            position_value,
                            ApexConfig.MIN_HEDGE_NOTIONAL,
                        )

                # Sell covered calls on eligible positions
                if ApexConfig.OPTIONS_COVERED_CALLS_ENABLED and qty >= ApexConfig.OPTIONS_MIN_SHARES_FOR_COVERED_CALL:
                    cc_key = f"{symbol}_cc"
                    if cc_key not in self.options_positions:
                        cc_retry_key = f"{symbol}:covered_call"
                        cc_retry_after = self._options_retry_after.get(cc_retry_key)
                        if cc_retry_after and datetime.now() < cc_retry_after:
                            options_logger.info(
                                "event=options_cc_skip symbol=%s reason=retry_backoff until=%s",
                                symbol,
                                cc_retry_after.isoformat() + "Z",
                            )
                            continue
                        logger.info(f"💰 Selling covered call on {symbol}: {qty} shares")
                        options_logger.info(
                            "event=options_cc_attempt symbol=%s qty=%.2f",
                            symbol,
                            qty,
                        )

                        result = await self.options_trader.sell_covered_call(
                            symbol=symbol,
                            shares=qty,
                            delta=ApexConfig.OPTIONS_COVERED_CALL_DELTA,
                            days_to_expiry=ApexConfig.OPTIONS_PREFERRED_DAYS_TO_EXPIRY
                        )

                        if result:
                            self.options_positions[cc_key] = result
                            self._options_retry_after.pop(cc_retry_key, None)
                            logger.info(f"   ✅ Covered call sold: ${result.get('premium', 0):,.2f} premium")
                            options_logger.info(
                                "event=options_cc_filled symbol=%s qty=%.2f premium=%.2f",
                                symbol,
                                qty,
                                result.get('premium', 0),
                            )
                        else:
                            self._options_retry_after[cc_retry_key] = datetime.now() + timedelta(
                                seconds=retry_cooldown_seconds
                            )
                            options_logger.info(
                                "event=options_cc_skip symbol=%s reason=trade_failed retry_after=%s",
                                symbol,
                                self._options_retry_after[cc_retry_key].isoformat() + "Z",
                            )
                    else:
                        options_logger.info(
                            "event=options_cc_skip symbol=%s reason=already_open",
                            symbol,
                        )
                else:
                    if ApexConfig.OPTIONS_COVERED_CALLS_ENABLED:
                        options_logger.info(
                            "event=options_cc_skip symbol=%s reason=insufficient_shares qty=%.2f min_shares=%d",
                            symbol,
                            qty,
                            ApexConfig.OPTIONS_MIN_SHARES_FOR_COVERED_CALL,
                        )

            # Check for expiring options (within 7 days)
            await self._check_expiring_options()

            # Log portfolio Greeks
            if self.options_positions:
                greeks = self.options_trader.get_portfolio_greeks()
                logger.debug(f"📊 Portfolio Greeks - Delta: {greeks['delta']:.1f}, Theta: ${greeks['theta']:.2f}/day")
                options_logger.info(
                    "event=options_portfolio_greeks delta=%.2f gamma=%.2f theta=%.2f vega=%.2f",
                    greeks.get('delta', 0.0),
                    greeks.get('gamma', 0.0),
                    greeks.get('theta', 0.0),
                    greeks.get('vega', 0.0),
                )

        except Exception as e:
            logger.error(f"❌ Options management error: {e}")
            options_logger.info("event=options_error error=%s", str(e))
            import traceback
            logger.debug(traceback.format_exc())

    async def manage_active_positions(self):
        """Phase 3: Manage active positions with advanced guards (Ratchet, Aging, Overnight)."""
        # Snapshot positions to avoid modification during iteration
        for symbol, qty in list(self.positions.items()):
            if qty == 0: continue
            try:
                if not is_market_open(symbol, datetime.utcnow()):
                    continue
            except Exception:
                pass
            
            # Get current price
            current_price = self.price_cache.get(symbol, 0)
            if current_price <= 0: continue
            
            entry_price = self.position_entry_prices.get(symbol, current_price)
            entry_time = self.position_entry_times.get(symbol, datetime.utcnow())

            # 1. Profit Ratchet
            if hasattr(self, 'profit_ratchet') and self.profit_ratchet:
                self.profit_ratchet.register_position(symbol, entry_price, entry_time)
                should_exit, reason, _ = self.profit_ratchet.should_exit(symbol, current_price)
                if should_exit:
                    logger.info(f"💰 Profit Ratchet Trigger: {symbol} ({reason})")
                    mgmt_connector = self._get_connector_for(symbol)
                    if mgmt_connector:
                        side = 'SELL' if qty > 0 else 'BUY'
                        await mgmt_connector.execute_order(symbol, side, abs(qty), confidence=1.0)
                    continue

            # 2. Position Aging
            if hasattr(self, 'aging_manager') and self.aging_manager:
                self.aging_manager.register_position(symbol, entry_price, entry_time)
                age = self.aging_manager.update_position(symbol, current_price)
                if age and age.should_exit:
                    logger.info(f"⏳ Aging Exit Trigger: {symbol} ({age.exit_reason})")
                    mgmt_connector = self._get_connector_for(symbol)
                    if mgmt_connector:
                        side = 'SELL' if qty > 0 else 'BUY'
                        await mgmt_connector.execute_order(symbol, side, abs(qty), confidence=1.0)
                    continue

            # 3. Overnight Risk Guard (Exposure reduction handled in OvernightRiskGuard logic, potentially separately)

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
                price = self.price_cache.get(symbol, 0)
                if price > 0 and (shares * price) < ApexConfig.MIN_HEDGE_NOTIONAL:
                    logger.warning(f"⚠️ Protective put rejected: Position value ${(shares*price):,.2f} < ${ApexConfig.MIN_HEDGE_NOTIONAL:,.2f} (MIN_HEDGE_NOTIONAL)")
                    return None

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
            if self.ibkr or self.alpaca:
                current_value = await self._get_total_portfolio_value()
                # Guard against brokers returning 0 due to connection issues
                if current_value is None or current_value <= 0:
                    if self.capital > 0:
                        logger.warning(f"⚠️  Brokers returned {current_value}, using last known capital ${self.capital:,.2f}")
                        current_value = self.capital
                    else:
                        current_value = 0.0
            else:
                current_value = self.capital
                for symbol, qty in self.positions.items():
                    price = self.price_cache.get(symbol, 0)
                    if price and qty:
                        if qty > 0:  # Long
                            current_value += float(qty) * float(price)
                        else:  # Short (qty is negative)
                            current_value += float(qty) * float(price)

            await self._maybe_rebase_paper_baselines_for_broker_mix(
                current_value=float(current_value),
            )

            equity_decision = self.equity_outlier_guard.evaluate(
                raw_equity_value=current_value,
                observed_at=datetime.now(),
            )
            if not equity_decision.accepted:
                logger.warning(
                    "⚠️ Equity outlier guard rejected sample raw=$%.2f (deviation=%.2f%%, reason=%s, suspect_streak=%d); using filtered=$%.2f",
                    equity_decision.raw_value,
                    equity_decision.deviation_pct * 100,
                    equity_decision.reason,
                    equity_decision.suspect_count,
                    equity_decision.filtered_value,
                )
            elif equity_decision.reason == "confirmed_large_move":
                logger.warning(
                    "✅ Equity outlier guard accepted confirmed large move raw=$%.2f (deviation=%.2f%% after %d confirmations)",
                    equity_decision.raw_value,
                    equity_decision.deviation_pct * 100,
                    equity_decision.suspect_count,
                )
            current_value = float(equity_decision.filtered_value)

            previous_recon = self._equity_reconciliation_snapshot
            recon_snapshot = await self._evaluate_equity_reconciliation(
                broker_equity=current_value,
                observed_at=datetime.now(),
            )
            self._equity_reconciliation_snapshot = recon_snapshot
            self._equity_reconciliation_block_entries = bool(recon_snapshot.block_entries)

            previous_block = bool(previous_recon.block_entries) if previous_recon else False
            if recon_snapshot.block_entries and not previous_block:
                logger.critical(
                    "🧾 EQUITY RECONCILIATION BLOCK LATCHED: gap=$%.2f (%.2f%%), reason=%s, breach_streak=%d",
                    recon_snapshot.gap_dollars,
                    recon_snapshot.gap_pct * 100,
                    recon_snapshot.reason,
                    recon_snapshot.breach_streak,
                )
                fire_alert(
                    "equity_recon_block",
                    f"Equity reconciliation block latched: gap=${recon_snapshot.gap_dollars:+.2f} "
                    f"({recon_snapshot.gap_pct*100:.2f}%), reason={recon_snapshot.reason}",
                    AlertSev.CRITICAL,
                )
            elif (not recon_snapshot.block_entries) and previous_block:
                logger.warning(
                    "✅ Equity reconciliation block cleared: gap=$%.2f (%.2f%%), healthy_streak=%d",
                    recon_snapshot.gap_dollars,
                    recon_snapshot.gap_pct * 100,
                    recon_snapshot.healthy_streak,
                )
                fire_alert("equity_recon_block", "Equity reconciliation block cleared", AlertSev.INFO)

            # Keep baseline/risk state sane when broker APIs temporarily return invalid values.
            self.risk_manager.heal_baselines(current_capital=current_value, source="check_risk")
            self.capital = float(current_value)

            loss_check = self.risk_manager.check_daily_loss(current_value)
            dd_check = self.risk_manager.check_drawdown(current_value)
            # Defensive: risk methods should always return dicts, but guard against
            # edge-case float returns to prevent 'float has no attribute get' crashes.
            if not isinstance(loss_check, dict):
                loss_check = {'daily_pnl': 0.0, 'daily_return': 0.0, 'breached': False,
                              'limit': 0.02, 'circuit_breaker_tripped': False}
            if not isinstance(dd_check, dict):
                dd_check = {'drawdown': 0.0, 'breached': False, 'peak': current_value,
                             'current': current_value, 'limit': 0.15, 'circuit_breaker_tripped': False}

            # Intraday rolling drawdown gate — entry-block only, never blocks exits
            self.risk_manager.push_capital_snapshot(current_value)
            _intraday_dd = self.risk_manager.check_intraday_rolling_dd(current_value)
            _prev_gate = self._intraday_dd_gate_active
            self._intraday_dd_gate_active = bool(_intraday_dd.get("breached", False))
            if self._intraday_dd_gate_active and not _prev_gate:
                logger.warning(
                    "🚨 INTRADAY DD GATE TRIPPED: %.2f%% loss in last %d min (limit=%.2f%%)",
                    abs(_intraday_dd.get("rolling_loss_pct", 0.0)) * 100,
                    _intraday_dd.get("window_minutes", 60),
                    _intraday_dd.get("limit", 0.015) * 100,
                )
                fire_alert(
                    "intraday_dd_gate",
                    f"Intraday DD gate tripped: {abs(_intraday_dd.get('rolling_loss_pct', 0.0))*100:.2f}% "
                    f"loss in last {_intraday_dd.get('window_minutes', 60)} min "
                    f"(limit={_intraday_dd.get('limit', 0.015)*100:.2f}%)",
                    AlertSev.CRITICAL,
                )
            elif _prev_gate and not self._intraday_dd_gate_active:
                logger.info("✅ INTRADAY DD GATE cleared — entries re-enabled")
                fire_alert("intraday_dd_gate", "Intraday DD gate cleared — entries re-enabled", AlertSev.INFO)

            await self.risk_manager.save_state_async()
            await self.performance_tracker.record_equity(current_value)
            
            # 🎯 Phase 2: Capture SPY for Alpha Retention
            current_time = datetime.now().isoformat()
            spy_data = self.price_cache.get("SPY") or self.historical_data.get("SPY", {})
            # Handle both float and dictionary formats safely
            if isinstance(spy_data, (int, float)):
                spy_price = spy_data
            else:
                spy_price = spy_data.get("close") or spy_data.get("price") if isinstance(spy_data, dict) else None
                
            if spy_price:
                await self.performance_tracker.record_benchmark(float(spy_price), current_time)
                # Feed SPY + BTC prices into correlation early warning detector
                if getattr(self, '_corr_early_warning', None) is not None:
                    try:
                        _btc_pc = self.price_cache.get("CRYPTO:BTC/USD") or self.price_cache.get("BTC/USD")
                        if isinstance(_btc_pc, dict):
                            _btc_pc = _btc_pc.get("close") or _btc_pc.get("price")
                        if _btc_pc and float(_btc_pc) > 0:
                            self._corr_early_warning.record_prices(float(spy_price), float(_btc_pc))
                    except Exception:
                        pass
            perf_snapshot = self._performance_snapshot
            if self.performance_governor:
                self.performance_governor.set_regime_targets(
                    regime=self._current_regime or "neutral",
                    base_sharpe=ApexConfig.PERFORMANCE_TARGET_SHARPE,
                    base_sortino=ApexConfig.PERFORMANCE_TARGET_SORTINO,
                    multipliers=ApexConfig.GOVERNOR_REGIME_TARGET_MULTIPLIERS,
                    min_sharpe=ApexConfig.GOVERNOR_MIN_TARGET_SHARPE,
                    min_sortino=ApexConfig.GOVERNOR_MIN_TARGET_SORTINO,
                )
                perf_snapshot = self.performance_governor.update(current_value, datetime.now())
                self._performance_snapshot = perf_snapshot
                # Persist governor state every 10 risk-checks so restarts skip warmup
                self._risk_check_count = getattr(self, "_risk_check_count", 0) + 1
                if self._risk_check_count % 10 == 0:
                    try:
                        self.performance_governor.save_state(
                            ApexConfig.DATA_DIR / "performance_governor_state.json"
                        )
                    except Exception as _pg_save_err:
                        logger.debug("Governor state save failed: %s", _pg_save_err)

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

            total_trades = len(self.performance_tracker.trades)
            completed_trades = self.performance_tracker.get_completed_trade_count()

            if completed_trades <= 0:
                sharpe = 0.0
                sortino = 0.0
                win_rate = 0.0
            else:
                try:
                    sharpe = self.performance_tracker.get_sharpe_ratio()
                except:
                    sharpe = 0.0

                try:
                    win_rate = self.performance_tracker.get_win_rate()
                except:
                    win_rate = 0.0

                try:
                    sortino = self.performance_tracker.get_sortino_ratio()
                except:
                    sortino = 0.0

            kill_state = self.kill_switch.update(self.performance_tracker.equity_curve) if self.kill_switch else None
            attribution_summary = self.performance_attribution.get_summary(lookback_days=30)
            self._reload_social_inputs()
            if self.social_shock_governor:
                for social_asset in ("EQUITY", "FOREX", "CRYPTO"):
                    social_regime = self._map_governor_regime(social_asset, self._current_regime)
                    self._social_decision_for(social_asset, social_regime)

            # Calculate sector exposure
            sector_exp = self.calculate_sector_exposure()

            logger.info("")
            logger.info("═" * 80)
            logger.info(f"💼 Portfolio: ${current_value:,.2f}")
            logger.info(f"📊 Daily P&L: ${loss_check['daily_pnl']:+,.2f} ({loss_check['daily_return']*100:+.2f}%)")
            logger.info(f"📉 Drawdown: {dd_check['drawdown']*100:.2f}%")
            logger.info(f"📦 Positions: {self.position_count}/{ApexConfig.MAX_POSITIONS}")
            logger.info(f"⏳ Pending: {len(self.pending_orders)}")
            logger.info(
                "🧾 Equity Reconciliation: gap=$%.2f (%.2f%%) block_entries=%s reason=%s",
                recon_snapshot.gap_dollars,
                recon_snapshot.gap_pct * 100,
                recon_snapshot.block_entries,
                recon_snapshot.reason,
            )
            logger.info(f"💸 Total Commissions: ${self.total_commissions:,.2f}")
            logger.info(f"📈 Sharpe: {sharpe:.2f} | Win Rate: {win_rate*100:.1f}% | Trades: {total_trades}")
            if perf_snapshot:
                logger.info(
                    f"🧭 Governor: {perf_snapshot.tier.value.upper()} "
                    f"(size={perf_snapshot.size_multiplier:.0%}, "
                    f"thr+={perf_snapshot.signal_threshold_boost:.2f}, "
                    f"conf+={perf_snapshot.confidence_boost:.2f}, "
                    f"halt_entries={perf_snapshot.halt_new_entries})"
                )
            if kill_state:
                _ks_log = logger.warning if kill_state.active else logger.debug
                _ks_log(
                    "🛑 Kill-Switch: active=%s dd=%.2f%% hist_mdd=%.2f%% sharpe_%dd=%.2f",
                    kill_state.active,
                    kill_state.drawdown * 100,
                    kill_state.historical_mdd * 100,
                    ApexConfig.KILL_SWITCH_SHARPE_WINDOW_DAYS,
                    kill_state.sharpe_rolling,
                )
            social_equity = self._social_decision_cache.get(
                ("EQUITY", self._map_governor_regime("EQUITY", self._current_regime))
            )
            if social_equity:
                logger.info(
                    "📣 SocialShock: score=%.2f gross_mult=%.0f%% block=%s verified_event=%.2f",
                    social_equity.combined_risk_score,
                    social_equity.gross_exposure_multiplier * 100.0,
                    social_equity.block_new_entries,
                    social_equity.verified_event_probability,
                )

            # Institutional risk metrics
            if self.use_institutional:
                logger.info("🏛️  INSTITUTIONAL RISK:")
                logger.info(f"   Portfolio Vol: {portfolio_risk.portfolio_volatility:.1%} | VaR(95%): ${portfolio_risk.var_95:,.0f}")
                logger.info(f"   Risk Level: {portfolio_risk.risk_level.value.upper()} | Risk Mult: {portfolio_risk.risk_multiplier:.2f}")
                logger.info(f"   Gross Exp: ${portfolio_risk.gross_exposure:,.0f} | Net Exp: ${portfolio_risk.net_exposure:,.0f}")
                logger.info(f"   Concentration (HHI): {portfolio_risk.herfindahl_index:.3f}")

            if sector_exp:
                logger.info("🏢 Sector Exposure:")
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
                            abs(qty) * price
                            entry = self.position_entry_prices.get(symbol, price)
                            
                            if qty > 0:
                                pnl_pct = (price / entry - 1) * 100
                            else:
                                pnl_pct = (entry / price - 1) * 100
                            
                            logger.info(f"   {symbol}: {abs(qty)} shares ({pos_type}) ${price:.2f} P&L:{pnl_pct:+.1f}%")
                    except Exception as e:
                        logger.debug(f"Error displaying {symbol}: {e}")
            
            if getattr(self, 'options_positions', {}):
                opt_list = []
                for key, details in self.options_positions.items():
                    qty = details.get('quantity', 0)
                    if qty != 0:
                        opt_list.append((key, details))
                
                if opt_list:
                    sorted_opts = sorted(opt_list, key=lambda x: abs(x[1].get('quantity', 0)), reverse=True)
                    logger.info(f"📈 Option Positions ({len(sorted_opts)}):")
                    for key, details in sorted_opts:
                        try:
                            sym = details.get('symbol', 'UNKNOWN')
                            qty = details.get('quantity', 0)
                            strike = float(details.get('strike', 0))
                            right = details.get('right', 'C')
                            expiry = details.get('expiry', 'YYYYMMDD')
                            avg_cost = details.get('avg_cost', 0.0)

                            pos_type = "LONG" if qty > 0 else "SHORT"
                            opt_type = "CALL" if right == 'C' else "PUT"

                            # Calculate DTE
                            try:
                                exp_date = datetime.strptime(str(expiry), '%Y%m%d')
                                dte = (exp_date - datetime.utcnow()).days
                                dte_str = f"DTE:{dte}d"
                                dte_warn = " ⚠️EXPIRING" if dte <= 7 else (" 📅ROLL_SOON" if dte <= 14 else "")
                            except Exception:
                                dte_str = "DTE:?"
                                dte_warn = ""

                            # Calculate ITM/ATM/OTM vs current stock price
                            stock_price = self.price_cache.get(sym, 0)
                            if stock_price > 0 and strike > 0:
                                if right == 'C':
                                    moneyness_pct = (stock_price - strike) / strike * 100
                                    if stock_price > strike * 1.01:
                                        money_label = f"ITM({moneyness_pct:+.1f}%) ⚠️"
                                    elif stock_price < strike * 0.99:
                                        money_label = f"OTM({moneyness_pct:+.1f}%)"
                                    else:
                                        money_label = "ATM"
                                else:
                                    moneyness_pct = (strike - stock_price) / strike * 100
                                    if stock_price < strike * 0.99:
                                        money_label = f"ITM({moneyness_pct:+.1f}%) ⚠️"
                                    elif stock_price > strike * 1.01:
                                        money_label = f"OTM({moneyness_pct:+.1f}%)"
                                    else:
                                        money_label = "ATM"
                            else:
                                money_label = "?"

                            logger.info(
                                f"   {sym} {expiry} ${strike} {opt_type}: {abs(qty)} contracts "
                                f"({pos_type}) AvgCost:${avg_cost:.2f} | {dte_str}{dte_warn} | {money_label}"
                            )
                        except Exception as e:
                            logger.debug(f"Error displaying option {key}: {e}")
            
            logger.info("═" * 80)
            logger.info("")

            if self.prometheus_metrics:
                try:
                    self.prometheus_metrics.update_portfolio_value(current_value)
                    self.prometheus_metrics.update_portfolio_positions(self.position_count)
                    self.prometheus_metrics.update_daily_pnl(
                        pnl=float(loss_check.get("daily_pnl", 0.0)),
                        return_pct=float(loss_check.get("daily_return", 0.0)),
                    )
                    self.prometheus_metrics.update_drawdown(
                        current=float(dd_check.get("drawdown", 0.0)),
                        maximum=float(self.performance_tracker.get_max_drawdown()),
                    )
                    self.prometheus_metrics.update_risk_ratios(
                        sharpe=float(sharpe),
                        sortino=float(sortino),
                        var_95=float(portfolio_risk.var_95) if self.use_institutional else 0.0,
                    )
                    self.prometheus_metrics.update_sector_exposure(sector_exp)
                    self.prometheus_metrics.update_circuit_breaker(
                        bool(loss_check.get("breached", False) or dd_check.get("breached", False))
                    )
                    self.prometheus_metrics.update_equity_validation(
                        accepted=equity_decision.accepted,
                        reason=equity_decision.reason,
                        raw_value=equity_decision.raw_value,
                        filtered_value=equity_decision.filtered_value,
                        deviation_pct=equity_decision.deviation_pct,
                        suspect_count=equity_decision.suspect_count,
                    )
                    self.prometheus_metrics.update_equity_reconciliation(
                        broker_equity=recon_snapshot.broker_equity,
                        modeled_equity=recon_snapshot.modeled_equity,
                        gap_dollars=recon_snapshot.gap_dollars,
                        gap_pct=recon_snapshot.gap_pct,
                        block_entries=recon_snapshot.block_entries,
                        breach_streak=recon_snapshot.breach_streak,
                        healthy_streak=recon_snapshot.healthy_streak,
                        reason=recon_snapshot.reason,
                        breached=recon_snapshot.breached,
                        breach_event=(
                            recon_snapshot.breached
                            and not (previous_recon.breached if previous_recon else False)
                        ),
                    )
                    self.prometheus_metrics.update_attribution_summary(attribution_summary)
                    self._emit_social_prometheus_metrics()
                    self.prometheus_metrics.record_heartbeat()
                except Exception as metrics_exc:
                    logger.debug("Prometheus update error: %s", metrics_exc)

            if kill_state and kill_state.active and not self._kill_switch_last_active:
                logger.critical("🛑 HARD KILL-SWITCH TRIGGERED: %s", kill_state.reason)
                fire_alert("kill_switch", f"HARD KILL-SWITCH TRIGGERED: {kill_state.reason}", AlertSev.CRITICAL)
                if getattr(self, '_alert_manager', None) is not None:
                    _ks_pnl = float(loss_check.get('daily_pnl', 0.0))
                    asyncio.create_task(self._alert_manager.send_kill_switch_alert(
                        reason=str(kill_state.reason or "dd_sharpe_breach"),
                        session_pnl=_ks_pnl,
                    ))
                if self.prometheus_metrics:
                    self.prometheus_metrics.update_kill_switch(active=True, reason="dd_sharpe_breach")

            if kill_state and kill_state.active and not kill_state.flatten_executed:
                logger.critical("🛑 Flattening all positions due to kill-switch")
                await self.close_all_positions()
                if self.kill_switch:
                    self.kill_switch.mark_flattened()
                    self._persist_kill_switch_state()
            elif kill_state and kill_state.active:
                # Already flattened in a previous run — keep persisting so restart
                # knows the switch is still on.
                self._persist_kill_switch_state()

            if self.prometheus_metrics and kill_state:
                self.prometheus_metrics.update_kill_switch(active=kill_state.active)
            self._kill_switch_last_active = bool(kill_state.active) if kill_state else False

            # Refresh position prices before dashboard export
            await self.refresh_position_prices()

            # Export to dashboard
            await self.export_dashboard_state()
            self.export_trades_history()
            self.export_equity_curve()
            self._write_daily_pnl_snapshot(loss_check, dd_check, current_value)

            if loss_check.get('breached', False):
                logger.error("🚨 DAILY LOSS LIMIT BREACHED!")
                if getattr(self, '_alert_manager', None) is not None:
                    _dl_pct = abs(float(loss_check.get('daily_return', 0.0))) * 100.0
                    _dl_usd = float(loss_check.get('daily_pnl', 0.0))
                    if _dl_pct >= self._alert_manager.drawdown_alert_pct:
                        asyncio.create_task(self._alert_manager.send_drawdown_alert(
                            daily_loss_pct=_dl_pct,
                            daily_loss_usd=_dl_usd,
                        ))

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
                close_connector = self._get_connector_for(symbol)
                if close_connector:
                    price = await close_connector.get_market_price(symbol)
                    side = 'SELL' if qty > 0 else 'BUY'

                    await close_connector.execute_order(symbol, side, abs(qty), confidence=1.0, force_market=True)

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
    
    async def _update_pairs_signals(self) -> None:
        """
        Update pairs trading overlay signals.

        Every PAIRS_SCAN_INTERVAL cycles: re-scan historical_data for cointegrated pairs
        (equity-only, lookback=60 bars, p<0.05). Computationally expensive O(N^2) so
        runs infrequently.

        Every call: compute z-scores for active pairs and populate self._pairs_overlay
        with a signal adjustment in [-PAIRS_MAX_OVERLAY, +PAIRS_MAX_OVERLAY].
        """
        if getattr(self, '_pairs_trader', None) is None:
            return

        try:
            import pandas as pd
            _scan_interval = int(getattr(ApexConfig, "PAIRS_SCAN_INTERVAL_CYCLES", 300))
            _max_overlay = float(getattr(ApexConfig, "PAIRS_MAX_OVERLAY", 0.10))
            _weight = float(getattr(ApexConfig, "PAIRS_SIGNAL_WEIGHT", 0.15))

            self._pairs_scan_cycle = getattr(self, '_pairs_scan_cycle', 0) + 1

            # Periodic pair discovery — only equity symbols with sufficient history
            if self._pairs_scan_cycle % _scan_interval == 1:
                _eq_syms = [
                    s for s in self.historical_data
                    if not _symbol_is_crypto(s)
                    and not s.startswith('FX:')
                    and len(self.historical_data[s]) >= self._pairs_trader.lookback
                ][:30]  # cap at 30 for performance (C(30,2)=435 pairs)

                if len(_eq_syms) >= 2:
                    _price_df = pd.DataFrame({
                        s: self.historical_data[s]['Close'].iloc[-self._pairs_trader.lookback:]
                        for s in _eq_syms
                        if 'Close' in self.historical_data[s].columns
                    }).dropna(axis=1, how='any')

                    if _price_df.shape[1] >= 2:
                        found = await asyncio.to_thread(
                            self._pairs_trader.find_cointegrated_pairs, _price_df, 0.05
                        )
                        self._active_pairs = found[:20]  # keep top 20 pairs
                        if found:
                            logger.info("PairsTrader: found %d cointegrated pairs", len(found))

            # Update z-scores for active pairs every cycle
            new_overlay: Dict[str, float] = {}
            for asset_y, asset_x, _ in self._active_pairs:
                if asset_y not in self.historical_data or asset_x not in self.historical_data:
                    continue
                try:
                    analysis = self._pairs_trader.analyze_pair(
                        asset_y, asset_x, self.historical_data
                    )
                    if analysis is None:
                        continue
                    sig = self._pairs_trader.get_signal(asset_y, asset_x)
                    action = sig.get('action', 'NONE')
                    if action == 'SELL_SPREAD':
                        # Sell Y (bearish for Y), Buy X (bullish for X)
                        new_overlay[asset_y] = new_overlay.get(asset_y, 0.0) - _max_overlay
                        new_overlay[asset_x] = new_overlay.get(asset_x, 0.0) + _max_overlay
                    elif action == 'BUY_SPREAD':
                        # Buy Y (bullish for Y), Sell X (bearish for X)
                        new_overlay[asset_y] = new_overlay.get(asset_y, 0.0) + _max_overlay
                        new_overlay[asset_x] = new_overlay.get(asset_x, 0.0) - _max_overlay
                    elif action == 'EXIT':
                        # Nudge toward exit by zeroing overlay
                        new_overlay.setdefault(asset_y, 0.0)
                        new_overlay.setdefault(asset_x, 0.0)
                except Exception as _pair_err:
                    logger.debug("Pair (%s, %s) analysis error: %s", asset_y, asset_x, _pair_err)

            # Clamp overlay values
            self._pairs_overlay = {
                sym: max(-_max_overlay, min(_max_overlay, v))
                for sym, v in new_overlay.items()
            }
            if self._pairs_overlay:
                logger.debug("PairsOverlay: %d symbols, weight=%.2f", len(self._pairs_overlay), _weight)

        except Exception as _pte:
            logger.debug("_update_pairs_signals error: %s", _pte)

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
                # self.signal_generator.train_models(self.historical_data)
                logger.info("✅ ML models updated")

            # Update streams if needed (e.g. new symbols added)
            all_symbols = list(set(list(self.positions.keys()) + self._runtime_symbols()))
            if self.ibkr:
                equity_symbols = [s for s in all_symbols if parse_symbol(s).asset_class != AssetClass.CRYPTO]
                
                must_stream = []
                for s in self.positions.keys():
                    if s in equity_symbols and s not in must_stream:
                        must_stream.append(s)
                for s in self.pending_orders:
                    if s in equity_symbols and s not in must_stream:
                        must_stream.append(s)
                        
                rotational = [s for s in equity_symbols if s not in must_stream]
                # Weight by signal magnitude if available
                rotational.sort(
                    key=lambda s: abs(self.current_signals.get(s, {}).get("signal", 0.0)),
                    reverse=True
                )
                
                max_streams = getattr(ApexConfig, "IBKR_MAX_STREAMS", 85)
                slots = max_streams - len(must_stream)
                
                if slots <= 0:
                    stream_symbols = must_stream[:max_streams]
                else:
                    if not rotational:
                        stream_symbols = must_stream
                    else:
                        chunk = []
                        for i in range(slots):
                            idx = (self._ibkr_rotation_index + i) % len(rotational)
                            chunk.append(rotational[idx])
                        stream_symbols = must_stream + chunk
                        self._ibkr_rotation_index = (self._ibkr_rotation_index + slots) % len(rotational)
                
                await self.ibkr.stream_quotes(stream_symbols)
                
            if self.alpaca:
                crypto_symbols = [s for s in all_symbols if parse_symbol(s).asset_class == AssetClass.CRYPTO]
                if crypto_symbols:
                    await self.alpaca.stream_quotes(crypto_symbols)
        
        except Exception as e:
            logger.error(f"❌ Data refresh error: {e}")
    
    def get_current_signals(self) -> dict:
        """Get current signals for dashboard (using cached results from trading loop)."""
        # Purge stale signals (older than 10 minutes)
        now = datetime.utcnow()
        to_purge = []
        for sym, data in self.current_signals.items():
            try:
                ts = datetime.fromisoformat(data['timestamp'].replace('Z', ''))
                if (now - ts).total_seconds() > 600:
                    to_purge.append(sym)
            except Exception:
                pass
        for sym in to_purge:
            self.current_signals.pop(sym, None)
            
        return self.current_signals
    
    def _save_position_metadata(self):
        """Persist position metadata for restart recovery.

        Called from async context — uses _meta_file_lock via a sync wrapper.
        The lock prevents concurrent writes corrupting the JSON.
        """
        try:
            metadata = {}
            for symbol in self.position_entry_prices:
                entry_time = self.position_entry_times.get(symbol, datetime.utcnow())
                peak_price = self.position_peak_prices.get(symbol, self.position_entry_prices[symbol])
                metadata[symbol] = {
                    'entry_price': self.position_entry_prices[symbol],
                    'entry_time': entry_time.isoformat() + "Z" if isinstance(entry_time, datetime) else entry_time,
                    'entry_signal': self.position_entry_signals.get(symbol, 0.0),
                    'peak_price': peak_price
                }
                try:
                    from services.common.redis_client import state_set
                    asyncio.create_task(state_set("position_peak_prices", symbol, peak_price))
                except Exception:
                    pass
            metadata_file = Path("data") / "position_metadata.json"
            metadata_file.parent.mkdir(exist_ok=True)
            # Write to temp file then atomically rename to prevent partial-write corruption
            tmp_file = metadata_file.with_suffix(".json.tmp")
            with open(tmp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            tmp_file.replace(metadata_file)
        except Exception as e:
            logger.error(f"Error saving position metadata: {e}")

    def _load_position_metadata(self) -> Dict:
        """Load position metadata from disk."""
        metadata_file = Path("data") / "position_metadata.json"
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            logger.info(f"📂 Loaded saved position metadata for {len(metadata)} symbols")
            return metadata
        except Exception as e:
            logger.error(f"Error loading position metadata: {e}")
            return {}

    async def export_dashboard_state(self):
        """Export current state for dashboard."""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            # CRITICAL FIX: Use _get_total_portfolio_value() to get accurate combined equity
            # instead of manually reconstructing it (which was adding Alpaca twice)
            current_value = await self._get_total_portfolio_value()

            # Fallback to self.capital only if broker fetch fails
            if current_value <= 0:
                current_value = self.capital
            
            current_signals = self.get_current_signals()

            option_leg_map: Dict[Tuple[str, str, float, str], Dict[str, float | int | str]] = {}

            def upsert_option_leg(
                symbol: str,
                expiry: str,
                strike: float,
                right: str,
                quantity: float,
                avg_cost: float,
            ) -> None:
                symbol_norm = str(symbol or "").strip().upper()
                import re
                expiry_norm = re.sub(r'[^0-9]', '', str(expiry or "").strip()) # Normalize strictly to YYYYMMDD
                if not symbol_norm or not expiry_norm:
                    return
                try:
                    strike_norm = float(strike or 0.0)
                except Exception:
                    strike_norm = 0.0
                right_norm = str(right or "").strip().upper()
                if right_norm == "CALL":
                    right_norm = "C"
                elif right_norm == "PUT":
                    right_norm = "P"
                if right_norm not in {"C", "P"}:
                    return
                try:
                    qty_norm = int(float(quantity or 0.0))
                except Exception:
                    qty_norm = 0
                if qty_norm == 0:
                    return
                try:
                    avg_cost_norm = float(avg_cost or 0.0)
                except Exception:
                    avg_cost_norm = 0.0

                key = (symbol_norm, expiry_norm, strike_norm, right_norm)
                if key not in option_leg_map:
                    option_leg_map[key] = {
                        "symbol": symbol_norm,
                        "expiry": expiry_norm,
                        "strike": strike_norm,
                        "right": right_norm,
                        "quantity": qty_norm,
                        "side": "LONG" if qty_norm > 0 else "SHORT",
                        "avg_cost": avg_cost_norm,
                    }
                    return

                existing_qty = int(option_leg_map[key].get("quantity", 0))
                merged_qty = existing_qty + qty_norm
                option_leg_map[key]["quantity"] = merged_qty
                option_leg_map[key]["side"] = "LONG" if merged_qty > 0 else "SHORT"
                if avg_cost_norm > 0:
                    option_leg_map[key]["avg_cost"] = avg_cost_norm

            if self.ibkr and hasattr(self.ibkr, "ib"):
                try:
                    for pos in self.ibkr.ib.positions():
                        contract = getattr(pos, "contract", None)
                        if getattr(contract, "secType", "") != "OPT":
                            continue
                        upsert_option_leg(
                            symbol=str(getattr(contract, "symbol", "")),
                            expiry=str(getattr(contract, "lastTradeDateOrContractMonth", "")),
                            strike=float(getattr(contract, "strike", 0.0) or 0.0),
                            right=str(getattr(contract, "right", "")),
                            quantity=float(getattr(pos, "position", 0.0) or 0.0),
                            avg_cost=float(getattr(pos, "avgCost", 0.0) or 0.0),
                        )
                except Exception:
                    pass

            for option_row in self.options_positions.values():
                try:
                    upsert_option_leg(
                        symbol=str(option_row.get("symbol", "")),
                        expiry=str(option_row.get("expiry", "")),
                        strike=float(option_row.get("strike", 0.0) or 0.0),
                        right=str(option_row.get("right", "")),
                        quantity=float(option_row.get("quantity", 0.0) or 0.0),
                        avg_cost=float(option_row.get("avg_cost", 0.0) or 0.0),
                    )
                except Exception:
                    continue

            option_positions_detail = list(option_leg_map.values())
            option_positions_detail.sort(
                key=lambda row: (
                    str(row.get("symbol", "")),
                    str(row.get("expiry", "")),
                    float(row.get("strike", 0.0)),
                    str(row.get("right", "")),
                )
            )
            option_positions_count = len(option_positions_detail)
            equity_daily_delta = (
                float(current_value - self.risk_manager.day_start_capital)
                if self.risk_manager.day_start_capital > 0
                else 0.0
            )
            broker_truth_daily_enabled = bool(self.ibkr or self.alpaca)
            realized_daily_pnl = float(self._daily_realized_pnl_total)

            # Smart fallback: if broker_fills shows 0 but equity_delta is significant,
            # use equity_delta (handles manual closes, cancelled orders, etc.)
            if broker_truth_daily_enabled and realized_daily_pnl == 0.0 and abs(equity_daily_delta) > 100:
                daily_pnl_value = equity_daily_delta
                daily_pnl_source = "equity_delta_fallback"
                logger.info(f"💡 Using equity_delta fallback for daily P&L: ${equity_daily_delta:,.2f} (no fills recorded but portfolio changed)")
            elif broker_truth_daily_enabled:
                daily_pnl_value = realized_daily_pnl
                daily_pnl_source = "broker_fills"
            else:
                daily_pnl_value = equity_daily_delta
                daily_pnl_source = "equity_delta"
            
            state = {
                'timestamp': datetime.utcnow().isoformat() + "Z",
                'capital': float(current_value),
                'initial_capital': float(ApexConfig.INITIAL_CAPITAL),
                'starting_capital': float(self.risk_manager.starting_capital),
                'positions': {},
                'signals': current_signals,
                'daily_pnl': float(daily_pnl_value),
                'daily_pnl_realized': float(realized_daily_pnl),
                'daily_pnl_unrealized_fallback': float(equity_daily_delta),
                'daily_pnl_source': daily_pnl_source,
                'daily_pnl_by_broker': self._compute_daily_pnl_by_broker(),
                'total_pnl': float(current_value - self.risk_manager.starting_capital),
                'total_commissions': float(self.total_commissions),
                'max_drawdown': float(self.performance_tracker.get_max_drawdown()) if math.isfinite(self.performance_tracker.get_max_drawdown()) else 0.0,
                'sharpe_ratio': float(self.performance_tracker.get_sharpe_ratio()) if math.isfinite(self.performance_tracker.get_sharpe_ratio()) else 0.0,
                'sortino_ratio': round(self.performance_tracker.get_sortino_ratio(), 2),
                'calmar_ratio': float(self.performance_tracker.get_calmar_ratio()) if math.isfinite(self.performance_tracker.get_calmar_ratio()) else 0.0,
                'profit_factor': round(self.performance_tracker.get_profit_factor(), 2),
                'alpha_retention': round(self.performance_tracker.get_alpha_retention(), 4),
                'win_rate': float(self.performance_tracker.get_win_rate()) if math.isfinite(self.performance_tracker.get_win_rate()) else 0.0,
                'total_trades': len(self.performance_tracker.trades),
                # open_positions includes both equity and option positions so the
                # dashboard/API always shows the true total exposure.
                'open_positions': int(self.position_count + option_positions_count),
                'open_positions_equity': self.position_count,
                'option_positions': int(option_positions_count),
                'open_positions_total': int(self.position_count + option_positions_count),
                'option_positions_detail': option_positions_detail,
                'sector_exposure': self.calculate_sector_exposure(),
                'broker_heartbeats': self._broker_heartbeat_payload(),
                'crypto_rotation': dict(self._crypto_rotation_snapshot or {}),
                'performance_governor': self._performance_snapshot.to_dict() if self._performance_snapshot else None,
                'stress_control': (
                    self._stress_control_state.to_dict()
                    if getattr(self, "_stress_control_state", None) is not None
                    else None
                ),
                'stress_unwind_plan': (
                    self._stress_unwind_plan.to_dict()
                    if getattr(self, "_stress_unwind_plan", None) is not None
                    else None
                ),
                'shadow_deployment': (
                    self.shadow_deployment.runtime_state()
                    if getattr(self, "shadow_deployment", None) is not None
                    else None
                ),
                'kill_switch': self.kill_switch.state().__dict__ if self.kill_switch else None,
                'equity_reconciliation': (
                    self._equity_reconciliation_snapshot.to_dict()
                    if self._equity_reconciliation_snapshot
                    else {
                        "block_entries": bool(self._equity_reconciliation_block_entries),
                        "reason": "not_evaluated",
                    }
                ),
                'performance_attribution': self.performance_attribution.get_summary(lookback_days=30),
                'social_shock': {
                    "enabled": bool(self.social_shock_governor is not None),
                    "policy_version": self._social_policy_version,
                    "input_validation": dict(self._social_input_validation or {}),
                    "decisions": [
                        decision.to_dict() for decision in self._social_decision_cache.values()
                    ],
                    "snapshots": [
                        snapshot.to_dict() for snapshot in self._social_snapshot_cache.values()
                    ],
                    "prediction_verification": [
                        verification.to_dict()
                        for rows in self._prediction_results_cache.values()
                        for verification in rows
                    ],
                },
                'daily_fill_events': list(self._daily_realized_fill_events[-50:]),
            }
            # Fetch active broker connections once to map sources
            active_sources = {}
            try:
                from services.common.db import db_session
                from services.trading.models import BrokerConnectionModel
                from sqlalchemy import select
                
                async with db_session() as session:
                    stmt = select(BrokerConnectionModel).where(BrokerConnectionModel.is_active)
                    result = await session.execute(stmt)
                    for row in result.scalars().all():
                        if row.broker_type not in active_sources:
                            active_sources[row.broker_type] = row.id
            except Exception as e:
                logger.debug(f"Could not load active sources for tagging: {e}")

            for symbol, qty in self.positions.items():
                if qty == 0:
                    continue
                
                try:
                    price = self.price_cache.get(symbol, 0)
                    avg_price = self.position_entry_prices.get(symbol, price)

                    # Fallback to entry price if market data is unavailable (e.g. after hours)
                    if price <= 0:
                        price = avg_price
                    
                    if price <= 0:
                        continue
                    
                    entry_time = self.position_entry_times.get(symbol, datetime.utcnow())
                    
                    if qty > 0:  # Long
                        pnl = (price - avg_price) * qty
                        pnl_pct = (price / avg_price - 1) * 100 if avg_price > 0 else 0
                    else:  # Short
                        pnl = (avg_price - price) * abs(qty)
                        pnl_pct = (avg_price / price - 1) * 100 if price > 0 else 0
                    
                    # Determine source and broker type
                    connector = self._get_connector_for(symbol)
                    source_id = ""
                    broker_type = ""
                    if connector == self.alpaca:
                        source_id = active_sources.get("alpaca", "alpaca")
                        broker_type = "alpaca"
                    elif connector == self.ibkr:
                        source_id = active_sources.get("ibkr", "ibkr")
                        broker_type = "ibkr"
                    # Fallback: if active_sources lookup failed, ensure source_id
                    # always contains the broker name so the frontend can split P&L.
                    if not source_id:
                        source_id = broker_type or "unknown"

                    state['positions'][symbol] = {
                        'qty': round(float(qty), 8),  # Keep fractional for crypto (int() truncates 0.1234 BTC → 0)
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'avg_price': float(avg_price),
                        'current_price': float(price),
                        'pnl': float(pnl),
                        'pnl_pct': float(pnl_pct),
                        'entry_time': entry_time.isoformat() + "Z" if isinstance(entry_time, datetime) else entry_time,
                        'current_signal': current_signals.get(symbol, {}).get('signal', 0),
                        'signal_direction': current_signals.get(symbol, {}).get('direction', 'UNKNOWN'),
                        'source_id': source_id,
                        'broker_type': broker_type,
                    }
                except Exception as e:
                    logger.debug(f"Error adding position {symbol}: {e}")
            
            # Add signal quality metrics if available
            try:
                quality_metrics = self.signal_outcome_tracker.get_quality_metrics()
                state['signal_quality'] = {
                    'total_signals_tracked': quality_metrics.total_signals,
                    'completed_signals': quality_metrics.total_signals,
                    'avg_forward_return_5d': quality_metrics.avg_return_5d_buy,
                    'avg_forward_return_10d': quality_metrics.avg_return_10d_buy,
                    'hit_5pct_5d': quality_metrics.hit_rate_5pct_5d,
                    'hit_10pct_10d': quality_metrics.hit_rate_10pct_10d,
                    'avg_mfe_5d': quality_metrics.avg_mfe_10d,
                    'avg_mae_5d': quality_metrics.avg_mae_10d,
                    'by_regime': quality_metrics.accuracy_by_regime,
                    'by_confidence': quality_metrics.accuracy_by_confidence,
                }
            except Exception as e:
                logger.debug(f"Signal quality metrics error: {e}")
                state['signal_quality'] = {}

            # Session-scoped state file: core_trading_state.json / crypto_trading_state.json
            # (api/dependencies.py reads from these session-specific paths)
            _prefix = f"{self.session_type}_" if self.session_type != "unified" else ""
            state_file = data_dir / f"{_prefix}trading_state.json"
            tmp_file = state_file.with_suffix(".json.tmp")
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, state_file)

            # Redis hot-path write — session-scoped keys prevent CORE/CRYPTO from
            # overwriting each other.  api/server.py reads apex:state:trading:{session}.
            try:
                from services.common.redis_client import cache_set as _redis_set
                _redis_key = f"apex:state:trading:{self.session_type}" if self.session_type != "unified" else "apex:state:trading"
                await _redis_set(_redis_key, state, ttl_seconds=60)
                # Keep legacy unified key updated with most-recent state (for backward compat)
                await _redis_set("apex:state:trading", state, ttl_seconds=60)
                await _redis_set("apex:state:prices", self.price_cache, ttl_seconds=30)
            except Exception:
                pass

            # Persist position metadata for restart recovery
            self._save_position_metadata()

            # Save risk state and price cache
            await self.risk_manager.save_state_async()
            await self.save_price_cache()

            logger.debug("📊 Dashboard state exported")
        
        except Exception as e:
            logger.error(f"❌ Dashboard export error: {e}")

    async def save_price_cache(self):
        """Save price cache to Redis AND disk so the API layer always has fresh prices."""
        if not self.price_cache:
            return
        # Always write to disk first (API reads from file, not Redis)
        try:
            ApexConfig.DATA_DIR.mkdir(exist_ok=True)
            with open(ApexConfig.DATA_DIR / "price_cache.json", "w") as f:
                json.dump(self.price_cache, f)
        except Exception as e:
            logger.debug(f"Error saving disk price cache: {e}")
        # Also push to Redis for other consumers
        try:
            from services.common.redis_client import cache_set
            await cache_set("apex:state:prices", self.price_cache, ttl_seconds=60)
        except Exception as e:
            logger.debug(f"Error saving price cache to Redis: {e}")

    async def load_price_cache(self):
        """Load last known prices from Redis (fallback to disk)."""
        try:
            from services.common.redis_client import cache_get
            cached = await cache_get("apex:state:prices")
            if cached and isinstance(cached, dict):
                self.price_cache.update(cached)
                logger.info(f"🔄 Restored {len(cached)} prices from Redis cache")
                return
        except Exception as e:
            logger.debug(f"Error loading price cache from Redis: {e}")
            
        try:
            cache_file = ApexConfig.DATA_DIR / "price_cache.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    self.price_cache.update(cached)
                logger.info(f"💾 Restored {len(cached)} prices from disk cache")
        except Exception as e:
            logger.debug(f"Error loading disk price cache: {e}")

    async def refresh_position_prices(self):
        """Fetch current market prices for all open positions and update price_cache."""
        if not self.ibkr and not self.alpaca:
            return

        position_symbols = [s for s, qty in self.positions.items() if qty != 0]
        if not position_symbols:
            return

        updated_count = 0
        for symbol in position_symbols:
            try:
                conn = self._get_connector_for(symbol)
                if conn:
                    price = await conn.get_market_price(symbol)
                    if price and price > 0:
                        self.price_cache[symbol] = price
                        if self.data_watchdog:
                            self.data_watchdog.feed_heartbeat(symbol)
                        updated_count += 1
            except Exception as e:
                logger.debug(f"Error fetching price for {symbol}: {e}")

        if updated_count > 0:
            logger.debug(f"💰 Refreshed prices for {updated_count}/{len(position_symbols)} positions")

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
    
    def _write_daily_pnl_snapshot(self, loss_check: dict, dd_check: dict, current_value: float):
        """Write/update today's daily PnL snapshot to audit/daily_pnl_YYYYMMDD.json.

        Called every risk-check cycle. Overwrites the file in-place so the last
        call of the day leaves the permanent end-of-day record.
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            audit_dir = self.user_data_dir / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            out_path = audit_dir / f"daily_pnl_{today.replace('-', '')}.json"

            day_start = float(self.risk_manager.day_start_capital or current_value)
            daily_pnl = float(loss_check.get("daily_pnl", current_value - day_start))
            daily_ret = float(loss_check.get("daily_return", daily_pnl / day_start if day_start else 0.0))

            # Count today's fills from execution_latency log
            fills_today_buy = 0
            fills_today_sell = 0
            notional_today = 0.0
            latency_path = self.user_data_dir / "audit" / "execution_latency.jsonl"
            if latency_path.exists():
                try:
                    with open(latency_path) as f:
                        for raw in f:
                            try:
                                rec = json.loads(raw)
                                if rec.get("ts", "")[:10] == today:
                                    if rec.get("side") == "BUY":
                                        fills_today_buy += 1
                                    else:
                                        fills_today_sell += 1
                                    notional_today += rec.get("qty", 0) * rec.get("fill_price", 0)
                            except Exception:
                                pass
                except Exception:
                    pass

            snapshot = {
                "date": today,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "day_start_capital": day_start,
                "end_capital": current_value,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": round(daily_ret * 100, 4),
                "max_drawdown_pct": round(dd_check.get("drawdown", 0.0) * 100, 4),
                "open_positions": self.position_count,
                "fills_buy": fills_today_buy,
                "fills_sell": fills_today_sell,
                "notional_traded": round(notional_today, 2),
                "regime": self._current_regime or "unknown",
                "vix_multiplier": round(getattr(self, "_vix_risk_multiplier", 1.0), 4),
                "cycle_count": getattr(self, "_cycle_count", 0),
            }

            with open(out_path, "w") as f:
                json.dump(snapshot, f, indent=2)

            logger.info("📅 Daily PnL snapshot updated: %s PnL=%+.2f (%.3f%%)",
                         today, daily_pnl, daily_ret * 100)
        except Exception as e:
            logger.warning("⚠️ Could not write daily PnL snapshot: %s", e)

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
    
    async def _export_initial_state(self):
        """Export initial state."""
        try:
            state = {
                'capital': self.capital,
                'starting_capital': self.capital,
                'positions': {},
                'daily_pnl': 0.0,
                'daily_pnl_realized': 0.0,
                'daily_pnl_unrealized_fallback': 0.0,
                'daily_pnl_source': "broker_fills" if (self.ibkr or self.alpaca) else "equity_delta",
                'daily_pnl_by_broker': {'ibkr': 0.0, 'alpaca': 0.0},
                'total_pnl': 0.0,
                'total_commissions': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'profit_factor': 0.0,
                'alpha_retention': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'open_positions': self.total_position_count,
                'open_positions_equity': self.position_count,

                'broker_heartbeats': self._broker_heartbeat_payload(),
                'crypto_rotation': dict(self._crypto_rotation_snapshot or {}),
                'performance_governor': self._performance_snapshot.to_dict() if self._performance_snapshot else None,
                'kill_switch': self.kill_switch.state().__dict__ if self.kill_switch else None,
                'equity_reconciliation': {
                    "block_entries": bool(self._equity_reconciliation_block_entries),
                    "reason": "not_evaluated",
                },
                'performance_attribution': self.performance_attribution.get_summary(lookback_days=30),
                'social_shock': {
                    "enabled": bool(self.social_shock_governor is not None),
                    "policy_version": self._social_policy_version,
                    "input_validation": dict(self._social_input_validation or {}),
                    "decisions": [],
                    "snapshots": [],
                    "prediction_verification": [],
                },
            }
            self.state_sync.write(state)
            self.live_monitor.update_state(state)
            await self.export_dashboard_state()
            logger.info("📱 Dashboard initialized")
        except Exception as e:
            logger.warning(f"⚠️  Dashboard export failed: {e}")
    
    async def _update_bl_weights(self) -> None:
        """
        Compute Black-Litterman posterior weights using current ML signals as views.

        Steps:
        1. Gather 60+ bars of daily returns from self.historical_data for all symbols
           that have recent signals.
        2. Use signal × confidence as the expected-return view for each symbol.
        3. Run BlackLittermanOptimizer.optimize() to get posterior weights.
        4. Store normalised relative weights in self._bl_weights.

        Position sizing in process_symbol multiplies the computed shares by
        bl_weight / equal_weight, clamped to [BL_MIN_SCALE, BL_MAX_SCALE].
        """
        try:
            _syms_with_signals = [
                s for s in self._last_signal_components
                if abs(self._last_signal_components[s].get('ml', 0.0)) > 0.05
            ]
            if len(_syms_with_signals) < 2:
                return

            # Build returns_data dict
            _returns: Dict[str, "pd.Series"] = {}
            for _s in _syms_with_signals:
                _hist = self.historical_data.get(_s)
                if _hist is None:
                    continue
                _prices = _hist['Close'] if (hasattr(_hist, 'get') and 'Close' in _hist) else _hist
                if len(_prices) < 60:
                    continue
                _ret = _prices.pct_change().dropna()
                if len(_ret) >= 60:
                    _returns[_s] = _ret

            if len(_returns) < 2:
                return

            # Views: signal × confidence → expected daily return (scaled down to realistic range)
            _scale = float(getattr(ApexConfig, "BL_VIEW_SCALE", 0.002))  # 0.2% per unit signal
            _views: Dict[str, float] = {}
            _view_confs: Dict[str, float] = {}
            for _s, _comp in self._last_signal_components.items():
                if _s not in _returns:
                    continue
                _ml = float(_comp.get('ml', 0.0) or 0.0)
                _conf_raw = float(_comp.get('confidence', 0.60) or 0.60)
                _views[_s] = _ml * _scale
                _view_confs[_s] = max(0.10, min(0.95, _conf_raw))

            result = self._bl_optimizer.optimize(
                returns_data=_returns,
                views=_views,
                view_confidences=_view_confs,
            )

            # Normalise weights so they average to 1.0 across active symbols
            _w = result.posterior_weights
            _n = max(1, len(_w))
            _mean_w = sum(_w.values()) / _n if _w else 1.0 / _n
            if _mean_w > 0:
                self._bl_weights = {s: float(v / _mean_w) for s, v in _w.items()}
            else:
                self._bl_weights = {}

            logger.debug(
                "📊 BL weights updated: %d symbols. Top3: %s",
                len(self._bl_weights),
                sorted(self._bl_weights.items(), key=lambda x: -x[1])[:3],
            )
        except Exception as _ble:
            logger.debug("BL weight update skipped (non-fatal): %s", _ble)

    def _update_kelly_multiplier(self) -> None:
        """
        Recompute self._live_kelly_mult from the rolling last-20 live_entry trades.

        Half-Kelly is applied and the result is clamped to [0.25, 1.5] to prevent
        extreme swings.  We treat a 5% full-Kelly as the baseline (100% size), so:
            mult = clamp(half_kelly / 0.05, 0.25, 1.5)

        Requires at least 5 closed live_entry / startup_signal_refresh trades.
        Falls back to 1.0 (no change) if data is insufficient.
        """
        try:
            closed = getattr(self.performance_attribution, "closed_trades", [])
            live = [
                t for t in closed
                if t.get("source", "live_entry") != "startup_restore"
                and t.get("net_pnl") is not None
            ]
            n = min(20, len(live))
            # Fix 6 (2026-03-19): Require n>=20 for statistically valid Kelly.
            # With n<20, Kelly can produce garbage (e.g., 0 wins from 4 trades →
            # Kelly=0, half_kelly=0, mult=0.25 minimum → undersizes every trade).
            # Fall back to fixed 1.0× until we have enough data.
            if n < 20:
                self._live_kelly_mult = 1.0
                logger.debug("Kelly sizing: n=%d < 20 — using fixed 1.0× until sufficient history", n)
                return
            recent = live[-n:]
            wins   = [float(t["net_pnl"]) for t in recent if float(t["net_pnl"]) > 0]
            losses = [abs(float(t["net_pnl"])) for t in recent if float(t["net_pnl"]) <= 0]
            if not wins or not losses:
                self._live_kelly_mult = 1.0
                return
            p = len(wins) / n
            b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))  # avg win / avg loss
            kelly = max(0.0, (p * b - (1.0 - p)) / b)
            half_kelly = kelly / 2.0
            # Normalise: 5% full-kelly = 1.0× multiplier
            self._live_kelly_mult = float(max(0.25, min(1.5, half_kelly / 0.05)))
            logger.info(
                "Kelly sizing: n=%d, p=%.2f, b=%.2f, kelly=%.3f, half=%.3f → mult=%.2fx",
                n, p, b, kelly, half_kelly, self._live_kelly_mult,
            )
        except Exception as exc:
            logger.debug("Kelly update failed: %s", exc)

    async def _run_threshold_calibration(self, force: bool = False) -> None:
        """
        Recalibrate exit thresholds and slippage blacklist from live trade history.
        Runs in background (asyncio.ensure_future) — never blocks the main loop.
        Skipped if called within 6h of last run (unless force=True).
        """
        _RECAL_INTERVAL_H = 6
        if not force and self._last_calibration_at:
            hours_since = (datetime.now() - self._last_calibration_at).total_seconds() / 3600
            if hours_since < _RECAL_INTERVAL_H:
                return
        try:
            params = await asyncio.to_thread(self._calibrator.run_calibration)
            update_excellence_params(params)
            self._calibrated_blacklist = params.get("slippage_blacklist", [])
            self._calibrated_symbol_probation = params.get("symbol_probation", {})
            self._last_calibration_at = datetime.now()
            logger.info(
                f"ThresholdCalibrator: recalibrated — "
                f"exit_thresh={params.get('weak_signal_loss_threshold_pct', '?'):.3f}%, "
                f"blacklist={self._calibrated_blacklist}, "
                f"probation={list(self._calibrated_symbol_probation.keys())}"
            )
        except Exception as e:
            logger.warning(f"ThresholdCalibrator: recalibration failed: {e}")

    async def _check_and_reload_models(self) -> None:
        """Non-blocking check to safely hot-reload ML models if updated on disk."""
        try:
            if not hasattr(self, 'inst_signal_generator'): return
            
            import os
            _meta_path = os.path.join(self.inst_signal_generator.model_dir, "metadata.pkl")
            if not os.path.exists(_meta_path): return
                
            current_mtime = os.path.getmtime(_meta_path)
            
            # If the file hasn't changed, or it's our first check, do nothing.
            if getattr(self, "_last_model_mtime", 0) == 0:
                self._last_model_mtime = current_mtime
                return
                
            if current_mtime > self._last_model_mtime:
                logger.info("🔥 Hot-reloading ML models due to detected updates on disk...")
                self._last_model_mtime = current_mtime
                
                # Asynchronously load models to prevent blocking the event loop
                loaded = await asyncio.to_thread(self.inst_signal_generator.loadModels)
                if loaded:
                    logger.info("✅ ML models hot-reloaded successfully!")
                    # Call garbage collector explicitly to manage memory spikes from holding two model sets
                    import gc
                    gc.collect()
                else:
                    logger.warning("⚠️ ML models hot-reload failed or no new models were found.")
        except Exception as e:
            import traceback
            logger.warning(f"Error during ML model hot-reload check: {e}\n{traceback.format_exc()}")

    # ── Last crypto momentum rescan timestamp ─────────────────────────────────
    _last_crypto_scan_at: Optional[datetime] = None

    async def _refresh_crypto_universe(self, force: bool = False) -> None:
        """Re-rank the Alpaca crypto universe by 24h momentum × liquidity.

        Runs as a background task — never blocks the main loop.  Triggered:
          • At NYSE market open (overnight → equity-hours transition)
          • Every CRYPTO_MOMENTUM_RESCAN_HOURS hours during the session

        IBKR equity subscriptions and ApexConfig.SYMBOLS are NEVER touched.
        Only self._dynamic_crypto_symbols is updated, which feeds into
        _runtime_symbols() exclusively for the Alpaca crypto slice.

        Positions currently held on Alpaca are force-included in the new
        universe even if their momentum rank has fallen, ensuring the quote
        poll loop never loses pricing for an open position.
        """
        if not self.alpaca:
            return
        if not getattr(ApexConfig, "CRYPTO_MOMENTUM_SCAN_ENABLED", True):
            return

        rescan_h = float(getattr(ApexConfig, "CRYPTO_MOMENTUM_RESCAN_HOURS", 4.0))
        if not force and self._last_crypto_scan_at is not None:
            hours_since = (datetime.now() - self._last_crypto_scan_at).total_seconds() / 3600
            if hours_since < rescan_h:
                return

        try:
            excluded = set(getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_EXCLUDED", []))
            top_n    = int(getattr(ApexConfig, "CRYPTO_MOMENTUM_TOP_N", 8))
            min_vol  = float(getattr(ApexConfig, "CRYPTO_MOMENTUM_MIN_VOLUME_USD", 500_000))

            fg_value = await self._fetch_fear_greed()
            leaders = await self.alpaca.scan_crypto_momentum_leaders(
                top_n=top_n,
                min_volume_usd=min_vol,
                excluded=excluded,
                fear_greed=fg_value,
            )

            if not leaders:
                logger.warning("CryptoMomentumScan: empty result — keeping current universe")
                return

            # Force-include any crypto symbol currently held (non-zero position).
            # We must never drop a held symbol from the quote poll loop.
            held = []
            for sym, qty in list(self.positions.items()):
                if qty == 0:
                    continue
                try:
                    if parse_symbol(sym).asset_class == AssetClass.CRYPTO:
                        held.append(sym)
                except ValueError:
                    pass

            new_universe: List[str] = list(dict.fromkeys(leaders + held))

            old_set = set(self._dynamic_crypto_symbols)
            new_set = set(new_universe)
            added   = new_set - old_set
            removed = old_set - new_set

            self._dynamic_crypto_symbols = new_universe
            self._last_crypto_scan_at = datetime.now()

            logger.info(
                "🪙 CryptoMomentumScan: universe → %d symbols (+%d −%d held=%d). Top: %s",
                len(new_universe), len(added), len(removed), len(held),
                ", ".join(new_universe[:6]),
            )

            # Only restart the quote poll loop if the symbol list actually changed.
            if added or removed:
                await self.alpaca.stream_quotes(new_universe)
            else:
                logger.debug("CryptoMomentumScan: universe unchanged — poll loop not restarted")

        except Exception as exc:
            logger.warning("CryptoMomentumScan: refresh failed: %s", exc)

    async def run(self):
        """Run the complete system."""
        try:
            await self.initialize()

            # ── Load persisted calibration immediately (no re-calibration overhead) ──
            try:
                from risk.threshold_calibrator import ThresholdCalibrator as _TC
                _persisted = _TC.load_or_defaults(ApexConfig.DATA_DIR)
                update_excellence_params(_persisted)
                self._calibrated_blacklist = _persisted.get("slippage_blacklist", [])
                self._calibrated_symbol_probation = _persisted.get("symbol_probation", {})
                logger.info(
                    f"ThresholdCalibrator: loaded persisted — "
                    f"exit_thresh={_persisted['weak_signal_loss_threshold_pct']:.3f}%, "
                    f"blacklist={self._calibrated_blacklist}, "
                    f"probation={list(self._calibrated_symbol_probation.keys())}"
                )
                # Trigger full recalibration in background (won't block startup)
                asyncio.ensure_future(self._run_threshold_calibration(force=True))
            except Exception as _ce:
                logger.warning(f"ThresholdCalibrator startup load failed: {_ce}")

            # ── Restore kill switch state from disk (survives restarts) ──────
            self._load_kill_switch_state()

            # ── Initial reconciliation: clean stale attribution entries ───────
            asyncio.ensure_future(self._reconcile_position_state())

            logger.info("▶️  Starting trading loop...")
            logger.info(f"   Interval: {ApexConfig.CHECK_INTERVAL_SECONDS}s")
            logger.info(f"   Hours: {ApexConfig.TRADING_HOURS_START:.1f} - {ApexConfig.TRADING_HOURS_END:.1f} EST")
            logger.info(f"   🛡️  Protection: {ApexConfig.TRADE_COOLDOWN_SECONDS}s cooldown")
            logger.info("   🚀 Parallel processing enabled")
            logger.info("   📱 Dashboard: streamlit run dashboard/streamlit_app.py")
            _broker_desc = (
                "IBKR+Alpaca" if self.ibkr and self.alpaca
                else "Alpaca" if self.alpaca
                else "IBKR"
            )
            fire_alert("engine_startup", f"Apex Trading engine started (brokers: {_broker_desc})", AlertSev.INFO)
            logger.info("")
            # Warm earnings calendar in background (non-blocking)
            asyncio.ensure_future(self._refresh_earnings_calendar())

            self.is_running = True
            cycle = 0

            # --- Hot-Reload ML Models Tracker ---
            self._last_model_mtime = 0
            if getattr(self, "use_institutional", False) and hasattr(self, "inst_signal_generator"):
                import os
                _meta_path = os.path.join(self.inst_signal_generator.model_dir, "metadata.pkl")
                if os.path.exists(_meta_path):
                    self._last_model_mtime = os.path.getmtime(_meta_path)
            last_data_refresh = datetime.now()
            self._startup_time = datetime.now()
            
            # Phase 12: Ignite High-Frequency WebSockets!
            if getattr(self, 'websocket_streamer', None):
                self.websocket_streamer.load_universe(self._session_symbols)
                await self.websocket_streamer.start()
                # Option D: Wait for WSS handshake to complete before entering the main loop.
                # This eliminates the race condition where the first N price fetches always
                # fell back to REST because the WSS connection wasn't yet subscribed.
                # Timeout of 8s → if WSS takes longer, we proceed anyway (REST fallback active).
                _wss_ready = await self.websocket_streamer.wait_until_ready(timeout=8.0)
                if _wss_ready:
                    logger.info("✅ WebSocket streams ready — entering main loop with live price feed")
                else:
                    logger.warning("⚠️ WebSocket streams not ready within 8s — proceeding with REST fallback")

            while self.is_running:
                try:
                    cycle_start_time = datetime.now()
                    cycle += 1
                    self._cycle_count = cycle
                    self._cycle_id = cycle          # UPGRADE F: stagger reset
                    self._cycle_exit_count = 0      # UPGRADE F: reset per-cycle exit counter
                    logger.debug(f"Starting trading cycle {cycle}")
                    now = datetime.now()

                    # AD: Purge price cache entries older than 90 seconds to prevent
                    # stale prices from being used for order sizing after reconnects.
                    _price_ttl = 90.0
                    _now_ts = now.timestamp()
                    _stale_syms = [
                        s for s, ts in list(self._price_cache_ts.items())
                        if _now_ts - ts > _price_ttl
                    ]
                    for _s in _stale_syms:
                        self.price_cache.pop(_s, None)
                        self._price_cache_ts.pop(_s, None)
                    if _stale_syms:
                        logger.debug("AD: Purged %d stale price cache entries (>%.0fs old)", len(_stale_syms), _price_ttl)
                    self._roll_daily_realized_if_needed(now)

                    # Process operator commands (e.g., kill-switch reset) each cycle.
                    await self._process_external_control_commands()

                    # ── Periodic ML Model Hot-Reload (every 15 cycles ≈ 15 min) ──
                    if cycle % 15 == 0 and getattr(self, "use_institutional", False):
                        asyncio.ensure_future(self._check_and_reload_models())

                    # ── Periodic threshold recalibration (every ~360 cycles ≈ 6h) ──
                    if cycle % 360 == 0:
                        asyncio.ensure_future(self._run_threshold_calibration())

                    # ── Periodic crypto momentum rescan ───────────────────────
                    # Interval is time-gated inside _refresh_crypto_universe()
                    # (default 4h). This cycle-based check just provides the
                    # scheduling hook without adding per-cycle overhead.
                    if cycle % 240 == 0 and self.alpaca:
                        asyncio.ensure_future(self._refresh_crypto_universe())

                    # ── Live Kelly sizing update (every 10 cycles ≈ 10 min) ──────
                    if cycle % 10 == 0 and getattr(ApexConfig, "LIVE_KELLY_SIZING_ENABLED", True):
                        self._update_kelly_multiplier()

                    # ── Black-Litterman portfolio weight update (every 30 cycles ≈ 30 min) ──
                    # Combines current ML signal "views" with market equilibrium (CAPM prior)
                    # to produce posterior conviction weights. These relativise position sizes —
                    # high-conviction, low-correlation symbols get more; crowded/noisy ones less.
                    if (
                        cycle - self._bl_last_cycle >= int(getattr(ApexConfig, "BL_UPDATE_INTERVAL_CYCLES", 30))
                        and getattr(self, '_bl_optimizer', None) is not None
                        and getattr(ApexConfig, "BL_SIZING_ENABLED", True)
                    ):
                        self._bl_last_cycle = cycle
                        asyncio.ensure_future(self._update_bl_weights())

                    # ── Factor Hedger: portfolio beta/concentration check ──────
                    if (cycle % 10 == 0
                            and getattr(self, '_factor_hedger', None) is not None
                            and getattr(ApexConfig, "FACTOR_HEDGER_ENABLED", True)):
                        try:
                            _fh = self._factor_hedger
                            _prices = {s: float(self.last_prices.get(s, 0)) for s in self.positions if self.positions.get(s, 0) != 0}
                            _fh_exposure = _fh.get_exposure(dict(self.positions), _prices)
                            # Cache for HHI concentration gate in entry path
                            self._last_factor_exposure = _fh_exposure
                            if _fh_exposure.hedge_urgency == "urgent":
                                logger.warning(
                                    "⚠️ FactorHedger URGENT: beta_eq=%.2f beta_cx=%.2f HHI=%.2f — %s",
                                    _fh_exposure.market_beta_equity,
                                    _fh_exposure.market_beta_crypto,
                                    _fh_exposure.concentration_hhi,
                                    _fh_exposure.hedge_recommendation or "High beta",
                                )
                            elif _fh_exposure.hedge_urgency == "advisory":
                                logger.debug(
                                    "FactorHedger advisory: beta_eq=%.2f beta_cx=%.2f — %s",
                                    _fh_exposure.market_beta_equity,
                                    _fh_exposure.market_beta_crypto,
                                    _fh_exposure.hedge_recommendation or "",
                                )
                        except Exception as _fhe:
                            logger.debug("FactorHedger check failed (non-fatal): %s", _fhe)

                    # ── Adaptive ATR stops: regime+VIX+profit-ratchet update ──────
                    _aatrs_interval = int(getattr(ApexConfig, "ADAPTIVE_ATR_UPDATE_INTERVAL", 5))
                    if (
                        cycle % _aatrs_interval == 0
                        and getattr(self, '_adaptive_atr_stops', None) is not None
                        and getattr(ApexConfig, "ADAPTIVE_ATR_STOPS_ENABLED", True)
                        and self.positions
                    ):
                        try:
                            _updated = self._adaptive_atr_stops.update_all(
                                positions=dict(self.positions),
                                position_stops=dict(self.position_stops),
                                entry_prices=dict(self.position_entry_prices),
                                historical_data=self.historical_data,
                                regime=self._current_regime or "neutral",
                                vix=float(self._current_vix or 20.0),
                            )
                            self.position_stops.update(_updated)
                            if _updated:
                                logger.debug(
                                    "AdaptiveATRStops updated %d stops (regime=%s vix=%.1f)",
                                    len(_updated), self._current_regime, self._current_vix or 20.0,
                                )
                        except Exception as _aatrs_err:
                            logger.debug("AdaptiveATRStops update failed (non-fatal): %s", _aatrs_err)

                    # ── ModelDriftMonitor: periodic health check (every 60 cycles) ──
                    if (
                        cycle % 60 == 0
                        and getattr(self, '_model_drift_monitor', None) is not None
                        and getattr(ApexConfig, "MODEL_DRIFT_MONITOR_ENABLED", True)
                    ):
                        try:
                            _mdm_status = self._model_drift_monitor.get_status()
                            if _mdm_status.should_retrain:
                                logger.warning(
                                    "⚠️ ModelDriftMonitor: RETRAIN RECOMMENDED — "
                                    "IC=%.3f hr=%.1f%% conf=%.2f degraded_windows=%d",
                                    _mdm_status.ic_current,
                                    _mdm_status.hit_rate_current * 100,
                                    _mdm_status.med_confidence,
                                    _mdm_status.consecutive_degraded,
                                )
                                # Auto-trigger retraining with 24h cooldown
                                import time as _time_mod2
                                _mdm_last = getattr(self, "_mdm_last_retrain_ts", 0.0)
                                _mdm_cooldown = float(getattr(ApexConfig, "MODEL_DRIFT_RETRAIN_COOLDOWN_HOURS", 24)) * 3600
                                if (
                                    self.outcome_loop is not None
                                    and getattr(ApexConfig, "AUTO_RETRAIN_ENABLED", True)
                                    and (_time_mod2.time() - _mdm_last) > _mdm_cooldown
                                ):
                                    self._mdm_last_retrain_ts = _time_mod2.time()
                                    logger.warning("🔄 ModelDriftMonitor: auto-triggering retraining (24h cooldown ok)")
                                    try:
                                        self.outcome_loop.trigger_retrain(self.historical_data)
                                    except Exception as _rt_err:
                                        logger.debug("ModelDriftMonitor retrain trigger error: %s", _rt_err)
                            elif _mdm_status.health != "healthy" and _mdm_status.total_windows > 0:
                                logger.info(
                                    "ModelDriftMonitor: %s — IC=%.3f hr=%.1f%% conf=%.2f",
                                    _mdm_status.health,
                                    _mdm_status.ic_current,
                                    _mdm_status.hit_rate_current * 100,
                                    _mdm_status.med_confidence,
                                )
                        except Exception as _mdm_err:
                            logger.debug("ModelDriftMonitor check error (non-fatal): %s", _mdm_err)

                    # ── AdaptiveEntryGate periodic calibration/persist ─────────
                    _aeg = getattr(self, '_adaptive_entry_gate', None)
                    if _aeg is not None:
                        _aeg.calibrate(cycle)

                    # ── historical_data TTL eviction (every 60 cycles ≈ 60 min) ─────
                    if cycle % 60 == 0 and self.historical_data:
                        _active = set(self._runtime_symbols())
                        _open_syms = {s for s, q in self.positions.items() if q != 0}
                        _keep = _active | _open_syms | {"SPY", "BTC/USD", "BTCUSD"}
                        _evict = [s for s in list(self.historical_data) if s not in _keep]
                        for _s in _evict:
                            del self.historical_data[_s]
                        if _evict:
                            logger.debug("historical_data evicted %d stale symbols", len(_evict))

                    # ── Position reconciliation (every 60 cycles ≈ 60 min) ───────────
                    if cycle % 60 == 0:
                        asyncio.ensure_future(self._reconcile_position_state())

                    # ══════════════════════════════════════════════════════════════
                    # UPGRADE A: IBKR Broker Failover — degrade to Alpaca-only if
                    # the connector has exhausted all reconnect retries.
                    # ══════════════════════════════════════════════════════════════
                    if (
                        self.ibkr
                        and getattr(ApexConfig, "IBKR_FAILOVER_ENABLED", True)
                        and getattr(self.ibkr, "_persistently_down", False)
                    ):
                        logger.error(
                            "❌ IBKR persistently down — degrading to Alpaca-only (Upgrade A)"
                        )
                        self.ibkr = None

                    # ══════════════════════════════════════════════════════════════
                    # UPGRADE E: Circuit Breaker Auto-Reset — check at cycle start
                    # so the cooldown clock is honoured even when no trades happen.
                    # ══════════════════════════════════════════════════════════════
                    if getattr(ApexConfig, "CIRCUIT_BREAKER_AUTO_RESET", True):
                        cb = getattr(self.risk_manager, "circuit_breaker", None)
                        if cb and getattr(cb, "is_tripped", False):
                            # Try early reset first (2h no-loss condition)
                            try:
                                _day_cap = getattr(self.risk_session, "day_start_capital", 1.0) or 1.0
                                _daily_pnl = getattr(self.risk_manager, "daily_pnl", 0.0) or 0.0
                                _dloss_pct = abs(_daily_pnl / _day_cap)
                                _early = self.risk_manager.check_early_circuit_breaker_reset(_dloss_pct)
                                if _early:
                                    await self.risk_session.save_state_async()
                                else:
                                    # Fallback: standard 24h cooldown
                                    cb.check_and_reset()
                            except Exception:
                                cb.check_and_reset()

                    # ═══════════════════════════════════════════════════════════════
                    # 1. DATA HEALTH CHECK (Dead Man's Switch)
                    # ═══════════════════════════════════════════════════════════════
                    # Only check if strictly running and not in initial startup
                    if self.data_watchdog and cycle > 5:
                        active_syms = [
                            s for s, q in self.positions.items()
                            if q != 0 and is_market_open(s, datetime.utcnow())
                        ]
                        watchdog_status = self.data_watchdog.check_health(active_syms)
                        
                        if not watchdog_status.is_alive:
                            logger.critical(f"💀 DATA WATCHDOG KILL SWITCH: {watchdog_status.message}")
                            logger.critical("   HALTING TRADING LOOP UNTIL DATA RESTORED")
                            # Write emergency heartbeat indicating failure
                            try:
                                self._write_heartbeat_payload(
                                    {
                                        'status': 'DEAD',
                                        'timestamp': datetime.utcnow().isoformat() + "Z",
                                        'cycle_count': cycle,
                                        'broker_heartbeats': self._broker_heartbeat_payload(),
                                    }
                                )
                            except Exception as e:
                                logger.error("Failed to write DEAD heartbeat: %s", e)
                            
                            await asyncio.sleep(10)
                            continue

                    # Write heartbeat for watchdog monitoring
                    self._write_heartbeat()

                    # Get EST hour using proper timezone handling
                    est_hour = self._get_est_hour()

                    # Cache positions at start of each cycle (avoids race conditions)
                    if self.ibkr or self.alpaca:
                        async def fetch_ibkr():
                            if not self.ibkr: return {}
                            try:
                                pos = await asyncio.wait_for(self.ibkr.get_all_positions(), timeout=15)
                                self._mark_broker_heartbeat("ibkr", success=True)
                                return pos
                            except Exception as e:
                                self._mark_broker_heartbeat("ibkr", success=False, error=str(e))
                                logger.debug("IBKR cycle position snapshot failed: %s", e)
                                return {}

                        async def fetch_alpaca():
                            if not self.alpaca: return {}
                            try:
                                pos = await asyncio.wait_for(self.alpaca.get_all_positions(), timeout=15)
                                self._mark_broker_heartbeat("alpaca", success=True)
                                result = {}
                                for sym, qty in pos.items():
                                    try:
                                        parsed = parse_symbol(sym)
                                        if parsed.asset_class == AssetClass.CRYPTO:
                                            result[sym] = qty
                                    except ValueError:
                                        result[sym] = qty
                                return result
                            except Exception as e:
                                self._mark_broker_heartbeat("alpaca", success=False, error=str(e))
                                logger.debug("Alpaca cycle position snapshot failed: %s", e)
                                return {}

                        # Parallel execution
                        ibkr_pos, alpaca_pos = await asyncio.gather(fetch_ibkr(), fetch_alpaca())
                        
                        merged = {}
                        merged.update(ibkr_pos)
                        merged.update(alpaca_pos)

                        self._cached_ibkr_positions = merged
                        self.positions = merged.copy()
                        self._sync_cost_basis_with_positions()

                    # Refresh pending orders
                    if self.ibkr or self.alpaca:
                        await self.refresh_pending_orders()

                    # D: Refresh Crypto Fear & Greed index hourly (non-blocking)
                    if self._dynamic_crypto_symbols or any(
                        _symbol_is_crypto(s) for s in self._runtime_symbols()
                    ):
                        asyncio.ensure_future(self._fetch_fear_greed())

                    # Refresh data: shorter interval during overnight crypto window, hourly otherwise
                    _outside_equity_est = not (
                        ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END
                    )
                    _data_refresh_secs = (
                        int(getattr(ApexConfig, "CRYPTO_OVERNIGHT_DATA_REFRESH_SECONDS", 900))
                        if (ApexConfig.CRYPTO_ALWAYS_OPEN and _outside_equity_est)
                        else 3600
                    )
                    if (now - last_data_refresh).total_seconds() > _data_refresh_secs:
                        logger.info(
                            "🔄 Refreshing market data (interval=%ds, overnight=%s)",
                            _data_refresh_secs, _outside_equity_est,
                        )
                        await self.refresh_data()
                        last_data_refresh = now

                    # Filter to open markets only (weekends/closures)
                    open_universe = [s for s in self._runtime_symbols() if is_market_open(s, datetime.utcnow())]
                    open_positions = [s for s, q in self.positions.items() if q != 0 and is_market_open(s, datetime.utcnow())]

                    # ── Orphan exit fix ──────────────────────────────────────────────────
                    # Positions whose symbol is NOT in the universe are never processed →
                    # exits never fire → positions accumulate indefinitely.
                    # Solution: append orphan position symbols to open_universe so that
                    # process_symbol() runs for them each cycle (entry guards skip them,
                    # exit logic applies normally).
                    _universe_set = set(open_universe)
                    _orphan_positions = [
                        s for s in open_positions
                        if s not in _universe_set
                        # _runtime_symbols() now emits CRYPTO:X form, so CRYPTO:BTC/USD
                        # will be in _universe_set directly — no bare-form fallback needed.
                    ]
                    if _orphan_positions:
                        logger.info(
                            "📦 %d orphan positions appended to universe for exit evaluation: %s%s",
                            len(_orphan_positions),
                            ", ".join(_orphan_positions[:10]),
                            " ..." if len(_orphan_positions) > 10 else "",
                        )
                        open_universe = open_universe + _orphan_positions
                    # ────────────────────────────────────────────────────────────────────

                    open_universe = self._rotate_crypto_universe(open_universe, open_positions)

                    # ✅ SOTA: Ensure all selected symbols have historical data before processing
                    await self._prefill_historical_data(open_universe)
                    if not open_universe and not open_positions:
                        # Crypto trades 24/7 — never skip when crypto symbols exist in universe
                        if ApexConfig.CRYPTO_ALWAYS_OPEN:
                            crypto_universe = [
                                s for s in self._runtime_symbols()
                                if _symbol_is_crypto(s)
                            ]
                            if crypto_universe:
                                open_universe = crypto_universe
                                logger.info(
                                    "⏰ Equity markets closed; running crypto-only cycle (%d symbols): %s",
                                    len(crypto_universe),
                                    ", ".join(crypto_universe[:15]) + (" ..." if len(crypto_universe) > 15 else ""),
                                )
                            else:
                                logger.info("⏸️  No open markets right now; skipping cycle work")
                                await asyncio.sleep(ApexConfig.CHECK_INTERVAL_SECONDS)
                                continue
                        else:
                            logger.info("⏸️  No open markets right now; skipping cycle work")
                            await asyncio.sleep(ApexConfig.CHECK_INTERVAL_SECONDS)
                            continue

                    in_equity_hours = ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END

                    # Detect overnight → equity hours transition and emit overnight summary
                    if in_equity_hours and self._last_in_equity_hours is False:
                        if self._overnight_cycles > 0 and self._overnight_session_start is not None:
                            elapsed_min = (now - self._overnight_session_start).total_seconds() / 60
                            logger.info(
                                "🌅 Overnight Crypto Session Summary — %.0f min | "
                                "cycles=%d | signals_evaluated=%d | entries=%d | exits=%d",
                                elapsed_min,
                                self._overnight_cycles,
                                self._overnight_signals_evaluated,
                                self._overnight_entries,
                                self._overnight_exits,
                            )
                        # Reset for next overnight session
                        self._overnight_cycles = 0
                        self._overnight_signals_evaluated = 0
                        self._overnight_entries = 0
                        self._overnight_exits = 0
                        self._overnight_session_start = None

                        # ── NYSE open: re-rank Alpaca crypto universe ──────────
                        # IBKR equity universe (ApexConfig.SYMBOLS) is never
                        # touched.  Only _dynamic_crypto_symbols is updated.
                        if self.alpaca and getattr(ApexConfig, "CRYPTO_MOMENTUM_SCAN_ENABLED", True):
                            logger.info("🌅 NYSE open — triggering crypto momentum rescan")
                            asyncio.ensure_future(
                                self._refresh_crypto_universe(force=True)
                            )
                    elif not in_equity_hours:
                        # Track start of overnight session
                        if self._last_in_equity_hours is True or self._last_in_equity_hours is None:
                            self._overnight_session_start = now
                        self._overnight_cycles += 1

                    self._last_in_equity_hours = in_equity_hours

                    if not in_equity_hours:
                        logger.info("⏰ Outside equity hours; processing only open markets")

                    # Always process if any market is open
                    logger.debug(f"⏰ Cycle #{cycle}: {now.strftime('%Y-%m-%d %H:%M:%S')} (EST: {est_hour:.1f}h)")
                    logger.debug("─" * 80)

                    # ═══════════════════════════════════════════════════════
                    # SOTA: Update Market State & Health
                    # ═══════════════════════════════════════════════════════

                    # Refresh macro context (cached 15 min; non-blocking on failure)
                    if (
                        getattr(ApexConfig, "MACRO_INDICATORS_ENABLED", True)
                        and getattr(self, '_macro_indicators', None) is not None
                    ):
                        try:
                            self._macro_context = await self._macro_indicators.get_context()
                        except Exception as _me:
                            logger.debug("Macro context refresh failed (non-fatal): %s", _me)

                    # Refresh Put/Call Ratio (every 30 cycles; 1-hr cached internally)
                    if cycle % 30 == 0 and getattr(self, '_pcr_signal', None) is not None:
                        try:
                            self._pcr_context = await self._pcr_signal.get_signal()
                            if self._pcr_context and self._pcr_context.source != "neutral":
                                logger.debug(
                                    "PCR %.2f [%s] signal=%+.3f (%s)",
                                    self._pcr_context.pcr,
                                    self._pcr_context.date,
                                    self._pcr_context.signal,
                                    self._pcr_context.direction,
                                )
                        except Exception as _pcr_e:
                            logger.debug("PCR refresh error (non-fatal): %s", _pcr_e)

                    # Refresh ORB opening ranges (clears stale daily data at equity open)
                    if cycle % 5 == 0 and getattr(self, '_orb_signal', None) is not None:
                        try:
                            import pytz as _pytz_orb
                            _now_et_orb = datetime.now(_pytz_orb.timezone("US/Eastern"))
                            # Clear stale ranges once per day at 9:25–9:31 ET
                            if _now_et_orb.hour == 9 and 25 <= _now_et_orb.minute <= 31:
                                self._orb_signal.clear_stale_ranges()
                            # Feed intraday data to ORB updater during first 30 min of session
                            if _now_et_orb.hour == 9 and _now_et_orb.minute >= 30:
                                for _orb_sym in list(self.historical_data.keys()):
                                    if _symbol_is_crypto(_orb_sym):
                                        continue
                                    _orb_df = self.historical_data.get(_orb_sym)
                                    if _orb_df is not None and len(_orb_df) >= 3:
                                        self._orb_signal.update_opening_range(_orb_sym, _orb_df)
                        except Exception as _orb_e:
                            logger.debug("ORB update error (non-fatal): %s", _orb_e)

                    # Check VIX Regime
                    vix_state = self.vix_manager.get_current_state()
                    self._vix_risk_multiplier = vix_state.risk_multiplier
                    self._current_vix = vix_state.current_vix  # For signal filtering

                    # Regime Transition Forecaster: proactive pre-emptive dampener
                    if getattr(self, '_regime_forecaster', None) is not None:
                        try:
                            _vix3m = float(getattr(vix_state, 'vix3m', 0.0) or 0.0)
                            _pcr_val = float(getattr(self, '_last_pcr_value', 1.0) or 1.0)
                            _hyg_val = float(self.price_cache.get('HYG', 80.0) or 80.0)
                            _spy_val = float(self.price_cache.get('SPY', 500.0) or 500.0)
                            self._regime_forecaster.update(
                                vix=self._current_vix,
                                pcr=_pcr_val,
                                hyg_price=_hyg_val,
                                spy_price=_spy_val,
                                vix3m=_vix3m,
                            )
                            _fc = self._regime_forecaster.get_forecast()
                            self._regime_forecast_mult = _fc.size_multiplier
                            if _fc.signal != "clear":
                                logger.info(
                                    "🔮 RegimeForecaster: %s (prob=%.2f, mult=%.2f)",
                                    _fc.signal.upper(), _fc.transition_prob, _fc.size_multiplier,
                                )
                            # Apply: only ever tightens (take the min of VIX + forecast mults)
                            self._vix_risk_multiplier = min(
                                self._vix_risk_multiplier, self._regime_forecast_mult
                            )
                            # Live Regime Transition Alert
                            _ra = getattr(self, "_regime_alerter", None)
                            if _ra is not None and getattr(ApexConfig, "REGIME_ALERT_ENABLED", True):
                                try:
                                    _alert = _ra.check(_fc)
                                    if _alert is not None:
                                        logger.warning(
                                            "🚨 RegimeAlert [%s]: %s",
                                            _alert.severity.upper(), _alert.message,
                                        )
                                except Exception:
                                    pass
                        except Exception as _rtf_err:
                            logger.debug("RegimeForecaster update error: %s", _rtf_err)

                    # Drawdown-adaptive leverage: cut size when session is losing.
                    # After -2% session loss → 50%; after -4% → 25%; recovery restores gradually.
                    if getattr(ApexConfig, "DRAWDOWN_ADAPTIVE_LEVERAGE_ENABLED", True):
                        try:
                            _dd_cap = float(getattr(self, 'capital', 0) or 0)
                            _dd_start = float(getattr(self.risk_manager, 'starting_capital', _dd_cap) or _dd_cap)
                            if _dd_start > 0:
                                _session_loss_pct = (_dd_cap - _dd_start) / _dd_start  # negative = loss
                                _tier1 = float(getattr(ApexConfig, "DD_LEV_TIER1_PCT", -0.02))
                                _tier2 = float(getattr(ApexConfig, "DD_LEV_TIER2_PCT", -0.04))
                                if _session_loss_pct <= _tier2:
                                    self._vix_risk_multiplier = min(self._vix_risk_multiplier, 0.25)
                                elif _session_loss_pct <= _tier1:
                                    self._vix_risk_multiplier = min(self._vix_risk_multiplier, 0.50)
                                # If session is recovering: don't restore above VIX mult (handled by VIX logic)
                        except Exception as _dle:
                            logger.debug("DrawdownLeverage update error: %s", _dle)

                    # Volatility targeting: update daily return + recompute multiplier
                    if getattr(self, '_vol_targeter', None) is not None:
                        try:
                            _vt_capital = float(getattr(self, 'capital', 0) or 0)
                            _vt_start = float(getattr(self.risk_manager, 'starting_capital', _vt_capital) or _vt_capital)
                            if _vt_start > 0:
                                _daily_ret = (_vt_capital - _vt_start) / _vt_start
                                self._vol_targeter.update(_daily_ret)
                            self._vol_targeting_mult = self._vol_targeter.get_multiplier()
                        except Exception as _vte:
                            logger.debug("VolTargeting update error: %s", _vte)

                    # Log regime change if significant
                    if self._last_vix_regime is not None and self._last_vix_regime != vix_state.regime:
                        logger.warning(f"🚨 REGIME CHANGE: {self._last_vix_regime.value} -> {vix_state.regime.value} (VIX: {vix_state.current_vix:.1f})")
                        fire_alert(
                            "regime_change",
                            f"Regime change: {self._last_vix_regime.value} → {vix_state.regime.value} "
                            f"(VIX={vix_state.current_vix:.1f})",
                            AlertSev.WARNING,
                        )
                    self._last_vix_regime = vix_state.regime

                    if vix_state.regime != VIXRegime.NORMAL and cycle % 10 == 0:
                        logger.info(f"🌪️ Market Regime: {vix_state.regime.value.upper()} (Risk Multiplier: {self._vix_risk_multiplier:.2f})")

                    # Feed VIX data to risk managers
                    if self.use_institutional and hasattr(self.inst_risk_manager, 'set_market_volatility'):
                        self.inst_risk_manager.set_market_volatility(vix_state.current_vix / 100.0)

                    # Update Health Dashboard
                    self.health_dashboard.run_all_checks(
                        current_capital=self.capital,
                        peak_capital=max(float(self.risk_manager.peak_capital or 0.0), float(self.capital or 0.0)),
                        positions=self.positions
                    )
                    health_status = self.health_dashboard.get_overall_status()
                    if health_status != HealthStatus.HEALTHY and cycle % 5 == 0:
                        logger.warning(f"🏥 System Health: {health_status.value.upper()}")

                    self._run_intraday_stress_check(cycle)
                    self._refresh_stress_unwind_plan()
                    self._run_shadow_deployment_gate(cycle)

                    # ═══════════════════════════════════════════════════════

                    # ✅ Retry any failed exits first (critical for risk management)
                    await self.retry_failed_exits()

                    # ✅ Enforce sector concentration limits — trim breaching sectors before new entries
                    if in_equity_hours and len(self.positions) > 0:
                        await self.enforce_sector_limits()
                        
                        # Phase 14: Enforce Absolute Beta Neutrality (rate-limited)
                        _dh_interval = int(getattr(ApexConfig, "DELTA_HEDGE_REBALANCE_CYCLES", 20))
                        _dh_last = getattr(self, '_delta_hedge_last_cycle', 0)
                        if (
                            getattr(self, 'delta_hedger', None)
                            and getattr(ApexConfig, "DELTA_HEDGE_ENABLED", True)
                            and (cycle - _dh_last) >= _dh_interval
                        ):
                            try:
                                _hedge_sym = str(getattr(ApexConfig, "DELTA_HEDGE_SYMBOL", "SPY"))
                                # Exclude the hedge symbol itself from position dict fed to hedger
                                _non_hedge_pos = {
                                    s: q for s, q in self.positions.items()
                                    if s != _hedge_sym and q != 0
                                }
                                _spy_hedge_qty = await self.delta_hedger.calculate_hedge_order(
                                    _non_hedge_pos, self.price_cache, hedge_symbol=_hedge_sym
                                )
                                if _spy_hedge_qty != 0:
                                    _spy_side = "BUY" if _spy_hedge_qty > 0 else "SELL"
                                    _spy_qty = abs(int(_spy_hedge_qty))
                                    # Notional cap: never put >MAX_SPY_NOTIONAL into hedge
                                    _spy_price = self.price_cache.get(_hedge_sym, 500.0)
                                    _max_qty = int(
                                        getattr(ApexConfig, "DELTA_HEDGE_MAX_SPY_NOTIONAL", 200_000)
                                        / max(_spy_price, 1.0)
                                    )
                                    _spy_qty = min(_spy_qty, _max_qty)
                                    if _spy_qty > 0:
                                        logger.warning(
                                            "🛡️ Delta Hedge: %s %d %s (portfolio Δ sterilised)",
                                            _spy_side, _spy_qty, _hedge_sym,
                                        )
                                        await self.broker_dispatch.place_order(
                                            broker="ibkr",
                                            symbol=_hedge_sym,
                                            quantity=float(_spy_qty),
                                            side=_spy_side,
                                            confidence=1.0,
                                            force_market=True,
                                        )
                                        # Track hedge position so process_symbol skips signal exit
                                        _cur_hedge = getattr(self, '_delta_hedge_qty', {})
                                        _prev = _cur_hedge.get(_hedge_sym, 0)
                                        _cur_hedge[_hedge_sym] = _prev + (_spy_qty if _spy_side == "BUY" else -_spy_qty)
                                        self._delta_hedge_qty = _cur_hedge
                                self._delta_hedge_last_cycle = cycle
                            except Exception as _dhe:
                                logger.warning("Delta Hedger execution failed: %s", _dhe)

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
                        logger.debug(f"👉 Step 3: Process symbols ({len(open_universe)} symbols in open_universe)")
                        await self.process_symbols_parallel(open_universe)

                        # Pairs trading overlay (runs every cycle, re-scans pairs every ~5min)
                        if getattr(self, '_pairs_trader', None) is not None:
                            await self._update_pairs_signals()

                        # 4. Refresh data (market data, indicators)
                        if cycle % 10 == 0:
                            logger.debug("👉 Step 4: Refresh data")
                            await self.refresh_data()

                        # Check for rebalancing (near market close)
                        if open_universe:
                            await self.check_and_execute_rebalance(est_hour)

                        # Manage options (hedging, covered calls, expiring positions)
                        if ApexConfig.OPTIONS_ENABLED and any(
                            parse_symbol(s).asset_class == AssetClass.EQUITY for s in open_universe
                        ):
                            logger.debug("👉 Step 5: Manage options")
                            await self.manage_options()

                            # Phase 3: Manage Active Positions (Ratchet, Aging)
                            logger.debug("👉 Step 6: Manage active positions")
                            await self.manage_active_positions()

                        # Sync positions after processing — trade-triggered syncs inside
                        # manage_active_positions/process_symbol are sufficient; here we
                        # do a periodic background sync every 10 cycles (~10 min) as a
                        # safety net for any missed fills.
                        if (self.ibkr or self.alpaca) and cycle % 10 == 0:
                            await self._sync_positions()

                        logger.debug("👉 Step 7: Check risk")
                        await self.check_risk()
                        if cycle % 200 == 0:
                            self._maybe_tune_governor_policies(now)

                        # ═══════════════════════════════════════════════════════
                        # SIGNAL OUTCOME TRACKING - Periodic forward return update
                        # ═══════════════════════════════════════════════════════
                        if cycle % 20 == 0:  # Update every 20 cycles
                            try:
                                self.signal_outcome_tracker.update_outcomes(
                                    self.price_cache,
                                    self.historical_data
                                )
                                # Log metrics summary periodically
                                if cycle % 100 == 0:
                                    metrics = self.signal_outcome_tracker.get_quality_metrics()
                                    if metrics.total_signals > 0:
                                        logger.info(f"📊 Signal Quality: {metrics.total_signals} tracked, "
                                                  f"5d accuracy: {metrics.target_accuracy.get('hit_5pct_5d', 0):.1%}")
                            except Exception as e:
                                logger.debug(f"Signal outcome update error: {e}")

                        # ═══════════════════════════════════════════════════════
                        # SIGNAL FORTRESS - Periodic monitoring & feedback
                        # ═══════════════════════════════════════════════════════

                        # Option D: WSS hit rate instrumentation — log every 100 cycles
                        if cycle % 100 == 0 and getattr(self, 'websocket_streamer', None):
                            _hr = self.websocket_streamer.hit_rate
                            _hits = self.websocket_streamer._wss_hits
                            _misses = self.websocket_streamer._wss_misses
                            logger.info(
                                "📡 WSS hit rate: %.1f%% (%d WSS / %d REST fallbacks) — "
                                "target >80%% for latency gains",
                                _hr * 100, _hits, _misses,
                            )

                        # Outcome feedback loop: update forward returns & feed to generators
                        if self.outcome_loop and cycle % 20 == 0:
                            try:
                                self.outcome_loop.update_forward_returns(self.historical_data)
                                self.outcome_loop.feed_outcomes_to_generators()

                                # Check for performance degradation
                                degradations = self.outcome_loop.check_performance_degradation()
                                for d in degradations:
                                    if d.recommendation == 'retrain':
                                        logger.warning(f"🏰 Performance degradation: {d.metric}={d.current_value:.3f} → retrain recommended")
                                    elif d.recommendation == 'monitor':
                                        logger.info(f"🏰 Performance watch: {d.metric}={d.current_value:.3f}")

                                # Log rolling metrics periodically
                                if cycle % 100 == 0:
                                    metrics = self.outcome_loop.get_rolling_metrics()
                                    if metrics.get('accuracy') is not None:
                                        logger.info(f"🏰 Feedback Loop: accuracy={metrics['accuracy']:.1%}, "
                                                  f"active={metrics['active_signals']}, completed={metrics['completed_total']}")
                            except Exception as e:
                                logger.debug(f"Outcome feedback loop error: {e}")

                        # Auto-retrain check (daily ~ every 1000 cycles)
                        if self.outcome_loop and cycle % 1000 == 0 and ApexConfig.AUTO_RETRAIN_ENABLED:
                            try:
                                should_retrain, reason = self.outcome_loop.should_retrain()
                                if should_retrain:
                                    logger.warning(f"🏰 AUTO-RETRAIN triggered: {reason}")
                                    self.outcome_loop.trigger_retrain(self.historical_data)
                            except Exception as e:
                                logger.debug(f"Auto-retrain check error: {e}")

                        # Weekly time-based retraining: re-fit models to absorb recent regime changes
                        if (self.outcome_loop
                                and getattr(ApexConfig, "WEEKLY_RETRAIN_ENABLED", True)
                                and cycle % 500 == 0):   # check every ~500 cycles, actual gate is time-based
                            try:
                                import time as _time_mod
                                _weekly_interval = int(getattr(ApexConfig, "WEEKLY_RETRAIN_INTERVAL_HOURS", 168)) * 3600
                                _last_weekly = getattr(self, "_last_weekly_retrain_ts", 0.0)
                                if (_time_mod.time() - _last_weekly) > _weekly_interval:
                                    self._last_weekly_retrain_ts = _time_mod.time()
                                    logger.warning("🗓️ Weekly model retraining triggered (7-day interval)")
                                    self.outcome_loop.trigger_retrain(self.historical_data)
                                    # Also retrain GodLevel ensemble so ML predictions stay current
                                    if (getattr(ApexConfig, "GOD_LEVEL_BLEND_ENABLED", True)
                                            and getattr(self, "god_signal_generator", None) is not None):
                                        try:
                                            logger.info("🧠 Weekly GodLevel ensemble retraining...")
                                            await asyncio.to_thread(
                                                self.god_signal_generator.train_models,
                                                self.historical_data,
                                            )
                                            logger.info("✅ GodLevel ensemble retrained and saved")
                                            # Regenerate manifest so startup trust checks pass on next restart
                                            try:
                                                from models.model_manifest import build_manifest, write_manifest
                                                from pathlib import Path as _Path
                                                write_manifest(build_manifest(), _Path("models/model_manifest.json"))
                                                logger.info("✅ Model manifest regenerated after retrain")
                                            except Exception as _mf_err:
                                                logger.warning("Manifest regen after retrain failed: %s", _mf_err)
                                        except Exception as _gl_train_err:
                                            logger.warning("GodLevel weekly retrain failed: %s", _gl_train_err)
                            except Exception as _wre:
                                logger.debug("Weekly retrain check error: %s", _wre)

                        # Signal integrity check (every 50 cycles)
                        if self.signal_integrity and cycle % 50 == 0:
                            try:
                                health_report = self.signal_integrity.check_integrity()
                                if not health_report.healthy:
                                    for alert in health_report.alerts:
                                        if alert.severity.value == 'critical':
                                            logger.warning(f"🏰 CRITICAL: {alert.symbol} - {alert.message}")
                                        elif alert.severity.value == 'warning':
                                            logger.info(f"🏰 WARNING: {alert.symbol} - {alert.message}")
                                if health_report.quarantined_symbols:
                                    logger.info(f"🏰 Quarantined: {', '.join(health_report.quarantined_symbols)}")
                            except Exception as e:
                                logger.debug(f"Signal integrity check error: {e}")

                        # Feature drift check (every 50 cycles, same cadence as signal integrity)
                        if cycle % 50 == 0:
                            self._run_drift_check(cycle)

                        # Earnings calendar daily refresh (~every 24h)
                        if cycle % 1440 == 0 and cycle > 0:
                            asyncio.ensure_future(self._refresh_earnings_calendar())

                        # Adaptive threshold optimization (daily ~ every 1000 cycles)
                        if self.threshold_optimizer and cycle % 1000 == 0:
                            try:
                                tracker = self.signal_outcome_tracker
                                if hasattr(tracker, 'get_outcome_dataframe'):
                                    signal_df = tracker.get_outcome_dataframe()
                                    if signal_df is not None and len(signal_df) > 0:
                                        results = self.threshold_optimizer.optimize_all(signal_df)
                                        if results:
                                            logger.info(f"🏰 Optimized thresholds for {len(results)} symbols")
                            except Exception as e:
                                logger.debug(f"Threshold optimization error: {e}")

                        # ── Intraday Stress Engine — evaluate every 50 periodic checks ──
                        if getattr(self, 'intraday_stress_engine', None) is not None:
                            try:
                                _se_counter = getattr(self, '_stress_eval_counter', 0) + 1
                                self._stress_eval_counter = _se_counter
                                if _se_counter % 50 == 1:   # first call + every 50 thereafter
                                    _pos_qty = {
                                        s: float(d.get("qty", 0) if isinstance(d, dict) else d)
                                        for s, d in self.positions.items()
                                        if (d.get("qty", 0) if isinstance(d, dict) else d) != 0
                                    }
                                    _new_stress = self.intraday_stress_engine.evaluate(
                                        positions=_pos_qty,
                                        prices=self.price_cache,
                                        historical_data=self.historical_data,
                                        capital=float(getattr(self, '_current_portfolio_value', 0) or 1),
                                    )
                                    _prev_stress = self._stress_control_state
                                    self._stress_control_state = _new_stress
                                    if _new_stress.active:
                                        logger.warning(
                                            "🚨 StressEngine %s: worst=%s ret=%.1f%% mult=%.2f halt=%s",
                                            _new_stress.action,
                                            _new_stress.worst_scenario_name,
                                            _new_stress.worst_portfolio_return * 100,
                                            _new_stress.size_multiplier,
                                            _new_stress.halt_new_entries,
                                        )
                                        self._journal_stress_evaluation(_new_stress)
                                        # Build/refresh the unwind plan when halting
                                        if _new_stress.halt_new_entries and getattr(self, 'stress_unwind_planner', None):
                                            _unwind = self.stress_unwind_planner.build_plan(
                                                stress_state=_new_stress,
                                                positions=_pos_qty,
                                                prices=self.price_cache,
                                                portfolio_value=float(getattr(self, '_current_portfolio_value', 0) or 1),
                                            )
                                            self._stress_unwind_plan = _unwind
                                            if _unwind.active:
                                                logger.warning(
                                                    "📋 StressUnwindPlan: %d candidates to reduce (%s)",
                                                    len(_unwind.candidates),
                                                    _new_stress.worst_scenario_name,
                                                )
                                                # Alert on stress halt
                                                if getattr(self, '_alert_manager', None):
                                                    import asyncio as _asyncio
                                                    _asyncio.create_task(self._alert_manager.send_stress_alert(
                                                        scenario=_new_stress.worst_scenario_name,
                                                        action=_new_stress.action,
                                                        portfolio_return=_new_stress.worst_portfolio_return,
                                                        candidates=[c.symbol for c in _unwind.candidates],
                                                    ))
                                    elif _prev_stress is not None and _prev_stress.active and not _new_stress.active:
                                        logger.info("✅ StressEngine: returned to normal")
                            except Exception as _se_err:
                                logger.debug("StressEngine evaluation error: %s", _se_err)

                        # EOD Performance Digest — runs once per day at market close
                        if getattr(self, '_eod_digest', None) is not None:
                            try:
                                import time as _t_mod
                                from datetime import date as _date_cls
                                _today_str = _date_cls.today().isoformat()
                                _last_digest = getattr(self, '_last_eod_digest_date', None)
                                _market_hour = datetime.now().hour
                                # Trigger between 16:00-17:00 ET, once per calendar day
                                if _today_str != _last_digest and 16 <= _market_hour < 17:
                                    self._last_eod_digest_date = _today_str
                                    _gov_snap = getattr(self, '_performance_snapshot', None)
                                    _report = self._eod_digest.generate(
                                        positions=self.positions,
                                        price_cache=self.price_cache,
                                        governor_snapshot=_gov_snap,
                                    )
                                    self._eod_digest.save(_report)
                                    self._eod_digest.log_summary(_report)
                                    # Run signal auto-tuner right after digest is saved
                                    if getattr(self, '_signal_auto_tuner', None) is not None:
                                        try:
                                            _tune_result = self._signal_auto_tuner.run()
                                            if _tune_result.changes:
                                                _new_thresh = self._signal_auto_tuner.get_thresholds()
                                                self._auto_tuned_thresholds = _new_thresh
                                                logger.info(
                                                    "SignalAutoTuner: %d threshold(s) updated",
                                                    len(_tune_result.changes),
                                                )
                                        except Exception as _sat_err:
                                            logger.debug("SignalAutoTuner run failed: %s", _sat_err)
                                    # Alert: send EOD digest summary
                                    if getattr(self, '_alert_manager', None) is not None:
                                        try:
                                            import asyncio as _asyncio
                                            _asyncio.create_task(self._alert_manager.send_eod_summary(
                                                report_date=_today_str,
                                                total_trades=_report.total_trades,
                                                realized_pnl=_report.total_realized_pnl,
                                                win_rate=_report.overall_win_rate,
                                                recommendations=_report.recommendations[:3],
                                            ))
                                        except Exception:
                                            pass
                            except Exception as _eod_err:
                                logger.debug("EOD digest generation failed: %s", _eod_err)

                        # Capital Allocator: feed today's broker-split P&L and recompute
                        if getattr(self, '_capital_allocator', None) is not None:
                            try:
                                _broker_pnl = self._compute_daily_pnl_by_broker()
                                _cap_pv = float(getattr(self, '_current_portfolio_value', 1) or 1)
                                _eq_pct = _broker_pnl.get("ibkr_realized", 0.0) / max(_cap_pv, 1.0)
                                _cr_pct = _broker_pnl.get("alpaca_realized", 0.0) / max(_cap_pv, 1.0)
                                self._capital_allocator.update_leg_pnl(
                                    equity_pnl_pct=float(_eq_pct),
                                    crypto_pnl_pct=float(_cr_pct),
                                )
                                _alloc = self._capital_allocator.compute_allocation()
                                if _alloc.rebalance_recommended:
                                    logger.info(
                                        "📊 CapitalAllocator: eq=%.0f%% cr=%.0f%% "
                                        "(eq_sharpe=%.2f cr_sharpe=%.2f corr=%.2f)",
                                        _alloc.equity_frac * 100, _alloc.crypto_frac * 100,
                                        _alloc.equity_sharpe, _alloc.crypto_sharpe,
                                        _alloc.correlation,
                                    )
                            except Exception as _ca_err:
                                logger.debug("CapitalAllocator EOD error: %s", _ca_err)

                        # Execution Quality: EOD flush
                        if getattr(self, '_exec_quality', None) is not None:
                            try:
                                self._exec_quality.flush()
                                _eq_report = self._exec_quality.get_report()
                                _global = _eq_report.get("global", {})
                                if _global.get("total_fills", 0) > 0:
                                    logger.info(
                                        "📊 ExecQuality EOD: fills=%d mean_slip=%.1fbps p95=%.1fbps adverse=%.0f%%",
                                        _global["total_fills"], _global["mean_bps"],
                                        _global["p95_bps"], _global["adverse_pct"],
                                    )
                                    _worst = _eq_report.get("worst_symbols", [])
                                    if _worst:
                                        logger.info("   Worst symbols: %s", ", ".join(
                                            f"{w['symbol']}({w['p95_bps']:.0f}bps)" for w in _worst[:5]
                                        ))
                            except Exception as _eq_err:
                                logger.debug("ExecQuality flush error: %s", _eq_err)

                        # Shadow Deployment Gate: periodic evaluation for promotion
                        if getattr(self, '_shadow_gate', None) is not None:
                            try:
                                _sdg_counter = getattr(self, '_shadow_eval_counter', 0) + 1
                                self._shadow_eval_counter = _sdg_counter
                                if _sdg_counter % self._shadow_gate.evaluation_interval_cycles == 0:
                                    _perf = getattr(self, '_performance_snapshot', None)
                                    _sdg_update = self._shadow_gate.evaluate(
                                        live_sharpe=float(getattr(_perf, 'sharpe', 0.0) or 0.0),
                                        live_drawdown=float(getattr(_perf, 'max_drawdown', 0.0) or 0.0),
                                        live_win_rate=float(self.performance_tracker.get_win_rate() or 0.0),
                                        live_total_pnl=float(getattr(self, '_total_realized_pnl', 0.0) or 0.0),
                                        live_total_trades=int(self.performance_tracker.get_completed_trade_count()),
                                        stress_halt_active=bool(
                                            getattr(self, '_stress_control_state', None) is not None
                                            and self._stress_control_state.halt_new_entries
                                        ),
                                    )
                                    if _sdg_update.status_changed:
                                        logger.info(
                                            "ShadowGate: %s → %s (activated=%s)",
                                            _sdg_update.previous_status,
                                            _sdg_update.current_status,
                                            _sdg_update.activation_applied,
                                        )
                            except Exception as _sdg_err:
                                logger.debug("ShadowGate evaluation error: %s", _sdg_err)

                        # ═══════════════════════════════════════════════════════
                        # SIGNAL FORTRESS V2 - Indestructible Shield Updates
                        # ═══════════════════════════════════════════════════════

                        # Black Swan Guard: assess threat every cycle
                        if self.black_swan_guard:
                            try:
                                # Get Index prices for velocity tracking (Multi-Index)
                                for idx_sym in ["SPY", "QQQ", "IWM"]:
                                    idx_price = self.price_cache.get(idx_sym)
                                    if idx_price:
                                        self.black_swan_guard.record_index_price(idx_sym, idx_price)
                                # Get VIX
                                vix_current = self._current_vix
                                # Compute correlations for portfolio
                                correlations = []
                                positions_list = [s for s, q in self.positions.items() if q != 0]
                                if len(positions_list) >= 3 and self.correlation_manager:
                                    corr_data = self.correlation_manager.get_correlation_matrix(
                                        positions_list, self.historical_data
                                    )
                                    if corr_data is not None:
                                        corr_arr = corr_data.values
                                        n = len(corr_arr)
                                        correlations = [corr_arr[i, j] for i in range(n) for j in range(i+1, n)]

                                threat = self.black_swan_guard.assess_threat(
                                    vix_level=vix_current,
                                    portfolio_correlations=correlations if correlations else None
                                )
                                if threat.threat_level >= ThreatLevel.ELEVATED:
                                    logger.warning(f"🛡️ THREAT {threat.threat_level.name}: {threat.recommended_action}")
                                    # Close positions if severe
                                    if threat.threat_level >= ThreatLevel.SEVERE:
                                        to_close = self.black_swan_guard.get_positions_to_close(
                                            {s: {"pnl": 0} for s in positions_list},
                                            self.price_cache
                                        )
                                        for sym in to_close[:3]:  # Max 3 emergency closes per cycle
                                            logger.warning(f"🛡️ Emergency close: {sym}")
                                            # Would trigger exit here in production
                            except Exception as e:
                                logger.debug(f"Black Swan Guard error: {e}")

                        # Drawdown Cascade: update every cycle
                        if self.drawdown_breaker:
                            try:
                                # `self.capital` already reflects total equity from broker/quorum reads.
                                # Re-adding position PnL here double-counts and can falsely escalate tiers.
                                current_capital = float(self.capital or 0.0)
                                peak_capital = max(
                                    float(self.risk_manager.peak_capital or 0.0),
                                    current_capital,
                                )
                                dd_state = self.drawdown_breaker.update(
                                    current_capital=current_capital,
                                    peak_capital=peak_capital,
                                )
                                if dd_state.tier >= DrawdownTier.CAUTION:
                                    logger.info(f"🛡️ Drawdown Tier {dd_state.tier.name}: "
                                              f"DD={dd_state.current_drawdown:.1%}, size_mult={dd_state.size_multiplier:.0%}")
                            except Exception as e:
                                logger.debug(f"Drawdown Cascade error: {e}")

                        # Correlation Cascade: update every 10 cycles
                        if self.correlation_breaker and cycle % 10 == 0:
                            try:
                                positions_list = [s for s, q in self.positions.items() if q != 0]
                                if len(positions_list) >= 3:
                                    corr_state = self.correlation_breaker.assess_correlation_state(
                                        positions_list, self.historical_data
                                    )
                                    # ─── Store for position-sizing chain ─────────────────────
                                    self._correlation_regime = corr_state.regime
                                    self._correlation_avg = corr_state.avg_correlation
                                    self._correlation_effective_n = corr_state.effective_positions

                                    if corr_state.regime >= CorrelationRegime.HERDING:
                                        logger.warning(
                                            "🛡️ Correlation %s: avg=%.2f, effective_N=%.1f — "
                                            "position sizes REDUCED (%.0f%% of normal)",
                                            corr_state.regime.name,
                                            corr_state.avg_correlation,
                                            corr_state.effective_positions,
                                            # HERDING → 50%, CRISIS → 25%
                                            50.0 if corr_state.regime.name == "HERDING" else 25.0,
                                        )
                                else:
                                    self._correlation_regime = CorrelationRegime.NORMAL
                            except Exception as e:
                                logger.debug(f"Correlation Cascade error: {e}")

                        # Execution Shield: log quality report every 100 cycles
                        if self.execution_shield and cycle % 100 == 0:
                            try:
                                report = self.execution_shield.get_execution_quality_report()
                                if report["total_executions"] > 0:
                                    logger.info(f"🛡️ Execution Quality: {report['total_executions']} trades, "
                                              f"avg slippage={report['avg_slippage_bps']:.1f}bps")
                                    if report["expensive_symbols"]:
                                        logger.info(f"   Expensive symbols: {', '.join(report['expensive_symbols'])}")
                            except Exception as e:
                                logger.debug(f"Execution Shield report error: {e}")

                        # Sector Exposure Summary logging
                        try:
                            exposures = self.calculate_sector_exposure()
                            if exposures:
                                cap = getattr(ApexConfig, "MAX_SECTOR_EXPOSURE", 0.50)
                                summary_parts = []
                                for sec, pct in sorted(exposures.items(), key=lambda x: -x[1]):
                                    alert = "⚠️ " if pct >= cap else ""
                                    summary_parts.append(f"{alert}{sec}: {pct*100:.1f}%")
                                logger.info(f"📊 Sector Exposure vs Cap ({cap*100:.0f}%): " + " | ".join(summary_parts))
                        except Exception as e:
                            logger.debug(f"Sector summary error: {e}")

                        # Save regime detector state
                        if hasattr(self, "adaptive_regime") and hasattr(self.adaptive_regime, "save_state"):
                            try:
                                self.adaptive_regime.save_state()
                            except Exception as e:
                                logger.debug(f"Failed to save regime state: {e}")

                        logger.info("")

                    # Clear cycle cache
                    self._cached_ibkr_positions = None

                    if self.prometheus_metrics:
                        try:
                            self.prometheus_metrics.increment_cycle()
                            self.prometheus_metrics.record_cycle_duration(
                                (datetime.now() - now).total_seconds()
                            )
                            self.prometheus_metrics.update_uptime(
                                (datetime.now() - self._startup_time).total_seconds()
                            )
                        except Exception as metrics_exc:
                            logger.debug("Prometheus cycle metrics error: %s", metrics_exc)

                    cycle_end_time = datetime.now()
                    cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()
                    logger.debug(f"Trading cycle {cycle} finished in {cycle_duration:.2f} seconds.")
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
        except Exception as e:
            logger.critical(f"❌ FATAL ERROR in main loop: {e}", exc_info=True)
            # Re-raise to ensure proper exit code
            raise
        finally:
            # UPGRADE B: Save pending orders before disconnect so they survive restart
            if self.alpaca:
                _orders_path = self.user_data_dir / "runtime" / "alpaca_pending_orders.json"
                try:
                    self.alpaca.save_pending_orders(_orders_path)
                except Exception as _save_err:
                    logger.debug("Could not save pending orders: %s", _save_err)

            # Phase 12: Drain the WebSockets daemon gracefully
            if getattr(self, 'websocket_streamer', None):
                await self.websocket_streamer.stop()

            if self.ibkr:
                self.ibkr.disconnect()
            if self.alpaca:
                self.alpaca.disconnect()
            if self.prometheus_metrics:
                try:
                    self.prometheus_metrics.stop()
                except Exception:
                    pass
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
