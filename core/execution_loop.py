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
from monitoring.performance_tracker import PerformanceTracker
from monitoring.live_monitor import LiveMonitor
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
from models.adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORTRESS PHASE 2 - Indestructible Shield
# ═══════════════════════════════════════════════════════════════════════════════
from risk.black_swan_guard import BlackSwanGuard, ThreatLevel
from monitoring.signal_decay_shield import SignalDecayShield
from risk.exit_quality_guard import ExitQualityGuard
from risk.correlation_cascade_breaker import CorrelationCascadeBreaker, CorrelationRegime
from risk.drawdown_cascade_breaker import DrawdownCascadeBreaker, DrawdownTier
from execution.execution_shield import (
    ExecutionShield,
    ExecutionAlgo,
    Urgency as ExecUrgency,
)
from risk.macro_shield import MacroShield
from monitoring.data_watchdog import DataWatchdog
from core.logging_config import setup_logging

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FORTRESS PHASE 3 - Autonomous Money Machine
# ═══════════════════════════════════════════════════════════════════════════════
from risk.macro_event_shield import MacroEventShield, EventType
from risk.overnight_risk_guard import OvernightRiskGuard, MarketPhase
from risk.profit_ratchet import ProfitRatchet, ProfitTier
from risk.liquidity_guard import LiquidityGuard, LiquidityRegime
from risk.position_aging_manager import PositionAgingManager, AgingTier
from risk.trading_excellence import TradingExcellenceManager, quick_mismatch_check, ProfitAction
from risk.performance_governor import PerformanceGovernor, GovernorSnapshot, GovernorTier
from risk.governor_policy import (
    GovernorPolicyRepository,
    GovernorPolicyResolver,
    PolicyPromotionService,
    TierControls,
    default_policy_for,
)
from risk.kill_switch import KillSwitchConfig, RiskKillSwitch
from risk.pretrade_risk_gateway import PreTradeLimitConfig, PreTradeRiskGateway
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
    json_format=False,
    console_output=True,
    max_bytes=ApexConfig.LOG_MAX_BYTES,
    backup_count=ApexConfig.LOG_BACKUP_COUNT,
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
    
    def __init__(self, tenant_id: str = "default", broker_service=None):
        self.tenant_id = tenant_id
        if self.tenant_id == "default":
            self.user_data_dir = ApexConfig.DATA_DIR
        else:
            self.user_data_dir = ApexConfig.DATA_DIR / "users" / self.tenant_id
            self.user_data_dir.mkdir(parents=True, exist_ok=True)

        self.print_banner()
        logger.info("=" * 80)
        logger.info(f"🚀 {ApexConfig.SYSTEM_NAME} V{ApexConfig.VERSION} [Tenant: {self.tenant_id}]")
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
                        logger.info(f"[{self.tenant_id}] Mode: {conn.environment.upper()} TRADING (IBKR)")
                        self.ibkr = IBKRConnector(
                            host=creds.get("host", ApexConfig.IBKR_HOST),
                            port=creds.get("port", ApexConfig.IBKR_PORT),
                            client_id=conn.client_id or ApexConfig.IBKR_CLIENT_ID
                        )
                        if ApexConfig.USE_ADVANCED_EXECUTION:
                            self.advanced_executor = AdvancedOrderExecutor(self.ibkr)
                            logger.info(f"[{self.tenant_id}] Advanced execution (TWAP/VWAP) enabled")
                    
                    elif conn.broker_type == BrokerType.ALPACA:
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
                        self.advanced_executor = AdvancedOrderExecutor(self.ibkr)
                        logger.info("Advanced execution (TWAP/VWAP) enabled")
    
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
            
        self.broker_dispatch = BrokerDispatch(self.ibkr, self.alpaca)
        self.state_sync = StateSync(self.user_data_dir / "trading_state.json")
        
        # Initialize modules
        self.signal_generator = AdvancedSignalGenerator()
        self.risk_manager = RiskManager(
            max_daily_loss=ApexConfig.MAX_DAILY_LOSS,
            max_drawdown=ApexConfig.MAX_DRAWDOWN,
            user_id=self.tenant_id
        )
        self.portfolio_optimizer = PortfolioOptimizer()
        self.market_data = MarketDataFetcher()
        self.performance_tracker = PerformanceTracker()
        self.live_monitor = LiveMonitor()
        self.performance_governor: Optional[PerformanceGovernor] = None
        self._performance_snapshot: Optional[GovernorSnapshot] = None
        self.prometheus_metrics: Optional[PrometheusMetrics] = None
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
                max_order_shares=ApexConfig.PRETRADE_MAX_ORDER_SHARES,
                max_price_deviation_bps=ApexConfig.PRETRADE_MAX_PRICE_DEVIATION_BPS,
                max_participation_rate=ApexConfig.PRETRADE_MAX_PARTICIPATION_RATE,
                max_gross_exposure_ratio=ApexConfig.PRETRADE_MAX_GROSS_EXPOSURE_RATIO,
            ),
            audit_dir=self.user_data_dir / "audit" / "pretrade_gateway",
        )
        logger.info(
            "🧱 Pre-trade gateway %s (notional<=%s, shares<=%s, price_band<=%.0fbps, adv<=%.1f%%, gross<=%.2fx, fail_closed=%s)",
            "enabled" if self.pretrade_gateway.config.enabled else "disabled",
            f"${self.pretrade_gateway.config.max_order_notional:,.0f}",
            self.pretrade_gateway.config.max_order_shares,
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
                min_regime_duration=ApexConfig.MIN_REGIME_DURATION_DAYS,
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

        # Drawdown Cascade Breaker (5-tier drawdown response)
        if ApexConfig.DRAWDOWN_CASCADE_ENABLED:
            self.drawdown_breaker = DrawdownCascadeBreaker(
                initial_capital=ApexConfig.INITIAL_CAPITAL,
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

        # Macro Event Shield (FOMC/CPI/NFP blackouts + Game Over protection)
        self.macro_event_shield = MacroEventShield(
            blackout_before_fomc=60,
            blackout_after_fomc=30,
            blackout_before_data=30,
            blackout_after_data=15,
            game_over_threshold=ApexConfig.GAME_OVER_LOSS_THRESHOLD,
        )
        logger.info("🤖 Phase 3: MacroEventShield enabled (FOMC/CPI/NFP blackouts)")

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
            tier_1_threshold=0.02,  # 2% gain
            tier_2_threshold=0.05,  # 5% gain
            tier_3_threshold=0.10,  # 10% gain
            tier_4_threshold=0.20,  # 20% gain
            initial_trailing_pct=0.03,
        )
        logger.info("🤖 Phase 3: ProfitRatchet enabled (lock in profits)")

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

        # Feature flag for institutional mode
        self.use_institutional = True  # Toggle to enable/disable institutional components
        
        # State
        self.capital = ApexConfig.INITIAL_CAPITAL
        self._last_good_total_equity: float = float(self.capital)
        self.positions: Dict[str, int] = {}  # symbol -> quantity (positive=long, negative=short)
        self.is_running = False
        self._cached_ibkr_positions: Optional[Dict[str, int]] = None  # Cycle-level cache
        self._broker_equity_cache: Dict[str, Tuple[float, datetime]] = {}
        self._broker_cash_cache: Dict[str, Tuple[float, datetime]] = {}
        self._last_good_total_cash: Optional[float] = None
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
        if self.ibkr and getattr(self.ibkr, "port", None) == 7497:
            return True
        alpaca_base = str(getattr(ApexConfig, "ALPACA_BASE_URL", "") or "").lower()
        if self.alpaca and "paper" in alpaca_base:
            return True
        return not bool(getattr(ApexConfig, "LIVE_TRADING", True))

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

    def _sanitize_startup_state_for_paper(self):
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
            self._last_good_total_equity = float(capital)
            drawdown_breaker = getattr(self, "drawdown_breaker", None)
            if drawdown_breaker:
                drawdown_breaker.reset_peak(capital)
                logger.warning(
                    "🩹 Paper startup risk self-heal: reset DrawdownCascadeBreaker peak to $%.2f",
                    capital,
                )
            if (
                ApexConfig.PAPER_STARTUP_RESET_CIRCUIT_BREAKER
                and self.risk_manager.circuit_breaker.is_tripped
            ):
                self.risk_manager.circuit_breaker.reset()
                logger.warning("🩹 Paper startup risk self-heal: cleared persisted circuit breaker latch")
            self.risk_manager.save_state()

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
                self.performance_tracker.reset_history(
                    starting_capital=capital,
                    reason="paper_startup_rebase",
                )

    def _rebase_latches_after_reset_for_paper(self, requested_by: str, reason: str, reset_notes: List[str]) -> None:
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
            self.risk_manager.current_day = datetime.now().strftime("%Y-%m-%d")
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
            self.performance_tracker.reset_history(
                starting_capital=baseline,
                reason="unified_latch_reset",
            )
            reset_notes.append("paper_performance_rebase=applied")
        else:
            reset_notes.append("paper_performance_rebase=disabled")

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
        
        # Load risk state (day_start_capital, etc)
        self.risk_manager.load_state()
        self.load_price_cache()
        # Connect to IBKR
        if self.ibkr:
            try:
                await self.ibkr.connect()
                if self.prometheus_metrics:
                    self.prometheus_metrics.update_ibkr_status(True)
            except Exception as ibkr_exc:
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
            else:
                logger.warning(f"⚠️  IBKR returned ${ibkr_capital:,.2f}, keeping initial capital ${self.capital:,.2f}")
                self.risk_manager.set_starting_capital(self.capital)
                self.risk_manager.day_start_capital = self.capital  # ✅ CRITICAL
                logger.info(f"✅ IBKR Account ${self.capital:,.2f}")

            # Load existing positions from IBKR
            await self.sync_positions_with_ibkr()

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
                                    self.position_entry_times[symbol] = datetime.now()
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
            stream_symbols = list(set(list(self.positions.keys()) + ApexConfig.SYMBOLS))
            backtest_only = getattr(ApexConfig, "BACKTEST_ONLY_SYMBOLS", set()) or set()
            if backtest_only:
                backtest_only_norm = {normalize_symbol(s) for s in backtest_only}
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

            await self.ibkr.stream_quotes(stream_symbols)

        # Connect to Alpaca (crypto paper trading)
        # Connect to Alpaca (crypto paper trading)
        if self.alpaca:
            try:
                await self.alpaca.connect()
                alpaca_equity = await self.alpaca.get_portfolio_value()
                if alpaca_equity > 0 and not self.ibkr:
                    # Only use Alpaca capital when running Alpaca-only mode
                    self.capital = alpaca_equity
                    self.equity_outlier_guard.seed(float(self.capital))
                    self.risk_manager.set_starting_capital(self.capital)
                logger.info(f"Alpaca Account: ${alpaca_equity:,.2f}")

                # Sync Alpaca positions
                alpaca_positions = await self.alpaca.get_all_positions()
                for sym, qty in alpaca_positions.items():
                    if qty != 0:
                        self.positions[sym] = int(qty) if qty == int(qty) else qty
                        logger.info(f"  Alpaca position: {sym} = {qty}")

                # Start crypto quote polling
                crypto_symbols = [
                    s for s in ApexConfig.SYMBOLS
                    if parse_symbol(s).asset_class == AssetClass.CRYPTO
                ]
                if crypto_symbols:
                    if self.data_watchdog:
                        self.alpaca.set_data_callback(self.data_watchdog.feed_heartbeat)
                    await self.alpaca.stream_quotes(crypto_symbols)
            except Exception as e:
                logger.warning(f"⚠️  Alpaca connection failed: {e}. Disabling Alpaca.")
                self.alpaca = None

        # Sync startup capital from broker APIs before self-healing persisted paper state.
        await self._refresh_capital_from_brokers_for_startup()

        # Prevent stale persisted paper state from tripping risk/kill controls at startup.
        self._sanitize_startup_state_for_paper()

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

        # Train institutional signal generator
        if self.use_institutional:
            logger.info("")
            if self.inst_signal_generator.is_trained and not ApexConfig.FORCE_RETRAIN:
                logger.info("✅ Institutional ML models loaded from disk. Skipping training.")
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
            actual_positions = {s: d['qty'] for s, d in detailed_positions.items()}
            
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

            # Update entry prices from IBKR avgCost for positions we don't have metadata for
            for symbol, data in detailed_positions.items():
                if symbol not in self.position_entry_prices or self.position_entry_prices[symbol] == 0:
                    self.position_entry_prices[symbol] = data['avg_cost']
                    if symbol not in self.position_entry_times:
                        self.position_entry_times[symbol] = datetime.now()
                    logger.debug(f"ℹ️ Captured IBKR avgCost for {symbol}: ${data['avg_cost']:.2f}")

            logger.debug(f"✅ Position sync: {self.position_count} active positions")
        
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    async def sync_positions_with_alpaca(self):
        """Sync positions from Alpaca into self.positions with metadata."""
        if not self.alpaca:
            return
        try:
            detailed_positions = await self.alpaca.get_detailed_positions()
            actual_positions = {s: d['qty'] for s, d in detailed_positions.items()}
            
            # Sync quantities
            for sym, qty in actual_positions.items():
                if qty != 0:
                    self.positions[sym] = int(qty) if qty == int(qty) else qty
                    
            # Remove Alpaca symbols that are no longer held
            for sym in list(self.positions.keys()):
                try:
                    parsed = parse_symbol(sym)
                    if parsed.asset_class == AssetClass.CRYPTO and sym not in actual_positions:
                        del self.positions[sym]
                except ValueError:
                    pass
                    
            # Populate price cache and entry prices so Modeled Equity can value them correctly
            for sym, data in detailed_positions.items():
                qty = data['qty']
                if qty == 0:
                    continue
                
                # Update local caches with Alpaca's reality
                if data.get('current_price', 0) > 0:
                    self.price_cache[sym] = data['current_price']
                
                if sym not in self.position_entry_prices or self.position_entry_prices[sym] == 0:
                    self.position_entry_prices[sym] = data.get('avg_cost', 0)
                    if sym not in self.position_entry_times:
                        self.position_entry_times[sym] = datetime.now()
            
            logger.debug(f"✅ Alpaca position sync complete")
        except Exception as e:
            logger.error(f"Error syncing Alpaca positions: {e}")

    async def _sync_positions(self):
        """Sync positions from all active brokers."""
        if self.ibkr:
            await self.sync_positions_with_ibkr()
        if self.alpaca:
            await self.sync_positions_with_alpaca()

    async def _read_broker_equity(self, broker_name: str, connector) -> Optional[float]:
        """Read broker equity and refresh cache on valid updates."""
        if connector is None:
            return None
        try:
            value = float(await connector.get_portfolio_value())
        except Exception as exc:
            logger.debug("Broker equity read failed for %s: %s", broker_name, exc)
            return None
        if value > 0:
            self._broker_equity_cache[broker_name] = (value, datetime.utcnow())
            return value
        return None

    async def _read_broker_cash(self, broker_name: str, connector) -> Optional[float]:
        """Read broker cash and refresh cache on valid updates."""
        if connector is None or not hasattr(connector, "get_account_cash"):
            return None
        try:
            value = float(await connector.get_account_cash())
        except Exception as exc:
            logger.debug("Broker cash read failed for %s: %s", broker_name, exc)
            return None
        if math.isfinite(value):
            self._broker_cash_cache[broker_name] = (value, datetime.utcnow())
            return value
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
        """Return spread gate threshold by asset class."""
        normalized = str(asset_class or "EQUITY").upper()
        if normalized == "FOREX":
            return float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_FX)
        if normalized == "CRYPTO":
            return float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_CRYPTO)
        return float(ApexConfig.EXECUTION_MAX_SPREAD_BPS_EQUITY)

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
        return abs(fill - expected) / expected * 10000.0

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
        for symbol, qty in self.positions.items():
            if qty == 0:
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

            entry_time = self.position_entry_times.get(symbol, datetime.now())
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

    async def _get_total_portfolio_value(self) -> float:
        """Get combined portfolio value across brokers with quorum + cache fallback."""
        brokers = {}
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
        included_values: Dict[str, float] = {}

        for broker_name, connector in brokers.items():
            fresh = await self._read_broker_equity(broker_name, connector)
            if fresh is not None and fresh > 0:
                included_values[broker_name] = fresh
                continue

            cached = self._broker_equity_cache.get(broker_name)
            if not cached:
                continue
            cached_value, cached_at = cached
            age_seconds = (now - cached_at).total_seconds()
            if cached_value > 0 and age_seconds <= stale_seconds:
                included_values[broker_name] = cached_value
                logger.warning(
                    "⚠️ %s equity unavailable, reusing cached value $%.2f (age=%.0fs)",
                    broker_name,
                    cached_value,
                    age_seconds,
                )

        if len(included_values) >= required_quorum:
            total = float(sum(included_values.values()))
            if total > 0:
                self._last_good_total_equity = total
                return total

        if self._last_good_total_equity > 0:
            logger.warning(
                "⚠️ Broker equity quorum not met (%d/%d), using last good total $%.2f",
                len(included_values),
                required_quorum,
                self._last_good_total_equity,
            )
            return float(self._last_good_total_equity)

        return float(self.capital)

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
            if math.isfinite(cached_value) and age_seconds <= stale_seconds:
                included_values[broker_name] = cached_value
                logger.warning(
                    "⚠️ %s cash unavailable, reusing cached value $%.2f (age=%.0fs)",
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
                "⚠️ Broker cash quorum not met (%d/%d), using last good cash $%.2f",
                len(included_values),
                required_quorum,
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
    
    async def check_sector_limit(self, symbol: str) -> bool:
        """
        Check if adding this symbol would exceed sector limits.
        
        Dynamic logic:
        - If < 5 positions, allow up to 80% in a sector.
        - Otherwise, use ApexConfig.MAX_SECTOR_EXPOSURE (0.50).
        """
        sector = ApexConfig.get_sector(symbol)
        current_exposure = self.calculate_sector_exposure()
        
        # Determine dynamic limit
        limit = ApexConfig.MAX_SECTOR_EXPOSURE
        if self.position_count < 5:
            limit = 0.80  # Allow more concentration in early stage
            logger.debug(f"ℹ️ {symbol}: Using relaxed sector limit {limit*100:.0f}% (small portfolio)")
        
        # Check if adding this position would breach limit
        if current_exposure.get(sector, 0) >= limit:
            logger.warning(f"⚠️ {symbol}: Sector limit reached ({sector}: {current_exposure[sector]*100:.1f}% >= {limit*100:.0f}%)")
            return False
        
        return True

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
                asset_class: ts.isoformat()
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

            self._rebase_latches_after_reset_for_paper(
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
            self.risk_manager.save_state()
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

        # Portfolio-level hard kill-switch: block only new entries, never exits.
        if self.kill_switch and self.kill_switch.active:
            existing_qty = self.positions.get(symbol, 0)
            if existing_qty == 0:
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime="default",
                        reason="kill_switch",
                    )
                logger.debug("🛑 %s: New entries blocked by kill-switch", symbol)
                return

        # Equity reconciliation hard block: fail-closed for new entries only.
        if self._equity_reconciliation_block_entries:
            existing_qty = self.positions.get(symbol, 0)
            if existing_qty == 0:
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

        # Check 2.5: Signal Fortress - skip quarantined symbols
        if self.signal_integrity and self.signal_integrity.is_quarantined(symbol):
            logger.debug(f"🏰 {symbol}: Quarantined by signal integrity monitor")
            return

        # Check 2.6: Black Swan Guard - block entries during crash
        if self.black_swan_guard and self.black_swan_guard.should_block_entry():
            logger.debug(f"🛡️ {symbol}: Entry blocked by BlackSwanGuard")
            return

        # Check 2.7: Drawdown Cascade - block entries in high-tier drawdown
        if self.drawdown_breaker and not self.drawdown_breaker.get_entry_allowed():
            logger.debug(f"🛡️ {symbol}: Entry blocked by DrawdownCascadeBreaker")
            return

        # Check 2.8: Correlation Cascade - block entries during herding
        if self.correlation_breaker:
            existing_positions = [s for s, qty in self.positions.items() if qty != 0]
            if self.correlation_breaker.should_block_entry(symbol, existing_positions, self.historical_data):
                logger.debug(f"🛡️ {symbol}: Entry blocked by CorrelationCascadeBreaker")
                return

        # Check 2.9: Macro Shield - block entries during economic events
        if self.macro_shield and self.macro_shield.is_blackout_active():
            # Only log occasionally to avoid spam
            if random.random() < 0.01:
                event = self.macro_shield.get_active_event()
                evt_name = event.title if event else "Unknown Event"
                logger.info(f"🛡️ {symbol}: Entry blocked by Macro Shield ({evt_name})")
            return

        # Check 2.10: Phase 3 Macro Event Shield (More advanced)
        if hasattr(self, 'macro_event_shield') and self.macro_event_shield:
            blocked, reason = self.macro_event_shield.should_block_entry(symbol)
            if blocked:
                if random.random() < 0.01:
                    logger.debug(f"🛡️ {symbol}: Entry blocked by MacroEventShield ({reason})")
                return

        # Check 2.11: Phase 3 Overnight Risk Guard (skip for 24/7 assets)
        if hasattr(self, 'overnight_guard') and self.overnight_guard:
            try:
                _parsed_overnight = parse_symbol(symbol)
                _skip_overnight = _parsed_overnight.asset_class in (AssetClass.CRYPTO, AssetClass.FOREX)
            except ValueError:
                _skip_overnight = False
            if not _skip_overnight:
                blocked, reason = self.overnight_guard.should_block_entry()
                if blocked:
                    if random.random() < 0.01:
                        logger.debug(f"🛡️ {symbol}: Entry blocked by OvernightRiskGuard ({reason})")
                    return

        # Check 2.12: Phase 3 Liquidity Guard
        if hasattr(self, 'liquidity_guard') and self.liquidity_guard:
            blocked, reason = self.liquidity_guard.should_block_entry(symbol)
            if blocked:
                if random.random() < 0.01:
                    logger.debug(f"🛡️ {symbol}: Entry blocked by LiquidityGuard ({reason})")
                return

        # Get data
        if symbol not in self.historical_data:
            return
        
        try:
            # Use cached positions (refreshed at cycle start) to avoid race conditions
            connector = self._get_connector_for(symbol)
            if connector:
                # Use cycle-level cached positions if available
                if self._cached_ibkr_positions is not None:
                    current_pos = self._cached_ibkr_positions.get(symbol, 0)
                else:
                    current_pos = self.positions.get(symbol, 0)

                # Get current price
                price = await connector.get_market_price(symbol)
                if not price or price == 0:
                    logger.debug(f"⚠️ {symbol}: No price available")
                    return

                self.price_cache[symbol] = price
                # Record price freshness for decay shield
                if self.signal_decay_shield:
                    self.signal_decay_shield.record_data_timestamp(symbol, "price")
            else:
                current_pos = self.positions.get(symbol, 0)
                price = float(self.historical_data[symbol]['Close'].iloc[-1])
                self.price_cache[symbol] = price
                if self.signal_decay_shield:
                    self.signal_decay_shield.record_data_timestamp(symbol, "price")

            # Check data freshness before signal generation
            if self.signal_decay_shield and not self.signal_decay_shield.is_data_tradeable(symbol):
                logger.debug(f"🛡️ {symbol}: Data too stale, skipping signal generation")
                return

            # Generate signal (use institutional or standard)
            prices = self.historical_data[symbol]['Close']
            data = self.historical_data[symbol]

            # SOTA: Get Cross-Sectional Momentum
            cs_data = self.cs_momentum.get_signal(symbol, self.historical_data)
            cs_signal = cs_data.get('signal', 0)
            
            # SOTA: Get News Sentiment
            sent_result = self.sentiment_analyzer.analyze(symbol)
            sent_signal = sent_result.sentiment_score
            sent_conf = sent_result.confidence

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
                
                # Note: Blending is now handled within the ML feature matrix once models retrain.
                # For now, we keep the signature-passing logic and let the generator decide.

                # Log component breakdown for quant transparency
                if abs(signal) >= 0.30:
                    direction = "BULLISH" if signal > 0 else "BEARISH"
                    strength = "STRONG" if abs(signal) > 0.50 else "MODERATE"
                    logger.info(f"📊 {symbol}: {strength} {direction} signal={signal:+.3f} conf={confidence:.2f}")
                    logger.debug(f"   Breakdown: Tech={inst_signal.signal:.2f} Mom={cs_signal:.2f}({cs_data.get('rank_percentile', 0.5):.0%}) Sent={sent_signal:.2f}")
                    logger.debug(f"   Components: mom={inst_signal.momentum_signal:.2f} rev={inst_signal.mean_reversion_signal:.2f} "
                                f"trend={inst_signal.trend_signal:.2f} vol={inst_signal.volatility_signal:.2f}")
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
                            'model_agreement': inst_signal.model_agreement,
                        })

                    self.signal_outcome_tracker.record_signal(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        signal_value=signal,
                        confidence=confidence,
                        price=price,
                        direction='LONG' if signal > 0 else 'SHORT',
                        metadata=signal_metadata
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

                # 🏆 EXCELLENCE: Check signal-position mismatch
                if hasattr(self, 'excellence_manager') and not should_exit:
                    mismatch_exit, mismatch_reason = quick_mismatch_check(
                        position_side=side,
                        signal=signal,
                        confidence=confidence,
                        pnl_pct=pnl_pct
                    )
                    if mismatch_exit:
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

                    exit_connector = self._get_connector_for(symbol)
                    if exit_connector:
                        self.pending_orders.add(symbol)

                        # Determine order side
                        order_side = 'SELL' if current_pos > 0 else 'BUY'

                        trade = await exit_connector.execute_order(
                            symbol=symbol,
                            side=order_side,
                            quantity=abs(current_pos),
                            confidence=abs(signal) if signal != 0 else 0.8
                        )

                        if trade:
                            exit_fill_price = float(price)
                            exit_expected_price = float(price)
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

                            # ✅ CRITICAL: Force sync after trade
                            await self._sync_positions()

                            # Track commission
                            commission = ApexConfig.COMMISSION_PER_TRADE
                            self.total_commissions += commission

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
                            self._save_position_metadata()

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

            # ✅ Phase 3.2: Detect market regime (adaptive or legacy)
            if self.adaptive_regime:
                try:
                    regime_assessment = self.adaptive_regime.assess_regime(
                        prices=prices,
                        vix_level=self._current_vix,
                    )
                    self._current_regime = regime_assessment.primary_regime
                    # Log transitions
                    if regime_assessment.transition_probability > 0.7:
                        logger.info(f"🏰 {symbol}: Regime transition likely ({regime_assessment.transition_probability:.0%})")
                except Exception:
                    regime_enum = self.god_signal_generator.detect_market_regime(prices)
                    self._current_regime = regime_enum.value
            else:
                try:
                    regime_enum = self.god_signal_generator.detect_market_regime(prices)
                    self._current_regime = regime_enum.value
                except Exception:
                    self._current_regime = 'neutral'

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

            if current_pos == 0 and governor_controls.halt_new_entries:
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
            if self.threshold_optimizer:
                sym_thresholds = self.threshold_optimizer.get_thresholds(symbol, self._current_regime)
                signal_threshold = sym_thresholds.entry_threshold
            else:
                signal_threshold = ApexConfig.SIGNAL_THRESHOLDS_BY_REGIME.get(
                    self._current_regime, ApexConfig.MIN_SIGNAL_THRESHOLD
                )

            effective_signal_threshold = min(
                0.95, signal_threshold + governor_controls.signal_threshold_boost
            )
            effective_confidence_threshold = min(
                0.95,
                max(
                    ApexConfig.MIN_CONFIDENCE,
                    ApexConfig.MIN_CONFIDENCE + governor_controls.confidence_boost,
                ),
            )

            if abs(signal) < effective_signal_threshold:
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason="threshold",
                    )
                logger.info(
                    f"⏭️ {symbol}: Signal {signal:.3f} below effective threshold "
                    f"{effective_signal_threshold:.3f} ({self._current_regime}, perf={perf_snapshot.tier.value})"
                )
                return

            if confidence < effective_confidence_threshold:
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_governor_blocked_entry(
                        asset_class=asset_class,
                        regime=governor_regime_key,
                        reason="confidence",
                    )
                logger.info(
                    f"⏭️ {symbol}: Confidence {confidence:.3f} below effective minimum "
                    f"{effective_confidence_threshold:.3f} (perf={perf_snapshot.tier.value})"
                )
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

            # ✅ Phase 3: Aggregate Risk Check (Pre-Trade)
            # Check if the account is allowed to open new positions based on total equity
            agg_risk = await self.risk_manager.check_aggregate_risk("default")
            if not agg_risk["allowed"]:
                 logger.warning(f"🛑 {symbol}: Aggregate Risk Limit Breached - {agg_risk['reason']}")
                 if self.prometheus_metrics:
                     self.prometheus_metrics.record_governor_blocked_entry(
                         asset_class=asset_class, regime="default", reason="aggregate_risk"
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

                        # 🏰 Signal Fortress: Apply per-symbol size multiplier
                        if self.threshold_optimizer:
                            size_mult = sym_thresholds.position_size_multiplier
                            if size_mult != 1.0:
                                shares = max(1, int(shares * size_mult))
                                logger.debug(f"   🏰 Adaptive size: {size_mult:.2f}x → {shares} shares")

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
                                shares = max(1, int(shares * bsg_mult))
                                logger.info(f"   🛡️ BlackSwanGuard: {bsg_mult:.0%} → {shares} shares")

                        # 🛡️ Signal Fortress V2: Apply Drawdown Cascade size multiplier
                        if self.drawdown_breaker:
                            dd_mult = self.drawdown_breaker.get_position_size_multiplier()
                            if dd_mult < 1.0:
                                shares = max(1, int(shares * dd_mult))
                                logger.info(f"   🛡️ DrawdownBreaker: {dd_mult:.0%} → {shares} shares")

                        # 🛡️ Signal Fortress V2: Apply Execution Shield slippage adjustment
                        if self.execution_shield:
                            slip_adj = self.execution_shield.get_slippage_adjustment(symbol)
                            if slip_adj < 1.0:
                                shares = max(1, int(shares * slip_adj))
                                logger.debug(f"   🛡️ ExecutionShield slippage adj: {slip_adj:.0%} → {shares} shares")

                        if governor_controls.size_multiplier < 1.0:
                            shares = max(1, int(shares * governor_controls.size_multiplier))
                            logger.info(
                                f"   🧭 PerformanceGovernor: {governor_controls.size_multiplier:.0%} "
                                f"size ({perf_snapshot.tier.value}, {governor_regime_key}) → {shares} shares"
                            )
                        shares = self._apply_social_size_multiplier(
                            symbol=symbol,
                            shares=shares,
                            decision=social_decision,
                            price=price,
                        )

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

                        if governor_controls.size_multiplier < 1.0:
                            shares = max(1, int(shares * governor_controls.size_multiplier))
                            logger.info(
                                f"   🧭 PerformanceGovernor: {governor_controls.size_multiplier:.0%} "
                                f"size ({perf_snapshot.tier.value}, {governor_regime_key}) → {shares} shares"
                            )
                        shares = self._apply_social_size_multiplier(
                            symbol=symbol,
                            shares=shares,
                            decision=social_decision,
                            price=price,
                        )

                        if shares < 1:
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
                        if len(prices) > 1:
                            reference_price = float(prices.iloc[-2])
                    except Exception:
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
                        if self.pretrade_gateway.config.fail_closed:
                            async with self._position_lock:
                                if symbol in self.positions:
                                    del self.positions[symbol]
                            return

                    # ExecutionShield hard gates: spread gate + slippage budget.
                    if self.execution_shield:
                        quote = await self._get_connector_quote(entry_connector, symbol)
                        bid = float(quote.get("bid", 0.0) or 0.0)
                        ask = float(quote.get("ask", 0.0) or 0.0)
                        spread_limit_bps = self._spread_limit_bps_for_asset(asset_class)
                        slippage_budget_bps = self._slippage_budget_bps_for_asset(asset_class)
                        edge_buffer_bps = (
                            self._edge_buffer_bps_for_asset(asset_class)
                            if ApexConfig.EXECUTION_EDGE_GATE_ENABLED
                            else 0.0
                        )
                        gate_ok, gate_reason = self.execution_shield.can_enter_order(
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            max_spread_bps=spread_limit_bps,
                            slippage_budget_bps=slippage_budget_bps,
                            signal_strength=float(signal),
                            confidence=float(confidence),
                            min_edge_over_cost_bps=edge_buffer_bps,
                            signal_to_edge_bps=float(ApexConfig.EXECUTION_SIGNAL_TO_EDGE_BPS),
                        )
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
                    if self.use_institutional:
                        logger.debug(f"   Vol-adjusted: ${sizing.vol_adjusted_size:,.0f} | Corr penalty: {sizing.correlation_penalty:.2f}")

                    if entry_connector:
                        self.pending_orders.add(symbol)

                        # SOTA: Record arrival price
                        arrival_price = price
                        start_time = datetime.now()

                        # ✅ Phase 2.1: Use TWAP/VWAP for large orders (IBKR only)
                        order_value = shares * price
                        use_advanced = (
                            entry_connector is self.ibkr and
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
                            self.position_entry_prices[symbol] = price
                            self.position_entry_times[symbol] = entry_ts
                            self.position_peak_prices[symbol] = price
                            self.position_entry_signals[symbol] = signal  # Track entry signal for dynamic exits
                            self._save_position_metadata()

                            attribution_entry_price = float(fill_price if fill_price > 0 else price)
                            entry_slippage_bps = self._compute_slippage_bps(
                                arrival_price,
                                attribution_entry_price,
                            )
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
        if not self.failed_exits or (not self.ibkr and not self.alpaca):
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
                    try:
                        asset_class = parse_symbol(symbol).asset_class.value
                    except Exception:
                        asset_class = "EQUITY"
                    side_label = "LONG" if current_pos > 0 else "SHORT"
                    entry_price = float(self.position_entry_prices.get(symbol, self.price_cache.get(symbol, 0.0)) or 0.0)
                    if entry_price <= 0:
                        entry_price = float(self.price_cache.get(symbol, 0.0) or 0.0)
                    entry_time = self.position_entry_times.get(symbol, datetime.now())
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
                                hedge_retry_after.isoformat(),
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
                                    self._options_retry_after[hedge_retry_key].isoformat(),
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
                                cc_retry_after.isoformat(),
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
                                self._options_retry_after[cc_retry_key].isoformat(),
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
            entry_time = self.position_entry_times.get(symbol, datetime.now())

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
                if current_value <= 0 and self.capital > 0:
                    logger.warning(f"⚠️  Brokers returned ${current_value:,.2f}, using last known capital ${self.capital:,.2f}")
                    current_value = self.capital
            else:
                current_value = self.capital
                for symbol, qty in self.positions.items():
                    price = self.price_cache.get(symbol, 0)
                    if price and qty:
                        if qty > 0:  # Long
                            current_value += float(qty) * float(price)
                        else:  # Short (qty is negative)
                            current_value += float(qty) * float(price)

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
            elif (not recon_snapshot.block_entries) and previous_block:
                logger.warning(
                    "✅ Equity reconciliation block cleared: gap=$%.2f (%.2f%%), healthy_streak=%d",
                    recon_snapshot.gap_dollars,
                    recon_snapshot.gap_pct * 100,
                    recon_snapshot.healthy_streak,
                )

            # Keep baseline/risk state sane when broker APIs temporarily return invalid values.
            self.risk_manager.heal_baselines(current_capital=current_value, source="check_risk")
            self.capital = float(current_value)

            loss_check = self.risk_manager.check_daily_loss(current_value)
            dd_check = self.risk_manager.check_drawdown(current_value)
            self.performance_tracker.record_equity(current_value)
            perf_snapshot = self._performance_snapshot
            if self.performance_governor:
                perf_snapshot = self.performance_governor.update(current_value, datetime.now())
                self._performance_snapshot = perf_snapshot

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

            try:
                sortino = self.performance_tracker.get_sortino_ratio()
            except:
                sortino = 0.0

            total_trades = len(self.performance_tracker.trades)
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
                logger.info(
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
                if self.prometheus_metrics:
                    self.prometheus_metrics.update_kill_switch(active=True, reason="dd_sharpe_breach")

            if kill_state and kill_state.active and not kill_state.flatten_executed:
                logger.critical("🛑 Flattening all positions due to kill-switch")
                await self.close_all_positions()
                if self.kill_switch:
                    self.kill_switch.mark_flattened()

            if self.prometheus_metrics and kill_state:
                self.prometheus_metrics.update_kill_switch(active=kill_state.active)
            self._kill_switch_last_active = bool(kill_state.active) if kill_state else False

            # Refresh position prices before dashboard export
            await self.refresh_position_prices()

            # Export to dashboard
            await self.export_dashboard_state()
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
            all_symbols = list(set(list(self.positions.keys()) + ApexConfig.SYMBOLS))
            if self.ibkr:
                equity_symbols = [s for s in all_symbols if parse_symbol(s).asset_class != AssetClass.CRYPTO]
                pos_keys = list(self.positions.keys())
                equity_symbols.sort(key=lambda x: x in pos_keys, reverse=True)
                await self.ibkr.stream_quotes(equity_symbols)
            if self.alpaca:
                crypto_symbols = [s for s in all_symbols if parse_symbol(s).asset_class == AssetClass.CRYPTO]
                if crypto_symbols:
                    await self.alpaca.stream_quotes(crypto_symbols)
        
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
    
    def _save_position_metadata(self):
        """Persist position metadata for restart recovery."""
        try:
            metadata = {}
            for symbol in self.position_entry_prices:
                entry_time = self.position_entry_times.get(symbol, datetime.now())
                metadata[symbol] = {
                    'entry_price': self.position_entry_prices[symbol],
                    'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
                    'entry_signal': self.position_entry_signals.get(symbol, 0.0),
                    'peak_price': self.position_peak_prices.get(symbol, self.position_entry_prices[symbol])
                }
            metadata_file = Path("data") / "position_metadata.json"
            metadata_file.parent.mkdir(exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
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
            
            current_value = self.capital  # fallback
            if self.ibkr and hasattr(self.ibkr, 'ib'):
                try:
                    # Get NetLiquidation (total account equity) from IBKR
                    account_values = self.ibkr.ib.accountValues()
                    for av in account_values:
                        if av.tag == 'NetLiquidation' and av.currency == 'USD':
                            current_value = float(av.value)
                            break
                except:
                    pass
            if self.alpaca:
                try:
                    # Add Alpaca account equity (sync call via cached value)
                    alpaca_val = getattr(self.alpaca, '_last_equity', 0)
                    if alpaca_val > 0:
                        if self.ibkr:
                            current_value += alpaca_val  # Add to IBKR value
                        else:
                            current_value = alpaca_val  # Alpaca-only
                except:
                    pass
            
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
                expiry_norm = str(expiry or "").strip()
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
            
            state = {
                'timestamp': datetime.now().isoformat(),
                'capital': float(current_value),
                'initial_capital': float(ApexConfig.INITIAL_CAPITAL),
                'starting_capital': float(self.risk_manager.starting_capital),
                'positions': {},
                'signals': current_signals,
                'daily_pnl': float(current_value - self.risk_manager.day_start_capital) if self.risk_manager.day_start_capital > 0 else 0.0,
                'total_pnl': float(current_value - self.risk_manager.starting_capital),
                'total_commissions': float(self.total_commissions),
                'max_drawdown': float(self.performance_tracker.get_max_drawdown()) if math.isfinite(self.performance_tracker.get_max_drawdown()) else 0.0,
                'sharpe_ratio': float(self.performance_tracker.get_sharpe_ratio()) if math.isfinite(self.performance_tracker.get_sharpe_ratio()) else 0.0,
                'win_rate': float(self.performance_tracker.get_win_rate()) if math.isfinite(self.performance_tracker.get_win_rate()) else 0.0,
                'total_trades': len(self.performance_tracker.trades),
                'open_positions': self.position_count,
                'option_positions': int(option_positions_count),
                'open_positions_total': int(self.position_count + option_positions_count),
                'option_positions_detail': option_positions_detail,
                'sector_exposure': self.calculate_sector_exposure(),
                'performance_governor': self._performance_snapshot.to_dict() if self._performance_snapshot else None,
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
            }
            # Fetch active broker connections once to map sources
            active_sources = {}
            try:
                from services.common.db import db_session
                from services.trading.models import BrokerConnectionModel
                from sqlalchemy import select
                
                async with db_session() as session:
                    stmt = select(BrokerConnectionModel).where(BrokerConnectionModel.is_active == True)
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
                    
                    entry_time = self.position_entry_times.get(symbol, datetime.now())
                    
                    if qty > 0:  # Long
                        pnl = (price - avg_price) * qty
                        pnl_pct = (price / avg_price - 1) * 100 if avg_price > 0 else 0
                    else:  # Short
                        pnl = (avg_price - price) * abs(qty)
                        pnl_pct = (avg_price / price - 1) * 100 if price > 0 else 0
                    
                    # Determine source
                    connector = self._get_connector_for(symbol)
                    source_id = ""
                    if connector == self.alpaca:
                        source_id = active_sources.get("alpaca", "alpaca")
                    elif connector == self.ibkr:
                        source_id = active_sources.get("ibkr", "ibkr")

                    state['positions'][symbol] = {
                        'qty': int(qty),
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'avg_price': float(avg_price),
                        'current_price': float(price),
                        'pnl': float(pnl),
                        'pnl_pct': float(pnl_pct),
                        'entry_time': entry_time.isoformat() if isinstance(entry_time, datetime) else entry_time,
                        'current_signal': current_signals.get(symbol, {}).get('signal', 0),
                        'signal_direction': current_signals.get(symbol, {}).get('direction', 'UNKNOWN'),
                        'source_id': source_id
                    }
                except Exception as e:
                    logger.debug(f"Error adding position {symbol}: {e}")
            
            # Add signal quality metrics if available
            try:
                quality_metrics = self.signal_outcome_tracker.get_quality_metrics()
                state['signal_quality'] = {
                    'total_signals_tracked': quality_metrics.total_signals,
                    'completed_signals': quality_metrics.completed_signals,
                    'avg_forward_return_5d': quality_metrics.avg_forward_returns.get(5, 0.0),
                    'avg_forward_return_10d': quality_metrics.avg_forward_returns.get(10, 0.0),
                    'hit_5pct_5d': quality_metrics.target_accuracy.get('hit_5pct_5d', 0.0),
                    'hit_10pct_10d': quality_metrics.target_accuracy.get('hit_10pct_10d', 0.0),
                    'avg_mfe_5d': quality_metrics.avg_mfe.get(5, 0.0),
                    'avg_mae_5d': quality_metrics.avg_mae.get(5, 0.0),
                    'by_regime': quality_metrics.by_regime,
                    'by_confidence': quality_metrics.by_confidence,
                }
            except Exception as e:
                logger.debug(f"Signal quality metrics error: {e}")
                state['signal_quality'] = {}

            state_file = data_dir / "trading_state.json"
            tmp_file = state_file.with_suffix(".json.tmp")
            with open(tmp_file, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, state_file)

            # Persist position metadata for restart recovery
            self._save_position_metadata()
            
            # Save risk state and price cache
            self.risk_manager.save_state()
            self.save_price_cache()

            logger.debug(f"📊 Dashboard state exported")
        
        except Exception as e:
            logger.error(f"❌ Dashboard export error: {e}")

    def save_price_cache(self):
        """Save price cache to disk."""
        try:
            if not self.price_cache:
                return
            ApexConfig.DATA_DIR.mkdir(exist_ok=True)
            with open(ApexConfig.DATA_DIR / "price_cache.json", "w") as f:
                json.dump(self.price_cache, f)
        except Exception as e:
            logger.debug(f"Error saving price cache: {e}")

    def load_price_cache(self):
        """Load last known prices from disk."""
        try:
            cache_file = ApexConfig.DATA_DIR / "price_cache.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    self.price_cache.update(cached)
                logger.info(f"💾 Restored {len(cached)} prices from cache")
        except Exception as e:
            logger.debug(f"Error loading price cache: {e}")

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
                'total_pnl': 0.0,
                'total_commissions': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'open_positions': self.position_count,
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
            self._startup_time = datetime.now()
            
            while self.is_running:
                try:
                    cycle += 1
                    self._cycle_count = cycle
                    now = datetime.now()

                    # Process operator commands (e.g., kill-switch reset) each cycle.
                    await self._process_external_control_commands()

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
                                with open(ApexConfig.DATA_DIR / 'heartbeat.json', 'w') as f:
                                    json.dump({'status': 'DEAD', 'timestamp': now.isoformat()}, f)
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
                        merged = {}
                        if self.ibkr:
                            merged.update(await self.ibkr.get_all_positions())
                        if self.alpaca:
                            # Only take crypto positions from Alpaca to avoid
                            # overwriting IBKR equity positions with stale Alpaca data
                            for sym, qty in (await self.alpaca.get_all_positions()).items():
                                try:
                                    parsed = parse_symbol(sym)
                                    if parsed.asset_class == AssetClass.CRYPTO:
                                        merged[sym] = qty
                                except ValueError:
                                    merged[sym] = qty
                        self._cached_ibkr_positions = merged
                        self.positions = merged.copy()

                    # Refresh pending orders
                    if self.ibkr or self.alpaca:
                        await self.refresh_pending_orders()

                    # Refresh data hourly
                    if (now - last_data_refresh).total_seconds() > 3600:
                        await self.refresh_data()
                        last_data_refresh = now

                    # Filter to open markets only (weekends/closures)
                    open_universe = [s for s in ApexConfig.SYMBOLS if is_market_open(s, datetime.utcnow())]
                    open_positions = [s for s, q in self.positions.items() if q != 0 and is_market_open(s, datetime.utcnow())]
                    if not open_universe and not open_positions:
                        logger.info("⏸️  No open markets right now; skipping cycle work")
                        await asyncio.sleep(ApexConfig.CHECK_INTERVAL_SECONDS)
                        continue

                    in_equity_hours = ApexConfig.TRADING_HOURS_START <= est_hour <= ApexConfig.TRADING_HOURS_END
                    if not in_equity_hours:
                        logger.info("⏰ Outside equity hours; processing only open markets")

                    # Always process if any market is open
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
                        peak_capital=max(float(self.risk_manager.peak_capital or 0.0), float(self.capital or 0.0)),
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
                        await self.process_symbols_parallel(open_universe)

                        # 4. Refresh data (market data, indicators)
                        if cycle % 10 == 0:
                            logger.info("👉 Step 4: Refresh data")
                            await self.refresh_data()

                        # Check for rebalancing (near market close)
                        if open_universe:
                            await self.check_and_execute_rebalance(est_hour)

                        # Manage options (hedging, covered calls, expiring positions)
                        if ApexConfig.OPTIONS_ENABLED and any(
                            parse_symbol(s).asset_class == AssetClass.EQUITY for s in open_universe
                        ):
                            logger.info("👉 Step 5: Manage options")
                            await self.manage_options()

                            # Phase 3: Manage Active Positions (Ratchet, Aging)
                            logger.info("👉 Step 6: Manage active positions")
                            await self.manage_active_positions()

                        # Sync positions after processing (captures any trades)
                        if self.ibkr or self.alpaca:
                            await self._sync_positions()

                        logger.info("👉 Step 7: Check risk")
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
                                        import numpy as np
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
                                    if corr_state.regime >= CorrelationRegime.HERDING:
                                        logger.warning(f"🛡️ Correlation {corr_state.regime.name}: "
                                                     f"avg={corr_state.avg_correlation:.2f}, "
                                                     f"effective_N={corr_state.effective_positions:.1f}")
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


async def main():
    """Main entry point."""
    system = ApexTradingSystem()
    await system.run()
