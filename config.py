"""
config.py - APEX Trading System Configuration

Central configuration hub for all system parameters including:
- IBKR connection settings
- Capital and position sizing
- Risk limits and circuit breakers
- Trading hours and timing
- Universe selection

Environment variables can override defaults:
- APEX_IBKR_HOST, APEX_IBKR_PORT, APEX_IBKR_CLIENT_ID
- APEX_LIVE_TRADING (true/false)
- APEX_INITIAL_CAPITAL
"""

import os
import logging
from pathlib import Path
from typing import Dict, List

# Load .env file automatically (allows overriding any setting without shell exports)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

_config_logger = logging.getLogger(__name__)


def _resolve_environment_name() -> str:
    return (
        os.getenv("APEX_ENVIRONMENT")
        or os.getenv("APEX_ENV")
        or os.getenv("ENV")
        or "prod"
    ).strip().lower()


def _is_development_environment(environment: str) -> bool:
    return environment in {"dev", "development", "local"}


_RUNTIME_ENVIRONMENT = _resolve_environment_name()


def _env_default(name: str, production: object, development: object) -> str:
    default = development if _is_development_environment(_RUNTIME_ENVIRONMENT) else production
    return os.getenv(name, str(default))


class ApexConfig:
    # --- EXECUTION & SMART ORDER ROUTING ---
    SOR_ENABLED = True               # Use Limit mid-pegging instead of Market orders
    SOR_MAX_URGENCY_STEPS = 3        # Number of times to adjust price toward the ask/bid
    SOR_STEP_DELAY_SECONDS = 10      # How long to wait at each price level before adjusting

    # Passive Limit Execution (Pillar 1B — spread capture during market hours)
    PASSIVE_LIMIT_ENABLED = True     # Post at mid-price first; step toward touch if unfilled
    PASSIVE_LIMIT_STEP_SECONDS = 5   # Seconds to wait at each urgency step (Faster capture)
    PASSIVE_LIMIT_MAX_STEPS = 5      # Steps before falling back to market sweep (Wider sweep)
    PASSIVE_LIMIT_MIN_SPREAD_BPS = 2 # Skip passive limit if estimated spread < 2bps

    # --- PHASE B: DYNAMIC LIMITS ---
    CORRELATION_DYNAMIC_ENABLED = True
    DRAWDOWN_DYNAMIC_TIERS_ENABLED = True

    """
    Central configuration for APEX Trading System.

    All settings can be overridden via environment variables prefixed with APEX_.
    Example: APEX_LIVE_TRADING=false will disable live trading.
    """

    # System Info
    SYSTEM_NAME: str = "APEX Trading System"
    VERSION: str = "2.0.0-PRODUCTION"
    ENVIRONMENT: str = _RUNTIME_ENVIRONMENT
    IS_DEVELOPMENT: bool = _is_development_environment(ENVIRONMENT)
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING MODE
    # ═══════════════════════════════════════════════════════════════
    LIVE_TRADING: bool = os.getenv(
        "APEX_LIVE_TRADING",
        os.getenv("LIVE_TRADING", "false"),
    ).lower() == "true"
    LIVE_TRADING_CONFIRMED: bool = os.getenv(
        "APEX_LIVE_TRADING_CONFIRMED",
        os.getenv("LIVE_TRADING_CONFIRMED", "false"),
    ).lower() == "true"

    # ═══════════════════════════════════════════════════════════════
    # IBKR CONNECTION
    # ═══════════════════════════════════════════════════════════════
    IBKR_HOST: str = os.getenv("APEX_IBKR_HOST", os.getenv("IBKR_HOST", "127.0.0.1"))
    IBKR_PORT: int = int(
        os.getenv("APEX_IBKR_PORT", os.getenv("IBKR_PORT", "7497")).split("#")[0].strip()
    )  # 7497 = Paper, 7496 = Live
    import random
    IBKR_CLIENT_ID: int = int(
        os.getenv(
            "APEX_IBKR_CLIENT_ID",
            os.getenv("IBKR_CLIENT_ID", str(random.randint(100, 999))),
        )
    )
    IBKR_FX_EXCHANGE: str = os.getenv("APEX_IBKR_FX_EXCHANGE", "IDEALPRO")
    IBKR_CRYPTO_EXCHANGE: str = os.getenv("APEX_IBKR_CRYPTO_EXCHANGE", "PAXOS")
    IBKR_CONNECT_TIMEOUT: int = int(os.getenv("APEX_IBKR_CONNECT_TIMEOUT", "10"))

    # ═══════════════════════════════════════════════════════════════
    # FX/CRYPTO PAPER TRADING TUNING (OPTIMIZE FOR OBSERVABILITY)
    # ═══════════════════════════════════════════════════════════════
    # Set True for strict live-style checks. Keep False for paper.
    STRICT_IBKR_LIVE_RULES: bool = os.getenv("APEX_STRICT_IBKR_LIVE_RULES", "false").lower() == "true"

    # Optional mapping to make IBKR paper trading compatible with backtest pairs
    # (keys/values are slash pairs; prefixes are allowed but not required)
    IBKR_USE_PAIR_MAP: bool = os.getenv("APEX_IBKR_USE_PAIR_MAP", "true").lower() == "true"
    IBKR_PAIR_MAP = {
        "BTC/USDT": "BTC/USD",
        "ETH/USDC": "ETH/USD",
    }
    # Optional data provider mapping (e.g., yfinance lacks USDT/USDC pairs)
    DATA_PAIR_MAP = {
        "BTC/USDT": "BTC/USD",
        "ETH/USDT": "ETH/USD",
        "BTC/USDC": "BTC/USD",
        "ETH/USDC": "ETH/USD",
        "SOL/USDT": "SOL/USD",
        "DOGE/USDT": "DOGE/USD",
    }
    # ✅ NEW: If a crypto pair uses USDT/USDC and no explicit mapping exists,
    # map it to USD for data providers that lack stablecoin quotes.
    DATA_MAP_STABLECOINS_TO_USD: bool = os.getenv(
        "APEX_DATA_MAP_STABLECOINS_TO_USD", "true"
    ).lower() == "true"

    # Typical minimums (paper: warn only; live: enforce if STRICT_IBKR_LIVE_RULES=True)
    IBKR_MIN_FX_NOTIONAL = float(os.getenv("APEX_IBKR_MIN_FX_NOTIONAL", "1000"))
    IBKR_MIN_CRYPTO_NOTIONAL = float(os.getenv("APEX_IBKR_MIN_CRYPTO_NOTIONAL", "10"))

    # Fee model knobs for backtests and estimates (tune for sensitivity analysis)
    FX_SPREAD_BPS = float(os.getenv("APEX_FX_SPREAD_BPS", "1.5"))  # typical tight spread
    FX_COMMISSION_BPS = float(os.getenv("APEX_FX_COMMISSION_BPS", "0.2"))
    CRYPTO_SPREAD_BPS = float(os.getenv("APEX_CRYPTO_SPREAD_BPS", "3.0"))
    CRYPTO_COMMISSION_BPS = float(os.getenv("APEX_CRYPTO_COMMISSION_BPS", "8.0"))  # 0.08%

    # IBKR pacing (requests per second, for paper-safe throttling)
    IBKR_MAX_REQ_PER_SEC = float(os.getenv("APEX_IBKR_MAX_REQ_PER_SEC", "6"))
    # Max concurrent streaming subscriptions (increased for expanded universe)
    IBKR_MAX_STREAMS = int(os.getenv("APEX_IBKR_MAX_STREAMS", "100"))

    # Market hours overrides (set for stress tests)
    MARKET_ALWAYS_OPEN: bool = os.getenv("APEX_MARKET_ALWAYS_OPEN", "false").lower() == "true"
    FX_ALWAYS_OPEN: bool = os.getenv("APEX_FX_ALWAYS_OPEN", "false").lower() == "true"
    CRYPTO_ALWAYS_OPEN: bool = os.getenv("APEX_CRYPTO_ALWAYS_OPEN", "true").lower() == "true"
    # Separate daily-loss limits per asset class (crypto uses a rolling 24h window)
    CRYPTO_MAX_DAILY_LOSS: float = float(os.getenv("APEX_CRYPTO_MAX_DAILY_LOSS", "0.05"))   # 5%
    EQUITY_MAX_DAILY_LOSS: float = float(os.getenv("APEX_EQUITY_MAX_DAILY_LOSS", "0.03"))   # 3%
    CUSTOM_MARKET_SESSIONS = {
        # FOREX disabled intentionally until IDEALPRO minimum lots are structurally modeled
        # "FOREX": {"timezone": "America/New_York", "open": "17:00", "close": "17:00", "weekdays": [0,1,2,3,4,6]},
        # "EQUITY": {"timezone": "America/New_York", "open": "09:30", "close": "16:00", "weekdays": [0,1,2,3,4]},
    }
    USE_DATA_FALLBACK_FOR_PRICES: bool = os.getenv("APEX_USE_DATA_FALLBACK_FOR_PRICES", "true").lower() == "true"
    # ✅ NEW (Paper-safe): Allow offline IBKR mode when connection fails
    IBKR_ALLOW_OFFLINE: bool = os.getenv("APEX_IBKR_ALLOW_OFFLINE", "true").lower() == "true"
    # ✅ NEW (Observability): Prefer data-provider fallback when market is closed (weekends)
    PRICE_FALLBACK_WHEN_MARKET_CLOSED: bool = os.getenv("APEX_PRICE_FALLBACK_WHEN_MARKET_CLOSED", "true").lower() == "true"
    # ✅ NEW (Paper-safe): Toggle Data Watchdog (disable when running offline)
    DATA_WATCHDOG_ENABLED: bool = os.getenv("APEX_DATA_WATCHDOG_ENABLED", "true").lower() == "true"
    # Offline-session escape hatch: synthesize historical bars from cached prices
    # when external data providers are unavailable.
    MARKET_DATA_ALLOW_SYNTHETIC_HISTORY: bool = (
        os.getenv("APEX_MARKET_DATA_ALLOW_SYNTHETIC_HISTORY", "false").lower() == "true"
    )
    MARKET_DATA_SYNTHETIC_BASE_PRICE: float = float(
        os.getenv("APEX_MARKET_DATA_SYNTHETIC_BASE_PRICE", "100.0")
    )

    # ═══════════════════════════════════════════════════════════════
    # ALPACA CONNECTION (Crypto Paper Trading)
    # ═══════════════════════════════════════════════════════════════
    ALPACA_API_KEY: str = os.getenv("APEX_ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("APEX_ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "APEX_ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    ALPACA_ALLOW_OFFLINE: bool = (
        os.getenv("APEX_ALPACA_ALLOW_OFFLINE", "false").lower() == "true"
    )
    ALPACA_HTTP_TIMEOUT_SECONDS: float = float(os.getenv("APEX_ALPACA_HTTP_TIMEOUT_SECONDS", "20.0"))
    ALPACA_FILL_WAIT_SECONDS: int = int(os.getenv("APEX_ALPACA_FILL_WAIT_SECONDS", "10"))
    # Exit limit order: try limit at -30 bps first, fall back to market on miss
    # 30 bps = P75 of actual exit slippage — captures ~75% of exits without missed fills
    ALPACA_EXIT_USE_LIMIT: bool = (
        os.getenv("APEX_ALPACA_EXIT_USE_LIMIT", "true").lower() != "false"
    )
    ALPACA_EXIT_LIMIT_OFFSET_BPS: float = float(
        os.getenv("APEX_ALPACA_EXIT_LIMIT_OFFSET_BPS", "30")        # P75 of live exit slippage
    )
    ALPACA_EXIT_LIMIT_WAIT_SECONDS: int = int(
        os.getenv("APEX_ALPACA_EXIT_LIMIT_WAIT_SECONDS", "8")       # Try limit for 8s then market
    )

    # ── Adaptive circuit breaker early reset ────────────────────────
    CIRCUIT_BREAKER_EARLY_RESET_HOURS: float = float(os.getenv("APEX_CB_EARLY_RESET_HOURS", "2.0"))
    CIRCUIT_BREAKER_EARLY_RESET_MAX_LOSS_USD: float = float(os.getenv("APEX_CB_EARLY_RESET_MAX_LOSS_USD", "50.0"))
    CIRCUIT_BREAKER_EARLY_RESET_MAX_DAILY_LOSS_PCT: float = float(os.getenv("APEX_CB_EARLY_RESET_MAX_DAILY_LOSS_PCT", "0.015"))
    # ── Intraday rolling drawdown gate ───────────────────────────────────────
    # Blocks NEW entries (never exits) if the portfolio drops more than
    # INTRADAY_DD_MAX_LOSS_PCT within the rolling INTRADAY_DD_WINDOW_MINUTES window.
    # Flip ENABLED=false via env-var for instant kill-switch / rollback.
    INTRADAY_DD_GATE_ENABLED: bool = (
        os.getenv("APEX_INTRADAY_DD_GATE_ENABLED", "true").lower() == "true"
    )
    INTRADAY_DD_WINDOW_MINUTES: int = int(os.getenv("APEX_INTRADAY_DD_WINDOW_MINUTES", "60"))
    INTRADAY_DD_MAX_LOSS_PCT: float = float(os.getenv("APEX_INTRADAY_DD_MAX_LOSS_PCT", "0.015"))  # 1.5%
    # Disabled: dynamic discovery circumvents the static whitelist, pulling in illiquid
    # pairs (RENDER, ONDO, PAXG, etc.). Static CRYPTO_PAIRS already defines our universe.
    ALPACA_DISCOVER_CRYPTO_SYMBOLS: bool = (
        os.getenv("APEX_ALPACA_DISCOVER_CRYPTO_SYMBOLS", "false").lower() == "true"
    )
    ALPACA_DISCOVER_CRYPTO_LIMIT: int = int(
        os.getenv("APEX_ALPACA_DISCOVER_CRYPTO_LIMIT", "24")
    )
    ALPACA_DISCOVER_CRYPTO_PREFERRED_QUOTES: List[str] = [
        token.strip().upper()
        for token in os.getenv(
            "APEX_ALPACA_DISCOVER_CRYPTO_PREFERRED_QUOTES", "USD,USDT,USDC"
        ).split(",")
        if token.strip()
    ]
    # Symbols known to have no yfinance historical data — excluded from discovery universe
    ALPACA_DISCOVER_CRYPTO_EXCLUDED: List[str] = [
        s.strip().upper()
        for s in os.getenv(
            "APEX_ALPACA_DISCOVER_CRYPTO_EXCLUDED",
            # CRV and FIL excluded: chronic 15-30bps slippage observed in live fills.
            # BCH, XLM, ETC, AAVE, DOT, LTC also excluded to match static CRYPTO_PAIRS.
            # RENDER, ONDO, PAXG: observed in Alpaca discovery despite not being whitelisted.
            "CRYPTO:GRT/USD,CRYPTO:PEPE/USD,CRYPTO:POL/USD,GRT/USD,PEPE/USD,POL/USD,"
            "CRYPTO:CRV/USD,CRV/USD,CRYPTO:FIL/USD,FIL/USD,"
            "CRYPTO:BCH/USD,BCH/USD,CRYPTO:AVAX/USD,AVAX/USD,"
            "CRYPTO:DOGE/USD,DOGE/USD,CRYPTO:AAVE/USD,AAVE/USD,"
            "CRYPTO:DOT/USD,DOT/USD,CRYPTO:LTC/USD,LTC/USD,"
            "CRYPTO:RENDER/USD,RENDER/USD,CRYPTO:ONDO/USD,ONDO/USD,"
            "CRYPTO:PAXG/USD,PAXG/USD,"
            "CRYPTO:TRX/USD,TRX/USD,CRYPTO:ALGO/USD,ALGO/USD,"
            "CRYPTO:INJ/USD,INJ/USD,CRYPTO:OP/USD,OP/USD,"
            "CRYPTO:ARB/USD,ARB/USD,CRYPTO:APT/USD,APT/USD,"
            "CRYPTO:SUI/USD,SUI/USD"
        ).split(",")
        if s.strip()
    ]
    # Momentum scan: re-rank Alpaca crypto universe at NYSE open and periodically.
    # IBKR equity universe is never touched — only _dynamic_crypto_symbols is updated.
    CRYPTO_MOMENTUM_SCAN_ENABLED: bool = (
        os.getenv("APEX_CRYPTO_MOMENTUM_SCAN_ENABLED", "true").lower() == "true"
    )
    CRYPTO_MOMENTUM_TOP_N: int = int(os.getenv("APEX_CRYPTO_MOMENTUM_TOP_N", "12"))
    CRYPTO_MOMENTUM_MIN_VOLUME_USD: float = float(
        os.getenv("APEX_CRYPTO_MOMENTUM_MIN_VOLUME_USD", "500000")
    )
    CRYPTO_MOMENTUM_RESCAN_HOURS: float = float(
        os.getenv("APEX_CRYPTO_MOMENTUM_RESCAN_HOURS", "4.0")
    )

    # ═══════════════════════════════════════════════════════════════
    # AUTHENTICATION
    # ═══════════════════════════════════════════════════════════════
    AUTH_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("APEX_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
    )  # Default: 24 hours

    # Broker selection: "ibkr" | "alpaca" | "both"
    #   ibkr    = IBKR only
    #   alpaca  = Alpaca only (crypto paper trading)
    #   both    = IBKR for equities/forex, Alpaca for crypto (default)
    BROKER_MODE: str = os.getenv(
        "APEX_BROKER_MODE",
        os.getenv("BROKER_MODE", os.getenv("APEX_484BACKTEST_ONLY484", "both")),
    )
    # Preferred execution broker for UI/runtime role attribution.
    # In mixed mode this determines which broker is shown as "trading" vs "idle".
    PRIMARY_EXECUTION_BROKER: str = os.getenv(
        "APEX_PRIMARY_EXECUTION_BROKER",
        "alpaca",
    ).lower()
    BROKER_EQUITY_CACHE_TTL_SECONDS: int = int(
        os.getenv("APEX_BROKER_EQUITY_CACHE_TTL_SECONDS", "900")
    )
    BROKER_EQUITY_QUORUM_MIN_BROKERS: int = int(
        os.getenv("APEX_BROKER_EQUITY_QUORUM_MIN_BROKERS", "1")
    )
    BROKER_EQUITY_REFRESH_INTERVAL_SECONDS: int = int(
        os.getenv("APEX_BROKER_EQUITY_REFRESH_INTERVAL_SECONDS", "120")
    )
    BROKER_POSITIONS_REFRESH_INTERVAL_SECONDS: int = int(
        os.getenv("APEX_BROKER_POSITIONS_REFRESH_INTERVAL_SECONDS", "120")
    )
    # Paper-session startup hygiene: stale persisted state can block trading after restarts.
    PAPER_STARTUP_RISK_SELF_HEAL_ENABLED: bool = os.getenv(
        "APEX_PAPER_STARTUP_RISK_SELF_HEAL_ENABLED", "true"
    ).lower() == "true"
    PAPER_STARTUP_RISK_MISMATCH_RATIO: float = float(
        os.getenv("APEX_PAPER_STARTUP_RISK_MISMATCH_RATIO", "0.30")
    )
    PAPER_STARTUP_RESET_CIRCUIT_BREAKER: bool = os.getenv(
        "APEX_PAPER_STARTUP_RESET_CIRCUIT_BREAKER", "true"
    ).lower() == "true"
    PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED: bool = os.getenv(
        "APEX_PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", "true"
    ).lower() == "true"
    PAPER_STARTUP_PERFORMANCE_REBASE_RATIO: float = float(
        os.getenv("APEX_PAPER_STARTUP_PERFORMANCE_REBASE_RATIO", "0.30")
    )
    MODEL_MANIFEST_VERIFICATION_ENABLED: bool = os.getenv(
        "APEX_MODEL_MANIFEST_VERIFICATION_ENABLED", "true"
    ).lower() == "true"
    MODEL_MANIFEST_PATH: str = os.getenv(
        "APEX_MODEL_MANIFEST_PATH", "models/model_manifest.json"
    )
    MODEL_MANIFEST_FAIL_CLOSED: bool = os.getenv(
        "APEX_MODEL_MANIFEST_FAIL_CLOSED", "true"
    ).lower() == "true"
    MODEL_MANIFEST_MAX_AGE_DAYS: int = int(
        os.getenv("APEX_MODEL_MANIFEST_MAX_AGE_DAYS", "0")
    )
    STARTUP_TRUST_AUDIT_ENABLED: bool = os.getenv(
        "APEX_STARTUP_TRUST_AUDIT_ENABLED", "true"
    ).lower() == "true"
    SHADOW_DEPLOYMENT_ENABLED: bool = os.getenv(
        "APEX_SHADOW_DEPLOYMENT_ENABLED", "true"
    ).lower() == "true"
    SHADOW_DEPLOYMENT_EVALUATION_INTERVAL_CYCLES: int = int(
        os.getenv("APEX_SHADOW_DEPLOYMENT_EVALUATION_INTERVAL_CYCLES", "20")
    )
    SHADOW_DEPLOYMENT_MIN_DAYS: float = float(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MIN_DAYS", "1.0")
    )
    SHADOW_DEPLOYMENT_MIN_SIGNALS: int = int(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MIN_SIGNALS", "25")
    )
    SHADOW_DEPLOYMENT_MIN_DECISION_AGREEMENT_RATE: float = float(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MIN_DECISION_AGREEMENT_RATE", "0.60")
    )
    SHADOW_DEPLOYMENT_MIN_OFFLINE_SHARPE_DELTA: float = float(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MIN_OFFLINE_SHARPE_DELTA", "0.10")
    )
    SHADOW_DEPLOYMENT_MAX_DRAWDOWN_INCREASE: float = float(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MAX_DRAWDOWN_INCREASE", "0.02")
    )
    SHADOW_DEPLOYMENT_MAX_EXCESS_BLOCK_RATE: float = float(
        os.getenv("APEX_SHADOW_DEPLOYMENT_MAX_EXCESS_BLOCK_RATE", "0.35")
    )
    SHADOW_DEPLOYMENT_AUTO_PROMOTE_NON_PROD: bool = os.getenv(
        "APEX_SHADOW_DEPLOYMENT_AUTO_PROMOTE_NON_PROD", "true"
    ).lower() == "true"
    UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES: bool = os.getenv(
        "APEX_UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES", "true"
    ).lower() == "true"
    UNIFIED_LATCH_RESET_REBASE_PERFORMANCE: bool = os.getenv(
        "APEX_UNIFIED_LATCH_RESET_REBASE_PERFORMANCE", "true"
    ).lower() == "true"

    # ═══════════════════════════════════════════════════════════════
    # CAPITAL & POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    INITIAL_CAPITAL: int = int(
        os.getenv("APEX_INITIAL_CAPITAL", os.getenv("INITIAL_CAPITAL", "1100000"))
    )
    POSITION_SIZE_USD = float(os.getenv("APEX_POSITION_SIZE_USD", "20000"))  # Default $20k
    CRYPTO_POSITION_SIZE_USD: float = float(
        os.getenv("APEX_CRYPTO_POSITION_SIZE_USD", "10000")
    )  # Notional cap per crypto position — $10K (~13% of $79K account, 4 pairs max)
    MAX_POSITIONS = 20  # Reduced from 40: prevents capital dilution across too many tiny positions
    MAX_SHARES_PER_POSITION = 500  # Cap max shares per position
    MIN_HEDGE_NOTIONAL = 50_000  # Only hedge positions larger than $50k to save on costs
    
    # ═══════════════════════════════════════════════════════════════
    # RISK LIMITS
    # ═══════════════════════════════════════════════════════════════
    MAX_DAILY_LOSS = 0.03  # 3% max daily loss (Moderate risk profile)
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    MAX_SECTOR_EXPOSURE = 1.0  # Relaxed to 100% for initial trades as requested
    SECTOR_CONCENTRATION_ENABLED: bool = os.getenv("APEX_SECTOR_CONCENTRATION_ENABLED", "true").lower() == "true"
    SECTOR_CONCENTRATION_MAX_PCT: float = float(os.getenv("APEX_SECTOR_CONCENTRATION_MAX_PCT", "0.25"))

    # Phase 14: Delta / Market-Neutral Hedger
    DELTA_HEDGE_ENABLED: bool = os.getenv("APEX_DELTA_HEDGE_ENABLED", "true").lower() == "true"
    MIN_HEDGE_DOLLAR_IMBALANCE: float = float(os.getenv("APEX_MIN_HEDGE_DOLLAR_IMBALANCE", "5000"))
    DELTA_HEDGE_REBALANCE_CYCLES: int = int(os.getenv("APEX_DELTA_HEDGE_REBALANCE_CYCLES", "20"))
    DELTA_HEDGE_MAX_SPY_NOTIONAL: float = float(os.getenv("APEX_DELTA_HEDGE_MAX_SPY_NOTIONAL", "200000"))
    DELTA_HEDGE_SYMBOL: str = os.getenv("APEX_DELTA_HEDGE_SYMBOL", "SPY")

    # Options Flow Smart Money Gate
    OPTIONS_FLOW_GATE_ENABLED: bool = os.getenv("APEX_OPTIONS_FLOW_GATE_ENABLED", "true").lower() == "true"
    OPTIONS_FLOW_CONTRA_THRESHOLD: float = float(os.getenv("APEX_OPTIONS_FLOW_CONTRA_THRESHOLD", "0.40"))
    OPTIONS_FLOW_BLOCK_MIN_CONF: float = float(os.getenv("APEX_OPTIONS_FLOW_BLOCK_MIN_CONF", "0.75"))
    OPTIONS_FLOW_CONF_PENALTY: float = float(os.getenv("APEX_OPTIONS_FLOW_CONF_PENALTY", "0.93"))

    # Capital Allocator
    ALLOC_LOOKBACK_DAYS: int = int(os.getenv("APEX_ALLOC_LOOKBACK_DAYS", "20"))

    # Multi-Timeframe Signal Fusion
    MTF_FUSION_ENABLED: bool = os.getenv("APEX_MTF_FUSION_ENABLED", "true").lower() == "true"

    # Earnings Event Gate
    EARNINGS_GATE_ENABLED: bool = os.getenv("APEX_EARNINGS_GATE_ENABLED", "true").lower() == "true"

    # HRP Correlation Sizer
    HRP_SIZING_ENABLED: bool = os.getenv("APEX_HRP_SIZING_ENABLED", "true").lower() == "true"

    # Adaptive ATR stop manager — regime + VIX + profit-ratchet stop updates
    ADAPTIVE_ATR_STOPS_ENABLED: bool = os.getenv("APEX_ADAPTIVE_ATR_STOPS_ENABLED", "true").lower() == "true"
    ADAPTIVE_ATR_UPDATE_INTERVAL: int = int(os.getenv("APEX_ADAPTIVE_ATR_UPDATE_INTERVAL", "5"))

    # Model drift monitor — IC/hit-rate/confidence decay → auto-retrain trigger
    MODEL_DRIFT_MONITOR_ENABLED: bool = os.getenv("APEX_MODEL_DRIFT_MONITOR_ENABLED", "true").lower() == "true"
    MODEL_DRIFT_WINDOW_SIZE: int = int(os.getenv("APEX_MODEL_DRIFT_WINDOW_SIZE", "30"))
    MODEL_DRIFT_IC_RETRAIN_THRESHOLD: float = float(os.getenv("APEX_MODEL_DRIFT_IC_RETRAIN_THRESHOLD", "0.01"))
    MODEL_DRIFT_RETRAIN_COOLDOWN_HOURS: float = float(os.getenv("APEX_MODEL_DRIFT_RETRAIN_COOLDOWN_HOURS", "24"))

    # Alpha decay calibrator — signal IC by hold-time horizon
    ALPHA_DECAY_CALIBRATOR_ENABLED: bool = os.getenv("APEX_ALPHA_DECAY_ENABLED", "true").lower() == "true"

    # IC Tracker — rolling Spearman IC per signal component
    IC_TRACKER_ENABLED: bool = os.getenv("APEX_IC_TRACKER_ENABLED", "true").lower() == "true"
    # Dampen composite signal when IC is below dead threshold (0.50-1.0 scale; 0.80 = 20% reduction)
    IC_DEAD_SIGNAL_DAMPENER: float = float(os.getenv("APEX_IC_DEAD_SIGNAL_DAMPENER", "0.80"))
    # Boost confidence when IC is above strong threshold (1.0-1.15 range)
    IC_STRONG_CONFIDENCE_BOOST: float = float(os.getenv("APEX_IC_STRONG_CONF_BOOST", "1.08"))
    # Minimum observations before IC dampening is active (prevent premature dampening)
    IC_MIN_OBS_TO_DAMPEN: int = int(os.getenv("APEX_IC_MIN_OBS_TO_DAMPEN", "20"))

    # Optuna Bayesian Parameter Optimizer
    PARAM_OPTIMIZER_ENABLED: bool = os.getenv("APEX_PARAM_OPTIMIZER_ENABLED", "true").lower() == "true"
    PARAM_OPTIMIZER_N_TRIALS: int = int(os.getenv("APEX_PARAM_OPTIMIZER_N_TRIALS", "50"))
    PARAM_OPTIMIZER_INTERVAL_HOURS: float = float(os.getenv("APEX_PARAM_OPTIMIZER_INTERVAL_HOURS", "168"))
    PARAM_OPTIMIZER_LOOKBACK_DAYS: int = int(os.getenv("APEX_PARAM_OPTIMIZER_LOOKBACK_DAYS", "30"))

    # Execution Simulator — pre-trade cost estimation
    EXEC_SIM_ENABLED: bool = os.getenv("APEX_EXEC_SIM_ENABLED", "true").lower() == "true"
    EXEC_SIM_MIN_NOTIONAL: float = float(os.getenv("APEX_EXEC_SIM_MIN_NOTIONAL", "500.0"))

    # Regime Transition Predictor
    REGIME_TRANSITION_PREDICTOR_ENABLED: bool = os.getenv("APEX_REGIME_TP_ENABLED", "true").lower() == "true"
    REGIME_TP_HIGH_PROB_THRESHOLD: float = float(os.getenv("APEX_REGIME_TP_THRESHOLD", "0.60"))
    REGIME_TP_MIN_SIZE_MULT: float = float(os.getenv("APEX_REGIME_TP_MIN_SIZE_MULT", "0.60"))

    # Dynamic Universe Selector
    UNIVERSE_SELECTOR_ENABLED: bool = os.getenv("APEX_UNIVERSE_SELECTOR_ENABLED", "true").lower() == "true"
    UNIVERSE_SELECTOR_MIN_SCORE: float = float(os.getenv("APEX_UNIVERSE_MIN_SCORE", "0.25"))
    UNIVERSE_SELECTOR_REFRESH_CYCLES: int = int(os.getenv("APEX_UNIVERSE_REFRESH_CYCLES", "300"))
    UNIVERSE_SELECTOR_LOOKBACK_DAYS: int = int(os.getenv("APEX_UNIVERSE_LOOKBACK_DAYS", "21"))

    # Trade Diagnostics — gate-decision context tracker
    TRADE_DIAGNOSTICS_ENABLED: bool = os.getenv("APEX_TRADE_DIAGNOSTICS_ENABLED", "true").lower() == "true"

    # Signal A/B Gate — Thompson Sampling challenger/control
    SIGNAL_AB_GATE_ENABLED: bool = os.getenv("APEX_SIGNAL_AB_GATE_ENABLED", "true").lower() == "true"
    SIGNAL_AB_MIN_TRADES: int = int(os.getenv("APEX_SIGNAL_AB_MIN_TRADES", "30"))
    SIGNAL_AB_SHADOW_HOURS: float = float(os.getenv("APEX_SIGNAL_AB_SHADOW_HOURS", "48"))
    SIGNAL_AB_PROMOTION_PROB: float = float(os.getenv("APEX_SIGNAL_AB_PROMOTION_PROB", "0.95"))

    # Regime-Conditional Model Selector — per-regime isotonic calibration
    REGIME_MODEL_SELECTOR_ENABLED: bool = os.getenv("APEX_REGIME_MODEL_ENABLED", "true").lower() == "true"
    REGIME_MODEL_MIN_SAMPLES: int = int(os.getenv("APEX_REGIME_MODEL_MIN_SAMPLES", "30"))
    REGIME_MODEL_MAX_CAL_WEIGHT: float = float(os.getenv("APEX_REGIME_MODEL_MAX_CAL_WEIGHT", "0.40"))

    # Liquidity Sizer — spread-adjusted position sizing + concentration heat
    LIQUIDITY_SIZER_ENABLED: bool = os.getenv("APEX_LIQUIDITY_SIZER_ENABLED", "true").lower() == "true"
    LIQUIDITY_SIZER_PENALTY_K: float = float(os.getenv("APEX_LIQUIDITY_PENALTY_K", "1.5"))
    LIQUIDITY_SIZER_MULT_FLOOR: float = float(os.getenv("APEX_LIQUIDITY_MULT_FLOOR", "0.30"))
    LIQUIDITY_CONCENTRATION_CAP: float = float(os.getenv("APEX_LIQUIDITY_CONCENTRATION_CAP", "0.65"))

    # Edge Miner — unsupervised pattern discovery for confidence boosts
    EDGE_MINER_ENABLED: bool = os.getenv("APEX_EDGE_MINER_ENABLED", "true").lower() == "true"
    EDGE_MINER_N_CLUSTERS: int = int(os.getenv("APEX_EDGE_MINER_N_CLUSTERS", "12"))
    EDGE_MINER_MIN_WIN_RATE: float = float(os.getenv("APEX_EDGE_MINER_MIN_WIN_RATE", "0.60"))
    EDGE_MINER_MAX_BOOST: float = float(os.getenv("APEX_EDGE_MINER_MAX_BOOST", "0.08"))
    EDGE_MINER_REMINE_INTERVAL_HOURS: float = float(os.getenv("APEX_EDGE_MINER_REMINE_HOURS", "168"))

    # Exit Optimizer (self-calibrating ML exit scoring)
    EXIT_OPTIMIZER_ENABLED: bool = os.getenv("APEX_EXIT_OPTIMIZER_ENABLED", "true").lower() == "true"
    EXIT_OPTIMIZER_SCORE_THRESHOLD: float = float(os.getenv("APEX_EXIT_OPTIMIZER_SCORE_THRESHOLD", "0.70"))
    EXIT_OPTIMIZER_MIN_SAMPLES: int = int(os.getenv("APEX_EXIT_OPTIMIZER_MIN_SAMPLES", "30"))
    EXIT_OPTIMIZER_REFIT_INTERVAL_HOURS: float = float(os.getenv("APEX_EXIT_OPTIMIZER_REFIT_HOURS", "168"))

    # Cross-Asset Signal Cascade
    CASCADE_ENABLED: bool = os.getenv("APEX_CASCADE_ENABLED", "true").lower() == "true"
    CASCADE_WINDOW_BARS: int = int(os.getenv("APEX_CASCADE_WINDOW_BARS", "6"))
    CASCADE_TRIGGER_THRESHOLD: float = float(os.getenv("APEX_CASCADE_TRIGGER_THRESHOLD", "0.005"))
    CASCADE_GAIN: float = float(os.getenv("APEX_CASCADE_GAIN", "2.0"))
    CASCADE_FLOOR: float = float(os.getenv("APEX_CASCADE_FLOOR", "0.50"))
    CASCADE_CEILING: float = float(os.getenv("APEX_CASCADE_CEILING", "1.50"))
    CASCADE_MIN_CORRELATION: float = float(os.getenv("APEX_CASCADE_MIN_CORRELATION", "0.35"))

    # Monte Carlo Drawdown Sentinel
    MC_SENTINEL_ENABLED: bool = os.getenv("APEX_MC_SENTINEL_ENABLED", "true").lower() == "true"
    MC_SENTINEL_N_PATHS: int = int(os.getenv("APEX_MC_SENTINEL_N_PATHS", "1000"))
    MC_SENTINEL_LOOK_AHEAD: int = int(os.getenv("APEX_MC_SENTINEL_LOOK_AHEAD", "8"))
    MC_SENTINEL_BREACH_THRESH: float = float(os.getenv("APEX_MC_SENTINEL_BREACH_THRESH", "0.30"))
    MC_SENTINEL_DEFENSIVE_THRESH: float = float(os.getenv("APEX_MC_SENTINEL_DEFENSIVE_THRESH", "0.60"))
    MC_SENTINEL_AMBER_SIZE_MULT: float = float(os.getenv("APEX_MC_SENTINEL_AMBER_MULT", "0.70"))
    MC_SENTINEL_RED_SIZE_MULT: float = float(os.getenv("APEX_MC_SENTINEL_RED_MULT", "0.45"))
    MC_SENTINEL_DEFENSIVE_SIZE_MULT: float = float(os.getenv("APEX_MC_SENTINEL_DEFENSIVE_MULT", "0.20"))
    MC_SENTINEL_DEFENSIVE_MAX_POS: int = int(os.getenv("APEX_MC_SENTINEL_DEFENSIVE_MAX_POS", "2"))

    # Automated Daily Digest
    DAILY_DIGEST_ENABLED: bool = os.getenv("APEX_DAILY_DIGEST_ENABLED", "true").lower() == "true"

    # Signal Staleness Watchdog
    STALENESS_WATCHDOG_ENABLED: bool = os.getenv("APEX_STALENESS_WATCHDOG_ENABLED", "true").lower() == "true"
    STALENESS_WARN_SECONDS: float = float(os.getenv("APEX_STALENESS_WARN_SECONDS", "1800"))
    STALENESS_CRITICAL_SECONDS: float = float(os.getenv("APEX_STALENESS_CRITICAL_SECONDS", "3600"))

    # Intraday mean-reversion signal (VWAP + RSI extreme)
    INTRADAY_MR_ENABLED: bool = os.getenv("APEX_INTRADAY_MR_ENABLED", "true").lower() == "true"
    INTRADAY_MR_BLEND_WEIGHT: float = float(os.getenv("APEX_INTRADAY_MR_BLEND_WEIGHT", "0.12"))

    # Live Regime Transition Alerter — structured severity-escalation alerts
    REGIME_ALERT_ENABLED: bool = os.getenv("APEX_REGIME_ALERT_ENABLED", "true").lower() == "true"
    REGIME_ALERT_COOLDOWN_SECONDS: float = float(os.getenv("APEX_REGIME_ALERT_COOLDOWN_SECONDS", "900"))

    # Execution Timing Optimizer — slippage-bps tracking per (hour, dow, regime)
    EXECUTION_TIMING_ENABLED: bool = os.getenv("APEX_EXECUTION_TIMING_ENABLED", "true").lower() == "true"
    EXECUTION_TIMING_MIN_OBS: int = int(os.getenv("APEX_EXECUTION_TIMING_MIN_OBS", "5"))
    EXECUTION_TIMING_SCORE_FLOOR: float = float(os.getenv("APEX_EXECUTION_TIMING_SCORE_FLOOR", "0.55"))
    EXECUTION_TIMING_CONF_PENALTY: float = float(os.getenv("APEX_EXECUTION_TIMING_CONF_PENALTY", "0.10"))

    # Factor hedger — portfolio beta / factor exposure monitor
    FACTOR_HEDGER_ENABLED: bool = os.getenv("APEX_FACTOR_HEDGER_ENABLED", "true").lower() == "true"
    FACTOR_BETA_WARN_THRESHOLD: float = float(os.getenv("APEX_FACTOR_BETA_WARN_THRESHOLD", "1.20"))
    FACTOR_BETA_URGENT_THRESHOLD: float = float(os.getenv("APEX_FACTOR_BETA_URGENT_THRESHOLD", "1.80"))
    FACTOR_HEDGER_LOOKBACK_DAYS: int = int(os.getenv("APEX_FACTOR_HEDGER_LOOKBACK_DAYS", "20"))
    # HHI concentration gate: shrinks new position size when book is concentrated
    HHI_CONCENTRATION_GATE_ENABLED: bool = os.getenv("APEX_HHI_GATE", "true").lower() == "true"
    HHI_CONCENTRATION_WARN: float = float(os.getenv("APEX_HHI_WARN", "0.25"))   # start shrinking above this
    HHI_CONCENTRATION_HARD: float = float(os.getenv("APEX_HHI_HARD", "0.45"))   # max 50% shrink at this level
    EARNINGS_FILTER_ENABLED: bool = os.getenv("APEX_EARNINGS_FILTER_ENABLED", "true").lower() == "true"
    EARNINGS_BLACKOUT_HOURS_BEFORE: int = int(os.getenv("APEX_EARNINGS_BLACKOUT_HOURS_BEFORE", "24"))
    EARNINGS_BLACKOUT_HOURS_AFTER: int = int(os.getenv("APEX_EARNINGS_BLACKOUT_HOURS_AFTER", "2"))
    EQUITY_OUTLIER_GUARD_ENABLED: bool = os.getenv(
        "APEX_EQUITY_OUTLIER_GUARD_ENABLED", "true"
    ).lower() == "true"
    EQUITY_OUTLIER_MAX_STEP_MOVE_PCT: float = float(
        os.getenv("APEX_EQUITY_OUTLIER_MAX_STEP_MOVE_PCT", "0.25")
    )
    EQUITY_OUTLIER_CONFIRM_SAMPLES: int = int(
        os.getenv("APEX_EQUITY_OUTLIER_CONFIRM_SAMPLES", "3")
    )
    EQUITY_OUTLIER_MATCH_TOLERANCE_PCT: float = float(
        os.getenv("APEX_EQUITY_OUTLIER_MATCH_TOLERANCE_PCT", "0.02")
    )
    EQUITY_RECONCILIATION_ENABLED: bool = os.getenv(
        "APEX_EQUITY_RECONCILIATION_ENABLED", "true"
    ).lower() == "true"
    EQUITY_RECONCILIATION_MAX_GAP_DOLLARS: float = float(
        os.getenv("APEX_EQUITY_RECONCILIATION_MAX_GAP_DOLLARS", "20000")
    )
    EQUITY_RECONCILIATION_MAX_GAP_PCT: float = float(
        os.getenv("APEX_EQUITY_RECONCILIATION_MAX_GAP_PCT", "0.015")
    )
    EQUITY_RECONCILIATION_BREACH_CONFIRMATIONS: int = int(
        os.getenv("APEX_EQUITY_RECONCILIATION_BREACH_CONFIRMATIONS", "2")
    )
    EQUITY_RECONCILIATION_HEAL_CONFIRMATIONS: int = int(
        os.getenv("APEX_EQUITY_RECONCILIATION_HEAL_CONFIRMATIONS", "3")
    )
    EQUITY_RECONCILIATION_FAIL_CLOSED: bool = os.getenv(
        "APEX_EQUITY_RECONCILIATION_FAIL_CLOSED", "true"
    ).lower() == "true"

    # Live performance governor (adaptive risk throttling by realized metrics)
    PERFORMANCE_GOVERNOR_ENABLED: bool = os.getenv(
        "APEX_PERFORMANCE_GOVERNOR_ENABLED", "true"
    ).lower() == "true"
    PERFORMANCE_TARGET_SHARPE: float = float(os.getenv("APEX_PERFORMANCE_TARGET_SHARPE", "0.8"))
    PERFORMANCE_TARGET_SORTINO: float = float(os.getenv("APEX_PERFORMANCE_TARGET_SORTINO", "1.0"))
    PERFORMANCE_MAX_DRAWDOWN: float = float(os.getenv("APEX_PERFORMANCE_MAX_DRAWDOWN", "0.10"))
    PERFORMANCE_GOV_SAMPLE_MINUTES: int = int(os.getenv("APEX_PERFORMANCE_GOV_SAMPLE_MINUTES", "15"))
    PERFORMANCE_GOV_MIN_SAMPLES: int = int(os.getenv("APEX_PERFORMANCE_GOV_MIN_SAMPLES", "30"))
    PERFORMANCE_GOV_LOOKBACK_POINTS: int = int(os.getenv("APEX_PERFORMANCE_GOV_LOOKBACK_POINTS", "200"))
    PERFORMANCE_GOV_RECOVERY_POINTS: int = int(os.getenv("APEX_PERFORMANCE_GOV_RECOVERY_POINTS", "3"))
    PERFORMANCE_GOV_POINTS_PER_YEAR: int = int(os.getenv("APEX_PERFORMANCE_GOV_POINTS_PER_YEAR", "3276"))

    # Dynamic governor targets — regime multipliers with floor
    GOVERNOR_REGIME_TARGET_MULTIPLIERS: Dict = {
        "strong_bull": 1.2, "bull": 1.0, "neutral": 0.8,
        "bear": 0.6, "volatile": 0.5,
    }
    GOVERNOR_MIN_TARGET_SHARPE: float = 0.4
    GOVERNOR_MIN_TARGET_SORTINO: float = 0.5
    GOVERNOR_POLICY_SCOPE: str = os.getenv("APEX_GOVERNOR_POLICY_SCOPE", "asset_class_regime")
    GOVERNOR_TUNE_CADENCE_DEFAULT: str = os.getenv("APEX_GOVERNOR_TUNE_CADENCE_DEFAULT", "weekly")
    GOVERNOR_TUNE_CADENCE_CRYPTO: str = os.getenv("APEX_GOVERNOR_TUNE_CADENCE_CRYPTO", "daily")
    GOVERNOR_AUTO_PROMOTE_NON_PROD: bool = os.getenv(
        "APEX_GOVERNOR_AUTO_PROMOTE_NON_PROD", "true"
    ).lower() == "true"
    GOVERNOR_PROD_MANUAL_APPROVAL: bool = os.getenv(
        "APEX_GOVERNOR_PROD_MANUAL_APPROVAL", "true"
    ).lower() == "true"

    # Hard kill-switch (drawdown + Sharpe decay)
    KILL_SWITCH_ENABLED: bool = os.getenv("APEX_KILL_SWITCH_ENABLED", "true").lower() == "true"
    KILL_SWITCH_DD_MULTIPLIER: float = float(os.getenv("APEX_KILL_SWITCH_DD_MULTIPLIER", "1.5"))
    KILL_SWITCH_SHARPE_WINDOW_DAYS: int = int(os.getenv("APEX_KILL_SWITCH_SHARPE_WINDOW_DAYS", "63"))
    KILL_SWITCH_SHARPE_FLOOR: float = float(os.getenv("APEX_KILL_SWITCH_SHARPE_FLOOR", "0.2"))
    KILL_SWITCH_LOGIC: str = os.getenv("APEX_KILL_SWITCH_LOGIC", "AND")
    KILL_SWITCH_MIN_POINTS: int = int(os.getenv("APEX_KILL_SWITCH_MIN_POINTS", "20"))
    KILL_SWITCH_HISTORICAL_MDD_BASELINE: float = float(
        os.getenv("APEX_KILL_SWITCH_HISTORICAL_MDD_BASELINE", "0.08")
    )

    # ── Volatility Targeting ──────────────────────────────────────────────────
    VOL_TARGET_ANN: float = float(os.getenv("APEX_VOL_TARGET_ANN", "0.15"))          # 15% annual vol target
    VOL_TARGET_LOOKBACK_DAYS: int = int(os.getenv("APEX_VOL_TARGET_LOOKBACK_DAYS", "20"))
    VOL_TARGET_MIN_DAYS: int = int(os.getenv("APEX_VOL_TARGET_MIN_DAYS", "5"))
    VOL_TARGET_MIN_MULT: float = float(os.getenv("APEX_VOL_TARGET_MIN_MULT", "0.30"))
    VOL_TARGET_MAX_MULT: float = float(os.getenv("APEX_VOL_TARGET_MAX_MULT", "2.00"))

    # ── Drawdown-Adaptive Leverage ────────────────────────────────────────────
    DRAWDOWN_ADAPTIVE_LEVERAGE_ENABLED: bool = (
        os.getenv("APEX_DRAWDOWN_ADAPTIVE_LEVERAGE_ENABLED", "true").lower() == "true"
    )
    DD_LEV_TIER1_PCT: float = float(os.getenv("APEX_DD_LEV_TIER1_PCT", "-0.02"))   # -2% → 50% size
    DD_LEV_TIER2_PCT: float = float(os.getenv("APEX_DD_LEV_TIER2_PCT", "-0.04"))   # -4% → 25% size

    # Social-media shock governor (attention/sentiment + verified prediction odds)
    SOCIAL_SHOCK_GOVERNOR_ENABLED: bool = os.getenv(
        "APEX_SOCIAL_SHOCK_GOVERNOR_ENABLED", "true"
    ).lower() == "true"
    # Kill switch for dynamic adapters (NEWS_AGG, MARKET) — set False to disable
    SOCIAL_DYNAMIC_ADAPTERS_ENABLED: bool = os.getenv(
        "APEX_SOCIAL_DYNAMIC_ADAPTERS_ENABLED", "true"
    ).lower() == "true"
    SOCIAL_RISK_ATTENTION_TRIGGER_Z: float = float(
        os.getenv("APEX_SOCIAL_RISK_ATTENTION_TRIGGER_Z", "1.0")
    )
    SOCIAL_RISK_ATTENTION_EXTREME_Z: float = float(
        os.getenv("APEX_SOCIAL_RISK_ATTENTION_EXTREME_Z", "3.0")
    )
    SOCIAL_RISK_NEGATIVE_SENTIMENT_TRIGGER: float = float(
        os.getenv("APEX_SOCIAL_RISK_NEGATIVE_SENTIMENT_TRIGGER", "-0.35")
    )
    SOCIAL_RISK_POSITIVE_SENTIMENT_TRIGGER: float = float(
        os.getenv("APEX_SOCIAL_RISK_POSITIVE_SENTIMENT_TRIGGER", "0.75")
    )
    SOCIAL_RISK_ATTENTION_WEIGHT: float = float(
        os.getenv("APEX_SOCIAL_RISK_ATTENTION_WEIGHT", "0.60")
    )
    SOCIAL_RISK_SENTIMENT_WEIGHT: float = float(
        os.getenv("APEX_SOCIAL_RISK_SENTIMENT_WEIGHT", "0.40")
    )
    SOCIAL_RISK_MIN_PLATFORMS: int = int(
        os.getenv("APEX_SOCIAL_RISK_MIN_PLATFORMS", "2")
    )
    SOCIAL_SHOCK_REDUCE_THRESHOLD: float = float(
        os.getenv("APEX_SOCIAL_SHOCK_REDUCE_THRESHOLD", "0.60")
    )
    SOCIAL_SHOCK_BLOCK_THRESHOLD: float = float(
        os.getenv("APEX_SOCIAL_SHOCK_BLOCK_THRESHOLD", "0.85")
    )
    SOCIAL_SHOCK_MIN_GROSS_MULTIPLIER: float = float(
        os.getenv("APEX_SOCIAL_SHOCK_MIN_GROSS_MULTIPLIER", "0.35")
    )
    SOCIAL_SHOCK_VERIFIED_EVENT_WEIGHT: float = float(
        os.getenv("APEX_SOCIAL_SHOCK_VERIFIED_EVENT_WEIGHT", "0.30")
    )
    SOCIAL_SHOCK_VERIFIED_EVENT_FLOOR: float = float(
        os.getenv("APEX_SOCIAL_SHOCK_VERIFIED_EVENT_FLOOR", "0.55")
    )
    PREDICTION_VERIFY_MIN_SOURCES: int = int(
        os.getenv("APEX_PREDICTION_VERIFY_MIN_SOURCES", "2")
    )
    PREDICTION_VERIFY_MAX_PROB_DIVERGENCE: float = float(
        os.getenv("APEX_PREDICTION_VERIFY_MAX_PROB_DIVERGENCE", "0.15")
    )
    PREDICTION_VERIFY_MAX_SOURCE_DISAGREEMENT: float = float(
        os.getenv("APEX_PREDICTION_VERIFY_MAX_SOURCE_DISAGREEMENT", "0.20")
    )
    PREDICTION_VERIFY_MIN_MARKET_PROB: float = float(
        os.getenv("APEX_PREDICTION_VERIFY_MIN_MARKET_PROB", "0.05")
    )

    # Institutional pre-trade hard-limit gateway (fat-finger/leverage/ADV)
    PRETRADE_GATEWAY_ENABLED: bool = os.getenv(
        "APEX_PRETRADE_GATEWAY_ENABLED", "true"
    ).lower() == "true"
    PRETRADE_GATEWAY_FAIL_CLOSED: bool = os.getenv(
        "APEX_PRETRADE_GATEWAY_FAIL_CLOSED", "true"
    ).lower() == "true"
    PRETRADE_MAX_ORDER_NOTIONAL: float = float(
        os.getenv("APEX_PRETRADE_MAX_ORDER_NOTIONAL", "50000.0")
    )
    PRETRADE_MAX_PRICE_DEVIATION_BPS: float = float(
        os.getenv("APEX_PRETRADE_MAX_PRICE_DEVIATION_BPS", "500")  # Raised from 250: PANIC regimes see 3-8% intraday moves
    )
    PRETRADE_MAX_PARTICIPATION_RATE: float = float(
        os.getenv("APEX_PRETRADE_MAX_PARTICIPATION_RATE", "0.10")
    )
    PRETRADE_MAX_GROSS_EXPOSURE_RATIO: float = float(
        os.getenv("APEX_PRETRADE_MAX_GROSS_EXPOSURE_RATIO", "2.0")
    )

    # Intraday portfolio stress control loop
    INTRADAY_STRESS_ENGINE_ENABLED: bool = os.getenv(
        "APEX_INTRADAY_STRESS_ENGINE_ENABLED", "true"
    ).lower() == "true"
    INTRADAY_STRESS_INTERVAL_CYCLES: int = int(
        os.getenv("APEX_INTRADAY_STRESS_INTERVAL_CYCLES", "15")
    )
    INTRADAY_STRESS_SCENARIOS: list[str] = [
        item.strip()
        for item in os.getenv(
            "APEX_INTRADAY_STRESS_SCENARIOS",
            "2020_covid_crash,vix_spike,correlation_breakdown,rate_shock",
        ).split(",")
        if item.strip()
    ]
    INTRADAY_STRESS_WARNING_RETURN_THRESHOLD: float = float(
        os.getenv("APEX_INTRADAY_STRESS_WARNING_RETURN_THRESHOLD", "-0.04")
    )
    INTRADAY_STRESS_HALT_RETURN_THRESHOLD: float = float(
        os.getenv("APEX_INTRADAY_STRESS_HALT_RETURN_THRESHOLD", "-0.08")
    )
    INTRADAY_STRESS_WARNING_DRAWDOWN_THRESHOLD: float = float(
        os.getenv("APEX_INTRADAY_STRESS_WARNING_DRAWDOWN_THRESHOLD", "0.06")
    )
    INTRADAY_STRESS_HALT_DRAWDOWN_THRESHOLD: float = float(
        os.getenv("APEX_INTRADAY_STRESS_HALT_DRAWDOWN_THRESHOLD", "0.10")
    )
    INTRADAY_STRESS_WARNING_SIZE_MULTIPLIER: float = float(
        os.getenv("APEX_INTRADAY_STRESS_WARNING_SIZE_MULTIPLIER", "0.60")
    )
    INTRADAY_STRESS_HALT_SIZE_MULTIPLIER: float = float(
        os.getenv("APEX_INTRADAY_STRESS_HALT_SIZE_MULTIPLIER", "0.25")
    )
    STRESS_UNWIND_ENABLED: bool = os.getenv(
        "APEX_STRESS_UNWIND_ENABLED", "true"
    ).lower() == "true"
    STRESS_UNWIND_MAX_POSITIONS_PER_CYCLE: int = int(
        os.getenv("APEX_STRESS_UNWIND_MAX_POSITIONS_PER_CYCLE", "2")
    )
    STRESS_UNWIND_MAX_PARTICIPATION_RATE: float = float(
        os.getenv("APEX_STRESS_UNWIND_MAX_PARTICIPATION_RATE", "0.05")
    )
    STRESS_UNWIND_MIN_REDUCTION_PCT: float = float(
        os.getenv("APEX_STRESS_UNWIND_MIN_REDUCTION_PCT", "0.10")
    )
    STRESS_UNWIND_FALLBACK_REDUCTION_PCT: float = float(
        os.getenv("APEX_STRESS_UNWIND_FALLBACK_REDUCTION_PCT", "0.25")
    )

    # Trading-loop Prometheus metrics exporter
    PROMETHEUS_TRADING_METRICS_ENABLED: bool = os.getenv(
        "APEX_PROMETHEUS_TRADING_METRICS_ENABLED", "true"
    ).lower() == "true"
    PROMETHEUS_TRADING_METRICS_PORT: int = int(
        os.getenv("APEX_PROMETHEUS_TRADING_METRICS_PORT", "9108")
    )

    # ═══════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER (Automatic Trading Halt)
    # ═══════════════════════════════════════════════════════════════
    CIRCUIT_BREAKER_ENABLED = True  # Enable automatic trading halt
    CIRCUIT_BREAKER_DAILY_LOSS = 0.03  # Halt if daily loss exceeds 3% (Moderate)
    CIRCUIT_BREAKER_WARNING = 0.015  # Warning at 1.5% - reduce position sizes by 50%
    CIRCUIT_BREAKER_DRAWDOWN = 0.08  # Halt if drawdown exceeds 8%
    CIRCUIT_BREAKER_CONSECUTIVE_LOSSES = 5  # Halt after 5 consecutive losing trades
    CIRCUIT_BREAKER_COOLDOWN_HOURS = 24  # Hours before trading resumes after halt
    # Minimum absolute loss (USD) for a trade to count as a "loss" in the consecutive
    # loss counter. Prevents micro-losses from partial fills (e.g., 10 TWAP tranches each
    # losing $5) from falsely tripping the circuit breaker.
    CIRCUIT_BREAKER_MIN_LOSS_USD: float = float(
        os.getenv("APEX_CIRCUIT_BREAKER_MIN_LOSS_USD", "25.0")
    )

    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO REBALANCING
    # ═══════════════════════════════════════════════════════════════
    REBALANCE_ENABLED = True  # Enable automatic rebalancing
    REBALANCE_DRIFT_THRESHOLD = 0.10  # Rebalance when position drifts >10% from target
    REBALANCE_MIN_INTERVAL_HOURS = 24  # Minimum hours between rebalances
    REBALANCE_AT_MARKET_CLOSE = True  # Prefer rebalancing near market close (3:30 PM EST)

    # ═══════════════════════════════════════════════════════════════
    # NETWORK RESILIENCE
    # ═══════════════════════════════════════════════════════════════
    IBKR_MAX_RETRIES = 5  # Maximum retry attempts for IBKR operations
    IBKR_RETRY_BASE_DELAY = 2.0  # Base delay in seconds (exponential backoff)
    IBKR_RETRY_MAX_DELAY = 60.0  # Maximum delay between retries
    IBKR_CONNECTION_TIMEOUT = 30  # Connection timeout in seconds
    
    # ═══════════════════════════════════════════════════════════════
    # SIGNAL THRESHOLDS (QUALITY FOCUS - Reduced noise)    # ML Configuration
    # ═══════════════════════════════════════════════════════════════
    # FX Signal Calibration
    FX_SIGNAL_THRESHOLD = 0.05       # Lower threshold for low-volatility FX macro patterns
    FX_SIGNAL_GAIN_MULTIPLIER = 3.0  # Post-tanh gain amplification for FX signals

    # ── Intraday 5-minute bar features ───────────────────────────────────────
    INTRADAY_FEATURES_ENABLED: bool = True
    INTRADAY_INTERVAL: str = "5m"     # Bar interval for intraday data
    INTRADAY_PERIOD: str = "1d"       # Lookback for intraday data

    MIN_SIGNAL_THRESHOLD = 0.15      # Recalibrated: lowered from 0.18 to allow moderate signals through — hedge dampener already risk-adjusts
    MIN_CONFIDENCE = 0.60            # Raised from 0.55 — higher bar for fresh entries
    CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER: float = float(
        os.getenv("APEX_CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER", "0.80")  # raised from 0.60 → stronger signal required
    )
    CRYPTO_CONFIDENCE_THRESHOLD_MULTIPLIER: float = float(
        os.getenv("APEX_CRYPTO_CONFIDENCE_THRESHOLD_MULTIPLIER", "0.70")
    )
    # ATR hard stop enforcement (per-cycle price check against position_stops)
    ATR_STOP_ENFORCEMENT_ENABLED: bool = os.getenv("APEX_ATR_STOP_ENFORCEMENT", "true").lower() == "true"

    # Win-rate guards: signal stability + max concurrent crypto positions
    CRYPTO_SIGNAL_STABILITY_BARS: int = int(os.getenv("APEX_CRYPTO_SIGNAL_STABILITY_BARS", "2"))
    CRYPTO_MAX_CONCURRENT_POSITIONS: int = int(os.getenv("APEX_CRYPTO_MAX_CONCURRENT_POSITIONS", "4"))
    # Crypto-wide consecutive loss pause: after N crypto losses in a row, pause all crypto entries
    CRYPTO_CONSEC_LOSS_PAUSE_COUNT: int = int(os.getenv("APEX_CRYPTO_CONSEC_LOSS_PAUSE_COUNT", "3"))
    CRYPTO_CONSEC_LOSS_PAUSE_HOURS: float = float(os.getenv("APEX_CRYPTO_CONSEC_LOSS_PAUSE_HOURS", "4.0"))
    # ── Scale-in on winning positions ─────────────────────────────────────────
    # Adds 25% to an open position once it is up SCALE_IN_PROFIT_PCT% and signal
    # remains strong (≥ SCALE_IN_MIN_SIGNAL) in the same direction. Fires once per
    # position (reset when position closes). Long-only by design (scale-in shorts
    # disabled — gap risk asymmetric).
    SCALE_IN_ENABLED: bool = os.getenv("APEX_SCALE_IN_ENABLED", "true").lower() == "true"
    SCALE_IN_PROFIT_PCT: float = float(os.getenv("APEX_SCALE_IN_PROFIT_PCT", "1.5"))
    SCALE_IN_MIN_SIGNAL: float = float(os.getenv("APEX_SCALE_IN_MIN_SIGNAL", "0.18"))
    SCALE_IN_SIZE_PCT: float = float(os.getenv("APEX_SCALE_IN_SIZE_PCT", "0.25"))

    EXCELLENCE_MIN_HOLD_BARS: int = int(os.getenv("APEX_EXCELLENCE_MIN_HOLD_BARS", "2"))
    # Minimum minutes a position must be held before excellence exit can fire.
    # Raised to 90 min: crypto needs time to develop — 30 min was killing trades at first pullback.
    EXCELLENCE_MIN_HOLD_MINUTES: float = float(os.getenv("APEX_EXCELLENCE_MIN_HOLD_MINUTES", "90"))
    # Quick-mismatch grace loss threshold for crypto positions held < 1h (wider due to volatility).
    # Equity uses -0.80%; crypto uses this wider threshold so normal intraday swings don't trigger exits.
    QUICK_MISMATCH_GRACE_CRYPTO_PCT: float = float(os.getenv("APEX_QUICK_MISMATCH_GRACE_CRYPTO_PCT", "-2.0"))
    # Per-position hedge force-exit threshold (%) during correlation crisis.
    # Exit fires when corr >= 0.85 AND position loss exceeds this level.
    # Raised from 1.0% to 2.0% to avoid exiting on normal intraday volatility.
    HEDGE_PER_POSITION_FORCE_EXIT_PCT: float = float(os.getenv("APEX_HEDGE_PER_POSITION_FORCE_EXIT_PCT", "2.0"))
    CRYPTO_ROTATION_ENABLED: bool = os.getenv(
        "APEX_CRYPTO_ROTATION_ENABLED", "true"
    ).lower() == "true"
    CRYPTO_ROTATION_TOP_N: int = int(
        os.getenv("APEX_CRYPTO_ROTATION_TOP_N", "14")
    )
    CRYPTO_ROTATION_MOMENTUM_LOOKBACK: int = int(
        os.getenv("APEX_CRYPTO_ROTATION_MOMENTUM_LOOKBACK", "20")
    )
    CRYPTO_ROTATION_LIQUIDITY_LOOKBACK: int = int(
        os.getenv("APEX_CRYPTO_ROTATION_LIQUIDITY_LOOKBACK", "20")
    )
    CRYPTO_ROTATION_MIN_DOLLAR_VOLUME: float = float(
        os.getenv("APEX_CRYPTO_ROTATION_MIN_DOLLAR_VOLUME", "250000")
    )
    CRYPTO_ROTATION_MOMENTUM_WEIGHT: float = float(
        os.getenv("APEX_CRYPTO_ROTATION_MOMENTUM_WEIGHT", "0.65")
    )
    CRYPTO_ROTATION_LIQUIDITY_WEIGHT: float = float(
        os.getenv("APEX_CRYPTO_ROTATION_LIQUIDITY_WEIGHT", "0.35")
    )
    # Extra crypto gating controls:
    # - Top-ranked pairs get lower thresholds and momentum-alignment boosts.
    CRYPTO_ROTATION_THRESHOLD_DISCOUNT_MAX: float = float(
        os.getenv("APEX_CRYPTO_ROTATION_THRESHOLD_DISCOUNT_MAX", "0.30")
    )
    CRYPTO_MOMENTUM_ALIGN_SIGNAL_BOOST_MAX: float = float(
        os.getenv("APEX_CRYPTO_MOMENTUM_ALIGN_SIGNAL_BOOST_MAX", "0.08")
    )
    CRYPTO_MOMENTUM_ALIGN_CONFIDENCE_BOOST_MAX: float = float(
        os.getenv("APEX_CRYPTO_MOMENTUM_ALIGN_CONFIDENCE_BOOST_MAX", "0.10")
    )
    CRYPTO_MOMENTUM_CONFLICT_CONFIDENCE_PENALTY_MAX: float = float(
        os.getenv("APEX_CRYPTO_MOMENTUM_CONFLICT_CONFIDENCE_PENALTY_MAX", "0.08")
    )
    # During overnight crypto-only sessions, refresh market data every N seconds
    # (vs 3600s during equity hours). Lower = fresher signals, higher = less yfinance load.
    CRYPTO_OVERNIGHT_DATA_REFRESH_SECONDS: int = int(
        os.getenv("APEX_CRYPTO_OVERNIGHT_DATA_REFRESH_SECONDS", "900")
    )

    # Sentiment API configuration
    SENTIMENT_API_ENABLED = True     # Toggle Yahoo Finance news sentiment parsing
    
    FORCE_RETRAIN = False            # Set to True to force model retraining on startup

    # Regime-based entry thresholds — calibrated Mar 2026 to model output range [0.05, 0.272], mean 0.124.
    # MATHEMATICAL CONSTRAINT: all thresholds MUST be < model output max (0.272).
    # These thresholds apply to entries ALIGNED with the regime direction:
    #   bull  → LONG entries, bear → SHORT entries.
    # Counter-trend entries (LONG in bear, SHORT in bull) are gated by
    #   REGIME_COUNTER_TREND_SIGNAL_MULT applied on top of these thresholds.
    SIGNAL_THRESHOLDS_BY_REGIME = {
        'strong_bull': 0.14,    # ~p40: aggressive in confirmed uptrend (breadth wide)
        'bull': 0.17,           # ~p55: standard bull — moderate conviction
        'neutral': 0.18,        # ~p60: slightly tighter in directionless tape
        'bear': 0.15,           # ~p50: LOWER — aligned SHORT entries easier in bear
        'strong_bear': 0.13,    # ~p40: LOWER — aligned SHORT entries easy in strong bear
        'volatile': 0.17,       # ~p60: volatile — slightly tighter but allow both dirs
        'crisis': 0.22,         # ~p85: crisis — only high-conviction signals
    }

    # Counter-trend signal multiplier: going AGAINST the regime (LONG in bear, SHORT in bull)
    # requires this multiple of the regime threshold.
    # Example: bear threshold=0.15, mult=1.8 → counter-trend LONG needs signal >= 0.27
    # strong_bear: 0.13 × 2.0 = 0.26 → effectively blocked (near model max)
    REGIME_COUNTER_TREND_SIGNAL_MULT: float = float(
        os.getenv("APEX_REGIME_COUNTER_TREND_SIGNAL_MULT", "1.8")
    )
    # Hard block LONGs in strong_bear regardless of signal strength
    REGIME_STRONG_BEAR_LONG_BLOCK: bool = (
        os.getenv("APEX_REGIME_STRONG_BEAR_LONG_BLOCK", "true").lower() == "true"
    )

    # Tiered confidence gate: moderate signals require higher conviction at entry
    # ENTRY_SIGNAL_HIGH_CUTOFF: model ~p90 (0.272) — signals above this pass with floor confidence.
    # Signals below cutoff are "moderate" and require conf >= ENTRY_CONFIDENCE_MODERATE.
    # Raised back to 0.60 (2026-03-19 audit fix): 0.44 was too permissive — it allowed
    # trades with confidence 0.44 on weak signals (signal=0.12), driving 0% win rate.
    # With 0.60 base: weak signal (0.10) needs conf >= 0.55; strong signal needs >= 0.48.
    ENTRY_TIERED_CONFIDENCE_ENABLED: bool = True
    ENTRY_SIGNAL_HIGH_CUTOFF: float = float(os.getenv("APEX_ENTRY_SIGNAL_HIGH_CUTOFF", "0.22"))   # p85 of model
    ENTRY_CONFIDENCE_MODERATE: float = float(os.getenv("APEX_ENTRY_CONFIDENCE_MODERATE", "0.60"))  # raised from 0.44

    # Drawdown gate: after daily losses, require higher confidence for new entries
    ENTRY_DRAWDOWN_GATE_PCT: float = float(os.getenv("APEX_ENTRY_DRAWDOWN_GATE_PCT", "0.015"))
    ENTRY_DRAWDOWN_CONF_BOOST: float = float(os.getenv("APEX_ENTRY_DRAWDOWN_CONF_BOOST", "0.10"))  # raised from 0.08

    # Liquidity window for crypto exits (UTC hours — peak US/EU overlap session)
    CRYPTO_EXIT_LIQUIDITY_WINDOW_ENABLED: bool = (
        os.getenv("APEX_CRYPTO_EXIT_LIQUIDITY_WINDOW_ENABLED", "true").lower() == "true"
    )
    CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_START: int = int(
        os.getenv("APEX_CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_START", "13")   # 9am ET
    )
    CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_END: int = int(
        os.getenv("APEX_CRYPTO_EXIT_LIQUIDITY_WINDOW_UTC_END", "22")     # 6pm ET
    )
    CRYPTO_EXIT_URGENCY_THRESHOLD_PCT: float = float(
        os.getenv("APEX_CRYPTO_EXIT_URGENCY_THRESHOLD_PCT", "-2.5")      # Exit anyway below -2.5%
    )

    # Tier 4: Signal momentum gate — block entry if per-symbol signal is weakening
    SIGNAL_MOMENTUM_GATE_ENABLED: bool = (
        os.getenv("APEX_SIGNAL_MOMENTUM_GATE", "true").lower() != "false"
    )
    SIGNAL_MOMENTUM_HISTORY_BARS: int = int(os.getenv("APEX_SIGNAL_MOMENTUM_HISTORY_BARS", "4"))
    SIGNAL_MOMENTUM_MIN_SLOPE: float = float(os.getenv("APEX_SIGNAL_MOMENTUM_MIN_SLOPE", "-0.05"))
    EXPECTANCY_LEDGER_LOOKBACK_DAYS: int = int(
        os.getenv("APEX_EXPECTANCY_LEDGER_LOOKBACK_DAYS", "60")
    )
    EXPECTANCY_LEDGER_MIN_TRADES: int = int(
        os.getenv("APEX_EXPECTANCY_LEDGER_MIN_TRADES", "5")
    )
    EXPECTANCY_LEDGER_REFRESH_SECONDS: float = float(
        os.getenv("APEX_EXPECTANCY_LEDGER_REFRESH_SECONDS", "300.0")
    )
    GENERATOR_DEMOTION_ENABLED: bool = (
        os.getenv("APEX_GENERATOR_DEMOTION_ENABLED", "true").lower() != "false"
    )
    GENERATOR_DEMOTION_BLOCK_PNL_BPS: float = float(
        os.getenv("APEX_GENERATOR_DEMOTION_BLOCK_PNL_BPS", "-35.0")
    )
    GENERATOR_DEMOTION_SIGNAL_MULTIPLIER: float = float(
        os.getenv("APEX_GENERATOR_DEMOTION_SIGNAL_MULTIPLIER", "0.85")
    )
    GENERATOR_DEMOTION_CONFIDENCE_MULTIPLIER: float = float(
        os.getenv("APEX_GENERATOR_DEMOTION_CONFIDENCE_MULTIPLIER", "0.80")
    )
    NO_TRADE_BAND_ENABLED: bool = (
        os.getenv("APEX_NO_TRADE_BAND_ENABLED", "true").lower() != "false"
    )
    NO_TRADE_BAND_RATIO: float = float(os.getenv("APEX_NO_TRADE_BAND_RATIO", "0.12"))
    NO_TRADE_BAND_MAX_SLOPE: float = float(
        os.getenv("APEX_NO_TRADE_BAND_MAX_SLOPE", "0.01")
    )
    CONFIDENCE_AUDIT_NEAR_MISS_GAP: float = float(
        os.getenv("APEX_CONFIDENCE_AUDIT_NEAR_MISS_GAP", "0.02")
    )

    # Tier 5: Regime component agreement gate — penalize when sub-signals disagree with main signal
    REGIME_COMPONENT_AGREEMENT_ENABLED: bool = (
        os.getenv("APEX_REGIME_COMPONENT_AGREEMENT", "true").lower() != "false"
    )
    REGIME_COMPONENT_CONF_PENALTY: float = float(
        os.getenv("APEX_REGIME_COMPONENT_CONF_PENALTY", "0.06")   # Halved: ML conf rarely >0.70 in live data
    )

    # Excellence exit persistence: require N consecutive weak-signal bars before firing.
    # Prevents false exits on single-bar signal noise.
    EXCELLENCE_PERSIST_BARS: int = int(os.getenv("APEX_EXCELLENCE_PERSIST_BARS", "2"))

    # Signal 4-bar moving-average gate: require mean(last-4 signals) >= this before entry.
    SIGNAL_MA_GATE_ENABLED: bool = os.getenv("APEX_SIGNAL_MA_GATE", "true").lower() != "false"
    SIGNAL_MA_MIN: float = float(os.getenv("APEX_SIGNAL_MA_MIN", "0.18"))

    # BTC macro filter: block altcoin LONG entries when BTC signal is below threshold.
    BTC_MACRO_FILTER_ENABLED: bool = os.getenv("APEX_BTC_MACRO_FILTER", "true").lower() != "false"
    BTC_MACRO_FILTER_MIN_SIGNAL: float = float(os.getenv("APEX_BTC_MACRO_FILTER_MIN_SIGNAL", "0.0"))

    # Altcoin probation: symbols that require a higher signal threshold due to poor track record.
    ALTCOIN_PROBATION_SYMBOLS: List[str] = [
        s.strip() for s in os.getenv(
            "APEX_ALTCOIN_PROBATION_SYMBOLS",
            "CRYPTO:AVAX/USD,CRYPTO:AAVE/USD",   # 0% win rate in live data
        ).split(",") if s.strip()
    ]
    ALTCOIN_PROBATION_MIN_SIGNAL: float = float(os.getenv("APEX_ALTCOIN_PROBATION_MIN_SIGNAL", "0.28"))  # Recalibrated: model max is 0.272

    # Portfolio heat gate: block new entries when total unrealised portfolio loss > threshold.
    PORTFOLIO_HEAT_GATE_ENABLED: bool = os.getenv("APEX_PORTFOLIO_HEAT_GATE", "true").lower() != "false"
    PORTFOLIO_HEAT_MAX_LOSS_PCT: float = float(os.getenv("APEX_PORTFOLIO_HEAT_MAX_LOSS_PCT", "0.03"))

    # Option B: GodLevel signal blend — merges options flow, RL weights, VADER news,
    # and on-chain crypto whale data into the primary institutional signal at WEIGHT %.
    GOD_LEVEL_BLEND_ENABLED: bool = os.getenv("APEX_GOD_LEVEL_BLEND", "true").lower() != "false"
    GOD_LEVEL_BLEND_WEIGHT: float = float(os.getenv("APEX_GOD_LEVEL_BLEND_WEIGHT", "0.12"))

    # Live Kelly position sizing: scale position size based on rolling win/loss stats.
    LIVE_KELLY_SIZING_ENABLED: bool = os.getenv("APEX_LIVE_KELLY_SIZING", "true").lower() != "false"
    EXECUTION_TIMING_SIZE_ENABLED: bool = (
        os.getenv("APEX_EXECUTION_TIMING_SIZE_ENABLED", "true").lower() != "false"
    )
    EXECUTION_EXPECTANCY_SIZING_ENABLED: bool = (
        os.getenv("APEX_EXECUTION_EXPECTANCY_SIZING_ENABLED", "true").lower() != "false"
    )
    EXECUTION_EXPECTANCY_SIZE_FLOOR: float = float(
        os.getenv("APEX_EXECUTION_EXPECTANCY_SIZE_FLOOR", "0.75")
    )
    EXECUTION_EXPECTANCY_LOSS_FLOOR_BPS: float = float(
        os.getenv("APEX_EXECUTION_EXPECTANCY_LOSS_FLOOR_BPS", "-50.0")
    )

    # Regime transition caution window: reduce size for N hours after exiting a bear/volatile regime.
    REGIME_TRANSITION_CAUTION_ENABLED: bool = (
        os.getenv("APEX_REGIME_TRANSITION_CAUTION", "true").lower() != "false"
    )
    REGIME_TRANSITION_CAUTION_HOURS: float = float(os.getenv("APEX_REGIME_TRANSITION_CAUTION_HOURS", "4.0"))
    REGIME_TRANSITION_SIZE_MULT: float = float(os.getenv("APEX_REGIME_TRANSITION_SIZE_MULT", "0.70"))

    # ── Session-aware entry windows ──────────────────────────────────────────
    # Block new equity entries in the first N minutes after NYSE open (9:30 ET).
    # Wide spreads and erratic price action in the opening auction distort signals.
    SESSION_AWARE_ENTRY_ENABLED: bool = (
        os.getenv("APEX_SESSION_AWARE_ENTRY", "true").lower() == "true"
    )
    EQUITY_ENTRY_AVOID_OPEN_MINUTES: int = int(
        os.getenv("APEX_EQUITY_AVOID_OPEN_MIN", "5")   # avoid 9:30-9:35 ET
    )
    EQUITY_ENTRY_AVOID_CLOSE_MINUTES: int = int(
        os.getenv("APEX_EQUITY_AVOID_CLOSE_MIN", "5")  # avoid 15:55-16:00 ET
    )

    # ── Weekly model retraining ───────────────────────────────────────────────
    # Time-based retraining trigger to complement outcome-based trigger.
    # Re-fits ML models once per week to absorb recent market regime changes.
    WEEKLY_RETRAIN_ENABLED: bool = (
        os.getenv("APEX_WEEKLY_RETRAIN_ENABLED", "true").lower() == "true"
    )
    WEEKLY_RETRAIN_INTERVAL_HOURS: int = int(
        os.getenv("APEX_WEEKLY_RETRAIN_INTERVAL_HOURS", "168")  # 7 days
    )

    # ── Post-Earnings Announcement Drift (PEAD) signal ────────────────────────
    # Stocks with positive EPS surprise tend to drift upward for 30-60 days.
    # Stocks with negative EPS surprise drift down. This is a documented anomaly.
    EARNINGS_PEAD_ENABLED: bool = (
        os.getenv("APEX_EARNINGS_PEAD_ENABLED", "true").lower() == "true"
    )
    EARNINGS_PEAD_CACHE_TTL_SEC: int = int(
        os.getenv("APEX_EARNINGS_PEAD_CACHE_TTL", "3600")   # 1 hour TTL
    )
    EARNINGS_PEAD_DECAY_HALFLIFE_DAYS: int = int(
        os.getenv("APEX_EARNINGS_PEAD_DECAY_HALFLIFE_DAYS", "30")  # signal decays 50% in 30 days
    )
    EARNINGS_PEAD_MIN_SURPRISE: float = float(
        os.getenv("APEX_EARNINGS_PEAD_MIN_SURPRISE", "0.05")  # 5% EPS beat to trigger
    )
    EARNINGS_PEAD_CONF_BOOST: float = float(
        os.getenv("APEX_EARNINGS_PEAD_CONF_BOOST", "0.06")   # +6% conf when PEAD aligns with signal
    )
    EARNINGS_PEAD_CONTRA_PENALTY: float = float(
        os.getenv("APEX_EARNINGS_PEAD_CONTRA_PENALTY", "0.10")  # -10% conf when going against PEAD
    )
    EARNINGS_PEAD_BLOCK_THRESHOLD: float = float(
        os.getenv("APEX_EARNINGS_PEAD_BLOCK_THRESHOLD", "0.20")  # |surprise|>20% AND contra → block
    )

    # ── Put/Call Ratio signal ─────────────────────────────────────────────────
    PCR_GATE_ENABLED: bool = os.getenv("APEX_PCR_GATE_ENABLED", "true").lower() == "true"
    PCR_CACHE_TTL_SEC: int = int(os.getenv("APEX_PCR_CACHE_TTL", "3600"))
    PCR_ALIGN_BOOST: float = float(os.getenv("APEX_PCR_ALIGN_BOOST", "0.04"))
    PCR_CONTRA_PENALTY: float = float(os.getenv("APEX_PCR_CONTRA_PENALTY", "0.08"))

    # ── Opening Range Breakout signal ─────────────────────────────────────────
    ORB_GATE_ENABLED: bool = os.getenv("APEX_ORB_GATE_ENABLED", "true").lower() == "true"
    ORB_MIN_RVOL: float = float(os.getenv("APEX_ORB_MIN_RVOL", "1.20"))
    ORB_MIN_BREAKOUT_PCT: float = float(os.getenv("APEX_ORB_MIN_BREAKOUT_PCT", "0.003"))
    ORB_CONF_BOOST: float = float(os.getenv("APEX_ORB_CONF_BOOST", "0.07"))
    ORB_CONTRA_PENALTY: float = float(os.getenv("APEX_ORB_CONTRA_PENALTY", "0.10"))

    # Exit signal hysteresis (separate from entry threshold)
    SIGNAL_EXIT_BASE = 0.15

    # Regime-conditional consensus weights (per generator name)
    CONSENSUS_REGIME_WEIGHTS = {
        "bull": {"institutional": 1.1, "god_level": 1.0, "advanced": 0.9},
        "bear": {"institutional": 1.0, "god_level": 1.1, "advanced": 0.9},
        "neutral": {"institutional": 1.0, "god_level": 1.0, "advanced": 1.0},
        "volatile": {"institutional": 0.9, "god_level": 1.1, "advanced": 1.0},
    }

    # Model Weight Differentiation
    MODEL_WEIGHT_TEMPERATURE: float = 3.0       # Power scaling for inverse-MSE weights
    MODEL_WEIGHT_ACCURACY_BONUS: bool = True     # Include directional accuracy in weighting
    MODEL_OVERFIT_RATIO_THRESHOLD: float = 2.5   # val/train ratio gate (excludes GP at 2.63x)
    MODEL_OVERFIT_WARNING_RATIO: float = 2.0     # Log warning when val/train ratio exceeds this (G)

    # Signal Quality Filters (relaxed for more activity)
    MIN_MODEL_AGREEMENT = 0.50      # 50% agreement (majority rule)
    MIN_EXPECTED_RETURN = 0.003     # 0.3% expected return minimum
    VOLUME_CONFIRMATION = True      # Keep volume confirmation
    VOLUME_THRESHOLD_MULTIPLE = 0.6 # Volume at 60% of average

    # Multi-timeframe confirmation
    REQUIRE_MTF_CONFIRMATION = False  # Disabled - was killing too many valid signals
    MTF_AGREEMENT_THRESHOLD = 0.35   # Lower if re-enabled

    # VIX-based signal filtering
    VIX_FILTER_ENABLED = True
    VIX_NO_LONGS_ABOVE = 35         # Only block longs in extreme fear
    VIX_NO_SHORTS_BELOW = 10        # Only block shorts in extreme calm
    VIX_REDUCE_SIZE_ABOVE = 25      # Only reduce size in elevated VIX

    # ═══════════════════════════════════════════════════════════════
    # GOD LEVEL PARAMETERS (Moderate Risk Profile)
    # ═══════════════════════════════════════════════════════════════
    # Position Sizing (ATR-based)
    ATR_MULTIPLIER_STOP = 2.5  # Stop loss = ATR * this multiplier (wider room)
    ATR_MULTIPLIER_PROFIT = 4.0  # Take profit = ATR * this multiplier (let winners run)
    TRAILING_STOP_ATR = 2.5  # Trailing stop = ATR * this multiplier (avoid whipsaws)
    USE_KELLY_SIZING = True  # Use Kelly criterion for position sizing
    KELLY_FRACTION = 0.5  # Kelly fraction (50% - slightly more aggressive)
    KELLY_MAX_POSITION_PCT = 0.05  # Max 5% of capital per Kelly-sized position

    # Enable advanced risk features
    USE_ATR_STOPS = True  # Use dynamic ATR-based stops instead of fixed percentages
    USE_ADVANCED_EXECUTION = True  # Use TWAP/VWAP for large orders
    USE_CORRELATION_MANAGER = True  # Enable correlation monitoring

    # Market Regime
    REGIME_LOOKBACK_DAYS = 60  # Days to analyze for regime detection
    REGIME_BULL_THRESHOLD = 0.05  # MA crossover threshold for bull regime
    REGIME_BEAR_THRESHOLD = -0.05  # MA crossover threshold for bear regime
    HIGH_VOL_THRESHOLD = 0.35  # Annualized volatility threshold for high-vol regime

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL FORTRESS (Multi-Layer Signal Hardening)
    # ═══════════════════════════════════════════════════════════════

    # Signal Consensus Engine
    USE_CONSENSUS_ENGINE = True        # Run multiple generators and require agreement
    MIN_CONSENSUS_AGREEMENT = 0.60     # Min fraction of generators agreeing on direction
    MIN_CONVICTION_SCORE = 30          # Min conviction (0-100) to allow trade

    # Adaptive Regime Detection
    USE_ADAPTIVE_REGIME = True         # Use probability-based regime detector
    REGIME_SMOOTHING_ALPHA = 0.15      # EMA alpha for smooth regime transitions
    MIN_REGIME_DURATION_DAYS = 3       # Min days before allowing regime switch
    REGIME_MIN_MARGIN = 0.10           # Minimum probability margin required to switch regime

    # Signal Integrity Monitor
    SIGNAL_INTEGRITY_ENABLED = True    # Monitor signal stream for anomalies
    STUCK_SIGNAL_THRESHOLD = 30        # Alert after N identical signals (30×90s≈45min window for daily-bar models)
    KL_DIVERGENCE_THRESHOLD = 0.5      # Distribution shift detection sensitivity

    # Outcome Feedback Loop
    AUTO_RETRAIN_ENABLED = True        # Auto-retrain on performance degradation
    RETRAIN_ACCURACY_THRESHOLD = 0.45  # Retrain if accuracy drops below
    RETRAIN_SHARPE_THRESHOLD = 0.5     # Retrain if Sharpe drops below

    # Adaptive Thresholds
    ADAPTIVE_THRESHOLDS_ENABLED = True # Per-symbol threshold optimization
    THRESHOLD_MIN_SIGNALS = 30         # Min signals before symbol optimization
    THRESHOLD_OPTIMIZATION_INTERVAL_HOURS = 24  # Hours between re-optimization

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL FORTRESS PHASE 2 (Indestructible Shield)
    # ═══════════════════════════════════════════════════════════════

    # Black Swan Guard - Real-time crash detection
    BLACK_SWAN_GUARD_ENABLED = True
    CRASH_VELOCITY_THRESHOLD_10M = 0.02   # 2% drop in 10 min = ELEVATED
    CRASH_VELOCITY_THRESHOLD_30M = 0.04   # 4% drop in 30 min = SEVERE
    VIX_SPIKE_ELEVATED = 0.30             # 30% VIX increase = ELEVATED
    VIX_SPIKE_SEVERE = 0.50               # 50% VIX increase = SEVERE
    CORRELATION_CRISIS_THRESHOLD = 0.85   # Avg correlation for crisis

    # Signal Decay Shield - Time-decay & staleness guard
    SIGNAL_DECAY_ENABLED = True
    PRE_MARKET_STALENESS_GUARD = True     # Guard against stale overnight pre-market gaps
    MAX_PRICE_AGE_SECONDS = 120           # 2 minutes max for price data
    MAX_SENTIMENT_AGE_SECONDS = 1800      # 30 minutes max for sentiment
    MAX_FEATURE_AGE_SECONDS = 14400       # 4 hours max for features

    # Exit Quality Guard - Exit signal validation
    EXIT_QUALITY_GUARD_ENABLED = True
    EXIT_MIN_CONFIDENCE = 0.55            # Min confidence for signal-based exits
    EXIT_MAX_RETRY_ATTEMPTS = 20          # Never give up (was 5)
    EXIT_BACKOFF_BASE_SECONDS = 30        # Exponential backoff base
    EXIT_HARD_STOP_PNL = -0.03            # -3% hard stop bypasses validation

    # Correlation Cascade Breaker - Portfolio correlation shield
    CORRELATION_CASCADE_ENABLED = True
    CORRELATION_ELEVATED_THRESHOLD = 0.40
    CORRELATION_HERDING_THRESHOLD = 0.60
    CORRELATION_CRISIS_THRESHOLD_PORT = 0.80

    # Drawdown Cascade Breaker - 5-tier drawdown response
    DRAWDOWN_CASCADE_ENABLED = True
    DRAWDOWN_TIER_1 = 0.02               # 2% = Caution (75% sizing)
    DRAWDOWN_TIER_2 = 0.04               # 4% = Defensive (50% sizing)
    DRAWDOWN_TIER_3 = 0.06               # 6% = Survival (25% sizing)
    DRAWDOWN_TIER_4 = 0.08               # 8% = Emergency (close all)
    DRAWDOWN_VELOCITY_JUMP = 0.01        # 1%/day = jump up one tier

    # Execution Shield - Smart execution wrapper
    EXECUTION_SHIELD_ENABLED = True
    EXECUTION_TWAP_THRESHOLD = 50000     # $50K+ uses TWAP
    EXECUTION_VWAP_THRESHOLD = 200000    # $200K+ uses VWAP
    MAX_ACCEPTABLE_SLIPPAGE_BPS = 12     # Flag symbols with avg slippage > 12bps
    CRITICAL_SLIPPAGE_BPS = 20           # Reduce size 20% above this (lowered from 30)
    EXECUTION_SLIPPAGE_BUDGET_BPS: float = float(
        os.getenv("APEX_EXECUTION_SLIPPAGE_BUDGET_BPS", "250")
    )
    # Tuned from recent live fill logs (crypto-heavy sample) with conservative priors.
    EXECUTION_SLIPPAGE_BUDGET_BPS_EQUITY: float = float(
        os.getenv("APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_EQUITY", "221")
    )
    EXECUTION_SLIPPAGE_BUDGET_BPS_FX: float = float(
        os.getenv("APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_FX", "180")
    )
    EXECUTION_SLIPPAGE_BUDGET_BPS_CRYPTO: float = float(
        os.getenv("APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_CRYPTO", "156")
    )
    EXECUTION_SLIPPAGE_BUDGET_WINDOW: int = int(
        os.getenv("APEX_EXECUTION_SLIPPAGE_BUDGET_WINDOW", "20")
    )
    EXECUTION_MAX_SPREAD_BPS_EQUITY: float = float(
        os.getenv("APEX_EXECUTION_MAX_SPREAD_BPS_EQUITY", "12")
    )
    EXECUTION_MAX_SPREAD_BPS_FX: float = float(
        os.getenv("APEX_EXECUTION_MAX_SPREAD_BPS_FX", "8")
    )
    EXECUTION_MAX_SPREAD_BPS_CRYPTO: float = float(
        os.getenv("APEX_EXECUTION_MAX_SPREAD_BPS_CRYPTO", "50")   # Tightened from 80: exclude chronic wide-spread alts
    )
    EXECUTION_EDGE_GATE_ENABLED: bool = os.getenv(
        "APEX_EXECUTION_EDGE_GATE_ENABLED",
        "true",
    ).strip().lower() == "true"
    EXECUTION_SIGNAL_TO_EDGE_BPS: float = float(
        os.getenv("APEX_EXECUTION_SIGNAL_TO_EDGE_BPS", "80")
    )
    # Crypto: higher multiplier reflects crypto vol ~5-10%/day vs 1-2% equity.
    # At 400, a 0.10 signal estimates ~40bps edge, enough to clear a 15+15+15=45bps hurdle.
    EXECUTION_SIGNAL_TO_EDGE_BPS_CRYPTO: float = float(
        os.getenv("APEX_EXECUTION_SIGNAL_TO_EDGE_BPS_CRYPTO", "400")  # Raised from 300
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY", "12")  # Raised from 8
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_FX: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_FX", "999")  # FX disabled — belt+suspenders
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO", "15")  # Raised from 4: need real edge
    )

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL FORTRESS PHASE 4 (Macro & Event Risk)
    # ═══════════════════════════════════════════════════════════════
    
    # Macro Shield - Economic Event Protection
    MACRO_SHIELD_ENABLED = True
    MACRO_BLACKOUT_MINUTES_BEFORE = 60    # Stop entries 60 min before event
    MACRO_BLACKOUT_MINUTES_AFTER = 30     # Resume 30 min after event
    GAME_OVER_LOSS_THRESHOLD = 0.05       # 5% daily loss = Immediate Shutdown

    # ═══════════════════════════════════════════════════════════════
    # SIGNAL FORTRESS PHASE 3 (Autonomous Money Machine)
    # ═══════════════════════════════════════════════════════════════

    # Overnight Risk Guard - Gap protection
    OVERNIGHT_GUARD_ENABLED = True
    OVERNIGHT_NO_ENTRY_MINUTES = 30       # No entries in last 30 min
    OVERNIGHT_REDUCE_MINUTES = 60         # Start reducing exposure in last 60 min
    OVERNIGHT_MAX_VAR_PCT = 2.0           # Max 2% overnight VaR
    OVERNIGHT_HIGH_VIX_REDUCTION = 0.20   # 20% additional reduction when VIX > 25

    # Profit Ratchet - Progressive trailing stops
    PROFIT_RATCHET_ENABLED = True
    PROFIT_TIER_1 = 0.02                  # 2% gain = lock 50%
    PROFIT_TIER_2 = 0.05                  # 5% gain = lock 70%
    PROFIT_TIER_3 = 0.10                  # 10% gain = lock 80%
    PROFIT_TIER_4 = 0.20                  # 20% gain = lock 85%
    PROFIT_INITIAL_TRAILING = 0.03        # 3% initial trailing stop

    # Liquidity Guard - Illiquid condition detection
    LIQUIDITY_GUARD_ENABLED = True
    LIQUIDITY_THIN_SPREAD = 0.001         # 0.1% spread = THIN
    LIQUIDITY_STRESSED_SPREAD = 0.003     # 0.3% spread = STRESSED
    LIQUIDITY_CRISIS_SPREAD = 0.005       # 0.5% spread = CRISIS
    LIQUIDITY_MIN_DOLLAR_VOLUME = 1000000 # $1M min daily volume

    # Position Aging Manager - Time-based exits
    AGING_MANAGER_ENABLED = True
    AGING_MAX_DAYS = 30                   # Force exit after 30 days
    AGING_STALE_DAYS = 15                 # Position becomes stale
    AGING_CRITICAL_DAYS = 20              # Position becomes critical
    AGING_STALE_MIN_PNL = 0.0             # Stale positions must be profitable
    AGING_CRITICAL_MIN_PNL = 0.02         # Critical positions need 2%+ P&L

    # ═══════════════════════════════════════════════════════════════
    # OPTIONS TRADING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    OPTIONS_ENABLED = True  # Enable options trading
    OPTIONS_RISK_FREE_RATE = 0.05  # Risk-free rate for Black-Scholes pricing

    # Options Strategy Settings
    OPTIONS_AUTO_HEDGE = True  # Automatically hedge large stock positions with puts
    OPTIONS_HEDGE_THRESHOLD = 10000  # Hedge positions worth more than $10,000
    OPTIONS_HEDGE_DELTA = -0.30  # Target delta for protective puts (-0.30 = 30-delta)

    # Covered Calls
    OPTIONS_COVERED_CALLS_ENABLED = True  # Sell covered calls on long positions
    OPTIONS_COVERED_CALL_DELTA = 0.30  # Target delta for covered calls (30-delta)
    OPTIONS_MIN_SHARES_FOR_COVERED_CALL = 100  # Minimum shares to write covered call

    # Options Expiration Preferences
    OPTIONS_MIN_DAYS_TO_EXPIRY = 14  # Minimum days to expiration
    OPTIONS_MAX_DAYS_TO_EXPIRY = 45  # Maximum days to expiration
    OPTIONS_PREFERRED_DAYS_TO_EXPIRY = 30  # Preferred days to expiration

    # Options Risk Limits
    OPTIONS_MAX_PORTFOLIO_DELTA = 1000  # Maximum portfolio delta exposure
    OPTIONS_MAX_PORTFOLIO_GAMMA = 500  # Maximum portfolio gamma exposure
    OPTIONS_MAX_OPTIONS_CAPITAL_PCT = 0.10  # Max 10% of capital in options premiums
    OPTIONS_MAX_SINGLE_POSITION_PCT = 0.02  # Max 2% of capital per option position
    OPTIONS_FAILED_TRADE_RETRY_COOLDOWN_SECONDS: int = int(
        os.getenv("APEX_OPTIONS_FAILED_TRADE_RETRY_COOLDOWN_SECONDS", "1800")
    )

    # Correlation Management
    MAX_CORRELATION = 0.70  # Max correlation between positions
    MAX_PORTFOLIO_CORRELATION = 0.50  # Max average portfolio correlation (equities)
    # Crypto: all cryptos are inherently correlated; use a much higher cap
    MAX_PORTFOLIO_CORRELATION_CRYPTO = 0.92  # Max avg portfolio correlation for crypto
    CORRELATION_LOOKBACK = 60  # Days for correlation calculation
    # Graduated entry size reduction (below the hard-block threshold)
    CORRELATION_SIZE_WARN_LO: float = 0.70   # avg_corr ≥ this → 75% size
    CORRELATION_SIZE_WARN_HI: float = 0.80   # avg_corr ≥ this → 50% size
    # TP Laddering: partial exits as position becomes profitable.
    # Lowered from 6%/12% to 3%/6%: with 0% win rate, capturing smaller gains
    # is more important than waiting for large moves that never materialise.
    # Tier 1 at 3% locks in 50% of position; Tier 2 at 6% takes another 25%.
    TP_LADDER_TIER1_PCT:  float = 3.0   # pnl_pct threshold to fire Tier 1 (50% exit)
    TP_LADDER_TIER2_PCT:  float = 6.0   # pnl_pct threshold to fire Tier 2 (25% exit)
    TP_LADDER_TIER1_FRAC: float = 0.50  # fraction of position to sell at Tier 1
    TP_LADDER_TIER2_FRAC: float = 0.25  # fraction of position to sell at Tier 2

    # Hedge force-exit guard thresholds (Fix 2+3 from 2026-03-19 audit fixes)
    # Minimum hold time before hedge portfolio-alarm path can force-exit a position.
    # Prevents immediate re-exit of freshly entered positions on portfolio alarm days.
    HEDGE_FORCE_EXIT_MIN_HOLD_MINUTES: float = 15.0
    # Minimum per-position loss before portfolio-alarm force-exit fires.
    # Stops exiting positions that are slightly negative (-0.1%) during alarm days.
    HEDGE_FORCE_EXIT_MIN_POS_LOSS_PCT: float = 1.0

    # Execution
    LARGE_ORDER_THRESHOLD = 50000  # Dollar value threshold for TWAP/VWAP execution
    USE_LIVE_MARKET_DATA = False  # Set True for live trading with data subscription

    # Advanced Risk
    VAR_CONFIDENCE = 0.95  # VaR confidence level
    MAX_PORTFOLIO_VAR = 0.03  # Maximum daily VaR (3%) — equities
    # Crypto assets have 2-4× equity vol; dedicated Alpaca session uses a
    # higher VaR budget so a single BTC position doesn't block all other entries.
    CRYPTO_MAX_PORTFOLIO_VAR: float = float(
        os.getenv("APEX_CRYPTO_MAX_PORTFOLIO_VAR", "0.06")
    )  # 6% crypto VaR limit
    DRAWDOWN_REDUCE_THRESHOLD = 0.05  # Reduce position size after 5% drawdown
    DRAWDOWN_HALT_THRESHOLD = 0.10  # Halt new trades after 10% drawdown
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING HOURS (EST)
    # ═══════════════════════════════════════════════════════════════
    TRADING_HOURS_START = 9.5  # 9:30 AM EST (market open)
    TRADING_HOURS_END = 16.0   # 4:00 PM EST (market close)

    # Extended/Pre-market hours for ETFs (some commodity ETFs)
    EXTENDED_HOURS_START = 4.0   # 4:00 AM EST (pre-market)
    EXTENDED_HOURS_END = 20.0    # 8:00 PM EST (after-hours)

    # Commodity ETFs trading hours (same as stocks, but some have extended)
    # GLD, SLV, USO, UNG, PALL - trade regular + extended hours on some brokers
    COMMODITY_ETF_USE_EXTENDED = True  # Trade commodities in extended hours (4AM-8PM EST)
    
    # ═══════════════════════════════════════════════════════════════
    # BROKER FAILOVER (A)
    # ═══════════════════════════════════════════════════════════════
    IBKR_FAILOVER_ENABLED: bool = True          # Degrade to Alpaca-only on persistent IBKR outage
    IBKR_FAILOVER_MAX_RETRIES: int = 3          # Reconnect attempts before declaring IBKR down
    IBKR_FAILOVER_RETRY_SECONDS: float = 30.0   # Seconds between reconnect retries after failover
    IBKR_RECOVERY_INTERVAL_SECONDS: float = 300.0  # Background recovery loop interval after persistent down
    # WSS hit-rate threshold below which a WARNING is emitted every 100 cycles
    WSS_HIT_RATE_WARN_THRESHOLD: float = float(os.getenv("APEX_WSS_HIT_RATE_WARN_THRESHOLD", "0.50"))

    # ═══════════════════════════════════════════════════════════════
    # LIQUIDITY GATE (D)
    # ═══════════════════════════════════════════════════════════════
    LIQUIDITY_SPREAD_MAX_BPS: float = 100.0     # Skip crypto entries if spread > 100bps (1%)
    LIQUIDITY_GATE_ENABLED: bool = True         # Enable pre-trade spread check

    # ═══════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER AUTO-RESET (E)
    # ═══════════════════════════════════════════════════════════════
    CIRCUIT_BREAKER_AUTO_RESET: bool = True     # Auto-reset circuit breaker after cooldown

    # ═══════════════════════════════════════════════════════════════
    # ASYMMETRIC EXIT SIZING (2:1+ R:R, let winners run)
    # ═══════════════════════════════════════════════════════════════
    ASYMMETRIC_SIZING_ENABLED: bool = True
    ASYM_ATR_STOP_MULT: float = 1.5          # stop = ATR × 1.5
    ASYM_ATR_STOP_MULT_CRYPTO: float = 2.0   # wider for crypto volatility
    ASYM_RR_BASE: float = 2.0                # minimum 2:1 R:R
    ASYM_RR_CONF_SCALE: float = 1.5          # conf=1.0 → 3.5:1 R:R
    ASYM_TRAIL_ACTIVATION_FRAC: float = 0.55 # trailing only after 55% of TP distance
    ASYM_TRAIL_DIST_MULT: float = 1.2        # trailing = 1.2× stop distance
    ASYM_BREAKEVEN_LOCK_FRAC: float = 1.0    # lock at breakeven when PnL > 1× stop dist
    ASYM_HOLD_SIGNAL_THRESH_HIGH: float = -0.25  # high-conv: exit only on strong reversal
    ASYM_HOLD_SIGNAL_THRESH_BASE: float = -0.10  # base: exit on mild reversal
    ASYM_HIGH_CONF_THRESHOLD: float = 0.68   # confidence above this = high conviction
    ASYM_REGIME_TP_MULT: dict = {            # TP scaling per regime
        "strong_bull": 1.20, "bull": 1.10, "neutral": 1.00,
        "bear": 0.80, "strong_bear": 0.70, "volatile": 0.85, "crisis": 0.60,
    }

    # ═══════════════════════════════════════════════════════════════
    # CORRELATED EXIT STAGGER (F)
    # ═══════════════════════════════════════════════════════════════
    CORR_EXIT_STAGGER_ENABLED: bool = True      # Stagger simultaneous correlated exits
    CORR_EXIT_STAGGER_MAX_PER_CYCLE: int = 2    # Max exits per cycle when correlation is high
    CORR_EXIT_STAGGER_THRESHOLD: float = 0.70   # Pearson r above this triggers staggering

    # ═══════════════════════════════════════════════════════════════
    # CORRELATION ENTRY GATE
    # ═══════════════════════════════════════════════════════════════
    CORR_ENTRY_GATE_ENABLED: bool = True
    CORR_ENTRY_MAX_CORR: float = 0.65    # hard block when any open-pos corr >= this
    CORR_ENTRY_SOFT_CORR: float = 0.50   # soft warn + confidence penalty
    CORR_ENTRY_CONF_PENALTY: float = 0.05
    CORR_ENTRY_LOOKBACK: int = 30        # bars of returns to compare
    CORR_ENTRY_MIN_BARS: int = 10        # minimum bars required to run the gate

    # ═══════════════════════════════════════════════════════════════
    # ROLLING STRATEGY HEALTH MONITOR
    # ═══════════════════════════════════════════════════════════════
    STRATEGY_HEALTH_ENABLED: bool = True
    STRATEGY_HEALTH_LOOKBACK_DAYS: int = 30
    STRATEGY_HEALTH_MIN_TRADES: int = 5
    STRATEGY_HEALTH_PAPER_THRESHOLD: float = 0.0    # Sharpe < this → paper-only
    STRATEGY_HEALTH_RECOVER_THRESHOLD: float = 0.30  # Sharpe > this → back to live
    STRATEGY_HEALTH_PERSIST_PATH: str = "data/strategy_health_state.json"

    # ═══════════════════════════════════════════════════════════════
    # BEAR / NEUTRAL REGIME STRATEGIES
    # ═══════════════════════════════════════════════════════════════
    BEAR_MR_ENABLED: bool = True
    BEAR_MR_BLEND_WEIGHT: float = 0.10          # mean-reversion blend weight
    BEAR_MR_RSI_OVERSOLD: int = 32
    BEAR_MR_RSI_OVERBOUGHT: int = 68
    BEAR_MR_VWAP_ZSCORE_THRESH: float = 1.5
    BEAR_MR_ACTIVE_REGIMES: list = ["bear", "strong_bear", "neutral", "volatile"]
    SECTOR_ROTATION_ENABLED: bool = True
    SECTOR_ROTATION_BLEND_WEIGHT: float = 0.08  # sector rotation blend weight
    SECTOR_ROTATION_LOOKBACK: int = 20
    SECTOR_ROTATION_TOP_N: int = 3
    SECTOR_ETFS: list = [
        "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLC", "XLU", "XLB", "XLRE", "XLP", "XLY",
    ]

    # ═══════════════════════════════════════════════════════════════
    # FACTOR IC TRACKER
    # ═══════════════════════════════════════════════════════════════
    FACTOR_IC_ENABLED: bool = True
    FACTOR_IC_WINDOW: int = 50        # rolling window (trades) for IC calc
    FACTOR_IC_MIN_OBS: int = 10       # min observations to report IC
    FACTOR_IC_PERSIST_PATH: str = "data/factor_ic_state.json"

    # ═══════════════════════════════════════════════════════════════
    # ADAPTIVE SIGNAL WEIGHT MANAGER
    # ═══════════════════════════════════════════════════════════════
    ADAPTIVE_WEIGHTS_ENABLED: bool = True
    ADAPTIVE_WEIGHTS_MIN_MULT: float = 0.30   # floor: 30% of base weight
    ADAPTIVE_WEIGHTS_MAX_MULT: float = 2.50   # ceiling: 250% of base weight
    ADAPTIVE_WEIGHTS_EMA_ALPHA: float = 0.25  # EMA smoothing coefficient
    ADAPTIVE_WEIGHTS_IC_K: float = 4.0        # sigmoid IC→multiplier sensitivity
    ADAPTIVE_WEIGHTS_INTERVAL: int = 100      # update every N main-loop cycles
    ADAPTIVE_WEIGHTS_PERSIST_PATH: str = "data/adaptive_weights.json"

    # ── Nightly Backtest Gate ──────────────────────────────────────
    BACKTEST_GATE_ENABLED: bool = True
    BACKTEST_GATE_WINDOW_DAYS: int = 30
    BACKTEST_GATE_COMPARE_DAYS: int = 15
    BACKTEST_GATE_MIN_TRADES: int = 10
    BACKTEST_GATE_SHARPE_DEGRADE: float = 0.30
    BACKTEST_GATE_WIN_RATE_FLOOR: float = 0.40
    BACKTEST_GATE_CONSEC_DEGRADE: int = 2
    BACKTEST_GATE_CONSEC_RECOVER: int = 2
    BACKTEST_GATE_STATE_PATH: str = "data/backtest_gate_state.json"
    BACKTEST_GATE_HISTORY_LIMIT: int = 30

    # ── Daily Briefing ────────────────────────────────────────────
    DAILY_BRIEFING_ENABLED: bool = True
    DAILY_BRIEFING_TELEGRAM_ENABLED: bool = False
    DAILY_BRIEFING_TELEGRAM_TOKEN: str = ""
    DAILY_BRIEFING_TELEGRAM_CHAT_ID: str = ""
    DAILY_BRIEFING_OUTPUT_DIR: str = "data/daily_briefings"
    DAILY_BRIEFING_KEEP_DAYS: int = 30

    # ── Short Selling Gate ────────────────────────────────────────
    CRYPTO_ALLOW_SHORTS: bool = os.getenv("APEX_CRYPTO_ALLOW_SHORTS", "true").lower() == "true"
    SHORT_SELLING_ENABLED: bool = True
    SHORT_ACTIVE_REGIMES: list = ["bear", "strong_bear", "neutral", "volatile"]
    SHORT_MAX_POSITIONS: int = 3
    SHORT_MAX_NOTIONAL: float = 25000.0
    SHORT_MAX_TOTAL_NOTIONAL: float = 60000.0
    SHORT_MIN_PRICE: float = 10.0
    SHORT_VIX_BLOCK: float = 35.0
    SHORT_SIGNAL_FLOOR: float = 0.12
    SHORT_CONFIDENCE_FLOOR: float = 0.55
    SHORT_LARGE_CAP_ONLY: bool = False

    # ── Order Flow Imbalance Signal ───────────────────────────────
    OFI_ENABLED: bool = True
    OFI_WINDOW: int = 20
    OFI_BLEND_WEIGHT: float = 0.06
    OFI_MIN_VOLUME: int = 1000
    OFI_SMOOTHING_ALPHA: float = 0.30
    OFI_CACHE_TTL_SECONDS: int = 30

    # ── Black-Litterman Portfolio Allocation ──────────────────────
    BL_ENABLED: bool = True
    BL_TAU: float = 0.05
    BL_RISK_AVERSION: float = 2.5
    BL_VIEW_CONFIDENCE: float = 0.75
    BL_MIN_IC_FOR_VIEW: float = 0.10
    BL_MIN_OBS_FOR_VIEW: int = 10
    BL_WEIGHT_FLOOR: float = 0.0
    BL_WEIGHT_CAP: float = 0.25
    BL_REFRESH_INTERVAL: int = 50
    BL_PERSIST_PATH: str = "data/bl_weights.json"
    BL_LOOKBACK_BARS: int = 60

    # ── Correlation Matrix Regime Detector ────────────────────────
    CORR_REGIME_ENABLED: bool = True
    CORR_REGIME_LOOKBACK: int = 30
    CORR_REGIME_WARNING_THRESHOLD: float = 0.55
    CORR_REGIME_CRISIS_THRESHOLD: float = 0.75
    CORR_REGIME_MIN_SYMBOLS: int = 4
    CORR_REGIME_ELEVATED_MULT: float = 0.75
    CORR_REGIME_CRISIS_MULT: float = 0.50
    CORR_REGIME_UPDATE_INTERVAL: int = 10

    # ── Portfolio Volatility Targeting ────────────────────────────
    VOL_TARGET_ENABLED: bool = True
    VOL_TARGET_ANNUALISED: float = 0.12         # 12% annual portfolio vol
    VOL_TARGET_LOOKBACK_DAYS: int = 30
    VOL_TARGET_MIN_MULT: float = 0.50
    VOL_TARGET_MAX_MULT: float = 1.50
    VOL_TARGET_MIN_OBS: int = 10
    VOL_TARGET_BARS_PER_YEAR: int = 252

    # ── Crypto Liquidation Cascade Monitor ────────────────────────
    LIQUIDATION_MONITOR_ENABLED: bool = True
    LIQUIDATION_MONITOR_FUNDING_EXTREME: float = 0.10   # % per 8h (0.10 = 0.10%)
    LIQUIDATION_MONITOR_OI_DROP_WARNING: float = 0.05
    LIQUIDATION_MONITOR_OI_DROP_CRITICAL: float = 0.10
    LIQUIDATION_MONITOR_CACHE_TTL_SECONDS: int = 120
    LIQUIDATION_MONITOR_SIZING_MULT_FLOOR: float = 0.50

    # ── IV Skew Signal ─────────────────────────────────────────────
    IV_SKEW_ENABLED: bool = True
    IV_SKEW_PUT_CALL_WEIGHT: float = 0.60
    IV_SKEW_VIX_TERM_WEIGHT: float = 0.40
    IV_SKEW_CACHE_TTL_SECONDS: int = 600
    IV_SKEW_BLEND_WEIGHT: float = 0.05
    IV_SKEW_STRIKES_AROUND_ATM: int = 3

    # ── Macro Cross-Asset Signal ───────────────────────────────────
    MACRO_ENABLED: bool = True
    MACRO_VIX_WEIGHT: float = 0.40
    MACRO_YIELD_WEIGHT: float = 0.35
    MACRO_DXY_WEIGHT: float = 0.25
    MACRO_VIX_LOOKBACK: int = 10
    MACRO_YIELD_LOOKBACK: int = 20
    MACRO_DXY_LOOKBACK: int = 20
    MACRO_CACHE_TTL_SECONDS: int = 300
    MACRO_BLEND_WEIGHT: float = 0.08

    # ═══════════════════════════════════════════════════════════════
    # AUDIT LOGS (H, I)
    # ═══════════════════════════════════════════════════════════════
    TRADE_REJECTION_AUDIT_ENABLED: bool = True  # Write rejected signals to JSONL (H)
    DEAD_LETTER_QUEUE_ENABLED: bool = True       # Write failed orders to JSONL (I)

    # PROFIT RATCHET (AG)
    # ═══════════════════════════════════════════════════════════════
    # When equity grows > PROFIT_RATCHET_TRIGGER above peak, lock in a fraction of gains
    # and reduce max position size temporarily until equity makes a new high.
    PROFIT_RATCHET_ENABLED: bool = True
    PROFIT_RATCHET_TRIGGER_PCT: float = float(os.getenv("APEX_PROFIT_RATCHET_TRIGGER_PCT", "10.0"))
    PROFIT_RATCHET_LOCK_PCT: float = float(os.getenv("APEX_PROFIT_RATCHET_LOCK_PCT", "5.0"))
    PROFIT_RATCHET_SIZE_SCALE: float = float(os.getenv("APEX_PROFIT_RATCHET_SIZE_SCALE", "0.6"))  # 60% of normal size

    # EXECUTION LATENCY AUDIT (P)
    # ═══════════════════════════════════════════════════════════════
    EXECUTION_LATENCY_AUDIT_ENABLED: bool = True

    # ═══════════════════════════════════════════════════════════════
    # TIMING & EXECUTION
    # ═══════════════════════════════════════════════════════════════
    CHECK_INTERVAL_SECONDS = 30  # Check symbols every 30 seconds
    TRADE_COOLDOWN_SECONDS = 300  # 5 min minimum between same-symbol revisits (was 120s — too short, caused churn)
    POLL_INTERVAL_SECONDS: float = float(_env_default("APEX_POLL_INTERVAL_SECONDS", 1.0, 3.0))
    PUBLIC_WS_POLL_INTERVAL_SECONDS: float = float(
        _env_default("APEX_PUBLIC_WS_POLL_INTERVAL_SECONDS", 1.0, 5.0)
    )
    RETRAIN_INTERVAL_SECONDS: int = int(
        _env_default("APEX_RETRAIN_INTERVAL_SECONDS", 24 * 60 * 60, 7 * 24 * 60 * 60)
    )
    HOT_PATH_PROFILING_ENABLED: bool = os.getenv(
        "APEX_HOT_PATH_PROFILING_ENABLED",
        "true" if IS_DEVELOPMENT else "false",
    ).lower() == "true"
    HOT_PATH_PROFILING_THRESHOLD_MS: float = float(
        os.getenv("APEX_HOT_PATH_PROFILING_THRESHOLD_MS", "250.0")
    )
    HOT_PATH_PROFILING_SUMMARY_INTERVAL_CYCLES: int = int(
        os.getenv("APEX_HOT_PATH_PROFILING_SUMMARY_INTERVAL_CYCLES", "20")
    )
    
    # ═══════════════════════════════════════════════════════════════
    # TRANSACTION COSTS
    # ═══════════════════════════════════════════════════════════════
    COMMISSION_PER_TRADE = 1.00  # ✅ NEW: $1 per trade (IBKR Pro)
    SLIPPAGE_BPS = 5  # 5 basis points slippage (0.05%)

    # Advanced model training controls
    ADV_LABEL_VOL_LOOKBACK = int(os.getenv("APEX_ADV_LABEL_VOL_LOOKBACK", "20"))
    ADV_LABEL_VOL_CLIP = float(os.getenv("APEX_ADV_LABEL_VOL_CLIP", "6.0"))
    ADV_CROSS_SECTIONAL_NORM = os.getenv("APEX_ADV_CROSS_SECTIONAL_NORM", "true").lower() == "true"
    ADV_PURGE_DAYS = int(os.getenv("APEX_ADV_PURGE_DAYS", "5"))
    ADV_EMBARGO_DAYS = int(os.getenv("APEX_ADV_EMBARGO_DAYS", "2"))
    
    # ═══════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════
    LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = "logs/apex.log"
    # ✅ NEW: Log rotation controls (bytes)
    LOG_MAX_BYTES = int(os.getenv("APEX_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5MB
    LOG_BACKUP_COUNT = int(os.getenv("APEX_LOG_BACKUP_COUNT", "5"))
    TERMINAL_OBSERVABILITY_ENABLED: bool = (
        os.getenv("APEX_TERMINAL_OBSERVABILITY_ENABLED", "true").lower() == "true"
    )
    TERMINAL_METRICS_INTERVAL_SECONDS: float = float(
        os.getenv("APEX_TERMINAL_METRICS_INTERVAL_SECONDS", "60.0")
    )

    # Health check staleness (seconds). Mark backend offline if state is older than this.
    HEALTH_STALENESS_SECONDS = int(os.getenv("APEX_HEALTH_STALENESS_SECONDS", "30"))
    
    # ═══════════════════════════════════════════════════════════════
    # DUAL-SESSION MODE (Core Strategy + Crypto Sleeve)
    # ═══════════════════════════════════════════════════════════════
    SESSION_MODE: str = os.getenv("APEX_SESSION_MODE", "dual")  # "unified", "dual", "core_only", "crypto_only"
    CRYPTO_SLEEVE_ENABLED: bool = os.getenv("APEX_CRYPTO_SLEEVE_ENABLED", "true").lower() == "true"

    # Per-session capital allocation (must sum to INITIAL_CAPITAL)
    CORE_INITIAL_CAPITAL: int = int(os.getenv("APEX_CORE_INITIAL_CAPITAL", "1000000"))
    CRYPTO_INITIAL_CAPITAL: int = int(os.getenv("APEX_CRYPTO_INITIAL_CAPITAL", "100000"))

    # Per-session signal thresholds (tuned independently for higher Sharpe)
    CORE_MIN_SIGNAL_THRESHOLD: float = float(os.getenv("APEX_CORE_MIN_SIGNAL_THRESHOLD", "0.10"))
    CORE_MIN_CONFIDENCE: float = float(os.getenv("APEX_CORE_MIN_CONFIDENCE", "0.45"))
    CRYPTO_MIN_SIGNAL_THRESHOLD: float = float(os.getenv("APEX_CRYPTO_MIN_SIGNAL_THRESHOLD", "0.08"))
    CRYPTO_MIN_CONFIDENCE: float = float(os.getenv("APEX_CRYPTO_MIN_CONFIDENCE", "0.50"))
    # Hedge dampener: equity VIX/correlation thresholds are too aggressive for
    # crypto's higher natural volatility.  This floor prevents the HedgeManager
    # from choking crypto signals below 50% conviction during VIX 20-27 range.
    CRYPTO_HEDGE_DAMPENER_FLOOR: float = float(os.getenv("APEX_CRYPTO_HEDGE_DAMPENER_FLOOR", "0.50"))
    # Kelly cold-start: use this win-rate floor when live trade count is < 20
    # so early losses don't push Kelly to zero and freeze position sizing.
    KELLY_COLD_START_WIN_RATE_FLOOR: float = float(os.getenv("APEX_KELLY_COLD_START_WIN_RATE_FLOOR", "0.40"))

    # ── Macro Indicators ──────────────────────────────────────────────────────
    MACRO_INDICATORS_ENABLED: bool = os.getenv("APEX_MACRO_INDICATORS_ENABLED", "true").lower() == "true"
    # Equity size multiplier when yield curve is inverted (10Y < 2Y)
    MACRO_YIELD_CURVE_INVERSION_SIZE_MULT: float = float(
        os.getenv("APEX_MACRO_YIELD_CURVE_INVERSION_SIZE_MULT", "0.75")
    )
    # All-entry dampener when VIX futures are in backwardation (stress regime)
    MACRO_VIX_BACKWARDATION_DAMPENER: float = float(
        os.getenv("APEX_MACRO_VIX_BACKWARDATION_DAMPENER", "0.80")
    )

    # ── News Confirmation Gate ────────────────────────────────────────────────
    NEWS_CONFIRMATION_GATE_ENABLED: bool = os.getenv("APEX_NEWS_CONFIRMATION_GATE_ENABLED", "true").lower() == "true"
    # |sentiment| must exceed this to be considered a strong contradiction
    NEWS_STRONG_CONTRADICTION_THRESHOLD: float = float(
        os.getenv("APEX_NEWS_STRONG_CONTRADICTION_THRESHOLD", "0.40")
    )
    # Confidence below this + strong contradiction → block entry entirely
    NEWS_CONTRADICTION_MIN_CONFIDENCE: float = float(
        os.getenv("APEX_NEWS_CONTRADICTION_MIN_CONFIDENCE", "0.70")
    )

    # ── VWAP Deviation Gate ───────────────────────────────────────────────────
    VWAP_GATE_ENABLED: bool = os.getenv("APEX_VWAP_GATE_ENABLED", "true").lower() == "true"
    # Baseline max % deviation from 20d VWAP before blocking entry
    VWAP_MAX_DEVIATION_PCT: float = float(os.getenv("APEX_VWAP_MAX_DEVIATION_PCT", "2.0"))
    # Scale threshold with ATR so high-vol names aren't unfairly penalised
    VWAP_ATR_ADJUST: bool = os.getenv("APEX_VWAP_ATR_ADJUST", "true").lower() == "true"

    # ── RVOL (Relative Volume) Gate ───────────────────────────────────────────
    RVOL_GATE_ENABLED: bool = os.getenv("APEX_RVOL_GATE_ENABLED", "true").lower() == "true"
    # Block entries when today's volume < this fraction of 20-day average
    RVOL_MIN_THRESHOLD: float = float(os.getenv("APEX_RVOL_MIN_THRESHOLD", "0.30"))

    # ── Statistical (Hurst / OU) Features ────────────────────────────────────
    HURST_FEATURE_ENABLED: bool = os.getenv("APEX_HURST_FEATURE_ENABLED", "true").lower() == "true"
    HURST_LAGS: int = int(os.getenv("APEX_HURST_LAGS", "20"))

    # ── Binary Direction Classifier ───────────────────────────────────────────
    # Trains alongside regression ensemble; signal = prob_up * 2 - 1 ∈ [-1, 1]
    BINARY_SIGNAL_ENABLED: bool = os.getenv("APEX_BINARY_SIGNAL_ENABLED", "true").lower() == "true"
    BINARY_SIGNAL_WEIGHT: float = float(os.getenv("APEX_BINARY_SIGNAL_WEIGHT", "0.40"))
    BINARY_LABEL_HORIZON_DAYS: int = int(os.getenv("APEX_BINARY_LABEL_HORIZON_DAYS", "1"))

    # ── Funding Rate Signal (Binance perpetuals, crypto-only) ─────────────────
    FUNDING_RATE_SIGNAL_ENABLED: bool = os.getenv("APEX_FUNDING_RATE_ENABLED", "true").lower() == "true"
    FUNDING_EXTREME_THRESHOLD: float = float(os.getenv("APEX_FUNDING_EXTREME_THRESHOLD", "0.0010"))
    FUNDING_SIGNAL_SCALE: float = float(os.getenv("APEX_FUNDING_SIGNAL_SCALE", "0.0015"))
    FUNDING_RATE_CACHE_TTL: int = int(os.getenv("APEX_FUNDING_RATE_CACHE_TTL", "900"))

    # ── Candlestick Pattern Signal ─────────────────────────────────────────────
    PATTERN_SIGNAL_ENABLED: bool = os.getenv("APEX_PATTERN_SIGNAL_ENABLED", "true").lower() == "true"
    PATTERN_TREND_WINDOW: int = int(os.getenv("APEX_PATTERN_TREND_WINDOW", "5"))

    # ── Signal Aggregator (combines funding + pattern votes) ──────────────────
    SIGNAL_AGGREGATOR_AGREE_BOOST: float = float(os.getenv("APEX_SA_AGREE_BOOST", "0.08"))
    SIGNAL_AGGREGATOR_CONTRA_PENALTY: float = float(os.getenv("APEX_SA_CONTRA_PENALTY", "0.15"))
    SIGNAL_AGGREGATOR_CONTRA_THRESHOLD: float = float(os.getenv("APEX_SA_CONTRA_THRESHOLD", "0.35"))
    SIGNAL_AGGREGATOR_MIN_CONF_GATE: float = float(os.getenv("APEX_SA_MIN_CONF_GATE", "0.60"))

    # ── Confidence-Proportional Sizing ─────────────────────────────────────────
    # Scale size from CONF_SCALING_MIN_MULT at min_confidence → 1.0× at conf=1.0
    CONF_PROPORTIONAL_SIZING_ENABLED: bool = os.getenv("APEX_CONF_PROPORTIONAL_SIZING", "true").lower() == "true"
    CONF_SCALING_MIN_MULT: float = float(os.getenv("APEX_CONF_SCALING_MIN_MULT", "0.25"))

    # ── ML Model Quality Gate ──────────────────────────────────────────────────
    # Block live deployment if walk-forward directional accuracy is below this floor.
    # A model worse than random (dir_acc < 0.50) has negative edge and will lose money.
    # Set to 0.0 to disable the gate (always deploy regardless of quality).
    MODEL_QUALITY_GATE_ENABLED: bool = os.getenv("APEX_MODEL_QUALITY_GATE", "true").lower() == "true"
    MODEL_MIN_DIR_ACC: float = float(os.getenv("APEX_MODEL_MIN_DIR_ACC", "0.48"))

    # ── Signal Consensus Gate ──────────────────────────────────────────────────
    # N-of-M independent signal sources must agree on direction before entry.
    # Hard block when agreement < MIN_AGREEMENT; soft confidence penalty when
    # agreement < SOFT_THRESHOLD.
    SIGNAL_CONSENSUS_GATE_ENABLED: bool = os.getenv("APEX_SIGNAL_CONSENSUS_GATE", "true").lower() == "true"
    SIGNAL_CONSENSUS_MIN_AGREEMENT: float = float(os.getenv("APEX_SIGNAL_CONSENSUS_MIN_AGREE", "0.60"))
    SIGNAL_CONSENSUS_SOFT_THRESHOLD: float = float(os.getenv("APEX_SIGNAL_CONSENSUS_SOFT_THRESH", "0.55"))
    SIGNAL_CONSENSUS_CONF_PENALTY: float = float(os.getenv("APEX_SIGNAL_CONSENSUS_CONF_PENALTY", "0.10"))

    # ── OFI from OHLCV ────────────────────────────────────────────────────────
    # Number of recent OHLCV bars to measure order-flow imbalance over.
    OFI_LOOKBACK_BARS: int = int(os.getenv("APEX_OFI_LOOKBACK_BARS", "5"))

    # ── Regime-Adaptive Crypto Blend Weight ──────────────────────────────────
    # Override the per-regime ML weight map with a single fixed value if set.
    # Leave unset (empty string) to use the regime-aware defaults.
    CRYPTO_ML_BLEND_WEIGHT: float = float(os.getenv("APEX_CRYPTO_ML_BLEND_WEIGHT", "0.0")) or 0.0
    # 0.0 = use regime-adaptive map (default behaviour)

    # ── Macro Confidence Gate ─────────────────────────────────────────────────
    MACRO_CONFIDENCE_GATE_ENABLED: bool = os.getenv("APEX_MACRO_CONF_GATE", "true").lower() == "true"
    MACRO_YIELD_CURVE_CONF_PENALTY: float = float(os.getenv("APEX_MACRO_YC_CONF_PENALTY", "0.92"))
    MACRO_VIX_BACKWARDATION_CONF_PENALTY: float = float(os.getenv("APEX_MACRO_VIX_BACK_CONF_PENALTY", "0.88"))

    # ── Sentiment Exit Gate ───────────────────────────────────────────────────
    SENTIMENT_EXIT_GATE_ENABLED: bool = os.getenv("APEX_SENTIMENT_EXIT_GATE", "true").lower() == "true"
    SENTIMENT_EXIT_CONTRA_THRESHOLD: float = float(os.getenv("APEX_SENTIMENT_EXIT_CONTRA_THRESH", "0.45"))
    SENTIMENT_EXIT_MIN_LOSS_PCT: float = float(os.getenv("APEX_SENTIMENT_EXIT_MIN_LOSS_PCT", "-0.8"))

    # ── Cross-Asset Divergence Gate ───────────────────────────────────────────
    CROSS_ASSET_DIVERGENCE_GATE_ENABLED: bool = os.getenv("APEX_CROSS_ASSET_DIV_GATE", "true").lower() == "true"
    CROSS_ASSET_DIV_MAX_PENALTY: float = float(os.getenv("APEX_CROSS_ASSET_DIV_MAX_PENALTY", "0.12"))
    CROSS_ASSET_CONFIRM_BOOST: float = float(os.getenv("APEX_CROSS_ASSET_CONFIRM_BOOST", "0.03"))

    # ── Online Learning ────────────────────────────────────────────────────────
    # After each trade closes, the realized return is fed back to the GBM model
    # for that (asset_class, regime) bucket via a warm-start mini-update.
    ONLINE_LEARNING_ENABLED: bool = os.getenv("APEX_ONLINE_LEARNING", "true").lower() == "true"
    # Lower from 20 → 3: we only have 5 closed trades, trigger should fire now
    ONLINE_UPDATE_TRIGGER_N: int = int(os.getenv("APEX_ONLINE_TRIGGER_N", "3"))

    # ── ML/Tech Signal Conflict Detection (#3) ─────────────────────────────────
    # When ML and tech signals oppose each other, apply a confidence haircut.
    # MIN_MAG: both signals must exceed this to count as a "real" conflict.
    # MAX_HAIRCUT: max fraction of confidence to strip when conflict is total (0→1).
    SIGNAL_CONFLICT_MIN_MAG: float = float(os.getenv("APEX_SIGNAL_CONFLICT_MIN_MAG", "0.10"))
    SIGNAL_CONFLICT_MAX_HAIRCUT: float = float(os.getenv("APEX_SIGNAL_CONFLICT_MAX_HAIRCUT", "0.40"))

    # ── BL Portfolio Sizing ────────────────────────────────────────────────────
    BL_SIZING_ENABLED: bool = os.getenv("APEX_BL_SIZING", "true").lower() == "true"
    BL_UPDATE_INTERVAL_CYCLES: int = int(os.getenv("APEX_BL_UPDATE_CYCLES", "30"))
    BL_VIEW_SCALE: float = float(os.getenv("APEX_BL_VIEW_SCALE", "0.002"))
    BL_MIN_SCALE: float = float(os.getenv("APEX_BL_MIN_SCALE", "0.40"))
    BL_MAX_SCALE: float = float(os.getenv("APEX_BL_MAX_SCALE", "2.00"))

    # ── Volatility-Spike Exit ──────────────────────────────────────────────────
    VOL_SPIKE_EXIT_ENABLED: bool = os.getenv("APEX_VOL_SPIKE_EXIT", "true").lower() == "true"
    VOL_SPIKE_LOOKBACK: int = int(os.getenv("APEX_VOL_SPIKE_LOOKBACK", "20"))
    VOL_SPIKE_SIGMA: float = float(os.getenv("APEX_VOL_SPIKE_SIGMA", "2.0"))
    VOL_SPIKE_MIN_LOSS_PCT: float = float(os.getenv("APEX_VOL_SPIKE_MIN_LOSS_PCT", "0.005"))

    # ── Signal Aggregator v2 (signals/signal_aggregator.py) ──────────────────
    # Require the primary signal magnitude to exceed this floor before any
    # external vote can boost confidence. Without this, a vote aligned with
    # a near-zero (noise) primary signal adds spurious confidence.
    SIGNAL_AGGREGATOR_MIN_PRIMARY_ABS: float = float(
        os.getenv("APEX_SA_MIN_PRIMARY_ABS", "0.15")
    )
    # Normalise multi-vote boosts by count^exponent so four weak votes don't
    # out-boost one strong vote. exponent=1.0 → pure average; 0.5 → sqrt-N
    # (gives strong votes extra weight while still penalising solo noise).
    SIGNAL_AGGREGATOR_COUNT_EXPONENT: float = float(
        os.getenv("APEX_SA_COUNT_EXPONENT", "0.50")
    )
    # Absolute ceiling on total boost/penalty — prevents runaway stacking
    SIGNAL_AGGREGATOR_MAX_BOOST: float = float(
        os.getenv("APEX_SA_MAX_BOOST", "0.12")
    )
    SIGNAL_AGGREGATOR_MAX_PENALTY: float = float(
        os.getenv("APEX_SA_MAX_PENALTY", "0.25")
    )

    # ── Pattern Signal v2 (signals/pattern_signal.py) ────────────────────────
    # "max" = dominant pattern confidence (prevents averaging down),
    # "mean" = legacy behaviour, "weighted_max" = dominant × agreement bonus.
    PATTERN_CONFIDENCE_MODE: str = os.getenv("APEX_PATTERN_CONF_MODE", "weighted_max").lower()
    PATTERN_AGREEMENT_BONUS: float = float(os.getenv("APEX_PATTERN_AGREEMENT_BONUS", "0.10"))
    PATTERN_CONFLICT_PENALTY: float = float(os.getenv("APEX_PATTERN_CONFLICT_PENALTY", "0.30"))

    # ── ORB Signal v2 (signals/orb_signal.py) ────────────────────────────────
    ORB_MIN_RVOL: float = float(os.getenv("APEX_ORB_MIN_RVOL", "1.20"))
    # Minimum breakout extension as fraction of current price — floor only.
    # The actual breakout threshold scales with the opening range width
    # (range-relative breakouts are far more robust than fixed percentage).
    ORB_MIN_BREAKOUT_PCT: float = float(os.getenv("APEX_ORB_MIN_BREAKOUT_PCT", "0.003"))
    # Breakout must extend at least this fraction of the OR range width
    # beyond the boundary (e.g. 0.25 → break the range by 25% of its width).
    ORB_RANGE_EXTENSION_FRACTION: float = float(os.getenv("APEX_ORB_RANGE_EXT_FRAC", "0.25"))
    ORB_VOL_CONFIDENCE_SCALE: float = float(os.getenv("APEX_ORB_VOL_CONF_SCALE", "1.5"))
    ORB_DIST_CONFIDENCE_SCALE: float = float(os.getenv("APEX_ORB_DIST_CONF_SCALE", "0.015"))
    ORB_VOL_CONF_WEIGHT: float = float(os.getenv("APEX_ORB_VOL_CONF_WEIGHT", "0.60"))
    ORB_DIST_CONF_WEIGHT: float = float(os.getenv("APEX_ORB_DIST_CONF_WEIGHT", "0.40"))

    # ── Fee-Aware Entry Edge Gate (risk/fee_aware_edge_gate.py) ──────────────
    # Block entries whose expected edge cannot clear round-trip fees +
    # slippage + half-spread with a safety margin. Directly eliminates
    # negative-expectancy "paper cuts" that silently erode ROI.
    FEE_AWARE_EDGE_GATE_ENABLED: bool = (
        os.getenv("APEX_FEE_AWARE_EDGE_GATE_ENABLED", "true").lower() == "true"
    )
    # Required ratio of expected edge to round-trip cost. 1.5 = edge must be
    # at least 150% of cost. <1.0 guarantees losing money on average.
    FEE_AWARE_MIN_EDGE_COST_RATIO: float = float(
        os.getenv("APEX_FEE_AWARE_MIN_EDGE_COST_RATIO", "1.50")
    )
    # Default cost assumptions per asset class when the execution-layer
    # realised-cost history is unavailable (cold start).
    FEE_AWARE_DEFAULT_EQUITY_BPS: float = float(
        os.getenv("APEX_FEE_AWARE_DEFAULT_EQUITY_BPS", "8.0")
    )
    FEE_AWARE_DEFAULT_CRYPTO_BPS: float = float(
        os.getenv("APEX_FEE_AWARE_DEFAULT_CRYPTO_BPS", "18.0")
    )
    FEE_AWARE_DEFAULT_FX_BPS: float = float(
        os.getenv("APEX_FEE_AWARE_DEFAULT_FX_BPS", "4.0")
    )

    # ── Dynamic Exit Manager regime multipliers (per-regime × per-lever) ─────
    # Encoded as "stop,target,hold,signal" quad for each regime. All four
    # must parse or the regime falls back to the legacy defaults.
    EXIT_REGIME_STRONG_BULL: str = os.getenv("APEX_EXIT_REGIME_STRONG_BULL", "1.2,1.5,1.5,0.8")
    EXIT_REGIME_BULL: str = os.getenv("APEX_EXIT_REGIME_BULL", "1.1,1.3,1.2,0.9")
    EXIT_REGIME_NEUTRAL: str = os.getenv("APEX_EXIT_REGIME_NEUTRAL", "0.9,0.8,0.7,1.2")
    EXIT_REGIME_BEAR: str = os.getenv("APEX_EXIT_REGIME_BEAR", "0.8,1.2,0.8,1.1")
    EXIT_REGIME_STRONG_BEAR: str = os.getenv("APEX_EXIT_REGIME_STRONG_BEAR", "0.7,1.4,0.6,1.3")
    EXIT_REGIME_HIGH_VOLATILITY: str = os.getenv("APEX_EXIT_REGIME_HIGH_VOL", "0.6,0.7,0.5,1.5")

    # ── Adaptive Position Sizer (risk/adaptive_position_sizer.py) ────────────
    # Volatility percentile window (trading days). Was hard-coded 252.
    ADAPTIVE_SIZER_VOL_LOOKBACK: int = int(
        os.getenv("APEX_ADAPTIVE_SIZER_VOL_LOOKBACK", "252")
    )
    ADAPTIVE_SIZER_VOL_MIN_HISTORY: int = int(
        os.getenv("APEX_ADAPTIVE_SIZER_VOL_MIN_HISTORY", "10")
    )
    # Signal-confidence weight: size_mult = FLOOR + WEIGHT * signal_confidence
    ADAPTIVE_SIZER_CONF_FLOOR: float = float(
        os.getenv("APEX_ADAPTIVE_SIZER_CONF_FLOOR", "0.50")
    )
    ADAPTIVE_SIZER_CONF_WEIGHT: float = float(
        os.getenv("APEX_ADAPTIVE_SIZER_CONF_WEIGHT", "0.50")
    )
    # Inverse-vol scaling bounds: mult = clip(2 - vol_percentile, MIN, MAX)
    ADAPTIVE_SIZER_VOL_MULT_MIN: float = float(
        os.getenv("APEX_ADAPTIVE_SIZER_VOL_MULT_MIN", "0.50")
    )
    ADAPTIVE_SIZER_VOL_MULT_MAX: float = float(
        os.getenv("APEX_ADAPTIVE_SIZER_VOL_MULT_MAX", "1.50")
    )
    # Sharpe tiers (sharpe_ratio thresholds → size multiplier)
    ADAPTIVE_SIZER_SHARPE_EXCELLENT: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_EXCELLENT", "1.5"))
    ADAPTIVE_SIZER_SHARPE_GOOD: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_GOOD", "1.0"))
    ADAPTIVE_SIZER_SHARPE_OK: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_OK", "0.5"))
    ADAPTIVE_SIZER_SHARPE_MULT_EXCELLENT: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_MULT_EXCELLENT", "1.30"))
    ADAPTIVE_SIZER_SHARPE_MULT_GOOD: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_MULT_GOOD", "1.15"))
    ADAPTIVE_SIZER_SHARPE_MULT_OK: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_MULT_OK", "1.00"))
    ADAPTIVE_SIZER_SHARPE_MULT_WEAK: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_MULT_WEAK", "0.85"))
    ADAPTIVE_SIZER_SHARPE_MULT_NEG: float = float(os.getenv("APEX_ADAPTIVE_SIZER_SHARPE_MULT_NEG", "0.60"))
    # Drawdown protection tiers
    ADAPTIVE_SIZER_DD_SEVERE: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_SEVERE", "0.10"))
    ADAPTIVE_SIZER_DD_HIGH: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_HIGH", "0.08"))
    ADAPTIVE_SIZER_DD_MODERATE: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_MODERATE", "0.05"))
    ADAPTIVE_SIZER_DD_MULT_SEVERE: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_MULT_SEVERE", "0.30"))
    ADAPTIVE_SIZER_DD_MULT_HIGH: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_MULT_HIGH", "0.50"))
    ADAPTIVE_SIZER_DD_MULT_MODERATE: float = float(os.getenv("APEX_ADAPTIVE_SIZER_DD_MULT_MODERATE", "0.70"))
    # Win-rate adjustment: mult = FLOOR + WEIGHT * win_rate
    ADAPTIVE_SIZER_WR_FLOOR: float = float(os.getenv("APEX_ADAPTIVE_SIZER_WR_FLOOR", "0.50"))
    ADAPTIVE_SIZER_WR_WEIGHT: float = float(os.getenv("APEX_ADAPTIVE_SIZER_WR_WEIGHT", "1.00"))
    # Final clamp applied to the combined multiplier
    ADAPTIVE_SIZER_CLIP_MIN: float = float(os.getenv("APEX_ADAPTIVE_SIZER_CLIP_MIN", "0.20"))
    ADAPTIVE_SIZER_CLIP_MAX: float = float(os.getenv("APEX_ADAPTIVE_SIZER_CLIP_MAX", "2.50"))
    # Geometric normalization exponent — shrinks multi-factor compounding so
    # additive factors don't cascade to the clip floor (1.0 = classic multiplication,
    # 0.5 = geometric mean of two factors, 0.167 = 6th-root when all 6 factors stack).
    ADAPTIVE_SIZER_STACK_EXPONENT: float = float(
        os.getenv("APEX_ADAPTIVE_SIZER_STACK_EXPONENT", "0.65")
    )

    # ── Adaptive ATR Stops (risk/adaptive_atr_stops.py) ──────────────────────
    ATR_STOP_PERIOD: int = int(os.getenv("APEX_ATR_STOP_PERIOD", "14"))
    ATR_MIN_STOP_PCT: float = float(os.getenv("APEX_ATR_MIN_STOP_PCT", "0.006"))
    ATR_MAX_STOP_PCT: float = float(os.getenv("APEX_ATR_MAX_STOP_PCT", "0.15"))
    ATR_FALLBACK_PCT: float = float(os.getenv("APEX_ATR_FALLBACK_PCT", "0.02"))
    ATR_REGIME_MULT_STRONG_BULL: float = float(os.getenv("APEX_ATR_REGIME_MULT_STRONG_BULL", "2.20"))
    ATR_REGIME_MULT_BULL: float = float(os.getenv("APEX_ATR_REGIME_MULT_BULL", "2.00"))
    ATR_REGIME_MULT_NEUTRAL: float = float(os.getenv("APEX_ATR_REGIME_MULT_NEUTRAL", "2.00"))
    ATR_REGIME_MULT_BEAR: float = float(os.getenv("APEX_ATR_REGIME_MULT_BEAR", "2.00"))
    ATR_REGIME_MULT_STRONG_BEAR: float = float(os.getenv("APEX_ATR_REGIME_MULT_STRONG_BEAR", "2.20"))
    ATR_REGIME_MULT_VOLATILE: float = float(os.getenv("APEX_ATR_REGIME_MULT_VOLATILE", "2.60"))
    ATR_REGIME_MULT_CRISIS: float = float(os.getenv("APEX_ATR_REGIME_MULT_CRISIS", "3.00"))
    ATR_REGIME_MULT_DEFAULT: float = float(os.getenv("APEX_ATR_REGIME_MULT_DEFAULT", "2.00"))
    ATR_VIX_BASELINE: float = float(os.getenv("APEX_ATR_VIX_BASELINE", "20.0"))
    ATR_VIX_SCALE: float = float(os.getenv("APEX_ATR_VIX_SCALE", "40.0"))
    ATR_VIX_FACTOR_MAX: float = float(os.getenv("APEX_ATR_VIX_FACTOR_MAX", "2.0"))
    # Profit-tier pairs encoded as "threshold:tighten_factor" list
    # pnl >= threshold → trailing distance = base * tighten_factor (lock gains)
    ATR_PROFIT_TIERS: str = os.getenv(
        "APEX_ATR_PROFIT_TIERS",
        "0.12:0.40,0.08:0.50,0.05:0.70,0.03:0.85",
    )

    # ── Dynamic Exit Manager (risk/dynamic_exit_manager.py) ──────────────────
    EXIT_BASE_STOP_LOSS_PCT: float = float(os.getenv("APEX_EXIT_BASE_STOP_LOSS_PCT", "0.03"))
    EXIT_BASE_TAKE_PROFIT_PCT: float = float(os.getenv("APEX_EXIT_BASE_TAKE_PROFIT_PCT", "0.06"))
    EXIT_BASE_TRAILING_ACTIVATION: float = float(os.getenv("APEX_EXIT_BASE_TRAILING_ACTIVATION", "0.025"))
    EXIT_BASE_TRAILING_DISTANCE: float = float(os.getenv("APEX_EXIT_BASE_TRAILING_DISTANCE", "0.02"))
    EXIT_BASE_MAX_HOLD_DAYS: int = int(os.getenv("APEX_EXIT_BASE_MAX_HOLD_DAYS", "14"))
    EXIT_VIX_EXTREME: float = float(os.getenv("APEX_EXIT_VIX_EXTREME", "30.0"))
    EXIT_VIX_HIGH: float = float(os.getenv("APEX_EXIT_VIX_HIGH", "25.0"))
    EXIT_VIX_ELEVATED: float = float(os.getenv("APEX_EXIT_VIX_ELEVATED", "20.0"))
    EXIT_VIX_COMPLACENCY: float = float(os.getenv("APEX_EXIT_VIX_COMPLACENCY", "12.0"))
    EXIT_VIX_EXTREME_STOP_MULT: float = float(os.getenv("APEX_EXIT_VIX_EXTREME_STOP_MULT", "0.60"))
    EXIT_VIX_HIGH_STOP_MULT: float = float(os.getenv("APEX_EXIT_VIX_HIGH_STOP_MULT", "0.75"))
    EXIT_VIX_ELEVATED_STOP_MULT: float = float(os.getenv("APEX_EXIT_VIX_ELEVATED_STOP_MULT", "0.90"))
    EXIT_VIX_COMPLACENCY_STOP_MULT: float = float(os.getenv("APEX_EXIT_VIX_COMPLACENCY_STOP_MULT", "1.10"))
    EXIT_VIX_COMPLACENCY_TARGET_MULT: float = float(os.getenv("APEX_EXIT_VIX_COMPLACENCY_TARGET_MULT", "1.20"))
    EXIT_VIX_COMPLACENCY_HOLD_MULT: float = float(os.getenv("APEX_EXIT_VIX_COMPLACENCY_HOLD_MULT", "1.30"))
    EXIT_VIX_EXTREME_MAX_HOLD: int = int(os.getenv("APEX_EXIT_VIX_EXTREME_MAX_HOLD", "5"))
    EXIT_VIX_HIGH_MAX_HOLD: int = int(os.getenv("APEX_EXIT_VIX_HIGH_MAX_HOLD", "7"))
    EXIT_VIX_EXTREME_SIGNAL_MULT: float = float(os.getenv("APEX_EXIT_VIX_EXTREME_SIGNAL_MULT", "1.50"))
    EXIT_VIX_HIGH_SIGNAL_MULT: float = float(os.getenv("APEX_EXIT_VIX_HIGH_SIGNAL_MULT", "1.30"))
    EXIT_VIX_ELEVATED_SIGNAL_MULT: float = float(os.getenv("APEX_EXIT_VIX_ELEVATED_SIGNAL_MULT", "1.10"))
    EXIT_ATR_STOP_MULT: float = float(os.getenv("APEX_EXIT_ATR_STOP_MULT", "2.0"))
    EXIT_ATR_TARGET_MULT: float = float(os.getenv("APEX_EXIT_ATR_TARGET_MULT", "2.5"))
    EXIT_ATR_TRAIL_MULT: float = float(os.getenv("APEX_EXIT_ATR_TRAIL_MULT", "1.5"))
    EXIT_STOP_CLAMP_MIN: float = float(os.getenv("APEX_EXIT_STOP_CLAMP_MIN", "0.02"))
    EXIT_STOP_CLAMP_MAX: float = float(os.getenv("APEX_EXIT_STOP_CLAMP_MAX", "0.15"))
    EXIT_TARGET_CLAMP_MIN: float = float(os.getenv("APEX_EXIT_TARGET_CLAMP_MIN", "0.03"))
    EXIT_TARGET_CLAMP_MAX: float = float(os.getenv("APEX_EXIT_TARGET_CLAMP_MAX", "0.25"))
    EXIT_SIGNAL_EXIT_CLAMP_MIN: float = float(os.getenv("APEX_EXIT_SIGNAL_EXIT_CLAMP_MIN", "0.15"))
    EXIT_SIGNAL_EXIT_CLAMP_MAX: float = float(os.getenv("APEX_EXIT_SIGNAL_EXIT_CLAMP_MAX", "0.60"))
    EXIT_MAX_HOLD_CLAMP_MIN: int = int(os.getenv("APEX_EXIT_MAX_HOLD_CLAMP_MIN", "3"))
    EXIT_MAX_HOLD_CLAMP_MAX: int = int(os.getenv("APEX_EXIT_MAX_HOLD_CLAMP_MAX", "30"))

    # ── Composite Signal Quality Gate ─────────────────────────────────────────
    # Blocks entries where signal × confidence < this floor, preventing borderline
    # signal + borderline confidence combinations that have no real edge.
    # Example: signal=0.16 × confidence=0.62 = 0.099 → blocked (below 0.10).
    MIN_SIGNAL_QUALITY_COMPOSITE: float = float(
        os.getenv("APEX_MIN_SIGNAL_QUALITY_COMPOSITE", "0.10")
    )

    # ── Regime-Direction Alignment Gate ──────────────────────────────────────
    # Blocks LONG entries in bear/strong_bear regimes and SHORT entries in
    # bull/strong_bull regimes. Prevents fighting the macro trend.
    REGIME_DIRECTION_GATE_ENABLED: bool = (
        os.getenv("APEX_REGIME_DIRECTION_GATE_ENABLED", "true").lower() == "true"
    )
    # In volatile/crisis regimes, require a higher signal × confidence composite
    REGIME_VOLATILE_QUALITY_FLOOR: float = float(
        os.getenv("APEX_REGIME_VOLATILE_QUALITY_FLOOR", "0.15")
    )

    # ── Re-entry Gap After Exit ───────────────────────────────────────────────
    # Minimum seconds between a position close and the next entry in the same symbol.
    # Raised 2026-03-19 audit: SOL re-entered 38 min after loss exit with a WEAKER signal
    # (0.127 vs 0.233), getting immediately excellence-exited again — pure churn.
    # 30 min (1800s) was not enough; raising to 60 min (3600s) after a loss.
    REENTRY_GAP_SECONDS: float = float(os.getenv("APEX_REENTRY_GAP_SECONDS", "600"))       # 10 min after WIN
    LOSS_REENTRY_GAP_SECONDS: float = float(os.getenv("APEX_LOSS_REENTRY_GAP_SECONDS", "3600"))  # 60 min after LOSS

    # ── Crypto Entry Window (UTC hours) ───────────────────────────────────────
    # Block NEW crypto entries outside liquid trading hours to avoid midnight
    # order spam (577 pre-trade attempts on Mar 18, 469 before market open).
    # Mirrors the exit liquidity window: 13:00–22:00 UTC (9am–6pm ET).
    CRYPTO_ENTRY_WINDOW_ENABLED: bool = os.getenv("APEX_CRYPTO_ENTRY_WINDOW_ENABLED", "true").lower() == "true"
    CRYPTO_ENTRY_WINDOW_UTC_START: int = int(os.getenv("APEX_CRYPTO_ENTRY_WINDOW_UTC_START", "13"))
    CRYPTO_ENTRY_WINDOW_UTC_END: int = int(os.getenv("APEX_CRYPTO_ENTRY_WINDOW_UTC_END", "22"))

    # ── Crypto TWAP (large order slicing via Alpaca) ────────────────────────
    CRYPTO_TWAP_ENABLED: bool = os.getenv("APEX_CRYPTO_TWAP_ENABLED", "true").lower() == "true"
    CRYPTO_TWAP_MIN_NOTIONAL: float = float(os.getenv("APEX_CRYPTO_TWAP_MIN_NOTIONAL", "1000.0"))
    CRYPTO_TWAP_INTERVAL_SEC: float = float(os.getenv("APEX_CRYPTO_TWAP_INTERVAL_SEC", "30.0"))
    CRYPTO_TWAP_ABANDON_PCT: float = float(os.getenv("APEX_CRYPTO_TWAP_ABANDON_PCT", "0.005"))

    # ── Equity TWAP ───────────────────────────────────────────────────────────
    EQUITY_TWAP_ENABLED: bool = os.getenv("APEX_EQUITY_TWAP_ENABLED", "true").lower() == "true"
    EQUITY_TWAP_MIN_NOTIONAL: float = float(os.getenv("APEX_EQUITY_TWAP_MIN_NOTIONAL", "10000.0"))
    EQUITY_TWAP_SLICES: int = int(os.getenv("APEX_EQUITY_TWAP_SLICES", "5"))
    EQUITY_TWAP_INTERVAL_SEC: float = float(os.getenv("APEX_EQUITY_TWAP_INTERVAL_SEC", "60.0"))
    EQUITY_TWAP_ADVERSE_BPS: float = float(os.getenv("APEX_EQUITY_TWAP_ADVERSE_BPS", "50.0"))

    # ── Order Flow Imbalance ──────────────────────────────────────────────────
    ORDER_FLOW_GATE_ENABLED: bool = os.getenv("APEX_ORDER_FLOW_GATE_ENABLED", "true").lower() == "true"
    ORDER_FLOW_SIGNAL_WEIGHT: float = float(os.getenv("APEX_ORDER_FLOW_SIGNAL_WEIGHT", "0.08"))

    # ── Pairs Trading (Statistical Arbitrage) ────────────────────────────────
    PAIRS_TRADING_ENABLED: bool = os.getenv("APEX_PAIRS_TRADING_ENABLED", "true").lower() == "true"
    PAIRS_LOOKBACK_WINDOW: int = int(os.getenv("APEX_PAIRS_LOOKBACK_WINDOW", "60"))
    PAIRS_Z_ENTRY: float = float(os.getenv("APEX_PAIRS_Z_ENTRY", "2.0"))
    PAIRS_Z_EXIT: float = float(os.getenv("APEX_PAIRS_Z_EXIT", "0.5"))
    PAIRS_MIN_HALF_LIFE: int = int(os.getenv("APEX_PAIRS_MIN_HALF_LIFE", "1"))
    PAIRS_MAX_HALF_LIFE: int = int(os.getenv("APEX_PAIRS_MAX_HALF_LIFE", "20"))
    PAIRS_SCAN_INTERVAL_CYCLES: int = int(os.getenv("APEX_PAIRS_SCAN_INTERVAL_CYCLES", "300"))  # ~5 min
    PAIRS_MAX_OVERLAY: float = float(os.getenv("APEX_PAIRS_MAX_OVERLAY", "0.10"))   # max ±0.10 signal adj
    PAIRS_SIGNAL_WEIGHT: float = float(os.getenv("APEX_PAIRS_SIGNAL_WEIGHT", "0.15"))  # 15% blend weight

    # ── Per-symbol Consecutive Loss Protection ────────────────────────────────
    # Tracked dynamically in execution_loop._symbol_loss_streak.
    # ≥2 losses → require MIN_CONFIDENCE + 0.15 confidence; ≥3 → block for session.

    # Per-session risk parameters
    CORE_MAX_POSITIONS: int = int(os.getenv("APEX_CORE_MAX_POSITIONS", "25"))
    CRYPTO_MAX_POSITIONS: int = int(os.getenv("APEX_CRYPTO_MAX_POSITIONS", "20"))
    CORE_MAX_DAILY_LOSS: float = float(os.getenv("APEX_CORE_MAX_DAILY_LOSS", "0.025"))
    CRYPTO_MAX_DAILY_LOSS_SESSION: float = float(os.getenv("APEX_CRYPTO_MAX_DAILY_LOSS_SESSION", "0.06"))
    CORE_KELLY_FRACTION: float = float(os.getenv("APEX_CORE_KELLY_FRACTION", "0.60"))
    CRYPTO_KELLY_FRACTION: float = float(os.getenv("APEX_CRYPTO_KELLY_FRACTION", "0.40"))

    # Per-session position sizing (more aggressive for concentrated signals)
    CORE_POSITION_SIZE_USD: float = float(os.getenv("APEX_CORE_POSITION_SIZE_USD", "25000"))
    CORE_KELLY_MAX_POSITION_PCT: float = float(os.getenv("APEX_CORE_KELLY_MAX_POSITION_PCT", "0.08"))

    # Per-session ATR stops (tighter for better Sharpe)
    CORE_ATR_MULTIPLIER_STOP: float = float(os.getenv("APEX_CORE_ATR_MULTIPLIER_STOP", "2.0"))
    CORE_TRAILING_STOP_ATR: float = float(os.getenv("APEX_CORE_TRAILING_STOP_ATR", "1.8"))

    # Per-session regime thresholds (relaxed to capture more trades)
    CORE_SIGNAL_THRESHOLDS_BY_REGIME: Dict = {
        'strong_bull': 0.08, 'bull': 0.10, 'neutral': 0.10,
        'bear': 0.12, 'strong_bear': 0.10, 'volatile': 0.15
    }
    CRYPTO_SIGNAL_THRESHOLDS_BY_REGIME: Dict = {
        'strong_bull': 0.13, 'bull': 0.15, 'neutral': 0.16,
        'bear': 0.15, 'strong_bear': 0.14, 'volatile': 0.16   # raised +0.02 across all regimes for higher quality signals
    }

    # Per-session state files
    DATA_DIR: Path = Path(os.getenv("APEX_DATA_DIR", str(Path(__file__).parent / "data")))
    CORE_STATE_FILE: str = str(DATA_DIR / "core_trading_state.json")
    CRYPTO_STATE_FILE: str = str(DATA_DIR / "crypto_trading_state.json")

    # High-conviction assets (get 2x position sizing in Core session)
    # Top performers by risk-adjusted returns from backtest analysis
    CORE_HIGH_CONVICTION: List[str] = [
        "NVDA", "META", "AAPL", "MSFT", "GOOGL", "AMZN", "AVGO", "LLY",
        "JPM", "GS", "XOM", "CVX", "CAT", "GE", "NFLX",
        "SPY", "QQQ", "HD", "WMT", "UNH",
        "CRM", "ORCL", "AMD", "NEM", "MPC",
        "LIN", "HON", "ABBV", "MRK", "BA"
    ]

    # Crypto high-conviction (momentum leaders, get 1.5x sizing boost)
    CRYPTO_HIGH_CONVICTION: List[str] = [
        "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
        "XRP/USD", "ADA/USD", "UNI/USD", "AAVE/USD",
    ]

    @classmethod
    def get_session_config(cls, session_type: str) -> Dict:
        """Return config overrides for a specific session type."""
        if session_type == "core":
            return {
                "initial_capital": cls.CORE_INITIAL_CAPITAL,
                "min_signal_threshold": cls.CORE_MIN_SIGNAL_THRESHOLD,
                "min_confidence": cls.CORE_MIN_CONFIDENCE,
                "max_positions": cls.CORE_MAX_POSITIONS,
                "max_daily_loss": cls.CORE_MAX_DAILY_LOSS,
                "kelly_fraction": cls.CORE_KELLY_FRACTION,
                "position_size_usd": cls.CORE_POSITION_SIZE_USD,
                "kelly_max_position_pct": cls.CORE_KELLY_MAX_POSITION_PCT,
                "atr_multiplier_stop": cls.CORE_ATR_MULTIPLIER_STOP,
                "trailing_stop_atr": cls.CORE_TRAILING_STOP_ATR,
                "signal_thresholds_by_regime": cls.CORE_SIGNAL_THRESHOLDS_BY_REGIME,
                "state_file": cls.CORE_STATE_FILE,
                "high_conviction": cls.CORE_HIGH_CONVICTION,
            }
        elif session_type == "crypto":
            return {
                "initial_capital": cls.CRYPTO_INITIAL_CAPITAL,
                "min_signal_threshold": cls.CRYPTO_MIN_SIGNAL_THRESHOLD,
                "min_confidence": cls.CRYPTO_MIN_CONFIDENCE,
                "max_positions": cls.CRYPTO_MAX_POSITIONS,
                "max_daily_loss": cls.CRYPTO_MAX_DAILY_LOSS_SESSION,
                "kelly_fraction": cls.CRYPTO_KELLY_FRACTION,
                "position_size_usd": cls.CRYPTO_POSITION_SIZE_USD,
                "kelly_max_position_pct": cls.KELLY_MAX_POSITION_PCT,
                "atr_multiplier_stop": cls.ATR_MULTIPLIER_STOP,
                "trailing_stop_atr": cls.TRAILING_STOP_ATR,
                "signal_thresholds_by_regime": cls.CRYPTO_SIGNAL_THRESHOLDS_BY_REGIME,
                "state_file": cls.CRYPTO_STATE_FILE,
                "high_conviction": cls.CRYPTO_HIGH_CONVICTION,
            }
        else:
            return {
                "initial_capital": cls.INITIAL_CAPITAL,
                "min_signal_threshold": cls.MIN_SIGNAL_THRESHOLD,
                "min_confidence": cls.MIN_CONFIDENCE,
                "max_positions": cls.MAX_POSITIONS,
                "max_daily_loss": cls.MAX_DAILY_LOSS,
                "kelly_fraction": cls.KELLY_FRACTION,
                "position_size_usd": cls.POSITION_SIZE_USD,
                "kelly_max_position_pct": cls.KELLY_MAX_POSITION_PCT,
                "atr_multiplier_stop": cls.ATR_MULTIPLIER_STOP,
                "trailing_stop_atr": cls.TRAILING_STOP_ATR,
                "signal_thresholds_by_regime": cls.SIGNAL_THRESHOLDS_BY_REGIME,
                "state_file": str(cls.DATA_DIR / "trading_state.json"),
                "high_conviction": [],
            }

    # ═══════════════════════════════════════════════════════════════
    # UNIVERSE SELECTION
    # ═══════════════════════════════════════════════════════════════
    UNIVERSE_MODE = "SP500"  # Options: "SP500", "NASDAQ100", "CUSTOM"
    
    # S&P 500 Top Liquid Stocks & Multi-Asset Universe
    # ------------------------------------------------
    # 1. Major Indices
    INDICES = ["SPY", "QQQ", "IWM", "DIA"]

    # 2. Forex Pairs (Major G10)
    # ⚠️  Disabled: IBKR IDEALPRO minimum lots (EUR 20K, GBP 17K, etc.)
    # are well above paper sizing. Re-enable only when proper lot sizing is modeled.
    FOREX_PAIRS: list = []

    # 3. Crypto Pairs — Alpaca-confirmed tradeable universe
    # Removed (no Alpaca live price): ATOM, NEAR, TRX, INJ, OP (confirmed via debug log 150+ misses)
    # Removed (0 yfinance rows): APT/USD, SUI/USD
    # Removed (delisted yfinance Feb 2026): MATIC/USD, UNI/USD
    # Removed (chronic slippage >15bps): BCH, XLM, ETC
    CRYPTO_PAIRS = [
        # Tier 1: Blue-chip (highest liquidity, tightest spreads)
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "LINK/USD",
        "DOGE/USD",
        "AVAX/USD",
        # Tier 2: High-momentum mid-caps
        "XRP/USD",    # ~8-12bps spread, top-5 market cap
        "ADA/USD",    # Cardano, ~10-15bps
        "DOT/USD",    # Polkadot, ~12-14bps
        "LTC/USD",    # Litecoin, ~8-12bps, deepest order book
        "AAVE/USD",   # DeFi lending leader
        # Tier 3: Alpaca-confirmed Tier 4 (added 2026-03-26)
        "FIL/USD",    # Filecoin, moderate spread
        "ALGO/USD",   # Algorand, 6+ years history
        "ARB/USD",    # Arbitrum, largest L2 by TVL
    ]
    EXTRA_CRYPTO_PAIRS = [
        token.strip().upper()
        for token in os.getenv("APEX_EXTRA_CRYPTO_PAIRS", "").split(",")
        if token.strip()
    ]
    if EXTRA_CRYPTO_PAIRS:
        CRYPTO_PAIRS = list(dict.fromkeys(CRYPTO_PAIRS + EXTRA_CRYPTO_PAIRS))

    # 4. Top S&P 500 Components (aligned with SECTOR_MAP)
    # Exactly 89 equities so IBKR universe = 4 indices + 7 forex + 89 equities = 100
    SP500_TOP_100 = [
        # Technology (18)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CSCO", "ADBE",
        "CRM", "ACN", "AMD", "INTC", "IBM", "QCOM", "TXN", "AMAT",
        # Financials (10)
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB",
        # Healthcare (10)
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
        # Consumer (10)
        "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "DG", "DLTR",
        # Industrials (10)
        "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP",
        # Energy (8)
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        # Materials (5)
        "LIN", "APD", "SHW", "FCX",
        # Communication (6)
        "NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ",
        # Real Estate & Utilities (5)
        "AMT", "PLD", "CCI", "EQIX", "NEE",
        # Commodities (4) — ETFs with equity-like trading
        "GLD", "SLV", "USO", "UNG",
        # Additional high-momentum names (3)
        "MU", "LRCX", "OXY",
    ]

    # Combine all into master universe — priority-ordered so GLD/SLV/defensive assets are
    # processed first each cycle (avoids random set() ordering that previously buried them).
    _PRIORITY_SYMBOLS = ["GLD", "SLV", "XOM", "CVX", "UNH", "LLY", "JNJ", "ABBV", "MRK"]
    SYMBOLS = list(dict.fromkeys(_PRIORITY_SYMBOLS + INDICES + CRYPTO_PAIRS + SP500_TOP_100 + FOREX_PAIRS))

    # Session-scoped universes (for dual-session mode)
    CORE_SYMBOLS = list(set(INDICES + FOREX_PAIRS + SP500_TOP_100))  # No crypto
    CRYPTO_SYMBOLS = list(set(CRYPTO_PAIRS))  # Crypto only

    @classmethod
    def get_session_symbols(cls, session_type: str) -> list:
        """Return the symbol universe for a specific session."""
        if session_type == "core":
            return cls.CORE_SYMBOLS
        elif session_type == "crypto":
            return cls.CRYPTO_SYMBOLS
        return cls.SYMBOLS

    # Backtesting-only symbols (kept in universe, excluded from IBKR paper execution)
    BACKTEST_ONLY_SYMBOLS = {
        "CRYPTO:SOL/USDT", "CRYPTO:DOGE/USDT",
        "CRYPTO:SUSHI/USD", "CRYPTO:CRV/USD", "CRYPTO:GRT/USD", "CRYPTO:ICP/USD",  # Lower liquidity crypto
    }

    # Commodity symbols for special handling
    COMMODITY_SYMBOLS = {'GLD', 'SLV', 'USO', 'UNG', 'PALL'}

    # Defensive / commodity / energy symbols — receive a 1.5× position-size boost.
    # These outperform during elevated VIX / geopolitical shocks (gold +19% Jan–Mar 2026).
    DEFENSIVE_SYMBOLS: frozenset = frozenset(["GLD", "SLV", "XOM", "CVX", "UNH", "LLY", "JNJ", "ABBV", "MRK"])
    DEFENSIVE_POSITION_SIZE_MULTIPLIER: float = float(os.getenv("APEX_DEFENSIVE_SIZE_MULT", "1.5"))

    # Macro momentum overlay: amplify signals for assets with >10% 20-day return by this factor.
    MOMENTUM_SIGNAL_BOOST: float = float(os.getenv("APEX_MOMENTUM_SIGNAL_BOOST", "1.20"))

    # VIX threshold above which new crypto entries are completely blocked.
    VIX_CRYPTO_BLOCK_THRESHOLD: float = float(os.getenv("APEX_VIX_CRYPTO_BLOCK_THRESHOLD", "35.0"))
    
    # Sector Mappings (for exposure tracking)
    SECTOR_MAP = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Technology", "META": "Technology", "TSLA": "Technology",
        "AVGO": "Technology", "ORCL": "Technology", "CSCO": "Technology",
        "ADBE": "Technology", "CRM": "Technology", "ACN": "Technology",
        "AMD": "Technology", "INTC": "Technology", "IBM": "Technology",
        "QCOM": "Technology", "TXN": "Technology", "AMAT": "Technology",
        "MU": "Technology", "LRCX": "Technology",
        
        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "C": "Financials",
        "BLK": "Financials", "AXP": "Financials", "SCHW": "Financials",
        "USB": "Financials",
        
        # Healthcare
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
        "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
        "ABT": "Healthcare", "DHR": "Healthcare", "PFE": "Healthcare",
        "BMY": "Healthcare",
        
        # Consumer
        "AMZN": "Consumer", "WMT": "Consumer", "HD": "Consumer",
        "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
        "LOW": "Consumer", "TGT": "Consumer", "DG": "Consumer",
        "DLTR": "Consumer",
        
        # Industrials
        "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
        "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
        "LMT": "Industrials", "DE": "Industrials", "MMM": "Industrials",
        "UNP": "Industrials",
        
        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
        "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
        "HAL": "Energy",
        
        # Materials
        "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
        "SHW": "Materials", "FCX": "Materials", "NEM": "Materials",
        "DOW": "Materials", "DD": "Materials", "ALB": "Materials",
        "CE": "Materials",
        
        # Communication
        "NFLX": "Communication", "DIS": "Communication", "CMCSA": "Communication",
        "T": "Communication", "TMUS": "Communication", "VZ": "Communication",
        "CHTR": "Communication", "EA": "Communication",
        
        # Real Estate & Utilities
        "AMT": "RealEstate", "PLD": "RealEstate", "CCI": "RealEstate",
        "EQIX": "RealEstate", "PSA": "RealEstate", "NEE": "Utilities",
        "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
        "AEP": "Utilities",
        
        # Commodities
        "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
        "GLD": "Commodities", "SLV": "Commodities", "USO": "Commodities",
        "UNG": "Commodities", "PALL": "Commodities"
    }
    
    @classmethod
    def get_sector(cls, symbol: str) -> str:
        """Get sector for a symbol."""
        if symbol.startswith("CRYPTO:"):
            return "Crypto"
        return cls.SECTOR_MAP.get(symbol, "Unknown")

    @classmethod
    def is_commodity(cls, symbol: str) -> bool:
        """
        Check if symbol is a commodity.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is a commodity, False otherwise
        """
        return symbol in cls.COMMODITY_SYMBOLS or cls.SECTOR_MAP.get(symbol) == "Commodities"

    @classmethod
    def is_etf(cls, symbol: str) -> bool:
        """
        Check if symbol is an ETF.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if symbol is an ETF, False otherwise
        """
        return cls.SECTOR_MAP.get(symbol) == "ETF"

    # ═══════════════════════════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════════════════════════
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models" / "saved_ultimate"
    PRODUCTION_MODELS_DIR = BASE_DIR / "models" / "saved_ultimate"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)


# ═══════════════════════════════════════════════════════════════
# VALIDATE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
def validate_config() -> bool:
    """
    Validate configuration settings for safety and consistency.

    Checks:
        - Position size relative to capital
        - Max exposure doesn't exceed capital
        - Signal thresholds are reasonable
        - Cooldown periods are adequate

    Returns:
        bool: True if all validations pass, False otherwise
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Critical errors
    if ApexConfig.LIVE_TRADING and not ApexConfig.LIVE_TRADING_CONFIRMED:
        errors.append(
            "❌ Live trading requested without confirmation. "
            "Set APEX_LIVE_TRADING_CONFIRMED=true to explicitly opt in."
        )

    if ApexConfig.MAX_POSITIONS * ApexConfig.POSITION_SIZE_USD > ApexConfig.INITIAL_CAPITAL:
        errors.append(
            f"❌ Max exposure ({ApexConfig.MAX_POSITIONS} × ${ApexConfig.POSITION_SIZE_USD:,}) "
            f"exceeds capital (${ApexConfig.INITIAL_CAPITAL:,})"
        )

    if ApexConfig.IBKR_PORT not in [7496, 7497]:
        warnings.append(
            f"⚠️  Non-standard IBKR port ({ApexConfig.IBKR_PORT}). "
            "Expected 7496 (Live) or 7497 (Paper)"
        )

    # Warnings (non-fatal)
    if ApexConfig.POSITION_SIZE_USD > ApexConfig.INITIAL_CAPITAL * 0.05:
        warnings.append(
            f"⚠️  Position size (${ApexConfig.POSITION_SIZE_USD:,}) > 5% of capital"
        )

    if ApexConfig.MIN_SIGNAL_THRESHOLD < 0.15:
        warnings.append(
            f"⚠️  Signal threshold ({ApexConfig.MIN_SIGNAL_THRESHOLD}) is very low - risk of false signals"
        )

    if ApexConfig.TRADE_COOLDOWN_SECONDS < 60:
        warnings.append(
            f"⚠️  Cooldown ({ApexConfig.TRADE_COOLDOWN_SECONDS}s) is short - risk of overtrading"
        )

    if ApexConfig.PERFORMANCE_TARGET_SHARPE < 1.0:
        warnings.append(
            f"⚠️  Performance target Sharpe ({ApexConfig.PERFORMANCE_TARGET_SHARPE}) is lenient"
        )

    if ApexConfig.PERFORMANCE_MAX_DRAWDOWN > ApexConfig.MAX_DRAWDOWN:
        warnings.append(
            f"⚠️  Performance governor max drawdown ({ApexConfig.PERFORMANCE_MAX_DRAWDOWN:.1%}) "
            f"exceeds hard system max drawdown ({ApexConfig.MAX_DRAWDOWN:.1%})"
        )

    if ApexConfig.KILL_SWITCH_DD_MULTIPLIER < 1.0:
        warnings.append(
            f"⚠️  Kill-switch DD multiplier ({ApexConfig.KILL_SWITCH_DD_MULTIPLIER}) is very strict"
        )

    if ApexConfig.KILL_SWITCH_LOGIC.upper() not in {"OR", "AND"}:
        warnings.append(
            f"⚠️  Kill-switch logic ({ApexConfig.KILL_SWITCH_LOGIC}) invalid, expected OR/AND"
        )

    if ApexConfig.PRETRADE_MAX_ORDER_NOTIONAL <= 0:
        warnings.append("⚠️  PRETRADE max order notional must be > 0")

    if ApexConfig.PRETRADE_MAX_ORDER_NOTIONAL <= 0:
        errors.append("PRETRADE_MAX_ORDER_NOTIONAL must be positive")

    if not (0 < ApexConfig.PRETRADE_MAX_PARTICIPATION_RATE <= 1):
        warnings.append(
            f"⚠️  PRETRADE max participation ({ApexConfig.PRETRADE_MAX_PARTICIPATION_RATE}) should be in (0,1]"
        )

    if ApexConfig.PRETRADE_MAX_GROSS_EXPOSURE_RATIO < 1.0:
        warnings.append(
            f"⚠️  PRETRADE max gross exposure ratio ({ApexConfig.PRETRADE_MAX_GROSS_EXPOSURE_RATIO}) is very strict"
        )

    if not (0 < ApexConfig.PAPER_STARTUP_RISK_MISMATCH_RATIO <= 2):
        warnings.append(
            f"⚠️  PAPER startup risk mismatch ratio ({ApexConfig.PAPER_STARTUP_RISK_MISMATCH_RATIO}) should be in (0,2]"
        )

    if not (0 < ApexConfig.PAPER_STARTUP_PERFORMANCE_REBASE_RATIO <= 2):
        warnings.append(
            f"⚠️  PAPER startup performance rebase ratio ({ApexConfig.PAPER_STARTUP_PERFORMANCE_REBASE_RATIO}) should be in (0,2]"
        )

    if ApexConfig.OPTIONS_FAILED_TRADE_RETRY_COOLDOWN_SECONDS < 60:
        warnings.append(
            f"⚠️  OPTIONS failed-trade retry cooldown ({ApexConfig.OPTIONS_FAILED_TRADE_RETRY_COOLDOWN_SECONDS}s) is very short"
        )

    for warning in warnings:
        _config_logger.warning(warning)

    if errors:
        for error in errors:
            _config_logger.error(error)
        return False

    return True


def assert_live_trading_confirmation() -> None:
    """Raise when live trading is enabled without explicit confirmation."""
    if ApexConfig.LIVE_TRADING and not ApexConfig.LIVE_TRADING_CONFIRMED:
        raise RuntimeError(
            "Unsafe startup blocked: APEX_LIVE_TRADING=true requires "
            "APEX_LIVE_TRADING_CONFIRMED=true."
        )


if __name__ == "__main__":
    print(f"APEX Trading System Configuration v{ApexConfig.VERSION}")
    print(f"Mode: {'LIVE' if ApexConfig.LIVE_TRADING else 'SIMULATION'}")
    print(f"IBKR: {ApexConfig.IBKR_HOST}:{ApexConfig.IBKR_PORT}")
    print(f"Capital: ${ApexConfig.INITIAL_CAPITAL:,}")
    print()

    if validate_config():
        print("✅ Configuration validated successfully")
    else:
        print("❌ Configuration validation failed")

    BROKER_MODE = "both"
