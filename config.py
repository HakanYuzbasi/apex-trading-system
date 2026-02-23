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


class ApexConfig:
    # --- EXECUTION & SMART ORDER ROUTING ---
    SOR_ENABLED = True               # Use Limit mid-pegging instead of Market orders
    SOR_MAX_URGENCY_STEPS = 3        # Number of times to adjust price toward the ask/bid
    SOR_STEP_DELAY_SECONDS = 10      # How long to wait at each price level before adjusting

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
    ENVIRONMENT: str = os.getenv("APEX_ENVIRONMENT", "prod")
    
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
        os.getenv("APEX_IBKR_PORT", os.getenv("IBKR_PORT", "7497"))
    )  # 7497 = Paper, 7496 = Live
    import random
    IBKR_CLIENT_ID: int = int(
        os.getenv(
            "APEX_IBKR_CLIENT_ID",
            os.getenv("IBKR_CLIENT_ID", str(random.randint(10, 99))),
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
    CRYPTO_ALWAYS_OPEN: bool = os.getenv("APEX_CRYPTO_ALWAYS_OPEN", "false").lower() == "true"
    CUSTOM_MARKET_SESSIONS = {
        # Example:
        # "FOREX": {"timezone": "America/New_York", "open": "17:00", "close": "17:00", "weekdays": [0,1,2,3,4,6]},
        # "EQUITY": {"timezone": "America/New_York", "open": "09:30", "close": "16:00", "weekdays": [0,1,2,3,4]},
    }
    USE_DATA_FALLBACK_FOR_PRICES: bool = os.getenv("APEX_USE_DATA_FALLBACK_FOR_PRICES", "true").lower() == "true"
    # ✅ NEW (Paper-safe): Allow offline IBKR mode when connection fails
    IBKR_ALLOW_OFFLINE: bool = os.getenv("APEX_IBKR_ALLOW_OFFLINE", "false").lower() == "true"
    # ✅ NEW (Observability): Prefer data-provider fallback when market is closed (weekends)
    PRICE_FALLBACK_WHEN_MARKET_CLOSED: bool = os.getenv("APEX_PRICE_FALLBACK_WHEN_MARKET_CLOSED", "true").lower() == "true"
    # ✅ NEW (Paper-safe): Toggle Data Watchdog (disable when running offline)
    DATA_WATCHDOG_ENABLED: bool = os.getenv("APEX_DATA_WATCHDOG_ENABLED", "true").lower() == "true"

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
    ALPACA_DISCOVER_CRYPTO_SYMBOLS: bool = (
        os.getenv("APEX_ALPACA_DISCOVER_CRYPTO_SYMBOLS", "true").lower() == "true"
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
        os.getenv("APEX_BROKER_EQUITY_REFRESH_INTERVAL_SECONDS", "5")
    )
    BROKER_POSITIONS_REFRESH_INTERVAL_SECONDS: int = int(
        os.getenv("APEX_BROKER_POSITIONS_REFRESH_INTERVAL_SECONDS", "5")
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
    MAX_POSITIONS = 10  # Increased from 15 for better capital utilization
    MAX_SHARES_PER_POSITION = 500  # Cap max shares per position
    MIN_HEDGE_NOTIONAL = 50_000  # Only hedge positions larger than $50k to save on costs
    
    # ═══════════════════════════════════════════════════════════════
    # RISK LIMITS
    # ═══════════════════════════════════════════════════════════════
    MAX_DAILY_LOSS = 0.03  # 3% max daily loss (Moderate risk profile)
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    MAX_SECTOR_EXPOSURE = 0.20  # 20% max per sector for proper diversification (was 0.50)
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
    KILL_SWITCH_LOGIC: str = os.getenv("APEX_KILL_SWITCH_LOGIC", "OR")
    KILL_SWITCH_MIN_POINTS: int = int(os.getenv("APEX_KILL_SWITCH_MIN_POINTS", "20"))
    KILL_SWITCH_HISTORICAL_MDD_BASELINE: float = float(
        os.getenv("APEX_KILL_SWITCH_HISTORICAL_MDD_BASELINE", "0.08")
    )

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
        os.getenv("APEX_PRETRADE_MAX_ORDER_NOTIONAL", "250000")
    )
    PRETRADE_MAX_ORDER_SHARES: int = int(
        os.getenv("APEX_PRETRADE_MAX_ORDER_SHARES", "10000")
    )
    PRETRADE_MAX_PRICE_DEVIATION_BPS: float = float(
        os.getenv("APEX_PRETRADE_MAX_PRICE_DEVIATION_BPS", "250")
    )
    PRETRADE_MAX_PARTICIPATION_RATE: float = float(
        os.getenv("APEX_PRETRADE_MAX_PARTICIPATION_RATE", "0.10")
    )
    PRETRADE_MAX_GROSS_EXPOSURE_RATIO: float = float(
        os.getenv("APEX_PRETRADE_MAX_GROSS_EXPOSURE_RATIO", "2.0")
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
    MIN_SIGNAL_THRESHOLD = 0.18      # Lowered from 0.25 to let moderate signals through
    MIN_CONFIDENCE = 0.55            # Lowered from 0.35 (soft filters stack 0.85x*0.90x penalties)
    CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER: float = float(
        os.getenv("APEX_CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER", "0.60")
    )
    CRYPTO_CONFIDENCE_THRESHOLD_MULTIPLIER: float = float(
        os.getenv("APEX_CRYPTO_CONFIDENCE_THRESHOLD_MULTIPLIER", "0.70")
    )
    CRYPTO_ROTATION_ENABLED: bool = os.getenv(
        "APEX_CRYPTO_ROTATION_ENABLED", "true"
    ).lower() == "true"
    CRYPTO_ROTATION_TOP_N: int = int(
        os.getenv("APEX_CRYPTO_ROTATION_TOP_N", "10")
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
    FORCE_RETRAIN = False            # Set to True to force model retraining on startup

    # Regime-based entry thresholds (High filter)
    SIGNAL_THRESHOLDS_BY_REGIME = {
        'strong_bull': 0.15,    # Lowered from 0.20
        'bull': 0.18,          # Lowered from 0.23
        'neutral': 0.15,       # Lowered from 0.28
        'bear': 0.18,          # Lowered from 0.25
        'strong_bear': 0.15,   # Lowered from 0.22
        'volatile': 0.22       # Lowered from 0.30
    }

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

    # Signal Integrity Monitor
    SIGNAL_INTEGRITY_ENABLED = True    # Monitor signal stream for anomalies
    STUCK_SIGNAL_THRESHOLD = 10        # Alert after N identical signals
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
    MAX_ACCEPTABLE_SLIPPAGE_BPS = 15     # Flag symbols with avg slippage > 15bps
    CRITICAL_SLIPPAGE_BPS = 30           # Reduce size 20% above this
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
        os.getenv("APEX_EXECUTION_MAX_SPREAD_BPS_CRYPTO", "26")
    )
    EXECUTION_EDGE_GATE_ENABLED: bool = os.getenv(
        "APEX_EXECUTION_EDGE_GATE_ENABLED",
        "true",
    ).strip().lower() == "true"
    EXECUTION_SIGNAL_TO_EDGE_BPS: float = float(
        os.getenv("APEX_EXECUTION_SIGNAL_TO_EDGE_BPS", "80")
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY", "8")
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_FX: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_FX", "6")
    )
    EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO: float = float(
        os.getenv("APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO", "4")
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
    MAX_PORTFOLIO_CORRELATION = 0.50  # Max average portfolio correlation
    CORRELATION_LOOKBACK = 60  # Days for correlation calculation

    # Execution
    LARGE_ORDER_THRESHOLD = 50000  # Dollar value threshold for TWAP/VWAP execution
    USE_LIVE_MARKET_DATA = False  # Set True for live trading with data subscription

    # Advanced Risk
    VAR_CONFIDENCE = 0.95  # VaR confidence level
    MAX_PORTFOLIO_VAR = 0.03  # Maximum daily VaR (3%)
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
    # TIMING & EXECUTION
    # ═══════════════════════════════════════════════════════════════
    CHECK_INTERVAL_SECONDS = 30  # Check symbols every 30 seconds
    TRADE_COOLDOWN_SECONDS = 120  # Reduced from 240s; doubles opportunity window per symbol
    
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
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = "logs/apex.log"
    # ✅ NEW: Log rotation controls (bytes)
    LOG_MAX_BYTES = int(os.getenv("APEX_LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5MB
    LOG_BACKUP_COUNT = int(os.getenv("APEX_LOG_BACKUP_COUNT", "5"))

    # Health check staleness (seconds). Mark backend offline if state is older than this.
    HEALTH_STALENESS_SECONDS = int(os.getenv("APEX_HEALTH_STALENESS_SECONDS", "30"))
    
    # ═══════════════════════════════════════════════════════════════
    # UNIVERSE SELECTION
    # ═══════════════════════════════════════════════════════════════
    UNIVERSE_MODE = "SP500"  # Options: "SP500", "NASDAQ100", "CUSTOM"
    
    # S&P 500 Top Liquid Stocks & Multi-Asset Universe
    # ------------------------------------------------
    # 1. Major Indices
    INDICES = ["SPY", "QQQ", "IWM", "DIA"]

    # 2. Forex Pairs (Major G10)
    FOREX_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
        "USD/CAD", "NZD/USD"
    ]

    # 3. Crypto Pairs (Top Liquid, expanded for Alpaca crypto paper)
    CRYPTO_PAIRS = [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "DOGE/USD",
        "AVAX/USD",
        "LINK/USD",
        "MATIC/USD",
        "ADA/USD",
        "XRP/USD",
        "DOT/USD",
        "LTC/USD",
        "BCH/USD",
        "XLM/USD",
        "ETC/USD",
        "AAVE/USD",
        "UNI/USD",
    ]
    EXTRA_CRYPTO_PAIRS = [
        token.strip().upper()
        for token in os.getenv("APEX_EXTRA_CRYPTO_PAIRS", "").split(",")
        if token.strip()
    ]
    if EXTRA_CRYPTO_PAIRS:
        CRYPTO_PAIRS = list(dict.fromkeys(CRYPTO_PAIRS + EXTRA_CRYPTO_PAIRS))

    # 4. Top 100 S&P 500 Components (aligned with SECTOR_MAP)
    # Trimmed to exactly 85 to keep total universe at 100 (4 Indices + 7 Forex + 4 Crypto + 85 Stocks)
    SP500_TOP_100 = [
        # Technology (Top 15)
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CSCO", "ADBE",
        "CRM", "ACN", "AMD", "INTC", "IBM", "QCOM", "TXN",
        # Financials (Top 6)
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
        # Healthcare (Top 6)
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
        # Consumer (Top 6)
        "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "DLTR",
        # Industrials (Top 7)
        "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE",
        # Energy (Top 5)
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC",
        # Materials (Top 5)
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "ALB",
        # Communication (Top 6)
        "NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ",
        # Real Estate & Utilities (Top 5)
        "AMT", "PLD", "CCI", "EQIX", "NEE",
        # Commodities (Top 4)
        "GLD", "SLV", "USO", "UNG"
    ]

    # Combine all into master universe
    SYMBOLS = list(set(INDICES + FOREX_PAIRS + CRYPTO_PAIRS + SP500_TOP_100))

    # Backtesting-only symbols (kept in universe, excluded from IBKR paper execution)
    BACKTEST_ONLY_SYMBOLS = {"SOL/USDT", "DOGE/USDT"}

    # Commodity symbols for special handling
    COMMODITY_SYMBOLS = {'GLD', 'SLV', 'USO', 'UNG', 'PALL'}
    
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

    if ApexConfig.PRETRADE_MAX_ORDER_SHARES <= 0:
        warnings.append("⚠️  PRETRADE max order shares must be > 0")

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
