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
from pathlib import Path
from typing import Set, Dict, List


class ApexConfig:
    """
    Central configuration for APEX Trading System.

    All settings can be overridden via environment variables prefixed with APEX_.
    Example: APEX_LIVE_TRADING=false will disable live trading.
    """

    # System Info
    SYSTEM_NAME: str = "APEX Trading System"
    VERSION: str = "2.0.0-PRODUCTION"
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING MODE
    # ═══════════════════════════════════════════════════════════════
    LIVE_TRADING: bool = os.getenv("APEX_LIVE_TRADING", "true").lower() == "true"

    # ═══════════════════════════════════════════════════════════════
    # IBKR CONNECTION
    # ═══════════════════════════════════════════════════════════════
    IBKR_HOST: str = os.getenv("APEX_IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = int(os.getenv("APEX_IBKR_PORT", "7497"))  # 7497 = Paper, 7496 = Live
    IBKR_CLIENT_ID: int = int(os.getenv("APEX_IBKR_CLIENT_ID", "1"))

    # ═══════════════════════════════════════════════════════════════
    # CAPITAL & POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    INITIAL_CAPITAL: int = int(os.getenv("APEX_INITIAL_CAPITAL", "1100000"))
    POSITION_SIZE_USD = 5_000  # $5K per position (0.45% of capital)
    MAX_POSITIONS = 15  # Maximum concurrent positions
    MAX_SHARES_PER_POSITION = 200  # ✅ NEW: Cap max shares per position
    
    # ═══════════════════════════════════════════════════════════════
    # RISK LIMITS
    # ═══════════════════════════════════════════════════════════════
    MAX_DAILY_LOSS = 0.03  # 3% max daily loss (Moderate risk profile)
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    MAX_SECTOR_EXPOSURE = 0.40  # 40% max per sector

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
    # SIGNAL THRESHOLDS
    # ═══════════════════════════════════════════════════════════════
    MIN_SIGNAL_THRESHOLD = 0.40  # Default minimum signal strength
    MIN_CONFIDENCE = 0.35  # Minimum confidence for trade execution

    # ✅ Phase 3.1: Regime-based entry thresholds
    # Lower threshold in trending markets, higher in choppy/volatile markets
    SIGNAL_THRESHOLDS_BY_REGIME = {
        'strong_bull': 0.30,    # Easier to enter in strong trends
        'bull': 0.35,
        'neutral': 0.45,        # Stricter in sideways markets
        'bear': 0.40,
        'strong_bear': 0.35,
        'high_volatility': 0.55  # Much stricter in volatile markets
    }

    # ═══════════════════════════════════════════════════════════════
    # GOD LEVEL PARAMETERS (Moderate Risk Profile)
    # ═══════════════════════════════════════════════════════════════
    # Position Sizing (ATR-based)
    ATR_MULTIPLIER_STOP = 2.5  # Stop loss = ATR * this multiplier (Moderate)
    ATR_MULTIPLIER_PROFIT = 3.5  # Take profit = ATR * this multiplier (Moderate)
    TRAILING_STOP_ATR = 2.0  # Trailing stop = ATR * this multiplier (Moderate)
    USE_KELLY_SIZING = True  # Use Kelly criterion for position sizing
    KELLY_FRACTION = 0.6  # Kelly fraction (Moderate - 60%)

    # Enable advanced risk features
    USE_ATR_STOPS = True  # Use dynamic ATR-based stops instead of fixed percentages
    USE_ADVANCED_EXECUTION = True  # Use TWAP/VWAP for large orders
    USE_CORRELATION_MANAGER = True  # Enable correlation monitoring

    # Market Regime
    REGIME_LOOKBACK_DAYS = 60  # Days to analyze for regime detection
    REGIME_BULL_THRESHOLD = 0.05  # MA crossover threshold for bull regime
    REGIME_BEAR_THRESHOLD = -0.05  # MA crossover threshold for bear regime
    HIGH_VOL_THRESHOLD = 0.35  # Annualized volatility threshold for high-vol regime

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
    CHECK_INTERVAL_SECONDS = 60  # Check symbols every 60 seconds
    TRADE_COOLDOWN_SECONDS = 300  # ✅ NEW: 5 minutes between trades per symbol
    
    # ═══════════════════════════════════════════════════════════════
    # TRANSACTION COSTS
    # ═══════════════════════════════════════════════════════════════
    COMMISSION_PER_TRADE = 1.00  # ✅ NEW: $1 per trade (IBKR Pro)
    SLIPPAGE_BPS = 5  # 5 basis points slippage (0.05%)
    
    # ═══════════════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════════════
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = "logs/apex.log"
    
    # ═══════════════════════════════════════════════════════════════
    # UNIVERSE SELECTION
    # ═══════════════════════════════════════════════════════════════
    UNIVERSE_MODE = "SP500"  # Options: "SP500", "NASDAQ100", "CUSTOM"
    
    # S&P 500 Top Liquid Stocks (deduplicated)
    SYMBOLS = [
        # Technology
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CSCO", "ADBE",
        "CRM", "ACN", "AMD", "INTC", "IBM", "QCOM", "TXN", "AMAT", "MU", "LRCX",

        # Financials
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB",

        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",

        # Consumer
        "AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "DG", "DLTR",

        # Industrials
        "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP",

        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",

        # Materials
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "DOW", "DD", "ALB", "CE",

        # Communication (removed duplicate GOOGL, META)
        "NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ", "CHTR", "EA",

        # Real Estate & Utilities
        "AMT", "PLD", "CCI", "EQIX", "PSA", "NEE", "DUK", "SO", "D", "AEP",

        # ETFs & Commodities (removed duplicate CRM, AMAT)
        "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "UNG", "PALL"
    ]

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
    MODELS_DIR = BASE_DIR / "models" / "saved"
    
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

    if ApexConfig.MIN_SIGNAL_THRESHOLD < 0.3:
        warnings.append(
            f"⚠️  Signal threshold ({ApexConfig.MIN_SIGNAL_THRESHOLD}) is low - risk of false signals"
        )

    if ApexConfig.TRADE_COOLDOWN_SECONDS < 60:
        warnings.append(
            f"⚠️  Cooldown ({ApexConfig.TRADE_COOLDOWN_SECONDS}s) is short - risk of overtrading"
        )

    # Print results
    for warning in warnings:
        print(warning)

    if errors:
        for error in errors:
            print(error)
        return False

    return True


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
