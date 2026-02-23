"""
utils/constants.py - System-Wide Constants

Central location for all constants used throughout the trading system.
Avoid magic numbers scattered through the codebase.
"""

from enum import Enum
from typing import Dict, Set


# ============================================================================
# Time Constants
# ============================================================================

# Market Hours (EST)
MARKET_OPEN_HOUR = 9.5  # 9:30 AM
MARKET_CLOSE_HOUR = 16.0  # 4:00 PM
PRE_MARKET_START_HOUR = 4.0  # 4:00 AM
AFTER_HOURS_END_HOUR = 20.0  # 8:00 PM

# Extended hours for futures/commodities
EXTENDED_MARKET_OPEN_HOUR = 6.0
EXTENDED_MARKET_CLOSE_HOUR = 17.0

# Time periods
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
WEEKS_PER_YEAR = 52


# ============================================================================
# Trading Constants
# ============================================================================

# Default commission rates
DEFAULT_COMMISSION = 0.0  # Most brokers now offer commission-free trading
IBKR_COMMISSION_PER_SHARE = 0.005
IBKR_MIN_COMMISSION = 1.0
IBKR_MAX_COMMISSION_PCT = 0.01  # 1% of trade value

# Slippage estimates
DEFAULT_SLIPPAGE_BPS = 5  # 5 basis points
HIGH_VOLUME_SLIPPAGE_BPS = 2
LOW_VOLUME_SLIPPAGE_BPS = 15

# Order sizing
MIN_ORDER_VALUE = 100.0  # $100 minimum order
MAX_ORDER_VALUE = 1_000_000.0  # $1M maximum order
ROUND_LOT_SIZE = 100  # Standard round lot

# Position limits
MAX_POSITION_PCT = 0.10  # 10% of portfolio per position
MAX_SECTOR_EXPOSURE_PCT = 0.40  # 40% sector limit
MAX_SINGLE_STOCK_EXPOSURE = 0.05  # 5% single stock limit


# ============================================================================
# Risk Constants
# ============================================================================

# Daily limits
MAX_DAILY_LOSS_PCT = 0.03  # 3% daily loss limit
MAX_DRAWDOWN_PCT = 0.10  # 10% max drawdown
CIRCUIT_BREAKER_CONSECUTIVE_LOSSES = 5

# Volatility thresholds
LOW_VOLATILITY_THRESHOLD = 0.10  # 10% annualized
HIGH_VOLATILITY_THRESHOLD = 0.35  # 35% annualized
EXTREME_VOLATILITY_THRESHOLD = 0.50  # 50% annualized

# Gap protection
MAX_GAP_PCT = 0.05  # 5% gap down triggers halt


# ============================================================================
# Signal Constants
# ============================================================================

# Signal thresholds
SIGNAL_THRESHOLD_BULL = 0.30
SIGNAL_THRESHOLD_NEUTRAL = 0.40
SIGNAL_THRESHOLD_BEAR = 0.50
SIGNAL_THRESHOLD_VOLATILE = 0.55

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Regime definitions
class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


# ============================================================================
# Technical Analysis Constants
# ============================================================================

# Moving average periods
SMA_SHORT = 10
SMA_MEDIUM = 20
SMA_LONG = 50
SMA_VERY_LONG = 200

EMA_SHORT = 12
EMA_LONG = 26
EMA_SIGNAL = 9

# RSI parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2.0

# ATR
ATR_PERIOD = 14


# ============================================================================
# Data Constants
# ============================================================================

# Historical data
MIN_HISTORY_DAYS = 60
DEFAULT_HISTORY_DAYS = 252
MAX_HISTORY_DAYS = 1000

# Cache settings
CACHE_TTL_SECONDS = 60
CACHE_MAX_SIZE = 1000

# Data quality
MIN_DATA_POINTS = 20
MAX_MISSING_DATA_PCT = 0.10  # 10% max missing data


# ============================================================================
# API Constants
# ============================================================================

# Rate limits
API_RATE_LIMIT_PER_SECOND = 10
API_RATE_LIMIT_PER_MINUTE = 100

# Timeouts
CONNECTION_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 60
ORDER_TIMEOUT_SECONDS = 30

# Retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0


# ============================================================================
# IBKR Constants
# ============================================================================

# Connection
IBKR_DEFAULT_HOST = "127.0.0.1"
IBKR_PAPER_PORT = 7497
IBKR_LIVE_PORT = 7496
IBKR_DEFAULT_CLIENT_ID = 1

# Order types
class IBKROrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    MARKET_ON_CLOSE = "MOC"
    LIMIT_ON_CLOSE = "LOC"


# ============================================================================
# Symbol Constants
# ============================================================================

# Major indices
MAJOR_INDICES: Set[str] = {
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"
}

# Sector ETFs
SECTOR_ETFS: Set[str] = {
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLU",
    "XLP", "XLY", "XLB", "XLRE", "XLC"
}

# Commodity ETFs
COMMODITY_ETFS: Set[str] = {
    "GLD", "SLV", "USO", "UNG", "GDX", "GDXJ"
}

# Bond ETFs
BOND_ETFS: Set[str] = {
    "TLT", "IEF", "SHY", "AGG", "BND", "HYG", "LQD"
}

# Volatility
VOLATILITY_PRODUCTS: Set[str] = {
    "VXX", "UVXY", "SVXY", "VIXY"
}

# Sector mapping for exposure tracking
SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "ORCL": "Technology", "IBM": "Technology", "CSCO": "Technology",

    # Financial
    "JPM": "Financial", "BAC": "Financial", "WFC": "Financial",
    "GS": "Financial", "MS": "Financial", "C": "Financial",
    "BRK.B": "Financial", "V": "Financial", "MA": "Financial",

    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",

    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "NKE": "Consumer", "MCD": "Consumer", "SBUX": "Consumer",

    # Industrial
    "BA": "Industrial", "CAT": "Industrial", "GE": "Industrial",
    "MMM": "Industrial", "HON": "Industrial", "UPS": "Industrial",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",

    # ETFs
    "XLF": "Financial", "XLK": "Technology", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrial", "XLU": "Utilities",
    "XLP": "Consumer Staples", "XLY": "Consumer Discretionary",
    "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication",
}


# ============================================================================
# Status Enums
# ============================================================================

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SystemStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ============================================================================
# Formatting Constants
# ============================================================================

# Number formatting
PRICE_DECIMALS = 2
PERCENTAGE_DECIMALS = 2
QUANTITY_DECIMALS = 0

# Date formats
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIME_FORMAT = "%H:%M:%S"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
