"""
config.py - APEX Trading System Configuration
PRODUCTION SETTINGS - Conservative and Safe
"""

import os
from pathlib import Path


class ApexConfig:
    """Central configuration for APEX Trading System."""
    
    # System Info
    SYSTEM_NAME = "APEX Trading System"
    VERSION = "2.0.0-PRODUCTION"
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING MODE
    # ═══════════════════════════════════════════════════════════════
    LIVE_TRADING = True  # Set to False for simulation mode
    
    # ═══════════════════════════════════════════════════════════════
    # IBKR CONNECTION
    # ═══════════════════════════════════════════════════════════════
    IBKR_HOST = '127.0.0.1'
    IBKR_PORT = 7497  # 7497 = Paper Trading, 7496 = Live Trading
    IBKR_CLIENT_ID = 1
    
    # ═══════════════════════════════════════════════════════════════
    # CAPITAL & POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    INITIAL_CAPITAL = 1_100_000  # $1.1M starting capital
    POSITION_SIZE_USD = 5_000  # $5K per position (0.45% of capital)
    MAX_POSITIONS = 15  # Maximum concurrent positions
    MAX_SHARES_PER_POSITION = 200  # ✅ NEW: Cap max shares per position
    
    # ═══════════════════════════════════════════════════════════════
    # RISK LIMITS
    # ═══════════════════════════════════════════════════════════════
    MAX_DAILY_LOSS = 0.02  # 2% max daily loss
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    MAX_SECTOR_EXPOSURE = 0.40  # ✅ NEW: 40% max per sector
    
    # ═══════════════════════════════════════════════════════════════
    # SIGNAL THRESHOLDS
    # ═══════════════════════════════════════════════════════════════
    MIN_SIGNAL_THRESHOLD = 0.45  # Minimum signal strength (0-1) - Higher = fewer trades
    MIN_CONFIDENCE = 0.30  # Minimum confidence for trade execution
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING HOURS (EST)
    # ═══════════════════════════════════════════════════════════════
    TRADING_HOURS_START = 9.5  # 9:30 AM EST (market open)
    TRADING_HOURS_END = 16.0   # 4:00 PM EST (market close)
    
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
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if ApexConfig.POSITION_SIZE_USD > ApexConfig.INITIAL_CAPITAL * 0.01:
        errors.append(f"⚠️  Position size (${ApexConfig.POSITION_SIZE_USD}) > 1% of capital")
    
    if ApexConfig.MAX_POSITIONS * ApexConfig.POSITION_SIZE_USD > ApexConfig.INITIAL_CAPITAL:
        errors.append(f"⚠️  Max exposure ({ApexConfig.MAX_POSITIONS} * ${ApexConfig.POSITION_SIZE_USD}) > capital")
    
    if ApexConfig.MIN_SIGNAL_THRESHOLD < 0.3:
        errors.append(f"⚠️  Signal threshold too low ({ApexConfig.MIN_SIGNAL_THRESHOLD}) - Risk of false signals")
    
    if ApexConfig.TRADE_COOLDOWN_SECONDS < 60:
        errors.append(f"⚠️  Cooldown too short ({ApexConfig.TRADE_COOLDOWN_SECONDS}s) - Risk of overtrading")
    
    if errors:
        print("\n".join(errors))
        return False
    
    return True


if __name__ == "__main__":
    if validate_config():
        print("✅ Configuration validated successfully")
    else:
        print("❌ Configuration validation failed")
