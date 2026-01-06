"""
config.py - Complete Configuration
STATE-OF-THE-ART SETTINGS
"""

from pathlib import Path


class ApexConfig:
    """Complete system configuration."""
    
    # System
    SYSTEM_NAME = "APEX Trading System"
    VERSION = "3.0.0-STATE-OF-THE-ART"
    
    # Trading Mode
    LIVE_TRADING = True
    
    # IBKR
    IBKR_HOST = '127.0.0.1'
    IBKR_PORT = 7497  # 7497=Paper, 7496=Live
    IBKR_CLIENT_ID = 1
    
    # Capital & Sizing
    INITIAL_CAPITAL = 1_100_000
    POSITION_SIZE_USD = 5_000
    MAX_POSITIONS = 15
    MAX_SHARES_PER_POSITION = 200
    
    # Risk
    MAX_DAILY_LOSS = 0.02
    MAX_DRAWDOWN = 0.10
    MAX_SECTOR_EXPOSURE = 0.40
    
    # Signals
    MIN_SIGNAL_THRESHOLD = 0.45
    MIN_CONFIDENCE = 0.30
    MIN_CONSENSUS = 0.50  # Model agreement
    
    # Trading Hours (EST)
    TRADING_HOURS_START = 9.5
    TRADING_HOURS_END = 16.0
    
    # Timing
    CHECK_INTERVAL_SECONDS = 60
    TRADE_COOLDOWN_SECONDS = 300  # 5 minutes
    
    # Transaction Costs
    COMMISSION_PER_TRADE = 1.00
    SLIPPAGE_BPS = 5
    
    # Advanced Features
    USE_ENSEMBLE_ML = True
    USE_REGIME_DETECTION = True
    USE_ADVANCED_EXECUTION = True
    RUN_STRESS_TESTS = True
    USE_SMART_ROUTING = True
    
    # ML Settings
    ML_VALIDATION_SPLITS = 5
    ML_LOOKBACK_DAYS = 1260  # 5 years
    FEATURE_SELECTION_TOP_N = 20
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/apex.log"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models" / "saved"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    
    # S&P 500 Top Liquid Stocks (Example - Replace with your full list)
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
        
        # Communication
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ", "CHTR", "EA",
        
        # Real Estate & Utilities
        "AMT", "PLD", "CCI", "EQIX", "PSA", "NEE", "DUK", "SO", "D", "AEP",
        
        # ETFs & Commodities
        "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "UNG", "PALL", "CRM", "AMAT"
    ]
    
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
        return cls.SECTOR_MAP.get(symbol, "Unknown")