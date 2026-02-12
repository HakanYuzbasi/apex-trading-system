"""
Production-Grade Trading Configuration with Pydantic Validation
Critical Fix #4: Secure credential management

Usage:
    from config_secure import TradingConfig
    config = TradingConfig()  # Auto-loads from .env
"""

from pydantic import BaseSettings, SecretStr, validator, Field
from typing import Optional, Literal
import os
from pathlib import Path


class TradingConfig(BaseSettings):
    """Validated configuration with secret management and risk limits"""
    
    # ============================================
    # TRADING MODE & ENVIRONMENT
    # ============================================
    MODE: Literal["PAPER", "LIVE"] = Field(
        default="PAPER",
        description="Trading mode: PAPER for testing, LIVE for real money"
    )
    
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    
    # ============================================
    # INTERACTIVE BROKERS (IBKR) CONFIGURATION
    # ============================================
    IBKR_HOST: str = Field(default="127.0.0.1")
    IBKR_PORT: int = Field(
        default=7497,
        description="7497 for paper trading, 7496 for live trading"
    )
    IBKR_CLIENT_ID: int = Field(default=1)
    IBKR_PAPER_ACCOUNT: str = Field(default="")
    IBKR_LIVE_ACCOUNT: SecretStr = Field(default=SecretStr(""))
    IBKR_TIMEOUT: int = Field(default=30, description="Connection timeout in seconds")
    
    # ============================================
    # ALPACA (CRYPTO PAPER TRADING)
    # ============================================
    ALPACA_API_KEY: SecretStr = Field(default=SecretStr(""))
    ALPACA_SECRET_KEY: SecretStr = Field(default=SecretStr(""))
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets")
    
    # ============================================
    # RISK MANAGEMENT & CIRCUIT BREAKERS
    # ============================================
    MAX_POSITION_SIZE_PCT: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Maximum position size as % of portfolio (1-50%)"
    )
    
    MAX_DAILY_LOSS_PCT: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Maximum daily loss % before stopping trading (1-20%)"
    )
    
    CIRCUIT_BREAKER_THRESHOLD: float = Field(
        default=0.15,
        ge=0.05,
        le=0.50,
        description="Total loss % that triggers emergency stop (5-50%)"
    )
    
    MAX_OPEN_POSITIONS: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of simultaneous open positions"
    )
    
    MAX_LEVERAGE: float = Field(
        default=1.0,
        ge=1.0,
        le=4.0,
        description="Maximum portfolio leverage (1x-4x)"
    )
    
    # ============================================
    # ADVANCED EXECUTION SETTINGS
    # ============================================
    USE_ADVANCED_EXECUTION: bool = Field(default=True)
    SLIPPAGE_TOLERANCE_BPS: int = Field(
        default=10,
        description="Maximum acceptable slippage in basis points"
    )
    
    # ============================================
    # DATABASE & REDIS
    # ============================================
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # ============================================
    # LOGGING & MONITORING
    # ============================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    LOG_FILE: Path = Field(default=Path("apex_trading.log"))
    SENTRY_DSN: Optional[SecretStr] = Field(default=None)
    
    # ============================================
    # VALIDATORS
    # ============================================
    
    @validator("MODE")
    def validate_mode(cls, v):
        if v not in ["PAPER", "LIVE"]:
            raise ValueError("MODE must be 'PAPER' or 'LIVE'")
        return v
    
    @validator("IBKR_PORT")
    def validate_ibkr_port(cls, v, values):
        mode = values.get("MODE", "PAPER")
        if mode == "PAPER" and v not in [4001, 4002, 7497]:
            raise ValueError(f"PAPER mode requires port 7497 (TWS) or 4001/4002 (Gateway), got {v}")
        elif mode == "LIVE" and v not in [4000, 7496]:
            raise ValueError(f"LIVE mode requires port 7496 (TWS) or 4000 (Gateway), got {v}")
        return v
    
    @validator("MAX_DAILY_LOSS_PCT")
    def validate_daily_loss(cls, v):
        if v > 0.20:
            raise ValueError("MAX_DAILY_LOSS_PCT should not exceed 20% for safety")
        return v
    
    @validator("CIRCUIT_BREAKER_THRESHOLD")
    def validate_circuit_breaker(cls, v, values):
        daily_loss = values.get("MAX_DAILY_LOSS_PCT", 0.05)
        if v < daily_loss:
            raise ValueError(
                f"CIRCUIT_BREAKER_THRESHOLD ({v}) must be >= MAX_DAILY_LOSS_PCT ({daily_loss})"
            )
        return v
    
    @validator("IBKR_LIVE_ACCOUNT", always=True)
    def validate_live_account(cls, v, values):
        mode = values.get("MODE")
        if mode == "LIVE" and not v.get_secret_value():
            raise ValueError("IBKR_LIVE_ACCOUNT is required when MODE=LIVE")
        return v
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True
        
        # Allow extra fields for backward compatibility
        extra = "ignore"


# ============================================
# SINGLETON INSTANCE
# ============================================

_config: Optional[TradingConfig] = None

def get_config() -> TradingConfig:
    """Get singleton config instance"""
    global _config
    if _config is None:
        _config = TradingConfig()
    return _config


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print(f"\n{'='*50}")
    print("APEX TRADING CONFIGURATION")
    print(f"{'='*50}")
    print(f"Mode: {config.MODE}")
    print(f"Environment: {config.ENVIRONMENT}")
    print(f"IBKR Port: {config.IBKR_PORT}")
    print(f"Max Position Size: {config.MAX_POSITION_SIZE_PCT:.1%}")
    print(f"Max Daily Loss: {config.MAX_DAILY_LOSS_PCT:.1%}")
    print(f"Circuit Breaker: {config.CIRCUIT_BREAKER_THRESHOLD:.1%}")
    print(f"Max Open Positions: {config.MAX_OPEN_POSITIONS}")
    print(f"{'='*50}\n")
config_secure.py
