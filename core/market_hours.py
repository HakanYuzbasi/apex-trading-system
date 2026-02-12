"""
core/market_hours.py
Market session logic and trading hours validation.
"""

from datetime import datetime
import pytz
from typing import Union, Dict, Any, Optional

from config import ApexConfig
from core.symbols import AssetClass, ParsedSymbol, parse_symbol

def _to_eastern(timestamp: datetime) -> datetime:
    """Convert timestamp to US/Eastern."""
    try:
        eastern = pytz.timezone("America/New_York")
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=pytz.UTC).astimezone(eastern)
        return timestamp.astimezone(eastern)
    except Exception:
        return timestamp

def _parse_time(value: str):
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    return hour, minute

def _in_custom_session(dt: datetime, session: Dict[str, Any]) -> bool:
    """Check if time is within a custom session definition."""
    timezone = session.get("timezone", "America/New_York")
    try:
        tz = pytz.timezone(timezone)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC).astimezone(tz)
        else:
            dt = dt.astimezone(tz)
    except Exception:
        pass

    weekdays = session.get("weekdays")
    if weekdays is not None and dt.weekday() not in weekdays:
        return False

    open_time = session.get("open")
    close_time = session.get("close")
    if not open_time or not close_time:
        return True

    try:
        open_h, open_m = _parse_time(open_time)
        close_h, close_m = _parse_time(close_time)

        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        now_minutes = dt.hour * 60 + dt.minute

        if open_minutes == close_minutes:
            return True
        if open_minutes < close_minutes:
            return open_minutes <= now_minutes < close_minutes
        
        # Crosses midnight
        return now_minutes >= open_minutes or now_minutes < close_minutes
    except ValueError:
        return False

def is_market_open(
    raw_or_parsed: Union[str, ParsedSymbol], 
    timestamp: datetime, 
    assume_daily: bool = False
) -> bool:
    """
    Check if market is open for the given asset.
    
    Args:
        raw_or_parsed: Symbol string or ParsedSymbol
        timestamp: Datetime to check
        assume_daily: If True, ignores time of day (useful for daily execution checks)
    """
    # Global overrides
    if getattr(ApexConfig, "MARKET_ALWAYS_OPEN", False):
        return True

    parsed = raw_or_parsed if isinstance(raw_or_parsed, ParsedSymbol) else parse_symbol(raw_or_parsed)

    # Asset-specific overrides
    if parsed.asset_class == AssetClass.CRYPTO:
        if getattr(ApexConfig, "CRYPTO_ALWAYS_OPEN", False):
            return True
        sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("CRYPTO")
        if sessions:
            return _in_custom_session(timestamp, sessions)
        return True

    dt = _to_eastern(timestamp)
    weekday = dt.weekday()
    is_midnight = dt.hour == 0 and dt.minute == 0 and dt.second == 0

    if parsed.asset_class == AssetClass.FOREX:
        if getattr(ApexConfig, "FX_ALWAYS_OPEN", False):
            return True
        sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("FOREX")
        if sessions:
            return _in_custom_session(timestamp, sessions)
            
        if assume_daily or is_midnight:
            # Forex closed on Saturday
            return weekday != 5
            
        # Standard Forex: 
        # Opens Sunday 5PM EST (17:00)
        # Closes Friday 5PM EST (17:00)
        # Closed Saturday
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            return dt.hour >= 17
        if weekday == 4:  # Friday
            return dt.hour < 17
        return True

    # Equity / Options
    sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("EQUITY")
    if sessions:
        return _in_custom_session(timestamp, sessions)
        
    if assume_daily or is_midnight:
        return weekday < 5  # Mon-Fri
        
    if weekday >= 5: # Sat/Sun
        return False
        
    # NYSE/NASDAQ: 9:30 - 16:00
    market_open = (dt.hour > 9) or (dt.hour == 9 and dt.minute >= 30)
    market_close = dt.hour < 16
    return market_open and market_close
