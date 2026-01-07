"""
utils/timezone.py - Professional timezone handling for trading
Handles US market hours with proper DST support
"""

import logging
from datetime import datetime, time
from typing import Tuple
import pytz

logger = logging.getLogger(__name__)


class TradingHours:
    """
    Handle US market trading hours with proper timezone support.

    US Stock Market Hours (NYSE/NASDAQ):
    - Regular: 9:30 AM - 4:00 PM ET
    - Pre-market: 4:00 AM - 9:30 AM ET
    - After-hours: 4:00 PM - 8:00 PM ET
    """

    # Timezones
    ET = pytz.timezone('America/New_York')
    UTC = pytz.UTC

    # Regular market hours (ET)
    MARKET_OPEN = time(9, 30)  # 9:30 AM ET
    MARKET_CLOSE = time(16, 0)  # 4:00 PM ET

    # Extended hours
    PRE_MARKET_OPEN = time(4, 0)  # 4:00 AM ET
    AFTER_HOURS_CLOSE = time(20, 0)  # 8:00 PM ET

    # Market holidays (2026 - update annually)
    MARKET_HOLIDAYS_2026 = [
        datetime(2026, 1, 1),  # New Year's Day
        datetime(2026, 1, 19),  # Martin Luther King Jr. Day
        datetime(2026, 2, 16),  # Presidents Day
        datetime(2026, 4, 3),  # Good Friday
        datetime(2026, 5, 25),  # Memorial Day
        datetime(2026, 7, 3),  # Independence Day (observed)
        datetime(2026, 9, 7),  # Labor Day
        datetime(2026, 11, 26),  # Thanksgiving
        datetime(2026, 12, 25),  # Christmas
    ]

    @classmethod
    def get_current_et_time(cls) -> datetime:
        """
        Get current time in Eastern Time.

        Returns:
            Current datetime in ET (handles DST automatically)
        """
        return datetime.now(cls.ET)

    @classmethod
    def is_market_open(cls, dt: datetime = None, include_extended: bool = False) -> bool:
        """
        Check if market is currently open.

        Args:
            dt: Datetime to check (default: now)
            include_extended: Include pre-market and after-hours

        Returns:
            True if market is open
        """
        if dt is None:
            dt = cls.get_current_et_time()
        elif dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = cls.UTC.localize(dt).astimezone(cls.ET)
        else:
            dt = dt.astimezone(cls.ET)

        # Check if weekend
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check if holiday
        date_only = dt.date()
        for holiday in cls.MARKET_HOLIDAYS_2026:
            if date_only == holiday.date():
                return False

        # Check hours
        current_time = dt.time()

        if include_extended:
            return cls.PRE_MARKET_OPEN <= current_time < cls.AFTER_HOURS_CLOSE
        else:
            return cls.MARKET_OPEN <= current_time < cls.MARKET_CLOSE

    @classmethod
    def get_next_market_open(cls, dt: datetime = None) -> datetime:
        """
        Get the next market open time.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Next market open datetime in ET
        """
        if dt is None:
            dt = cls.get_current_et_time()
        elif dt.tzinfo is None:
            dt = cls.UTC.localize(dt).astimezone(cls.ET)
        else:
            dt = dt.astimezone(cls.ET)

        # Start from next day if after market close
        if dt.time() >= cls.MARKET_CLOSE:
            dt = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            dt += timedelta(days=1)
        else:
            dt = dt.replace(hour=9, minute=30, second=0, microsecond=0)

        # Skip weekends and holidays
        while dt.weekday() >= 5 or any(dt.date() == h.date() for h in cls.MARKET_HOLIDAYS_2026):
            dt += timedelta(days=1)

        return dt

    @classmethod
    def get_next_market_close(cls, dt: datetime = None) -> datetime:
        """
        Get the next market close time.

        Args:
            dt: Starting datetime (default: now)

        Returns:
            Next market close datetime in ET
        """
        if dt is None:
            dt = cls.get_current_et_time()
        elif dt.tzinfo is None:
            dt = cls.UTC.localize(dt).astimezone(cls.ET)
        else:
            dt = dt.astimezone(cls.ET)

        # Set to today's close
        close_dt = dt.replace(hour=16, minute=0, second=0, microsecond=0)

        # If already past close, move to next day
        if dt >= close_dt:
            close_dt += timedelta(days=1)

        # Skip weekends and holidays
        while close_dt.weekday() >= 5 or any(close_dt.date() == h.date() for h in cls.MARKET_HOLIDAYS_2026):
            close_dt += timedelta(days=1)

        return close_dt

    @classmethod
    def seconds_until_market_open(cls) -> float:
        """Get seconds until next market open."""
        now = cls.get_current_et_time()
        next_open = cls.get_next_market_open(now)
        return (next_open - now).total_seconds()

    @classmethod
    def seconds_until_market_close(cls) -> float:
        """Get seconds until next market close."""
        now = cls.get_current_et_time()
        next_close = cls.get_next_market_close(now)
        return (next_close - now).total_seconds()

    @classmethod
    def get_market_session_info(cls) -> dict:
        """
        Get detailed information about current market session.

        Returns:
            Dict with market status, times, and session info
        """
        now = cls.get_current_et_time()
        is_open = cls.is_market_open(now)
        is_extended = cls.is_market_open(now, include_extended=True)

        status = "CLOSED"
        if is_open:
            status = "OPEN"
        elif is_extended and now.time() < cls.MARKET_OPEN:
            status = "PRE_MARKET"
        elif is_extended and now.time() >= cls.MARKET_CLOSE:
            status = "AFTER_HOURS"

        return {
            'current_time_et': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'status': status,
            'is_open': is_open,
            'is_extended_hours': is_extended and not is_open,
            'next_open': cls.get_next_market_open(now).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'next_close': cls.get_next_market_close(now).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'seconds_until_open': cls.seconds_until_market_open() if not is_open else 0,
            'seconds_until_close': cls.seconds_until_market_close() if is_open else 0
        }


# Convenience functions
def is_market_open(include_extended: bool = False) -> bool:
    """Check if market is currently open."""
    return TradingHours.is_market_open(include_extended=include_extended)


def get_market_time() -> datetime:
    """Get current market time (ET)."""
    return TradingHours.get_current_et_time()


from datetime import timedelta

def format_time_until(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "now"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{int(seconds)}s"
