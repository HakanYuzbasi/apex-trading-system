from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from core.symbols import AssetClass, parse_symbol

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency path
    import pandas_market_calendars as mcal
except ImportError:  # pragma: no cover - tested via fallback behavior
    mcal = None

from quant_system.core.market_clock import AlpacaMarketClock



class SessionManager:
    """
    Session oracle for asset-aware market open/close decisions.

    Crypto is always tradable. Equities and options use the NYSE session. Forex
    falls back to a simple 24/5 rule because this harness is focused on crypto
    plus US listed instruments.
    """

    _NEW_YORK = ZoneInfo("America/New_York")
    _NYSE_OPEN = time(9, 30)
    _NYSE_CLOSE = time(16, 0)

    def __init__(
        self, 
        *, 
        nyse_calendar_name: str = "NYSE",
        market_clock: AlpacaMarketClock | None = None
    ) -> None:
        self._market_clock = market_clock
        self._calendar = mcal.get_calendar(nyse_calendar_name) if mcal is not None else None
        if self._calendar is None and self._market_clock is None:
            logger.warning(
                "pandas_market_calendars and AlpacaMarketClock are unavailable; SessionManager is using a weekday 09:30-16:00 ET fallback"
            )
        elif self._calendar is None:
            logger.info("SessionManager using AlpacaMarketClock for NYSE session resolution.")

    def is_market_open(self, instrument_id: str, when: datetime | None = None) -> bool:
        asset_class = parse_symbol(instrument_id).asset_class
        timestamp = self._normalize_timestamp(when)

        if asset_class == AssetClass.CRYPTO:
            return True
        if asset_class == AssetClass.FOREX:
            return self._is_forex_open(timestamp)
        if asset_class in {AssetClass.EQUITY, AssetClass.OPTION}:
            return self._is_nyse_open(timestamp)
        return False

    def minutes_until_close(self, instrument_id: str, when: datetime | None = None) -> float | None:
        asset_class = parse_symbol(instrument_id).asset_class
        if asset_class == AssetClass.CRYPTO:
            return None

        timestamp = self._normalize_timestamp(when)
        if asset_class == AssetClass.FOREX:
            return None
        if asset_class not in {AssetClass.EQUITY, AssetClass.OPTION}:
            return None

        session = self._session_bounds(timestamp)
        if session is None:
            return None
        _, market_close = session
        if timestamp < market_close and self._is_nyse_open(timestamp):
            return max(0.0, (market_close - timestamp).total_seconds() / 60.0)
        return None

    def _is_nyse_open(self, timestamp: datetime) -> bool:
        if self._market_clock is not None:
            # Note: We ignore the timestamp arg here because the live clock
            # only knows the "current" state. In backtesting, market_clock is None.
            return self._market_clock.is_open()
            
        session = self._session_bounds(timestamp)
        if session is None:
            return False
        market_open, market_close = session
        return market_open <= timestamp < market_close

    def _session_bounds(self, timestamp: datetime) -> tuple[datetime, datetime] | None:
        timestamp = self._normalize_timestamp(timestamp)
        if self._calendar is not None:
            session_date = timestamp.astimezone(self._NEW_YORK).date()
            schedule = self._calendar.schedule(start_date=session_date, end_date=session_date)
            if schedule.empty:
                return None
            market_open = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(timezone.utc)
            market_close = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(timezone.utc)
            return market_open, market_close

        local_timestamp = timestamp.astimezone(self._NEW_YORK)
        if local_timestamp.weekday() >= 5:
            return None
        market_open_local = datetime.combine(local_timestamp.date(), self._NYSE_OPEN, tzinfo=self._NEW_YORK)
        market_close_local = datetime.combine(local_timestamp.date(), self._NYSE_CLOSE, tzinfo=self._NEW_YORK)
        return market_open_local.astimezone(timezone.utc), market_close_local.astimezone(timezone.utc)

    @staticmethod
    def _is_forex_open(timestamp: datetime) -> bool:
        local = timestamp.astimezone(SessionManager._NEW_YORK)
        if local.weekday() == 5:
            return False
        if local.weekday() == 6:
            return local.hour >= 17
        if local.weekday() == 4:
            return local.hour < 17
        return True

    @staticmethod
    def _normalize_timestamp(when: datetime | None) -> datetime:
        timestamp = when or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
