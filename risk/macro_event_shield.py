"""
risk/macro_event_shield.py - Macro Economic Event Protection

Automatically blocks trading around high-impact economic events:
- FOMC announcements (2pm ET releases)
- CPI/PPI releases (8:30am ET)
- Non-Farm Payrolls (first Friday, 8:30am ET)
- Earnings for held positions

Blackout periods:
- FOMC: 60 min before, 30 min after
- CPI/NFP: 30 min before, 15 min after
- Earnings: 24h before, 2h after (for that symbol only)

Also implements "Game Over" protection:
- If daily loss > 5%: IMMEDIATE shutdown, no trading until manual reset
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from collections import defaultdict
import calendar

logger = logging.getLogger(__name__)


class EventType(Enum):
    FOMC = "fomc"
    CPI = "cpi"
    PPI = "ppi"
    NFP = "nfp"  # Non-Farm Payrolls
    JOBLESS_CLAIMS = "jobless_claims"
    GDP = "gdp"
    EARNINGS = "earnings"
    FED_SPEECH = "fed_speech"


class EventImpact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4  # FOMC, CPI, NFP


@dataclass
class MacroEvent:
    """Scheduled macro event."""
    event_type: EventType
    impact: EventImpact
    event_time: datetime
    symbol: Optional[str] = None  # For earnings
    description: str = ""
    blackout_before_minutes: int = 60
    blackout_after_minutes: int = 30


@dataclass
class MacroState:
    """Current macro risk assessment."""
    in_blackout: bool
    active_events: List[MacroEvent]
    next_event: Optional[MacroEvent]
    minutes_to_next: Optional[int]
    game_over: bool
    game_over_reason: Optional[str]
    trading_allowed: bool
    position_sizing_multiplier: float
    symbols_in_earnings_blackout: Set[str]


class MacroEventShield:
    """
    Protection against macro economic event risk.

    Maintains calendar of known high-impact events and enforces
    blackout periods where no new positions are opened.
    """

    def __init__(
        self,
        blackout_before_fomc: int = 60,
        blackout_after_fomc: int = 30,
        blackout_before_data: int = 30,
        blackout_after_data: int = 15,
        blackout_before_earnings: int = 1440,  # 24 hours
        blackout_after_earnings: int = 120,     # 2 hours
        game_over_threshold: float = 0.05,      # 5% daily loss
    ):
        self.blackout_before_fomc = blackout_before_fomc
        self.blackout_after_fomc = blackout_after_fomc
        self.blackout_before_data = blackout_before_data
        self.blackout_after_data = blackout_after_data
        self.blackout_before_earnings = blackout_before_earnings
        self.blackout_after_earnings = blackout_after_earnings
        self.game_over_threshold = game_over_threshold

        # State
        self._events: List[MacroEvent] = []
        self._earnings_calendar: Dict[str, datetime] = {}  # symbol -> earnings date
        self._game_over = False
        self._game_over_reason: Optional[str] = None
        self._game_over_time: Optional[datetime] = None

        # Performance tracking
        self._daily_start_capital: Optional[float] = None
        self._daily_low_capital: Optional[float] = None

        # Initialize with known recurring events
        self._initialize_known_events()

        logger.info(
            f"MacroEventShield initialized: "
            f"FOMC blackout={blackout_before_fomc}/{blackout_after_fomc}min, "
            f"game_over={game_over_threshold:.1%}"
        )

    def _initialize_known_events(self):
        """Initialize calendar with known recurring events."""
        now = datetime.now()

        # FOMC meetings (roughly every 6 weeks, 2pm ET)
        # 2024 dates - in production, fetch from API
        fomc_dates = self._get_fomc_dates(now.year)
        for dt in fomc_dates:
            if dt > now:
                self._events.append(MacroEvent(
                    event_type=EventType.FOMC,
                    impact=EventImpact.CRITICAL,
                    event_time=dt,
                    description="FOMC Rate Decision",
                    blackout_before_minutes=self.blackout_before_fomc,
                    blackout_after_minutes=self.blackout_after_fomc,
                ))

        # CPI (usually 2nd week, 8:30am ET)
        cpi_dates = self._get_cpi_dates(now.year)
        for dt in cpi_dates:
            if dt > now:
                self._events.append(MacroEvent(
                    event_type=EventType.CPI,
                    impact=EventImpact.CRITICAL,
                    event_time=dt,
                    description="Consumer Price Index",
                    blackout_before_minutes=self.blackout_before_data,
                    blackout_after_minutes=self.blackout_after_data,
                ))

        # NFP (first Friday, 8:30am ET)
        nfp_dates = self._get_nfp_dates(now.year)
        for dt in nfp_dates:
            if dt > now:
                self._events.append(MacroEvent(
                    event_type=EventType.NFP,
                    impact=EventImpact.CRITICAL,
                    event_time=dt,
                    description="Non-Farm Payrolls",
                    blackout_before_minutes=self.blackout_before_data,
                    blackout_after_minutes=self.blackout_after_data,
                ))

        # Sort by time
        self._events.sort(key=lambda e: e.event_time)

        logger.info(f"Loaded {len(self._events)} macro events for {now.year}")

    def _get_fomc_dates(self, year: int) -> List[datetime]:
        """Get FOMC meeting dates for a year (2pm ET releases)."""
        # Simplified: actual dates should come from Fed calendar
        # These are approximate - 8 meetings per year
        months = [1, 3, 5, 6, 7, 9, 11, 12]
        dates = []
        for month in months:
            # Usually mid-month Wednesday
            dt = datetime(year, month, 15, 14, 0)  # 2pm
            # Adjust to Wednesday
            while dt.weekday() != 2:
                dt += timedelta(days=1)
            dates.append(dt)
        return dates

    def _get_cpi_dates(self, year: int) -> List[datetime]:
        """Get CPI release dates (8:30am ET, usually 2nd week)."""
        dates = []
        for month in range(1, 13):
            # Usually around the 10th-14th
            dt = datetime(year, month, 12, 8, 30)
            # Adjust to weekday
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
            dates.append(dt)
        return dates

    def _get_nfp_dates(self, year: int) -> List[datetime]:
        """Get NFP dates (first Friday of each month, 8:30am ET)."""
        dates = []
        for month in range(1, 13):
            # Find first Friday
            dt = datetime(year, month, 1, 8, 30)
            while dt.weekday() != 4:  # Friday
                dt += timedelta(days=1)
            dates.append(dt)
        return dates

    def add_earnings_date(self, symbol: str, earnings_time: datetime):
        """Add earnings date for a symbol."""
        self._earnings_calendar[symbol] = earnings_time
        self._events.append(MacroEvent(
            event_type=EventType.EARNINGS,
            impact=EventImpact.HIGH,
            event_time=earnings_time,
            symbol=symbol,
            description=f"{symbol} Earnings",
            blackout_before_minutes=self.blackout_before_earnings,
            blackout_after_minutes=self.blackout_after_earnings,
        ))
        self._events.sort(key=lambda e: e.event_time)

    def add_custom_event(
        self,
        event_type: EventType,
        event_time: datetime,
        impact: EventImpact = EventImpact.HIGH,
        description: str = "",
        blackout_before: int = 30,
        blackout_after: int = 15,
    ):
        """Add a custom macro event."""
        self._events.append(MacroEvent(
            event_type=event_type,
            impact=impact,
            event_time=event_time,
            description=description,
            blackout_before_minutes=blackout_before,
            blackout_after_minutes=blackout_after,
        ))
        self._events.sort(key=lambda e: e.event_time)

    def update_daily_capital(self, current_capital: float, is_day_start: bool = False):
        """
        Update capital tracking for game-over detection.

        Args:
            current_capital: Current portfolio value
            is_day_start: True if this is the start of a new trading day
        """
        if is_day_start or self._daily_start_capital is None:
            self._daily_start_capital = current_capital
            self._daily_low_capital = current_capital
            # Reset game over at day start (manual review required)
            if is_day_start and self._game_over:
                logger.warning("New trading day - game over state preserved until manual reset")

        if current_capital < self._daily_low_capital:
            self._daily_low_capital = current_capital

        # Check for game over
        if self._daily_start_capital and self._daily_start_capital > 0:
            daily_loss = (self._daily_start_capital - current_capital) / self._daily_start_capital
            if daily_loss >= self.game_over_threshold:
                self._trigger_game_over(
                    f"Daily loss {daily_loss:.1%} exceeded {self.game_over_threshold:.1%} threshold"
                )

    def _trigger_game_over(self, reason: str):
        """Trigger game over state - immediate shutdown."""
        if not self._game_over:
            self._game_over = True
            self._game_over_reason = reason
            self._game_over_time = datetime.now()
            logger.critical(f"🚨 GAME OVER TRIGGERED: {reason}")
            logger.critical("All trading halted until manual reset!")

    def reset_game_over(self, manual_confirmation: bool = False):
        """
        Reset game over state (requires manual confirmation).

        Args:
            manual_confirmation: Must be True to reset
        """
        if not manual_confirmation:
            logger.warning("Game over reset requires manual_confirmation=True")
            return False

        self._game_over = False
        self._game_over_reason = None
        self._game_over_time = None
        self._daily_start_capital = None
        logger.info("Game over state reset - trading enabled")
        return True

    def assess_state(
        self,
        symbol: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> MacroState:
        """
        Assess current macro state for trading decisions.

        Args:
            symbol: Optional symbol to check for earnings blackout
            current_time: Current time (default: now)

        Returns:
            MacroState with blackout status and recommendations
        """
        now = current_time or datetime.now()

        # Check game over first
        if self._game_over:
            return MacroState(
                in_blackout=True,
                active_events=[],
                next_event=None,
                minutes_to_next=None,
                game_over=True,
                game_over_reason=self._game_over_reason,
                trading_allowed=False,
                position_sizing_multiplier=0.0,
                symbols_in_earnings_blackout=set(),
            )

        active_events = []
        symbols_in_blackout = set()

        # Check all events
        for event in self._events:
            blackout_start = event.event_time - timedelta(minutes=event.blackout_before_minutes)
            blackout_end = event.event_time + timedelta(minutes=event.blackout_after_minutes)

            if blackout_start <= now <= blackout_end:
                active_events.append(event)
                if event.event_type == EventType.EARNINGS and event.symbol:
                    symbols_in_blackout.add(event.symbol)

        # Find next upcoming event
        next_event = None
        minutes_to_next = None
        for event in self._events:
            if event.event_time > now:
                next_event = event
                minutes_to_next = int((event.event_time - now).total_seconds() / 60)
                break

        # Determine if in blackout
        # Global blackout for FOMC/CPI/NFP
        # Symbol-specific blackout for earnings
        global_blackout = any(
            e.event_type in [EventType.FOMC, EventType.CPI, EventType.NFP, EventType.GDP]
            for e in active_events
        )

        symbol_blackout = symbol and symbol in symbols_in_blackout
        in_blackout = global_blackout or symbol_blackout

        # Position sizing based on proximity to events
        size_mult = 1.0
        if next_event and minutes_to_next:
            if next_event.impact == EventImpact.CRITICAL:
                if minutes_to_next < 120:  # 2 hours
                    size_mult = 0.5
                elif minutes_to_next < 240:  # 4 hours
                    size_mult = 0.75

        if in_blackout:
            size_mult = 0.0

        return MacroState(
            in_blackout=in_blackout,
            active_events=active_events,
            next_event=next_event,
            minutes_to_next=minutes_to_next,
            game_over=False,
            game_over_reason=None,
            trading_allowed=not in_blackout,
            position_sizing_multiplier=size_mult,
            symbols_in_earnings_blackout=symbols_in_blackout,
        )

    def should_block_entry(self, symbol: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if new entries should be blocked.

        Returns:
            (blocked, reason) tuple
        """
        state = self.assess_state(symbol)

        if state.game_over:
            return True, f"GAME OVER: {state.game_over_reason}"

        if state.in_blackout:
            if state.active_events:
                events_str = ", ".join(e.description for e in state.active_events)
                return True, f"Macro blackout: {events_str}"
            return True, "In blackout period"

        return False, ""

    def should_reduce_exposure(self) -> Tuple[bool, float, str]:
        """
        Check if exposure should be reduced due to upcoming events.

        Returns:
            (should_reduce, target_multiplier, reason)
        """
        state = self.assess_state()

        if state.game_over:
            return True, 0.0, f"GAME OVER: {state.game_over_reason}"

        if state.next_event and state.minutes_to_next:
            if state.next_event.impact == EventImpact.CRITICAL:
                if state.minutes_to_next < 60:
                    return True, 0.25, f"{state.next_event.description} in {state.minutes_to_next} min"
                elif state.minutes_to_next < 120:
                    return True, 0.5, f"{state.next_event.description} in {state.minutes_to_next} min"

        return False, 1.0, ""

    def get_symbols_to_avoid(self) -> Set[str]:
        """Get symbols that should be avoided due to earnings."""
        state = self.assess_state()
        return state.symbols_in_earnings_blackout

    def get_upcoming_events(self, hours: int = 24) -> List[MacroEvent]:
        """Get events in the next N hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        return [e for e in self._events if now < e.event_time <= cutoff]

    def is_game_over(self) -> bool:
        """Check if in game over state."""
        return self._game_over

    def get_diagnostics(self) -> Dict:
        """Return shield state for monitoring."""
        state = self.assess_state()
        return {
            "in_blackout": state.in_blackout,
            "game_over": state.game_over,
            "game_over_reason": state.game_over_reason,
            "trading_allowed": state.trading_allowed,
            "position_sizing_mult": state.position_sizing_multiplier,
            "next_event": state.next_event.description if state.next_event else None,
            "minutes_to_next": state.minutes_to_next,
            "active_events": [e.description for e in state.active_events],
            "earnings_blackout_symbols": list(state.symbols_in_earnings_blackout),
            "total_events_loaded": len(self._events),
        }
