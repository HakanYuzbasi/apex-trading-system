"""
risk/macro_shield.py - Economic Event Risk Protection

Blocks new entries during high-impact economic events (FOMC, CPI, NFP).
Loads event schedules from a local JSON calendar or can be configured manually.

Key Features:
- Pre-event blackout window (e.g., 60 mins before)
- Post-event blackout window (e.g., 30 mins after)
- "Red Folder" event filtering
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, NamedTuple
from pathlib import Path

from config import ApexConfig

logger = logging.getLogger(__name__)

class EconomicEvent(NamedTuple):
    id: str
    title: str
    timestamp: datetime
    impact_level: str  # "HIGH", "MEDIUM", "LOW"

class MacroShield:
    """
    Manages trading blackouts around high-impact economic events.
    """
    
    def __init__(
        self,
        blackout_minutes_before: int = 60,
        blackout_minutes_after: int = 30,
        calendar_file: str = None
    ):
        self.minutes_before = blackout_minutes_before
        self.minutes_after = blackout_minutes_after
        self.calendar_file = Path(calendar_file) if calendar_file else (ApexConfig.DATA_DIR / "economic_calendar.json")
        self.events: List[EconomicEvent] = []
        
        # Load initial calendar
        self.reload_calendar()
        
    def reload_calendar(self):
        """Load or reload the economic calendar from disk."""
        if not self.calendar_file.exists():
            logger.warning(f"⚠️ Economic calendar file not found: {self.calendar_file}")
            # Create a sample file if it doesn't exist
            self._create_sample_calendar()
            return

        try:
            with open(self.calendar_file, 'r') as f:
                data = json.load(f)
            
            self.events = []
            for item in data:
                try:
                    # Parse timestamp (expecting ISO format or similar)
                    # "2026-02-18T14:00:00"
                    ts = datetime.fromisoformat(item["timestamp"])
                    
                    if item.get("impact") == "HIGH":
                        self.events.append(EconomicEvent(
                            id=item.get("id", "unknown"),
                            title=item.get("title", "Unknown Event"),
                            timestamp=ts,
                            impact_level="HIGH"
                        ))
                except Exception as e:
                    logger.warning(f"Failed to parse event {item}: {e}")
            
            # Sort by time
            self.events.sort(key=lambda x: x.timestamp)
            
            # Filter past events (keep recent past for post-event blackout)
            now = datetime.now()
            cutoff = now - timedelta(hours=2) # Keep recent events
            self.events = [e for e in self.events if e.timestamp > cutoff]
            
            logger.info(f"🛡️ Macro Shield loaded {len(self.events)} high-impact events")
            
        except Exception as e:
            logger.error(f"❌ Failed to load economic calendar: {e}")

    def _create_sample_calendar(self):
        """Create a sample calendar file if none exists."""
        try:
            self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy FOMC event for next week for demonstration
            # Note: User provided year is 2026
            next_fomc = datetime.now() + timedelta(days=7)
            next_fomc = next_fomc.replace(hour=14, minute=0, second=0, microsecond=0)
            
            sample_data = [
                {
                    "id": "fomc_rate_decision",
                    "title": "FOMC Interest Rate Decision",
                    "timestamp": next_fomc.isoformat(),
                    "impact": "HIGH",
                    "currency": "USD"
                }
            ]
            
            with open(self.calendar_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created sample economic calendar at {self.calendar_file}")
            self.reload_calendar()
            
        except Exception as e:
            logger.error(f"Could not create sample calendar: {e}")

    def is_blackout_active(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if we are currently in a blackout window.
        """
        if not self.events:
            return False
            
        now = current_time or datetime.now()
        
        # Optimization: Events are sorted, verify effective window
        for event in self.events:
            # Blackout start and end
            start_window = event.timestamp - timedelta(minutes=self.minutes_before)
            end_window = event.timestamp + timedelta(minutes=self.minutes_after)
            
            if start_window <= now <= end_window:
                time_to_release = (event.timestamp - now).total_seconds() / 60
                if time_to_release > 0:
                     logger.info(f"🛡️ MACRO BLACKOUT: {event.title} in {time_to_release:.0f} mins")
                else:
                     logger.info(f"🛡️ MACRO BLACKOUT: Post-event volatility ({event.title})")
                return True
                
        return False

    def get_active_event(self, current_time: Optional[datetime] = None) -> Optional[EconomicEvent]:
        """Get the specific event causing the blackout, if any."""
        if not self.events:
            return None
            
        now = current_time or datetime.now()
        for event in self.events:
            start_window = event.timestamp - timedelta(minutes=self.minutes_before)
            end_window = event.timestamp + timedelta(minutes=self.minutes_after)
            if start_window <= now <= end_window:
                return event
        return None
