"""
monitoring/data_watchdog.py - Dead Man's Switch for Data Feeds

Monitors the "heartbeat" of the IBKR data feed.
If market data updates stop arriving, this module triggers a CRITICAL alert
to halt trading and prevent the system from flying blind.

Checks:
1. Global Heartbeat: Last `tickPrice` received (System-wide)
2. active_symbol_freshness: Are active positions getting updates?
"""

import time
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class WatchdogStatus:
    is_alive: bool
    last_heartbeat: float
    stalled_duration: float
    critical_symbols: List[str]  # Symbols with stale data
    message: str

class DataWatchdog:
    """
    Monitors data feed freshness and triggers alerts on silence.
    """
    
    def __init__(
        self,
        max_silence_seconds: int = 300,  # 5 minutes global silence = CRITICAL
        active_symbol_timeout: int = 15, # 15s stale data for positions = WARNING/CRITICAL check
        min_updates_per_minute: int = 5
    ):
        self.max_silence_seconds = max_silence_seconds
        self.active_symbol_timeout = active_symbol_timeout
        self.min_updates_per_minute = min_updates_per_minute
        
        # State
        self.last_global_update = time.time()
        self.symbol_last_update: Dict[str, float] = {}
        self.update_counts: Dict[str, int] = {}  # Updates in current minute
        self.last_minute_bucket = int(time.time() / 60)
        
        logger.info(f"🐕 Data Watchdog initialized (Silence limit: {max_silence_seconds}s)")

    def feed_heartbeat(self, symbol: str = "SYSTEM"):
        """Call this whenever a price update is received."""
        now = time.time()
        self.last_global_update = now
        
        if symbol != "SYSTEM":
            self.symbol_last_update[symbol] = now
            
            # Track rate
            bucket = int(now / 60)
            if bucket != self.last_minute_bucket:
                self.update_counts.clear()
                self.last_minute_bucket = bucket
            
            self.update_counts[symbol] = self.update_counts.get(symbol, 0) + 1

    def check_health(self, active_positions: List[str]) -> WatchdogStatus:
        """
        Check if data feed is healthy.
        Returns False if:
        1. No updates for entire system > max_silence_seconds
        2. Active positions have stale data > active_symbol_timeout
        """
        now = time.time()
        
        # 1. Global Silence Check
        silence_duration = now - self.last_global_update
        if silence_duration > self.max_silence_seconds:
             return WatchdogStatus(
                 is_alive=False,
                 last_heartbeat=self.last_global_update,
                 stalled_duration=silence_duration,
                 critical_symbols=[],
                 message=f"CRITICAL: Data feed DEAD. Silence for {silence_duration:.0f}s"
             )

        # 2. Active Position Freshness Check
        stale_symbols = []
        for sym in active_positions:
            last_upd = self.symbol_last_update.get(sym, 0)
            age = now - last_upd
            
            # Only check freshness if we've seen at least one update ever
            # (Avoid flagging symbols we just subscribed to)
            if last_upd > 0 and age > self.active_symbol_timeout:
                stale_symbols.append(f"{sym}({age:.0f}s)")

        if stale_symbols and silence_duration > 60:
             # Escalating: Global updates slow AND active symbols stale
             return WatchdogStatus(
                 is_alive=False,
                 last_heartbeat=self.last_global_update,
                 stalled_duration=silence_duration,
                 critical_symbols=stale_symbols,
                 message=f"CRITICAL: Stale data for active positions: {', '.join(stale_symbols)}"
             )
        
        elif stale_symbols:
            # Warning level - updates happening generally, but some symbols lagging
            return WatchdogStatus(
                 is_alive=True, # Still alive, but warning
                 last_heartbeat=self.last_global_update,
                 stalled_duration=silence_duration,
                 critical_symbols=stale_symbols,
                 message=f"WARNING: Symbol lag detected: {', '.join(stale_symbols)}"
             )

        return WatchdogStatus(
            is_alive=True,
            last_heartbeat=self.last_global_update,
            stalled_duration=silence_duration,
            critical_symbols=[],
            message="OK"
        )
