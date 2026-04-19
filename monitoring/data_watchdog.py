"""
monitoring/data_watchdog.py - Dead Man's Switch for Data Feeds

Monitors the "heartbeat" of the IBKR data feed.
If market data updates stop arriving, this module triggers a CRITICAL alert
to halt trading and prevent the system from flying blind.

Checks:
1. Global Heartbeat: Last `tickPrice` received (System-wide)
2. active_symbol_freshness: Are active positions getting updates?
3. Reconnect attempts: Count reconnect tries so callers can enforce
   ``FEED_MAX_RECONNECT_ATTEMPTS`` before surrendering (Round 7 / GAP-10B).
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import ApexConfig

logger = logging.getLogger(__name__)

@dataclass
class WatchdogStatus:
    is_alive: bool
    last_heartbeat: float
    stalled_duration: float
    critical_symbols: List[str]  # Symbols with stale data
    message: str
    reconnect_attempts: int = 0
    reconnect_budget_exhausted: bool = False

class DataWatchdog:
    """
    Monitors data feed freshness and triggers alerts on silence.

    When instantiated without explicit arguments, every threshold is sourced
    from :class:`ApexConfig` so env-overrides actually take effect.
    """

    def __init__(
        self,
        max_silence_seconds: Optional[float] = None,
        active_symbol_timeout: Optional[float] = None,
        min_updates_per_minute: int = 5,
        max_reconnect_attempts: Optional[int] = None,
    ):
        """
        Args:
            max_silence_seconds: Global silence ceiling — defaults to
                ``ApexConfig.FEED_STALE_THRESHOLD_SECONDS``.
            active_symbol_timeout: Per-symbol staleness ceiling — defaults to
                ``ApexConfig.FEED_WATCHDOG_INTERVAL_SECONDS × 2``.
            min_updates_per_minute: Minimum heartbeat rate before a WARNING
                fires (soft signal — not a hard kill).
            max_reconnect_attempts: Reconnect budget — defaults to
                ``ApexConfig.FEED_MAX_RECONNECT_ATTEMPTS``. Exceeding this
                flips the watchdog into a dead state until an operator
                resets it explicitly.
        """
        self.max_silence_seconds: float = float(
            max_silence_seconds
            if max_silence_seconds is not None
            else getattr(ApexConfig, "FEED_STALE_THRESHOLD_SECONDS", 60.0)
        )
        if self.max_silence_seconds <= 0.0:
            raise ValueError(
                f"max_silence_seconds must be > 0, got {self.max_silence_seconds}"
            )

        _wd_default = float(getattr(ApexConfig, "FEED_WATCHDOG_INTERVAL_SECONDS", 15.0))
        self.active_symbol_timeout: float = float(
            active_symbol_timeout
            if active_symbol_timeout is not None
            else max(_wd_default * 2.0, 10.0)
        )

        self.min_updates_per_minute: int = int(min_updates_per_minute)

        self.max_reconnect_attempts: int = int(
            max_reconnect_attempts
            if max_reconnect_attempts is not None
            else getattr(ApexConfig, "FEED_MAX_RECONNECT_ATTEMPTS", 10)
        )
        if self.max_reconnect_attempts < 1:
            raise ValueError(
                f"max_reconnect_attempts must be >= 1, got {self.max_reconnect_attempts}"
            )

        # State
        self.last_global_update: float = time.time()
        self.symbol_last_update: Dict[str, float] = {}
        self.update_counts: Dict[str, int] = {}  # Updates in current minute
        self.last_minute_bucket: int = int(time.time() / 60)
        self.reconnect_attempts: int = 0

        logger.info(
            "🐕 Data Watchdog initialized (silence=%.0fs, active_timeout=%.0fs, "
            "reconnect_budget=%d)",
            self.max_silence_seconds,
            self.active_symbol_timeout,
            self.max_reconnect_attempts,
        )

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

    def register_reconnect_attempt(self) -> bool:
        """
        Increment the reconnect counter and return whether the caller may still
        retry. Returns ``False`` when the budget is exhausted — the caller
        must stop trying to reconnect and escalate.

        Returns:
            ``True`` if the attempt is within budget, ``False`` once the
            configured ``FEED_MAX_RECONNECT_ATTEMPTS`` ceiling has been hit.
        """
        self.reconnect_attempts += 1
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.critical(
                "🐕 Feed reconnect budget exhausted: %d > %d — halting reconnect loop",
                self.reconnect_attempts,
                self.max_reconnect_attempts,
            )
            return False
        logger.warning(
            "🐕 Feed reconnect attempt %d/%d",
            self.reconnect_attempts,
            self.max_reconnect_attempts,
        )
        return True

    def reset_reconnect_attempts(self) -> None:
        """Clear the reconnect counter after a successful feed recovery."""
        if self.reconnect_attempts:
            logger.info(
                "🐕 Feed reconnect counter cleared (was %d)", self.reconnect_attempts
            )
        self.reconnect_attempts = 0

    def check_health(self, active_positions: List[str]) -> WatchdogStatus:
        """
        Check if data feed is healthy.

        Args:
            active_positions: Symbols that require continuous live feed.

        Returns:
            :class:`WatchdogStatus` describing current health. ``is_alive``
            is ``False`` when the caller must halt new entries. When the
            reconnect budget is exhausted the ``reconnect_budget_exhausted``
            flag is set and callers should stop issuing reconnect attempts.
        """
        now = time.time()
        budget_done = self.reconnect_attempts > self.max_reconnect_attempts

        # 1. Global Silence Check
        silence_duration = now - self.last_global_update
        if silence_duration > self.max_silence_seconds:
            # Idle portfolios can run safely without hard-kill when there are
            # no active positions requiring streaming freshness guarantees.
            if not active_positions:
                return WatchdogStatus(
                    is_alive=True,
                    last_heartbeat=self.last_global_update,
                    stalled_duration=silence_duration,
                    critical_symbols=[],
                    message=f"IDLE: No active positions (silence {silence_duration:.0f}s)",
                    reconnect_attempts=self.reconnect_attempts,
                    reconnect_budget_exhausted=budget_done,
                )
            return WatchdogStatus(
                is_alive=False,
                last_heartbeat=self.last_global_update,
                stalled_duration=silence_duration,
                critical_symbols=[],
                message=f"CRITICAL: Data feed DEAD. Silence for {silence_duration:.0f}s",
                reconnect_attempts=self.reconnect_attempts,
                reconnect_budget_exhausted=budget_done,
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
                message=f"CRITICAL: Stale data for active positions: {', '.join(stale_symbols)}",
                reconnect_attempts=self.reconnect_attempts,
                reconnect_budget_exhausted=budget_done,
            )

        if stale_symbols:
            # Warning level - updates happening generally, but some symbols lagging
            return WatchdogStatus(
                is_alive=True,  # Still alive, but warning
                last_heartbeat=self.last_global_update,
                stalled_duration=silence_duration,
                critical_symbols=stale_symbols,
                message=f"WARNING: Symbol lag detected: {', '.join(stale_symbols)}",
                reconnect_attempts=self.reconnect_attempts,
                reconnect_budget_exhausted=budget_done,
            )

        return WatchdogStatus(
            is_alive=True,
            last_heartbeat=self.last_global_update,
            stalled_duration=silence_duration,
            critical_symbols=[],
            message="OK",
            reconnect_attempts=self.reconnect_attempts,
            reconnect_budget_exhausted=budget_done,
        )
