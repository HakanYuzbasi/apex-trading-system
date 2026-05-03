from __future__ import annotations

import time
import threading
import logging
import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Optional, Callable

logger = logging.getLogger("health_monitor")

class HealthMonitor:
    """
    Thread-based watchdog that monitors the main event loop's responsiveness.
    If the loop hangs (no heartbeats for > threshold), it triggers an alert.
    """

    def __init__(
        self,
        notifier: Optional[Any] = None,
        hang_threshold_seconds: float = 5.0,
        check_interval_seconds: float = 1.0,
    ) -> None:
        self._notifier = notifier
        self._hang_threshold = hang_threshold_seconds
        self._check_interval = check_interval_seconds
        
        # Hard exit threshold (default 120s)
        try:
            from config import ApexConfig
            self._hard_exit_threshold = getattr(ApexConfig, "WATCHDOG_HARD_EXIT_TIMEOUT", 120.0)
        except Exception:
            self._hard_exit_threshold = 120.0
            
        self._force_sync_callback: Optional[Callable] = None
        
        self._last_heartbeat = time.time()
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._is_hanging = False
        self._hard_exit_triggered = False

    def set_force_sync_callback(self, callback: Callable) -> None:
        """Set a callback to be executed before hard-exit (e.g. shadow_accounting.force_sync)."""
        self._force_sync_callback = callback

    def heartbeat(self) -> None:
        """Update the last heartbeat timestamp (call from main loop)."""
        self._last_heartbeat = time.time()
        if self._is_hanging:
            logger.info("✅ SYSTEM RECOVERED: Event loop is responding again.")
            self._is_hanging = False

    def start(self) -> None:
        """Start the watchdog thread."""
        if self._watchdog_thread is not None:
            return
            
        self._stop_event.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="HealthWatchdog",
            daemon=True
        )
        self._watchdog_thread.start()
        logger.info("🛡️ HealthMonitor: Watchdog thread started")

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Background thread loop checking for hangs."""
        while not self._stop_event.is_set():
            now = time.time()
            gap = now - self._last_heartbeat
            
            if gap > self._hang_threshold and not self._is_hanging:
                self._is_hanging = True
                msg = f"🚨 HA ALERT: SYSTEM HANG DETECTED! No heartbeats for {gap:.1f}s"
                logger.error(msg)
                
                if self._notifier:
                    try:
                        # Attempt async notification in a one-off loop if needed
                        asyncio.run(self._notifier.notify_text(msg))
                    except Exception as e:
                        logger.error(f"Failed to send hang alert: {e}")

            # Hard Exit Check
            if gap > self._hard_exit_threshold and not self._hard_exit_triggered:
                self._hard_exit_triggered = True
                logger.critical(
                    f"💀 CRITICAL: SYSTEM HANG PERSISTED FOR {gap:.1f}s. TRIGGERING HARD EXIT!"
                )
                
                # 1. Attempt Force Sync
                if self._force_sync_callback:
                    try:
                        logger.info("Attempting emergency state sync before exit...")
                        self._force_sync_callback()
                    except Exception as e:
                        logger.error(f"Emergency sync failed: {e}")
                
                # 2. Final log flush
                logging.shutdown()
                
                # 3. Hard Exit (force container restart)
                os._exit(1)
            
            time.sleep(self._check_interval)

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    hm = HealthMonitor()
    hm.start()
    time.sleep(2)
    hm.heartbeat()
    time.sleep(6) # Should trigger
    hm.heartbeat()
    time.sleep(2)
    hm.stop()
