from __future__ import annotations

import time
import threading
import logging
import asyncio
from datetime import datetime
from typing import Any, Optional

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
        
        self._last_heartbeat = time.time()
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._is_hanging = False

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
                    # Notifier is likely async, but we are in a thread.
                    # We use asyncio.run_coroutine_threadsafe if we have a loop,
                    # or just a synchronous fallback if possible.
                    # For this harness, we'll try to reach the notifier.
                    try:
                        asyncio.run(self._notifier.notify_text(msg))
                    except Exception as e:
                        logger.error(f"Failed to send hang alert: {e}")
            
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
