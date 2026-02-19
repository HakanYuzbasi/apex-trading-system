"""
Risk State Hot Reload

Watches risk_state.json for changes and reloads without requiring system restart.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class RiskStateWatcher:
    """Watch risk state file for changes and trigger reload."""
    
    def __init__(
        self,
        risk_state_file: Path,
        reload_callback: Callable,
        check_interval_seconds: int = 10
    ):
        self.risk_state_file = risk_state_file
        self.reload_callback = reload_callback
        self.check_interval = check_interval_seconds
        
        self.last_mtime: Optional[float] = None
        self.running = False
        self.watch_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start watching the file."""
        if self.running:
            logger.warning("Risk state watcher already running")
            return
        
        self.running = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        logger.info(f"Started risk state watcher for {self.risk_state_file}")
    
    def stop(self):
        """Stop watching the file."""
        self.running = False
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        logger.info("Stopped risk state watcher")
    
    def _watch_loop(self):
        """Main watch loop."""
        while self.running:
            try:
                if not self.risk_state_file.exists():
                    time.sleep(self.check_interval)
                    continue
                
                current_mtime = self.risk_state_file.stat().st_mtime
                
                if self.last_mtime is None:
                    # First check, just record mtime
                    self.last_mtime = current_mtime
                elif current_mtime > self.last_mtime:
                    # File changed, trigger reload
                    logger.warning(
                        f"🔄 Risk state file changed, reloading... "
                        f"(mtime: {self.last_mtime} → {current_mtime})"
                    )
                    
                    try:
                        self.reload_callback()
                        logger.info("✅ Risk state reloaded successfully")
                    except Exception as e:
                        logger.error(f"❌ Error reloading risk state: {e}")
                    
                    self.last_mtime = current_mtime
                
            except Exception as e:
                logger.error(f"Error in risk state watch loop: {e}")
            
            time.sleep(self.check_interval)
    
    def force_reload(self):
        """Force a reload regardless of file modification time."""
        logger.info("Forcing risk state reload...")
        try:
            self.reload_callback()
            if self.risk_state_file.exists():
                self.last_mtime = self.risk_state_file.stat().st_mtime
            logger.info("✅ Risk state force-reloaded successfully")
        except Exception as e:
            logger.error(f"❌ Error force-reloading risk state: {e}")
