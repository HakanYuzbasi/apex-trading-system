"""
core/alert_aggregator.py - Alert Aggregation Utility

Reduces log noise by aggregating similar alerts into periodic summaries.

Usage:
    from core.alert_aggregator import alert_agg
    
    # Instead of logging every occurrence
    alert_agg.add("price_fetch_failed", 
                  "Failed to fetch price",
                  data={"symbol": symbol})
    
    # Logs batched summary:
    # "Failed to fetch price (occurred 14x in last 60s) | Affected: AAPL, GOOGL..."
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Alert:
    """Single alert occurrence."""
    key: str
    message: str
    level: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class AlertAggregator:
    """
    Aggregate similar alerts to reduce log spam.
    
    Alerts are grouped by key and logged in batches either:
    - After window_seconds elapse
    - After max_count occurrences
    - Immediately if priority=True
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        window_seconds: float = 60.0,
        max_count: int = 100,
        auto_flush: bool = True
    ):
        self.logger = logger
        self.window_seconds = window_seconds
        self.max_count = max_count
        self.auto_flush = auto_flush
        
        self.alerts: Dict[str, List[Alert]] = defaultdict(list)
        self.first_seen: Dict[str, float] = {}
        self.lock = threading.RLock()
        self._flusher_thread = None
        
        # Start background flusher if auto-flush enabled
        if self.auto_flush:
            self._start_flusher()
    
    def add(
        self,
        key: str,
        message: str,
        level: str = "warning",
        data: Optional[Dict] = None,
        priority: bool = False
    ):
        """
        Add an alert to the aggregator.
        
        Args:
            key: Unique key for this alert type (e.g., "price_fetch_failed")
            message: Human-readable message
            level: Log level (debug, info, warning, error, critical)
            data: Optional metadata to attach
            priority: If True, log immediately without aggregation
        """
        if priority:
            # High-priority alerts bypass aggregation
            self._log_alert(message, level)
            return
        
        now = time.time()
        
        with self.lock:
            # Record alert
            self.alerts[key].append(Alert(
                key=key,
                message=message,
                level=level,
                timestamp=now,
                data=data or {}
            ))
            
            if key not in self.first_seen:
                self.first_seen[key] = now
            
            # Check if we should flush this key
            count = len(self.alerts[key])
            age = now - self.first_seen[key]
            
            if count >= self.max_count or age >= self.window_seconds:
                self._flush_key(key)
    
    def _flush_key(self, key: str):
        """Flush aggregated alerts for a specific key."""
        with self.lock:
            alerts = self.alerts.pop(key, [])
            self.first_seen.pop(key, None)
        
        if not alerts:
            return
        
        count = len(alerts)
        first = alerts[0]
        
        # Build summary message
        if count == 1:
            summary = first.message
        else:
            summary = f"{first.message} (occurred {count}x in last {self.window_seconds:.0f}s)"
            
            # Add unique values if data present
            if first.data:
                unique_vals = self._get_unique_values(alerts)
                if unique_vals:
                    summary += f" | Affected: {unique_vals}"
        
        # Log at appropriate level
        self._log_alert(summary, first.level)
    
    def _get_unique_values(self, alerts: List[Alert]) -> str:
        """Extract unique values from alert data."""
        # Collect unique values for common keys
        common_keys = ["symbol", "file", "service", "endpoint", "module"]
        values = defaultdict(set)
        
        for alert in alerts:
            for key in common_keys:
                if key in alert.data:
                    values[key].add(str(alert.data[key]))
        
        # Format output
        parts = []
        for key, val_set in values.items():
            if len(val_set) <= 5:
                parts.append(f"{', '.join(sorted(val_set))}")
            else:
                sample = sorted(val_set)[:3]
                parts.append(f"{', '.join(sample)}... (+{len(val_set)-3} more)")
        
        return "; ".join(parts)
    
    def flush_all(self):
        """Flush all pending alerts immediately."""
        with self.lock:
            keys = list(self.alerts.keys())
        
        for key in keys:
            self._flush_key(key)
    
    def _start_flusher(self):
        """Start background thread to auto-flush old alerts."""
        def flusher():
            while True:
                time.sleep(self.window_seconds / 2)
                
                # Check for old alerts
                now = time.time()
                with self.lock:
                    keys_to_flush = [
                        key for key, first_time in self.first_seen.items()
                        if now - first_time >= self.window_seconds
                    ]
                
                for key in keys_to_flush:
                    self._flush_key(key)
        
        self._flusher_thread = threading.Thread(target=flusher, daemon=True, name="AlertAggregatorFlusher")
        self._flusher_thread.start()
    
    def _log_alert(self, message: str, level: str):
        """Log a single alert immediately."""
        log_func = getattr(self.logger, level, self.logger.warning)
        log_func(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        with self.lock:
            return {
                "pending_keys": len(self.alerts),
                "pending_alerts": sum(len(alerts) for alerts in self.alerts.values()),
                "window_seconds": self.window_seconds,
                "max_count": self.max_count
            }


# Global instance for convenience (optional)
_global_aggregator = None

def get_alert_aggregator(logger: Optional[logging.Logger] = None) -> AlertAggregator:
    """Get or create global alert aggregator."""
    global _global_aggregator
    if _global_aggregator is None:
        if logger is None:
            logger = logging.getLogger("apex")
        _global_aggregator = AlertAggregator(logger, window_seconds=60.0)
    return _global_aggregator
