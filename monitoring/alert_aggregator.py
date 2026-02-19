"""
Alert Aggregation System

Centralized alert management with deduplication and throttling.
"""

import logging
import time
from typing import Dict, Optional, Set
from datetime import datetime
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertAggregator:
    """Aggregate and deduplicate alerts to prevent alert fatigue."""
    
    def __init__(
        self,
        cooldown_seconds: int = 300,  # 5 minutes
        log_file: Optional[Path] = None
    ):
        self.cooldown_seconds = cooldown_seconds
        self.log_file = log_file
        
        # Alert deduplication cache: {alert_key: last_sent_timestamp}
        self.alert_cache: Dict[str, float] = {}
        
        # Track alert counts
        self.alert_counts: Dict[str, int] = {}
        
    def _get_alert_key(self, alert_type: str, message: str) -> str:
        """Generate unique key for alert."""
        # Use hash of message to handle similar messages
        message_hash = hash(message) % 10000
        return f"{alert_type}:{message_hash}"
    
    def should_send_alert(self, alert_type: str, message: str) -> bool:
        """Check if alert should be sent based on cooldown."""
        alert_key = self._get_alert_key(alert_type, message)
        last_sent = self.alert_cache.get(alert_key, 0)
        current_time = time.time()
        
        if current_time - last_sent > self.cooldown_seconds:
            self.alert_cache[alert_key] = current_time
            self.alert_counts[alert_key] = self.alert_counts.get(alert_key, 0) + 1
            return True
        
        return False
    
    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict] = None
    ):
        """Send alert with deduplication."""
        if not self.should_send_alert(alert_type, message):
            logger.debug(f"Alert throttled: {alert_type}")
            return
        
        alert_key = self._get_alert_key(alert_type, message)
        count = self.alert_counts.get(alert_key, 1)
        
        # Add count to message if this is a repeat
        if count > 1:
            message = f"{message} (occurred {count} times)"
        
        # Log based on severity
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"🚨 {alert_type}: {message}")
        elif severity == AlertSeverity.ERROR:
            logger.error(f"❌ {alert_type}: {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"⚠️ {alert_type}: {message}")
        else:
            logger.info(f"ℹ️ {alert_type}: {message}")
        
        # Write to alert log file
        if self.log_file:
            self._write_to_log(alert_type, message, severity, metadata)
        
        # TODO: Send to external systems (Slack, PagerDuty, Email)
        # self._send_to_slack(alert_type, message, severity)
        # self._send_to_pagerduty(alert_type, message, severity)
    
    def _write_to_log(
        self,
        alert_type: str,
        message: str,
        severity: AlertSeverity,
        metadata: Optional[Dict]
    ):
        """Write alert to JSONL log file."""
        try:
            alert_record = {
                "timestamp": datetime.now().isoformat(),
                "type": alert_type,
                "message": message,
                "severity": severity.value,
                "metadata": metadata or {}
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to log: {e}")
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        return {
            "total_unique_alerts": len(self.alert_counts),
            "total_alerts_sent": sum(self.alert_counts.values()),
            "alerts_by_type": dict(self.alert_counts),
            "cooldown_seconds": self.cooldown_seconds
        }
    
    def reset_cooldowns(self):
        """Reset all cooldowns (for testing or manual override)."""
        self.alert_cache.clear()
        logger.info("Alert cooldowns reset")


# Global instance
_aggregator: Optional[AlertAggregator] = None


def get_alert_aggregator(
    cooldown_seconds: int = 300,
    log_file: Optional[Path] = None
) -> AlertAggregator:
    """Get or create global alert aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = AlertAggregator(cooldown_seconds, log_file)
    return _aggregator


def send_alert(
    alert_type: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    metadata: Optional[Dict] = None
):
    """Convenience function to send alert via global aggregator."""
    aggregator = get_alert_aggregator()
    aggregator.send_alert(alert_type, message, severity, metadata)
