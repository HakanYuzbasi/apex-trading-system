"""Alert aggregation and escalation for external notification channels."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity used by alert routing policy."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertAggregator:
    """Aggregate and deduplicate alerts with channel escalation policies."""

    def __init__(
        self,
        cooldown_seconds: int = 300,
        log_file: Optional[Path] = None,
        slack_webhook_url: Optional[str] = None,
        pagerduty_routing_key: Optional[str] = None,
        timeout_seconds: int = 10,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.log_file = log_file
        self.slack_webhook_url = (slack_webhook_url or os.getenv("APEX_SLACK_WEBHOOK_URL", "")).strip()
        self.pagerduty_routing_key = (
            pagerduty_routing_key or os.getenv("APEX_PAGERDUTY_ROUTING_KEY", "")
        ).strip()
        self.timeout_seconds = timeout_seconds

        self.alert_cache: Dict[str, float] = {}
        self.alert_counts: Dict[str, int] = {}

    def _get_alert_key(self, alert_type: str, message: str) -> str:
        """Generate deterministic dedupe key."""
        message_hash = hash(message) % 10000
        return f"{alert_type}:{message_hash}"

    def should_send_alert(self, alert_type: str, message: str) -> bool:
        """Return True when alert cooldown has elapsed."""
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send alert through configured channels based on severity policy."""
        if not self.should_send_alert(alert_type, message):
            logger.debug("Alert throttled: %s", alert_type)
            return

        alert_key = self._get_alert_key(alert_type, message)
        count = self.alert_counts.get(alert_key, 1)
        if count > 1:
            message = f"{message} (occurred {count} times)"

        event = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        if self.log_file:
            self._write_to_log(event)

        if severity == AlertSeverity.INFO:
            self._send_to_slack(event)
            return

        if severity == AlertSeverity.WARNING:
            logger.warning("ALERT_WARNING %s", json.dumps(event, sort_keys=True))
            self._send_to_slack(event)
            return

        logger.critical("ALERT_CRITICAL %s", json.dumps(event, sort_keys=True))
        self._send_to_slack(event)
        self._send_to_pagerduty(event)

    def _write_to_log(self, event: Dict[str, Any]) -> None:
        """Append alert record to JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as file_handle:
                file_handle.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception:
            logger.exception("Failed to write alert to log file")

    def _send_to_slack(self, event: Dict[str, Any]) -> None:
        """Post alert to Slack incoming webhook."""
        if not self.slack_webhook_url:
            logger.info("Slack webhook not configured; skipping Slack delivery")
            return

        payload = {
            "text": f"[{event['severity'].upper()}] {event['alert_type']}: {event['message']}",
            "attachments": [
                {
                    "color": self._slack_color(event["severity"]),
                    "fields": [
                        {"title": "Type", "value": event["alert_type"], "short": True},
                        {"title": "Severity", "value": event["severity"], "short": True},
                        {
                            "title": "Timestamp",
                            "value": event["timestamp"],
                            "short": False,
                        },
                    ],
                }
            ],
        }
        try:
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.exception("Slack delivery failed")
            raise RuntimeError("Slack delivery failed") from exc

    def _send_to_pagerduty(self, event: Dict[str, Any]) -> None:
        """Trigger PagerDuty incident using Events API v2."""
        if not self.pagerduty_routing_key:
            logger.warning("PagerDuty routing key not configured; critical alert not escalated")
            return

        payload = {
            "routing_key": self.pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{event['alert_type']}: {event['message']}",
                "severity": "critical",
                "source": "apex-trading",
                "timestamp": event["timestamp"],
                "custom_details": event.get("metadata") or {},
            },
        }
        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.exception("PagerDuty delivery failed")
            raise RuntimeError("PagerDuty delivery failed") from exc

    @staticmethod
    def _slack_color(severity: str) -> str:
        """Map severity to Slack attachment color."""
        if severity == AlertSeverity.INFO.value:
            return "#36a64f"
        if severity == AlertSeverity.WARNING.value:
            return "#ffcc00"
        return "#d40e0e"

    def get_alert_stats(self) -> Dict[str, Any]:
        """Return aggregate stats for emitted alerts."""
        return {
            "total_unique_alerts": len(self.alert_counts),
            "total_alerts_sent": sum(self.alert_counts.values()),
            "alerts_by_type": dict(self.alert_counts),
            "cooldown_seconds": self.cooldown_seconds,
        }

    def reset_cooldowns(self) -> None:
        """Reset in-memory alert cooldown state."""
        self.alert_cache.clear()
        logger.info("Alert cooldowns reset")


_aggregator: Optional[AlertAggregator] = None


def get_alert_aggregator(
    cooldown_seconds: int = 300,
    log_file: Optional[Path] = None,
    slack_webhook_url: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
) -> AlertAggregator:
    """Return singleton alert aggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = AlertAggregator(
            cooldown_seconds=cooldown_seconds,
            log_file=log_file,
            slack_webhook_url=slack_webhook_url,
            pagerduty_routing_key=pagerduty_routing_key,
        )
    return _aggregator


def send_alert(
    alert_type: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Send a routed alert using the global aggregator."""
    aggregator = get_alert_aggregator()
    aggregator.send_alert(alert_type, message, severity, metadata)
