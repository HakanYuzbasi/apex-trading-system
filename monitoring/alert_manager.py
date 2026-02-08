"""
monitoring/alert_manager.py - Real-Time Alert Management System

Provides real-time alerting for trading system events including:
- Risk threshold breaches (drawdown, VIX, correlation)
- System health issues (connection failures, data staleness)
- Trading events (circuit breaker triggers, large trades)
- Performance alerts (signal degradation, execution quality)

Supports multiple notification channels:
- In-app notifications (WebSocket push)
- Log-based alerts
- Webhook integrations (Slack, Discord, etc.)
- Email notifications (optional)
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta

from config import ApexConfig
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"           # Informational
    WARNING = "warning"     # Needs attention
    CRITICAL = "critical"   # Immediate action required
    EMERGENCY = "emergency" # System halt condition


class AlertCategory(Enum):
    """Categories of alerts."""
    RISK = "risk"               # Risk management alerts
    EXECUTION = "execution"     # Trade execution alerts
    SYSTEM = "system"           # System health alerts
    PERFORMANCE = "performance" # Performance degradation
    COMPLIANCE = "compliance"   # Compliance/regulatory
    MARKET = "market"          # Market condition alerts


@dataclass
class Alert:
    """Represents a single alert."""
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str                          # Component that raised alert
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'alert_id': self.alert_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    rule_id: str
    name: str
    condition: Callable[[], bool]     # Function that returns True when alert should fire
    category: AlertCategory
    severity: AlertSeverity
    message_template: str             # Can include {variable} placeholders
    cooldown_seconds: int = 300       # Minimum time between repeated alerts
    enabled: bool = True
    metadata_provider: Optional[Callable[[], Dict]] = None  # Function to get dynamic metadata


class AlertChannel:
    """Base class for alert notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel. Returns True on success."""
        raise NotImplementedError


class LogChannel(AlertChannel):
    """Log-based alert channel."""

    def __init__(self, logger_name: str = "alerts"):
        self.alert_logger = logging.getLogger(logger_name)

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }

        level = level_map.get(alert.severity, logging.WARNING)
        self.alert_logger.log(
            level,
            f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        )
        return True


class WebhookChannel(AlertChannel):
    """Webhook-based alert channel (Slack, Discord, etc.)."""

    def __init__(
        self,
        webhook_url: str,
        format_type: str = "slack",  # "slack", "discord", "generic"
        timeout: int = 10
    ):
        self.webhook_url = webhook_url
        self.format_type = format_type
        self.timeout = timeout

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            payload = self._format_payload(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    return response.status < 400

        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def _format_payload(self, alert: Alert) -> Dict:
        """Format payload based on webhook type."""
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ffcc00",   # Yellow
            AlertSeverity.CRITICAL: "#ff6600",  # Orange
            AlertSeverity.EMERGENCY: "#ff0000"  # Red
        }

        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔴"
        }

        if self.format_type == "slack":
            return {
                "attachments": [{
                    "color": severity_colors.get(alert.severity, "#808080"),
                    "title": f"{severity_emoji.get(alert.severity, '')} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%H:%M:%S"), "short": True}
                    ],
                    "footer": "APEX Trading System"
                }]
            }
        elif self.format_type == "discord":
            return {
                "embeds": [{
                    "title": f"{severity_emoji.get(alert.severity, '')} {alert.title}",
                    "description": alert.message,
                    "color": int(severity_colors.get(alert.severity, "#808080").lstrip('#'), 16),
                    "fields": [
                        {"name": "Category", "value": alert.category.value, "inline": True},
                        {"name": "Severity", "value": alert.severity.value, "inline": True}
                    ],
                    "timestamp": alert.timestamp.isoformat()
                }]
            }
        else:
            return alert.to_dict()


class WebSocketChannel(AlertChannel):
    """WebSocket channel for real-time frontend updates."""

    def __init__(self, broadcast_func: Optional[Callable[[Dict], Any]] = None):
        self.broadcast_func = broadcast_func
        self.pending_alerts: List[Alert] = []

    def set_broadcast_func(self, func: Callable[[Dict], Any]):
        """Set the broadcast function (typically from FastAPI WebSocket manager)."""
        self.broadcast_func = func

    async def send(self, alert: Alert) -> bool:
        """Broadcast alert to connected WebSocket clients."""
        if self.broadcast_func:
            try:
                await self.broadcast_func({
                    "type": "alert",
                    "data": alert.to_dict()
                })
                return True
            except Exception as e:
                logger.error(f"WebSocket alert broadcast failed: {e}")
                return False
        else:
            # Queue for later if no broadcast function set
            self.pending_alerts.append(alert)
            return True


class AlertManager:
    """
    Central alert management system.

    Usage:
        manager = AlertManager()

        # Register alert rules
        manager.register_rule(
            rule_id="high_vix",
            name="High VIX Alert",
            condition=lambda: self.vix > 30,
            category=AlertCategory.MARKET,
            severity=AlertSeverity.WARNING,
            message_template="VIX has risen to {vix:.1f}",
            metadata_provider=lambda: {"vix": self.vix}
        )

        # Check rules periodically
        await manager.check_rules()

        # Or trigger alerts directly
        manager.trigger_alert(
            category=AlertCategory.RISK,
            severity=AlertSeverity.CRITICAL,
            title="Circuit Breaker Triggered",
            message="Trading halted due to 5% drawdown",
            source="risk_manager"
        )
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_alerts: int = 1000
    ):
        """
        Initialize the alert manager.

        Args:
            storage_path: Path to persist alerts (optional)
            max_alerts: Maximum alerts to keep in memory
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_alerts = max_alerts

        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Alert rules
        self.rules: Dict[str, AlertRule] = {}
        self.last_triggered: Dict[str, datetime] = {}

        # Notification channels
        self.channels: List[AlertChannel] = [LogChannel()]  # Log by default

        # Metrics tracking
        self.metrics = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_category': defaultdict(int),
            'alerts_today': 0,
            'last_alert_time': None
        }

        # State for condition checking
        self._state: Dict[str, Any] = {}

        # Load persisted alerts
        self._load_alerts()

        logger.info("AlertManager initialized")

    def set_state(self, key: str, value: Any):
        """Set state value for condition checking."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.get(key, default)

    def add_channel(self, channel: AlertChannel):
        """Add a notification channel."""
        self.channels.append(channel)
        logger.info(f"Added alert channel: {type(channel).__name__}")

    def register_rule(
        self,
        rule_id: str,
        name: str,
        condition: Callable[[], bool],
        category: AlertCategory,
        severity: AlertSeverity,
        message_template: str,
        cooldown_seconds: int = 300,
        metadata_provider: Optional[Callable[[], Dict]] = None
    ):
        """Register an alert rule."""
        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            condition=condition,
            category=category,
            severity=severity,
            message_template=message_template,
            cooldown_seconds=cooldown_seconds,
            metadata_provider=metadata_provider
        )
        self.rules[rule_id] = rule
        logger.debug(f"Registered alert rule: {rule_id}")

    def unregister_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def enable_rule(self, rule_id: str):
        """Enable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """Disable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    async def check_rules(self) -> List[Alert]:
        """
        Check all registered rules and trigger alerts as needed.

        Returns:
            List of triggered alerts
        """
        triggered = []

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            last_time = self.last_triggered.get(rule_id)
            if last_time:
                elapsed = (datetime.now() - last_time).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue

            # Check condition
            try:
                if rule.condition():
                    # Get metadata
                    metadata = {}
                    if rule.metadata_provider:
                        metadata = rule.metadata_provider()

                    # Format message
                    message = rule.message_template.format(**metadata, **self._state)

                    # Trigger alert
                    alert = await self.trigger_alert(
                        category=rule.category,
                        severity=rule.severity,
                        title=rule.name,
                        message=message,
                        source=f"rule:{rule_id}",
                        metadata=metadata
                    )

                    if alert:
                        triggered.append(alert)
                        self.last_triggered[rule_id] = datetime.now()

            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {e}")

        return triggered

    async def trigger_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict] = None
    ) -> Optional[Alert]:
        """
        Trigger an alert manually.

        Args:
            category: Alert category
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Source component
            metadata: Additional metadata

        Returns:
            Created Alert object
        """
        alert_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        alert = Alert(
            alert_id=alert_id,
            category=category,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )

        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Trim history if needed
        if len(self.alert_history) > self.max_alerts:
            self.alert_history = self.alert_history[-self.max_alerts:]

        # Update metrics
        self.metrics['total_alerts'] += 1
        self.metrics['alerts_by_severity'][severity.value] += 1
        self.metrics['alerts_by_category'][category.value] += 1
        self.metrics['alerts_today'] += 1
        self.metrics['last_alert_time'] = datetime.now()

        # Send to all channels
        await self._broadcast_alert(alert)

        # Persist
        self._save_alerts()

        logger.info(
            f"Alert triggered: [{severity.value}] {title} - {message}"
        )

        return alert

    async def _broadcast_alert(self, alert: Alert):
        """Send alert to all channels."""
        for channel in self.channels:
            try:
                await channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(channel).__name__}: {e}")

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "system"
    ) -> bool:
        """Mark an alert as acknowledged."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.alerts[alert_id].acknowledged_at = datetime.now()
            self.alerts[alert_id].acknowledged_by = acknowledged_by
            self._save_alerts()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            self._save_alerts()
            return True
        return False

    def get_active_alerts(
        self,
        category: Optional[AlertCategory] = None,
        min_severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active (unresolved) alerts."""
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.CRITICAL,
            AlertSeverity.EMERGENCY
        ]

        alerts = [a for a in self.alerts.values() if not a.resolved]

        if category:
            alerts = [a for a in alerts if a.category == category]

        if min_severity:
            min_idx = severity_order.index(min_severity)
            alerts = [
                a for a in alerts
                if severity_order.index(a.severity) >= min_idx
            ]

        # Sort by severity (highest first) then timestamp (newest first)
        alerts.sort(
            key=lambda a: (
                -severity_order.index(a.severity),
                -a.timestamp.timestamp()
            )
        )

        return alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        active = self.get_active_alerts()

        return {
            'total_alerts': self.metrics['total_alerts'],
            'active_alerts': len(active),
            'critical_alerts': len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            'emergency_alerts': len([a for a in active if a.severity == AlertSeverity.EMERGENCY]),
            'alerts_today': self.metrics['alerts_today'],
            'last_alert_time': self.metrics['last_alert_time'].isoformat() if self.metrics['last_alert_time'] else None,
            'by_category': dict(self.metrics['alerts_by_category']),
            'by_severity': dict(self.metrics['alerts_by_severity'])
        }

    def _save_alerts(self):
        """Persist alerts to disk."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Save recent alerts
            data = {
                'alerts': [a.to_dict() for a in self.alert_history[-100:]],
                'metrics': {
                    'total_alerts': self.metrics['total_alerts'],
                    'alerts_today': self.metrics['alerts_today'],
                    'last_alert_time': self.metrics['last_alert_time'].isoformat() if self.metrics['last_alert_time'] else None
                },
                'saved_at': datetime.now().isoformat()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _load_alerts(self):
        """Load persisted alerts."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Restore alerts (mark old ones as resolved)
            for alert_data in data.get('alerts', []):
                alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                alert_data['category'] = AlertCategory(alert_data['category'])
                alert_data['severity'] = AlertSeverity(alert_data['severity'])
                if alert_data.get('acknowledged_at'):
                    alert_data['acknowledged_at'] = datetime.fromisoformat(alert_data['acknowledged_at'])
                if alert_data.get('resolved_at'):
                    alert_data['resolved_at'] = datetime.fromisoformat(alert_data['resolved_at'])

                alert = Alert(**alert_data)
                self.alerts[alert.alert_id] = alert
                self.alert_history.append(alert)

            logger.info(f"Loaded {len(self.alerts)} alerts from storage")

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")

    def reset_daily_metrics(self):
        """Reset daily metrics (call at market open)."""
        self.metrics['alerts_today'] = 0


def setup_trading_alerts(manager: AlertManager, get_state_func: Callable[[], Dict]):
    """
    Set up standard trading alert rules.

    Args:
        manager: AlertManager instance
        get_state_func: Function that returns current trading state
    """

    def get_vix():
        return get_state_func().get('vix', 0)

    def get_drawdown():
        return get_state_func().get('drawdown', 0)

    def get_daily_pnl():
        return get_state_func().get('daily_pnl', 0)

    def get_signal_accuracy():
        return get_state_func().get('signal_accuracy', 1.0)

    # VIX Alerts
    manager.register_rule(
        rule_id="high_vix_warning",
        name="Elevated VIX Warning",
        condition=lambda: get_vix() > 25,
        category=AlertCategory.MARKET,
        severity=AlertSeverity.WARNING,
        message_template="VIX has risen to {vix:.1f} - consider reducing exposure",
        cooldown_seconds=1800,  # 30 min cooldown
        metadata_provider=lambda: {"vix": get_vix()}
    )

    manager.register_rule(
        rule_id="high_vix_critical",
        name="High VIX Alert",
        condition=lambda: get_vix() > 35,
        category=AlertCategory.MARKET,
        severity=AlertSeverity.CRITICAL,
        message_template="VIX at {vix:.1f} - extreme volatility regime",
        cooldown_seconds=3600,
        metadata_provider=lambda: {"vix": get_vix()}
    )

    # Drawdown Alerts
    manager.register_rule(
        rule_id="drawdown_warning",
        name="Drawdown Warning",
        condition=lambda: get_drawdown() > 0.03,
        category=AlertCategory.RISK,
        severity=AlertSeverity.WARNING,
        message_template="Portfolio drawdown at {drawdown:.1%}",
        cooldown_seconds=1800,
        metadata_provider=lambda: {"drawdown": get_drawdown()}
    )

    manager.register_rule(
        rule_id="drawdown_critical",
        name="Critical Drawdown",
        condition=lambda: get_drawdown() > 0.05,
        category=AlertCategory.RISK,
        severity=AlertSeverity.CRITICAL,
        message_template="Critical drawdown of {drawdown:.1%} - circuit breaker imminent",
        cooldown_seconds=3600,
        metadata_provider=lambda: {"drawdown": get_drawdown()}
    )

    # Daily PnL Alerts
    manager.register_rule(
        rule_id="daily_loss_warning",
        name="Daily Loss Warning",
        condition=lambda: get_daily_pnl() < -5000,
        category=AlertCategory.RISK,
        severity=AlertSeverity.WARNING,
        message_template="Daily loss of ${daily_pnl:,.0f}",
        cooldown_seconds=3600,
        metadata_provider=lambda: {"daily_pnl": get_daily_pnl()}
    )

    # Signal Quality Alerts
    manager.register_rule(
        rule_id="signal_degradation",
        name="Signal Quality Degradation",
        condition=lambda: get_signal_accuracy() < 0.45,
        category=AlertCategory.PERFORMANCE,
        severity=AlertSeverity.WARNING,
        message_template="Signal accuracy has dropped to {accuracy:.1%}",
        cooldown_seconds=7200,  # 2 hour cooldown
        metadata_provider=lambda: {"accuracy": get_signal_accuracy()}
    )

    logger.info("Standard trading alert rules registered")


# Global instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(
            storage_path=str(ApexConfig.DATA_DIR / "alerts.json")
        )
    return _alert_manager
