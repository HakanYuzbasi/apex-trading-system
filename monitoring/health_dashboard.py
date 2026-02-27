"""
monitoring/health_dashboard.py - System Health Monitoring Dashboard

Real-time monitoring of:
- System health (heartbeat, connections)
- Trading performance (P&L, positions)
- Risk metrics (drawdown, VaR)
- Data quality
- Execution quality

Provides alerts and status checks.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    value: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'value': self.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemMetrics:
    """Current system metrics snapshot."""
    timestamp: datetime
    
    # System health
    is_running: bool
    last_heartbeat: Optional[datetime]
    ibkr_connected: bool
    data_feed_active: bool
    
    # Trading metrics
    total_capital: float
    daily_pnl: float
    daily_pnl_pct: float
    open_positions: int
    pending_orders: int
    
    # Risk metrics
    current_drawdown: float
    max_drawdown: float
    var_95: float
    portfolio_beta: float
    
    # Performance
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    
    # Data quality
    stale_symbols: int
    data_issues: int
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'system': {
                'is_running': self.is_running,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                'ibkr_connected': self.ibkr_connected,
                'data_feed_active': self.data_feed_active
            },
            'trading': {
                'total_capital': self.total_capital,
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': self.daily_pnl_pct,
                'open_positions': self.open_positions,
                'pending_orders': self.pending_orders
            },
            'risk': {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'var_95': self.var_95,
                'portfolio_beta': self.portfolio_beta
            },
            'performance': {
                'sharpe_ratio': self.sharpe_ratio,
                'win_rate': self.win_rate,
                'total_trades': self.total_trades
            },
            'data_quality': {
                'stale_symbols': self.stale_symbols,
                'data_issues': self.data_issues
            }
        }


class HealthDashboard:
    """
    System health monitoring and alerting.
    
    Features:
    - Real-time health checks
    - Alert generation
    - Metric tracking
    - Report generation
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize health dashboard.
        
        Args:
            data_dir: Directory for data files
            alert_callback: Optional function to call on alerts
        """
        self.data_dir = Path(data_dir)
        self.alert_callback = alert_callback
        
        # Alert thresholds
        self.thresholds = {
            'heartbeat_stale_seconds': 120,
            'max_daily_loss_pct': 0.02,
            'max_drawdown_pct': 0.10,
            'max_stale_symbols': 5,
            'max_pending_orders': 10,
            'alert_repeat_cooldown_seconds': 300,
        }
        
        # History
        self.health_checks: List[HealthCheck] = []
        self.alerts: List[Dict] = []
        self.metrics_history: List[SystemMetrics] = []
        self._last_alert_logged_at: Dict[str, datetime] = {}
        
        logger.info("HealthDashboard initialized")
    
    def check_heartbeat(self) -> HealthCheck:
        """Check if system heartbeat is recent."""
        heartbeat_candidates = [self.data_dir / 'heartbeat.json', Path('data') / 'heartbeat.json']
        
        try:
            heartbeat_file = next((p for p in heartbeat_candidates if p.exists()), None)
            if heartbeat_file:
                with open(heartbeat_file) as f:
                    data = json.load(f)
                
                # Parse timestamp — handle both naive and Z-suffix UTC
                raw_ts = data.get('timestamp', '')
                raw_ts_clean = raw_ts.replace('Z', '+00:00') if raw_ts.endswith('Z') else raw_ts
                last_beat = datetime.fromisoformat(raw_ts_clean)
                if last_beat.tzinfo is None:
                    last_beat = last_beat.replace(tzinfo=timezone.utc)
                age_seconds = (datetime.now(timezone.utc) - last_beat).total_seconds()
                
                if age_seconds < 60:
                    return HealthCheck(
                        name='heartbeat',
                        status=HealthStatus.HEALTHY,
                        message=f'Heartbeat OK ({age_seconds:.0f}s ago)',
                        value=age_seconds
                    )
                elif age_seconds < self.thresholds['heartbeat_stale_seconds']:
                    return HealthCheck(
                        name='heartbeat',
                        status=HealthStatus.WARNING,
                        message=f'Heartbeat stale ({age_seconds:.0f}s)',
                        value=age_seconds
                    )
                else:
                    return HealthCheck(
                        name='heartbeat',
                        status=HealthStatus.ERROR,
                        message=f'Heartbeat very stale ({age_seconds:.0f}s)',
                        value=age_seconds
                    )
            else:
                return HealthCheck(
                    name='heartbeat',
                    status=HealthStatus.ERROR,
                    message='No heartbeat file found'
                )
        except Exception as e:
            return HealthCheck(
                name='heartbeat',
                status=HealthStatus.ERROR,
                message=f'Heartbeat check failed: {e}'
            )
    
    def check_trading_state(self) -> HealthCheck:
        """Check trading state from state file."""
        state_file = self.data_dir / 'trading_state.json'
        
        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                
                capital = state.get('capital', 0)
                daily_pnl = state.get('daily_pnl', 0)
                daily_pnl_pct = daily_pnl / capital if capital > 0 else 0
                
                if daily_pnl_pct < -self.thresholds['max_daily_loss_pct']:
                    return HealthCheck(
                        name='trading_state',
                        status=HealthStatus.ERROR,
                        message=f'Daily loss exceeds threshold: {daily_pnl_pct*100:.2f}%',
                        value=daily_pnl_pct
                    )
                elif daily_pnl < 0:
                    return HealthCheck(
                        name='trading_state',
                        status=HealthStatus.WARNING,
                        message=f'Daily loss: ${abs(daily_pnl):,.2f} ({daily_pnl_pct*100:.2f}%)',
                        value=daily_pnl_pct
                    )
                else:
                    return HealthCheck(
                        name='trading_state',
                        status=HealthStatus.HEALTHY,
                        message=f'Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct*100:+.2f}%)',
                        value=daily_pnl_pct
                    )
            else:
                return HealthCheck(
                    name='trading_state',
                    status=HealthStatus.WARNING,
                    message='No trading state file'
                )
        except Exception as e:
            return HealthCheck(
                name='trading_state',
                status=HealthStatus.ERROR,
                message=f'State check failed: {e}'
            )
    
    def check_drawdown(self, current_capital: float, peak_capital: float) -> HealthCheck:
        """Check drawdown level."""
        if peak_capital <= 0:
            return HealthCheck(
                name='drawdown',
                status=HealthStatus.WARNING,
                message='No peak capital set'
            )
        
        drawdown = (peak_capital - current_capital) / peak_capital
        
        if drawdown >= self.thresholds['max_drawdown_pct']:
            return HealthCheck(
                name='drawdown',
                status=HealthStatus.CRITICAL,
                message=f'Drawdown critical: {drawdown*100:.1f}%',
                value=drawdown
            )
        elif drawdown >= self.thresholds['max_drawdown_pct'] * 0.7:
            return HealthCheck(
                name='drawdown',
                status=HealthStatus.WARNING,
                message=f'Drawdown elevated: {drawdown*100:.1f}%',
                value=drawdown
            )
        else:
            return HealthCheck(
                name='drawdown',
                status=HealthStatus.HEALTHY,
                message=f'Drawdown OK: {drawdown*100:.1f}%',
                value=drawdown
            )
    
    def check_positions(self, positions: Dict[str, int]) -> HealthCheck:
        """Check position health."""
        num_positions = len([q for q in positions.values() if q != 0])
        
        # Calculate concentration
        if num_positions == 0:
            return HealthCheck(
                name='positions',
                status=HealthStatus.HEALTHY,
                message='No open positions'
            )
        
        return HealthCheck(
            name='positions',
            status=HealthStatus.HEALTHY,
            message=f'{num_positions} open positions',
            value=num_positions
        )
    
    def run_all_checks(
        self,
        current_capital: float = 0,
        peak_capital: float = 0,
        positions: Optional[Dict[str, int]] = None
    ) -> List[HealthCheck]:
        """
        Run all health checks.
        
        Returns:
            List of HealthCheck results
        """
        checks = []
        
        # Core checks
        checks.append(self.check_heartbeat())
        checks.append(self.check_trading_state())
        
        # Risk checks
        if current_capital > 0 and peak_capital > 0:
            checks.append(self.check_drawdown(current_capital, peak_capital))
        
        # Position checks
        if positions:
            checks.append(self.check_positions(positions))
        
        # Store checks
        self.health_checks.extend(checks)
        
        # Trim history
        if len(self.health_checks) > 1000:
            self.health_checks = self.health_checks[-1000:]
        
        # Generate alerts for errors
        self._process_alerts(checks)
        
        return checks
    
    def _process_alerts(self, checks: List[HealthCheck]):
        """Process health checks and generate alerts."""
        now = datetime.now(timezone.utc)
        repeat_cooldown = max(1, int(self.thresholds.get('alert_repeat_cooldown_seconds', 300)))
        for check in checks:
            if check.status in [HealthStatus.ERROR, HealthStatus.CRITICAL]:
                signature = f"{check.name}:{check.status.value}:{check.message}"
                last_logged = self._last_alert_logged_at.get(signature)
                if last_logged and (now - last_logged.replace(tzinfo=timezone.utc) if last_logged.tzinfo is None else now - last_logged).total_seconds() < repeat_cooldown:
                    continue
                self._last_alert_logged_at[signature] = now
                alert = {
                    'check': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'timestamp': now.isoformat()
                }
                
                self.alerts.append(alert)
                
                # Log alert
                if check.status == HealthStatus.CRITICAL:
                    logger.critical(f"🚨 CRITICAL: {check.message}")
                else:
                    logger.error(f"⚠️ ERROR: {check.message}")
                
                # Callback
                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_checks:
            return HealthStatus.WARNING
        
        # Look at recent checks (last minute)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=1)
        recent = [c for c in self.health_checks if (c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=timezone.utc)) >= cutoff]
        
        if not recent:
            return HealthStatus.WARNING
        
        # Worst status
        if any(c.status == HealthStatus.CRITICAL for c in recent):
            return HealthStatus.CRITICAL
        if any(c.status == HealthStatus.ERROR for c in recent):
            return HealthStatus.ERROR
        if any(c.status == HealthStatus.WARNING for c in recent):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def get_summary(self) -> Dict:
        """Get health summary."""
        status = self.get_overall_status()
        
        # Recent alerts
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts if datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00')).replace(tzinfo=timezone.utc) >= cutoff]
        
        return {
            'status': status.value,
            'status_icon': self._status_icon(status),
            'recent_alerts': recent_alerts[-5:],
            'alert_count_1h': len(recent_alerts),
            'last_check': self.health_checks[-1].to_dict() if self.health_checks else None
        }
    
    def _status_icon(self, status: HealthStatus) -> str:
        """Get emoji icon for status."""
        return {
            HealthStatus.HEALTHY: '✅',
            HealthStatus.WARNING: '⚠️',
            HealthStatus.ERROR: '❌',
            HealthStatus.CRITICAL: '🚨'
        }.get(status, '❓')
    
    def generate_report(self) -> str:
        """Generate text health report."""
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            f"SYSTEM HEALTH REPORT  {summary['status_icon']} {summary['status'].upper()}",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Recent checks
        lines.append("Recent Health Checks:")
        for check in self.health_checks[-10:]:
            icon = self._status_icon(check.status)
            lines.append(f"  {icon} {check.name}: {check.message}")
        
        # Alerts
        if summary['recent_alerts']:
            lines.append("")
            lines.append(f"Recent Alerts ({summary['alert_count_1h']} in last hour):")
            for alert in summary['recent_alerts']:
                lines.append(f"  🔔 [{alert['status']}] {alert['message']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
