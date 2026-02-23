"""
monitoring/business_metrics.py - Business Metrics Tracking

Tracks trading-specific business metrics beyond basic system metrics.
Provides insights into trading performance and execution quality.

Features:
- Trade execution quality metrics (slippage, fill rates)
- P&L tracking (daily, cumulative)
- Signal accuracy tracking
- Risk metrics (drawdown, VaR)
- Alert thresholds with notifications
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert triggered by metric threshold."""
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TradeExecution:
    """Record of a trade execution for analysis."""
    symbol: str
    side: str
    quantity: int
    expected_price: float
    fill_price: float
    timestamp: datetime
    order_id: str = ""
    execution_time_ms: float = 0.0
    venue: str = ""

    @property
    def slippage_bps(self) -> float:
        """Calculate slippage in basis points."""
        if self.expected_price == 0:
            return 0.0
        return (self.fill_price - self.expected_price) / self.expected_price * 10000

    @property
    def slippage_value(self) -> float:
        """Calculate slippage in dollar value."""
        return (self.fill_price - self.expected_price) * self.quantity


@dataclass
class DailyPnL:
    """Daily P&L record."""
    date: date
    starting_capital: float
    ending_capital: float
    realized_pnl: float
    unrealized_pnl: float
    trades_count: int
    winning_trades: int
    losing_trades: int

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return self.total_pnl / self.starting_capital * 100

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total * 100


@dataclass
class SignalMetrics:
    """Metrics for signal quality."""
    total_signals: int = 0
    profitable_signals: int = 0
    total_return: float = 0.0
    avg_holding_period_days: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.profitable_signals / self.total_signals * 100


class MetricThreshold:
    """Threshold configuration for alerts."""

    def __init__(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison: str = "greater"  # "greater" or "less"
    ):
        self.metric_name = metric_name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.comparison = comparison

    def check(self, value: float) -> Optional[AlertSeverity]:
        """Check if value exceeds threshold."""
        if self.comparison == "greater":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        else:  # less
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        return None


class BusinessMetrics:
    """
    Central business metrics tracker.

    Example:
        metrics = BusinessMetrics()

        # Record trade execution
        metrics.record_execution(TradeExecution(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            expected_price=185.00,
            fill_price=185.05,
            timestamp=datetime.now()
        ))

        # Get execution quality
        quality = metrics.get_execution_quality()
        print(f"Avg slippage: {quality['avg_slippage_bps']:.2f} bps")

        # Check for alerts
        alerts = metrics.check_thresholds()
    """

    def __init__(self):
        # Execution tracking
        self.executions: deque = deque(maxlen=10000)
        self.executions_by_symbol: Dict[str, List[TradeExecution]] = {}

        # P&L tracking
        self.daily_pnl: Dict[date, DailyPnL] = {}
        self.current_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.starting_capital: float = 0.0

        # Signal tracking
        self.signal_metrics: Dict[str, SignalMetrics] = {}

        # Thresholds
        self.thresholds: List[MetricThreshold] = [
            MetricThreshold("daily_loss_pct", -2.0, -3.0, "less"),
            MetricThreshold("drawdown_pct", -7.0, -10.0, "less"),
            MetricThreshold("avg_slippage_bps", 10.0, 20.0, "greater"),
            MetricThreshold("consecutive_losses", 3, 5, "greater"),
        ]

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Alert history
        self.alerts: deque = deque(maxlen=1000)

        # Consecutive loss tracking
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0

    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)

    def _trigger_alert(self, alert: Alert):
        """Trigger alert callbacks."""
        self.alerts.append(alert)
        logger.log(
            logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.WARNING,
            f"Alert: {alert.message}"
        )
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    # ========================================================================
    # Execution Quality Metrics
    # ========================================================================

    def record_execution(self, execution: TradeExecution):
        """Record a trade execution."""
        self.executions.append(execution)

        if execution.symbol not in self.executions_by_symbol:
            self.executions_by_symbol[execution.symbol] = []
        self.executions_by_symbol[execution.symbol].append(execution)

        # Log if slippage is high
        if abs(execution.slippage_bps) > 10:
            logger.warning(
                f"High slippage on {execution.symbol}: {execution.slippage_bps:.2f} bps"
            )

    def get_execution_quality(
        self,
        symbol: str = None,
        since: datetime = None
    ) -> Dict[str, float]:
        """
        Get execution quality metrics.

        Args:
            symbol: Filter by symbol (None for all)
            since: Filter by time (None for all)

        Returns:
            Dict with slippage, fill rate, and execution time metrics
        """
        executions = list(self.executions)

        if symbol:
            executions = [e for e in executions if e.symbol == symbol]

        if since:
            executions = [e for e in executions if e.timestamp >= since]

        if not executions:
            return {
                'total_executions': 0,
                'avg_slippage_bps': 0.0,
                'total_slippage_value': 0.0,
                'avg_execution_time_ms': 0.0,
                'positive_slippage_pct': 0.0,
            }

        slippages = [e.slippage_bps for e in executions]
        slippage_values = [e.slippage_value for e in executions]
        exec_times = [e.execution_time_ms for e in executions if e.execution_time_ms > 0]

        positive_slippage = sum(1 for s in slippages if s > 0)

        return {
            'total_executions': len(executions),
            'avg_slippage_bps': sum(slippages) / len(slippages),
            'max_slippage_bps': max(slippages),
            'min_slippage_bps': min(slippages),
            'total_slippage_value': sum(slippage_values),
            'avg_execution_time_ms': sum(exec_times) / len(exec_times) if exec_times else 0,
            'positive_slippage_pct': positive_slippage / len(executions) * 100,
        }

    # ========================================================================
    # P&L Metrics
    # ========================================================================

    def set_starting_capital(self, capital: float):
        """Set starting capital."""
        self.starting_capital = capital
        self.current_capital = capital
        self.peak_capital = capital

    def update_capital(self, current_capital: float):
        """Update current capital and track peak."""
        self.current_capital = current_capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital

    def record_daily_pnl(self, pnl: DailyPnL):
        """Record daily P&L."""
        self.daily_pnl[pnl.date] = pnl

        # Update consecutive wins/losses
        if pnl.total_pnl >= 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Check thresholds
        self.check_thresholds()

    def get_drawdown(self) -> Dict[str, float]:
        """Calculate current drawdown metrics."""
        if self.peak_capital == 0:
            return {'current_drawdown': 0.0, 'max_drawdown': 0.0}

        current_dd = (self.current_capital - self.peak_capital) / self.peak_capital * 100

        # Calculate max drawdown from history
        max_dd = current_dd
        running_peak = self.starting_capital

        for d in sorted(self.daily_pnl.keys()):
            pnl = self.daily_pnl[d]
            if pnl.ending_capital > running_peak:
                running_peak = pnl.ending_capital
            dd = (pnl.ending_capital - running_peak) / running_peak * 100
            if dd < max_dd:
                max_dd = dd

        return {
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'peak_capital': self.peak_capital,
            'current_capital': self.current_capital,
        }

    def get_cumulative_pnl(self) -> Dict[str, float]:
        """Get cumulative P&L metrics."""
        if not self.daily_pnl:
            return {
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'trading_days': 0,
                'profitable_days': 0,
                'avg_daily_return': 0.0,
            }

        total_pnl = sum(p.total_pnl for p in self.daily_pnl.values())
        profitable_days = sum(1 for p in self.daily_pnl.values() if p.total_pnl > 0)
        returns = [p.return_pct for p in self.daily_pnl.values()]

        return {
            'total_pnl': total_pnl,
            'total_return_pct': total_pnl / self.starting_capital * 100 if self.starting_capital else 0,
            'trading_days': len(self.daily_pnl),
            'profitable_days': profitable_days,
            'avg_daily_return': sum(returns) / len(returns) if returns else 0,
            'best_day': max(returns) if returns else 0,
            'worst_day': min(returns) if returns else 0,
        }

    # ========================================================================
    # Signal Metrics
    # ========================================================================

    def record_signal_outcome(
        self,
        signal_type: str,
        profitable: bool,
        return_pct: float,
        holding_days: int
    ):
        """Record signal outcome for accuracy tracking."""
        if signal_type not in self.signal_metrics:
            self.signal_metrics[signal_type] = SignalMetrics()

        metrics = self.signal_metrics[signal_type]
        metrics.total_signals += 1
        if profitable:
            metrics.profitable_signals += 1
        metrics.total_return += return_pct

        # Update average holding period
        total = metrics.total_signals
        metrics.avg_holding_period_days = (
            (metrics.avg_holding_period_days * (total - 1) + holding_days) / total
        )

    def get_signal_metrics(self, signal_type: str = None) -> Dict[str, Any]:
        """Get signal accuracy metrics."""
        if signal_type:
            if signal_type not in self.signal_metrics:
                return {}
            m = self.signal_metrics[signal_type]
            return {
                'type': signal_type,
                'total_signals': m.total_signals,
                'accuracy': m.accuracy,
                'total_return': m.total_return,
                'avg_holding_days': m.avg_holding_period_days,
            }

        # Return all signal types
        return {
            name: {
                'total_signals': m.total_signals,
                'accuracy': m.accuracy,
                'total_return': m.total_return,
            }
            for name, m in self.signal_metrics.items()
        }

    # ========================================================================
    # Alert Thresholds
    # ========================================================================

    def add_threshold(self, threshold: MetricThreshold):
        """Add a metric threshold."""
        self.thresholds.append(threshold)

    def check_thresholds(self) -> List[Alert]:
        """Check all thresholds and trigger alerts."""
        triggered = []

        # Get current metric values
        metrics = {
            'daily_loss_pct': self._get_daily_loss_pct(),
            'drawdown_pct': self.get_drawdown()['current_drawdown'],
            'avg_slippage_bps': self.get_execution_quality()['avg_slippage_bps'],
            'consecutive_losses': self.consecutive_losses,
        }

        for threshold in self.thresholds:
            if threshold.metric_name in metrics:
                value = metrics[threshold.metric_name]
                severity = threshold.check(value)

                if severity:
                    alert = Alert(
                        metric_name=threshold.metric_name,
                        severity=severity,
                        message=f"{threshold.metric_name}: {value:.2f} exceeds threshold",
                        current_value=value,
                        threshold=threshold.critical_threshold if severity == AlertSeverity.CRITICAL else threshold.warning_threshold
                    )
                    triggered.append(alert)
                    self._trigger_alert(alert)

        return triggered

    def _get_daily_loss_pct(self) -> float:
        """Get today's P&L percentage."""
        today = date.today()
        if today in self.daily_pnl:
            return self.daily_pnl[today].return_pct
        return 0.0

    # ========================================================================
    # Reporting
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        return {
            'execution_quality': self.get_execution_quality(),
            'pnl': self.get_cumulative_pnl(),
            'drawdown': self.get_drawdown(),
            'signals': self.get_signal_metrics(),
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'active_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]),
        }

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'daily_pnl': {
                str(d): {
                    'pnl': p.total_pnl,
                    'return_pct': p.return_pct,
                    'trades': p.trades_count,
                    'win_rate': p.win_rate,
                }
                for d, p in sorted(self.daily_pnl.items())
            },
            'recent_alerts': [a.to_dict() for a in list(self.alerts)[-50:]],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")


# Global instance
_metrics: Optional[BusinessMetrics] = None


def get_business_metrics() -> BusinessMetrics:
    """Get global business metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = BusinessMetrics()
    return _metrics


def init_business_metrics(starting_capital: float) -> BusinessMetrics:
    """Initialize business metrics with starting capital."""
    global _metrics
    _metrics = BusinessMetrics()
    _metrics.set_starting_capital(starting_capital)
    return _metrics
