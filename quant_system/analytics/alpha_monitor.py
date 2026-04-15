from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from quant_system.analytics.notifier import TelegramNotifier
from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent, ExecutionEvent
from quant_system.portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlphaSnapshot:
    timestamp: datetime
    equity: float


class AlphaDecayMonitor:
    """
    Monitors live realized equity against expected walk-forward Sharpe.
    """

    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        notifier: TelegramNotifier,
        *,
        strategy_label: str = "Strategy",
        expected_oos_sharpe: float,
        decay_threshold: float = 0.50,
        evaluation_window_days: int = 30,
        alert_cooldown_hours: int = 24,
        alert_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        if expected_oos_sharpe <= 0:
            raise ValueError("expected_oos_sharpe must be positive")
        if not 0.0 < decay_threshold < 1.0:
            raise ValueError("decay_threshold must be in (0, 1)")
        if evaluation_window_days <= 0 or alert_cooldown_hours <= 0:
            raise ValueError("evaluation_window_days and alert_cooldown_hours must be positive")

        self._portfolio_ledger = portfolio_ledger
        self._event_bus = event_bus
        self._notifier = notifier
        self._strategy_label = strategy_label
        self._expected_oos_sharpe = float(expected_oos_sharpe)
        self._decay_threshold = float(decay_threshold)
        self._evaluation_window_days = int(evaluation_window_days)
        self._alert_cooldown = timedelta(hours=alert_cooldown_hours)
        self._alert_callback = alert_callback
        self._snapshots: list[AlphaSnapshot] = []
        self._last_alert_ts: datetime | None = None
        self._alert_active = False
        self._subscriptions: tuple[Subscription, ...] = (
            event_bus.subscribe("bar", self._on_bar, is_async=True),
            event_bus.subscribe("execution", self._on_execution, is_async=True),
        )

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    async def close(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription.token)

    async def _on_bar(self, event: BarEvent) -> None:
        await self._record_and_evaluate(event.exchange_ts)

    async def _on_execution(self, event: ExecutionEvent) -> None:
        if event.execution_status not in {"partial_fill", "filled"}:
            return
        await self._record_and_evaluate(event.exchange_ts)

    async def _record_and_evaluate(self, timestamp: datetime) -> None:
        equity = self._portfolio_ledger.total_equity()
        if self._snapshots:
            previous = self._snapshots[-1]
            if previous.timestamp == timestamp and abs(previous.equity - equity) <= 1e-9:
                return
        self._snapshots.append(AlphaSnapshot(timestamp=timestamp, equity=equity))
        live_sharpe = self.live_window_sharpe()
        if live_sharpe is None:
            return

        # Enforce a 24-hour warmup period for new deployments
        if (self._snapshots[-1].timestamp - self._snapshots[0].timestamp) < timedelta(hours=24):
            return

        threshold_sharpe = self._expected_oos_sharpe * (1.0 - self._decay_threshold)
        if live_sharpe >= threshold_sharpe:
            self._alert_active = False
            return
        if self._alert_active:
            return
        if self._last_alert_ts is not None and (timestamp - self._last_alert_ts) < self._alert_cooldown:
            return

        self._last_alert_ts = timestamp
        self._alert_active = True
        logger.warning(
            "Alpha decay detected strategy=%s live_sharpe=%.3f expected_sharpe=%.3f threshold=%.3f",
            self._strategy_label,
            live_sharpe,
            self._expected_oos_sharpe,
            threshold_sharpe,
        )
        await self._notifier.notify_text(
            f"🚨 ALPHA DECAY DETECTED: {self._strategy_label} performance deviating from historical norm."
        )
        if self._alert_callback is not None:
            await self._alert_callback(self._strategy_label)

    def live_window_sharpe(self) -> float | None:
        if len(self._snapshots) < 3:
            return None
        latest_timestamp = self._snapshots[-1].timestamp
        start_cutoff = latest_timestamp - timedelta(days=self._evaluation_window_days)
        filtered = [snapshot for snapshot in self._snapshots if snapshot.timestamp >= start_cutoff]
        if len(filtered) < 3:
            return None
        index = pd.DatetimeIndex([snapshot.timestamp for snapshot in filtered], tz="UTC")
        equity_curve = pd.Series([snapshot.equity for snapshot in filtered], index=index, dtype=float)
        metrics = PerformanceAnalyzer.compute_metrics_from_equity_curve(equity_curve)
        return float(metrics["annualized_sharpe"])
    def get_current_sortino(self) -> float:
        """Alias for localized sortino calculation used by v3 dashboard."""
        if len(self._snapshots) < 3:
            return 0.0
        latest_timestamp = self._snapshots[-1].timestamp
        start_cutoff = latest_timestamp - timedelta(days=self._evaluation_window_days)
        filtered = [snapshot for snapshot in self._snapshots if snapshot.timestamp >= start_cutoff]
        if len(filtered) < 3:
            return 0.0
            
        index = pd.DatetimeIndex([snapshot.timestamp for snapshot in filtered], tz="UTC")
        equity_curve = pd.Series([snapshot.equity for snapshot in filtered], index=index, dtype=float)
        metrics = PerformanceAnalyzer.compute_metrics_from_equity_curve(equity_curve)
        return float(metrics.get("annualized_sortino", 0.0))
