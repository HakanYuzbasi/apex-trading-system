from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from quant_system.analytics.notifier import TelegramNotifier
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import ExecutionEvent, OrderEvent, QuoteTick, SignalEvent, TradeTick


@dataclass(frozen=True, slots=True)
class TCAFillRecord:
    order_id: str
    instrument_id: str
    benchmark_price: float
    fill_price: float
    fill_qty: float
    side: str
    timestamp: datetime

    @property
    def per_share_slippage(self) -> float:
        if self.side == "buy":
            return self.fill_price - self.benchmark_price
        return self.benchmark_price - self.fill_price

    @property
    def total_slippage_dollars(self) -> float:
        return self.per_share_slippage * self.fill_qty


class TransactionCostAnalyzer:
    """
    Tracks signal benchmark prices against realized execution fills.
    """

    def __init__(self, event_bus: InMemoryEventBus, notifier: TelegramNotifier) -> None:
        self._event_bus = event_bus
        self._notifier = notifier
        self._latest_price_by_instrument: dict[str, float] = {}
        self._signal_benchmark_by_key: dict[tuple[str, str], float] = {}
        self._order_benchmark_by_order_id: dict[str, float] = {}
        self._fills: list[TCAFillRecord] = []
        self._subscriptions: tuple[Subscription, ...] = (
            event_bus.subscribe("quote_tick", self._on_quote),
            event_bus.subscribe("trade_tick", self._on_trade),
            event_bus.subscribe("signal", self._on_signal),
            event_bus.subscribe("order", self._on_order),
            event_bus.subscribe("execution", self._on_execution),
        )

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    def close(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription.token)

    async def send_summary(self, *, lookback_days: int = 7) -> None:
        metrics = self.report_metrics(lookback_days=lookback_days)
        await self._notifier.notify_text(
            "📊 TCA REPORT: Average Slippage is "
            f"${metrics['average_slippage_per_share']:.4f}/share. "
            f"Sniper saved us ${metrics['sniper_saved_dollars']:.2f} today."
        )

    def _on_quote(self, event: QuoteTick) -> None:
        self._latest_price_by_instrument[event.instrument_id] = (event.bid + event.ask) / 2.0

    def _on_trade(self, event: TradeTick) -> None:
        self._latest_price_by_instrument[event.instrument_id] = event.last_price

    def _on_signal(self, event: SignalEvent) -> None:
        benchmark = self._latest_price_by_instrument.get(event.instrument_id)
        if benchmark is None:
            return
        self._signal_benchmark_by_key[(event.strategy_id, event.instrument_id)] = benchmark

    def _on_order(self, event: OrderEvent) -> None:
        benchmark = self._signal_benchmark_by_key.get((event.strategy_id or "", event.instrument_id))
        if benchmark is None:
            benchmark = self._latest_price_by_instrument.get(event.instrument_id)
        if benchmark is not None:
            self._order_benchmark_by_order_id[event.order_id] = benchmark

    def _on_execution(self, event: ExecutionEvent) -> None:
        if event.execution_status not in {"partial_fill", "filled"}:
            return
        benchmark = self._order_benchmark_by_order_id.get(event.order_id)
        if benchmark is None:
            return
        self._fills.append(
            TCAFillRecord(
                order_id=event.order_id,
                instrument_id=event.instrument_id,
                benchmark_price=benchmark,
                fill_price=event.fill_price,
                fill_qty=event.fill_qty,
                side=event.side,
                timestamp=event.exchange_ts,
            )
        )

    def report_metrics(self, *, lookback_days: int = 7) -> dict[str, float]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        fills = [fill for fill in self._fills if fill.timestamp >= cutoff]
        if not fills:
            return {
                "average_slippage_per_share": 0.0,
                "total_slippage_dollars": 0.0,
                "sniper_saved_dollars": 0.0,
                "fill_count": 0.0,
            }
        total_shares = sum(fill.fill_qty for fill in fills)
        total_slippage = sum(fill.total_slippage_dollars for fill in fills)
        avg_per_share = total_slippage / total_shares if total_shares > 0 else 0.0
        sniper_saved = max(0.0, -total_slippage)
        return {
            "average_slippage_per_share": avg_per_share,
            "total_slippage_dollars": total_slippage,
            "sniper_saved_dollars": sniper_saved,
            "fill_count": float(len(fills)),
        }
