from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from core.symbols import AssetClass, parse_symbol
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events.execution import ExecutionEvent
from quant_system.events.market import BarEvent, QuoteTick, TradeTick
from quant_system.events.order import OrderEvent


class PaperBroker:
    """
    Event-driven paper broker for live simulation and integration testing.

    It listens to market data and order flow on the shared event bus, maintains a
    last-price cache, and emits filled execution events back into the bus.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        simulated_latency_ms: float = 50.0,
        fee_per_unit: float = 0.005,
        synchronous: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._latency_seconds = max(0.0, simulated_latency_ms / 1000.0)
        self._fee_per_unit = max(0.0, fee_per_unit)
        self._synchronous = synchronous
        self._latest_market: dict[str, QuoteTick | TradeTick] = {}
        order_handler = self._on_order_event_sync if synchronous else self._on_order_event
        self._subscriptions: tuple[Subscription, ...] = (
            self._event_bus.subscribe("bar", self._on_bar_event),
            self._event_bus.subscribe("quote_tick", self._on_quote_tick),
            self._event_bus.subscribe("trade_tick", self._on_trade_tick),
            self._event_bus.subscribe("order", order_handler),
        )

    @property
    def latest_market(self) -> dict[str, QuoteTick | TradeTick]:
        return dict(self._latest_market)

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    def close(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription.token)

    def _on_bar_event(self, event: BarEvent) -> None:
        synthetic_tick = TradeTick(
            instrument_id=event.instrument_id,
            exchange_ts=event.exchange_ts,
            received_ts=event.received_ts,
            processed_ts=event.processed_ts,
            sequence_id=event.sequence_id,
            source=event.source,
            last_price=event.close_price,
            last_size=max(event.volume, 1.0),
            aggressor_side="unknown",
            trade_id=event.bar_id,
            metadata=event.metadata,
        )
        self._latest_market[event.instrument_id] = synthetic_tick

    def _on_quote_tick(self, event: QuoteTick) -> None:
        self._latest_market[event.instrument_id] = event

    def _on_trade_tick(self, event: TradeTick) -> None:
        self._latest_market[event.instrument_id] = event

    def _on_order_event_sync(self, event: OrderEvent) -> None:
        if event.order_action != "submit":
            return
        if self._latency_seconds > 0:
            time.sleep(self._latency_seconds)
        execution_event = self._build_execution_event(event)
        if execution_event is None:
            return
        self._event_bus.publish(execution_event)

    async def _on_order_event(self, event: OrderEvent) -> None:
        if event.order_action != "submit":
            return

        await asyncio.sleep(self._latency_seconds)
        execution_event = self._build_execution_event(event)
        if execution_event is None:
            return
        await self._event_bus.publish_async(execution_event)

    def _build_execution_event(self, event: OrderEvent) -> ExecutionEvent | None:
        market_event = self._latest_market.get(event.instrument_id)
        if market_event is None:
            return None

        # The ledger owns long/short inventory semantics. The paper broker only
        # needs to emit the correct side and signed economics for the fill.
        fill_price = self._simulate_fill_price(event, market_event)
        fill_qty = event.quantity
        fees = abs(fill_qty) * self._fee_per_unit
        now = datetime.now(timezone.utc)
        return ExecutionEvent(
            instrument_id=event.instrument_id,
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=event.sequence_id,
            source="paper_broker",
            order_id=event.order_id,
            parent_order_id=event.parent_order_id if event.order_scope == "child" else event.order_id,
            broker="paper",
            venue=event.venue or "paper",
            side=event.side,
            execution_status="filled",
            fill_qty=fill_qty,
            fill_price=fill_price,
            fees=fees,
            slippage=self._compute_slippage(event, market_event, fill_price),
            remaining_qty=0.0,
        )

    def _simulate_fill_price(self, order: OrderEvent, market_event: QuoteTick | TradeTick) -> float:
        tick_size = self._tick_size(order.instrument_id)
        if isinstance(market_event, QuoteTick):
            base_price = market_event.ask if order.side == "buy" else market_event.bid
        else:
            base_price = market_event.last_price

        if order.side == "buy":
            return base_price + tick_size
        return max(tick_size, base_price - tick_size)

    def _compute_slippage(
        self,
        order: OrderEvent,
        market_event: QuoteTick | TradeTick,
        fill_price: float,
    ) -> float:
        if isinstance(market_event, QuoteTick):
            reference_price = market_event.ask if order.side == "buy" else market_event.bid
        else:
            reference_price = market_event.last_price

        if order.side == "buy":
            return fill_price - reference_price
        return reference_price - fill_price

    @staticmethod
    def _tick_size(instrument_id: str) -> float:
        try:
            parsed = parse_symbol(instrument_id)
        except Exception:
            return 0.01

        if parsed.asset_class == AssetClass.FOREX:
            return 0.0001
        if parsed.asset_class == AssetClass.CRYPTO:
            return 0.01
        return 0.01
