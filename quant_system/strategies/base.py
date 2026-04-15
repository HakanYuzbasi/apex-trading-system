from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Mapping
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent, SignalEvent, TradeTick, EventScalar


class BaseStrategy(ABC):
    def __init__(self, event_bus: InMemoryEventBus) -> None:
        self.event_bus = event_bus
        self.strategy_id = self.__class__.__name__
        self._subscriptions: tuple[Subscription, ...] = (
            self.event_bus.subscribe("bar", self._handle_bar),
            self.event_bus.subscribe("trade_tick", self._handle_tick),
        )

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.event_bus.unsubscribe(subscription.token)

    def _handle_bar(self, event: BarEvent) -> None:
        self.on_bar(event)

    def _handle_tick(self, event: TradeTick) -> None:
        self.on_tick(event)

    @abstractmethod
    def on_bar(self, event: BarEvent) -> None:
        pass

    @abstractmethod
    def on_tick(self, event: TradeTick) -> None:
        pass

    def emit_signal(
        self,
        instrument_id: str,
        target_type: str,
        target_value: float,
        confidence: float = 1.0,
        stop_model: str = "fixed_bps",
        stop_params: Mapping[str, EventScalar] = None,
        metadata: Mapping[str, EventScalar] = None,
    ) -> SignalEvent:
        now = self._utc_now()
        if target_value > 0:
            side = "buy"
        elif target_value < 0:
            side = "sell"
        else:
            side = "flatten"

        signal = SignalEvent(
            instrument_id=instrument_id,
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=0,
            source=f"strategy.{self.strategy_id}",
            strategy_id=self.strategy_id,
            side=side,
            target_type=target_type,
            target_value=target_value,
            confidence=confidence,
            stop_model=stop_model,
            stop_params=stop_params or {},
            metadata=metadata or {},
        )
        self._dispatch_event(signal)
        return signal

    def _dispatch_event(self, event: SignalEvent) -> None:
        subscriptions = self.event_bus.subscriptions_for(event.event_type)
        has_async_subscribers = any(subscription.is_async for subscription in subscriptions)
        if not has_async_subscribers:
            self.event_bus.publish(event)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.event_bus.publish_async(event))
            return

        loop.create_task(self.event_bus.publish_async(event))

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
