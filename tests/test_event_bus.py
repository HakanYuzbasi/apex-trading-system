from __future__ import annotations

import asyncio

import pytest

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import ExecutionEvent, OrderEvent, SignalEvent, utc_now


def _signal_event(sequence_id: int = 1) -> SignalEvent:
    now = utc_now()
    return SignalEvent(
        instrument_id="BTC-USD",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=sequence_id,
        source="strategy.test",
        strategy_id="trend_01",
        side="buy",
        target_type="weight",
        target_value=0.10,
        confidence=0.91,
        stop_model="atr_trailing",
        stop_params={"atr_multiple": 2.0},
    )


def _order_event(sequence_id: int = 2) -> OrderEvent:
    now = utc_now()
    return OrderEvent(
        instrument_id="BTC-USD",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=sequence_id,
        source="portfolio.test",
        order_action="submit",
        order_scope="parent",
        side="buy",
        order_type="limit",
        quantity=1.0,
        time_in_force="day",
        execution_algo="twap",
        limit_price=50_000.0,
    )


def _execution_event(order_id: str, sequence_id: int = 3) -> ExecutionEvent:
    now = utc_now()
    return ExecutionEvent(
        instrument_id="BTC-USD",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=sequence_id,
        source="venue.test",
        order_id=order_id,
        side="buy",
        execution_status="filled",
        fill_qty=1.0,
        fill_price=49_995.0,
        fees=1.5,
        slippage=-5.0,
        remaining_qty=0.0,
    )


def test_publish_is_deterministic_for_sync_handlers() -> None:
    bus = InMemoryEventBus()
    calls: list[str] = []

    def first(event: SignalEvent) -> None:
        calls.append(f"first:{event.event_type}")

    def second(event: SignalEvent) -> None:
        calls.append(f"second:{event.event_type}")

    bus.subscribe("signal", first)
    bus.subscribe("signal", second)

    result = bus.publish(_signal_event())

    assert calls == ["first:signal", "second:signal"]
    assert result.subscriber_count == 2
    assert result.sync_handler_count == 2
    assert result.async_handler_count == 0
    assert len(bus.published_events) == 1


def test_wildcard_subscription_receives_all_matching_events() -> None:
    bus = InMemoryEventBus()
    seen: list[str] = []

    def capture(event) -> None:
        seen.append(event.event_type)

    bus.subscribe("*", capture)
    order_event = _order_event()
    execution_event = _execution_event(order_event.order_id)

    bus.publish(_signal_event())
    bus.publish(order_event)
    bus.publish(execution_event)

    assert seen == ["signal", "order", "execution"]


def test_publish_rejects_async_handlers_in_sync_mode() -> None:
    bus = InMemoryEventBus()

    async def async_handler(event) -> None:
        await asyncio.sleep(0)

    bus.subscribe("signal", async_handler)

    with pytest.raises(RuntimeError, match="publish_async"):
        bus.publish(_signal_event())


@pytest.mark.asyncio
async def test_publish_async_supports_sync_and_async_subscribers() -> None:
    bus = InMemoryEventBus()
    started: list[str] = []
    completed: list[str] = []

    def sync_handler(event) -> None:
        started.append(f"sync:{event.event_type}")
        completed.append("sync")

    async def async_handler(event) -> None:
        started.append(f"async:{event.event_type}")
        await asyncio.sleep(0.01)
        completed.append("async")

    bus.subscribe("signal", sync_handler)
    bus.subscribe("signal", async_handler)

    result = await bus.publish_async(_signal_event())

    assert started == ["sync:signal", "async:signal"]
    assert completed == ["sync", "async"]
    assert result.subscriber_count == 2
    assert result.sync_handler_count == 1
    assert result.async_handler_count == 1


def test_unsubscribe_removes_handler() -> None:
    bus = InMemoryEventBus()
    seen: list[str] = []

    def handler(event) -> None:
        seen.append(event.event_type)

    subscription = bus.subscribe("signal", handler)
    assert bus.unsubscribe(subscription.token) is True

    bus.publish(_signal_event())

    assert seen == []
