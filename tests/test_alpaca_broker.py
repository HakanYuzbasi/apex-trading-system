from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from quant_system.core.bus import InMemoryEventBus
from quant_system.events.market import QuoteTick
from quant_system.events.order import OrderEvent
from quant_system.execution.brokers.alpaca_broker import AlpacaBroker


class DummyTradingClient:
    def __init__(self) -> None:
        self._api_key = "key"
        self._secret_key = "secret"
        self._sandbox = True
        self.submitted_orders: list[object] = []

    def submit_order(self, order_data: object) -> object:
        self.submitted_orders.append(order_data)
        return SimpleNamespace(id="alpaca-order-1")


def _build_order_event() -> OrderEvent:
    now = datetime.now(timezone.utc)
    return OrderEvent(
        instrument_id="AAPL",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=1,
        source="test",
        order_action="submit",
        order_scope="parent",
        side="buy",
        order_type="market",
        quantity=5.0,
        time_in_force="day",
        execution_algo="direct",
    )


@pytest.mark.asyncio
async def test_alpaca_broker_submits_market_orders_with_client_order_id() -> None:
    event_bus = InMemoryEventBus()
    client = DummyTradingClient()
    broker = AlpacaBroker(client, event_bus)

    order_event = _build_order_event()
    await event_bus.publish_async(order_event)

    assert len(client.submitted_orders) == 1
    submitted_order = client.submitted_orders[0]
    assert submitted_order.symbol == "AAPL"
    assert float(submitted_order.qty) == pytest.approx(5.0)
    assert submitted_order.client_order_id == order_event.order_id

    broker.close()


@pytest.mark.asyncio
async def test_alpaca_broker_translates_trade_updates_into_execution_events() -> None:
    event_bus = InMemoryEventBus()
    client = DummyTradingClient()
    broker = AlpacaBroker(client, event_bus)
    executions = []

    def on_execution(event) -> None:
        executions.append(event)

    event_bus.subscribe("execution", on_execution)

    now = datetime.now(timezone.utc)
    await event_bus.publish_async(
        QuoteTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=0,
            source="test.quote",
            bid=99.99,
            ask=100.00,
            bid_size=100.0,
            ask_size=100.0,
        )
    )
    order_event = _build_order_event()
    await event_bus.publish_async(order_event)

    await broker._on_trade_update(
        {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "execution_id": "execution-1",
                "timestamp": now.isoformat(),
                "price": 100.05,
                "qty": 5.0,
                "order": {
                    "id": "alpaca-order-1",
                    "client_order_id": order_event.order_id,
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": "5",
                    "filled_qty": "5",
                },
            },
        }
    )

    assert len(executions) == 1
    execution_event = executions[0]
    assert execution_event.order_id == order_event.order_id
    assert execution_event.venue_order_id == "alpaca-order-1"
    assert execution_event.execution_status == "filled"
    assert execution_event.fill_qty == pytest.approx(5.0)
    assert execution_event.fill_price == pytest.approx(100.05)
    assert execution_event.slippage == pytest.approx(0.05)

    broker.close()
