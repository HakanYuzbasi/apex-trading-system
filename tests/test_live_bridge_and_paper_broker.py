from __future__ import annotations

import pytest

from quant_system.core.bus import InMemoryEventBus
from quant_system.data.normalizers.live_bridge import LiveDataBridge
from quant_system.events.execution import ExecutionEvent
from quant_system.events.market import QuoteTick, TradeTick
from quant_system.events.order import OrderEvent
from quant_system.events import utc_now
from quant_system.execution.brokers.paper import PaperBroker


@pytest.mark.asyncio
async def test_live_bridge_normalizes_and_publishes_alpaca_trade() -> None:
    bus = InMemoryEventBus()
    bridge = LiveDataBridge(bus)
    captured: list[TradeTick] = []

    def on_trade(event) -> None:
        captured.append(event)

    bus.subscribe("trade_tick", on_trade)

    events = await bridge.publish_alpaca(
        {
            "T": "t",
            "S": "BTC/USD",
            "p": 65000.5,
            "s": 0.25,
            "t": "2026-04-04T12:00:00Z",
            "i": "alp-trade-1",
        }
    )

    assert len(events) == 1
    assert isinstance(events[0], TradeTick)
    assert events[0].instrument_id == "CRYPTO:BTC/USD"
    assert captured[0].trade_id == "alp-trade-1"


@pytest.mark.asyncio
async def test_live_bridge_normalizes_ibkr_quote_and_trade_from_single_payload() -> None:
    bus = InMemoryEventBus()
    bridge = LiveDataBridge(bus)

    events = await bridge.publish_ibkr(
        {
            "symbol": "AAPL",
            "bid": 199.95,
            "ask": 200.05,
            "bidSize": 100,
            "askSize": 200,
            "last": 200.0,
            "lastSize": 50,
            "time": "2026-04-04T12:00:00+00:00",
        }
    )

    assert len(events) == 2
    assert isinstance(events[0], QuoteTick)
    assert isinstance(events[1], TradeTick)
    assert events[0].instrument_id == "AAPL"
    assert events[1].instrument_id == "AAPL"


@pytest.mark.asyncio
async def test_paper_broker_fills_order_from_latest_quote() -> None:
    bus = InMemoryEventBus()
    bridge = LiveDataBridge(bus)
    broker = PaperBroker(bus, simulated_latency_ms=0.0)
    captured: list[ExecutionEvent] = []

    def on_execution(event) -> None:
        captured.append(event)

    bus.subscribe("execution", on_execution)

    await bridge.publish_ibkr(
        {
            "symbol": "AAPL",
            "bid": 199.95,
            "ask": 200.05,
            "bidSize": 100,
            "askSize": 200,
            "time": "2026-04-04T12:00:00+00:00",
        }
    )

    now = utc_now()
    order = OrderEvent(
        instrument_id="AAPL",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=10,
        source="portfolio.test",
        order_action="submit",
        order_scope="parent",
        side="buy",
        order_type="market",
        quantity=100.0,
        time_in_force="day",
        execution_algo="direct",
    )

    await bus.publish_async(order)

    assert len(captured) == 1
    assert captured[0].execution_status == "filled"
    assert captured[0].fill_price == pytest.approx(200.06)
    assert captured[0].fees == pytest.approx(0.5)
    assert captured[0].slippage == pytest.approx(0.01)

    broker.close()


@pytest.mark.asyncio
async def test_paper_broker_supports_short_side_execution() -> None:
    bus = InMemoryEventBus()
    bridge = LiveDataBridge(bus)
    broker = PaperBroker(bus, simulated_latency_ms=0.0)
    captured: list[ExecutionEvent] = []

    bus.subscribe("execution", lambda event: captured.append(event))

    await bridge.publish_ibkr(
        {
            "symbol": "AAPL",
            "bid": 199.95,
            "ask": 200.05,
            "bidSize": 100,
            "askSize": 200,
            "time": "2026-04-04T12:00:00+00:00",
        }
    )

    now = utc_now()
    order = OrderEvent(
        instrument_id="AAPL",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=11,
        source="portfolio.test",
        order_action="submit",
        order_scope="parent",
        side="sell",
        order_type="market",
        quantity=25.0,
        time_in_force="day",
        execution_algo="direct",
    )

    await bus.publish_async(order)

    assert len(captured) == 1
    assert captured[0].side == "sell"
    assert captured[0].execution_status == "filled"
    assert captured[0].fill_price == pytest.approx(199.94)
    assert captured[0].fees == pytest.approx(0.125)
    assert captured[0].slippage == pytest.approx(0.01)

    broker.close()
