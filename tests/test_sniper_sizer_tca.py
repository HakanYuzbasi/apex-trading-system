from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quant_system.analytics.notifier import TelegramNotifier
from quant_system.analytics.tca import TransactionCostAnalyzer
from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, ExecutionEvent, OrderEvent, QuoteTick, SignalEvent
from quant_system.execution.brokers.alpaca_broker import AlpacaBroker
from quant_system.risk.sizer import VolatilitySizer


class DummyTradingClient:
    def __init__(self) -> None:
        self._api_key = "key"
        self._secret_key = "secret"
        self._sandbox = True
        self.submitted_orders: list[object] = []
        self.replaced_orders: list[tuple[str, object]] = []

    def submit_order(self, order_data: object) -> object:
        self.submitted_orders.append(order_data)
        return SimpleNamespace(id=f"venue-{len(self.submitted_orders)}")

    def replace_order_by_id(self, order_id: str, order_data: object) -> object:
        self.replaced_orders.append((order_id, order_data))
        return SimpleNamespace(id=f"{order_id}-r")


@pytest.mark.asyncio
async def test_alpaca_broker_uses_limit_chaser_for_market_order() -> None:
    bus = InMemoryEventBus()
    client = DummyTradingClient()
    broker = AlpacaBroker(client, bus, enable_limit_chaser=True)
    now = datetime.now(timezone.utc)

    await bus.publish_async(
        QuoteTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            bid=100.00,
            ask=100.10,
            bid_size=100.0,
            ask_size=100.0,
        )
    )
    await bus.publish_async(
        OrderEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="test",
            order_action="submit",
            order_scope="parent",
            side="buy",
            order_type="market",
            quantity=10.0,
            time_in_force="day",
            execution_algo="direct",
        )
    )

    assert len(client.submitted_orders) == 1
    submitted = client.submitted_orders[0]
    assert submitted.limit_price == pytest.approx(100.00)
    await broker.stop()
    broker.close()


def test_volatility_sizer_scales_down_when_pair_is_more_volatile() -> None:
    bus = InMemoryEventBus()
    sizer = VolatilitySizer(bus, pair_configs=[("AAPL", "MSFT")], lookback_window=5, target_risk_dollars=1_000.0)
    start = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    calm = [(100.0, 100.0), (100.1, 100.0), (100.0, 100.0), (100.2, 100.0), (100.1, 100.0), (100.0, 100.0)]
    for idx, (a, b) in enumerate(calm):
        ts = start + timedelta(hours=idx)
        for instrument, price in (("AAPL", a), ("MSFT", b)):
            bus.publish(
                BarEvent(
                    instrument_id=instrument,
                    exchange_ts=ts,
                    received_ts=ts,
                    processed_ts=ts,
                    sequence_id=1,
                    source="test",
                    open_price=price,
                    high_price=price,
                    low_price=price,
                    close_price=price,
                    volume=100.0,
                )
            )
    calm_size = sizer.position_size("AAPL", "MSFT")

    volatile = [(100.0, 100.0), (110.0, 100.0), (90.0, 100.0), (115.0, 100.0), (85.0, 100.0), (120.0, 100.0)]
    sizer = VolatilitySizer(bus, pair_configs=[("AAPL", "MSFT")], lookback_window=5, target_risk_dollars=1_000.0)
    for idx, (a, b) in enumerate(volatile):
        ts = start + timedelta(hours=idx)
        bus.publish(
            BarEvent(
                instrument_id="AAPL",
                exchange_ts=ts,
                received_ts=ts,
                processed_ts=ts,
                sequence_id=1,
                source="test",
                open_price=a,
                high_price=a,
                low_price=a,
                close_price=a,
                volume=100.0,
            )
        )
        bus.publish(
            BarEvent(
                instrument_id="MSFT",
                exchange_ts=ts,
                received_ts=ts,
                processed_ts=ts,
                sequence_id=1,
                source="test",
                open_price=b,
                high_price=b,
                low_price=b,
                close_price=b,
                volume=100.0,
            )
        )
    volatile_size = sizer.position_size("AAPL", "MSFT")

    assert volatile_size < calm_size


@pytest.mark.asyncio
async def test_tca_tracks_signal_to_fill_slippage_and_sends_summary() -> None:
    bus = InMemoryEventBus()
    notifier = TelegramNotifier(bus)
    messages: list[str] = []

    async def capture_message(text: str) -> None:
        messages.append(text)

    notifier.notify_text = capture_message  # type: ignore[method-assign]
    tca = TransactionCostAnalyzer(bus, notifier)
    now = datetime.now(timezone.utc)

    await bus.publish_async(
        QuoteTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            bid=100.0,
            ask=100.1,
            bid_size=100.0,
            ask_size=100.0,
        )
    )
    await bus.publish_async(
        SignalEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="test",
            strategy_id="Demo",
            side="buy",
            target_type="units",
            target_value=10.0,
            confidence=0.9,
            stop_model="none",
            stop_params={},
        )
    )
    await bus.publish_async(
        OrderEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=3,
            source="test",
            order_id="ord-1",
            strategy_id="Demo",
            order_action="submit",
            order_scope="parent",
            side="buy",
            order_type="market",
            quantity=10.0,
            time_in_force="day",
            execution_algo="direct",
        )
    )
    await bus.publish_async(
        ExecutionEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=4,
            source="test",
            order_id="ord-1",
            side="buy",
            execution_status="filled",
            fill_qty=10.0,
            fill_price=100.2,
            fees=0.0,
            slippage=0.1,
        )
    )

    metrics = tca.report_metrics(lookback_days=7)
    assert metrics["average_slippage_per_share"] == pytest.approx(0.15)

    await tca.send_summary(lookback_days=7)
    assert "TCA REPORT" in messages[0]

    tca.close()
    await notifier.close()
