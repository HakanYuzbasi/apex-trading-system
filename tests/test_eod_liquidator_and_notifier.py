from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from quant_system.analytics.notifier import TelegramNotifier
from quant_system.core.bus import InMemoryEventBus
from quant_system.events.execution import ExecutionEvent
from quant_system.events.market import BarEvent
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.eod_liquidator import EODLiquidator


def _bar_at_new_york(hour: int, minute: int) -> BarEvent:
    ny = ZoneInfo("America/New_York")
    local_dt = datetime(2026, 1, 5, hour, minute, tzinfo=ny)
    ts = local_dt.astimezone(timezone.utc)
    return BarEvent(
        instrument_id="AAPL",
        exchange_ts=ts,
        received_ts=ts,
        processed_ts=ts,
        sequence_id=1,
        source="test",
        open_price=100.0,
        high_price=101.0,
        low_price=99.0,
        close_price=100.5,
        volume=1_000.0,
    )


def test_eod_liquidator_emits_flatten_signal_once_for_same_position() -> None:
    event_bus = InMemoryEventBus()
    ledger = PortfolioLedger(event_bus, starting_cash=10_000.0)
    liquidator = EODLiquidator(ledger, event_bus)
    signals = []

    event_bus.subscribe("signal", lambda event: signals.append(event))

    position = ledger.get_position("AAPL")
    position.quantity = 25.0
    position.avg_price = 100.0

    liquidation_bar = _bar_at_new_york(15, 55)
    event_bus.publish(liquidation_bar)
    event_bus.publish(liquidation_bar)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.instrument_id == "AAPL"
    assert signal.side == "flatten"
    assert signal.target_value == 0.0
    assert signal.confidence == 1.0

    liquidator.close()
    ledger.close()


def test_eod_liquidator_ignores_crypto_positions() -> None:
    event_bus = InMemoryEventBus()
    ledger = PortfolioLedger(event_bus, starting_cash=10_000.0)
    liquidator = EODLiquidator(ledger, event_bus)
    signals = []

    event_bus.subscribe("signal", lambda event: signals.append(event))

    crypto_position = ledger.get_position("CRYPTO:BTC/USD")
    crypto_position.quantity = 1.25
    crypto_position.avg_price = 60_000.0

    liquidation_bar = _bar_at_new_york(15, 55)
    event_bus.publish(liquidation_bar)

    assert signals == []

    liquidator.close()
    ledger.close()


@pytest.mark.asyncio
async def test_telegram_notifier_formats_and_sends_filled_execution_alert() -> None:
    event_bus = InMemoryEventBus()
    notifier = TelegramNotifier(event_bus)
    messages: list[str] = []

    async def capture_message(content: str) -> None:
        messages.append(content)

    notifier._post_message = capture_message  # type: ignore[method-assign]

    now = datetime.now(timezone.utc)
    await event_bus.publish_async(
        ExecutionEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            order_id="order-1",
            side="buy",
            execution_status="filled",
            fill_qty=50.0,
            fill_price=175.20,
            fees=0.05,
            slippage=0.01,
        )
    )

    assert messages == ["🚨 BOUGHT 50 AAPL @ $175.20 | Fees: $0.05"]

    await notifier.close()
