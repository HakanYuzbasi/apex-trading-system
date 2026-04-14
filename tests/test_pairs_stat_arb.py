from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, SignalEvent
from quant_system.strategies.pairs_stat_arb import PairsStatArbStrategy


def _bar(instrument_id: str, when: datetime, close_price: float) -> BarEvent:
    return BarEvent(
        instrument_id=instrument_id,
        exchange_ts=when,
        received_ts=when,
        processed_ts=when,
        sequence_id=1,
        source="test",
        open_price=close_price,
        high_price=close_price,
        low_price=close_price,
        close_price=close_price,
        volume=100.0,
    )


def test_pairs_strategy_waits_for_lookback_then_emits_entry_and_exit_signals() -> None:
    event_bus = InMemoryEventBus()
    strategy = PairsStatArbStrategy(
        event_bus,
        instrument_a="AAPL",
        instrument_b="MSFT",
        lookback_window=3,
        entry_z_score=1.0,
        exit_z_score=0.9,
        leg_notional=1_000.0,
    )
    signals: list[SignalEvent] = []
    event_bus.subscribe("signal", lambda event: signals.append(event))

    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    paired_prices = [
        (100.0, 100.0),
        (101.0, 100.0),
        (102.0, 100.0),
        (120.0, 100.0),
        (100.0, 100.0),
    ]

    for index, (price_a, price_b) in enumerate(paired_prices):
        ts = start + timedelta(hours=index)
        event_bus.publish(_bar("AAPL", ts, price_a))
        event_bus.publish(_bar("MSFT", ts, price_b))

    assert len(signals) == 4
    assert signals[0].instrument_id == "AAPL"
    assert signals[0].target_value == -1_000.0
    assert signals[1].instrument_id == "MSFT"
    assert signals[1].target_value == 1_000.0
    assert signals[2].instrument_id == "AAPL"
    assert signals[2].target_value == 0.0
    assert signals[3].instrument_id == "MSFT"
    assert signals[3].target_value == 0.0

    strategy.close()
