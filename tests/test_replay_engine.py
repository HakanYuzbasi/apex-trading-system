from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_system.core.bus import InMemoryEventBus
from quant_system.core.clock import SimulatedClock
from quant_system.data.replay.engine import ReplayEngine
from quant_system.data.replay.source import HistoricalReplaySource
from quant_system.events import BarEvent, TradeTick


def test_historical_replay_source_yields_events_in_strict_chronological_order() -> None:
    datasets = {
        "AAPL": pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.5, 100.5],
                "Close": [100.5, 101.5],
                "Volume": [1_000_000, 1_100_000],
            },
            index=pd.to_datetime(["2026-01-01T14:30:00Z", "2026-01-01T14:31:00Z"], utc=True),
        ),
        "MSFT": pd.DataFrame(
            {
                "Open": [200.0],
                "High": [201.0],
                "Low": [199.0],
                "Close": [200.5],
                "Volume": [900_000],
            },
            index=pd.to_datetime(["2026-01-01T14:30:30Z"], utc=True),
        ),
    }

    source = HistoricalReplaySource(datasets)
    events = list(source.iter_events())

    assert [event.instrument_id for event in events] == ["AAPL", "MSFT", "AAPL"]
    assert all(isinstance(event, BarEvent) for event in events)


def test_historical_replay_source_falls_back_to_trade_tick_for_print_data() -> None:
    datasets = {
        "CRYPTO:BTC/USD": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
                "price": [95_000.0],
                "size": [0.25],
            }
        )
    }

    source = HistoricalReplaySource(datasets)
    event = next(iter(source))

    assert isinstance(event, TradeTick)
    assert event.last_price == 95_000.0
    assert event.last_size == 0.25


def test_replay_engine_advances_clock_before_sync_publish() -> None:
    first_ts = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    second_ts = datetime(2026, 1, 1, 14, 31, tzinfo=timezone.utc)
    datasets = {
        "AAPL": pd.DataFrame(
            {
                "timestamp": [first_ts, second_ts],
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [1_000_000, 1_000_000],
            }
        )
    }

    clock = SimulatedClock(current_time=datetime(2026, 1, 1, 14, 29, tzinfo=timezone.utc))
    source = HistoricalReplaySource(datasets)
    bus = InMemoryEventBus()
    observed: list[tuple[datetime, datetime]] = []

    def handler(event) -> None:
        observed.append((clock.current_time, event.exchange_ts))

    bus.subscribe("bar", handler)
    engine = ReplayEngine(clock, source, bus)

    processed = engine.run()

    assert processed == 2
    assert observed == [(first_ts, first_ts), (second_ts, second_ts)]
    assert clock.current_time == second_ts


def test_historical_replay_source_can_stream_from_database_client() -> None:
    class FakeClient:
        def stream_event_rows(self, **kwargs):
            assert kwargs["instrument_ids"] == ("AAPL", "MSFT")
            return iter(
                [
                    {
                        "event_kind": "bar",
                        "exchange_ts": datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc),
                        "instrument_id": "AAPL",
                        "event_id": "evt-1",
                        "bar_id": "bar-1",
                        "source": "db",
                        "payload_version": "1.0",
                        "metadata": {"origin": "db"},
                        "open_price": 100.0,
                        "high_price": 101.0,
                        "low_price": 99.0,
                        "close_price": 100.5,
                        "volume": 1_000.0,
                    },
                    {
                        "event_kind": "trade_tick",
                        "exchange_ts": datetime(2026, 1, 1, 14, 30, 1, tzinfo=timezone.utc),
                        "instrument_id": "MSFT",
                        "event_id": "evt-2",
                        "trade_id": "trade-1",
                        "source": "db",
                        "payload_version": "1.0",
                        "metadata": {"origin": "db"},
                        "last_price": 200.0,
                        "last_size": 50.0,
                        "aggressor_side": "buy",
                    },
                ]
            )

    source = HistoricalReplaySource(
        client=FakeClient(),
        start_ts=datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc),
        end_ts=datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
        instrument_ids=("AAPL", "MSFT"),
    )

    events = list(source)

    assert isinstance(events[0], BarEvent)
    assert isinstance(events[1], TradeTick)
    assert events[0].instrument_id == "AAPL"
    assert events[1].instrument_id == "MSFT"
