from __future__ import annotations

from typing import Iterable

from quant_system.core.bus import InMemoryEventBus
from quant_system.core.clock import SimulatedClock
from quant_system.events.base import BaseEvent


class ReplayEngine:
    """
    Deterministic historical event loop.

    Order of operations per event:
    1. Pull event from the source.
    2. Advance the simulated clock to the event timestamp.
    3. Publish synchronously to the event bus.
    """

    def __init__(
        self,
        clock: SimulatedClock,
        source: Iterable[BaseEvent],
        event_bus: InMemoryEventBus,
    ) -> None:
        self._clock = clock
        self._source = source
        self._event_bus = event_bus

    @property
    def clock(self) -> SimulatedClock:
        return self._clock

    def run(self, *, max_events: int | None = None) -> int:
        processed_events = 0
        for event in self._source:
            if max_events is not None and processed_events >= max_events:
                break
            self._clock.advance_to(event.exchange_ts)
            self._event_bus.publish(event)
            processed_events += 1
        return processed_events
