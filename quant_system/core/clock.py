from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class SimulatedClock:
    current_time: datetime

    def __post_init__(self) -> None:
        if self.current_time.tzinfo is None or self.current_time.utcoffset() is None:
            raise ValueError("current_time must be timezone-aware")

    def advance_to(self, new_time: datetime) -> datetime:
        if new_time.tzinfo is None or new_time.utcoffset() is None:
            raise ValueError("new_time must be timezone-aware")
        if new_time < self.current_time:
            raise ValueError("simulated clock cannot move backwards")
        self.current_time = new_time
        return self.current_time

    def advance(self, new_time: datetime) -> datetime:
        return self.advance_to(new_time)
