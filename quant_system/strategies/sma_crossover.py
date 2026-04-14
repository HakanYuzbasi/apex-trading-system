from __future__ import annotations

from collections import deque
from typing import Deque

from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    def __init__(
        self,
        event_bus,
        *,
        instrument_id: str,
        short_window: int = 10,
        long_window: int = 30,
        long_notional: float = 1_000.0,
    ) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("SMA windows must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        super().__init__(event_bus)
        self.instrument_id = instrument_id
        self.short_window = short_window
        self.long_window = long_window
        self.long_notional = float(long_notional)
        self._closes: Deque[float] = deque(maxlen=long_window)
        self._previous_signal_state: str | None = None

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument_id:
            return

        self._closes.append(event.close_price)
        if len(self._closes) < self.long_window:
            return

        closes = list(self._closes)
        short_sma = sum(closes[-self.short_window:]) / self.short_window
        long_sma = sum(closes) / self.long_window
        current_state = "long" if short_sma > long_sma else "flat"

        if self._previous_signal_state is None:
            self._previous_signal_state = current_state
            return

        if self._previous_signal_state == "flat" and current_state == "long":
            self.emit_signal(
                instrument_id=self.instrument_id,
                target_type="notional",
                target_value=self.long_notional,
                confidence=0.75,
            )
        elif self._previous_signal_state == "long" and current_state == "flat":
            self.emit_signal(
                instrument_id=self.instrument_id,
                target_type="notional",
                target_value=0.0,
                confidence=0.75,
            )

        self._previous_signal_state = current_state

    def on_tick(self, event: TradeTick) -> None:
        return
