from __future__ import annotations

import logging
from collections import deque
import numpy as np

from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutPodStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy based on Opening Range Breakout (ORB).
    Enters a trade when price breaks out of the high/low of the first N bars of the session.
    """
    def __init__(
        self,
        event_bus,
        *,
        instrument_id: str,
        window: int = 30, # Number of bars for the opening range
        leg_notional: float = 1_000.0,
    ) -> None:
        super().__init__(event_bus)
        self.instrument_id = instrument_id
        self.window = window
        self.leg_notional = float(leg_notional)
        
        self.state = "flat"
        self.current_date = None
        self.bar_count = 0
        self.opening_high = -np.inf
        self.opening_low = np.inf
        self.midpoint = 0.0
        
    def on_tick(self, event: TradeTick) -> None:
        pass

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument_id:
            return
            
        event_date = event.exchange_ts.date()
        if self.current_date != event_date:
            # Reset daily parameters
            self.current_date = event_date
            self.bar_count = 0
            self.opening_high = -np.inf
            self.opening_low = np.inf
            self.state = "flat"
            self._emit_state_transition("flat")
            
        self.bar_count += 1
        
        if self.bar_count <= self.window:
            # Build opening range
            self.opening_high = max(self.opening_high, event.high_price)
            self.opening_low = min(self.opening_low, event.low_price)
            self.midpoint = (self.opening_high + self.opening_low) / 2.0
            return
            
        # Post-ORB phase
        desired_state = self.state
        
        if self.state == "flat":
            if event.close_price > self.opening_high:
                desired_state = "long"
            elif event.close_price < self.opening_low:
                desired_state = "short"
        elif self.state == "long":
            # Exit if we retrace back below the midpoint of the ORB (whipsaw filter)
            if event.close_price < self.midpoint:
                desired_state = "flat"
        elif self.state == "short":
            # Exit if we retrace back above the midpoint of the ORB
            if event.close_price > self.midpoint:
                desired_state = "flat"
                
        if desired_state != self.state:
            self._emit_state_transition(desired_state)
            self.state = desired_state

    def _emit_state_transition(self, desired_state: str) -> None:
        if desired_state == "long":
            target_value = self.leg_notional
        elif desired_state == "short":
            target_value = -self.leg_notional
        else:
            target_value = 0.0
            
        self.emit_signal(
            instrument_id=self.instrument_id,
            target_type="notional",
            target_value=target_value,
            confidence=0.85,
            metadata={"strategy_type": "breakout"}
        )
