from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent

MarketState = Literal["low_vol", "high_vol", "breakout"]


@dataclass(slots=True)
class _RegimeState:
    last_price: float | None = None
    ewma_variance: float = 0.0
    recent_abs_returns: deque[float] | None = None
    regime: MarketState = "low_vol"


class RegimeDetector:
    """
    O(1) volatility regime detector for live trading veto logic.

    Uses EWMA variance and a short breakout window of absolute returns. This is
    deliberately lighter than a full GARCH fit and is suitable for 24/7 event
    processing.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        ewma_lambda: float = 0.94,
        high_vol_threshold: float = 0.02,
        breakout_return_threshold: float = 0.04,
        breakout_window: int = 12,
    ) -> None:
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1)")
        if high_vol_threshold <= 0 or breakout_return_threshold <= 0:
            raise ValueError("volatility thresholds must be positive")
        if breakout_window <= 0:
            raise ValueError("breakout_window must be positive")

        self._ewma_lambda = float(ewma_lambda)
        self._high_vol_threshold = float(high_vol_threshold)
        self._breakout_return_threshold = float(breakout_return_threshold)
        self._breakout_window = int(breakout_window)
        self._event_bus = event_bus
        self._states: dict[str, _RegimeState] = {}
        self._subscription: Subscription = event_bus.subscribe("bar", self._on_bar)

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    def close(self) -> None:
        self._event_bus.unsubscribe(self._subscription.token)

    def regime_for(self, instrument_id: str) -> MarketState:
        return self._states.get(instrument_id, _RegimeState()).regime

    def is_extreme_volatility(self, instrument_id: str | None = None) -> bool:
        if instrument_id is not None:
            return self.regime_for(instrument_id) in {"high_vol", "breakout"}
        return any(state.regime in {"high_vol", "breakout"} for state in self._states.values())

    def _on_bar(self, event: BarEvent) -> None:
        state = self._states.setdefault(
            event.instrument_id,
            _RegimeState(recent_abs_returns=deque(maxlen=self._breakout_window)),
        )
        if state.last_price is None or state.last_price <= 0:
            state.last_price = event.close_price
            return

        simple_return = (event.close_price / state.last_price) - 1.0
        squared_return = simple_return * simple_return
        state.ewma_variance = (
            self._ewma_lambda * state.ewma_variance
            + (1.0 - self._ewma_lambda) * squared_return
        )
        abs_return = abs(simple_return)
        state.recent_abs_returns.append(abs_return)
        realized_vol = math.sqrt(max(0.0, state.ewma_variance))
        recent_breakout = max(state.recent_abs_returns, default=0.0)

        if recent_breakout >= self._breakout_return_threshold:
            state.regime = "breakout"
        elif realized_vol >= self._high_vol_threshold:
            state.regime = "high_vol"
        else:
            state.regime = "low_vol"

        state.last_price = event.close_price
