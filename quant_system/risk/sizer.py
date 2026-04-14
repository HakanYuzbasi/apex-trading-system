from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import sqrt

import numpy as np

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent


@dataclass(slots=True)
class _PairVolState:
    recent_spread_returns: deque[float]
    latest_close_a: float | None = None
    latest_close_b: float | None = None
    latest_ts_a: datetime | None = None
    latest_ts_b: datetime | None = None
    last_pair_ts: datetime | None = None
    last_pair_spread: float | None = None


class VolatilitySizer:
    """
    Inverse-volatility position sizer for synchronized pairs.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        pair_configs: list[tuple[str, str]],
        lookback_window: int = 50,
        target_risk_dollars: float = 1_000.0,
    ) -> None:
        if lookback_window <= 1:
            raise ValueError("lookback_window must be greater than 1")
        if target_risk_dollars <= 0:
            raise ValueError("target_risk_dollars must be positive")
        self.lookback_window = int(lookback_window)
        self.target_risk_dollars = float(target_risk_dollars)
        self._pair_by_instrument: dict[str, list[str]] = {}
        self._states: dict[str, _PairVolState] = {}
        for instrument_a, instrument_b in pair_configs:
            pair_label = f"{instrument_a}/{instrument_b}"
            self._states[pair_label] = _PairVolState(recent_spread_returns=deque(maxlen=self.lookback_window))
            self._pair_by_instrument.setdefault(instrument_a, []).append(pair_label)
            self._pair_by_instrument.setdefault(instrument_b, []).append(pair_label)
        self._subscription: Subscription = event_bus.subscribe("bar", self._on_bar)

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    def close(self, event_bus: InMemoryEventBus) -> None:
        event_bus.unsubscribe(self._subscription.token)

    def position_size(self, instrument_a: str, instrument_b: str, *, target_risk_dollars: float | None = None) -> float:
        pair_label = f"{instrument_a}/{instrument_b}"
        state = self._states.get(pair_label)
        if state is None or len(state.recent_spread_returns) < 2:
            return target_risk_dollars or self.target_risk_dollars
        sigma_pair = float(np.std(np.array(state.recent_spread_returns, dtype=float), ddof=0))
        if sigma_pair <= 1e-12:
            return target_risk_dollars or self.target_risk_dollars
        risk_budget = float(target_risk_dollars or self.target_risk_dollars)
        return risk_budget / (sigma_pair * sqrt(252.0))

    def _on_bar(self, event: BarEvent) -> None:
        for pair_label in self._pair_by_instrument.get(event.instrument_id, []):
            instrument_a, instrument_b = pair_label.split("/", 1)
            state = self._states[pair_label]
            if event.instrument_id == instrument_a:
                state.latest_close_a = event.close_price
                state.latest_ts_a = event.exchange_ts
            else:
                state.latest_close_b = event.close_price
                state.latest_ts_b = event.exchange_ts

            if state.latest_ts_a is None or state.latest_ts_b is None:
                continue
            if state.latest_ts_a != state.latest_ts_b or state.latest_ts_a == state.last_pair_ts:
                continue

            previous_spread = state.last_pair_spread
            current_spread = self._spread(state)
            state.last_pair_ts = state.latest_ts_a
            state.last_pair_spread = current_spread
            if previous_spread is None or current_spread is None or previous_spread <= 0 or current_spread <= 0:
                continue
            spread_return = (current_spread / previous_spread) - 1.0
            state.recent_spread_returns.append(spread_return)

    @staticmethod
    def _spread(state: _PairVolState) -> float | None:
        if state.latest_close_a is None or state.latest_close_b is None or state.latest_close_b <= 0:
            return None
        return state.latest_close_a / state.latest_close_b
