from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Literal

from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy
from quant_system.risk.bayesian_vol import BayesianVolatilityAdjuster

PairState = Literal["flat", "long_a_short_b", "short_a_long_b"]


@dataclass(slots=True)
class _RollingSeries:
    values: Deque[float]
    sum_values: float = 0.0
    sum_squares: float = 0.0

    @classmethod
    def with_window(cls, window: int) -> _RollingSeries:
        return cls(values=deque(maxlen=window))

    def append(self, value: float) -> None:
        if len(self.values) == self.values.maxlen:
            oldest = self.values[0]
            self.sum_values -= oldest
            self.sum_squares -= oldest * oldest
        self.values.append(value)
        self.sum_values += value
        self.sum_squares += value * value

    @property
    def count(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_values / self.count

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        variance = max(0.0, (self.sum_squares / self.count) - (mean * mean))
        return math.sqrt(variance)


class PairsStatArbStrategy(BaseStrategy):
    """
    Market-neutral pairs strategy using a rolling z-score of the price ratio.

    The spread is defined as `price_a / price_b`, which is scale-invariant and
    more stable than a raw price difference for same-sector equities such as
    AAPL/MSFT. Signals are emitted only after both assets have produced a bar
    for the same timestamp and the rolling lookback window is fully populated.
    """

    def __init__(
        self,
        event_bus,
        *,
        instrument_a: str,
        instrument_b: str,
        lookback_window: int,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.0,
        leg_notional: float = 1_000.0,
    ) -> None:
        if not instrument_a.strip() or not instrument_b.strip():
            raise ValueError("Both instrument identifiers must be non-empty")
        if instrument_a == instrument_b:
            raise ValueError("Pairs strategy requires two distinct instruments")
        if lookback_window <= 1:
            raise ValueError("lookback_window must be greater than 1")
        if entry_z_score <= 0:
            raise ValueError("entry_z_score must be positive")
        if exit_z_score < 0:
            raise ValueError("exit_z_score must be non-negative")
        if exit_z_score > entry_z_score:
            raise ValueError("exit_z_score cannot exceed entry_z_score")
        if leg_notional <= 0:
            raise ValueError("leg_notional must be positive")

        super().__init__(event_bus)
        self.instrument_a = instrument_a
        self.instrument_b = instrument_b
        self.lookback_window = lookback_window
        self.entry_z_score = float(entry_z_score)
        self.exit_z_score = float(exit_z_score)
        self.leg_notional = float(leg_notional)

        self._bayesian_vol = BayesianVolatilityAdjuster(event_bus)

        self._close_history: dict[str, Deque[float]] = {
            self.instrument_a: deque(maxlen=lookback_window),
            self.instrument_b: deque(maxlen=lookback_window),
        }
        self._latest_bar_ts: dict[str, datetime | None] = {
            self.instrument_a: None,
            self.instrument_b: None,
        }
        self._latest_close: dict[str, float | None] = {
            self.instrument_a: None,
            self.instrument_b: None,
        }
        self._last_processed_pair_ts: datetime | None = None
        self._spread_window = _RollingSeries.with_window(lookback_window)
        self._pair_state: PairState = "flat"

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id not in self._close_history:
            return

        self._close_history[event.instrument_id].append(event.close_price)
        self._latest_close[event.instrument_id] = event.close_price
        self._latest_bar_ts[event.instrument_id] = event.exchange_ts

        paired_ts = self._synchronized_timestamp()
        if paired_ts is None or paired_ts == self._last_processed_pair_ts:
            return

        spread = self._compute_spread()
        if spread is None:
            return

        self._spread_window.append(spread)
        self._last_processed_pair_ts = paired_ts
        if self._spread_window.count < self.lookback_window:
            return

        spread_mean = self._spread_window.mean()
        spread_std = self._spread_window.std()
        if spread_std <= 1e-12:
            return

        z_score = (spread - spread_mean) / spread_std
        desired_state = self._desired_state(z_score)
        if desired_state is None or desired_state == self._pair_state:
            return

        self._emit_state_transition(desired_state)
        self._pair_state = desired_state

    def on_tick(self, event: TradeTick) -> None:
        return

    def _desired_state(self, z_score: float) -> PairState | None:
        if z_score > self.entry_z_score:
            return "short_a_long_b"
        if z_score < -self.entry_z_score:
            return "long_a_short_b"
        if abs(z_score) <= self.exit_z_score:
            return "flat"
        return None

    def _emit_state_transition(self, desired_state: PairState) -> None:
        # 1. Check Bayesian Volatility Adjuster for pre-emptive scaling
        prob_a = self._bayesian_vol.probability_of_high_vol(self.instrument_a)
        prob_b = self._bayesian_vol.probability_of_high_vol(self.instrument_b)
        avg_prob = (prob_a + prob_b) / 2.0

        scaling_factor = 1.0
        if avg_prob > 0.80:
            scaling_factor = 0.5
            logger.warning(
                "Bayesian scaling active for %s/%s (P(High Vol)=%.2f). Reducing notional by 50%%.",
                self.instrument_a,
                self.instrument_b,
                avg_prob,
            )

        current_leg_notional = self.leg_notional * scaling_factor

        if desired_state == "short_a_long_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=-current_leg_notional,
                confidence=0.90,
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=current_leg_notional,
                confidence=0.90,
            )
            return

        if desired_state == "long_a_short_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=current_leg_notional,
                confidence=0.90,
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=-current_leg_notional,
                confidence=0.90,
            )
            return

        self.emit_signal(
            instrument_id=self.instrument_a,
            target_type="notional",
            target_value=0.0,
            confidence=0.95,
        )
        self.emit_signal(
            instrument_id=self.instrument_b,
            target_type="notional",
            target_value=0.0,
            confidence=0.95,
        )

    def _synchronized_timestamp(self) -> datetime | None:
        ts_a = self._latest_bar_ts[self.instrument_a]
        ts_b = self._latest_bar_ts[self.instrument_b]
        if ts_a is None or ts_b is None:
            return None
        if ts_a != ts_b:
            return None
        return ts_a

    def _compute_spread(self) -> float | None:
        price_a = self._latest_close[self.instrument_a]
        price_b = self._latest_close[self.instrument_b]
        if price_a is None or price_b is None:
            return None
        if price_b <= 0:
            return None
        return price_a / price_b
