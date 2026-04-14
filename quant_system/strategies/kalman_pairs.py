from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy
from quant_system.strategies.stabilizer import JohansenStabilizer
from quant_system.execution.fast_math import cython_kalman_update

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class KalmanState:
    theta: np.ndarray
    covariance: np.ndarray
    observation_count: int = 0

class KalmanPairsStrategy(BaseStrategy):
    """
    Adaptive pairs strategy using a Kalman filter hedge ratio estimate.
    Includes Johansen Cointegration Test and VIX-adaptive entry thresholds.
    """

    def __init__(
        self,
        event_bus,
        *,
        instrument_a: str,
        instrument_b: str,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        leg_notional: float = 1_000.0,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-3,
        warmup_bars: int = 20,
        decay_half_life_hours: float = 4.0,
        vix_median: float = 18.5,
    ) -> None:
        super().__init__(event_bus)
        self.instrument_a = instrument_a
        self.instrument_b = instrument_b
        self.base_entry_z = float(entry_z_score)
        self.exit_z_score = float(exit_z_score)
        self.leg_notional = float(leg_notional)
        self.process_noise = float(process_noise)
        self.observation_noise = float(observation_noise)
        self.warmup_bars = int(warmup_bars)
        self.decay_half_life_hours = float(decay_half_life_hours)
        self.vix_median = float(vix_median)
        
        self.last_latency_micros: float = 0.0

        self._latest_close: dict[str, float | None] = {
            self.instrument_a: None,
            self.instrument_b: None,
        }
        self._latest_bar_ts: dict[str, datetime | None] = {
            self.instrument_a: None,
            self.instrument_b: None,
        }
        self._price_history_a = []
        self._price_history_b = []
        
        self._vix_level: float = self.vix_median
        self._last_processed_pair_ts: datetime | None = None
        self._kalman = KalmanState(
            theta=np.zeros(2, dtype=float),
            covariance=np.eye(2, dtype=float) * 1_000.0,
        )
        self._stabilizer = JohansenStabilizer()
        self._pair_state = "flat"
        self._entry_ts: datetime | None = None
        self._is_read_only: bool = False
        self._entry_z_score: float | None = None
        self._entry_vix: float | None = None
        self.last_z_score: float = 0.0

    @property
    def current_entry_z(self) -> float:
        """Adaptive Z-score: higher VIX requires higher Z-score for entry."""
        vix_factor = (self._vix_level - self.vix_median) / self.vix_median
        return self.base_entry_z * (1.0 + max(0.0, vix_factor))

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id == "VIX":
            self._vix_level = event.close_price
            return

        if event.instrument_id not in self._latest_close:
            return

        self._latest_close[event.instrument_id] = event.close_price
        self._latest_bar_ts[event.instrument_id] = event.exchange_ts

        paired_ts = self._synchronized_timestamp()
        if paired_ts is None or paired_ts == self._last_processed_pair_ts:
            return

        price_a = self._latest_close[self.instrument_a]
        price_b = self._latest_close[self.instrument_b]
        
        if price_a is None or price_b is None:
            return

        # Keep history for cointegration check
        self._price_history_a.append(price_a)
        self._price_history_b.append(price_b)
        if len(self._price_history_a) > 500: # Max window for stability test
            self._price_history_a.pop(0)
            self._price_history_b.pop(0)

        z_score = self._update_filter_and_score(price_a=float(price_a), price_b=float(price_b))
        if z_score is None:
            return

        # Stability Check (Johansen) — debounce logic lives in the stabilizer
        self._stabilizer.test(self._price_history_a, self._price_history_b)
        self._is_read_only = self._stabilizer.is_read_only

        desired_state = self._desired_state(z_score, event.exchange_ts)
        if desired_state is None or desired_state == self._pair_state:
            return

        # Veto entry if in read-only mode
        if self._pair_state == "flat" and desired_state != "flat" and self._is_read_only:
            return

        if self._pair_state == "flat" and desired_state != "flat":
            self._entry_ts = event.exchange_ts
            self._entry_z_score = z_score
            self._entry_vix = self._vix_level
        elif desired_state == "flat":
            self._entry_ts = None
            self._entry_z_score = None
            self._entry_vix = None

        self._emit_state_transition(desired_state)
        self._pair_state = desired_state
        self._last_processed_pair_ts = paired_ts

    def on_tick(self, event: TradeTick) -> None:
        pass

    def _update_filter_and_score(self, price_a: float, price_b: float) -> float | None:
        import time
        t0 = time.perf_counter_ns()
        
        new_theta, new_cov, innovation, innovation_variance = cython_kalman_update(
            price_a,
            price_b,
            self._kalman.theta,
            self._kalman.covariance,
            self.process_noise,
            self.observation_noise
        )
        
        self._kalman.theta = new_theta
        self._kalman.covariance = new_cov
        self._kalman.observation_count += 1
        
        t1 = time.perf_counter_ns()
        self.last_latency_micros = (t1 - t0) / 1000.0

        self._last_innovation = innovation
        self._last_innovation_variance = innovation_variance

        self._last_innovation = innovation
        self._last_innovation_variance = innovation_variance

        if self._kalman.observation_count < self.warmup_bars:
            return None

        z_score = self._last_innovation / np.sqrt(self._last_innovation_variance)
        self.last_z_score = float(z_score)
        if self._kalman.observation_count == self.warmup_bars or self._kalman.observation_count % 5 == 0:
            logger.info(
                "✅ [%s/%s] warmup done (obs=%d) z=%.3f entry_z=%.2f",
                self.instrument_a, self.instrument_b,
                self._kalman.observation_count, z_score, self.base_entry_z,
            )
        return self.last_z_score

    def _desired_state(self, z_score: float, current_ts: datetime) -> str | None:
        entry_z = self.current_entry_z
        
        if z_score > entry_z:
            return "short_a_long_b"
        if z_score < -entry_z:
            return "long_a_short_b"
        
        current_exit_z = self.exit_z_score
        if self._pair_state != "flat" and self._entry_ts:
            # 1. Time-based decay (existing)
            dt_hours = (current_ts - self._entry_ts).total_seconds() / 3600.0
            decay_factor = math.pow(0.5, dt_hours / self.decay_half_life_hours)
            current_exit_z = self.exit_z_score * decay_factor
            
            # 2. Asymmetric Alpha Patch: Exit faster if profitable
            # If we are long the spread (z was < -entry_z) and z is moving towards 0 (positive direction)
            # OR if we are short the spread (z was > entry_z) and z is moving towards 0 (negative direction)
            is_profitable = False
            if self._pair_state == "long_a_short_b" and z_score > self._entry_z_score:
                is_profitable = True
            elif self._pair_state == "short_a_long_b" and z_score < self._entry_z_score:
                is_profitable = True
                
            if is_profitable:
                # Institutional Success Secret: Exit 2x faster if in the money
                current_exit_z *= 0.5
                
        if abs(z_score) <= current_exit_z:
            return "flat"
        return None

    def _emit_state_transition(self, desired_state: str) -> None:
        snr_val = abs(self._last_innovation) / np.sqrt(self._last_innovation_variance)
        
        # Add metadata for Meta-Labeler
        metadata = {
            "kalman_residual": float(self._last_innovation),
            "z_score": float(self._last_innovation / np.sqrt(self._last_innovation_variance)),
            "vix_level": self._vix_level,
            "snr": float(snr_val)
        }

        if desired_state == "short_a_long_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=-self.leg_notional,
                confidence=0.92,
                metadata=metadata
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=self.leg_notional,
                confidence=0.92,
                metadata=metadata
            )
        elif desired_state == "long_a_short_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=self.leg_notional,
                confidence=0.92,
                metadata=metadata
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=-self.leg_notional,
                confidence=0.92,
                metadata=metadata
            )
        elif desired_state == "flat":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=0.0,
                confidence=1.0,
                metadata=metadata
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=0.0,
                confidence=1.0,
                metadata=metadata
            )

    def _synchronized_timestamp(self) -> datetime | None:
        ts_a = self._latest_bar_ts[self.instrument_a]
        ts_b = self._latest_bar_ts[self.instrument_b]
        if ts_a is None or ts_b is None:
            return None
        # Use the most-recent of the two timestamps as the canonical clock tick.
        # Alpaca delivers bars for different symbols at different wall-clock times
        # (BTC may arrive 1-2 minutes before ETH for the same calendar minute),
        # so strict equality would permanently block all pair updates.
        # Using max(ts_a, ts_b) and guarding with _last_processed_pair_ts ensures
        # each unique canonical minute is processed exactly once.
        return max(ts_a, ts_b)
