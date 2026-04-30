from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

_ET = ZoneInfo("America/New_York")
# No new crypto entries 22:00–07:00 ET (highest overnight volatility).
_CRYPTO_NO_ENTRY_START = 22
_CRYPTO_NO_ENTRY_END   = 7

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
        # Rolling z-score history for velocity filter (last 3 bars).
        self._z_history: list[float] = []

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

        if self._kalman.observation_count < self.warmup_bars:
            return None

        # On exact warmup completion: OLS warm-start so the filter inherits a
        # sensible hedge-ratio prior instead of starting cold at theta=[0,0].
        # Without this, BTC/XRP (price ratio ~40,800) produces z>500 on bar 20.
        if self._kalman.observation_count == self.warmup_bars:
            self._warm_start_kalman()
            # Re-run with corrected theta so this bar's z-score is valid.
            new_theta, new_cov, innovation, innovation_variance = cython_kalman_update(
                price_a, price_b, self._kalman.theta, self._kalman.covariance,
                self.process_noise, self.observation_noise
            )
            self._kalman.theta = new_theta
            self._kalman.covariance = new_cov
            self._last_innovation = innovation
            self._last_innovation_variance = innovation_variance

        z_score = self._last_innovation / np.sqrt(self._last_innovation_variance)
        self.last_z_score = float(z_score)

        # Maintain 3-bar z-score history for velocity filter.
        self._z_history.append(float(z_score))
        if len(self._z_history) > 3:
            self._z_history.pop(0)

        # Divergence guard: if |z| > 10 after a proper warm-start, the filter
        # has genuinely blown up on live data — block entries until restart.
        if abs(z_score) > 10.0 and not getattr(self, "_kalman_diverged", False):
            logger.warning(
                "⚠️  [%s/%s] Kalman filter DIVERGED z=%.1f — entries blocked until restart",
                self.instrument_a, self.instrument_b, z_score,
            )
            self._kalman_diverged = True

        if self._kalman.observation_count == self.warmup_bars or self._kalman.observation_count % 50 == 0:
            logger.info(
                "✅ [%s/%s] warmup done (obs=%d) z=%.3f entry_z=%.2f",
                self.instrument_a, self.instrument_b,
                self._kalman.observation_count, z_score, self.base_entry_z,
            )
        return self.last_z_score

    def _warm_start_kalman(self) -> None:
        """
        Reinitialise Kalman theta/covariance from an OLS estimate over the
        warmup window.  Prevents cold-start divergence when price_a / price_b
        differs by orders of magnitude (e.g. BTC/XRP ~40,800×).
        """
        n = self.warmup_bars
        pa = np.array(self._price_history_a[-n:], dtype=float)
        pb = np.array(self._price_history_b[-n:], dtype=float)
        if len(pa) < 2:
            return
        pb_mat = np.column_stack([np.ones_like(pb), pb])
        try:
            theta_ols, _, _, _ = np.linalg.lstsq(pb_mat, pa, rcond=None)
        except np.linalg.LinAlgError:
            return
        residuals = pa - pb_mat @ theta_ols
        resid_var = max(float(np.var(residuals)), self.observation_noise)
        # OLS gives estimation covariance (X'X)^{-1} σ² which scales as 1/n.
        # Kalman needs prediction-scale uncertainty so S ≈ resid_var after the
        # re-run.  Multiplying by n restores prediction variance and prevents
        # outlier bars near warmup from triggering |z| > 10.
        try:
            ols_cov = np.linalg.inv(pb_mat.T @ pb_mat) * resid_var * n
        except np.linalg.LinAlgError:
            ols_cov = np.eye(2, dtype=float) * resid_var
        self._kalman.theta = theta_ols
        self._kalman.covariance = ols_cov
        logger.warning(
            "🎯 [%s/%s] Kalman OLS warm-start: beta=[%.4f, %.6f] resid_std=%.4f",
            self.instrument_a, self.instrument_b,
            float(theta_ols[0]), float(theta_ols[1]), float(np.std(residuals)),
        )

    def _is_crypto_overnight(self) -> bool:
        """True during 22:00–07:00 ET for crypto pairs — no new entries allowed."""
        if not self.instrument_a.startswith("CRYPTO:"):
            return False
        hour = datetime.now(_ET).hour
        return hour >= _CRYPTO_NO_ENTRY_START or hour < _CRYPTO_NO_ENTRY_END

    def _is_equity_auction_window(self) -> bool:
        """True during the first/last 15 min of the equity session (9:30–9:45, 15:45–16:00 ET).
        Spreads are widest and fills are worst in these windows; skip new entries."""
        if self.instrument_a.startswith("CRYPTO:") or self.instrument_a.startswith("FOREX:"):
            return False
        now = datetime.now(_ET)
        # Convert to minutes since midnight for easy comparison
        mins = now.hour * 60 + now.minute
        open_end  = 9 * 60 + 45   # 9:45 ET
        close_start = 15 * 60 + 45  # 15:45 ET
        open_start = 9 * 60 + 30   # 9:30 ET
        close_end  = 16 * 60        # 16:00 ET
        return (open_start <= mins < open_end) or (close_start <= mins < close_end)

    def _ou_half_life_days(self) -> float:
        """Estimate OU mean-reversion half-life in calendar days from Kalman beta[1]."""
        beta1 = float(self._kalman.theta[1])
        if beta1 >= 0.0 or abs(beta1) < 1e-9:
            return float("inf")
        # beta[1] is the hedge ratio slope; negative means mean-reverting.
        # Half-life in bars: log(2)/|beta[1]|.  Bars are 1-minute, so /390 for days.
        return math.log(2.0) / abs(beta1) / 390.0

    def _spread_converging(self) -> bool:
        """True if the last 2 z-scores are moving toward zero (convergence confirmed)."""
        if len(self._z_history) < 2:
            return True  # not enough history — allow entry during warmup completion
        z_prev, z_curr = self._z_history[-2], self._z_history[-1]
        return abs(z_curr) < abs(z_prev)

    def _desired_state(self, z_score: float, current_ts: datetime) -> str | None:
        entry_z = self.current_entry_z

        # Hard stop-loss: if spread has blown past 4σ while in a position, cut losses.
        # This caps drawdown from structural breaks that the Kalman filter can't predict.
        if self._pair_state != "flat" and abs(z_score) > 4.0:
            logger.warning(
                "🛑 [%s/%s] Hard stop-loss triggered z=%.2f > 4.0 — exiting %s",
                self.instrument_a, self.instrument_b, z_score, self._pair_state,
            )
            return "flat"

        if z_score > entry_z:
            if self._pair_state == "flat":
                if (getattr(self, "_kalman_diverged", False)
                        or self._is_crypto_overnight()
                        or self._is_equity_auction_window()):
                    return None
                # OU half-life filter: skip slow-reverting pairs (> 5 calendar days).
                if self._ou_half_life_days() > 5.0:
                    return None
                # Velocity filter: only enter when spread is already converging.
                if not self._spread_converging():
                    return None
            return "short_a_long_b"
        if z_score < -entry_z:
            if self._pair_state == "flat":
                if (getattr(self, "_kalman_diverged", False)
                        or self._is_crypto_overnight()
                        or self._is_equity_auction_window()):
                    return None
                if self._ou_half_life_days() > 5.0:
                    return None
                if not self._spread_converging():
                    return None
            return "long_a_short_b"

        current_exit_z = self.exit_z_score
        if self._pair_state != "flat" and self._entry_ts:
            # Time-based decay
            dt_hours = (current_ts - self._entry_ts).total_seconds() / 3600.0
            decay_factor = math.pow(0.5, dt_hours / self.decay_half_life_hours)
            current_exit_z = self.exit_z_score * decay_factor

            # Asymmetric exit: exit 2× faster when profitable
            is_profitable = (
                (self._pair_state == "long_a_short_b" and z_score > self._entry_z_score)
                or (self._pair_state == "short_a_long_b" and z_score < self._entry_z_score)
            )
            if is_profitable:
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

    def save_state(self, state_dir: str | Path) -> None:
        """Persist Kalman filter state to disk so a restart can resume without cold-start divergence."""
        path = Path(state_dir) / f"{self.instrument_a}_{self.instrument_b}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "theta": self._kalman.theta.tolist(),
            "covariance": self._kalman.covariance.tolist(),
            "observation_count": self._kalman.observation_count,
            "pair_state": self._pair_state,
            "kalman_diverged": getattr(self, "_kalman_diverged", False),
            "price_history_a": self._price_history_a[-100:],
            "price_history_b": self._price_history_b[-100:],
        }
        path.write_text(json.dumps(payload))
        logger.debug("💾 [%s/%s] Kalman state saved → %s", self.instrument_a, self.instrument_b, path)

    def load_state(self, state_dir: str | Path) -> bool:
        """Load persisted Kalman state. Returns True if state was restored."""
        path = Path(state_dir) / f"{self.instrument_a}_{self.instrument_b}.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
            self._kalman.theta = np.array(payload["theta"], dtype=float)
            self._kalman.covariance = np.array(payload["covariance"], dtype=float)
            self._kalman.observation_count = int(payload["observation_count"])
            self._pair_state = payload.get("pair_state", "flat")
            if payload.get("kalman_diverged", False):
                self._kalman_diverged = True
            self._price_history_a = payload.get("price_history_a", [])
            self._price_history_b = payload.get("price_history_b", [])
            logger.warning(
                "♻️  [%s/%s] Kalman state restored: obs=%d pair_state=%s",
                self.instrument_a, self.instrument_b,
                self._kalman.observation_count, self._pair_state,
            )
            return True
        except Exception as exc:
            logger.warning("⚠️  [%s/%s] Failed to load Kalman state (%s) — starting fresh", self.instrument_a, self.instrument_b, exc)
            return False

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
