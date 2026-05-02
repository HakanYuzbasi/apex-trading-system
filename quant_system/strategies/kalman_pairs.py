from __future__ import annotations

import json
import logging
import math
from collections import deque
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
        # Innovation history for OU half-life estimation (last 50 bars).
        self._innovation_history: list[float] = []
        # Layered exit: True once the 50% partial close has fired for this trade.
        self._kalman_half_closed: bool = False
        # Soft-exit cooldown: when a position exits with |z| still elevated (spread
        # failed to revert), block re-entry for 4 hours — the cointegration may be breaking.
        self._soft_exit_ts: datetime | None = None
        self._SOFT_EXIT_COOLDOWN_HOURS = 4.0
        self._kalman_diverged: bool = False
        self._diverge_ts: float = 0.0
        # IMP-1: rolling SNR history for notional scaling
        self._snr_history: deque[float] = deque(maxlen=20)
        self._snr_notional_mult: float = 1.0

    @property
    def current_entry_z(self) -> float:
        """Adaptive Z-score: higher VIX requires higher Z-score for entry."""
        vix_factor = (self._vix_level - self.vix_median) / self.vix_median
        return self.base_entry_z * (1.0 + max(0.0, vix_factor))

    @property
    def _safe_innovation_std(self) -> float:
        """sqrt(innovation_variance) with a floor to prevent crash on near-zero or
        negative variance (possible due to floating-point accumulation in the Kalman
        covariance update).  1e-8 is orders of magnitude below any real price spread."""
        return float(np.sqrt(max(self._last_innovation_variance, 1e-8)))

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

        # Mark this pair timestamp consumed NOW — before any early returns — so that the
        # second leg's bar (which arrives with the same canonical ts) doesn't trigger a
        # duplicate Kalman update with stale prices during the warmup period.
        self._last_processed_pair_ts = paired_ts

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

        # Layered exit: close 50% when spread has converged past PARTIAL_EXIT_Z.
        # Only fires once per trade (guarded by _kalman_half_closed).
        # Condition: entered beyond PARTIAL_EXIT_Z and z has now crossed back.
        _PARTIAL_EXIT_Z = 1.5
        if (
            self._pair_state != "flat"
            and not self._kalman_half_closed
            and self._entry_z_score is not None
            and abs(self._entry_z_score) > _PARTIAL_EXIT_Z
            and abs(z_score) <= _PARTIAL_EXIT_Z
        ):
            self._emit_partial_exit()
            self._kalman_half_closed = True

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
            self._kalman_half_closed = False   # reset for each new trade
        elif desired_state == "flat":
            # Soft-exit detection: spread still dislocated at exit means it never reverted.
            # Stamp a cooldown so we don't immediately re-enter a breaking pair.
            if abs(z_score) > 0.8:
                self._soft_exit_ts = event.exchange_ts
                logger.info(
                    "Kalman soft exit [%s/%s] z=%.2f — re-entry blocked for %.0fh",
                    self.instrument_a, self.instrument_b,
                    z_score, self._SOFT_EXIT_COOLDOWN_HOURS,
                )
            self._entry_ts = None
            self._entry_z_score = None
            self._entry_vix = None

        self._emit_state_transition(desired_state)
        self._pair_state = desired_state

    def on_tick(self, event: TradeTick) -> None:
        pass

    def _update_filter_and_score(self, price_a: float, price_b: float) -> float | None:
        import time

        # #3: Auto-reinit after 4h divergence cooldown — pairs self-heal without restart.
        if self._kalman_diverged and time.monotonic() - self._diverge_ts > 4.0 * 3600.0:
            logger.info(
                "🔄 [%s/%s] Divergence cooldown expired — reinitialising Kalman filter",
                self.instrument_a, self.instrument_b,
            )
            self._kalman_diverged = False
            self._kalman.observation_count = 0
            self._kalman.theta = np.zeros(2, dtype=float)
            self._kalman.covariance = np.eye(2, dtype=float) * 1_000.0
            self._innovation_history.clear()
            self._z_history.clear()
            # Price history is preserved — OLS warm-start will use the most recent bars.

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
            if self._kalman_diverged:
                # ADF test or obs_noise calibration flagged this pair as non-cointegrated.
                return None
            # Re-run with corrected theta and updated obs_noise so this bar's z is valid.
            new_theta, new_cov, innovation, innovation_variance = cython_kalman_update(
                price_a, price_b, self._kalman.theta, self._kalman.covariance,
                self.process_noise, self.observation_noise
            )
            self._kalman.theta = new_theta
            self._kalman.covariance = new_cov
            self._last_innovation = innovation
            self._last_innovation_variance = innovation_variance

        z_score = self._last_innovation / self._safe_innovation_std
        self.last_z_score = float(z_score)

        # #1: Post-warm-start gate — if the first corrected z is already outside ±2.5σ,
        # the spread is dislocated right now; block entries and auto-reinit in 4h.
        if (self._kalman.observation_count == self.warmup_bars
                and abs(z_score) > 2.5
                and not self._kalman_diverged):
            logger.warning(
                "🚫 [%s/%s] Post-warm-start z=%.1f > 2.5 — spread dislocated, "
                "blocking entries for 4h",
                self.instrument_a, self.instrument_b, z_score,
            )
            self._kalman_diverged = True
            self._diverge_ts = time.monotonic()
            return None

        # Maintain 3-bar z-score history for velocity filter.
        self._z_history.append(float(z_score))
        if len(self._z_history) > 3:
            self._z_history.pop(0)

        # Maintain 50-bar innovation history for OU half-life estimation.
        self._innovation_history.append(float(self._last_innovation))
        if len(self._innovation_history) > 50:
            self._innovation_history.pop(0)

        # Ongoing divergence guard: if |z| > 10 the filter has blown up — auto-reinit in 4h.
        if abs(z_score) > 10.0 and not self._kalman_diverged:
            logger.warning(
                "⚠️  [%s/%s] Kalman filter DIVERGED z=%.1f — auto-reinit in 4h",
                self.instrument_a, self.instrument_b, z_score,
            )
            self._kalman_diverged = True
            self._diverge_ts = time.monotonic()

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

        Also runs ADF stationarity test (#2) and auto-calibrates observation_noise
        (#4) — sets _kalman_diverged if the spread is not cointegrated right now.
        """
        import time as _time
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

        # #2: ADF stationarity test — residuals must be stationary (p ≤ 0.10) for the
        # pair to be cointegrated right now.  Non-stationary spreads produce persistent
        # z-drift that looks like signal but never reverts.
        # Skip when warmup window is too small for a meaningful test (need ≥ 20 obs).
        try:
            from statsmodels.tsa.stattools import adfuller
            if len(residuals) >= 20:
                _, adf_pval, *_ = adfuller(residuals, maxlag=1, autolag=None)
            else:
                adf_pval = 0.0  # not enough data — assume stationary, rely on z-gate
            if float(adf_pval) > 0.10:
                logger.warning(
                    "📉 [%s/%s] Spread not stationary ADF p=%.3f > 0.10 — "
                    "blocking pair for 4h",
                    self.instrument_a, self.instrument_b, float(adf_pval),
                )
                self._kalman_diverged = True
                self._diverge_ts = _time.monotonic()
                return
        except Exception as exc:
            logger.debug("[%s/%s] ADF test skipped: %s", self.instrument_a, self.instrument_b, exc)

        # #4: Auto-calibrate observation_noise from OLS residual variance.
        # A fixed tiny obs_noise (default 1e-3) against BTC/LTC resid_var≈690 makes
        # every bar's innovation 100× larger than expected → z explodes on first live bar.
        self.observation_noise = max(resid_var, self.observation_noise)

        # OLS gives estimation covariance (X'X)^{-1} σ² which scales as 1/n.
        # Kalman needs prediction-scale uncertainty so S ≈ resid_var after the
        # re-run.  Multiplying by n restores prediction variance and prevents
        # outlier bars near warmup from triggering |z| > 10.
        try:
            ols_cov = np.linalg.inv(pb_mat.T @ pb_mat) * resid_var * n
        except np.linalg.LinAlgError:
            ols_cov = np.eye(2, dtype=float) * resid_var
        # OLS uses [intercept, slope] column order; Kalman uses [slope, intercept].
        self._kalman.theta = np.array([theta_ols[1], theta_ols[0]], dtype=float)
        self._kalman.covariance = ols_cov
        logger.warning(
            "🎯 [%s/%s] Kalman OLS warm-start: beta=[%.4f, %.6f] resid_std=%.4f "
            "obs_noise=%.4f",
            self.instrument_a, self.instrument_b,
            float(theta_ols[0]), float(theta_ols[1]), float(np.std(residuals)),
            self.observation_noise,
        )

    def _is_crypto_overnight(self) -> bool:
        """True during 22:00–07:00 ET for crypto pairs — no new entries allowed."""
        if not self.instrument_a.startswith("CRYPTO:"):
            return False
        hour = datetime.now(_ET).hour
        return hour >= _CRYPTO_NO_ENTRY_START or hour < _CRYPTO_NO_ENTRY_END

    def _is_equity_auction_window(self) -> bool:
        """True during windows when new equity pair entries are blocked:
          - 9:30–9:45 ET  (opening auction, wide spreads)
          - 14:30–16:00 ET (afternoon + closing auction; EOD rebalancing creates
                            spurious spread divergences that are not mean-reverting)
        """
        if self.instrument_a.startswith("CRYPTO:") or self.instrument_a.startswith("FOREX:"):
            return False
        mins = datetime.now(_ET).hour * 60 + datetime.now(_ET).minute
        open_start      = 9 * 60 + 30    # 9:30 ET
        open_end        = 9 * 60 + 45    # 9:45 ET
        afternoon_start = 14 * 60 + 30   # 14:30 ET — no new entries from here on
        close_end       = 16 * 60        # 16:00 ET
        return (open_start <= mins < open_end) or (afternoon_start <= mins < close_end)

    def _ou_half_life_days(self) -> float:
        """Estimate OU mean-reversion half-life in calendar days.

        Uses lag-1 autocorrelation of the Kalman innovations:
          rho  = corr(e_t, e_{t-1})
          kappa = -log(rho) per bar
          half_life = log(2) / kappa bars  →  divide by 390 for US-equity trading days

        Returns inf when history is too short or the spread is not mean-reverting.
        """
        hist = self._innovation_history
        if len(hist) < 10:
            return 0.0  # not enough data — allow entry during warmup
        e = np.array(hist, dtype=float)
        lag0 = np.dot(e[:-1], e[:-1])
        lag1 = np.dot(e[:-1], e[1:])
        if lag0 < 1e-12:
            return float("inf")
        rho = lag1 / lag0
        if rho <= 0.0 or rho >= 1.0:
            return float("inf")
        kappa = -math.log(rho)  # per bar (1-min bars)
        return math.log(2.0) / kappa / 390.0

    def _spread_converging(self) -> bool:
        """True if the last 2 z-scores are moving toward zero (convergence confirmed)."""
        if len(self._z_history) < 2:
            return True  # not enough history — allow entry during warmup completion
        z_prev, z_curr = self._z_history[-2], self._z_history[-1]
        return abs(z_curr) < abs(z_prev)

    def _regime_entry_z_mult(self) -> float:
        """
        Multiply entry z-score in trending regimes where equity mean-reversion
        is unreliable.  Only applied to equity pairs (not crypto/forex).
        """
        a = self.instrument_a
        if a.startswith("CRYPTO:") or a.startswith("FOREX:"):
            return 1.0
        try:
            from risk.regime_router import get_global_regime_router
            rr = get_global_regime_router()
            if rr is not None:
                regime = getattr(rr.last_regime, "value", "neutral")
                if regime in ("bull_trend", "strong_bull"):
                    return 1.5   # require much larger dislocation
                if regime in ("bear_trend", "strong_bear"):
                    return 1.3   # trend overrides reversion here too
        except Exception as exc:
            logger.debug("[%s/%s] Regime z-mult unavailable: %s", self.instrument_a, self.instrument_b, exc)
        return 1.0

    def _in_soft_exit_cooldown(self, current_ts: datetime) -> bool:
        """True if the pair exited with an unrevert spread recently (< 4h ago)."""
        if self._soft_exit_ts is None:
            return False
        elapsed_hours = (current_ts - self._soft_exit_ts).total_seconds() / 3600.0
        return elapsed_hours < self._SOFT_EXIT_COOLDOWN_HOURS

    def _half_life_limit(self) -> float:
        """#7: Return max tolerable OU half-life in days.
        Crypto pairs in ranging/neutral/volatile regimes require intraday reversion
        (≤1 day) — longer half-lives mean the spread won't close before EOD.
        Equity pairs keep the 5-day limit.
        """
        if not self.instrument_a.startswith("CRYPTO:"):
            return 5.0
        try:
            from risk.regime_router import get_global_regime_router
            rr = get_global_regime_router()
            if rr is not None:
                regime = getattr(rr.last_regime, "value", "neutral")
                if regime in ("ranging", "neutral", "volatile"):
                    return 1.0
        except Exception as exc:
            logger.debug("[%s/%s] Half-life limit check failed: %s", self.instrument_a, self.instrument_b, exc)
        return 3.0  # default crypto limit when regime unknown

    def _is_altcoin_pair(self) -> bool:
        """True when both legs are crypto and neither is BTC."""
        a, b = self.instrument_a, self.instrument_b
        both_crypto = a.startswith("CRYPTO:") and b.startswith("CRYPTO:")
        either_btc  = "BTC" in a or "BTC" in b
        return both_crypto and not either_btc

    def _btc_dominance_rising(self) -> bool:
        try:
            from quant_system.risk.btc_dominance_monitor import get_btc_dominance_monitor
            return get_btc_dominance_monitor().is_btc_dominance_rising()
        except Exception:
            # Conservative: assume rising when CoinGecko is unreachable — block new entries.
            return True

    def _desired_state(self, z_score: float, current_ts: datetime) -> str | None:
        entry_z = self.current_entry_z * self._regime_entry_z_mult()

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
                if (self._kalman_diverged
                        or self._is_crypto_overnight()
                        or self._is_equity_auction_window()):
                    return None
                if self._ou_half_life_days() > self._half_life_limit():
                    return None
                if not self._spread_converging():
                    return None
                if self._is_altcoin_pair() and self._btc_dominance_rising():
                    return None
                if self._in_soft_exit_cooldown(current_ts):
                    return None
            return "short_a_long_b"
        if z_score < -entry_z:
            if self._pair_state == "flat":
                if (self._kalman_diverged
                        or self._is_crypto_overnight()
                        or self._is_equity_auction_window()):
                    return None
                if self._ou_half_life_days() > self._half_life_limit():
                    return None
                if not self._spread_converging():
                    return None
                if self._is_altcoin_pair() and self._btc_dominance_rising():
                    return None
                if self._in_soft_exit_cooldown(current_ts):
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

    def _emit_partial_exit(self) -> None:
        """Emit signals that reduce each leg to 50% notional (layered exit)."""
        half = self.leg_notional * 0.5
        z    = self.last_z_score
        meta = {"trigger": "partial_z_exit", "z": round(z, 3)}
        if self._pair_state == "long_a_short_b":
            self.emit_signal(self.instrument_a, "notional", +half, confidence=0.92, metadata=meta)
            self.emit_signal(self.instrument_b, "notional", -half, confidence=0.92, metadata=meta)
        else:  # short_a_long_b
            self.emit_signal(self.instrument_a, "notional", -half, confidence=0.92, metadata=meta)
            self.emit_signal(self.instrument_b, "notional", +half, confidence=0.92, metadata=meta)
        logger.info(
            "Kalman PARTIAL EXIT 50%% [%s/%s] z=%.3f | remaining $%.0f each leg",
            self.instrument_a, self.instrument_b, z, half,
        )

    def _emit_state_transition(self, desired_state: str) -> None:
        snr_val = abs(self._last_innovation) / self._safe_innovation_std

        # IMP-1: update rolling SNR history and recompute notional multiplier
        self._snr_history.append(float(snr_val))
        if len(self._snr_history) >= 10:
            median_snr = float(np.median(list(self._snr_history)))
            if median_snr > 1e-6:
                ratio = float(snr_val) / median_snr
                self._snr_notional_mult = float(np.clip(ratio, 0.7, 1.3))
            else:
                self._snr_notional_mult = 1.0

        metadata = {
            "kalman_residual": float(self._last_innovation),
            "z_score": float(self._last_innovation / self._safe_innovation_std),
            "vix_level": self._vix_level,
            "snr": float(snr_val),
            "snr_mult": round(self._snr_notional_mult, 3),
        }

        # SNR-scaled notional: high-quality pairs get up to 1.3×, low-quality 0.7×
        scaled_notional = self.leg_notional * self._snr_notional_mult

        if desired_state == "short_a_long_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=-scaled_notional,
                confidence=0.92,
                metadata=metadata
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=scaled_notional,
                confidence=0.92,
                metadata=metadata
            )
        elif desired_state == "long_a_short_b":
            self.emit_signal(
                instrument_id=self.instrument_a,
                target_type="notional",
                target_value=scaled_notional,
                confidence=0.92,
                metadata=metadata
            )
            self.emit_signal(
                instrument_id=self.instrument_b,
                target_type="notional",
                target_value=-scaled_notional,
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
            "kalman_diverged": self._kalman_diverged,
            "diverge_ts_monotonic": self._diverge_ts,
            "price_history_a": self._price_history_a[-100:],
            "price_history_b": self._price_history_b[-100:],
            "innovation_history": self._innovation_history[-50:],
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
                import time as _time
                saved_ts = float(payload.get("diverge_ts_monotonic", 0.0))
                # Remap saved wall-clock offset to current monotonic clock.
                # If the saved value is 0, treat the diverge as happening now.
                if saved_ts > 0:
                    self._diverge_ts = _time.monotonic() - (_time.time() - saved_ts)
                else:
                    self._diverge_ts = _time.monotonic()
            self._price_history_a = payload.get("price_history_a", [])
            self._price_history_b = payload.get("price_history_b", [])
            self._innovation_history = payload.get("innovation_history", [])
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
