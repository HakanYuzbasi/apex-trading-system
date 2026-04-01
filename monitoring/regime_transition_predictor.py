"""
monitoring/regime_transition_predictor.py — Forward-Looking Regime Transition Predictor

Estimates the probability that the current market regime will shift within the
next 1-4 hours, using 5 composite indicators that are purely in-process
(no external API calls, no paid services):

  1. VIX velocity          — rapid VIX rise precedes risk-off transitions
  2. VIX extreme reversion — very high / very low VIX reverts toward mean
  3. SPY momentum exhaustion — momentum sign-flips near 0 signal regime fatigue
  4. Regime age pressure   — the longer in a regime, the higher the base transition rate
  5. Signal variance gate  — high recent signal dispersion = unstable regime

Outputs a TransitionPrediction with:
  - probability (0-1): P(regime changes within 1-4h)
  - direction: "risk_on", "risk_off", or "unknown"
  - size_multiplier: 1.0 normally; reduces toward 0.60 as probability rises

Usage (execution_loop.py):
    predictor.update(vix=vix_level, spy_return_1h=spy_ret, current_regime=regime_str)
    mult = predictor.get_size_multiplier()  # applied to entry size
    pred = predictor.predict()
    if pred.probability > 0.65:
        logger.warning("Regime transition imminent: p=%.2f dir=%s", ...)
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional


# ── Indicator weights (must sum to 1.0) ───────────────────────────────────────
_WEIGHTS = {
    "vix_velocity":       0.30,
    "vix_extreme":        0.20,
    "momentum_exhaustion": 0.25,
    "regime_age":         0.15,
    "signal_variance":    0.10,
}

# ── Regime classification helpers ────────────────────────────────────────────
_RISK_OFF_REGIMES = frozenset({"bear", "strong_bear", "volatile", "crisis"})
_RISK_ON_REGIMES  = frozenset({"bull", "strong_bull"})


@dataclass
class TransitionPrediction:
    probability: float      # 0-1
    direction: str          # "risk_off" | "risk_on" | "unknown"
    indicator_scores: dict  # per-indicator raw scores
    size_multiplier: float  # 1.0 → 0.60 as probability rises
    regime_age_hours: float
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "direction": self.direction,
            "indicator_scores": {k: round(v, 4) for k, v in self.indicator_scores.items()},
            "size_multiplier": round(self.size_multiplier, 4),
            "regime_age_hours": round(self.regime_age_hours, 2),
            "updated_at": self.updated_at,
        }


class RegimeTransitionPredictor:
    """
    Lightweight forward-looking regime transition probability estimator.

    Thread-safe for reading (predict / get_size_multiplier).
    update() should be called from the main async loop (no lock needed for single writer).
    """

    def __init__(
        self,
        vix_window: int = 24,        # number of readings (~hours at 1/cycle)
        spy_window: int = 12,        # SPY return rolling window
        signal_window: int = 20,     # composite signal history for variance
        high_prob_threshold: float = 0.60,  # above this → reduce size
        min_size_mult: float = 0.60,        # floor for size multiplier
    ) -> None:
        self._vix_window = vix_window
        self._spy_window = spy_window
        self._signal_window = signal_window
        self._high_prob_threshold = high_prob_threshold
        self._min_size_mult = min_size_mult

        # Rolling history buffers
        self._vix_history:    Deque[float] = deque(maxlen=vix_window)
        self._spy_returns:    Deque[float] = deque(maxlen=spy_window)
        self._signal_history: Deque[float] = deque(maxlen=signal_window)

        # Regime state tracking
        self._current_regime: str = "neutral"
        self._regime_start_ts: float = datetime.now(timezone.utc).timestamp()

        # Cached last prediction
        self._last_prediction: Optional[TransitionPrediction] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        vix: float,
        spy_return_1h: float = 0.0,
        current_regime: str = "neutral",
        composite_signal: float = 0.0,
    ) -> None:
        """
        Record latest market observations.
        Call once per main loop cycle (or at least hourly).
        """
        if vix > 0:
            self._vix_history.append(float(vix))
        self._spy_returns.append(float(spy_return_1h))
        self._signal_history.append(float(composite_signal))

        # Track regime transitions
        if current_regime != self._current_regime:
            self._current_regime = current_regime
            self._regime_start_ts = datetime.now(timezone.utc).timestamp()

    def predict(self) -> TransitionPrediction:
        """Compute and return a fresh TransitionPrediction."""
        scores = self._compute_indicator_scores()
        prob = self._aggregate_probability(scores)
        direction = self._infer_direction(scores)
        mult = self._size_multiplier(prob)
        age_h = self._regime_age_hours()

        pred = TransitionPrediction(
            probability=prob,
            direction=direction,
            indicator_scores=scores,
            size_multiplier=mult,
            regime_age_hours=age_h,
        )
        self._last_prediction = pred
        return pred

    def get_size_multiplier(self) -> float:
        """
        Quick accessor — returns last cached multiplier or recomputes if stale.
        Safe to call from hot path without re-running full predict().
        """
        if self._last_prediction is not None:
            return self._last_prediction.size_multiplier
        return self.predict().size_multiplier

    def get_last_prediction(self) -> Optional[dict]:
        if self._last_prediction is None:
            return None
        return self._last_prediction.to_dict()

    # ── Indicator scoring ─────────────────────────────────────────────────────

    def _compute_indicator_scores(self) -> dict:
        return {
            "vix_velocity":        self._score_vix_velocity(),
            "vix_extreme":         self._score_vix_extreme(),
            "momentum_exhaustion": self._score_momentum_exhaustion(),
            "regime_age":          self._score_regime_age(),
            "signal_variance":     self._score_signal_variance(),
        }

    def _score_vix_velocity(self) -> float:
        """
        Measures rate of VIX change over the recent window.
        Fast rising VIX → high score (risk-off transition imminent).
        Fast falling VIX → small negative score (possible risk-on recovery).
        Returns [-1, 1]; positive = transition risk.
        """
        vix = list(self._vix_history)
        if len(vix) < 4:
            return 0.0
        # Compare latest 4 readings vs prior 4 readings
        recent = vix[-4:]
        prior_start = max(0, len(vix) - 8)
        prior = vix[prior_start: len(vix) - 4]
        if not prior:
            return 0.0
        recent_mean = sum(recent) / len(recent)
        prior_mean  = sum(prior)  / len(prior)
        if prior_mean <= 0:
            return 0.0
        pct_change = (recent_mean - prior_mean) / prior_mean
        # Normalise: a 30% VIX rise → score 1.0
        return float(max(-1.0, min(1.0, pct_change / 0.30)))

    def _score_vix_extreme(self) -> float:
        """
        Extreme VIX readings revert. >35 → likely risk-off peak (small positive score).
        <12 → complacency, potential risk-off snap (positive score).
        Returns [0, 1].
        """
        vix = list(self._vix_history)
        if not vix:
            return 0.0
        current = vix[-1]
        if current >= 35:
            return float(min(1.0, (current - 35) / 20))  # 35→0, 55→1
        if current <= 12:
            return float(min(1.0, (12 - current) / 4))   # 12→0, 8→1
        return 0.0

    def _score_momentum_exhaustion(self) -> float:
        """
        SPY return momentum near sign-flip indicates exhaustion.
        Uses sum of recent returns vs total window — momentum near 0 = high score.
        Returns [0, 1].
        """
        rets = list(self._spy_returns)
        if len(rets) < 4:
            return 0.0
        cumulative = sum(rets)
        abs_sum    = sum(abs(r) for r in rets)
        if abs_sum < 1e-8:
            return 0.0
        # If cumulative ≈ 0 but abs_sum is large → momentum oscillating → score high
        directional_ratio = abs(cumulative) / abs_sum
        return float(max(0.0, 1.0 - directional_ratio))

    def _score_regime_age(self) -> float:
        """
        Regime persistence decay: higher score the longer we've been in a regime.
        Empirically, >48h in any regime significantly raises transition probability.
        Returns [0, 1].
        """
        age_h = self._regime_age_hours()
        # Sigmoid: score=0.5 at 48h, 0.9 at 120h
        return float(1 / (1 + math.exp(-0.06 * (age_h - 48))))

    def _score_signal_variance(self) -> float:
        """
        High variance in composite signal → model disagrees → regime unstable.
        Returns [0, 1].
        """
        sigs = list(self._signal_history)
        if len(sigs) < 5:
            return 0.0
        try:
            std = statistics.stdev(sigs)
        except statistics.StatisticsError:
            return 0.0
        # Typical stable std ~0.02; chaotic std ~0.08+
        return float(min(1.0, std / 0.06))

    # ── Aggregation and helpers ───────────────────────────────────────────────

    def _aggregate_probability(self, scores: dict) -> float:
        """Weighted sum of indicator scores, clamped to [0, 1]."""
        total = 0.0
        for key, weight in _WEIGHTS.items():
            raw = scores.get(key, 0.0)
            # Negative scores (e.g. risk-on recovery from vix_velocity) still count as 0
            total += weight * max(0.0, raw)
        return float(max(0.0, min(1.0, total)))

    def _infer_direction(self, scores: dict) -> str:
        """Infer likely direction of transition from indicator combination."""
        vix_vel = scores.get("vix_velocity", 0.0)
        mom_ex  = scores.get("momentum_exhaustion", 0.0)
        current = self._current_regime

        if current in _RISK_ON_REGIMES:
            # Currently risk-on; VIX rising fast → going risk-off
            if vix_vel > 0.20 or mom_ex > 0.65:
                return "risk_off"
            return "unknown"
        if current in _RISK_OFF_REGIMES:
            # Currently risk-off; VIX falling, momentum recovering → risk-on
            if vix_vel < -0.10 and mom_ex < 0.30:
                return "risk_on"
            return "unknown"
        return "unknown"

    def _size_multiplier(self, probability: float) -> float:
        """
        Linear reduction from 1.0 at threshold to min_size_mult at 1.0.
        Below threshold: always 1.0.
        """
        thr = self._high_prob_threshold
        if probability <= thr:
            return 1.0
        # Interpolate: thr→1.0, 1.0→min_size_mult
        slope = (self._min_size_mult - 1.0) / (1.0 - thr)
        mult = 1.0 + slope * (probability - thr)
        return float(max(self._min_size_mult, min(1.0, mult)))

    def _regime_age_hours(self) -> float:
        now = datetime.now(timezone.utc).timestamp()
        return (now - self._regime_start_ts) / 3600.0
