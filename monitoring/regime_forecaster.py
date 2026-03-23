"""
monitoring/regime_forecaster.py — Proactive Regime Transition Forecaster

Detects an impending regime shift 1-2 bars BEFORE it's confirmed by the
main VIXRegimeManager, using leading indicator momentum:

  1. VIX term structure slope (VIX3M - VIX spot): inversion → fear rising
  2. Put-call ratio velocity (rate of PCR acceleration)
  3. Credit spread momentum (HYG price ROC as proxy for IG spreads)
  4. Advance/Decline breadth divergence (SPY vs A/D momentum)
  5. VIX Z-score momentum (z rising fast even if not in PANIC yet)

Outputs:
  - transition_probability: float [0, 1] — prob a bearish shift is imminent
  - size_multiplier: float [0.40, 1.0] — sizing dampener to apply
  - signal: str — "warning" | "caution" | "clear"
  - features: dict of raw indicator values for dashboard

The multiplier is applied ADDITIVELY to the VIX multiplier (min of the two
wins), so it only ever tightens, never loosens, existing risk controls.
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_WARN_PROB = 0.60     # → "warning":  size_mult = 0.65
_CAUTION_PROB = 0.40  # → "caution":  size_mult = 0.82
_WINDOW = 20          # rolling window for z-scores and momentum


class RegimeTransitionForecast:
    __slots__ = ("transition_prob", "size_multiplier", "signal", "features", "timestamp")

    def __init__(self, prob: float, mult: float, signal: str, features: dict):
        self.transition_prob = round(prob, 4)
        self.size_multiplier = round(mult, 4)
        self.signal = signal
        self.features = features
        self.timestamp = datetime.utcnow().isoformat() + "Z"


class RegimeTransitionForecaster:
    """
    Proactive regime-transition detector.

    Wire in two steps:
      1. forecaster.update(vix, pcr, hyg_price, spy_price, vix3m)  — each cycle
      2. forecast = forecaster.get_forecast()                        — read multiplier
    """

    def __init__(
        self,
        window: int = _WINDOW,
        warn_prob: float = _WARN_PROB,
        caution_prob: float = _CAUTION_PROB,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._window = window
        self._warn_prob = warn_prob
        self._caution_prob = caution_prob
        self._data_dir = Path(data_dir) if data_dir else None

        # Rolling history
        self._vix_hist: deque = deque(maxlen=window)
        self._pcr_hist: deque = deque(maxlen=window)
        self._hyg_hist: deque = deque(maxlen=window)
        self._spy_hist: deque = deque(maxlen=window)
        self._vix3m_hist: deque = deque(maxlen=window)

        self._last_forecast: Optional[RegimeTransitionForecast] = None
        self._load_state()

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        vix: float,
        pcr: float = 1.0,
        hyg_price: float = 80.0,
        spy_price: float = 500.0,
        vix3m: float = 0.0,
    ) -> None:
        """Record one observation. Call once per main-loop cycle."""
        self._vix_hist.append(float(vix))
        self._pcr_hist.append(float(pcr))
        self._hyg_hist.append(float(hyg_price))
        self._spy_hist.append(float(spy_price))
        self._vix3m_hist.append(float(vix3m) if vix3m > 0 else float(vix))
        self._last_forecast = self._compute()
        self._persist()

    def get_forecast(self) -> RegimeTransitionForecast:
        """Return most recent forecast (neutral/clear if insufficient data)."""
        if self._last_forecast is None:
            return RegimeTransitionForecast(0.0, 1.0, "clear", {})
        return self._last_forecast

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute(self) -> RegimeTransitionForecast:
        n = len(self._vix_hist)
        features: Dict[str, float] = {}

        if n < 5:
            return RegimeTransitionForecast(0.0, 1.0, "clear", features)

        vix_arr = list(self._vix_hist)
        pcr_arr = list(self._pcr_hist)
        hyg_arr = list(self._hyg_hist)
        spy_arr = list(self._spy_hist)
        v3m_arr = list(self._vix3m_hist)

        scores: list[float] = []

        # Use _stress_score: 0 when neutral/calm, positive when stressed.
        # score = max(0, sigmoid(x) - 0.5) * 2  → [0, 1], 0 at neutral

        # 1. VIX Z-score: how many σ above recent mean (stress when z > 1)
        vix_z = self._zscore(vix_arr)
        features["vix_z"] = round(vix_z, 3)
        scores.append(self._stress_score(vix_z - 1.0))

        # 2. VIX momentum: rate of change over last 3 bars
        vix_roc = (vix_arr[-1] - vix_arr[-3]) / (vix_arr[-3] + 1e-9)
        features["vix_roc_3bar"] = round(vix_roc, 4)
        scores.append(self._stress_score(vix_roc * 20))

        # 3. VIX term structure: spot vs 3-month (inversion = backwardation = fear)
        term_slope = (v3m_arr[-1] - vix_arr[-1]) / (vix_arr[-1] + 1e-9)
        features["vix_term_slope"] = round(term_slope, 4)
        scores.append(self._stress_score(-term_slope * 10))  # neg slope = VIX > VIX3M

        # 4. PCR acceleration: put-call ratio rising fast
        if n >= 4:
            pcr_roc = (pcr_arr[-1] - pcr_arr[-4]) / (pcr_arr[-4] + 1e-9)
            features["pcr_roc_4bar"] = round(pcr_roc, 4)
            scores.append(self._stress_score(pcr_roc * 8))
        else:
            features["pcr_roc_4bar"] = 0.0

        # 5. HYG (credit proxy) falling = credit stress
        if n >= 5:
            hyg_roc = (hyg_arr[-1] - hyg_arr[-5]) / (hyg_arr[-5] + 1e-9)
            features["hyg_roc_5bar"] = round(hyg_roc, 4)
            scores.append(self._stress_score(-hyg_roc * 30))
        else:
            features["hyg_roc_5bar"] = 0.0

        # 6. SPY vs VIX divergence: VIX spiking while SPY still positive
        spy_roc = (spy_arr[-1] - spy_arr[-3]) / (spy_arr[-3] + 1e-9) if n >= 4 else 0.0
        features["spy_roc_3bar"] = round(spy_roc, 4)
        divergence = vix_roc + max(0.0, -spy_roc)  # VIX up + SPY down = additive stress
        features["divergence"] = round(divergence, 4)
        scores.append(self._stress_score(divergence * 15))

        # Weighted sum: 0 = completely calm, 1 = maximum stress
        weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        weights = weights[:len(scores)]
        w_sum = sum(weights)
        prob = sum(s * w / w_sum for s, w in zip(scores, weights))
        prob = max(0.0, min(1.0, prob))
        features["raw_prob"] = round(prob, 4)

        # Map prob to signal + multiplier
        if prob >= self._warn_prob:
            signal = "warning"
            mult = 0.65
        elif prob >= self._caution_prob:
            signal = "caution"
            mult = 0.82
        else:
            signal = "clear"
            mult = 1.0

        features["n_obs"] = n
        return RegimeTransitionForecast(prob, mult, signal, features)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _stress_score(x: float) -> float:
        """Map a raw stress signal to [0, 1]; returns 0 for neutral/calm x."""
        s = 1.0 / (1.0 + math.exp(-x))
        return max(0.0, (s - 0.5) * 2.0)

    @staticmethod
    def _zscore(series: list) -> float:
        n = len(series)
        if n < 2:
            return 0.0
        mean = sum(series) / n
        variance = sum((x - mean) ** 2 for x in series) / (n - 1)
        std = math.sqrt(variance)
        return (series[-1] - mean) / (std + 1e-9)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if not self._data_dir or not self._last_forecast:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / "regime_forecaster.json"
            payload = {
                "last_forecast": {
                    "transition_prob": self._last_forecast.transition_prob,
                    "size_multiplier": self._last_forecast.size_multiplier,
                    "signal": self._last_forecast.signal,
                    "features": self._last_forecast.features,
                    "timestamp": self._last_forecast.timestamp,
                },
                "vix_hist": list(self._vix_hist)[-10:],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.debug("RegimeForecaster persist error: %s", exc)

    def _load_state(self) -> None:
        if not self._data_dir:
            return
        path = Path(self._data_dir) / "regime_forecaster.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            fc = data.get("last_forecast", {})
            if fc:
                self._last_forecast = RegimeTransitionForecast(
                    prob=fc.get("transition_prob", 0.0),
                    mult=fc.get("size_multiplier", 1.0),
                    signal=fc.get("signal", "clear"),
                    features=fc.get("features", {}),
                )
            # Restore last known VIX history for continuity
            for v in data.get("vix_hist", []):
                self._vix_hist.append(float(v))
        except Exception as exc:
            logger.debug("RegimeForecaster load error: %s", exc)
