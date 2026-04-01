"""
monitoring/hmm_regime_detector.py — HMM-Based Market Regime Detector

Replaces hard-coded VIX thresholds with a learned Hidden Markov Model that
discovers latent market states from price returns + volatility features.

4 hidden states are auto-labelled after fitting:
  • bull      — low vol, positive mean return
  • neutral   — low vol, near-zero mean return
  • bear      — high vol, negative mean return
  • volatile  — very high vol (infrequent spike state)

Usage:
    from monitoring.hmm_regime_detector import HMMRegimeDetector
    det = HMMRegimeDetector(state_dir=Path("data/hmm_regime"))
    det.fit(spy_daily_returns, vix_series)        # weekly retrain
    label, conf, probs = det.classify(spy_daily_returns[-20:], vix_series[-20:])
    # label: "bull" | "neutral" | "bear" | "volatile"
    # conf:  posterior probability of current state [0, 1]
    # probs: {label: prob} full posterior
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_STATES = 4
MIN_FIT_SAMPLES = 60          # Min observations before training
DEFAULT_RETRAIN_INTERVAL = 7 * 86400   # 1 week in seconds
_FALLBACK_REGIME = "neutral"

_STATE_LABELS = ("bull", "neutral", "bear", "volatile")


@dataclass
class RegimeState:
    """Output of HMMRegimeDetector.classify()."""
    label: str                          # "bull" | "neutral" | "bear" | "volatile"
    confidence: float                   # posterior prob of current state
    state_probs: Dict[str, float]       # full posterior
    viterbi_path: List[str] = field(default_factory=list)   # last N labels
    trained_at: float = 0.0
    n_train_samples: int = 0
    method: str = "hmm"


class HMMRegimeDetector:
    """
    Gaussian HMM (4 states) trained on SPY daily returns + VIX features.

    Falls back gracefully to VIX-threshold labelling when:
      - hmmlearn is not installed
      - training data is insufficient
      - the model is not yet trained
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        n_states: int = N_STATES,
        min_fit_samples: int = MIN_FIT_SAMPLES,
        retrain_interval_s: float = DEFAULT_RETRAIN_INTERVAL,
    ):
        self._n_states = n_states
        self._min_fit_samples = min_fit_samples
        self._retrain_interval = retrain_interval_s

        self._model = None          # GaussianHMM instance
        self._state_map: Dict[int, str] = {}    # HMM state idx → label
        self._trained_at: float = 0.0
        self._n_train: int = 0
        self._last_path: List[str] = []

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Public API ─────────────────────────────────────────────────────────

    def should_retrain(self) -> bool:
        """True when the retrain interval has elapsed."""
        return time.time() - self._trained_at >= self._retrain_interval

    def fit(self, spy_returns: List[float], vix_series: List[float]) -> bool:
        """
        Fit the 4-state Gaussian HMM on (returns, vix) features.

        Returns True on success, False if hmmlearn unavailable or insufficient data.
        """
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore
        except ImportError:
            logger.debug("HMMRegimeDetector: hmmlearn not installed — using VIX fallback")
            return False

        if len(spy_returns) < self._min_fit_samples:
            logger.debug("HMMRegimeDetector: too few samples (%d < %d)", len(spy_returns), self._min_fit_samples)
            return False

        try:
            X = self._build_features(spy_returns, vix_series)
            if X is None or len(X) < self._min_fit_samples:
                return False

            model = GaussianHMM(
                n_components=self._n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            model.fit(X)

            # Label states by return mean + volatility
            state_means = model.means_[:, 0]  # feature 0 = daily return
            state_vols  = np.sqrt(model.covars_[:, 0])  # std of return feature
            self._state_map = self._label_states(state_means, state_vols)
            self._model = model
            self._trained_at = time.time()
            self._n_train = len(X)
            logger.info(
                "HMMRegimeDetector: trained on %d samples, state_map=%s",
                self._n_train, self._state_map,
            )
            self._save()
            return True
        except Exception as exc:
            logger.warning("HMMRegimeDetector.fit() failed: %s", exc)
            return False

    def classify(
        self,
        spy_returns: List[float],
        vix_series: List[float],
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify the current market regime.

        Returns (label, confidence, state_probs).
        Falls back to VIX threshold if model unavailable.
        """
        if self._model is None or not self._state_map:
            return self._vix_fallback(vix_series)

        try:
            X = self._build_features(spy_returns, vix_series)
            if X is None or len(X) < 2:
                return self._vix_fallback(vix_series)

            # Posterior state probabilities for the last observation
            _, posteriors = self._model.score_samples(X)
            last_posteriors = posteriors[-1]   # shape (n_states,)

            # Map to labels
            probs: Dict[str, float] = {lbl: 0.0 for lbl in _STATE_LABELS}
            for state_idx, prob in enumerate(last_posteriors):
                label = self._state_map.get(state_idx, "neutral")
                probs[label] = probs.get(label, 0.0) + float(prob)

            # Viterbi path for context
            states_seq = self._model.predict(X)
            self._last_path = [self._state_map.get(int(s), "neutral") for s in states_seq[-10:]]

            # Best label and its confidence
            best_label = max(probs, key=lambda k: probs[k])
            confidence = float(probs[best_label])
            return best_label, confidence, probs

        except Exception as exc:
            logger.debug("HMMRegimeDetector.classify() failed: %s — using VIX fallback", exc)
            return self._vix_fallback(vix_series)

    def get_state(self) -> RegimeState:
        """Return a snapshot of the detector state for API/dashboard."""
        if self._model is None:
            return RegimeState(
                label=_FALLBACK_REGIME,
                confidence=0.0,
                state_probs={lbl: 0.25 for lbl in _STATE_LABELS},
                method="vix_fallback",
            )
        return RegimeState(
            label=self._last_path[-1] if self._last_path else _FALLBACK_REGIME,
            confidence=0.0,
            state_probs={lbl: 0.25 for lbl in _STATE_LABELS},
            viterbi_path=list(self._last_path),
            trained_at=self._trained_at,
            n_train_samples=self._n_train,
            method="hmm",
        )

    def get_snapshot(self) -> Dict:
        """Return serialisable snapshot for REST endpoint."""
        s = self.get_state()
        return {
            "available": self._model is not None,
            "method": s.method,
            "current_label": s.label,
            "confidence": round(s.confidence, 4),
            "state_probs": {k: round(v, 4) for k, v in s.state_probs.items()},
            "viterbi_path": s.viterbi_path,
            "trained_at": s.trained_at,
            "n_train_samples": s.n_train_samples,
            "state_map": {str(k): v for k, v in self._state_map.items()},
            "n_states": self._n_states,
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_features(
        spy_returns: List[float],
        vix_series: List[float],
    ) -> Optional[np.ndarray]:
        """
        Build feature matrix for HMM.

        Features per observation:
          0 — daily return (scaled ×100)
          1 — VIX normalised to [0, 1] by dividing by 80
          2 — 5-day rolling return std (vol proxy, scaled ×100)
        """
        ret = np.array([float(r) for r in spy_returns if np.isfinite(r)])
        vix = np.array([float(v) for v in vix_series if np.isfinite(v)])

        n = min(len(ret), len(vix))
        if n < 10:
            return None

        ret = ret[-n:]
        vix = vix[-n:]

        # Rolling 5-day std
        rol_std = np.zeros(n)
        for i in range(n):
            window = ret[max(0, i - 4): i + 1]
            rol_std[i] = float(np.std(window)) if len(window) > 1 else 0.0

        X = np.column_stack([
            ret * 100.0,
            np.clip(vix / 80.0, 0.0, 1.0),
            rol_std * 100.0,
        ])
        return X.astype(np.float64)

    @staticmethod
    def _label_states(
        state_means: np.ndarray,
        state_vols: np.ndarray,
    ) -> Dict[int, str]:
        """
        Auto-label HMM states using return mean + volatility heuristics:
          - Highest vol → "volatile"
          - Of remaining 3: best return → "bull", worst → "bear", middle → "neutral"
        """
        n = len(state_means)
        label_map: Dict[int, str] = {}

        # Highest vol state = volatile
        vol_idx = int(np.argmax(state_vols))
        label_map[vol_idx] = "volatile"
        remaining = [i for i in range(n) if i != vol_idx]

        # Sort remaining by mean return descending
        remaining.sort(key=lambda i: -state_means[i])
        if len(remaining) >= 1:
            label_map[remaining[0]] = "bull"
        if len(remaining) >= 2:
            label_map[remaining[-1]] = "bear"
        if len(remaining) == 3:
            label_map[remaining[1]] = "neutral"
        # For <3 remaining states (e.g. n_states=2 or 3), fill defaults
        for i in range(n):
            if i not in label_map:
                label_map[i] = "neutral"
        return label_map

    def _vix_fallback(self, vix_series: List[float]) -> Tuple[str, float, Dict[str, float]]:
        """Simple VIX threshold fallback when HMM is unavailable."""
        vix = float(vix_series[-1]) if vix_series else 20.0
        if vix < 15:
            label, conf = "bull", 0.70
        elif vix < 25:
            label, conf = "neutral", 0.65
        elif vix < 35:
            label, conf = "bear", 0.65
        else:
            label, conf = "volatile", 0.75
        probs = {lbl: (conf if lbl == label else (1 - conf) / 3) for lbl in _STATE_LABELS}
        return label, conf, probs

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None or self._model is None:
            return
        try:
            import pickle
            p = self._state_dir / "hmm_model.pkl"
            tmp = p.with_suffix(".pkl.tmp")
            tmp.write_bytes(pickle.dumps(self._model))
            tmp.replace(p)

            meta = {
                "state_map": {str(k): v for k, v in self._state_map.items()},
                "trained_at": self._trained_at,
                "n_train": self._n_train,
                "n_states": self._n_states,
            }
            mp = self._state_dir / "hmm_meta.json"
            mp_tmp = mp.with_suffix(".json.tmp")
            mp_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            mp_tmp.replace(mp)
        except Exception as exc:
            logger.debug("HMMRegimeDetector: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            import pickle
            mp = self._state_dir / "hmm_meta.json"
            pp = self._state_dir / "hmm_model.pkl"
            if not mp.exists() or not pp.exists():
                return
            meta = json.loads(mp.read_text(encoding="utf-8"))
            self._state_map = {int(k): v for k, v in meta.get("state_map", {}).items()}
            self._trained_at = float(meta.get("trained_at", 0))
            self._n_train = int(meta.get("n_train", 0))
            self._model = pickle.loads(pp.read_bytes())
            logger.info(
                "HMMRegimeDetector: loaded model (trained_at=%s, n_train=%d)",
                time.strftime("%Y-%m-%d %H:%M", time.localtime(self._trained_at)),
                self._n_train,
            )
        except Exception as exc:
            logger.debug("HMMRegimeDetector: load failed: %s", exc)
