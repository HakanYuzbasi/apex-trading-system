"""
monitoring/exit_optimizer.py
────────────────────────────
Self-Calibrating Exit Optimizer.

Learns from closed-trade history WHEN to hold vs exit a position.
Produces an exit_score in [0, 1] per trade context.  score > threshold →
recommend exit (even if the rule-based exit_manager has not fired yet).

Feature vector
──────────────
  hold_hours       : float  — position age
  regime_code      : int    — 0-5 (bull/neutral/bear/strong_bear/volatile/crisis)
  vix_bucket       : int    — 0-3 (low/normal/high/extreme)
  entry_signal     : float  — signal captured at entry
  signal_decay     : float  — current_signal / entry_signal  (1.0 = no decay)
  pnl_pct          : float  — unrealized P&L as fraction  (e.g. 0.02 = +2%)
  day_of_week      : int    — 0=Mon … 4=Fri
  hour_of_day      : int    — 0-23 UTC

Label
─────
  1  = exit was "good" (closed with pnl_pct > 0 OR avoided further loss)
  0  = exit was "bad"  (closed at loss when holding further would have been better)

We approximate the label from the audit: any completed trade where
  pnl_pct_final >= 0                              → good exit
  pnl_pct_final <  0 AND |pnl_pct_final| < 0.02  → neutral (skip)
  pnl_pct_final <  -0.02                          → bad exit
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_REGIME_CODES: Dict[str, int] = {
    "strong_bull": 0,
    "bull": 1,
    "neutral": 2,
    "bear": 3,
    "strong_bear": 4,
    "volatile": 5,
    "crisis": 5,
}

_DEFAULT_REGIME = 2

_STATE_FILE = "exit_optimizer_state.json"
_MODEL_FILE = "exit_optimizer_model.pkl"

_BAD_PNL_THRESHOLD  = -0.02   # pnl worse than -2% → bad exit
_GOOD_PNL_THRESHOLD =  0.00   # pnl >= 0 → good exit
_MIN_FEATURE_ROWS   = 30      # need at least this many labelled rows before fitting


# ── Feature helpers ────────────────────────────────────────────────────────────

def _regime_code(regime: str) -> int:
    return _REGIME_CODES.get(str(regime).lower(), _DEFAULT_REGIME)


def _vix_bucket(vix: float) -> int:
    if vix < 15:
        return 0  # low
    if vix < 20:
        return 1  # normal
    if vix < 30:
        return 2  # high
    return 3      # extreme


def _make_features(
    hold_hours: float,
    regime: str,
    vix: float,
    entry_signal: float,
    current_signal: float,
    pnl_pct: float,
    day_of_week: int,
    hour_of_day: int,
) -> np.ndarray:
    entry_s = entry_signal if entry_signal != 0.0 else 1e-6
    decay = current_signal / entry_s
    return np.array([
        hold_hours,
        float(_regime_code(regime)),
        float(_vix_bucket(vix)),
        entry_signal,
        decay,
        pnl_pct,
        float(day_of_week),
        float(hour_of_day),
    ], dtype=float)


# ── Audit loading ─────────────────────────────────────────────────────────────

def _load_audit_rows(data_dir: Path, lookback_days: int = 90) -> List[Dict]:
    """Load EXIT rows from trade_audit_*.jsonl files."""
    rows: List[Dict] = []
    cutoff_dt = datetime.now(timezone.utc).timestamp() - lookback_days * 86400
    for user_dir in (data_dir / "users").glob("*/audit"):
        for jf in sorted(user_dir.glob("trade_audit_*.jsonl")):
            try:
                with jf.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Only EXIT events carry a final pnl_pct
                        if rec.get("event_type") != "EXIT":
                            continue
                        ts_raw = rec.get("timestamp", "")
                        try:
                            ts = datetime.fromisoformat(ts_raw).timestamp()
                        except Exception:
                            continue
                        if ts < cutoff_dt:
                            continue
                        rows.append(rec)
            except Exception as exc:
                logger.debug("audit load error %s: %s", jf, exc)
    return rows


def _audit_rows_to_xy(
    rows: List[Dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert audit EXIT rows to (X, y) arrays.  Returns (None, None) if insufficient."""
    Xs: List[np.ndarray] = []
    ys: List[int] = []

    for rec in rows:
        try:
            pnl = float(rec.get("pnl_pct", rec.get("pnl", 0.0)))
        except (TypeError, ValueError):
            continue

        # Label
        if pnl >= _GOOD_PNL_THRESHOLD:
            label = 1
        elif pnl < _BAD_PNL_THRESHOLD:
            label = 0
        else:
            continue  # neutral zone — skip

        try:
            entry_signal  = float(rec.get("entry_signal",  rec.get("signal", 0.0)))
            exit_signal   = float(rec.get("exit_signal",   rec.get("signal", entry_signal)))
            regime        = str(rec.get("regime", "neutral"))
            vix           = float(rec.get("vix", 18.0))
            hold_hours    = float(rec.get("hold_hours",    rec.get("holding_hours", 1.0)))
            ts_raw        = rec.get("timestamp", "")
            ts_dt         = datetime.fromisoformat(ts_raw)
            dow           = ts_dt.weekday()
            hod           = ts_dt.hour
        except Exception:
            continue

        Xs.append(_make_features(hold_hours, regime, vix, entry_signal, exit_signal, pnl, dow, hod))
        ys.append(label)

    if len(Xs) < _MIN_FEATURE_ROWS:
        return None, None

    return np.array(Xs, dtype=float), np.array(ys, dtype=int)


# ── Gradient-boosted classifier (pure-numpy fallback if sklearn unavailable) ──

class _GBModel:
    """Thin sklearn wrapper with pickle-safe serialisation."""

    def __init__(self) -> None:
        self._clf = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            clf.fit(X, y)
            self._clf = clf
            self._fitted = True
        except ImportError:
            logger.warning("sklearn not available — ExitOptimizer using logistic fallback")
            self._fitted = False
            self._logistic_fit(X, y)

    def _logistic_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Simple logistic regression via gradient descent (no sklearn)."""
        n_features = X.shape[1]
        self._w = np.zeros(n_features)
        self._b = 0.0
        lr = 0.01
        X_norm = X / (np.std(X, axis=0) + 1e-9)
        for _ in range(200):
            z = X_norm @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            err = p - y
            self._w -= lr * X_norm.T @ err / len(y)
            self._b -= lr * err.mean()
        self._X_std = np.std(X, axis=0) + 1e-9
        self._fitted = True

    def predict_proba_pos(self, x: np.ndarray) -> float:
        """Return P(exit is 'good') for a single feature vector."""
        if not self._fitted:
            return 0.5
        if self._clf is not None:
            return float(self._clf.predict_proba(x.reshape(1, -1))[0, 1])
        # Logistic fallback
        x_norm = x / self._X_std
        z = float(x_norm @ self._w + self._b)
        return float(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z)))))


# ── Main class ─────────────────────────────────────────────────────────────────

class ExitOptimizer:
    """
    Self-Calibrating Exit Optimizer.

    Usage
    ─────
    score = optimizer.get_exit_score(symbol, regime, hold_hours,
                                     unrealized_pnl_pct, entry_signal,
                                     current_signal, vix)
    # score > EXIT_OPTIMIZER_SCORE_THRESHOLD → recommend exit

    # at trade close:
    optimizer.record_outcome(symbol, regime, hold_hours, entry_signal,
                             exit_signal, pnl_pct, exit_reason)

    # weekly refit:
    optimizer.fit()
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        score_threshold: float = 0.65,
        min_samples: int = 30,
    ) -> None:
        self._data_dir     = Path(data_dir)
        self._state_path   = self._data_dir / _STATE_FILE
        self._model_path   = self._data_dir / _MODEL_FILE
        self._score_threshold = score_threshold
        self._min_samples     = min_samples

        self._model       : Optional[_GBModel] = None
        self._fitted      : bool               = False
        self._n_train     : int                = 0
        self._last_fit_ts : float              = 0.0

        # In-memory buffer: recent closed trades awaiting next refit
        self._buffer: List[Dict[str, Any]] = []

        self._load_state()

    # ── Inference ──────────────────────────────────────────────────────────────

    def get_exit_score(
        self,
        symbol: str,
        regime: str,
        hold_hours: float,
        unrealized_pnl_pct: float,
        entry_signal: float,
        current_signal: float,
        vix: float,
    ) -> float:
        """
        Return exit recommendation score in [0, 1].
        0.5 = uncertain (model not fitted or feature error).
        >threshold → recommend exit.
        """
        if not self._fitted or self._model is None:
            return 0.5

        now = datetime.now(timezone.utc)
        try:
            feat = _make_features(
                hold_hours, regime, vix,
                entry_signal, current_signal,
                unrealized_pnl_pct,
                now.weekday(), now.hour,
            )
            return self._model.predict_proba_pos(feat)
        except Exception as exc:
            logger.debug("ExitOptimizer.get_exit_score error: %s", exc)
            return 0.5

    def should_exit(
        self,
        symbol: str,
        regime: str,
        hold_hours: float,
        unrealized_pnl_pct: float,
        entry_signal: float,
        current_signal: float,
        vix: float,
    ) -> Tuple[bool, str, float]:
        """
        Returns (recommend_exit, reason, score).
        Convenience wrapper around get_exit_score.
        """
        score = self.get_exit_score(
            symbol, regime, hold_hours, unrealized_pnl_pct,
            entry_signal, current_signal, vix,
        )
        if score > self._score_threshold:
            return True, f"exit_optimizer_score={score:.3f}", score
        return False, "", score

    # ── Training data accumulation ─────────────────────────────────────────────

    def record_outcome(
        self,
        symbol: str,
        regime: str,
        hold_hours: float,
        entry_signal: float,
        exit_signal: float,
        pnl_pct: float,
        exit_reason: str = "",
        vix: float = 18.0,
    ) -> None:
        """Buffer a closed trade for the next refit."""
        now = datetime.now(timezone.utc)
        self._buffer.append({
            "event_type": "EXIT",
            "symbol": symbol,
            "regime": regime,
            "hold_hours": hold_hours,
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "vix": vix,
            "timestamp": now.isoformat(),
        })
        # Keep buffer bounded
        if len(self._buffer) > 2000:
            self._buffer = self._buffer[-2000:]

    # ── Model fitting ──────────────────────────────────────────────────────────

    def fit(self, lookback_days: int = 90) -> bool:
        """
        Reload audit + buffer, fit gradient boosting model.
        Returns True if fitting succeeded.
        """
        # Combine audit rows and in-memory buffer
        audit_rows = _load_audit_rows(self._data_dir, lookback_days)
        all_rows   = audit_rows + self._buffer

        X, y = _audit_rows_to_xy(all_rows)
        if X is None:
            logger.info(
                "ExitOptimizer: insufficient labelled samples (%d rows, need %d)",
                len(all_rows), _MIN_FEATURE_ROWS,
            )
            return False

        model = _GBModel()
        try:
            model.fit(X, y)
        except Exception as exc:
            logger.warning("ExitOptimizer.fit failed: %s", exc)
            return False

        self._model       = model
        self._fitted      = True
        self._n_train     = len(X)
        self._last_fit_ts = datetime.now(timezone.utc).timestamp()

        self._save_state()
        logger.info(
            "ExitOptimizer fitted: n_train=%d  last_fit=%s",
            self._n_train,
            datetime.fromtimestamp(self._last_fit_ts, tz=timezone.utc).isoformat(),
        )
        return True

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "fitted":       self._fitted,
            "n_train":      self._n_train,
            "last_fit_ts":  self._last_fit_ts,
            "buffer_len":   len(self._buffer),
        }
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.warning("ExitOptimizer state save failed: %s", exc)

        if self._model is not None:
            try:
                tmp = self._model_path.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    pickle.dump(self._model, fh, protocol=4)
                tmp.replace(self._model_path)
            except Exception as exc:
                logger.warning("ExitOptimizer model save failed: %s", exc)

    def _load_state(self) -> None:
        # Load metadata
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text())
                self._n_train     = int(state.get("n_train", 0))
                self._last_fit_ts = float(state.get("last_fit_ts", 0.0))
            except Exception as exc:
                logger.debug("ExitOptimizer state load error: %s", exc)

        # Load model
        if self._model_path.exists():
            try:
                with self._model_path.open("rb") as fh:
                    self._model = pickle.load(fh)
                self._fitted = True
                logger.info(
                    "ExitOptimizer model loaded (n_train=%d  last_fit=%s)",
                    self._n_train,
                    datetime.fromtimestamp(self._last_fit_ts, tz=timezone.utc).isoformat()
                    if self._last_fit_ts else "never",
                )
            except Exception as exc:
                logger.warning("ExitOptimizer model load failed: %s", exc)
                self._model  = None
                self._fitted = False

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted":           self._fitted,
            "n_train":          self._n_train,
            "buffer_size":      len(self._buffer),
            "last_fit_ts":      self._last_fit_ts,
            "score_threshold":  self._score_threshold,
            "model_type":       type(self._model._clf).__name__
                                if (self._model and self._model._clf is not None)
                                else ("logistic_fallback" if self._fitted else "none"),
        }
