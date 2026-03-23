"""
monitoring/model_drift_monitor.py — Model Drift Monitor & Auto-Retrain Trigger

Tracks predictive performance decay in real time and fires a retrain recommendation
when degradation is confirmed across multiple evaluation windows.

Three independent decay signals:

  1. IC (Information Coefficient) — Pearson correlation between signal value and
     the actual next-bar return. Measures whether the model retains *direction*
     predictive power.  IC > 0.05 → healthy, IC < 0 → critical.

  2. Hit Rate — fraction of completed trades where sign(signal) matched
     sign(actual_return).  < 50% for two consecutive windows triggers "degrading".

  3. Confidence distribution — rolling median confidence emitted by the signal
     generator. A steadily declining median indicates the model is hedging its
     bets (less certainty about its own predictions).

The monitor accumulates observations window-by-window (default 30 trades per
window). After each full window a `DriftStatus` snapshot is produced and
persisted to disk. When `consecutive_degraded_windows` reaches the configured
threshold (default 2) and IC is below `ic_retrain_threshold`, a retraining
recommendation is issued at severity "critical".

Wire-in (execution_loop.py):
    from monitoring.model_drift_monitor import ModelDriftMonitor
    _drift_monitor = ModelDriftMonitor(data_dir=ApexConfig.DATA_DIR)

    # At trade entry (after signal generation):
    _drift_monitor.record_signal(symbol, signal_value, confidence)

    # At trade exit (or when 1-bar forward return is available):
    _drift_monitor.record_outcome(symbol, signal_value, actual_return_1bar)

    # Each cycle (for dashboard / periodic check):
    status = _drift_monitor.get_status()
    if status.should_retrain:
        # trigger retraining pipeline
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_WINDOW_SIZE = 30           # observations per evaluation window
_IC_HEALTHY = 0.05          # IC above this → healthy
_IC_DEGRADED = 0.02         # IC below this → degrading
_IC_CRITICAL = -0.01        # IC below this → critical (negative predictive power)
_HIT_RATE_HEALTHY = 0.55    # hit rate above this → healthy
_HIT_RATE_DEGRADED = 0.50   # hit rate below this → degrading
_CONF_HEALTHY = 0.60        # median confidence above this → healthy
_CONF_DEGRADED = 0.50       # median confidence below this → degrading
_CONSECUTIVE_DEGRADE = 2    # windows in a row before retrain recommendation
_IC_RETRAIN_THRESHOLD = 0.01  # IC below this AND consecutive → retrain


@dataclass
class WindowStats:
    """Statistics for a completed evaluation window."""
    window_id: int
    n_obs: int
    ic: float           # information coefficient
    hit_rate: float     # fraction directionally correct
    med_confidence: float
    health: str         # "healthy" | "degrading" | "critical"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class DriftStatus:
    """Current drift monitor status."""
    health: str                 # "healthy" | "degrading" | "critical"
    should_retrain: bool
    ic_current: float
    ic_trend: float             # IC delta vs previous window (negative = decaying)
    hit_rate_current: float
    med_confidence: float
    consecutive_degraded: int
    total_windows: int
    pending_obs: int            # observations in current incomplete window
    last_updated: str
    window_history: List[WindowStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "health": self.health,
            "should_retrain": self.should_retrain,
            "ic_current": round(self.ic_current, 4),
            "ic_trend": round(self.ic_trend, 4),
            "hit_rate_current": round(self.hit_rate_current, 4),
            "med_confidence": round(self.med_confidence, 4),
            "consecutive_degraded": self.consecutive_degraded,
            "total_windows": self.total_windows,
            "pending_obs": self.pending_obs,
            "last_updated": self.last_updated,
        }


class ModelDriftMonitor:
    """
    Real-time model predictive quality monitor.

    Call `record_signal()` when a signal is generated, and `record_outcome()`
    when the 1-bar forward return is known. The monitor will automatically
    evaluate each full window and update its health status.
    """

    def __init__(
        self,
        window_size: int = _WINDOW_SIZE,
        ic_retrain_threshold: float = _IC_RETRAIN_THRESHOLD,
        consecutive_degrade_limit: int = _CONSECUTIVE_DEGRADE,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._window_size = window_size
        self._ic_retrain_threshold = ic_retrain_threshold
        self._consecutive_degrade_limit = consecutive_degrade_limit
        self._data_dir = Path(data_dir) if data_dir else None

        # Pending observations (signal awaiting matched outcome)
        self._pending: Dict[str, Tuple[float, float]] = {}  # symbol → (signal, confidence)

        # Completed (signal, return) pairs for the current window
        self._window_obs: List[Tuple[float, float, float]] = []  # (signal, return, confidence)

        # Completed windows
        self._windows: Deque[WindowStats] = deque(maxlen=20)

        # State
        self._consecutive_degraded: int = 0
        self._total_windows: int = 0
        self._last_status: Optional[DriftStatus] = None

        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_signal(
        self,
        symbol: str,
        signal_value: float,
        confidence: float,
    ) -> None:
        """Register a new signal. Must be followed by record_outcome()."""
        self._pending[symbol] = (float(signal_value), float(confidence))

    def record_outcome(
        self,
        symbol: str,
        signal_value: float,
        actual_return: float,
    ) -> None:
        """
        Register the actual 1-bar return for a prior signal.

        Args:
            symbol:        Symbol that was traded.
            signal_value:  The signal that was used (sign = predicted direction).
            actual_return: Realised 1-bar return (positive = price went up).
        """
        sig = float(signal_value)
        ret = float(actual_return)
        conf = 0.60  # default if we lost the pending entry
        if symbol in self._pending:
            stored_sig, stored_conf = self._pending.pop(symbol)
            # Use stored signal if consistent in sign
            if sig == 0.0 or math.copysign(1, stored_sig) == math.copysign(1, sig):
                sig = stored_sig
            conf = stored_conf

        if sig == 0.0:
            return  # skip neutral signals — not directionally meaningful

        self._window_obs.append((sig, ret, conf))

        if len(self._window_obs) >= self._window_size:
            self._evaluate_window()

    def get_status(self) -> DriftStatus:
        """Return current drift status."""
        if self._last_status is not None:
            return self._last_status

        # No completed windows yet — return a default healthy status
        return DriftStatus(
            health="healthy",
            should_retrain=False,
            ic_current=0.0,
            ic_trend=0.0,
            hit_rate_current=0.0,
            med_confidence=0.0,
            consecutive_degraded=0,
            total_windows=0,
            pending_obs=len(self._window_obs),
            last_updated=datetime.utcnow().isoformat() + "Z",
        )

    def get_report(self) -> dict:
        """JSON-serialisable status dict for dashboard."""
        s = self.get_status()
        d = s.to_dict()
        d["window_history"] = [
            {
                "window_id": w.window_id,
                "n_obs": w.n_obs,
                "ic": round(w.ic, 4),
                "hit_rate": round(w.hit_rate, 4),
                "med_confidence": round(w.med_confidence, 4),
                "health": w.health,
                "timestamp": w.timestamp,
            }
            for w in self._windows
        ]
        return d

    # ── Core computation ──────────────────────────────────────────────────────

    def _evaluate_window(self) -> None:
        obs = list(self._window_obs)
        self._window_obs.clear()
        self._total_windows += 1

        signals = [o[0] for o in obs]
        returns = [o[1] for o in obs]
        confs = [o[2] for o in obs]

        ic = self._pearson_corr(signals, returns)
        hit_rate = sum(
            1 for s, r in zip(signals, returns)
            if s != 0 and math.copysign(1, s) == math.copysign(1, r)
        ) / max(len(obs), 1)

        import statistics
        med_conf = statistics.median(confs)

        health = self._classify_health(ic, hit_rate, med_conf)

        if health in ("degrading", "critical"):
            self._consecutive_degraded += 1
        else:
            self._consecutive_degraded = 0

        win = WindowStats(
            window_id=self._total_windows,
            n_obs=len(obs),
            ic=ic,
            hit_rate=hit_rate,
            med_confidence=med_conf,
            health=health,
        )
        self._windows.append(win)

        # IC trend vs previous window
        ic_trend = 0.0
        if len(self._windows) >= 2:
            ic_trend = ic - self._windows[-2].ic

        should_retrain = (
            self._consecutive_degraded >= self._consecutive_degrade_limit
            and ic < self._ic_retrain_threshold
        )

        self._last_status = DriftStatus(
            health=health,
            should_retrain=should_retrain,
            ic_current=ic,
            ic_trend=ic_trend,
            hit_rate_current=hit_rate,
            med_confidence=med_conf,
            consecutive_degraded=self._consecutive_degraded,
            total_windows=self._total_windows,
            pending_obs=0,
            last_updated=datetime.utcnow().isoformat() + "Z",
            window_history=list(self._windows),
        )

        level = "CRITICAL" if should_retrain else health.upper()
        logger.info(
            "ModelDriftMonitor window %d: IC=%.3f HR=%.2f%% conf=%.2f → %s%s",
            self._total_windows, ic, hit_rate * 100, med_conf, health,
            " ⚠️ RETRAIN RECOMMENDED" if should_retrain else "",
        )

        self._persist()

    @staticmethod
    def _classify_health(ic: float, hit_rate: float, med_conf: float) -> str:
        """Classify window health from three signals (majority vote)."""
        scores = []
        # IC vote
        if ic >= _IC_HEALTHY:
            scores.append("healthy")
        elif ic < _IC_CRITICAL:
            scores.append("critical")
        else:
            scores.append("degrading")
        # Hit rate vote
        if hit_rate >= _HIT_RATE_HEALTHY:
            scores.append("healthy")
        elif hit_rate < _HIT_RATE_DEGRADED:
            scores.append("degrading")
        else:
            scores.append("healthy")   # borderline — give benefit of doubt
        # Confidence vote
        if med_conf >= _CONF_HEALTHY:
            scores.append("healthy")
        elif med_conf < _CONF_DEGRADED:
            scores.append("degrading")
        else:
            scores.append("healthy")

        if scores.count("critical") >= 1:
            return "critical"
        if scores.count("degrading") >= 2:
            return "degrading"
        return "healthy"

    @staticmethod
    def _pearson_corr(x: List[float], y: List[float]) -> float:
        """Pearson correlation; returns 0.0 on degenerate input."""
        n = len(x)
        if n < 3:
            return 0.0
        mx, my = sum(x) / n, sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if dx < 1e-12 or dy < 1e-12:
            return 0.0
        return float(max(-1.0, min(1.0, num / (dx * dy))))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if not self._data_dir or not self._last_status:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / "model_drift_monitor.json"
            path.write_text(json.dumps(self.get_report(), indent=2))
        except Exception as exc:
            logger.debug("ModelDriftMonitor persist error: %s", exc)

    def _load_state(self) -> None:
        if not self._data_dir:
            return
        path = Path(self._data_dir) / "model_drift_monitor.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._total_windows = data.get("total_windows", 0)
            self._consecutive_degraded = data.get("consecutive_degraded", 0)
            # Restore last status fields into a DriftStatus
            self._last_status = DriftStatus(
                health=data.get("health", "healthy"),
                should_retrain=data.get("should_retrain", False),
                ic_current=data.get("ic_current", 0.0),
                ic_trend=data.get("ic_trend", 0.0),
                hit_rate_current=data.get("hit_rate_current", 0.0),
                med_confidence=data.get("med_confidence", 0.0),
                consecutive_degraded=self._consecutive_degraded,
                total_windows=self._total_windows,
                pending_obs=0,
                last_updated=data.get("last_updated", ""),
            )
        except Exception as exc:
            logger.debug("ModelDriftMonitor load error: %s", exc)
