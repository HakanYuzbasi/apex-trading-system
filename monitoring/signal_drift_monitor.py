"""
monitoring/signal_drift_monitor.py — Signal Accuracy Drift Detector

Monitors rolling win-rate of signal-driven trades against a 30-day baseline.
When accuracy drops by more than a configurable threshold, emits a drift alert
so the engine can trigger an async model retrain.

Algorithm:
  1. Maintains a ring-buffer of recent trade outcomes (direction correct or not).
  2. Computes rolling win-rate over the last N trades (short window).
  3. Computes baseline win-rate over the prior M trades (long window).
  4. Emits drift alert when: (baseline - rolling) > threshold  AND  min_trades reached.
  5. Drift clears automatically once rolling win-rate recovers within tolerance.

Integration (execution_loop.py, at trade close):
    self._signal_drift_monitor.record_outcome(symbol, signal_was_correct)
    if self._signal_drift_monitor.is_drifting():
        asyncio.ensure_future(self._retrain_god_level_signal())
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional


# ── Defaults ────────────────────────────────────────────────────────────────────
_SHORT_WINDOW  = 20    # rolling accuracy window
_LONG_WINDOW   = 100   # baseline window
_DRIFT_THRESHOLD = 0.10  # 10% accuracy drop triggers alert
_MIN_TRADES    = 10    # min trades before drift detection activates
_RECOVER_TOLERANCE = 0.04  # alert clears when rolling is within 4% of baseline


@dataclass
class DriftState:
    """Snapshot of current drift detection state."""
    is_drifting: bool
    rolling_win_rate: float         # last SHORT_WINDOW trades
    baseline_win_rate: float        # prior LONG_WINDOW trades
    accuracy_drop: float            # baseline - rolling (positive = drop)
    total_trades: int
    drift_since: Optional[str]      # ISO timestamp when drift started (None if clear)
    last_evaluated: str

    def to_dict(self) -> dict:
        return {
            "is_drifting": self.is_drifting,
            "rolling_win_rate": round(self.rolling_win_rate, 4),
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "accuracy_drop": round(self.accuracy_drop, 4),
            "total_trades": self.total_trades,
            "drift_since": self.drift_since,
            "last_evaluated": self.last_evaluated,
        }


class SignalDriftMonitor:
    """
    Lightweight rolling accuracy monitor for live signal drift detection.

    Thread-safe for reading (get_state).
    record_outcome() called synchronously at every trade close — non-blocking.
    """

    def __init__(
        self,
        short_window: int = _SHORT_WINDOW,
        long_window: int = _LONG_WINDOW,
        drift_threshold: float = _DRIFT_THRESHOLD,
        min_trades: int = _MIN_TRADES,
        recover_tolerance: float = _RECOVER_TOLERANCE,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.drift_threshold = drift_threshold
        self.min_trades = min_trades
        self.recover_tolerance = recover_tolerance

        # Ring buffer: True = signal correct, False = signal wrong
        self._outcomes: Deque[bool] = deque(maxlen=long_window)
        self._total: int = 0
        self._is_drifting: bool = False
        self._drift_since: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_outcome(self, symbol: str, signal_correct: bool) -> None:
        """Record a single trade outcome (signal direction correct or not)."""
        self._outcomes.append(signal_correct)
        self._total += 1
        self._evaluate()

    def is_drifting(self) -> bool:
        """Return True when rolling accuracy has dropped significantly."""
        return self._is_drifting

    def get_state(self) -> DriftState:
        """Return current drift detection state as a dataclass."""
        rolling = self._rolling_win_rate()
        baseline = self._baseline_win_rate()
        drop = max(0.0, baseline - rolling)
        return DriftState(
            is_drifting=self._is_drifting,
            rolling_win_rate=rolling,
            baseline_win_rate=baseline,
            accuracy_drop=drop,
            total_trades=self._total,
            drift_since=self._drift_since,
            last_evaluated=datetime.now(timezone.utc).isoformat(),
        )

    def reset(self) -> None:
        """Clear all history (e.g. after model retrain)."""
        self._outcomes.clear()
        self._total = 0
        self._is_drifting = False
        self._drift_since = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _rolling_win_rate(self) -> float:
        """Win-rate of last SHORT_WINDOW outcomes."""
        n = min(self.short_window, len(self._outcomes))
        if n == 0:
            return 0.0
        recent = list(self._outcomes)[-n:]
        return sum(recent) / n

    def _baseline_win_rate(self) -> float:
        """Win-rate of ALL stored outcomes (up to LONG_WINDOW)."""
        n = len(self._outcomes)
        if n == 0:
            return 0.0
        return sum(self._outcomes) / n

    def _evaluate(self) -> None:
        """Re-evaluate drift state after a new outcome is recorded."""
        if self._total < self.min_trades:
            return

        rolling = self._rolling_win_rate()
        baseline = self._baseline_win_rate()
        drop = baseline - rolling

        if not self._is_drifting:
            if drop >= self.drift_threshold:
                self._is_drifting = True
                self._drift_since = datetime.now(timezone.utc).isoformat()
        else:
            # Clear drift once rolling recovers within tolerance of baseline
            if drop <= self.recover_tolerance:
                self._is_drifting = False
                self._drift_since = None
