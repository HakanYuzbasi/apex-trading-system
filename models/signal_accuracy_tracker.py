"""
models/signal_accuracy_tracker.py — Live Signal Accuracy Comparison
=====================================================================
Purely observational: records actual price outcomes and compares
directional accuracy of the regression signal vs the binary classifier.

Does NOT gate trades.  Used for logging + the comparison report script.

Usage:
    tracker = SignalAccuracyTracker()
    # At entry:
    tracker.record_prediction(symbol, timestamp, regression_signal, binary_signal, entry_price)
    # At exit:
    tracker.record_outcome(symbol, exit_price)
    # Report:
    summary = tracker.get_comparison_summary()
"""

import json
import os
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 50  # rolling predictions to track


class SignalAccuracyTracker:
    """
    Tracks regression vs binary signal directional accuracy on a rolling window.
    """

    def __init__(
        self,
        window: int = _DEFAULT_WINDOW,
        state_path: str = "data/signal_accuracy_state.json",
    ):
        self._window = window
        self._state_path = state_path

        # Pending predictions keyed by symbol
        self._pending: Dict[str, dict] = {}

        # Rolling deque of resolved records:
        # {regression_correct: bool, binary_correct: bool, regime: str}
        self._resolved: deque = deque(maxlen=window)

        self._load_state()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        symbol: str,
        timestamp: datetime,
        regression_signal: float,
        binary_signal: float,
        entry_price: float,
        regime: str = "unknown",
    ) -> None:
        """Store a pending prediction until the outcome is known."""
        self._pending[symbol] = {
            "timestamp": timestamp.isoformat(),
            "regression_signal": float(regression_signal),
            "binary_signal": float(binary_signal),
            "entry_price": float(entry_price),
            "regime": regime,
        }

    def record_outcome(self, symbol: str, exit_price: float) -> None:
        """
        Resolve a pending prediction with the actual exit price.
        actual_direction = 1 if exit > entry else 0.
        """
        pending = self._pending.pop(symbol, None)
        if pending is None:
            return

        entry_price = pending["entry_price"]
        if entry_price == 0:
            return

        actual_up = exit_price > entry_price

        reg_signal = pending["regression_signal"]
        bin_signal = pending["binary_signal"]

        regression_correct = (reg_signal > 0) == actual_up
        binary_correct = (bin_signal > 0) == actual_up

        record = {
            "symbol": symbol,
            "timestamp": pending["timestamp"],
            "regime": pending["regime"],
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "regression_signal": reg_signal,
            "binary_signal": bin_signal,
            "regression_correct": regression_correct,
            "binary_correct": binary_correct,
        }
        self._resolved.append(record)

        logger.debug(
            "SignalAccuracyTracker %s: reg=%s bin=%s actual_up=%s",
            symbol,
            "✓" if regression_correct else "✗",
            "✓" if binary_correct else "✗",
            actual_up,
        )

        self._persist_state()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_comparison_summary(self) -> Dict:
        """Return rolling accuracy stats for regression vs binary."""
        n = len(self._resolved)
        if n == 0:
            return {
                "n": 0,
                "regression_accuracy": None,
                "binary_accuracy": None,
                "binary_advantage": None,
            }

        records = list(self._resolved)
        reg_acc = float(np.mean([r["regression_correct"] for r in records]))
        bin_acc = float(np.mean([r["binary_correct"] for r in records]))

        # Per-regime breakdown
        regimes: Dict[str, Dict] = {}
        for r in records:
            reg = r["regime"]
            if reg not in regimes:
                regimes[reg] = {"reg": [], "bin": []}
            regimes[reg]["reg"].append(r["regression_correct"])
            regimes[reg]["bin"].append(r["binary_correct"])

        regime_summary = {}
        for reg, data in regimes.items():
            regime_summary[reg] = {
                "n": len(data["reg"]),
                "regression_accuracy": float(np.mean(data["reg"])),
                "binary_accuracy": float(np.mean(data["bin"])),
            }

        return {
            "n": n,
            "window": self._window,
            "regression_accuracy": reg_acc,
            "binary_accuracy": bin_acc,
            "binary_advantage": round(bin_acc - reg_acc, 4),
            "recommended_primary": "binary" if bin_acc > reg_acc else "regression",
            "by_regime": regime_summary,
        }

    def log_summary(self) -> None:
        s = self.get_comparison_summary()
        if s["n"] == 0:
            logger.info("SignalAccuracyTracker: no resolved predictions yet")
            return
        logger.info(
            "SignalAccuracyTracker (n=%d): regression=%.1f%% binary=%.1f%% advantage=%+.1f%%",
            s["n"],
            (s["regression_accuracy"] or 0) * 100,
            (s["binary_accuracy"] or 0) * 100,
            (s["binary_advantage"] or 0) * 100,
        )

    # ------------------------------------------------------------------
    # Persistence (lightweight JSON)
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            state = {
                "resolved": list(self._resolved),
                "pending": self._pending,
                "window": self._window,
            }
            with open(self._state_path, "w") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            logger.debug("SignalAccuracyTracker persist error: %s", exc)

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as fh:
                state = json.load(fh)
            self._window = int(state.get("window", _DEFAULT_WINDOW))
            self._resolved = deque(state.get("resolved", []), maxlen=self._window)
            self._pending = state.get("pending", {})
            logger.debug(
                "SignalAccuracyTracker loaded: %d resolved, %d pending",
                len(self._resolved),
                len(self._pending),
            )
        except Exception as exc:
            logger.debug("SignalAccuracyTracker load error: %s", exc)
