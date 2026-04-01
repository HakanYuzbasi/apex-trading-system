"""
models/regime_model_selector.py

Per-regime model selection layer for GodLevelSignalGenerator.

Instead of using a single global ensemble for all regimes, this class
maintains 8 regime-specific model weight calibrations.  At prediction time
it blends the global signal with the regime-specific calibration:

    final = regime_conf × regime_signal + (1 - regime_conf) × global_signal

Regime-specific calibrations are fitted from the closed-trade audit JSONL
using Isotonic Regression on the signal→return relationship, per regime.
When a regime has insufficient data (< min_samples), it falls back to the
global model with calibration weight = 0.

Architecture
------------
- ``RegimeModelCalibration``  one isotonic curve per regime
- ``RegimeModelSelector``      wraps GodLevelSignalGenerator + calibrations

Wire in execution_loop::

    from models.regime_model_selector import RegimeModelSelector
    self._regime_selector = RegimeModelSelector(data_dir=ApexConfig.DATA_DIR)

    # at signal generation
    base_signal = god_signal_generator.generate_ml_signal(symbol, data)
    signal = self._regime_selector.apply(
        base_signal=base_signal,
        regime=str(self._current_regime),
    )

    # at trade close (for fitting)
    self._regime_selector.record_outcome(symbol, regime, signal_at_entry, pnl_pct)

    # offline / weekly retrain
    self._regime_selector.fit()
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_REGIMES = (
    "strong_bull", "bull", "neutral", "bear", "strong_bear",
    "volatile", "crisis", "high_vol",
)
_DEFAULT_MIN_SAMPLES = 30
_DEFAULT_MAX_CALIBRATION_WEIGHT = 0.40   # caps regime blend at 40%
_DEFAULT_LOOKBACK_DAYS = 90


@dataclass
class _OutcomeRecord:
    regime: str
    signal: float
    pnl_pct: float
    ts: float = field(default_factory=time.time)


@dataclass
class RegimeCalibration:
    """Per-regime isotonic calibration state."""
    regime: str
    n_samples: int = 0
    # Monotone breakpoints (signal → return multiplier), fitted via isotonic regression
    signal_knots: List[float] = field(default_factory=list)
    return_knots: List[float] = field(default_factory=list)
    ic: float = 0.0          # information coefficient (rank correlation)
    fitted_at: Optional[float] = None

    def predict(self, signal: float) -> float:
        """Interpolate fitted calibration curve."""
        if not self.signal_knots or len(self.signal_knots) < 2:
            return signal
        return float(np.interp(signal, self.signal_knots, self.return_knots))

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "n_samples": self.n_samples,
            "signal_knots": self.signal_knots,
            "return_knots": self.return_knots,
            "ic": round(self.ic, 4),
            "fitted_at": self.fitted_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeCalibration":
        return cls(
            regime=d.get("regime", "unknown"),
            n_samples=int(d.get("n_samples", 0)),
            signal_knots=list(d.get("signal_knots", [])),
            return_knots=list(d.get("return_knots", [])),
            ic=float(d.get("ic", 0.0)),
            fitted_at=d.get("fitted_at"),
        )


class RegimeModelSelector:
    """Per-regime signal calibration layer on top of the global ML ensemble.

    Parameters
    ----------
    data_dir : Path | None
        Used for loading trade audit JSONL and persisting calibrations.
    min_samples : int
        Minimum per-regime samples before calibration is applied.
    max_calibration_weight : float
        Maximum weight given to the regime-specific calibration signal
        (0 = always use global model, 1 = always use regime calibration).
    lookback_days : int
        How far back to look in trade audit when fitting calibrations.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        max_calibration_weight: float = _DEFAULT_MAX_CALIBRATION_WEIGHT,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._min_samples = min_samples
        self._max_cal_weight = max_calibration_weight
        self._lookback_days = lookback_days

        self._calibrations: Dict[str, RegimeCalibration] = {
            r: RegimeCalibration(regime=r) for r in _REGIMES
        }
        self._outcome_buffer: List[_OutcomeRecord] = []

        if self._data_dir:
            self._load_calibrations()

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(self, base_signal: float, regime: str) -> float:
        """Blend global signal with regime-specific calibration."""
        cal = self._calibrations.get(_norm_regime(regime))
        if cal is None or cal.n_samples < self._min_samples:
            return float(base_signal)

        calibrated = cal.predict(float(base_signal))
        # Weight proportional to IC, capped at max_calibration_weight
        weight = min(
            self._max_cal_weight,
            max(0.0, cal.ic) * self._max_cal_weight,
        )
        blended = weight * calibrated + (1.0 - weight) * float(base_signal)
        return float(np.clip(blended, -1.0, 1.0))

    def record_outcome(
        self,
        symbol: str,  # noqa: ARG002 — future use for per-symbol curves
        regime: str,
        signal_at_entry: float,
        pnl_pct: float,
    ) -> None:
        """Buffer a live outcome for the next fit() call."""
        self._outcome_buffer.append(
            _OutcomeRecord(
                regime=_norm_regime(regime),
                signal=float(signal_at_entry),
                pnl_pct=float(pnl_pct),
            )
        )

    def fit(self) -> Dict[str, dict]:
        """Fit (or refit) isotonic calibrations from audit data + buffer.

        Returns summary of per-regime IC values.
        """
        records = self._load_audit_records() + self._outcome_buffer
        cutoff = time.time() - self._lookback_days * 86400.0
        records = [r for r in records if r.ts >= cutoff]

        by_regime: Dict[str, List[_OutcomeRecord]] = defaultdict(list)
        for r in records:
            by_regime[_norm_regime(r.regime)].append(r)

        summary: Dict[str, dict] = {}
        for regime in _REGIMES:
            recs = by_regime.get(regime, [])
            if len(recs) < self._min_samples:
                summary[regime] = {"n": len(recs), "ic": 0.0, "status": "insufficient_data"}
                continue
            cal = self._fit_one(regime, recs)
            self._calibrations[regime] = cal
            summary[regime] = {
                "n": cal.n_samples,
                "ic": cal.ic,
                "status": "fitted",
            }

        if self._data_dir:
            self._save_calibrations()
        return summary

    def get_calibration_report(self) -> dict:
        return {
            r: cal.to_dict()
            for r, cal in self._calibrations.items()
        }

    # ── Fitting ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_one(regime: str, records: List[_OutcomeRecord]) -> RegimeCalibration:
        """Fit an isotonic regression curve for a single regime."""
        signals = np.array([r.signal for r in records], dtype=float)
        returns = np.array([r.pnl_pct for r in records], dtype=float)

        # Rank IC
        from scipy.stats import spearmanr  # type: ignore[import]
        try:
            ic, _ = spearmanr(signals, returns)
            ic = float(ic) if np.isfinite(ic) else 0.0
        except Exception:
            ic = 0.0

        # Isotonic regression (monotone non-decreasing in signal space)
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore[import]
            order = np.argsort(signals)
            x_sorted = signals[order]
            y_sorted = returns[order]
            iso = IsotonicRegression(out_of_bounds="clip")
            y_fit = iso.fit_transform(x_sorted, y_sorted)
            # Deduplicate knots
            knots: List[Tuple[float, float]] = []
            prev_x = None
            for xi, yi in zip(x_sorted, y_fit):
                if prev_x is None or abs(xi - prev_x) > 1e-6:
                    knots.append((float(xi), float(yi)))
                    prev_x = xi
            signal_knots = [k[0] for k in knots]
            return_knots = [k[1] for k in knots]
        except Exception:
            # Fallback: linear fit
            coeffs = np.polyfit(signals, returns, 1)
            x_grid = np.linspace(float(signals.min()), float(signals.max()), 10)
            signal_knots = list(x_grid)
            return_knots = list(np.polyval(coeffs, x_grid))

        return RegimeCalibration(
            regime=regime,
            n_samples=len(records),
            signal_knots=signal_knots,
            return_knots=return_knots,
            ic=ic,
            fitted_at=time.time(),
        )

    # ── Audit loading ─────────────────────────────────────────────────────────

    def _load_audit_records(self) -> List[_OutcomeRecord]:
        if self._data_dir is None:
            return []
        records: List[_OutcomeRecord] = []
        audit_dirs = [
            self._data_dir / "users" / "admin" / "audit",
            self._data_dir / "audit",
        ]
        for audit_dir in audit_dirs:
            if not audit_dir.exists():
                continue
            for path in sorted(audit_dir.glob("trade_audit_*.jsonl")):
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        if d.get("event") != "EXIT":
                            continue
                        signal = float(d.get("entry_signal", d.get("signal", 0.0)) or 0.0)
                        pnl = float(d.get("pnl_pct", 0.0) or 0.0)
                        regime = str(d.get("regime", "neutral") or "neutral")
                        ts_str = str(d.get("ts", "") or "")
                        try:
                            from datetime import datetime
                            ts = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00")
                            ).timestamp()
                        except Exception:
                            ts = time.time()
                        records.append(_OutcomeRecord(
                            regime=_norm_regime(regime),
                            signal=signal,
                            pnl_pct=pnl,
                            ts=ts,
                        ))
                except Exception:
                    pass
        return records

    # ── Persistence ───────────────────────────────────────────────────────────

    def _cal_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "regime_model_calibrations.json"

    def _save_calibrations(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            state = {r: cal.to_dict() for r, cal in self._calibrations.items()}
            tmp = self._cal_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._cal_path())
        except Exception:
            pass

    def _load_calibrations(self) -> None:
        try:
            p = self._cal_path()
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            for regime, d in raw.items():
                self._calibrations[regime] = RegimeCalibration.from_dict(d)
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_regime(regime: str) -> str:
    r = str(regime or "neutral").strip().lower()
    # Normalize aliases
    aliases = {
        "bearish": "bear",
        "bullish": "bull",
        "high_volatility": "high_vol",
        "stress": "crisis",
    }
    return aliases.get(r, r) if r in _REGIMES or r in aliases else "neutral"
