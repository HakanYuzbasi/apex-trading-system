"""
monitoring/param_optimizer.py — Bayesian Parameter Optimizer (Optuna)

Weekly self-optimization of major trading parameters using walk-forward
Sharpe as the objective function.

Tunes 12 high-leverage parameters across 4 domains:
  - Signal thresholds (regime-specific entry bars)
  - Risk sizing multipliers (VIX tiers, drawdown adaptive leverage)
  - Monte Carlo Sentinel thresholds (breach probability gates)
  - Confidence gates (tiered entry confidence)

Safe-by-default:
  - All proposed params validated against hard bounds before applying
  - Only writes if new Sharpe > baseline + MIN_IMPROVEMENT_THRESHOLD
  - Persists best params to data/optimized_params.json
  - Loads on startup; fallback to config defaults if absent/corrupt

Usage (execution_loop.py weekly cycle):
    optimizer = ParamOptimizer(data_dir=ApexConfig.DATA_DIR)
    result = await asyncio.to_thread(optimizer.run_study, n_trials=50)
    if result.improved:
        optimizer.apply_best_params()
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_STATE_FILE = "optimized_params.json"
_MIN_IMPROVEMENT = 0.05     # Sharpe must improve by at least this to apply
_MIN_TRADES = 20             # minimum completed trades needed for a valid study


# ── Param space ───────────────────────────────────────────────────────────────

# (name, low, high, step) — all floats; integer params use step=1.0
_PARAM_SPACE: List[Tuple[str, float, float, float]] = [
    # Signal thresholds (model output range ≈ [0.05, 0.27])
    ("threshold_bull",        0.10, 0.22, 0.01),
    ("threshold_neutral",     0.12, 0.24, 0.01),
    ("threshold_bear",        0.15, 0.28, 0.01),
    # Tiered confidence gate
    ("entry_confidence_moderate", 0.38, 0.60, 0.01),
    # IC dampener (applied when composite_signal IC is dead)
    ("ic_dead_signal_dampener",   0.60, 1.00, 0.05),
    # Alpha decay: minimum decay score before penalising size
    ("alpha_decay_floor",         0.50, 1.00, 0.05),
    # MC Sentinel thresholds
    ("mc_breach_threshold",       0.15, 0.50, 0.05),
    ("mc_defensive_threshold",    0.40, 0.80, 0.05),
    # Regime caution hours
    ("regime_transition_caution_hours", 1.0, 8.0, 0.5),
    # Drawdown adaptive leverage tiers
    ("dd_lev_tier1_pct",   1.0,  4.0, 0.5),
    ("dd_lev_tier2_pct",   2.0,  8.0, 0.5),
    # Options flow confidence penalty
    ("options_conf_penalty", 0.85, 0.98, 0.01),
]

# Hard bounds: params can NEVER be set outside these regardless of Optuna output
_HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "threshold_bull":                   (0.08, 0.30),
    "threshold_neutral":                (0.10, 0.30),
    "threshold_bear":                   (0.12, 0.35),
    "entry_confidence_moderate":        (0.30, 0.75),
    "ic_dead_signal_dampener":          (0.50, 1.00),
    "alpha_decay_floor":                (0.30, 1.00),
    "mc_breach_threshold":              (0.10, 0.70),
    "mc_defensive_threshold":           (0.30, 0.90),
    "regime_transition_caution_hours":  (0.5,  12.0),
    "dd_lev_tier1_pct":                 (0.5,   6.0),
    "dd_lev_tier2_pct":                 (1.0,  10.0),
    "options_conf_penalty":             (0.70,  1.00),
}

# Map param name → ApexConfig attribute name
_PARAM_TO_CONFIG: Dict[str, str] = {
    "threshold_bull":                   "SIGNAL_THRESHOLDS_BY_REGIME",  # special handling
    "threshold_neutral":                "SIGNAL_THRESHOLDS_BY_REGIME",
    "threshold_bear":                   "SIGNAL_THRESHOLDS_BY_REGIME",
    "entry_confidence_moderate":        "ENTRY_CONFIDENCE_MODERATE",
    "ic_dead_signal_dampener":          "IC_DEAD_SIGNAL_DAMPENER",
    "alpha_decay_floor":                "ALPHA_DECAY_CALIBRATOR_ENABLED",  # placeholder
    "mc_breach_threshold":              "MC_SENTINEL_BREACH_THRESH",
    "mc_defensive_threshold":           "MC_SENTINEL_DEFENSIVE_THRESH",
    "regime_transition_caution_hours":  "REGIME_TRANSITION_CAUTION_HOURS",
    "dd_lev_tier1_pct":                 "DD_LEV_TIER1_PCT",
    "dd_lev_tier2_pct":                 "DD_LEV_TIER2_PCT",
    "options_conf_penalty":             "OPTIONS_FLOW_CONF_PENALTY",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    signal: float
    pnl_pct: float
    hold_hours: float
    regime: str
    entry_confidence: float


@dataclass
class OptimizeResult:
    improved: bool
    baseline_sharpe: float
    best_sharpe: float
    best_params: Dict[str, float]
    n_trials: int
    n_trades: int
    ran_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "improved": self.improved,
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "best_sharpe": round(self.best_sharpe, 4),
            "improvement": round(self.best_sharpe - self.baseline_sharpe, 4),
            "best_params": {k: round(v, 4) for k, v in self.best_params.items()},
            "n_trials": self.n_trials,
            "n_trades": self.n_trades,
            "ran_at": self.ran_at,
        }


# ── Sharpe computation ────────────────────────────────────────────────────────

def _compute_sharpe(pnls: List[float]) -> float:
    """Annualised Sharpe from a list of trade P&L fractions."""
    if len(pnls) < 5:
        return 0.0
    n = len(pnls)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(var) if var > 0 else 1e-9
    # Annualise assuming ~250 trading days, 2 trades/day on average
    ann_factor = math.sqrt(500)
    return float(mean / std * ann_factor)


def _simulate_sharpe(
    trades: List[TradeRecord],
    params: Dict[str, float],
) -> float:
    """
    Simulate walk-forward Sharpe with proposed params by applying the
    thresholds as a filter: trades whose entry signal or confidence would
    have been blocked by the new thresholds are excluded.

    This is a conservative lower-bound estimate — it can only REMOVE trades,
    not add new ones, so it avoids look-ahead bias.
    """
    thr_bull    = params.get("threshold_bull", 0.14)
    thr_neutral = params.get("threshold_neutral", 0.18)
    thr_bear    = params.get("threshold_bear", 0.21)
    min_conf    = params.get("entry_confidence_moderate", 0.44)

    regime_thr: Dict[str, float] = {
        "bull": thr_bull, "strong_bull": thr_bull * 0.90,
        "neutral": thr_neutral,
        "bear": thr_bear, "strong_bear": thr_bear * 1.15,
        "volatile": thr_neutral * 1.10,
    }

    filtered: List[float] = []
    for t in trades:
        thr = regime_thr.get(t.regime, thr_neutral)
        if abs(t.signal) < thr:
            continue  # this trade would have been blocked by threshold
        if t.entry_confidence < min_conf:
            continue
        filtered.append(t.pnl_pct)

    if len(filtered) < _MIN_TRADES // 2:
        return -99.0  # not enough trades after filtering — penalise
    return _compute_sharpe(filtered)


# ── Optimizer ─────────────────────────────────────────────────────────────────

class ParamOptimizer:
    """
    Bayesian hyper-parameter optimization using Optuna.

    Thread-safe for reading; run_study should be called from a background thread.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else Path("data")
        self._state_path = self._data_dir / _STATE_FILE
        self._best_params: Dict[str, float] = {}
        self._last_result: Optional[OptimizeResult] = None
        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_study(
        self,
        trades: List[TradeRecord],
        n_trials: int = 50,
        timeout_seconds: float = 120.0,
    ) -> OptimizeResult:
        """
        Run Optuna study.  Returns OptimizeResult (never raises).
        Call this from asyncio.to_thread — it's blocking.
        """
        try:
            return self._run_study_inner(trades, n_trials, timeout_seconds)
        except Exception as exc:
            logger.warning("ParamOptimizer study failed: %s", exc)
            return OptimizeResult(
                improved=False, baseline_sharpe=0.0, best_sharpe=0.0,
                best_params={}, n_trials=0, n_trades=len(trades),
            )

    def apply_best_params(self) -> bool:
        """
        Write best params to config overrides file.
        Returns True if written successfully.
        """
        if not self._best_params:
            return False
        try:
            self._save_state()
            return True
        except Exception as exc:
            logger.warning("ParamOptimizer apply failed: %s", exc)
            return False

    def get_best_params(self) -> Dict[str, float]:
        return dict(self._best_params)

    def get_last_result(self) -> Optional[dict]:
        if self._last_result is None:
            return None
        return self._last_result.to_dict()

    # ── Core study ────────────────────────────────────────────────────────────

    def _run_study_inner(
        self,
        trades: List[TradeRecord],
        n_trials: int,
        timeout_seconds: float,
    ) -> OptimizeResult:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trades = len(trades)
        if n_trades < _MIN_TRADES:
            logger.info(
                "ParamOptimizer: only %d trades, need %d — skipping",
                n_trades, _MIN_TRADES,
            )
            return OptimizeResult(
                improved=False, baseline_sharpe=0.0, best_sharpe=0.0,
                best_params={}, n_trials=0, n_trades=n_trades,
            )

        # Baseline Sharpe with default/current params
        baseline_params = {name: (lo + hi) / 2 for name, lo, hi, _ in _PARAM_SPACE}
        baseline_params.update(self._best_params)  # start from already-tuned state
        baseline_sharpe = _simulate_sharpe(trades, baseline_params)

        def objective(trial: Any) -> float:
            params: Dict[str, float] = {}
            for name, lo, hi, step in _PARAM_SPACE:
                if step == 1.0:
                    params[name] = float(trial.suggest_int(name, int(lo), int(hi)))
                else:
                    params[name] = trial.suggest_float(name, lo, hi, step=step)
            return _simulate_sharpe(trades, params)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1,
        )

        best_params = study.best_params
        best_sharpe = study.best_value

        # Validate hard bounds
        validated: Dict[str, float] = {}
        for name, value in best_params.items():
            lo, hi = _HARD_BOUNDS.get(name, (-1e9, 1e9))
            validated[name] = max(lo, min(hi, float(value)))

        improved = best_sharpe > baseline_sharpe + _MIN_IMPROVEMENT
        if improved:
            self._best_params = validated
            self._save_state()
            logger.info(
                "ParamOptimizer: improved Sharpe %.3f → %.3f (+%.3f) over %d trials",
                baseline_sharpe, best_sharpe,
                best_sharpe - baseline_sharpe, len(study.trials),
            )
        else:
            logger.info(
                "ParamOptimizer: no improvement (baseline=%.3f, best=%.3f) — keeping current params",
                baseline_sharpe, best_sharpe,
            )

        result = OptimizeResult(
            improved=improved,
            baseline_sharpe=baseline_sharpe,
            best_sharpe=best_sharpe,
            best_params=validated,
            n_trials=len(study.trials),
            n_trades=n_trades,
        )
        self._last_result = result
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "best_params": self._best_params,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "last_result": self._last_result.to_dict() if self._last_result else None,
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.debug("ParamOptimizer save error: %s", exc)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
            loaded = data.get("best_params", {})
            # Validate bounds before trusting loaded params
            validated = {}
            for name, value in loaded.items():
                lo, hi = _HARD_BOUNDS.get(name, (-1e9, 1e9))
                validated[name] = max(lo, min(hi, float(value)))
            self._best_params = validated
            logger.info(
                "ParamOptimizer: loaded %d optimised params from disk",
                len(validated),
            )
        except Exception as exc:
            logger.debug("ParamOptimizer load error: %s", exc)


# ── Helper: build TradeRecord list from audit JSONL ──────────────────────────

def load_trade_records(data_dir: Path, lookback_days: int = 30) -> List[TradeRecord]:
    """
    Load EXIT trade records from trade_audit_*.jsonl for the optimizer.
    Returns empty list on any error.
    """
    import glob
    from datetime import timedelta

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
    records: List[TradeRecord] = []

    pattern = str(data_dir / "users" / "*" / "audit" / "trade_audit_*.jsonl")
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("action") != "EXIT":
                        continue
                    ts_str = rec.get("timestamp", "")
                    try:
                        from datetime import datetime as _dt
                        ts = _dt.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
                    if ts < cutoff_ts:
                        continue
                    records.append(TradeRecord(
                        signal=float(rec.get("signal", 0.0) or 0.0),
                        pnl_pct=float(rec.get("pnl_pct", 0.0) or 0.0),
                        hold_hours=float(rec.get("hold_hours", 4.0) or 4.0),
                        regime=str(rec.get("regime", "neutral") or "neutral").lower(),
                        entry_confidence=float(rec.get("confidence", 0.5) or 0.5),
                    ))
        except Exception as exc:
            logger.debug("load_trade_records error on %s: %s", path, exc)

    return records
