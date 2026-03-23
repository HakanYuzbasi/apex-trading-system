"""
monitoring/alpha_decay_calibrator.py — Signal Alpha Decay Calibrator

Measures how quickly signal predictive power (IC) decays across hold-time horizons.
Each completed trade contributes one data point: (signal, actual_return, hold_hours, regime).
Trades are bucketed by hold duration and IC is computed per bucket per regime.

Output:
  - get_optimal_hold_hours(regime)  → recommended hold duration (hours)
  - get_alpha_half_life(regime)     → estimated hours until IC halves
  - get_decay_score(hold_hours, regime) → [0.5, 1.0] sizing modifier based on hold horizon

Wire-in (execution_loop.py):
    from monitoring.alpha_decay_calibrator import AlphaDecayCalibrator
    _decay_cal = AlphaDecayCalibrator(data_dir=ApexConfig.DATA_DIR)

    # At trade exit:
    _decay_cal.record_trade(
        signal=float(_entry_sig),
        actual_return=float(_pnl_pct),
        hold_hours=float(hold_minutes / 60),
        regime=str(_outcome_regime),
    )

    # When sizing a new trade, penalise if targeting a hold horizon with low IC:
    _timing_mult = _decay_cal.get_decay_score(expected_hold_hours, regime)
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Hold-time buckets (hours) ─────────────────────────────────────────────────
_BUCKETS: List[Tuple[float, float]] = [
    (0.0,   2.0),   # intraday scalp
    (2.0,   8.0),   # intraday swing
    (8.0,  24.0),   # overnight
    (24.0, 120.0),  # multi-day
]
_BUCKET_LABELS = ["0-2h", "2-8h", "8-24h", "24h+"]
_MIN_OBS = 10           # minimum obs per bucket for reliable IC
_DEFAULT_HOLD = 4.0     # default optimal hold when not enough data (hours)


@dataclass
class BucketStats:
    label: str
    n_obs: int
    ic: float           # Pearson IC of signal vs return
    mean_return: float  # avg actual return in this bucket
    hit_rate: float     # fraction directionally correct


@dataclass
class RegimeDecay:
    regime: str
    buckets: List[BucketStats]
    optimal_hold_hours: float       # midpoint of best-IC bucket
    alpha_half_life: Optional[float]  # estimated hours to IC/2 (None if insufficient data)
    peak_ic: float


class AlphaDecayCalibrator:
    """
    Tracks and calibrates signal alpha decay across hold-time horizons.

    Thread-safe for reads; record_trade is assumed to be called from a
    single async loop.
    """

    def __init__(
        self,
        min_obs: int = _MIN_OBS,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._min_obs = min_obs
        self._data_dir = Path(data_dir) if data_dir else None

        # regime → bucket_idx → [(signal, return)]
        self._obs: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(
            lambda: [[] for _ in _BUCKETS]
        )

        # Cached decay analysis per regime
        self._cache: Dict[str, RegimeDecay] = {}
        self._total_trades = 0

        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(
        self,
        signal: float,
        actual_return: float,
        hold_hours: float,
        regime: str = "neutral",
    ) -> None:
        """
        Record a completed trade for calibration.

        Args:
            signal:        Signal used at entry [-1, 1].
            actual_return: Realised P&L fraction (positive = profit).
            hold_hours:    Duration position was held, in hours.
            regime:        Regime at entry.
        """
        if signal == 0.0:
            return  # neutral signal carries no directional information
        bucket = self._bucket_idx(hold_hours)
        self._obs[regime.lower()][bucket].append((float(signal), float(actual_return)))
        self._total_trades += 1

        # Rebuild cache for this regime when we have enough new data
        if self._total_trades % 10 == 0:
            self._cache.pop(regime.lower(), None)

        self._persist()

    def get_optimal_hold_hours(self, regime: str = "neutral") -> float:
        """
        Return the recommended hold duration for this regime.
        Falls back to DEFAULT_HOLD when insufficient data.
        """
        decay = self._get_regime_decay(regime)
        if decay is None:
            return _DEFAULT_HOLD
        return decay.optimal_hold_hours

    def get_alpha_half_life(self, regime: str = "neutral") -> Optional[float]:
        """Estimated hours until IC decays to half of its peak value."""
        decay = self._get_regime_decay(regime)
        if decay is None:
            return None
        return decay.alpha_half_life

    def get_decay_score(
        self,
        hold_hours: float,
        regime: str = "neutral",
    ) -> float:
        """
        Sizing modifier [0.50, 1.0] based on whether the intended hold horizon
        overlaps with the highest-IC bucket for this regime.

        Returns 1.0 when data is insufficient (no penalty for unknowns).
        """
        decay = self._get_regime_decay(regime)
        if decay is None or decay.peak_ic <= 0:
            return 1.0

        bucket = self._bucket_idx(hold_hours)
        target_bucket_stats = next(
            (b for b in decay.buckets if b.label == _BUCKET_LABELS[bucket]), None
        )
        if target_bucket_stats is None or target_bucket_stats.n_obs < self._min_obs:
            return 1.0

        # Scale modifier by ratio of target IC to peak IC
        ic_ratio = max(0.0, target_bucket_stats.ic) / max(decay.peak_ic, 1e-9)
        return round(max(0.50, min(1.0, 0.50 + ic_ratio * 0.50)), 4)

    def get_decay_report(self) -> dict:
        """JSON-serialisable decay analysis across all observed regimes."""
        out: dict = {
            "total_trades": self._total_trades,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "regimes": {},
        }
        all_regimes = set(self._obs.keys())
        for regime in all_regimes:
            decay = self._get_regime_decay(regime)
            if decay is None:
                continue
            out["regimes"][regime] = {
                "optimal_hold_hours": decay.optimal_hold_hours,
                "alpha_half_life": decay.alpha_half_life,
                "peak_ic": round(decay.peak_ic, 4),
                "buckets": [
                    {
                        "label": b.label,
                        "n_obs": b.n_obs,
                        "ic": round(b.ic, 4),
                        "hit_rate": round(b.hit_rate, 4),
                        "mean_return": round(b.mean_return, 5),
                    }
                    for b in decay.buckets
                ],
            }
        return out

    # ── Core computation ──────────────────────────────────────────────────────

    def _get_regime_decay(self, regime: str) -> Optional[RegimeDecay]:
        regime = regime.lower()
        if regime in self._cache:
            return self._cache[regime]

        buckets_data = self._obs.get(regime)
        if not buckets_data:
            return None

        bucket_stats: List[BucketStats] = []
        for idx, obs_list in enumerate(buckets_data):
            if len(obs_list) < self._min_obs:
                bucket_stats.append(BucketStats(
                    label=_BUCKET_LABELS[idx], n_obs=len(obs_list),
                    ic=0.0, mean_return=0.0, hit_rate=0.0,
                ))
                continue
            sigs = [o[0] for o in obs_list]
            rets = [o[1] for o in obs_list]
            ic = self._pearson_corr(sigs, rets)
            hit_rate = sum(
                1 for s, r in zip(sigs, rets)
                if math.copysign(1, s) == math.copysign(1, r)
            ) / len(obs_list)
            bucket_stats.append(BucketStats(
                label=_BUCKET_LABELS[idx],
                n_obs=len(obs_list),
                ic=ic,
                mean_return=sum(rets) / len(rets),
                hit_rate=hit_rate,
            ))

        # Best IC bucket → optimal hold
        valid = [(i, b) for i, b in enumerate(bucket_stats) if b.n_obs >= self._min_obs]
        if not valid:
            return None

        best_idx, best_bucket = max(valid, key=lambda x: x[1].ic)
        lo, hi = _BUCKETS[best_idx]
        optimal_hold = (lo + hi) / 2 if hi < 120 else lo + 24

        peak_ic = best_bucket.ic
        half_life = self._estimate_half_life(bucket_stats, peak_ic)

        decay = RegimeDecay(
            regime=regime,
            buckets=bucket_stats,
            optimal_hold_hours=optimal_hold,
            alpha_half_life=half_life,
            peak_ic=peak_ic,
        )
        self._cache[regime] = decay
        return decay

    @staticmethod
    def _estimate_half_life(buckets: List[BucketStats], peak_ic: float) -> Optional[float]:
        """
        Fit a simple exponential decay to IC vs bucket midpoint.
        Returns estimated hours until IC = peak_ic / 2.
        None if insufficient valid buckets.
        """
        valid_pts = []
        for i, b in enumerate(buckets):
            if b.n_obs < _MIN_OBS or b.ic <= 0:
                continue
            lo, hi = _BUCKETS[i]
            mid = (lo + hi) / 2 if hi < 120 else lo + 24
            valid_pts.append((mid, b.ic))

        if len(valid_pts) < 2:
            return None

        # Fit ln(IC) = ln(IC0) - t/tau via linear regression on log scale
        try:
            xs = [p[0] for p in valid_pts]
            ys = [math.log(max(p[1], 1e-9)) for p in valid_pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
            den = sum((xi - mx) ** 2 for xi in xs)
            if abs(den) < 1e-12:
                return None
            slope = num / den  # slope of log(IC) vs t
            if slope >= 0:
                return None   # IC not decaying
            # half-life = ln(2) / (-slope)
            return round(math.log(2) / (-slope), 1)
        except Exception:
            return None

    @staticmethod
    def _bucket_idx(hold_hours: float) -> int:
        h = max(0.0, float(hold_hours))
        for i, (lo, hi) in enumerate(_BUCKETS):
            if h < hi:
                return i
        return len(_BUCKETS) - 1

    @staticmethod
    def _pearson_corr(x: List[float], y: List[float]) -> float:
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
        if not self._data_dir:
            return
        if self._total_trades % 10 != 0:
            return  # only persist every 10 trades to reduce I/O
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / "alpha_decay_calibrator.json"
            payload = {
                "total_trades": self._total_trades,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "obs": {
                    regime: [bucket for bucket in buckets]
                    for regime, buckets in self._obs.items()
                },
            }
            path.write_text(json.dumps(payload))
        except Exception as exc:
            logger.debug("AlphaDecayCalibrator persist error: %s", exc)

    def _load_state(self) -> None:
        if not self._data_dir:
            return
        path = Path(self._data_dir) / "alpha_decay_calibrator.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._total_trades = data.get("total_trades", 0)
            for regime, buckets in data.get("obs", {}).items():
                for i, bucket in enumerate(buckets):
                    if i < len(_BUCKETS):
                        self._obs[regime][i] = [tuple(pair) for pair in bucket]
        except Exception as exc:
            logger.debug("AlphaDecayCalibrator load error: %s", exc)
