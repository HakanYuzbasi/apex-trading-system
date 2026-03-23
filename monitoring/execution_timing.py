"""Execution Timing Optimizer.

Tracks fill quality (slippage bps) per (hour_of_day, day_of_week, regime) bucket
and produces a [0, 1] timing score that execution_loop uses to penalise entries
during historically poor execution windows.

Score = 1.0  →  historically good window (low slippage)
Score < 1.0  →  poor window (penalises confidence / sizing)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────────────

_MIN_OBS: int = 5        # minimum fills per bucket before scoring
_LOOKBACK_DAYS: int = 30 # rolling window (prune older entries)
_DEFAULT_SCORE: float = 1.0  # returned when bucket has no history
_SCORE_FLOOR: float = 0.55   # worst possible score
_EWMA_ALPHA: float = 0.10    # weight of new obs in exponential moving avg


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BucketStats:
    obs: int = 0
    mean_slippage_bps: float = 0.0   # EWMA
    var_slippage_bps: float = 0.0    # running variance (Welford)
    worst_bps: float = 0.0
    _m2: float = 0.0                 # Welford M2 accumulator

    def update(self, slippage_bps: float) -> None:
        """Welford online mean/variance update + EWMA mean."""
        self.obs += 1
        # Welford running variance
        delta = slippage_bps - self.mean_slippage_bps
        self.mean_slippage_bps += delta / self.obs
        delta2 = slippage_bps - self.mean_slippage_bps
        self._m2 += delta * delta2
        self.var_slippage_bps = self._m2 / self.obs if self.obs > 1 else 0.0
        # Track worst
        if slippage_bps > self.worst_bps:
            self.worst_bps = slippage_bps

    def to_dict(self) -> dict:
        return {
            "obs": self.obs,
            "mean_slippage_bps": round(self.mean_slippage_bps, 3),
            "var_slippage_bps": round(self.var_slippage_bps, 3),
            "worst_bps": round(self.worst_bps, 3),
        }


@dataclass
class TimingScore:
    score: float          # [0.55, 1.0] — multiply against confidence
    hour: int
    day_of_week: int
    regime: str
    mean_slippage_bps: float
    obs: int
    has_data: bool


# ── Main class ────────────────────────────────────────────────────────────────

class ExecutionTimingOptimizer:
    """Track slippage per (hour, dow, regime) bucket and return timing scores."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        min_obs: int = _MIN_OBS,
        score_floor: float = _SCORE_FLOOR,
    ) -> None:
        self._min_obs = min_obs
        self._score_floor = score_floor
        self._data_dir = data_dir

        # Buckets keyed by (hour, dow, regime)
        self._buckets: Dict[Tuple[int, int, str], BucketStats] = {}

        # Raw fill log for percentile calibration: [(timestamp, bps)]
        self._fill_log: List[Tuple[float, float]] = []

        # Global percentiles (calibrated after enough data)
        self._p50_bps: float = 0.0
        self._p95_bps: float = 0.0

        if data_dir is not None:
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_fill(
        self,
        slippage_bps: float,
        hour: int,
        day_of_week: int,
        regime: str = "neutral",
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a single fill's slippage into the matching bucket.

        Args:
            slippage_bps: Positive = paid more than mid (cost), negative = saved.
            hour: Hour of day 0-23 (UTC or local, be consistent).
            day_of_week: 0=Mon … 6=Sun.
            regime: Regime label string.
            timestamp: Unix time (defaults to now).
        """
        if timestamp is None:
            timestamp = time.time()

        key = (hour, day_of_week, regime)
        if key not in self._buckets:
            self._buckets[key] = BucketStats()
        self._buckets[key].update(slippage_bps)

        self._fill_log.append((timestamp, slippage_bps))
        self._prune_fill_log()
        self._recalibrate_percentiles()

        if self._data_dir is not None:
            self._save()

    def get_timing_score(
        self,
        hour: int,
        day_of_week: int,
        regime: str = "neutral",
    ) -> TimingScore:
        """Return a timing quality score for the given window.

        Score of 1.0 = historically average or better (no penalty).
        Score < 1.0 = historically worse than average (penalise).
        """
        key = (hour, day_of_week, regime)
        bucket = self._buckets.get(key)

        if bucket is None or bucket.obs < self._min_obs:
            return TimingScore(
                score=_DEFAULT_SCORE,
                hour=hour,
                day_of_week=day_of_week,
                regime=regime,
                mean_slippage_bps=0.0,
                obs=bucket.obs if bucket else 0,
                has_data=False,
            )

        score = self._compute_score(bucket.mean_slippage_bps)
        return TimingScore(
            score=score,
            hour=hour,
            day_of_week=day_of_week,
            regime=regime,
            mean_slippage_bps=bucket.mean_slippage_bps,
            obs=bucket.obs,
            has_data=True,
        )

    def get_worst_hours(self, regime: str = "neutral", top_n: int = 5) -> List[dict]:
        """Return top-N worst execution hours for the given regime."""
        rows = []
        for (h, d, r), stats in self._buckets.items():
            if r != regime or stats.obs < self._min_obs:
                continue
            score = self._compute_score(stats.mean_slippage_bps)
            rows.append({
                "hour": h,
                "day_of_week": d,
                "mean_slippage_bps": round(stats.mean_slippage_bps, 2),
                "obs": stats.obs,
                "score": round(score, 3),
            })
        rows.sort(key=lambda x: x["score"])
        return rows[:top_n]

    def get_report(self) -> dict:
        """Full summary: bucket stats, global percentiles, worst windows."""
        buckets_out: dict = {}
        for (h, d, r), stats in self._buckets.items():
            k = f"h{h:02d}_d{d}_{r}"
            buckets_out[k] = stats.to_dict()
            buckets_out[k]["score"] = round(
                self._compute_score(stats.mean_slippage_bps), 3
            )

        return {
            "total_fills": len(self._fill_log),
            "p50_bps": round(self._p50_bps, 3),
            "p95_bps": round(self._p95_bps, 3),
            "buckets": buckets_out,
            "worst_neutral": self.get_worst_hours("neutral"),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _compute_score(self, mean_bps: float) -> float:
        """Convert mean slippage into a [_SCORE_FLOOR, 1.0] penalty score.

        Logic:
        - If mean_bps <= p50 (median) → score 1.0 (no penalty)
        - If mean_bps >= p95 → score = _score_floor (maximum penalty)
        - Between p50 and p95 → linear interpolation
        """
        if self._p95_bps <= self._p50_bps:
            return _DEFAULT_SCORE
        if mean_bps <= self._p50_bps:
            return _DEFAULT_SCORE
        if mean_bps >= self._p95_bps:
            return self._score_floor

        t = (mean_bps - self._p50_bps) / (self._p95_bps - self._p50_bps)
        return _DEFAULT_SCORE - t * (_DEFAULT_SCORE - self._score_floor)

    def _recalibrate_percentiles(self) -> None:
        """Compute p50 and p95 from recent fill log."""
        vals = [bps for _, bps in self._fill_log]
        if len(vals) < 2:
            return
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        self._p50_bps = sorted_vals[int(0.50 * n)]
        self._p95_bps = sorted_vals[min(int(0.95 * n), n - 1)]

    def _prune_fill_log(self) -> None:
        """Remove fill log entries older than _LOOKBACK_DAYS."""
        cutoff = time.time() - _LOOKBACK_DAYS * 86_400
        self._fill_log = [(ts, bps) for ts, bps in self._fill_log if ts >= cutoff]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "execution_timing.json"

    def _save(self) -> None:
        try:
            state: dict = {
                "p50_bps": self._p50_bps,
                "p95_bps": self._p95_bps,
                "fill_log": self._fill_log[-500:],  # keep last 500 fills
                "buckets": {
                    f"{h},{d},{r}": asdict(stats)
                    for (h, d, r), stats in self._buckets.items()
                },
            }
            tmp = self._state_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self._state_path())
        except Exception:
            pass

    def _load(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._p50_bps = float(raw.get("p50_bps", 0.0))
            self._p95_bps = float(raw.get("p95_bps", 0.0))
            self._fill_log = [tuple(x) for x in raw.get("fill_log", [])]
            for key_str, stats_dict in raw.get("buckets", {}).items():
                parts = key_str.split(",", 2)
                h, d, r = int(parts[0]), int(parts[1]), parts[2]
                b = BucketStats(**{k: v for k, v in stats_dict.items()})
                self._buckets[(h, d, r)] = b
        except Exception:
            pass
