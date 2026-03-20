"""
ic_tracker.py — Rolling Information Coefficient (IC) tracker per feature.

IC = Spearman rank correlation between a feature at time T and the
     forward N-day return at time T+N.

Renaissance Technologies, Two Sigma, and virtually every quant fund track
IC per signal daily. Features with IC consistently below 0.02 are dead —
they predict nothing and add noise. Features with IC > 0.05 are gold.

Usage:
    tracker = ICTracker()

    # At signal generation time, record feature snapshot + signal
    tracker.record_features(symbol="AAPL", date="2026-03-20",
                            features={"rsi_14": 62.3, "macd_hist": 0.12, ...},
                            signal=0.18)

    # At trade evaluation (5 days later) or every cycle for live prices
    tracker.record_return(symbol="AAPL", entry_date="2026-03-20",
                          fwd_return_5d=0.024)

    # Query
    ic = tracker.get_ic("rsi_14", window=30)
    summary = tracker.get_summary()
    dead = tracker.get_dead_features(threshold=0.01)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "ic_tracker_state.json"


@dataclass
class FeatureSnapshot:
    symbol: str
    date: str                        # ISO-format date string
    features: Dict[str, float]       # feature_name → value at generation time
    signal: float                    # composite signal at that time
    fwd_return: Optional[float] = None   # filled in later (5-day fwd)
    filled_ts: Optional[float] = None   # epoch seconds when fwd_return was filled


@dataclass
class ICStats:
    feature: str
    ic_30d: float       # 30-day rolling IC
    ic_90d: float       # 90-day rolling IC
    n_obs: int          # number of observations
    status: str         # "live" | "dead" | "suspect" | "strong"


class ICTracker:
    """
    Tracks rolling Information Coefficient per feature.

    Thread-safe for read; NOT safe for concurrent writes.
    """

    IC_DEAD_THRESHOLD: float = 0.015     # IC below this = feature is not predicting
    IC_STRONG_THRESHOLD: float = 0.05    # IC above this = feature has real edge
    IC_SUSPECT_THRESHOLD: float = 0.03   # between dead and strong = watch list

    def __init__(
        self,
        state_path: str = str(_DEFAULT_STATE_PATH),
        max_buffer: int = 2000,          # max snapshots kept in memory
        fwd_return_days: int = 5,        # how many days ahead to compute IC
        persist: bool = True,
    ) -> None:
        self._state_path = Path(state_path)
        self._max_buffer = max_buffer
        self._fwd_days = fwd_return_days
        self._persist = persist

        # Pending snapshots: symbol+date → FeatureSnapshot (waiting for fwd return)
        self._pending: Dict[str, FeatureSnapshot] = {}

        # Completed observations: feature_name → deque of (feature_value, fwd_return) pairs
        self._observations: Dict[str, Deque[Tuple[float, float]]] = {}

        # Load state from disk
        if persist:
            self._load_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Public write API
    # ──────────────────────────────────────────────────────────────────────────

    def record_features(
        self,
        symbol: str,
        date: str,
        features: Dict[str, float],
        signal: float,
    ) -> None:
        """
        Record a feature snapshot at signal-generation time.
        Call this each time `generate_ml_signal()` is called for a new entry candidate.
        """
        key = f"{symbol}_{date}"
        if key in self._pending:
            return  # Already recorded today
        self._pending[key] = FeatureSnapshot(
            symbol=symbol,
            date=date,
            features={k: float(v) for k, v in features.items() if v is not None},
            signal=float(signal),
        )
        # Purge old pending entries (> 30 days waiting = give up)
        self._purge_stale_pending()

    def record_return(
        self,
        symbol: str,
        entry_date: str,
        fwd_return_5d: float,
    ) -> None:
        """
        Fill in the forward return for a previously recorded snapshot.
        Call this when a price is available N days after the snapshot.
        For live trading: call this at trade close with actual P&L %.
        For batch processing: call this when price data is available.
        """
        key = f"{symbol}_{entry_date}"
        snapshot = self._pending.get(key)
        if snapshot is None:
            return

        snapshot.fwd_return = float(fwd_return_5d)
        snapshot.filled_ts = time.time()

        # Update per-feature observation buffers
        for feat_name, feat_val in snapshot.features.items():
            if feat_name not in self._observations:
                self._observations[feat_name] = deque(maxlen=self._max_buffer)
            self._observations[feat_name].append((feat_val, snapshot.fwd_return))

        del self._pending[key]

        # Persist periodically (every 50 observations)
        if self._persist:
            total = sum(len(v) for v in self._observations.values())
            if total % 50 == 0:
                self._save_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Query API
    # ──────────────────────────────────────────────────────────────────────────

    def get_ic(self, feature: str, window: int = 30) -> float:
        """
        Compute Spearman rank IC for the last *window* observations of *feature*.
        Returns 0.0 if insufficient data.
        """
        obs = self._observations.get(feature)
        if not obs or len(obs) < max(5, window // 3):
            return 0.0
        recent = list(obs)[-window:]
        vals = np.array([x[0] for x in recent])
        rets = np.array([x[1] for x in recent])
        if np.std(vals) < 1e-10 or np.std(rets) < 1e-10:
            return 0.0
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(vals, rets)
            return float(ic) if not math.isnan(ic) else 0.0
        except Exception:
            # Fallback: Pearson on ranked values
            v_rank = np.argsort(np.argsort(vals)).astype(float)
            r_rank = np.argsort(np.argsort(rets)).astype(float)
            corr = float(np.corrcoef(v_rank, r_rank)[0, 1])
            return corr if not math.isnan(corr) else 0.0

    def get_stats(self, feature: str) -> ICStats:
        """Return full IC stats for a feature."""
        ic_30 = self.get_ic(feature, 30)
        ic_90 = self.get_ic(feature, 90)
        n = len(self._observations.get(feature, []))
        if ic_30 >= self.IC_STRONG_THRESHOLD:
            status = "strong"
        elif ic_30 <= self.IC_DEAD_THRESHOLD:
            status = "dead"
        elif ic_30 <= self.IC_SUSPECT_THRESHOLD:
            status = "suspect"
        else:
            status = "live"
        return ICStats(feature=feature, ic_30d=ic_30, ic_90d=ic_90, n_obs=n, status=status)

    def get_summary(self, min_obs: int = 10) -> Dict[str, float]:
        """
        Return {feature: ic_30d} for all features with >= min_obs observations.
        Sorted by |IC| descending.
        """
        results = {}
        for feat in self._observations:
            if len(self._observations[feat]) >= min_obs:
                results[feat] = self.get_ic(feat, 30)
        return dict(sorted(results.items(), key=lambda x: -abs(x[1])))

    def get_dead_features(self, threshold: float = None) -> Set[str]:
        """Return set of features whose 30-day IC is below threshold."""
        t = threshold if threshold is not None else self.IC_DEAD_THRESHOLD
        return {
            feat for feat in self._observations
            if len(self._observations[feat]) >= 20 and abs(self.get_ic(feat, 30)) < t
        }

    def get_strong_features(self, threshold: float = None) -> Set[str]:
        """Return features with IC consistently above threshold."""
        t = threshold if threshold is not None else self.IC_STRONG_THRESHOLD
        return {
            feat for feat in self._observations
            if len(self._observations[feat]) >= 20 and self.get_ic(feat, 30) > t
        }

    def get_pending_count(self) -> int:
        return len(self._pending)

    def get_observation_counts(self) -> Dict[str, int]:
        return {feat: len(obs) for feat, obs in self._observations.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _purge_stale_pending(self, max_age_days: int = 30) -> None:
        """Drop pending snapshots older than max_age_days."""
        import datetime
        cutoff = (datetime.date.today() - datetime.timedelta(days=max_age_days)).isoformat()
        stale = [k for k, v in self._pending.items() if v.date < cutoff]
        for k in stale:
            del self._pending[k]

    def _save_state(self) -> None:
        try:
            state = {
                "observations": {
                    feat: list(obs)
                    for feat, obs in self._observations.items()
                },
                "saved_at": time.time(),
            }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(state, f, separators=(",", ":"))
        except Exception as e:
            logger.debug("ICTracker save error: %s", e)

    def _load_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            with open(self._state_path) as f:
                state = json.load(f)
            for feat, pairs in state.get("observations", {}).items():
                self._observations[feat] = deque(
                    [(float(p[0]), float(p[1])) for p in pairs],
                    maxlen=self._max_buffer,
                )
            logger.info(
                "ICTracker: loaded %d features, %d total observations",
                len(self._observations),
                sum(len(v) for v in self._observations.values()),
            )
        except Exception as e:
            logger.debug("ICTracker load error: %s", e)
