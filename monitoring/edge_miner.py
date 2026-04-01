"""
monitoring/edge_miner.py

Unsupervised pattern discovery on trade history.

Clusters closed trades by their context features and identifies
combinations that produce systematically high win rates.  Discovered
clusters are exported as named "edge profiles" that can be used to
boost confidence at entry when the current context matches.

Algorithm
---------
1. Load EXIT rows from trade_audit_*.jsonl.
2. Extract feature vector per trade:
   [regime_code, hour_block_code, signal_source_code, vix_bucket,
    day_of_week, hold_hours_bucket]
3. K-Means clustering (n_clusters configurable, default 10).
4. For each cluster: compute win rate, avg P&L, sample size.
5. Clusters with win_rate > threshold AND n >= min_cluster_size → EdgeProfile.

Usage in execution_loop::

    # init (once)
    self._edge_miner = EdgeMiner(audit_dir=Path(ApexConfig.DATA_DIR)/"users/admin/audit")
    self._edge_miner.load_and_mine()   # run offline / weekly

    # at entry gate (hot path — pure dict lookup, no ML)
    boost = self._edge_miner.get_confidence_boost({
        "regime": str(self._current_regime),
        "hour_block": _hour_block_for_now(),
        "signal_source": dominant_component,
        "vix": current_vix,
        "day_of_week": datetime.now().weekday(),
    })
    confidence = min(1.0, confidence + boost)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_DEFAULT_N_CLUSTERS = 12
_DEFAULT_MIN_CLUSTER = 15
_DEFAULT_MIN_WIN_RATE = 0.60
_DEFAULT_MAX_BOOST = 0.08
_DEFAULT_LOOKBACK_DAYS = 120


# ── Feature encoding helpers ──────────────────────────────────────────────────

_REGIME_MAP = {
    "strong_bull": 0, "bull": 1, "neutral": 2, "bear": 3,
    "strong_bear": 4, "volatile": 5, "crisis": 6, "high_vol": 7,
}
_HOUR_BLOCK_MAP = {"open": 0, "midday": 1, "close": 2, "extended": 3, "overnight": 4}
_SOURCE_MAP = {"ml": 0, "tech": 1, "sentiment": 2, "momentum": 3,
               "pairs": 4, "earnings": 5, "intraday_mr": 6, "unknown": 7}


def _regime_code(r: str) -> float:
    return float(_REGIME_MAP.get(str(r).lower(), 2))


def _hour_block(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        h = dt.hour
    except Exception:
        return "unknown"
    if 9 <= h < 11:
        return "open"
    if 11 <= h < 15:
        return "midday"
    if 15 <= h < 17:
        return "close"
    if 17 <= h <= 23:
        return "extended"
    return "overnight"


def _hour_block_code(ts_str: str) -> float:
    return float(_HOUR_BLOCK_MAP.get(_hour_block(ts_str), 4))


def _dominant_source(components: dict) -> str:
    if not components:
        return "unknown"
    best = max(components.items(), key=lambda x: abs(float(x[1] or 0)), default=("unknown", 0))
    return best[0] if abs(float(best[1] or 0)) > 0 else "unknown"


def _source_code(components: dict) -> float:
    return float(_SOURCE_MAP.get(_dominant_source(components), 7))


def _vix_bucket(vix: float) -> float:
    if vix < 15:
        return 0.0
    if vix < 20:
        return 1.0
    if vix < 25:
        return 2.0
    if vix < 35:
        return 3.0
    return 4.0


def _hold_bucket(hours: float) -> float:
    if hours < 1:
        return 0.0
    if hours < 4:
        return 1.0
    if hours < 24:
        return 2.0
    return 3.0


def _now_hour_block() -> str:
    h = datetime.utcnow().hour
    if 9 <= h < 11:
        return "open"
    if 11 <= h < 15:
        return "midday"
    if 15 <= h < 17:
        return "close"
    if 17 <= h <= 23:
        return "extended"
    return "overnight"


# ── Edge profile ──────────────────────────────────────────────────────────────

@dataclass
class EdgeProfile:
    """A named cluster with high win rate."""
    cluster_id: int
    name: str
    regime: str
    hour_block: str
    signal_source: str
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    confidence_boost: float        # how much to add to confidence at match
    centroid: List[float] = field(default_factory=list)
    mined_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "regime": self.regime,
            "hour_block": self.hour_block,
            "signal_source": self.signal_source,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_pct": round(self.avg_pnl_pct, 6),
            "confidence_boost": round(self.confidence_boost, 4),
            "mined_at": self.mined_at,
        }


# ── Miner ─────────────────────────────────────────────────────────────────────

class EdgeMiner:
    """Unsupervised edge discovery from closed trade history.

    Parameters
    ----------
    audit_dir : Path
        Directory containing ``trade_audit_*.jsonl`` files.
    n_clusters : int
        Number of K-Means clusters.
    min_cluster_size : int
        Clusters with fewer trades are ignored.
    min_win_rate : float
        Win-rate threshold for an EdgeProfile.
    max_boost : float
        Maximum confidence boost per matched edge profile.
    lookback_days : int
        How far back to read trade audit files.
    """

    def __init__(
        self,
        audit_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        n_clusters: int = _DEFAULT_N_CLUSTERS,
        min_cluster_size: int = _DEFAULT_MIN_CLUSTER,
        min_win_rate: float = _DEFAULT_MIN_WIN_RATE,
        max_boost: float = _DEFAULT_MAX_BOOST,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._audit_dir = Path(audit_dir) if audit_dir else None
        self._data_dir = Path(data_dir) if data_dir else None
        self._n_clusters = n_clusters
        self._min_cluster = min_cluster_size
        self._min_win_rate = min_win_rate
        self._max_boost = max_boost
        self._lookback_days = lookback_days

        self._profiles: List[EdgeProfile] = []
        self._last_mined: Optional[float] = None

        if self._data_dir:
            self._load_profiles()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_and_mine(self) -> List[EdgeProfile]:
        """Load trade history and run clustering. Returns discovered EdgeProfiles."""
        trades = self._load_trades()
        if len(trades) < self._min_cluster * 2:
            return []
        self._profiles = self._mine(trades)
        self._last_mined = time.time()
        if self._data_dir:
            self._save_profiles()
        return self._profiles

    def get_confidence_boost(self, context: dict) -> float:
        """Return confidence boost if context matches a known edge profile.

        Parameters
        ----------
        context : dict
            Keys: ``regime``, ``hour_block``, ``signal_source``,
            ``vix`` (optional), ``day_of_week`` (optional).
        """
        if not self._profiles:
            return 0.0

        regime = str(context.get("regime", "neutral")).lower()
        hour_block = str(context.get("hour_block", _now_hour_block()))
        signal_source = str(context.get("signal_source", "unknown"))

        best_boost = 0.0
        for profile in self._profiles:
            # Require regime AND (hour_block OR signal_source) match
            regime_match = profile.regime == regime or regime in profile.regime
            hour_match = profile.hour_block == hour_block
            source_match = profile.signal_source == signal_source
            if regime_match and (hour_match or source_match):
                best_boost = max(best_boost, profile.confidence_boost)

        return float(best_boost)

    @property
    def profiles(self) -> List[EdgeProfile]:
        return list(self._profiles)

    def get_report(self) -> dict:
        return {
            "total_profiles": len(self._profiles),
            "last_mined": self._last_mined,
            "profiles": [p.to_dict() for p in self._profiles],
        }

    # ── Mining ────────────────────────────────────────────────────────────────

    def _mine(self, trades: List[dict]) -> List[EdgeProfile]:
        """Run K-Means and extract high-win-rate clusters."""
        from sklearn.cluster import KMeans  # type: ignore[import]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import]

        X, labels = self._build_features(trades)
        if len(X) < self._min_cluster:
            return []

        n_clust = min(self._n_clusters, len(X) // max(1, self._min_cluster))
        if n_clust < 2:
            return []

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X_scaled)

        # Group trades by cluster
        by_cluster: Dict[int, List[dict]] = defaultdict(list)
        for trade, cid in zip(trades, cluster_ids):
            by_cluster[int(cid)].append(trade)

        profiles: List[EdgeProfile] = []
        for cid, group in by_cluster.items():
            if len(group) < self._min_cluster:
                continue
            wins = sum(1 for t in group if float(t.get("pnl_pct", 0) or 0) > 0)
            wr = wins / len(group)
            if wr < self._min_win_rate:
                continue

            avg_pnl = sum(float(t.get("pnl_pct", 0) or 0) for t in group) / len(group)
            # Representative features (plurality)
            regimes = [str(t.get("regime", "neutral")).lower() for t in group]
            hours = [_hour_block(str(t.get("ts", ""))) for t in group]
            sources = [_dominant_source(t.get("components", {})) for t in group]

            dominant_regime = max(set(regimes), key=regimes.count)
            dominant_hour = max(set(hours), key=hours.count)
            dominant_source = max(set(sources), key=sources.count)

            # Boost proportional to win rate excess over threshold
            excess = wr - self._min_win_rate
            boost = float(min(self._max_boost, excess * self._max_boost * 3.0))

            name = f"{dominant_regime}_{dominant_hour}_{dominant_source}"
            profiles.append(EdgeProfile(
                cluster_id=cid,
                name=name,
                regime=dominant_regime,
                hour_block=dominant_hour,
                signal_source=dominant_source,
                n_trades=len(group),
                win_rate=round(wr, 4),
                avg_pnl_pct=round(avg_pnl, 6),
                confidence_boost=round(boost, 4),
                centroid=list(km.cluster_centers_[cid]),
            ))

        return sorted(profiles, key=lambda p: p.win_rate, reverse=True)

    @staticmethod
    def _build_features(trades: List[dict]) -> Tuple[np.ndarray, List[str]]:
        rows = []
        labels = []
        for t in trades:
            try:
                vix = float(t.get("vix", t.get("vix_at_entry", 20.0)) or 20.0)
                hold = float(t.get("hold_hours", 4.0) or 4.0)
                row = [
                    _regime_code(str(t.get("regime", "neutral"))),
                    _hour_block_code(str(t.get("ts", ""))),
                    _source_code(t.get("components", {})),
                    _vix_bucket(vix),
                    float(t.get("day_of_week", 0) or 0),
                    _hold_bucket(hold),
                ]
                rows.append(row)
                labels.append(str(t.get("symbol", "")))
            except Exception:
                pass
        return np.array(rows, dtype=float) if rows else np.empty((0, 6)), labels

    # ── Trade loading ─────────────────────────────────────────────────────────

    def _load_trades(self) -> List[dict]:
        dirs = []
        if self._audit_dir:
            dirs.append(self._audit_dir)
        if self._data_dir:
            dirs.extend([
                self._data_dir / "users" / "admin" / "audit",
                self._data_dir / "audit",
            ])

        cutoff = time.time() - self._lookback_days * 86400.0
        trades: List[dict] = []
        seen = set()
        for d in dirs:
            if not d or not Path(d).exists():
                continue
            for path in sorted(Path(d).glob("trade_audit_*.jsonl")):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get("event") != "EXIT":
                            continue
                        ts_str = str(row.get("ts", "") or "")
                        try:
                            ts = datetime.fromisoformat(
                                ts_str.replace("Z", "+00:00")
                            ).timestamp()
                        except Exception:
                            ts = time.time()
                        if ts < cutoff:
                            continue
                        trades.append(row)
                except Exception:
                    pass
        return trades

    # ── Persistence ───────────────────────────────────────────────────────────

    def _profiles_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "edge_miner_profiles.json"

    def _save_profiles(self) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            state = {
                "profiles": [p.to_dict() for p in self._profiles],
                "mined_at": self._last_mined,
            }
            tmp = self._profiles_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._profiles_path())
        except Exception:
            pass

    def _load_profiles(self) -> None:
        try:
            p = self._profiles_path()
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._last_mined = raw.get("mined_at")
            for d in raw.get("profiles", []):
                self._profiles.append(EdgeProfile(
                    cluster_id=int(d.get("cluster_id", 0)),
                    name=d.get("name", ""),
                    regime=d.get("regime", "neutral"),
                    hour_block=d.get("hour_block", "unknown"),
                    signal_source=d.get("signal_source", "unknown"),
                    n_trades=int(d.get("n_trades", 0)),
                    win_rate=float(d.get("win_rate", 0.0)),
                    avg_pnl_pct=float(d.get("avg_pnl_pct", 0.0)),
                    confidence_boost=float(d.get("confidence_boost", 0.0)),
                    mined_at=float(d.get("mined_at", time.time())),
                ))
        except Exception:
            pass
