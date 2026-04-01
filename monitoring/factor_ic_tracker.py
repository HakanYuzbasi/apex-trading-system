"""
monitoring/factor_ic_tracker.py — Factor Information Coefficient (IC) Tracker

Tracks how predictive each signal component is over recent live trades.
IC = Spearman rank correlation between signal value at entry and realized P&L.

Records:
    - signal_name (str): e.g. "god_level", "mean_reversion", "sector_rotation",
                          "earnings_catalyst", "options_flow", "news_sentiment", etc.
    - signal_value (float): the signal at entry time
    - realized_pnl (float): realized P&L percentage on close

Provides:
    - Per-signal IC (rolling last N trades)
    - Ranked leaderboard: which signals are actually working?
    - Signals with IC < threshold flagged for review/suppression

Usage:
    tracker = FactorICTracker()
    tracker.record_entry(symbol, {"god_level": 0.18, "mean_reversion": 0.42, ...})
    tracker.record_exit(symbol, realized_pnl_pct=0.025)
    report = tracker.get_report()

Config keys:
    FACTOR_IC_ENABLED      = True
    FACTOR_IC_WINDOW        = 50      # rolling window for IC calculation
    FACTOR_IC_MIN_OBS       = 10      # minimum obs per signal to report
    FACTOR_IC_PERSIST_PATH  = "data/factor_ic_state.json"
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEF = {
    "FACTOR_IC_ENABLED":      True,
    "FACTOR_IC_WINDOW":       50,
    "FACTOR_IC_MIN_OBS":      10,
    "FACTOR_IC_PERSIST_PATH": "data/factor_ic_state.json",
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between x and y."""
    n = len(x)
    if n < 3:
        return 0.0
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.std(x_arr) < 1e-9 or np.std(y_arr) < 1e-9:
        return 0.0
    rx = np.argsort(np.argsort(x_arr)).astype(float)
    ry = np.argsort(np.argsort(y_arr)).astype(float)
    # Pearson on ranks = Spearman
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    if den < 1e-9:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


@dataclass
class _EntryRecord:
    symbol: str
    signals: Dict[str, float]    # factor_name → signal_value
    ts: str                       # ISO entry timestamp


@dataclass
class FactorICResult:
    signal_name: str
    ic: float                # Spearman IC (rolling window)
    obs: int                 # number of observations used
    mean_signal: float       # average signal value
    win_rate: float          # fraction of positive-signal trades that were profitable
    is_reliable: bool        # obs >= min_obs
    status: str              # "active", "weak", "unreliable"


@dataclass
class FactorICReport:
    signals: List[FactorICResult]
    top_factors: List[str]    # sorted by IC descending
    weak_factors: List[str]   # IC below IC_WEAK_THRESHOLD
    timestamp: str


class FactorICTracker:
    IC_WEAK_THRESHOLD = 0.05  # IC below this = "weak" signal

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = persist_path or str(_cfg("FACTOR_IC_PERSIST_PATH"))
        # Pending entries awaiting exit
        self._pending: Dict[str, _EntryRecord] = {}
        # Ring buffer of (signal_value, pnl) per factor
        self._observations: Dict[str, List[tuple]] = defaultdict(list)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_entry(self, symbol: str, signal_components: Dict[str, float]) -> None:
        """
        Record the signal values at entry time.
        Call immediately after an entry order is confirmed.
        """
        if not _cfg("FACTOR_IC_ENABLED"):
            return
        self._pending[symbol] = _EntryRecord(
            symbol=symbol,
            signals={k: float(v) for k, v in signal_components.items()},
            ts=datetime.now(timezone.utc).isoformat(),
        )

    def record_exit(self, symbol: str, realized_pnl_pct: float) -> None:
        """
        Record the realized P&L for a closed position.
        Matches against the pending entry for this symbol.
        """
        if not _cfg("FACTOR_IC_ENABLED"):
            return
        entry = self._pending.pop(symbol, None)
        if entry is None:
            return

        pnl = float(realized_pnl_pct)
        window = int(_cfg("FACTOR_IC_WINDOW"))

        for factor, sig_val in entry.signals.items():
            buf = self._observations[factor]
            buf.append((sig_val, pnl))
            if len(buf) > window:
                del buf[:-window]  # keep only latest window

        self._save()

    def get_report(self) -> FactorICReport:
        """Compute and return the current IC report for all tracked factors."""
        min_obs = int(_cfg("FACTOR_IC_MIN_OBS"))
        results: List[FactorICResult] = []

        for factor, obs_list in self._observations.items():
            n = len(obs_list)
            if n == 0:
                continue

            signals = [o[0] for o in obs_list]
            pnls    = [o[1] for o in obs_list]
            ic = _spearman(signals, pnls) if n >= 3 else 0.0
            mean_sig = float(np.mean(signals))
            # Win rate: fraction where signal>0 and pnl>0, or signal<0 and pnl<0
            directional = [
                (s > 0 and p > 0) or (s < 0 and p < 0)
                for s, p in obs_list if abs(s) > 1e-4
            ]
            win_rate = float(sum(directional) / len(directional)) if directional else 0.5

            reliable = n >= min_obs
            if not reliable:
                status = "unreliable"
            elif abs(ic) < self.IC_WEAK_THRESHOLD:
                status = "weak"
            else:
                status = "active"

            results.append(FactorICResult(
                signal_name=factor,
                ic=round(ic, 4),
                obs=n,
                mean_signal=round(mean_sig, 4),
                win_rate=round(win_rate, 3),
                is_reliable=reliable,
                status=status,
            ))

        results.sort(key=lambda r: r.ic, reverse=True)
        top_factors  = [r.signal_name for r in results if r.status == "active"]
        weak_factors = [r.signal_name for r in results if r.status == "weak"]

        return FactorICReport(
            signals=results,
            top_factors=top_factors,
            weak_factors=weak_factors,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_report_dict(self) -> dict:
        """Serialisable version of get_report() for API responses."""
        report = self.get_report()
        return {
            "signals": [asdict(s) for s in report.signals],
            "top_factors": report.top_factors,
            "weak_factors": report.weak_factors,
            "timestamp": report.timestamp,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            path = Path(self._persist_path)
            if not path.exists():
                return
            with open(path) as f:
                state = json.load(f)
            for factor, obs_list in state.get("observations", {}).items():
                self._observations[factor] = [tuple(o) for o in obs_list]
        except Exception as e:
            logger.debug("FactorICTracker: could not load state — %s", e)

    def _save(self) -> None:
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "observations": {k: list(v) for k, v in self._observations.items()},
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("FactorICTracker: could not save state — %s", e)
