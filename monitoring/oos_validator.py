"""Out-of-Sample (OOS) Validator.

Slices the trade audit by (regime, hour_block, signal_source) cells and
computes per-cell win rate, average P&L, and Sharpe to identify which
combinations of regime + time window + signal source are genuinely
profitable out-of-sample.

This complements walkforward_validator.py (which focuses on rolling
monthly periods) by providing a cross-sectional breakdown useful for:
  - Identifying which signal sources work in which regimes
  - Finding hour blocks with consistently poor performance
  - Pruning low-alpha regime×signal combinations

Usage::

    from monitoring.oos_validator import OOSValidator
    v = OOSValidator()
    report = v.build_report()
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_AUDIT_GLOB = "trade_audit_*.jsonl"
_DEFAULT_AUDIT_DIR = Path("data/users/admin/audit")
_MIN_TRADES_CELL = 3   # minimum trades for a cell to be reported
_COMPONENTS = ("ml", "tech", "sentiment", "momentum", "pairs", "earnings", "intraday_mr")
_HOUR_BLOCKS = {
    "open": (9, 11),   # 09–10 UTC
    "midday": (11, 15),
    "close": (15, 17),
    "extended": (17, 24),
    "overnight": (0, 9),
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CellStats:
    cell: str           # "{regime}|{hour_block}|{signal_source}"
    regime: str
    hour_block: str
    signal_source: str
    trades: int
    wins: int
    win_rate: float
    avg_pnl_pct: float
    sharpe: float
    avg_hold_hours: float
    edge_score: float   # win_rate × avg_pnl_pct (quality score)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_score"] = round(d["edge_score"], 6)
        return d


@dataclass
class OOSReport:
    total_trades: int
    cells_analyzed: int
    best_cells: List[CellStats]    # top N by edge_score
    worst_cells: List[CellStats]   # bottom N by edge_score
    by_regime: Dict[str, dict]     # regime-level aggregates
    by_signal_source: Dict[str, dict]
    by_hour_block: Dict[str, dict]
    regime_signal_matrix: Dict[str, Dict[str, float]]  # regime → signal → win_rate

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "cells_analyzed": self.cells_analyzed,
            "best_cells": [c.to_dict() for c in self.best_cells],
            "worst_cells": [c.to_dict() for c in self.worst_cells],
            "by_regime": self.by_regime,
            "by_signal_source": self.by_signal_source,
            "by_hour_block": self.by_hour_block,
            "regime_signal_matrix": self.regime_signal_matrix,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class OOSValidator:
    """Load trade audit and compute per-cell out-of-sample metrics."""

    def __init__(
        self,
        audit_dir: Optional[Path] = None,
        min_trades: int = _MIN_TRADES_CELL,
    ) -> None:
        self._audit_dir = audit_dir or _DEFAULT_AUDIT_DIR
        self._min_trades = min_trades
        self._trades: List[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def load_trades(self) -> int:
        """Load all EXIT rows from trade_audit_*.jsonl files.

        Returns number of trades loaded.
        """
        self._trades = []
        try:
            for path in sorted(self._audit_dir.glob(_AUDIT_GLOB)):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            if row.get("event") == "EXIT" or row.get("exit_reason"):
                                self._trades.append(row)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("OOSValidator: failed to load trades: %s", e)
        return len(self._trades)

    def build_report(
        self,
        top_n: int = 10,
        auto_load: bool = True,
    ) -> OOSReport:
        """Build the full OOS report.

        Args:
            top_n: Number of best/worst cells to include.
            auto_load: If True, calls ``load_trades()`` first.
        """
        if auto_load:
            self.load_trades()

        if not self._trades:
            return OOSReport(
                total_trades=0,
                cells_analyzed=0,
                best_cells=[],
                worst_cells=[],
                by_regime={},
                by_signal_source={},
                by_hour_block={},
                regime_signal_matrix={},
            )

        # Bin trades into cells
        cell_bins: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        for trade in self._trades:
            regime = _normalize_regime(trade.get("regime", "unknown"))
            hour_block = _hour_block_for(trade)
            source = _dominant_component(trade)
            cell_bins[(regime, hour_block, source)].append(trade)

        # Compute cell stats
        cell_stats: List[CellStats] = []
        for (regime, hour_block, source), trades in cell_bins.items():
            if len(trades) < self._min_trades:
                continue
            stats = _compute_cell_stats(regime, hour_block, source, trades)
            cell_stats.append(stats)

        cell_stats.sort(key=lambda c: c.edge_score, reverse=True)

        # Aggregates
        by_regime = _aggregate_by(self._trades, key_fn=lambda t: _normalize_regime(t.get("regime", "unknown")), min_n=1)
        by_source = _aggregate_by(self._trades, key_fn=_dominant_component, min_n=1)
        by_hour = _aggregate_by(self._trades, key_fn=_hour_block_for, min_n=1)
        matrix = _regime_signal_matrix(self._trades)

        return OOSReport(
            total_trades=len(self._trades),
            cells_analyzed=len(cell_stats),
            best_cells=cell_stats[:top_n],
            worst_cells=list(reversed(cell_stats[-top_n:])) if len(cell_stats) >= top_n else list(reversed(cell_stats)),
            by_regime=by_regime,
            by_signal_source=by_source,
            by_hour_block=by_hour,
            regime_signal_matrix=matrix,
        )

    def get_prune_candidates(
        self,
        win_rate_threshold: float = 0.40,
        min_trades: int = 5,
    ) -> List[dict]:
        """Return cells with win_rate < threshold and enough data to be meaningful.

        These are candidates for signal/regime combinations to suppress.
        """
        report = self.build_report(auto_load=False)
        candidates = []
        for cell in report.best_cells + report.worst_cells:
            if cell.trades >= min_trades and cell.win_rate < win_rate_threshold:
                candidates.append({
                    "cell": cell.cell,
                    "regime": cell.regime,
                    "hour_block": cell.hour_block,
                    "signal_source": cell.signal_source,
                    "win_rate": cell.win_rate,
                    "avg_pnl_pct": cell.avg_pnl_pct,
                    "trades": cell.trades,
                })
        return sorted(candidates, key=lambda x: x["win_rate"])


# ── Helper functions ──────────────────────────────────────────────────────────

def _normalize_regime(r: str) -> str:
    if not r or r == "unknown":
        return "unknown"
    return str(r).lower().strip()


def _hour_block_for(trade: dict) -> str:
    """Return the hour block label for a trade based on its timestamp."""
    ts_str = trade.get("ts") or trade.get("exit_ts") or trade.get("timestamp") or ""
    if not ts_str:
        return "unknown"
    try:
        # Parse hour from ISO string "2026-03-22T14:35:00Z"
        hour = int(ts_str[11:13])
        for block, (lo, hi) in _HOUR_BLOCKS.items():
            if lo <= hour < hi:
                return block
        return "unknown"
    except Exception:
        return "unknown"


def _dominant_component(trade: dict) -> str:
    """Return the signal source with the largest absolute component value."""
    components = trade.get("components") or trade.get("signal_components") or {}
    if not components:
        # Fallback: check top-level keys
        for key in _COMPONENTS:
            if key in trade:
                return key
        return "unknown"
    best_key, best_val = "unknown", 0.0
    for key in _COMPONENTS:
        v = abs(float(components.get(key, 0.0) or 0.0))
        if v > best_val:
            best_val, best_key = v, key
    return best_key if best_val > 0.0 else "unknown"


def _compute_cell_stats(
    regime: str, hour_block: str, source: str, trades: List[dict]
) -> CellStats:
    pnl_pcts = [float(t.get("pnl_pct", 0.0) or 0.0) for t in trades]
    wins = sum(1 for p in pnl_pcts if p > 0)
    win_rate = wins / len(pnl_pcts)
    avg_pnl = sum(pnl_pcts) / len(pnl_pcts)

    hold_hours_list = [float(t.get("hold_hours", 0.0) or 0.0) for t in trades]
    avg_hold = sum(hold_hours_list) / len(hold_hours_list)

    sharpe = _sharpe(pnl_pcts)
    edge = win_rate * max(0.0, avg_pnl)

    return CellStats(
        cell=f"{regime}|{hour_block}|{source}",
        regime=regime,
        hour_block=hour_block,
        signal_source=source,
        trades=len(trades),
        wins=wins,
        win_rate=round(win_rate, 4),
        avg_pnl_pct=round(avg_pnl, 4),
        sharpe=round(sharpe, 4),
        avg_hold_hours=round(avg_hold, 2),
        edge_score=round(edge, 6),
    )


def _aggregate_by(
    trades: List[dict],
    key_fn,
    min_n: int = 1,
) -> Dict[str, dict]:
    bins: Dict[str, List[dict]] = defaultdict(list)
    for t in trades:
        bins[key_fn(t)].append(t)
    out: Dict[str, dict] = {}
    for k, ts in bins.items():
        if len(ts) < min_n:
            continue
        pnls = [float(t.get("pnl_pct", 0.0) or 0.0) for t in ts]
        wins = sum(1 for p in pnls if p > 0)
        out[k] = {
            "trades": len(ts),
            "win_rate": round(wins / len(ts), 4),
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 4),
            "sharpe": round(_sharpe(pnls), 4),
        }
    return out


def _regime_signal_matrix(trades: List[dict]) -> Dict[str, Dict[str, float]]:
    """regime → signal_source → win_rate."""
    bins: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for t in trades:
        r = _normalize_regime(t.get("regime", "unknown"))
        s = _dominant_component(t)
        pnl = float(t.get("pnl_pct", 0.0) or 0.0)
        bins[(r, s)].append(pnl)

    matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (r, s), pnls in bins.items():
        if len(pnls) < 2:
            continue
        wins = sum(1 for p in pnls if p > 0)
        matrix[r][s] = round(wins / len(pnls), 4)
    return dict(matrix)


def _sharpe(pnl_pcts: List[float], ann_factor: float = 252.0) -> float:
    """Annualised Sharpe from a list of per-trade P&L %."""
    n = len(pnl_pcts)
    if n < 2:
        return 0.0
    mean = sum(pnl_pcts) / n
    variance = sum((p - mean) ** 2 for p in pnl_pcts) / (n - 1)
    # Guard against near-zero variance (e.g. all-same values with float rounding)
    if variance < 1e-14:
        return 0.0
    return round(mean / math.sqrt(variance) * math.sqrt(ann_factor), 4)
