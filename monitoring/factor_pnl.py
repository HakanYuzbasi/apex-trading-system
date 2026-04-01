"""
monitoring/factor_pnl.py
─────────────────────────
Live Factor P&L Decomposition.

Reads closed-trade attribution data and decomposes realised P&L into
factor contributions:
    ml        — ML signal alpha component
    technical — technical analysis signal
    sentiment — sentiment signal
    momentum  — momentum signal
    residual  — unexplained (actual P&L minus sum of components)

Also breaks down by:
    asset_class  (EQUITY / CRYPTO / FX)
    regime       (bull / neutral / bear / …)
    time_bucket  (last_1d / last_7d / last_30d)

Output is a FactorPnlReport with per-factor attribs and a JSON-serialisable
summary dict for the API endpoint and frontend panel.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class FactorBucket:
    factor: str
    pnl_sum: float = 0.0
    pnl_pct_sum: float = 0.0
    trade_count: int = 0
    win_count: int = 0

    @property
    def avg_pnl_pct(self) -> Optional[float]:
        return self.pnl_pct_sum / self.trade_count if self.trade_count else None

    @property
    def win_rate(self) -> Optional[float]:
        return self.win_count / self.trade_count if self.trade_count else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor":        self.factor,
            "pnl_sum":       round(self.pnl_sum, 2),
            "pnl_pct_sum":   round(self.pnl_pct_sum, 4),
            "avg_pnl_pct":   round(self.avg_pnl_pct or 0.0, 4),
            "win_rate":      round(self.win_rate or 0.0, 3),
            "trade_count":   self.trade_count,
        }


@dataclass
class FactorPnlReport:
    generated_at: str = ""
    lookback_days: int = 7
    total_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    # By factor: list[FactorBucket]
    by_factor: List[FactorBucket] = field(default_factory=list)

    # By asset class: {asset_class → {factor → FactorBucket}}
    by_asset: Dict[str, Dict[str, FactorBucket]] = field(default_factory=dict)

    # By regime: {regime → {factor → FactorBucket}}
    by_regime: Dict[str, Dict[str, FactorBucket]] = field(default_factory=dict)

    # Time breakdown
    last_1d: List[FactorBucket] = field(default_factory=list)
    last_7d: List[FactorBucket] = field(default_factory=list)
    last_30d: List[FactorBucket] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at":  self.generated_at,
            "lookback_days": self.lookback_days,
            "total_trades":  self.total_trades,
            "total_pnl":     round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "by_factor":     [b.to_dict() for b in self.by_factor],
            "by_asset":      {
                ac: {f: b.to_dict() for f, b in fmap.items()}
                for ac, fmap in self.by_asset.items()
            },
            "by_regime":     {
                r: {f: b.to_dict() for f, b in fmap.items()}
                for r, fmap in self.by_regime.items()
            },
            "last_1d":       [b.to_dict() for b in self.last_1d],
            "last_7d":       [b.to_dict() for b in self.last_7d],
            "last_30d":      [b.to_dict() for b in self.last_30d],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_FACTORS = ("ml", "technical", "sentiment", "momentum", "residual")

def _empty_factor_map() -> Dict[str, FactorBucket]:
    return {f: FactorBucket(factor=f) for f in _FACTORS}


def _dominant_factor(rec: Dict) -> str:
    """Return the factor with the highest absolute signal value in this trade."""
    candidates = {
        "ml":        abs(float(rec.get("ml_signal",        rec.get("ml", 0.0)))),
        "technical": abs(float(rec.get("tech_signal",      rec.get("technical", 0.0)))),
        "sentiment": abs(float(rec.get("sentiment_signal", rec.get("sentiment", 0.0)))),
        "momentum":  abs(float(rec.get("cs_momentum_signal",
                                       rec.get("momentum", 0.0)))),
    }
    best = max(candidates, key=lambda k: candidates[k])
    return best if candidates[best] > 0.001 else "residual"


def _factor_weights(rec: Dict) -> Dict[str, float]:
    """
    Proportional weights for each factor from signal magnitudes.
    Returns weights that sum to 1.0 (or 0 if all signals absent → residual=1).
    """
    raw = {
        "ml":        abs(float(rec.get("ml_signal",        rec.get("ml", 0.0)))),
        "technical": abs(float(rec.get("tech_signal",      rec.get("technical", 0.0)))),
        "sentiment": abs(float(rec.get("sentiment_signal", rec.get("sentiment", 0.0)))),
        "momentum":  abs(float(rec.get("cs_momentum_signal",
                                       rec.get("momentum", 0.0)))),
    }
    total = sum(raw.values())
    if total < 1e-9:
        return {"ml": 0.0, "technical": 0.0, "sentiment": 0.0, "momentum": 0.0, "residual": 1.0}
    weights = {k: v / total for k, v in raw.items()}
    weights["residual"] = 0.0
    return weights


def _load_closed_trades(data_dir: Path, lookback_days: int) -> List[Dict]:
    """Load EXIT rows from trade_audit_*.jsonl within lookback window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
    rows: List[Dict] = []
    for jf in sorted((data_dir / "users").glob("*/audit/trade_audit_*.jsonl"), reverse=True):
        try:
            for line in jf.read_text().strip().split("\n"):
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event_type") != "EXIT":
                    continue
                try:
                    ts = datetime.fromisoformat(rec.get("timestamp", "")).timestamp()
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                rows.append(rec)
        except Exception as exc:
            logger.debug("factor_pnl audit read error %s: %s", jf, exc)
    return rows


def _also_try_attribution_jsonl(data_dir: Path, lookback_days: int) -> List[Dict]:
    """Also read from PerformanceAttribution closed records if present."""
    p = data_dir / "performance_attribution.json"
    if not p.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
    try:
        state = json.loads(p.read_text())
        closed = state.get("closed_trades", [])
        rows = []
        for rec in closed:
            try:
                ts = datetime.fromisoformat(rec.get("exit_time", "")).timestamp()
            except Exception:
                continue
            if ts < cutoff:
                continue
            rows.append(rec)
        return rows
    except Exception:
        return []


def _accumulate(
    bucket_map: Dict[str, FactorBucket],
    pnl: float,
    pnl_pct: float,
    weights: Dict[str, float],
) -> None:
    is_win = pnl_pct > 0
    for factor, w in weights.items():
        b = bucket_map[factor]
        b.pnl_sum     += pnl * w
        b.pnl_pct_sum += pnl_pct * w
        if w > 0:
            b.trade_count += 1
            if is_win:
                b.win_count += 1


# ── Main class ─────────────────────────────────────────────────────────────────

class FactorPnlAnalyzer:
    """
    Live Factor P&L Decomposition.

    Usage
    ─────
        analyzer = FactorPnlAnalyzer(data_dir="data")
        report   = analyzer.build_report(lookback_days=7)
        json_out = report.to_dict()
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._data_dir = Path(data_dir)

    def build_report(self, lookback_days: int = 7) -> FactorPnlReport:
        report = FactorPnlReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            lookback_days=lookback_days,
        )

        # Combine audit rows + attribution JSONL
        rows = _load_closed_trades(self._data_dir, lookback_days)
        rows += _also_try_attribution_jsonl(self._data_dir, lookback_days)

        if not rows:
            return report

        # Global factor map
        global_map = _empty_factor_map()
        asset_maps: Dict[str, Dict[str, FactorBucket]] = defaultdict(lambda: _empty_factor_map())
        regime_maps: Dict[str, Dict[str, FactorBucket]] = defaultdict(lambda: _empty_factor_map())

        # Time window maps
        now_ts = datetime.now(timezone.utc).timestamp()
        cutoffs = {
            "last_1d":  now_ts - 86400,
            "last_7d":  now_ts - 7 * 86400,
            "last_30d": now_ts - 30 * 86400,
        }
        window_maps: Dict[str, Dict[str, FactorBucket]] = {
            k: _empty_factor_map() for k in cutoffs
        }

        for rec in rows:
            try:
                pnl_pct = float(rec.get("pnl_pct", rec.get("pnl", 0.0)))
                pnl     = float(rec.get("realized_pnl",
                                        rec.get("pnl_dollars", 0.0)))
                if pnl == 0.0 and pnl_pct != 0.0:
                    # Approximate from P&L % if absolute not available
                    pnl = pnl_pct * float(rec.get("entry_value",
                                                   rec.get("notional", 10_000.0)))
            except (TypeError, ValueError):
                continue

            weights    = _factor_weights(rec)
            asset_cls  = str(rec.get("asset_class", "EQUITY")).upper()
            regime     = str(rec.get("regime",      "neutral")).lower()

            _accumulate(global_map, pnl, pnl_pct, weights)
            _accumulate(asset_maps[asset_cls], pnl, pnl_pct, weights)
            _accumulate(regime_maps[regime],   pnl, pnl_pct, weights)

            report.total_trades += 1
            report.total_pnl    += pnl
            report.total_pnl_pct += pnl_pct

            # Time windows
            try:
                ts = datetime.fromisoformat(
                    rec.get("timestamp", rec.get("exit_time", ""))
                ).timestamp()
            except Exception:
                ts = 0.0
            for wname, wcut in cutoffs.items():
                if ts >= wcut:
                    _accumulate(window_maps[wname], pnl, pnl_pct, weights)

        # Assemble report
        report.by_factor = sorted(
            global_map.values(),
            key=lambda b: abs(b.pnl_pct_sum),
            reverse=True,
        )
        report.by_asset  = dict(asset_maps)
        report.by_regime = dict(regime_maps)
        report.last_1d   = list(window_maps["last_1d"].values())
        report.last_7d   = list(window_maps["last_7d"].values())
        report.last_30d  = list(window_maps["last_30d"].values())

        return report
