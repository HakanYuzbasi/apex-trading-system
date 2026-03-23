"""
monitoring/walkforward_validator.py — Walk-Forward Validation Report

Reads trade audit JSONL files and daily P&L snapshots to produce a
rolling walk-forward performance report suitable for the dashboard.

Report contents:
  - Rolling monthly Sharpe (realized vs equal-weight baseline)
  - Per-regime win rate and avg P&L trend by period
  - Signal component alpha decomposition over time
  - Regime time-in-state distribution
  - Slippage trend by period

Usage:
    from monitoring.walkforward_validator import build_walkforward_report
    report = build_walkforward_report()
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_AUDIT_GLOB = "trade_audit_*.jsonl"
_DAILY_GLOB = "daily_pnl_*.json"
_DEFAULT_AUDIT_DIR = Path("data/users/admin/audit")
_COMPONENTS = ("ml", "tech", "sentiment", "momentum", "pairs")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PeriodMetrics:
    period: str              # e.g. "2026-03"
    trades: int
    wins: int
    win_rate: float
    avg_pnl_pct: float
    sharpe: float            # annualised from daily P&L in period
    baseline_sharpe: float   # 50/50 equal-weight buy-and-hold approximation
    max_dd_pct: float
    avg_slippage_bps: float
    regime_counts: Dict[str, int] = field(default_factory=dict)
    component_alpha: Dict[str, float] = field(default_factory=dict)
    gross_pnl_usd: float = 0.0


@dataclass
class RegimePeriod:
    regime: str
    period: str
    trades: int
    wins: int
    win_rate: float
    avg_pnl_pct: float


@dataclass
class WalkForwardReport:
    periods: List[PeriodMetrics]
    regime_trend: List[RegimePeriod]          # per-regime, per-period
    component_alpha_trend: Dict[str, List]    # component → [{period, alpha}]
    regime_distribution: Dict[str, int]       # overall regime trade counts
    overall: Dict                             # aggregate summary
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _period_key(ts: str) -> str:
    """Extract YYYY-MM from ISO timestamp string."""
    return ts[:7]


def _sharpe(daily_pnls: List[float]) -> float:
    """Annualised Sharpe from a list of daily P&L percentages."""
    n = len(daily_pnls)
    if n < 2:
        return 0.0
    mean = sum(daily_pnls) / n
    variance = sum((x - mean) ** 2 for x in daily_pnls) / (n - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return 0.0
    return round((mean / std) * math.sqrt(252), 4)


def _max_drawdown(cumulative_pnls: List[float]) -> float:
    """Max drawdown from list of cumulative equity values (as fractions)."""
    if not cumulative_pnls:
        return 0.0
    peak = cumulative_pnls[0]
    max_dd = 0.0
    for v in cumulative_pnls:
        if v > peak:
            peak = v
        dd = (peak - v) / (1 + abs(peak)) if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 4)


def _load_trade_audit(audit_dir: Path) -> List[dict]:
    """Load all trade audit events sorted by timestamp."""
    rows: List[dict] = []
    for f in sorted(audit_dir.glob(_AUDIT_GLOB)):
        try:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as exc:
            logger.debug("walkforward: skip %s: %s", f.name, exc)
    rows.sort(key=lambda r: r.get("ts", ""))
    return rows


def _load_daily_pnl(audit_dir: Path) -> Dict[str, dict]:
    """Load all daily_pnl snapshots keyed by date (YYYY-MM-DD)."""
    result: Dict[str, dict] = {}
    for f in sorted(audit_dir.glob(_DAILY_GLOB)):
        try:
            data = json.loads(f.read_text())
            date = data.get("date") or f.stem.replace("daily_pnl_", "")
            result[date] = data
        except Exception as exc:
            logger.debug("walkforward: skip %s: %s", f.name, exc)
    return result


def _dominant_component(row: dict) -> Optional[str]:
    """Determine the dominant signal component for an EXIT row."""
    # Exit rows may carry component weights if the strategy rotation wires them in.
    # Fall back to signal heuristic: if entry_signal >= 0.20 → ml, else tech.
    comps: dict = row.get("components", {})
    if comps:
        return max(comps, key=comps.get)
    # heuristic fallback
    sig = abs(float(row.get("entry_signal", row.get("signal", 0.0))))
    return "ml" if sig >= 0.18 else "tech"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_period_metrics(
    period: str,
    exits: List[dict],
    daily_rows: List[dict],
) -> PeriodMetrics:
    trades = len(exits)
    wins = sum(1 for r in exits if float(r.get("pnl_pct", 0)) > 0)
    win_rate = round(wins / trades, 4) if trades else 0.0
    avg_pnl = round(sum(float(r.get("pnl_pct", 0)) for r in exits) / trades, 4) if trades else 0.0
    gross_pnl = sum(float(r.get("pnl_usd", 0)) for r in exits)
    avg_slip = round(
        sum(float(r.get("slippage_bps", 0)) for r in exits) / trades, 2
    ) if trades else 0.0

    # Regime counts
    regime_counts: Dict[str, int] = defaultdict(int)
    for r in exits:
        regime_counts[r.get("regime", "unknown")] += 1

    # Component alpha: mean(pnl_pct × component_weight)
    comp_accum: Dict[str, List[float]] = {c: [] for c in _COMPONENTS}
    for r in exits:
        comps = r.get("components", {})
        pnl = float(r.get("pnl_pct", 0))
        if comps:
            for c in _COMPONENTS:
                comp_accum[c].append(pnl * float(comps.get(c, 0.0)))
        else:
            dom = _dominant_component(r)
            if dom in comp_accum:
                comp_accum[dom].append(pnl)
    component_alpha = {
        c: round(sum(v) / len(v), 6) if v else 0.0
        for c, v in comp_accum.items()
    }

    # Sharpe from daily rows
    daily_pnls = [float(d.get("daily_pnl_pct", 0)) for d in daily_rows]
    sharpe = _sharpe(daily_pnls)

    # Baseline: half the daily P&L (50/50 equity/crypto held passively)
    baseline_pnls = [p * 0.5 for p in daily_pnls]
    baseline_sharpe = _sharpe(baseline_pnls)

    # Max drawdown from cumulative
    cumul = 0.0
    cumul_series: List[float] = []
    for p in daily_pnls:
        cumul += p
        cumul_series.append(cumul)
    max_dd = _max_drawdown(cumul_series)

    return PeriodMetrics(
        period=period,
        trades=trades,
        wins=wins,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl,
        sharpe=sharpe,
        baseline_sharpe=baseline_sharpe,
        max_dd_pct=max_dd,
        avg_slippage_bps=avg_slip,
        regime_counts=dict(regime_counts),
        component_alpha=component_alpha,
        gross_pnl_usd=round(gross_pnl, 2),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_walkforward_report(
    audit_dir: Optional[Path] = None,
    min_trades_per_period: int = 1,
) -> dict:
    """
    Build the full walk-forward validation report.

    Args:
        audit_dir: Override the default audit directory.
        min_trades_per_period: Only include periods with at least this many EXIT trades.

    Returns:
        Dict suitable for JSON serialisation.
    """
    if audit_dir is None:
        audit_dir = Path(_DEFAULT_AUDIT_DIR)

    rows = _load_trade_audit(audit_dir)
    daily_snapshots = _load_daily_pnl(audit_dir)

    # Bucket EXIT rows by period
    exits_by_period: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("event") == "EXIT":
            p = _period_key(row.get("ts", ""))
            exits_by_period[p].append(row)

    # Bucket daily snapshots by period
    daily_by_period: Dict[str, List[dict]] = defaultdict(list)
    for date, snap in sorted(daily_snapshots.items()):
        p = date[:7]
        daily_by_period[p].append(snap)

    all_periods = sorted(
        set(exits_by_period.keys()) | set(daily_by_period.keys())
    )

    # Per-period metrics
    period_metrics: List[PeriodMetrics] = []
    for period in all_periods:
        exits = exits_by_period.get(period, [])
        daily_rows = daily_by_period.get(period, [])
        if len(exits) < min_trades_per_period and not daily_rows:
            continue
        pm = _compute_period_metrics(period, exits, daily_rows)
        period_metrics.append(pm)

    # Per-regime trend across periods
    regime_trend: List[RegimePeriod] = []
    for pm in period_metrics:
        exits = exits_by_period.get(pm.period, [])
        regime_exits: Dict[str, List[dict]] = defaultdict(list)
        for r in exits:
            regime_exits[r.get("regime", "unknown")].append(r)
        for regime, regs in sorted(regime_exits.items()):
            n = len(regs)
            w = sum(1 for r in regs if float(r.get("pnl_pct", 0)) > 0)
            regime_trend.append(RegimePeriod(
                regime=regime,
                period=pm.period,
                trades=n,
                wins=w,
                win_rate=round(w / n, 4) if n else 0.0,
                avg_pnl_pct=round(
                    sum(float(r.get("pnl_pct", 0)) for r in regs) / n, 4
                ) if n else 0.0,
            ))

    # Component alpha trend
    component_alpha_trend: Dict[str, List] = {c: [] for c in _COMPONENTS}
    for pm in period_metrics:
        for c in _COMPONENTS:
            component_alpha_trend[c].append({
                "period": pm.period,
                "alpha": pm.component_alpha.get(c, 0.0),
            })

    # Overall regime distribution
    regime_distribution: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("event") == "EXIT":
            regime_distribution[row.get("regime", "unknown")] += 1

    # Overall summary
    all_exits = [r for r in rows if r.get("event") == "EXIT"]
    total_trades = len(all_exits)
    total_wins = sum(1 for r in all_exits if float(r.get("pnl_pct", 0)) > 0)
    overall_pnl = sum(float(r.get("pnl_pct", 0)) for r in all_exits)
    all_daily = [float(d.get("daily_pnl_pct", 0)) for d in daily_snapshots.values()]

    overall = {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": round(total_wins / total_trades, 4) if total_trades else 0.0,
        "avg_pnl_pct": round(overall_pnl / total_trades, 4) if total_trades else 0.0,
        "sharpe": _sharpe(all_daily),
        "baseline_sharpe": _sharpe([p * 0.5 for p in all_daily]),
        "total_periods": len(period_metrics),
        "gross_pnl_usd": round(
            sum(float(r.get("pnl_usd", 0)) for r in all_exits), 2
        ),
    }

    report = WalkForwardReport(
        periods=period_metrics,
        regime_trend=regime_trend,
        component_alpha_trend=component_alpha_trend,
        regime_distribution=dict(regime_distribution),
        overall=overall,
    )

    # Convert to plain dict for JSON
    result = asdict(report)
    # Flatten periods list so regime_counts stays a dict not a list of tuples
    return result
