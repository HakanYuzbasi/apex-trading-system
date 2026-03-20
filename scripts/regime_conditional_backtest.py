#!/usr/bin/env python3
"""
scripts/regime_conditional_backtest.py - APEX Regime-Conditional Backtest

Reads trade audit JSONL files (data/users/admin/audit/trade_audit_YYYYMMDD.jsonl)
and the mandate copilot audit log, then produces a regime-conditional performance
breakdown showing how the system behaves across different market regimes.

Usage:
  python3 scripts/regime_conditional_backtest.py
  python3 scripts/regime_conditional_backtest.py --date 2026-03-18
  python3 scripts/regime_conditional_backtest.py --save
  python3 scripts/regime_conditional_backtest.py --date 2026-03-18 --save

The primary trade data source is the per-day trade audit files.
The mandate audit log is used for an additional mandate evaluation breakdown section.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config import ApexConfig
    DATA_DIR = Path(ApexConfig.DATA_DIR)
except Exception:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file, silently skipping malformed lines."""
    records: List[Dict] = []
    if not path.exists():
        return records
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            # Silently skip malformed lines
            pass
    return records


def _discover_trade_audit_files(data_dir: Path, date_filter: Optional[str]) -> List[Path]:
    """
    Find trade_audit_YYYYMMDD.jsonl files under data/users/admin/audit/.
    If date_filter (YYYY-MM-DD) is given, only load that day's file.
    Returns files sorted chronologically.
    """
    audit_dir = data_dir / "users" / "admin" / "audit"
    if not audit_dir.exists():
        return []

    if date_filter:
        compact = date_filter.replace("-", "")
        candidate = audit_dir / f"trade_audit_{compact}.jsonl"
        return [candidate] if candidate.exists() else []

    return sorted(audit_dir.glob("trade_audit_*.jsonl"))


# ─── Trade loading & normalisation ────────────────────────────────────────────

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string to UTC datetime."""
    if not ts_str:
        return None
    try:
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def _compute_pnl_pct(record: Dict) -> Optional[float]:
    """
    Try to obtain pnl_pct from the record.
    Priority:
      1. pnl_pct field (already a percentage, e.g. -0.415 means -0.415%)
      2. Compute from entry_price / exit_price / fill_price fields
    """
    val = record.get("pnl_pct")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

    # Fallback: compute from price fields
    entry = record.get("entry_price") or record.get("entry_fill_price")
    exit_ = record.get("exit_price") or record.get("fill_price")
    if entry and exit_:
        try:
            return (float(exit_) / float(entry) - 1.0) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def _load_exit_trades(data_dir: Path, date_filter: Optional[str]) -> List[Dict]:
    """
    Load and normalise EXIT events from trade audit files.
    Only EXIT events carry pnl_pct and can be used for performance analysis.
    ENTRY events are loaded only to pair entry timestamps with exits via symbol+fill_price matching.
    """
    files = _discover_trade_audit_files(data_dir, date_filter)

    # First pass: collect all ENTRY events by symbol for hold-time calculation
    entries_by_symbol: Dict[str, List[Dict]] = defaultdict(list)
    raw_exits: List[Dict] = []

    for fpath in files:
        for rec in _load_jsonl(fpath):
            event = (rec.get("event") or "").upper()
            sym = rec.get("symbol", "")
            if event == "ENTRY":
                entries_by_symbol[sym].append(rec)
            elif event == "EXIT":
                raw_exits.append(rec)

    # Second pass: normalise exit records
    trades: List[Dict] = []
    for rec in raw_exits:
        pnl_pct = _compute_pnl_pct(rec)
        if pnl_pct is None:
            # Cannot compute any return — skip
            continue

        sym = rec.get("symbol", "unknown")
        regime = rec.get("regime") or "unknown"
        broker = rec.get("broker") or "unknown"
        ts = _parse_ts(rec.get("ts"))
        signal = rec.get("signal")
        entry_signal = rec.get("entry_signal")
        confidence = rec.get("confidence")
        exit_reason = rec.get("exit_reason") or ""
        holding_days = rec.get("holding_days")

        # Determine hold time in hours
        hold_hours: Optional[float] = None
        if holding_days is not None:
            try:
                hold_hours = float(holding_days) * 24.0
            except (TypeError, ValueError):
                pass

        # If holding_days is 0 and we have ts, try to find the matching ENTRY by symbol
        # to estimate intra-day hold time from timestamps
        if (hold_hours is None or hold_hours == 0.0) and ts and sym in entries_by_symbol:
            # Find the most recent ENTRY before this EXIT
            best_entry_ts: Optional[datetime] = None
            for entry_rec in entries_by_symbol[sym]:
                ets = _parse_ts(entry_rec.get("ts"))
                if ets and ets <= ts:
                    if best_entry_ts is None or ets > best_entry_ts:
                        best_entry_ts = ets
            if best_entry_ts:
                delta_h = (ts - best_entry_ts).total_seconds() / 3600.0
                if delta_h >= 0:
                    hold_hours = delta_h

        trades.append({
            "symbol": sym,
            "regime": regime,
            "broker": broker,
            "ts": ts,
            "pnl_pct": pnl_pct,
            "pnl_usd": rec.get("pnl_usd"),
            "signal": signal,
            "entry_signal": entry_signal,
            "confidence": confidence,
            "exit_reason": exit_reason,
            "hold_hours": hold_hours,
        })

    return trades


# ─── Statistics ───────────────────────────────────────────────────────────────

def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _safe_mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def _annualised_sharpe(returns_pct: List[float], avg_hold_hours: Optional[float]) -> float:
    """
    Annualised Sharpe ratio from trade returns.
    Formula: mean(r) / std(r) * sqrt(252 / avg_hold_days)
    Returns 0.0 if fewer than 3 trades or std == 0.
    """
    if len(returns_pct) < 3:
        return 0.0
    m = _safe_mean(returns_pct)
    s = _safe_std(returns_pct)
    if s == 0.0:
        return 0.0
    hold_days = (avg_hold_hours / 24.0) if avg_hold_hours and avg_hold_hours > 0 else 1.0
    # Clamp to avoid unrealistic annualisation from very short hold times
    hold_days = max(hold_days, 1.0 / 24.0)  # minimum 1 hour
    ann_factor = math.sqrt(252.0 / hold_days)
    return m / s * ann_factor


def _compute_regime_stats(trades: List[Dict]) -> Dict:
    """Compute all per-regime statistics from a list of normalised exit trades."""
    returns = [t["pnl_pct"] for t in trades]
    hold_hours = [t["hold_hours"] for t in trades if t["hold_hours"] is not None]

    n = len(trades)
    wins = [r for r in returns if r > 0]
    win_rate = len(wins) / n * 100.0 if n > 0 else 0.0
    avg_pnl = _safe_mean(returns)
    total_pnl = sum(returns)
    avg_hold = _safe_mean(hold_hours) if hold_hours else None
    sharpe = _annualised_sharpe(returns, avg_hold)

    best = max(trades, key=lambda t: t["pnl_pct"]) if trades else None
    worst = min(trades, key=lambda t: t["pnl_pct"]) if trades else None

    return {
        "n": n,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "avg_hold_hours": avg_hold,
        "sharpe": sharpe,
        "best": best,
        "worst": worst,
        "trades": trades,
    }


def _group_by_regime(trades: List[Dict]) -> Dict[str, Dict]:
    """Group trades by regime field and compute stats for each group."""
    by_regime: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        by_regime[t["regime"]].append(t)

    # Sort regimes in a logical order
    order = ["strong_bull", "bull", "neutral", "bear", "strong_bear", "volatile", "crisis", "unknown"]
    result: Dict[str, Dict] = {}
    for regime in order:
        if regime in by_regime:
            result[regime] = _compute_regime_stats(by_regime[regime])
    # Any regimes not in the predefined order (e.g. typos, new additions)
    for regime in sorted(by_regime.keys()):
        if regime not in result:
            result[regime] = _compute_regime_stats(by_regime[regime])
    return result


def _group_by_broker(trades: List[Dict]) -> Dict[str, Dict]:
    """Group trades by broker (ibkr / alpaca / unknown)."""
    by_broker: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        by_broker[t["broker"]].append(t)
    return {b: _compute_regime_stats(ts) for b, ts in sorted(by_broker.items())}


def _group_by_exit_reason(trades: List[Dict]) -> Dict[str, int]:
    """Bucket exit reasons into broad categories and count."""
    buckets: Dict[str, int] = defaultdict(int)
    for t in trades:
        r = t.get("exit_reason") or ""
        if "Excellence" in r or "Weak signal" in r:
            buckets["excellence_exit"] += 1
        elif "Take profit" in r or "take_profit" in r or "TP" in r:
            buckets["take_profit"] += 1
        elif "Stop" in r or "stop" in r:
            buckets["stop_loss"] += 1
        elif "Hedge" in r or "hedge" in r:
            buckets["hedge_exit"] += 1
        elif "max_hold" in r or "max hold" in r.lower():
            buckets["max_hold"] += 1
        elif "mismatch" in r.lower() or "quick" in r.lower():
            buckets["quick_mismatch"] += 1
        else:
            buckets["other"] += 1
    return dict(buckets)


# ─── Mandate audit helpers ────────────────────────────────────────────────────

def _load_mandate_stats(data_dir: Path, date_filter: Optional[str]) -> Dict:
    """
    Load mandate_copilot_audit.jsonl and compute summary statistics.
    Fields used: timestamp, event, model_version, recommendation_mode,
                 response_summary.feasible, response_summary.feasibility_band,
                 response_summary.probability_target_hit, response_summary.confidence.
    """
    path = data_dir / "mandate_copilot_audit.jsonl"
    records = _load_jsonl(path)

    if date_filter:
        records = [
            r for r in records
            if (r.get("timestamp") or "")[:10] == date_filter
        ]

    total = len(records)
    if total == 0:
        return {"total": 0}

    feasible_count = sum(
        1 for r in records
        if (r.get("response_summary") or {}).get("feasible") is True
    )

    by_band: Dict[str, int] = defaultdict(int)
    by_model: Dict[str, int] = defaultdict(int)
    by_mode: Dict[str, int] = defaultdict(int)
    confidences: List[float] = []
    hit_probs: List[float] = []

    for r in records:
        rs = r.get("response_summary") or {}
        band = rs.get("feasibility_band") or "unknown"
        by_band[band] += 1

        mv = r.get("model_version") or "unknown"
        by_model[mv] += 1

        mode = r.get("recommendation_mode") or "unknown"
        by_mode[mode] += 1

        c = rs.get("confidence")
        if c is not None:
            try:
                confidences.append(float(c))
            except (TypeError, ValueError):
                pass

        p = rs.get("probability_target_hit")
        if p is not None:
            try:
                hit_probs.append(float(p))
            except (TypeError, ValueError):
                pass

    return {
        "total": total,
        "feasible_count": feasible_count,
        "infeasible_count": total - feasible_count,
        "feasibility_rate_pct": (feasible_count / total * 100.0) if total else 0.0,
        "avg_confidence": _safe_mean(confidences),
        "avg_probability_target_hit": _safe_mean(hit_probs),
        "by_feasibility_band": dict(by_band),
        "by_model_version": dict(by_model),
        "by_recommendation_mode": dict(by_mode),
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def _fmt_pct(v: Optional[float], sign: bool = True) -> str:
    if v is None:
        return "   N/A"
    if sign:
        return f"{v:>+.2f}%"
    return f"{v:.2f}%"


def _fmt_hold(h: Optional[float]) -> str:
    if h is None:
        return "  N/A"
    if h < 1.0:
        return f"{h*60:.0f}m"
    return f"{h:.1f}h"


def _fmt_sharpe(v: float) -> str:
    if v == 0.0:
        return "  n/a"
    return f"{v:>+.2f}"


def _regime_table_row(
    label: str,
    stats: Dict,
    col_widths: Tuple[int, int, int, int, int, int, int],
) -> str:
    cw = col_widths
    n = stats["n"]
    win = f"{stats['win_rate']:.0f}%"
    avg_pnl = _fmt_pct(stats["avg_pnl"])
    total_pnl = _fmt_pct(stats["total_pnl"])
    sharpe = _fmt_sharpe(stats["sharpe"])
    hold = _fmt_hold(stats["avg_hold_hours"])
    return (
        f" {label:<{cw[0]}} \u2502"
        f" {n:>{cw[1]}} \u2502"
        f" {win:>{cw[2]}} \u2502"
        f" {avg_pnl:>{cw[3]}} \u2502"
        f" {total_pnl:>{cw[4]}} \u2502"
        f" {sharpe:>{cw[5]}} \u2502"
        f" {hold:>{cw[6]}}"
    )


def format_report(
    regime_stats: Dict[str, Dict],
    broker_stats: Dict[str, Dict],
    exit_buckets: Dict[str, int],
    mandate_stats: Dict,
    all_stats: Dict,
    date_filter: Optional[str],
    source_files: List[Path],
) -> str:
    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    title = "REGIME-CONDITIONAL BACKTEST REPORT"
    if date_filter:
        title += f"  ({date_filter})"
    box_w = 56
    lines.append("\u2554" + "\u2550" * box_w + "\u2557")
    lines.append("\u2551" + title.center(box_w) + "\u2551")
    lines.append("\u255a" + "\u2550" * box_w + "\u255d")
    lines.append("")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"  Generated : {generated}")
    if source_files:
        lines.append(f"  Data files: {', '.join(str(f.name) for f in source_files)}")
    lines.append("")

    if all_stats["n"] == 0:
        lines.append("  No EXIT trades found for the selected date range.")
        if date_filter:
            lines.append(f"  Trade audit file: data/users/admin/audit/trade_audit_{date_filter.replace('-', '')}.jsonl")
        return "\n".join(lines)

    # ── Regime performance table ───────────────────────────────────────────
    # Column widths: regime, trades, win%, avg pnl, total pnl, sharpe, avg hold
    CW = (16, 6, 5, 8, 10, 6, 8)
    h_regime  = "Regime"
    h_trades  = "Trades"
    h_win     = "Win%"
    h_avg_pnl = "Avg P&L"
    h_tot_pnl = "Total P&L"
    h_sharpe  = "Sharpe"
    h_hold    = "Avg Hold"

    header = (
        f" {h_regime:<{CW[0]}} \u2502"
        f" {h_trades:>{CW[1]}} \u2502"
        f" {h_win:>{CW[2]}} \u2502"
        f" {h_avg_pnl:>{CW[3]}} \u2502"
        f" {h_tot_pnl:>{CW[4]}} \u2502"
        f" {h_sharpe:>{CW[5]}} \u2502"
        f" {h_hold:>{CW[6]}}"
    )
    sep_parts = ["\u2500" * (w + 2) for w in CW]
    sep_middle = "\u253c".join(sep_parts)
    sep_bottom = "\u2534".join(sep_parts)

    lines.append(header)
    lines.append(sep_middle)

    for regime, stats in regime_stats.items():
        lines.append(_regime_table_row(regime, stats, CW))

    lines.append(sep_middle)
    lines.append(_regime_table_row("ALL", all_stats, CW))
    lines.append(sep_bottom)
    lines.append("")

    # ── Best / worst trade callouts ────────────────────────────────────────
    if all_stats.get("best") and all_stats.get("worst"):
        best = all_stats["best"]
        worst = all_stats["worst"]
        lines.append("  Best trade : "
                      f"{best['symbol']:<18} "
                      f"{_fmt_pct(best['pnl_pct']):>8}  "
                      f"regime={best['regime']}  "
                      f"hold={_fmt_hold(best['hold_hours'])}")
        lines.append("  Worst trade: "
                      f"{worst['symbol']:<18} "
                      f"{_fmt_pct(worst['pnl_pct']):>8}  "
                      f"regime={worst['regime']}  "
                      f"hold={_fmt_hold(worst['hold_hours'])}")
        lines.append("")

    # ── Per-broker breakdown ───────────────────────────────────────────────
    lines.append("  Broker Breakdown")
    lines.append("  " + "\u2500" * 54)
    b_header = (
        f" {'Broker':<10} \u2502"
        f" {'Trades':>6} \u2502"
        f" {'Win%':>5} \u2502"
        f" {'Avg P&L':>8} \u2502"
        f" {'Total P&L':>10} \u2502"
        f" {'Sharpe':>6}"
    )
    lines.append(b_header)
    lines.append("  " + "\u2500" * 54)
    for broker, stats in broker_stats.items():
        row = (
            f" {broker:<10} \u2502"
            f" {stats['n']:>6} \u2502"
            f" {stats['win_rate']:.0f}%{' ':>2} \u2502"
            f" {_fmt_pct(stats['avg_pnl']):>8} \u2502"
            f" {_fmt_pct(stats['total_pnl']):>10} \u2502"
            f" {_fmt_sharpe(stats['sharpe']):>6}"
        )
        lines.append(row)
    lines.append("")

    # ── Exit reason breakdown ──────────────────────────────────────────────
    if exit_buckets:
        lines.append("  Exit Reason Breakdown")
        lines.append("  " + "\u2500" * 38)
        total_exits = sum(exit_buckets.values())
        for reason, count in sorted(exit_buckets.items(), key=lambda x: -x[1]):
            pct = count / total_exits * 100.0 if total_exits else 0
            lines.append(f"  {'  ' + reason:<28}  {count:>4}  ({pct:.0f}%)")
        lines.append("")

    # ── Mandate copilot evaluation breakdown ──────────────────────────────
    ms = mandate_stats
    if ms.get("total", 0) > 0:
        lines.append("  Mandate Copilot Evaluation Breakdown")
        lines.append("  " + "\u2500" * 54)
        lines.append(f"  {'Total evaluations':<32} {ms['total']:>6}")
        lines.append(f"  {'Feasible':<32} {ms['feasible_count']:>6}  ({ms['feasibility_rate_pct']:.0f}%)")
        lines.append(f"  {'Infeasible':<32} {ms['infeasible_count']:>6}  ({100 - ms['feasibility_rate_pct']:.0f}%)")
        lines.append(f"  {'Avg confidence':<32} {ms['avg_confidence']:.3f}")
        lines.append(f"  {'Avg probability target hit':<32} {ms['avg_probability_target_hit']:.3f}")
        lines.append("")

        lines.append("  By feasibility band:")
        for band, cnt in sorted(ms["by_feasibility_band"].items()):
            pct = cnt / ms["total"] * 100.0
            lines.append(f"    {band:<24}  {cnt:>4}  ({pct:.0f}%)")
        lines.append("")

        lines.append("  By model version:")
        for mv, cnt in sorted(ms["by_model_version"].items()):
            pct = cnt / ms["total"] * 100.0
            lines.append(f"    {mv:<32}  {cnt:>4}  ({pct:.0f}%)")
        lines.append("")
    else:
        filter_note = f" for date {date_filter}" if date_filter else ""
        lines.append(f"  Mandate Copilot: no evaluations found{filter_note}.")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append("  " + "\u2550" * 54)
    lines.append(f"  Total trades analysed: {all_stats['n']}")
    lines.append("")

    return "\n".join(lines)


# ─── JSON summary ─────────────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serialisable form."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items() if k != "trades"}
    if isinstance(obj, list):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    return obj


def build_json_summary(
    regime_stats: Dict[str, Dict],
    broker_stats: Dict[str, Dict],
    exit_buckets: Dict[str, int],
    mandate_stats: Dict,
    all_stats: Dict,
    date_filter: Optional[str],
) -> Dict:
    def _regime_summary(stats: Dict) -> Dict:
        return {
            "trade_count": stats["n"],
            "win_rate_pct": round(stats["win_rate"], 2),
            "avg_pnl_pct": round(stats["avg_pnl"], 4),
            "total_pnl_pct": round(stats["total_pnl"], 4),
            "avg_hold_hours": round(stats["avg_hold_hours"], 2) if stats["avg_hold_hours"] else None,
            "sharpe_annualised": round(stats["sharpe"], 4),
            "best_trade": {
                "symbol": stats["best"]["symbol"],
                "pnl_pct": round(stats["best"]["pnl_pct"], 4),
                "regime": stats["best"]["regime"],
                "broker": stats["best"]["broker"],
            } if stats.get("best") else None,
            "worst_trade": {
                "symbol": stats["worst"]["symbol"],
                "pnl_pct": round(stats["worst"]["pnl_pct"], 4),
                "regime": stats["worst"]["regime"],
                "broker": stats["worst"]["broker"],
            } if stats.get("worst") else None,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_filter": date_filter,
        "total_trades": all_stats["n"],
        "aggregate": _regime_summary(all_stats),
        "by_regime": {regime: _regime_summary(stats) for regime, stats in regime_stats.items()},
        "by_broker": {broker: _regime_summary(stats) for broker, stats in broker_stats.items()},
        "exit_reason_breakdown": exit_buckets,
        "mandate_evaluation_summary": mandate_stats,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="APEX Regime-Conditional Backtest — analyse trade performance by market regime"
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Filter to a single trading day (e.g. 2026-03-18). "
             "Without this flag all available trade audit files are loaded.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a JSON summary to data/regime_backtest_summary.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of the formatted report",
    )
    args = parser.parse_args()

    date_filter: Optional[str] = args.date

    # Validate date format if provided
    if date_filter:
        try:
            datetime.strptime(date_filter, "%Y-%m-%d")
        except ValueError:
            print(f"ERROR: --date must be YYYY-MM-DD, got: {date_filter!r}", file=sys.stderr)
            sys.exit(1)

    # ── Load trade data ────────────────────────────────────────────────────
    trades = _load_exit_trades(DATA_DIR, date_filter)
    source_files = _discover_trade_audit_files(DATA_DIR, date_filter)

    # ── Load mandate audit data ────────────────────────────────────────────
    mandate_stats = _load_mandate_stats(DATA_DIR, date_filter)

    # ── Compute statistics ─────────────────────────────────────────────────
    regime_stats = _group_by_regime(trades)
    broker_stats = _group_by_broker(trades)
    exit_buckets = _group_by_exit_reason(trades)
    all_stats = _compute_regime_stats(trades)

    # ── Output ────────────────────────────────────────────────────────────
    if args.json:
        summary = build_json_summary(
            regime_stats, broker_stats, exit_buckets,
            mandate_stats, all_stats, date_filter,
        )
        print(json.dumps(_to_json_safe(summary), indent=2))
    else:
        report = format_report(
            regime_stats, broker_stats, exit_buckets,
            mandate_stats, all_stats, date_filter, source_files,
        )
        print(report)

    # ── Optionally save JSON ───────────────────────────────────────────────
    if args.save:
        summary = build_json_summary(
            regime_stats, broker_stats, exit_buckets,
            mandate_stats, all_stats, date_filter,
        )
        out_path = DATA_DIR / "regime_backtest_summary.json"
        out_path.write_text(json.dumps(_to_json_safe(summary), indent=2))
        print(f"  Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
