"""
monitoring/tca_report.py - Transaction Cost Analysis (TCA) Report

Reads live execution data and produces a structured report covering:
1. Per-symbol execution quality (slippage, latency, fill rate)
2. P&L attribution (signal alpha vs execution drag)
3. Rejection analysis (circuit breaker, risk guard, etc.)
4. Overall execution health score

Can be run standalone (python -m monitoring.tca_report) or called via API.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data loading helpers ──────────────────────────────────────────────────────

def _norm(sym: str) -> str:
    """Normalize symbol: strip CRYPTO:/FX: prefix for grouping."""
    for prefix in ("CRYPTO:", "FX:"):
        if sym.startswith(prefix):
            return sym[len(prefix):]
    return sym


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _load_performance_attribution(path: Path) -> Dict:
    if not path.exists():
        return {"open_positions": {}, "closed_trades": []}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"open_positions": {}, "closed_trades": []}


# ── Core analysis ─────────────────────────────────────────────────────────────

def _analyze_closed_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Per-symbol and aggregate P&L + slippage from closed trade history."""
    by_sym: Dict[str, Dict] = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0,
        "gross_pnl": 0.0, "net_pnl": 0.0, "commissions": 0.0,
        "execution_drag": 0.0, "entry_slippage_bps": [],
        "exit_slippage_bps": [], "holding_hours": [],
        "exit_reasons": defaultdict(int),
    })

    for t in trades:
        sym = _norm(t.get("symbol", "UNKNOWN"))
        d = by_sym[sym]
        d["trades"] += 1
        net = float(t.get("net_pnl", 0))
        gross = float(t.get("gross_pnl", 0))
        drag = float(t.get("modeled_execution_drag", 0))
        comm = float(t.get("commissions", 0))
        d["net_pnl"] += net
        d["gross_pnl"] += gross
        d["commissions"] += comm
        d["execution_drag"] += drag
        if net > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

        esl = t.get("entry_slippage_bps")
        xsl = t.get("exit_slippage_bps")
        if esl is not None:
            d["entry_slippage_bps"].append(float(esl))
        if xsl is not None:
            d["exit_slippage_bps"].append(float(xsl))

        h = t.get("holding_hours")
        if h is not None:
            d["holding_hours"].append(float(h))

        reason = t.get("exit_reason", "unknown")
        # Bucket exit reasons
        if "Excellence" in reason or "Weak signal" in reason or "No signal" in reason:
            d["exit_reasons"]["excellence_exit"] += 1
        elif "Take profit" in reason or "take_profit" in reason:
            d["exit_reasons"]["take_profit"] += 1
        elif "Stop" in reason or "stop_loss" in reason:
            d["exit_reasons"]["stop_loss"] += 1
        elif "trailing" in reason.lower():
            d["exit_reasons"]["trailing_stop"] += 1
        else:
            d["exit_reasons"]["other"] += 1

    return dict(by_sym)


def _analyze_latency(records: List[Dict]) -> Dict[str, Any]:
    """Per-symbol latency stats from execution_latency.jsonl."""
    by_sym: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        sym = _norm(r.get("symbol", "UNKNOWN"))
        ms = r.get("total_ms") or r.get("order_to_fill_ms")
        if ms is not None:
            by_sym[sym].append(float(ms))

    result = {}
    for sym, vals in by_sym.items():
        vals_s = sorted(vals)
        n = len(vals_s)
        result[sym] = {
            "fills": n,
            "median_ms": vals_s[n // 2],
            "p95_ms": vals_s[int(n * 0.95)] if n >= 20 else vals_s[-1],
            "max_ms": vals_s[-1],
        }
    return result


def _analyze_rejections(records: List[Dict]) -> Dict[str, Any]:
    """Rejection counts by symbol and reason from trade_rejections.jsonl."""
    by_sym: Dict[str, Dict] = defaultdict(lambda: defaultdict(int))
    reason_totals: Dict[str, int] = defaultdict(int)

    for r in records:
        sym = _norm(r.get("symbol", "UNKNOWN"))
        reason = r.get("reason", "unknown")
        # Bucket
        if "circuit_breaker" in reason.lower():
            bucket = "circuit_breaker"
        elif "risk_guard" in reason.lower() or "pretrade" in reason.lower():
            bucket = "risk_guard"
        elif "daily_loss" in reason.lower():
            bucket = "daily_loss_limit"
        elif "drawdown" in reason.lower():
            bucket = "drawdown_limit"
        elif "spread" in reason.lower():
            bucket = "spread_too_wide"
        elif "cooldown" in reason.lower():
            bucket = "cooldown"
        else:
            bucket = "other"

        by_sym[sym][bucket] += 1
        reason_totals[bucket] += 1

    return {
        "by_symbol": {sym: dict(counts) for sym, counts in by_sym.items()},
        "totals": dict(reason_totals),
        "total_rejections": sum(reason_totals.values()),
    }


def _analyze_open_positions(positions: Dict) -> Dict[str, Any]:
    """Slippage on current open book."""
    result = {}
    for sym, pos in positions.items():
        esl = pos.get("entry_slippage_bps", 0)
        result[_norm(sym)] = {
            "entry_slippage_bps": float(esl),
            "entry_signal": pos.get("entry_signal", 0),
            "entry_time": pos.get("entry_time"),
            "notional": float(pos.get("quantity", 0)) * float(pos.get("entry_price", 0)),
        }
    return result


# ── Executive summary ─────────────────────────────────────────────────────────

def _execution_health_score(
    trade_stats: Dict, total_rejections: int, n_trades: int
) -> float:
    """
    Composite 0-100 execution health score.
    Penalizes: high slippage, high rejection rate, low fill rate, high excellence exits.
    """
    score = 100.0

    # Win rate component (40 pts)
    total_wins = sum(v["wins"] for v in trade_stats.values())
    if n_trades > 0:
        win_rate = total_wins / n_trades
        score -= max(0, (0.50 - win_rate)) * 80   # -1pt per 1.25% below 50% win rate

    # Rejection rate (20 pts)
    if n_trades > 0:
        rej_ratio = total_rejections / max(n_trades, 1)
        score -= min(20, rej_ratio * 0.5)

    # Excellence exit rate (20 pts) — excellence exits = signal problem
    total_exc = sum(v["exit_reasons"].get("excellence_exit", 0) for v in trade_stats.values())
    if n_trades > 0:
        exc_rate = total_exc / n_trades
        score -= exc_rate * 30

    # Slippage penalty (20 pts)
    all_slip = []
    for v in trade_stats.values():
        all_slip.extend(v["exit_slippage_bps"])
    if all_slip:
        avg_slip = sum(all_slip) / len(all_slip)
        score -= min(20, max(0, (avg_slip - 10) * 0.5))

    return max(0.0, min(100.0, round(score, 1)))


# ── Main entry point ──────────────────────────────────────────────────────────

def build_tca_report(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Build the full TCA report dict. Call from API or CLI.
    """
    from config import ApexConfig
    data_dir = Path(data_dir or ApexConfig.DATA_DIR)
    admin_dir = data_dir / "users" / "admin"
    audit_dir = admin_dir / "audit"

    # Load all data sources
    perf = _load_performance_attribution(admin_dir / "performance_attribution.json")
    latency_recs = _load_jsonl(audit_dir / "execution_latency.jsonl")
    rejection_recs = _load_jsonl(audit_dir / "trade_rejections.jsonl")

    closed_trades = perf.get("closed_trades", [])
    open_positions = perf.get("open_positions", {})

    # Analyze
    trade_stats = _analyze_closed_trades(closed_trades)
    latency_stats = _analyze_latency(latency_recs)
    rejection_stats = _analyze_rejections(rejection_recs)
    open_stats = _analyze_open_positions(open_positions)

    n_trades = len(closed_trades)
    total_net_pnl = sum(v["net_pnl"] for v in trade_stats.values())
    total_drag = sum(v["execution_drag"] for v in trade_stats.values())
    total_wins = sum(v["wins"] for v in trade_stats.values())
    total_rejections = rejection_stats["total_rejections"]

    health_score = _execution_health_score(trade_stats, total_rejections, n_trades)

    # Merge per-symbol data into unified table
    all_syms = set(trade_stats) | set(latency_stats) | set(open_stats)
    per_symbol: List[Dict] = []
    for sym in sorted(all_syms):
        ts = trade_stats.get(sym, {})
        ls = latency_stats.get(sym, {})
        os_ = open_stats.get(sym, {})

        entry_slips = ts.get("entry_slippage_bps", [])
        exit_slips = ts.get("exit_slippage_bps", [])

        per_symbol.append({
            "symbol": sym,
            "closed_trades": ts.get("trades", 0),
            "win_rate_pct": round(ts["wins"] / ts["trades"] * 100, 1) if ts.get("trades") else None,
            "net_pnl": round(ts.get("net_pnl", 0), 2),
            "execution_drag": round(ts.get("execution_drag", 0), 2),
            "avg_entry_slip_bps": round(sum(entry_slips) / len(entry_slips), 1) if entry_slips else (
                round(os_.get("entry_slippage_bps", 0), 1) if os_ else None
            ),
            "avg_exit_slip_bps": round(sum(exit_slips) / len(exit_slips), 1) if exit_slips else None,
            "median_fill_ms": ls.get("median_ms"),
            "p95_fill_ms": ls.get("p95_ms"),
            "fills": ls.get("fills", 0),
            "exit_reasons": dict(ts.get("exit_reasons", {})),
            "open_position": bool(os_),
            "rejections": rejection_stats["by_symbol"].get(sym, {}),
        })

    # Sort by abs(net_pnl) descending
    per_symbol.sort(key=lambda x: abs(x["net_pnl"]), reverse=True)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "closed_trades": n_trades,
            "win_rate_pct": round(total_wins / n_trades * 100, 1) if n_trades else 0,
            "total_net_pnl": round(total_net_pnl, 2),
            "total_execution_drag": round(total_drag, 2),
            "alpha_before_costs": round(total_net_pnl + total_drag, 2),
            "cost_ratio_pct": round(total_drag / max(abs(total_net_pnl + total_drag), 1) * 100, 1),
            "total_fills": len(latency_recs),
            "total_rejections": total_rejections,
            "rejection_breakdown": rejection_stats["totals"],
            "execution_health_score": health_score,
        },
        "per_symbol": per_symbol,
        "open_book": open_stats,
    }


def print_tca_report(report: Dict) -> None:
    """Print a formatted TCA report to stdout."""
    s = report["summary"]
    print("\n" + "=" * 72)
    print("  APEX TRADING — TRANSACTION COST ANALYSIS (TCA)")
    print(f"  {report['generated_at'][:19]} UTC")
    print("=" * 72)
    print(f"\n{'EXECUTION HEALTH SCORE':.<40} {s['execution_health_score']:.1f} / 100")
    print(f"{'Closed trades':.<40} {s['closed_trades']}")
    print(f"{'Win rate':.<40} {s['win_rate_pct']:.1f}%")
    print(f"{'Net P&L':.<40} ${s['total_net_pnl']:+,.2f}")
    print(f"{'Gross alpha (before costs)':.<40} ${s['alpha_before_costs']:+,.2f}")
    print(f"{'Execution drag (commissions+slip)':.<40} ${s['total_execution_drag']:+,.2f}")
    print(f"{'Cost ratio':.<40} {s['cost_ratio_pct']:.1f}% of gross alpha")
    print(f"{'Total fills':.<40} {s['total_fills']}")
    print(f"{'Total rejections':.<40} {s['total_rejections']}")
    if s["rejection_breakdown"]:
        for reason, count in sorted(s["rejection_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {reason:<36} {count:>6}")

    print(f"\n{'─' * 72}")
    header = f"{'Symbol':<14} {'Trades':>6} {'WR%':>5} {'NetPnL':>9} {'Drag':>8} {'eSl':>6} {'xSl':>6} {'MedMs':>7} {'P95Ms':>7}"
    print(header)
    print("─" * 72)

    for row in report["per_symbol"]:
        if row["closed_trades"] == 0:
            continue
        sym = row["symbol"][:13]
        wr = f"{row['win_rate_pct']:.0f}%" if row["win_rate_pct"] is not None else "  —"
        esl = f"{row['avg_entry_slip_bps']:.0f}" if row["avg_entry_slip_bps"] is not None else "  —"
        xsl = f"{row['avg_exit_slip_bps']:.0f}" if row["avg_exit_slip_bps"] is not None else "  —"
        mms = f"{row['median_fill_ms']:.0f}" if row["median_fill_ms"] else "  —"
        p95 = f"{row['p95_fill_ms']:.0f}" if row["p95_fill_ms"] else "  —"

        # Flag problematic rows
        flag = ""
        if (row["avg_exit_slip_bps"] or 0) > 80:
            flag += " ⚠SLIP"
        exc = row["exit_reasons"].get("excellence_exit", 0)
        tot = row["closed_trades"]
        if tot > 0 and exc / tot > 0.7:
            flag += " ⚠EXCEL"

        print(
            f"{sym:<14} {row['closed_trades']:>6} {wr:>5} "
            f"${row['net_pnl']:>8,.0f} ${row['execution_drag']:>7,.0f} "
            f"{esl:>6} {xsl:>6} {mms:>7} {p95:>7}{flag}"
        )

    # Exit reason breakdown
    print(f"\n{'─' * 72}")
    print("EXIT REASON BREAKDOWN (closed trades only):")
    reason_totals: Dict[str, int] = defaultdict(int)
    for row in report["per_symbol"]:
        for r, c in row["exit_reasons"].items():
            reason_totals[r] += c
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        pct = count / max(s["closed_trades"], 1) * 100
        bar = "█" * int(pct / 3)
        print(f"  {reason:<28} {count:>4} ({pct:>5.1f}%)  {bar}")

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    report = build_tca_report()
    print_tca_report(report)
