#!/usr/bin/env python3
"""
scripts/daily_report.py - APEX Daily P&L Morning Report

Generates a concise daily trading summary covering:
- P&L vs previous sessions
- Open book (unrealized)
- Today's closed trades with exit reasons
- Execution quality snapshot
- System health (circuit breaker, VIX, kill switch)
- Week-over-week trend

Usage:
  python scripts/daily_report.py              # Print to stdout
  python scripts/daily_report.py --save       # Also save to data/reports/
  python scripts/daily_report.py --date 2026-03-07  # Historical date

Can also be imported and called from the API:
  from scripts.daily_report import build_daily_report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ApexConfig


# ── Loaders ──────────────────────────────────────────────────────────────────

def _j(path: Path) -> Any:
    """Load JSON file, return {} on failure."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _load_daily_history(data_dir: Path, days: int = 7) -> List[Dict]:
    """Load last N daily_pnl_*.json files sorted oldest→newest."""
    audit_dir = data_dir / "users" / "admin" / "audit"
    files = sorted(audit_dir.glob("daily_pnl_*.json"))[-days:]
    results = []
    for f in files:
        d = _j(f)
        if d:
            results.append(d)
    return results


def _load_open_positions(data_dir: Path) -> Dict:
    perf = _j(data_dir / "users" / "admin" / "performance_attribution.json")
    return perf.get("open_positions", {})


def _load_closed_trades(data_dir: Path, since_date: Optional[str] = None) -> List[Dict]:
    """Load closed trades, optionally filtered to on/after since_date (YYYY-MM-DD)."""
    perf = _j(data_dir / "users" / "admin" / "performance_attribution.json")
    trades = perf.get("closed_trades", [])
    if since_date:
        trades = [
            t for t in trades
            if (t.get("exit_time") or "")[:10] >= since_date
        ]
    return trades


def _load_risk_state(data_dir: Path) -> Dict:
    return _j(data_dir / "users" / "admin" / "risk_state.json")


def _load_rejections_today(data_dir: Path, date_str: str) -> int:
    f = data_dir / "users" / "admin" / "audit" / "trade_rejections.jsonl"
    if not f.exists():
        return 0
    count = 0
    for line in f.read_text().splitlines():
        if date_str in line:
            count += 1
    return count


def _load_calibration(data_dir: Path) -> Dict:
    return _j(data_dir / "calibrated_thresholds.json")


# ── Unrealized P&L ───────────────────────────────────────────────────────────

def _get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for open positions via yfinance (best-effort)."""
    prices: Dict[str, float] = {}
    try:
        import yfinance as yf
        # Strip CRYPTO:/FX: prefixes, convert / to -
        yf_syms = {}
        for sym in symbols:
            base = sym.replace("CRYPTO:", "").replace("FX:", "").replace("/", "-")
            yf_syms[base] = sym
        if not yf_syms:
            return prices
        tickers = yf.download(
            list(yf_syms.keys()), period="1d", interval="5m",
            progress=False, auto_adjust=True, threads=True,
        )
        close = tickers.get("Close") if hasattr(tickers, "get") else tickers["Close"] if "Close" in tickers else None
        if close is None:
            return prices
        for yf_sym, orig_sym in yf_syms.items():
            try:
                col = close[yf_sym] if yf_sym in close.columns else None
                if col is not None and len(col.dropna()) > 0:
                    prices[orig_sym] = float(col.dropna().iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return prices


# ── Report builder ────────────────────────────────────────────────────────────

def build_daily_report(
    data_dir: Optional[Path] = None,
    target_date: Optional[str] = None,
    fetch_prices: bool = True,
) -> Dict[str, Any]:
    """
    Build the full daily report dict.
    target_date: YYYY-MM-DD string (default: today)
    """
    data_dir = Path(data_dir or ApexConfig.DATA_DIR)
    today = target_date or datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Load data ──────────────────────────────────────────────────────────
    history = _load_daily_history(data_dir, days=7)
    open_pos = _load_open_positions(data_dir)
    trades_today = _load_closed_trades(data_dir, since_date=today)
    risk_state = _load_risk_state(data_dir)
    calibration = _load_calibration(data_dir)
    rejections_today = _load_rejections_today(data_dir, today)

    # Find today's daily snapshot (may not exist yet)
    today_snap = next((h for h in reversed(history) if h.get("date") == today), {})
    yesterday_snap = next((h for h in reversed(history) if h.get("date") == yesterday), {})

    # ── Unrealized P&L ────────────────────────────────────────────────────
    unrealized_by_sym: Dict[str, Dict] = {}
    if open_pos and fetch_prices:
        prices = _get_current_prices(list(open_pos.keys()))
        for sym, pos in open_pos.items():
            entry = float(pos.get("entry_price", 0))
            qty = float(pos.get("quantity", 0))
            curr = prices.get(sym, 0)
            if curr > 0 and entry > 0:
                unreal_pnl = (curr - entry) * qty
                pnl_pct = (curr / entry - 1) * 100
                unrealized_by_sym[sym] = {
                    "entry_price": entry,
                    "current_price": curr,
                    "quantity": qty,
                    "unrealized_pnl": round(unreal_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "holding_hours": round(
                        (datetime.now() - datetime.fromisoformat(pos["entry_time"].replace("Z", "")))
                        .total_seconds() / 3600, 1
                    ) if pos.get("entry_time") else None,
                }
    else:
        for sym, pos in open_pos.items():
            unrealized_by_sym[sym] = {
                "entry_price": float(pos.get("entry_price", 0)),
                "current_price": None,
                "quantity": float(pos.get("quantity", 0)),
                "unrealized_pnl": None,
                "pnl_pct": None,
            }

    total_unrealized = sum(
        v["unrealized_pnl"] for v in unrealized_by_sym.values()
        if v["unrealized_pnl"] is not None
    )

    # ── Closed trades today ───────────────────────────────────────────────
    realized_today = sum(float(t.get("net_pnl", 0)) for t in trades_today)
    exit_buckets: Dict[str, int] = defaultdict(int)
    for t in trades_today:
        r = t.get("exit_reason", "")
        if "Excellence" in r or "Weak signal" in r or "No signal" in r:
            exit_buckets["excellence_exit"] += 1
        elif "Take profit" in r or "take_profit" in r:
            exit_buckets["take_profit"] += 1
        elif "Stop" in r:
            exit_buckets["stop_loss"] += 1
        else:
            exit_buckets["other"] += 1

    # ── Circuit breaker ───────────────────────────────────────────────────
    cb = risk_state.get("circuit_breaker", {})
    cb_tripped = cb.get("is_tripped", False)

    # ── 7-day P&L trend ───────────────────────────────────────────────────
    trend = []
    for snap in history:
        pnl = snap.get("daily_pnl")
        pnl_pct = snap.get("daily_pnl_pct")
        if pnl is not None:
            trend.append({
                "date": snap.get("date"),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4) if pnl_pct else None,
                "regime": snap.get("regime"),
            })

    week_pnl = sum(d["pnl"] for d in trend if d["pnl"] is not None)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "report_date": today,
        "daily": {
            "realized_pnl": round(realized_today, 2),
            "unrealized_pnl": round(total_unrealized, 2) if unrealized_by_sym else None,
            "total_pnl": round(realized_today + total_unrealized, 2),
            "capital_start": today_snap.get("day_start_capital"),
            "capital_end": today_snap.get("end_capital") or risk_state.get("peak_capital"),
            "max_drawdown_pct": today_snap.get("max_drawdown_pct"),
            "fills_buy": today_snap.get("fills_buy", 0),
            "fills_sell": today_snap.get("fills_sell", 0),
            "trades_closed": len(trades_today),
            "exit_breakdown": dict(exit_buckets),
            "rejections": rejections_today,
            "regime": today_snap.get("regime", "unknown"),
        },
        "open_positions": unrealized_by_sym,
        "closed_trades_today": [
            {
                "symbol": t.get("symbol", "").replace("CRYPTO:", ""),
                "net_pnl": round(float(t.get("net_pnl", 0)), 2),
                "pnl_bps": round(float(t.get("pnl_bps_on_entry_notional", 0)), 1),
                "exit_reason": t.get("exit_reason", ""),
                "holding_hours": round(float(t.get("holding_hours", 0)), 1),
            }
            for t in sorted(trades_today, key=lambda x: float(x.get("net_pnl", 0)), reverse=True)
        ],
        "system_health": {
            "circuit_breaker_tripped": cb_tripped,
            "circuit_breaker_reason": cb.get("reason") if cb_tripped else None,
            "starting_capital": risk_state.get("starting_capital"),
            "peak_capital": risk_state.get("peak_capital"),
            "exit_threshold_pct": calibration.get("weak_signal_loss_threshold_pct"),
            "slippage_blacklist": calibration.get("slippage_blacklist", []),
            "calibrated_at": calibration.get("calibrated_at"),
        },
        "trend_7d": trend,
        "week_pnl": round(week_pnl, 2),
    }


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_report(r: Dict) -> str:
    lines = []
    sep = "═" * 68
    thin = "─" * 68

    lines += [
        f"\n{sep}",
        f"  APEX DAILY REPORT — {r['report_date']}",
        f"  Generated: {r['generated_at'][:19]} UTC",
        sep,
    ]

    d = r["daily"]
    sh = r["system_health"]

    # ── System health banner ──────────────────────────────────────────────
    if sh["circuit_breaker_tripped"]:
        lines.append(f"\n  🚨 CIRCUIT BREAKER TRIPPED: {sh['circuit_breaker_reason']}")
    else:
        lines.append(f"\n  ✅ System OK | VIX regime: {d['regime'].upper()}")

    # ── P&L summary ───────────────────────────────────────────────────────
    lines += [f"\n{thin}", "  TODAY'S P&L", thin]
    realized = d["realized_pnl"]
    unrealized = d["unrealized_pnl"]
    total = d["total_pnl"]

    lines.append(f"  {'Realized (closed today)':<35} ${realized:>+10,.2f}")
    if unrealized is not None:
        lines.append(f"  {'Unrealized (open book)':<35} ${unrealized:>+10,.2f}")
        lines.append(f"  {'Total (realized + unrealized)':<35} ${total:>+10,.2f}")
    if d["capital_start"]:
        lines.append(f"  {'Capital (start of day)':<35} ${d['capital_start']:>10,.2f}")
    if d["max_drawdown_pct"]:
        lines.append(f"  {'Max intraday drawdown':<35} {d['max_drawdown_pct']:>9.2f}%")

    # ── Activity ──────────────────────────────────────────────────────────
    lines += [f"\n{thin}", "  ACTIVITY", thin]
    lines.append(f"  {'Trades closed today':<35} {d['trades_closed']:>6}")
    lines.append(f"  {'Fills (buy/sell)':<35} {d['fills_buy']:>3} / {d['fills_sell']:>3}")
    lines.append(f"  {'Order rejections':<35} {d['rejections']:>6}")
    if d["exit_breakdown"]:
        lines.append("  Exit reasons:")
        for reason, count in sorted(d["exit_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"    {reason:<32} {count:>4}")

    # ── Open positions ────────────────────────────────────────────────────
    open_pos = r["open_positions"]
    if open_pos:
        lines += [f"\n{thin}", f"  OPEN POSITIONS ({len(open_pos)})", thin]
        sorted_pos = sorted(
            open_pos.items(),
            key=lambda x: x[1].get("unrealized_pnl") or 0,
            reverse=True
        )
        lines.append(f"  {'Symbol':<14} {'Qty':>10} {'Entry':>9} {'Current':>9} {'Unreal PnL':>12} {'%':>6} {'Hold h':>7}")
        for sym, pos in sorted_pos:
            curr = f"${pos['current_price']:.4f}" if pos.get("current_price") else "  ???"
            unr = f"${pos['unrealized_pnl']:>+,.0f}" if pos.get("unrealized_pnl") is not None else "     ???"
            pct = f"{pos['pnl_pct']:>+.1f}%" if pos.get("pnl_pct") is not None else "   ???"
            hold = f"{pos['holding_hours']:.0f}h" if pos.get("holding_hours") else "  ???"
            lines.append(
                f"  {sym[:13]:<14} {pos['quantity']:>10,.2f} "
                f"${pos['entry_price']:>8.4f} {curr:>9} {unr:>12} {pct:>6} {hold:>7}"
            )

    # ── Today's closed trades ─────────────────────────────────────────────
    closed = r["closed_trades_today"]
    if closed:
        lines += [f"\n{thin}", "  CLOSED TRADES TODAY", thin]
        for t in closed:
            sym = t["symbol"][:14]
            pnl = t["net_pnl"]
            reason = t["exit_reason"][:40]
            flag = "✅" if pnl > 0 else "❌"
            lines.append(f"  {flag} {sym:<14} ${pnl:>+8,.2f}  {t['holding_hours']:>4.1f}h  {reason}")

    # ── 7-day trend ───────────────────────────────────────────────────────
    trend = r["trend_7d"]
    if trend:
        lines += [f"\n{thin}", "  7-DAY P&L TREND", thin]
        for day in trend:
            pnl = day["pnl"]
            pct = f"{day['pnl_pct']:>+.2f}%" if day.get("pnl_pct") else ""
            bar_len = max(0, min(20, int(abs(pnl) / 100)))
            bar = ("█" if pnl >= 0 else "░") * bar_len
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  {day['date']}  {sign}${pnl:>8,.0f} {pct:>8}  {bar}")
        lines.append(f"\n  {'Week total':<35} ${r['week_pnl']:>+10,.2f}")

    # ── System config ─────────────────────────────────────────────────────
    lines += [f"\n{thin}", "  SYSTEM CONFIG", thin]
    if sh["exit_threshold_pct"]:
        lines.append(f"  {'Excellence exit threshold':<35} {sh['exit_threshold_pct']:>+.3f}%")
    if sh["slippage_blacklist"]:
        lines.append(f"  {'Slippage blacklist':<35} {', '.join(sh['slippage_blacklist'])}")
    if sh["starting_capital"]:
        lines.append(f"  {'Starting capital':<35} ${sh['starting_capital']:>10,.2f}")
    if sh["peak_capital"]:
        lines.append(f"  {'Peak capital':<35} ${sh['peak_capital']:>10,.2f}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


# ── API helper ─────────────────────────────────────────────────────────────────

def build_and_format(data_dir: Optional[Path] = None, target_date: Optional[str] = None) -> str:
    """Convenience wrapper: build + format in one call."""
    report = build_daily_report(data_dir=data_dir, target_date=target_date)
    return format_report(report)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="APEX Daily P&L Report")
    parser.add_argument("--date", help="Report date YYYY-MM-DD (default: today)")
    parser.add_argument("--save", action="store_true", help="Save report to data/reports/")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    report = build_daily_report(target_date=args.date)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    text = format_report(report)
    print(text)

    if args.save:
        reports_dir = Path(ApexConfig.DATA_DIR) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        fname = f"daily_{report['report_date']}.txt"
        (reports_dir / fname).write_text(text)
        print(f"Saved to {reports_dir / fname}")


if __name__ == "__main__":
    main()
