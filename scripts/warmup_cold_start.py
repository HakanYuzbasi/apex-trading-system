#!/usr/bin/env python3
"""
scripts/warmup_cold_start.py — Cold-Start Bootstrap (Issue 2)

Synthesises a realistic trade audit seed from the existing equity_curve.csv
so downstream systems (BacktestGate, PerformanceGovernor, BL optimizer,
SignalAutoTuner) have real data to work with from the first cycle instead of
waiting for 20-50 live trades.

What it does:
  1. Reads equity_curve.csv to infer paper-trade entries/exits
  2. Writes ENTRY + EXIT events to data/trade_audit_warmup.jsonl
  3. Seeds performance_governor_state.json with realistic rolling samples
  4. Seeds backtest_gate_state.json with a synthetic evaluation baseline

Usage:
    python scripts/warmup_cold_start.py [--dry-run]

Exit codes: 0 = success, 1 = no usable data, 2 = error
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_equity_curve(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("equity", "portfolio_value", "value"):
                try:
                    r["_val"] = float(r[k])
                    break
                except (KeyError, ValueError):
                    pass
            if "_val" in r:
                try:
                    r["_ts"] = datetime.fromisoformat(
                        r.get("timestamp", "").replace("Z", "+00:00")
                    )
                except Exception:
                    r["_ts"] = datetime.now(timezone.utc)
                rows.append(r)
    return rows


def _infer_trades(rows: list[dict], min_trade_pct: float = 0.003) -> list[dict]:
    """
    Detect significant equity swings as synthetic trade round-trips.
    A 'trade' = equity moves > min_trade_pct between two consecutive readings.
    """
    trades = []
    if len(rows) < 2:
        return trades

    # Downsample to ~hourly buckets to reduce noise
    hourly: dict[str, dict] = {}
    for r in rows:
        bucket = r["_ts"].strftime("%Y-%m-%dT%H")
        hourly[bucket] = r  # last row of hour wins
    sampled = list(hourly.values())

    for i in range(1, len(sampled)):
        prev = sampled[i - 1]["_val"]
        curr = sampled[i]["_val"]
        if prev <= 0:
            continue
        ret = (curr - prev) / prev
        if abs(ret) < min_trade_pct:
            continue
        ts_entry = sampled[i - 1]["_ts"]
        ts_exit = sampled[i]["_ts"]
        side = "BUY" if ret > 0 else "SELL"
        pnl_pct = ret * 100
        trades.append(
            {
                "trade_id": str(uuid.uuid4()),
                "symbol": "CRYPTO:SOL/USD",  # Only live symbol we know exists
                "side": side,
                "entry_ts": ts_entry.isoformat(),
                "exit_ts": ts_exit.isoformat(),
                "entry_price": prev,
                "exit_price": curr,
                "shares": 1.0,
                "gross_pnl_pct": round(pnl_pct, 4),
                "net_pnl_pct": round(pnl_pct * 0.95, 4),  # ~5% slippage/fee
                "win": ret > 0,
                "source": "warmup_synthetic",
            }
        )
    return trades


def _build_audit_events(trades: list[dict]) -> list[dict]:
    events = []
    for t in trades:
        events.append(
            {
                "event": "ENTRY",
                "trade_id": t["trade_id"],
                "symbol": t["symbol"],
                "side": t["side"],
                "timestamp": t["entry_ts"],
                "price": t["entry_price"],
                "shares": t["shares"],
                "source": t["source"],
            }
        )
        events.append(
            {
                "event": "EXIT",
                "trade_id": t["trade_id"],
                "symbol": t["symbol"],
                "side": t["side"],
                "timestamp": t["exit_ts"],
                "price": t["exit_price"],
                "shares": t["shares"],
                "gross_pnl_pct": t["gross_pnl_pct"],
                "net_pnl_pct": t["net_pnl_pct"],
                "win": t["win"],
                "source": t["source"],
            }
        )
    return events


def _governor_samples(rows: list[dict], n: int = 100) -> list:
    """Build rolling portfolio value samples at ~30-min intervals."""
    if len(rows) < 2:
        return []
    step = max(1, len(rows) // n)
    samples = []
    for i in range(0, len(rows), step):
        r = rows[i]
        samples.append([r["_ts"].isoformat(), r["_val"]])
    return samples[-200:]  # cap at 200 matching PerformanceGovernor default


def _rolling_sharpe(samples: list) -> float:
    if len(samples) < 5:
        return 0.0
    vals = [s[1] for s in samples]
    rets = [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals)) if vals[i - 1] > 0]
    if len(rets) < 4:
        return 0.0
    mean_r = sum(rets) / len(rets)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1))
    return round(mean_r / std_r * math.sqrt(252 * 24), 4) if std_r > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apex cold-start warm-up bootstrapper")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written, don't write")
    args = parser.parse_args()

    try:
        from config import ApexConfig
        data_dir = Path(ApexConfig.DATA_DIR)
    except Exception:
        data_dir = Path("data")

    equity_csv = data_dir / "equity_curve.csv"
    rows = _read_equity_curve(equity_csv)
    if not rows:
        print(f"ERROR: No usable rows in {equity_csv}")
        return 1

    trades = _infer_trades(rows)
    print(f"Inferred {len(trades)} synthetic trades from {len(rows)} equity curve rows")

    if not trades:
        print("WARNING: No significant equity swings found — try lowering min_trade_pct")
        return 1

    wins = sum(1 for t in trades if t["win"])
    win_rate = wins / len(trades)
    avg_pnl = sum(t["net_pnl_pct"] for t in trades) / len(trades)
    print(f"  Win rate: {win_rate:.1%}  |  Avg net P&L: {avg_pnl:+.3f}%")

    events = _build_audit_events(trades)
    audit_path = data_dir / "trade_audit_warmup.jsonl"

    samples = _governor_samples(rows)
    sharpe = _rolling_sharpe(samples)

    gov_state = {
        "tier": "NORMAL",
        "consecutive_bad": 0,
        "consecutive_good": max(0, int(win_rate * 5)),
        "sharpe": sharpe,
        "sortino": sharpe * 1.2,
        "max_drawdown": abs(min((rows[i]["_val"] - rows[i - 1]["_val"]) / rows[i - 1]["_val"]
                                for i in range(1, len(rows)) if rows[i - 1]["_val"] > 0)),
        "samples": samples,
        "last_updated": _utc(),
        "warmup_source": "warmup_cold_start.py",
    }

    bg_state = {
        "mode": "unknown",
        "last_eval": _utc(),
        "current": {
            "sharpe": sharpe,
            "win_rate": round(win_rate, 4),
            "trades": len(trades),
            "avg_pnl_pct": round(avg_pnl, 4),
        },
        "warmup_source": "warmup_cold_start.py",
    }

    if args.dry_run:
        print(f"\n[DRY RUN] Would write:")
        print(f"  {audit_path}  ({len(events)} events)")
        print(f"  {data_dir}/performance_governor_state.json  (tier=NORMAL, sharpe={sharpe:.3f})")
        print(f"  {data_dir}/backtest_gate_state.json  (mode=unknown, trades={len(trades)})")
        return 0

    # Write audit
    with open(audit_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev, default=str) + "\n")
    print(f"✅ Wrote {len(events)} audit events → {audit_path}")

    # Write governor state
    gov_path = data_dir / "performance_governor_state.json"
    gov_path.write_text(json.dumps(gov_state, indent=2, default=str))
    print(f"✅ PerformanceGovernor seeded → tier=NORMAL, sharpe={sharpe:.3f}, samples={len(samples)}")

    # Write backtest gate state
    bg_path = data_dir / "backtest_gate_state.json"
    bg_path.write_text(json.dumps(bg_state, indent=2, default=str))
    print(f"✅ BacktestGate seeded → {len(trades)} synthetic trades, win_rate={win_rate:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
