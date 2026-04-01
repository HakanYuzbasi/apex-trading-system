#!/usr/bin/env python3
"""
scripts/audit_confidence_thresholds.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audit confidence thresholds against backtest win rates.

For each threshold in [0.30 … 0.70] (step 0.05), shows what win rate and
profit factor result from *allowing* all trades with confidence >= threshold.
Compares against the live thresholds defined in config.py.

Output: a human-readable table + a JSON file (if --json is specified).

Usage:
  PYTHONPATH=. venv/bin/python scripts/audit_confidence_thresholds.py
  PYTHONPATH=. venv/bin/python scripts/audit_confidence_thresholds.py --csv data/backtest_trades.csv
  PYTHONPATH=. venv/bin/python scripts/audit_confidence_thresholds.py --json out/threshold_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Live thresholds (read from config.py) ──────────────────────────────

def _load_live_thresholds() -> dict:
    """Pull the currently configured confidence thresholds from config.py."""
    try:
        from config import ApexConfig
        return {
            "apex_MIN_CONFIDENCE":           float(getattr(ApexConfig, "MIN_CONFIDENCE", 0.60)),
            "apex_ENTRY_CONFIDENCE_MODERATE": float(getattr(ApexConfig, "ENTRY_CONFIDENCE_MODERATE", 0.60)),
            "apex_CRYPTO_MIN_CONFIDENCE":     float(getattr(ApexConfig, "CRYPTO_MIN_CONFIDENCE", 0.50)),
            "apex_CORE_MIN_CONFIDENCE":       float(getattr(ApexConfig, "CORE_MIN_CONFIDENCE", 0.45)),
        }
    except Exception as e:
        print(f"[WARNING] Could not load config.py: {e}")
        return {
            "apex_MIN_CONFIDENCE":           0.60,
            "apex_ENTRY_CONFIDENCE_MODERATE": 0.60,
            "apex_CRYPTO_MIN_CONFIDENCE":     0.50,
            "apex_CORE_MIN_CONFIDENCE":       0.45,
        }


# ── Load trade CSV ──────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[dict]:
    """Load backtest_trades.csv (or any compatible CSV with pnl & confidence)."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fnames_lower = [c.lower() for c in (reader.fieldnames or [])]
        col_map = {c.lower(): c for c in (reader.fieldnames or [])}

        pnl_col = next(
            (c for c in ("pnl_percent", "pnl_pct", "pnl") if c in fnames_lower), None
        )
        if pnl_col is None:
            return []
        pnl_key = col_map[pnl_col]
        conf_key = col_map.get("confidence")
        regime_key = col_map.get("regime")
        sym_key = col_map.get("symbol", "symbol")

        for row in reader:
            raw_pnl = row.get(pnl_key, "").strip()
            raw_conf = row.get(conf_key, "").strip() if conf_key else ""
            if not raw_pnl:
                continue
            try:
                pnl = float(raw_pnl)
            except ValueError:
                continue
            conf = None
            if raw_conf:
                try:
                    conf = float(raw_conf)
                except ValueError:
                    pass
            rows.append({
                "symbol":     row.get(sym_key, "?"),
                "pnl":        pnl,
                "confidence": conf,
                "regime":     row.get(regime_key, "") if regime_key else "",
            })
    return rows


# ── Audit engine ────────────────────────────────────────────────────────

def audit_thresholds(
    trades: List[dict],
    thresholds: Optional[List[float]] = None,
    min_trades_for_recommendation: int = 20,
) -> dict:
    """
    For each threshold T in `thresholds`:
      - Consider only trades with confidence >= T
      - Compute: count, win_rate, profit_factor, net_pnl

    Also recommends the threshold that maximises profit factor
    while keeping trade count >= min_trades_for_recommendation.
    """
    if thresholds is None:
        thresholds = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 … 0.70

    trades_with_conf = [t for t in trades if t["confidence"] is not None]
    trades_without   = len(trades) - len(trades_with_conf)

    rows = []
    for t in thresholds:
        subset = [x for x in trades_with_conf if x["confidence"] >= t]
        if not subset:
            rows.append({
                "threshold": t, "count": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "net_pnl": 0.0,
            })
            continue
        wins     = sum(1 for x in subset if x["pnl"] > 0)
        gp       = sum(x["pnl"] for x in subset if x["pnl"] > 0)
        gl       = abs(sum(x["pnl"] for x in subset if x["pnl"] < 0))
        net      = sum(x["pnl"] for x in subset)
        win_rate = wins / len(subset) * 100
        pf       = gp / gl if gl > 0 else float("inf")
        rows.append({
            "threshold":     t,
            "count":         len(subset),
            "win_rate":      round(win_rate, 2),
            "profit_factor": round(pf, 4) if pf != float("inf") else None,
            "net_pnl":       round(net, 6),
        })

    # Best threshold: highest PF with enough trades
    eligible = [r for r in rows if r["count"] >= min_trades_for_recommendation
                and r["profit_factor"] is not None]
    best = max(eligible, key=lambda r: r["profit_factor"]) if eligible else None

    return {
        "total_trades":           len(trades),
        "trades_with_confidence": len(trades_with_conf),
        "trades_no_confidence":   trades_without,
        "threshold_rows":         rows,
        "recommended_threshold":  best,
    }


# ── Formatted output ────────────────────────────────────────────────────

def print_report(result: dict, live: dict) -> None:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"

    print()
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║       APEX — CONFIDENCE THRESHOLD AUDIT              ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")
    print(f"  Total trades in dataset         : {result['total_trades']}")
    print(f"  Trades with confidence recorded : {result['trades_with_confidence']}")
    if result["trades_no_confidence"]:
        print(f"  {YELLOW}Trades WITHOUT confidence field : {result['trades_no_confidence']} "
              f"(run new backtest to populate){RESET}")
    print()

    # Live threshold panel
    print(f"{BOLD}── CURRENT LIVE THRESHOLDS (config.py) ─────────────────{RESET}")
    for k, v in live.items():
        key_label = k.replace("apex_", "").replace("_", " ")
        print(f"  {key_label:<35} : {BOLD}{v:.2f}{RESET}")
    print()

    # Table
    print(f"{BOLD}── PERFORMANCE BY THRESHOLD ────────────────────────────{RESET}")
    print(f"  {'Threshold':>10} {'Trades':>7} {'WinRate':>8} {'ProfitFactor':>13} {'NetPnL':>10}")
    print("  " + "─" * 54)
    for row in result["threshold_rows"]:
        t = row["threshold"]
        is_live = any(abs(t - v) < 0.01 for v in live.values())
        marker   = f" {YELLOW}← LIVE{RESET}" if is_live else ""

        pf = row["profit_factor"]
        pf_str = f"{pf:.3f}" if pf is not None else "  N/A"
        pf_col = GREEN if (pf or 0) >= 1.0 else RED

        wr_col = GREEN if row["win_rate"] >= 50 else RED
        net_col = GREEN if row["net_pnl"] > 0 else RED
        count = row["count"]
        low_sample = f"  {DIM}(low n){RESET}" if 0 < count < 20 else ""

        print(
            f"  {t:>10.2f} {count:>7}  {wr_col}{row['win_rate']:>6.1f}%{RESET}  "
            f"{pf_col}{pf_str:>12}{RESET}  {net_col}{row['net_pnl']:>+10.4f}{RESET}"
            f"{marker}{low_sample}"
        )
    print()

    # Recommendation
    best = result["recommended_threshold"]
    if best:
        print(f"{BOLD}── RECOMMENDATION ──────────────────────────────────────{RESET}")
        live_min = min(live.values())
        opt_t    = best["threshold"]
        diff     = opt_t - live_min
        if abs(diff) < 0.02:
            print(f"  {GREEN}✓ Current threshold ({live_min:.2f}) is near-optimal.{RESET}")
        elif diff > 0:
            print(
                f"  {YELLOW}⚠ Optimal threshold: {opt_t:.2f}  "
                f"(current live minimum: {live_min:.2f} — raising by {diff:+.2f} may improve quality){RESET}"
            )
        else:
            print(
                f"  {GREEN}↓ Optimal threshold: {opt_t:.2f}  "
                f"(current live minimum: {live_min:.2f} — LOWERING by {diff:.2f} "
                f"admits more trades with good stats){RESET}"
            )
        print(
            f"     Best stat   : PF = {best['profit_factor']:.3f}  "
            f"Win Rate = {best['win_rate']:.1f}%  "
            f"N = {best['count']} trades"
        )
    else:
        print(f"  {RED}Insufficient trades for confident recommendation (need ≥ 20).{RESET}")
    print()


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Audit confidence thresholds vs. backtest results")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="CSV with trade history (default: data/backtest_trades.csv)")
    p.add_argument("--json", default=None, metavar="OUTPUT_PATH",
                   help="Write full result to JSON file")
    p.add_argument("--min-n", type=int, default=20,
                   help="Minimum trade count for a threshold to be considered 'eligible' for recommendation")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else (ROOT / "data" / "backtest_trades.csv")
    if not csv_path.exists():
        # Fallback
        fallback = ROOT / "data" / "trades.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            print(f"[ERROR] No trade data found. Run scripts/run_backtest_with_synthetic_data.py first.")
            return 2

    trades = _load_csv(csv_path)
    if not trades:
        print(f"[ERROR] Could not load trade data from {csv_path}")
        return 2

    print(f"[INFO] Loaded {len(trades)} trades from {csv_path}")

    live    = _load_live_thresholds()
    result  = audit_thresholds(trades, min_trades_for_recommendation=args.min_n)
    print_report(result, live)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_csv":    str(csv_path),
            "live_thresholds": live,
            **result,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[✓] Audit results saved to {out}")

    best = result.get("recommended_threshold")
    if best and best.get("profit_factor") and best["profit_factor"] < 1.0:
        return 1  # No threshold produces a profitable strategy
    return 0


if __name__ == "__main__":
    sys.exit(main())
