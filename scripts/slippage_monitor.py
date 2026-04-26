"""slippage_monitor.py — Real-time slippage analysis from execution_log.jsonl.
Alert thresholds:
  Single trade > 15 bps      → WARNING
  Rolling-20 avg > 10 bps    → CRITICAL + ALERT file
  Rolling-20 avg > 20 bps    → HALT file (live_trader must check before trading)
Usage:
  python scripts/slippage_monitor.py            # continuous monitor
  python scripts/slippage_monitor.py --report   # one-shot report, exit
"""
import argparse, json, logging, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
EXEC_LOG  = DATA_DIR / "execution_log.jsonl"
HALT_FILE = DATA_DIR / "HALT_slippage_exceeded.txt"
WARN_BPS  = 15    # single trade warning threshold
CRIT_BPS  = 10    # rolling-20 critical threshold
HALT_BPS  = int(
    __import__("os").getenv("SLIPPAGE_HALT_BPS", "20")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s slippage — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("slippage_monitor")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_filled() -> pd.DataFrame:
    """Return DataFrame of FILLED records that have a slippage_bps value."""
    sys.path.insert(0, str(ROOT))
    from scripts.execution_log import ExecutionLog
    records = ExecutionLog.read_all()
    filled  = [r for r in records if r.get("status") == "FILLED"
               and r.get("slippage_bps") is not None]
    if not filled:
        return pd.DataFrame()
    return pd.DataFrame(filled)


def _write_alert(tag: str, msg: str) -> Path:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / f"ALERT_{tag}_{ts}.txt"
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{datetime.now(timezone.utc).isoformat()}\n{msg}\n")
    log.critical("ALERT file written: %s", path)
    return path


def _write_halt(rolling_avg: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HALT_FILE.write_text(
        f"HALT issued at {datetime.now(timezone.utc).isoformat()}\n"
        f"Rolling-20 slippage avg = {rolling_avg:.2f} bps >= {HALT_BPS} bps\n"
        "Resolve: investigate broker fills, then delete this file to resume.\n"
    )
    log.critical("HALT FILE written: %s — live trading suspended.", HALT_FILE)


def halt_active() -> bool:
    """Return True if a halt file exists (live_trader checks this)."""
    return HALT_FILE.exists()


# ── Core analysis ─────────────────────────────────────────────────────────────
def analyse(df: pd.DataFrame) -> dict:
    """Compute per-trade alerts and rolling metrics. Return summary dict."""
    if df.empty:
        log.info("No filled trades to analyse.")
        return {"trades": 0}

    bps_series = df["slippage_bps"].astype(float)

    # Per-trade alerts
    for _, row in df.iterrows():
        bps = float(row["slippage_bps"])
        if bps > WARN_BPS:
            log.warning(
                "HIGH SLIPPAGE: %s %s %.2f bps (threshold %d bps)",
                row.get("date", "?"), row.get("symbol", "?"), bps, WARN_BPS,
            )

    # Rolling-20 analysis
    rolling20 = bps_series.rolling(20).mean().dropna()
    summary = {
        "trades":       len(bps_series),
        "avg_bps":      round(float(bps_series.mean()), 3),
        "p50_bps":      round(float(bps_series.median()), 3),
        "p95_bps":      round(float(bps_series.quantile(0.95)), 3),
        "rolling20_latest": round(float(rolling20.iloc[-1]), 3) if not rolling20.empty else None,
        "halt_active":  halt_active(),
    }

    if not rolling20.empty:
        latest_r20 = float(rolling20.iloc[-1])
        if latest_r20 >= HALT_BPS:
            _write_halt(latest_r20)
            summary["halt_written"] = True
        elif latest_r20 >= CRIT_BPS:
            _write_alert(
                "slippage_critical",
                f"Rolling-20 avg slippage = {latest_r20:.2f} bps >= {CRIT_BPS} bps",
            )
            summary["alert_written"] = True

    return summary


def print_report(summary: dict) -> None:
    print("\n══════════════════════════════════════")
    print("  SLIPPAGE MONITOR — REPORT")
    print("══════════════════════════════════════")
    if summary.get("trades", 0) == 0:
        print("  No filled trades in execution log.")
        return
    print(f"  Filled trades   : {summary['trades']}")
    print(f"  Avg slippage    : {summary['avg_bps']:.2f} bps")
    print(f"  P50 slippage    : {summary['p50_bps']:.2f} bps")
    print(f"  P95 slippage    : {summary['p95_bps']:.2f} bps")
    r20 = summary.get("rolling20_latest")
    print(f"  Rolling-20 avg  : {r20:.2f} bps" if r20 is not None else "  Rolling-20 avg  : n/a (<20 trades)")
    print(f"  Halt active     : {summary['halt_active']}")
    print(f"  Thresholds      : warn>{WARN_BPS}bps  crit>{CRIT_BPS}bps  halt>={HALT_BPS}bps")
    print("══════════════════════════════════════\n")


def run_monitor(poll_seconds: int = 300) -> None:
    """Continuously poll every poll_seconds seconds."""
    log.info("Slippage monitor running (poll=%ds, halt_bps=%d)", poll_seconds, HALT_BPS)
    while True:
        df      = _load_filled()
        summary = analyse(df)
        log.info("Monitor cycle: trades=%d avg=%.2f bps rolling20=%s",
                 summary["trades"], summary.get("avg_bps", 0),
                 summary.get("rolling20_latest"))
        time.sleep(poll_seconds)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Slippage monitor / report")
    parser.add_argument("--report", action="store_true", help="One-shot report then exit")
    parser.add_argument("--poll",   type=int, default=300, help="Poll interval in seconds")
    args = parser.parse_args()

    df      = _load_filled()
    summary = analyse(df)

    if args.report:
        print_report(summary)
        sys.exit(0)

    run_monitor(poll_seconds=args.poll)


if __name__ == "__main__":
    main()

