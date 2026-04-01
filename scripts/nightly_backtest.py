#!/usr/bin/env python3
"""
scripts/nightly_backtest.py — Nightly Backtest Runner

Runs the BacktestGate evaluation and logs the result.
Designed to be called by cron / scheduler once per trading day after close.

Usage:
    python scripts/nightly_backtest.py                  # evaluate and print result
    python scripts/nightly_backtest.py --force-live     # override to live mode
    python scripts/nightly_backtest.py --force-paper    # override to paper mode
    python scripts/nightly_backtest.py --status         # print current gate state only

Exit codes:
    0 — gate is LIVE (or unknown with insufficient data)
    1 — gate is PAPER (strategy degraded)
    2 — error during evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.backtest_gate import BacktestGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nightly_backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly backtest gate runner")
    parser.add_argument("--force-live",  action="store_true", help="Force gate to LIVE mode")
    parser.add_argument("--force-paper", action="store_true", help="Force gate to PAPER mode")
    parser.add_argument("--status",      action="store_true", help="Print current state only")
    parser.add_argument("--json",        action="store_true", help="Output JSON")
    args = parser.parse_args()

    try:
        gate = BacktestGate()

        if args.force_live:
            gate.force_live()
            logger.info("Gate forced to LIVE")
            return 0

        if args.force_paper:
            gate.force_paper()
            logger.info("Gate forced to PAPER")
            return 1

        if args.status:
            state = gate.get_state()
            if args.json:
                print(json.dumps(state, indent=2))
            else:
                print(f"Mode: {state['mode'].upper()}")
                print(f"Last eval: {state.get('last_eval_ts', 'never')}")
                m = state.get("last_metrics", {})
                if m:
                    print(f"Sharpe: {m.get('sharpe', 0):.3f}  Win-rate: {m.get('win_rate', 0)*100:.1f}%  Trades: {m.get('trades', 0)}")
            return 0 if gate.is_live or gate.mode == "unknown" else 1

        # Full evaluation
        record = gate.run_evaluation()

        if args.json:
            import dataclasses
            def _default(o):
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                return str(o)
            print(json.dumps(dataclasses.asdict(record), indent=2, default=_default))
        else:
            flags_str = ", ".join(record.triggered_flags) if record.triggered_flags else "none"
            print(f"\n{'='*60}")
            print(f"  NIGHTLY BACKTEST GATE — {record.ts[:10]}")
            print(f"{'='*60}")
            print(f"  Mode:        {record.mode.upper()}")
            print(f"  Flags:       {flags_str}")
            print(f"  Sharpe:      {record.current.sharpe:.3f}")
            print(f"  Win-rate:    {record.current.win_rate*100:.1f}%")
            print(f"  Avg P&L:     {record.current.avg_pnl_pct*100:.3f}%")
            print(f"  Trades:      {record.current.trades}")
            print(f"  Sharpe Δ:    {record.sharpe_delta:+.3f}")
            print(f"{'='*60}\n")

        return 0 if gate.is_live or gate.mode == "unknown" else 1

    except Exception as e:
        logger.error("Nightly backtest failed: %s", e, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
