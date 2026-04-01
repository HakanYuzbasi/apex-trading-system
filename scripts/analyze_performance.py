#!/usr/bin/env python3
"""
scripts/analyze_performance.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI runner for the Apex Trade Performance Analytics module.

Usage examples:
  # Auto-detect data sources (fastest path):
  python scripts/analyze_performance.py

  # Specify a custom CSV file:
  python scripts/analyze_performance.py --csv path/to/mytrades.csv

  # Output JSON metrics to a file:
  python scripts/analyze_performance.py --json out/metrics.json

  # Quiet mode (JSON only, no ANSI summary):
  python scripts/analyze_performance.py --quiet --json out/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when running with:
#   python scripts/analyze_performance.py   OR
#   PYTHONPATH=. python scripts/analyze_performance.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.trade_performance import TradePerformanceAnalytics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apex Trading — Strategy Performance Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Path to the Apex data directory (default: <project>/data)",
    )
    p.add_argument(
        "--csv", default=None,
        metavar="PATH",
        help="Custom CSV file with trade history (overrides auto-detection)",
    )
    p.add_argument(
        "--jsonl", default=None,
        metavar="PATH",
        help="Custom JSONL diagnostics file (overrides auto-detection)",
    )
    p.add_argument(
        "--equity-csv", default=None,
        metavar="PATH",
        help="Equity curve CSV for drawdown (overrides auto-detection)",
    )
    p.add_argument(
        "--json", default=None,
        metavar="OUTPUT_PATH",
        help="Also write metrics as JSON to this file",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the formatted console summary (useful with --json)",
    )
    p.add_argument(
        "--min-trades", type=int, default=1,
        help="Minimum trade count threshold before issuing a sample-size warning",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    analytics = TradePerformanceAnalytics(
        data_dir=args.data_dir,
        csv_path=args.csv,
        jsonl_path=args.jsonl,
        equity_csv=args.equity_csv,
        min_trades=args.min_trades,
    )

    report = analytics.run()

    if not args.quiet:
        print(report.summary())

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[✓] Metrics saved to {out_path}")

    # Exit 1 if strategy is clearly losing (negative expectancy)
    return 1 if report.expectancy < 0 and report.total_trades > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
