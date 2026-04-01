#!/usr/bin/env python3
"""
scripts/run_daily_briefing.py — Daily Strategy Briefing Runner (Build 3)

Generates the automated daily strategy briefing and optionally sends it via
Telegram. Designed to be called by cron / scheduler after market close, or
run on-demand.

Usage:
    python scripts/run_daily_briefing.py                  # today's briefing
    python scripts/run_daily_briefing.py --date 2026-03-24
    python scripts/run_daily_briefing.py --json            # JSON output
    python scripts/run_daily_briefing.py --text            # human-readable text (default)
    python scripts/run_daily_briefing.py --latest          # print most recent saved briefing

Exit codes:
    0 — briefing generated (or loaded) successfully
    1 — briefing could not be generated (no data / error)
    2 — unexpected error
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.daily_briefing import DailyBriefingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_daily_briefing")


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily strategy briefing generator")
    parser.add_argument(
        "--date",
        default=None,
        help="Date to generate briefing for (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output human-readable text (default if neither flag given)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Print most recently saved briefing without regenerating",
    )
    args = parser.parse_args()

    try:
        try:
            from config import ApexConfig
            data_dir = str(ApexConfig.DATA_DIR)
        except Exception:
            data_dir = "data"

        generator = DailyBriefingGenerator(data_dir=data_dir)

        if args.latest:
            latest = generator.get_latest()
            if latest is None:
                logger.warning("No saved briefings found.")
                return 1
            if args.json:
                print(json.dumps(latest, indent=2))
            else:
                from monitoring.daily_briefing import DailyBriefing, TradeStats, SignalStats
                import dataclasses
                # Re-hydrate and render as text (best-effort)
                ts_raw = latest.get("trade_stats", {})
                ts = TradeStats(
                    total_trades=ts_raw.get("total_trades", 0),
                    wins=ts_raw.get("wins", 0),
                    losses=ts_raw.get("losses", 0),
                    win_rate=ts_raw.get("win_rate", 0.0),
                    avg_pnl_pct=ts_raw.get("avg_pnl_pct", 0.0),
                    total_pnl_pct=ts_raw.get("total_pnl_pct", 0.0),
                    best_trade=ts_raw.get("best_trade"),
                    worst_trade=ts_raw.get("worst_trade"),
                )
                briefing = DailyBriefing(
                    date=latest.get("date", "?"),
                    generated_at=latest.get("generated_at", "?"),
                    trade_stats=ts,
                    top_signals=[
                        SignalStats(
                            name=s["name"], ic=s["ic"], obs=s["obs"],
                            win_rate=s["win_rate"], status=s["status"],
                        )
                        for s in latest.get("top_signals", [])
                    ],
                    weak_signals=latest.get("weak_signals", []),
                    regime=latest.get("regime", "unknown"),
                    backtest_gate_mode=latest.get("backtest_gate_mode", "unknown"),
                    strategy_health_paper_only=latest.get("strategy_health_paper_only", False),
                    adaptive_weights=latest.get("adaptive_weights", {}),
                    recommendations=latest.get("recommendations", []),
                    notes=latest.get("notes", []),
                )
                print(briefing.to_text())
            return 0

        # Generate fresh briefing
        report_date = None
        if args.date:
            try:
                report_date = date.fromisoformat(args.date)
            except ValueError as e:
                logger.error("Invalid date '%s': %s", args.date, e)
                return 2

        # Detect current regime from state file if possible
        regime = "unknown"
        try:
            import json as _json
            _rs = Path(data_dir) / "regime_state.json"
            if _rs.exists():
                _state = _json.loads(_rs.read_text())
                regime = _state.get("regime", "unknown")
        except Exception:
            pass

        briefing = generator.generate(report_date=report_date, regime=regime)

        if args.json:
            print(json.dumps(briefing.to_dict(), indent=2, default=str))
        else:
            # Default: human-readable text
            print(briefing.to_text())

        return 0

    except Exception as exc:
        logger.error("Daily briefing runner failed: %s", exc, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
