#!/usr/bin/env python3
"""
scripts/weekly_retrain.py

Weekly GodLevel ML model retrain on last 30 days of live price data.

Run manually or schedule with cron / Docker COMMAND:
    python scripts/weekly_retrain.py [--symbols AAPL MSFT ...] [--days 30] [--model-dir run_state/models/god_level]

The script:
  1. Pulls 30-day daily OHLCV from yfinance for the target universe
  2. Instantiates GodLevelSignalGenerator pointing at the live model dir
  3. Calls train_models() — rewrites god_level_models.pkl in place
  4. The running container loads the updated file on next GodLevel refresh (TTL 290s)

Safety: the old model file is backed up to *.bak before overwriting.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weekly_retrain")

_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD",  "V",    "MA",    "JPM",
    "BAC",  "KO",   "PEP",  "SPY",   "QQQ",
    "TSLA", "NFLX", "CRM",  "GLD",   "TLT",
]
_DEFAULT_DAYS    = 30
_DEFAULT_DIR     = "run_state/models/god_level"


def _fetch_data(symbols: list[str], days: int) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — pip install yfinance")
        sys.exit(1)

    end   = datetime.now()
    start = end - timedelta(days=days + 5)  # extra buffer for weekends

    logger.info("Fetching %d days of daily OHLCV for %d symbols …", days, len(symbols))
    data = {}
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if len(df) >= 15:
                data[sym] = df.tail(days)
                logger.info("  %-6s — %d bars", sym, len(data[sym]))
            else:
                logger.warning("  %-6s — only %d bars, skipping", sym, len(df))
        except Exception as exc:
            logger.warning("  %-6s — fetch failed: %s", sym, exc)

    logger.info("Fetched data for %d / %d symbols", len(data), len(symbols))
    return data


def _backup_model(model_dir: Path) -> None:
    model_file = model_dir / "god_level_models.pkl"
    if model_file.exists():
        backup = model_dir / "god_level_models.pkl.bak"
        shutil.copy2(model_file, backup)
        logger.info("Backed up existing model to %s", backup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly GodLevel ML retrain")
    parser.add_argument("--symbols", nargs="+", default=_DEFAULT_UNIVERSE)
    parser.add_argument("--days",    type=int,   default=_DEFAULT_DAYS)
    parser.add_argument("--model-dir", default=_DEFAULT_DIR)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model directory: %s", model_dir)

    data = _fetch_data(args.symbols, args.days)
    if len(data) < 5:
        logger.error("Too few symbols fetched (%d) — aborting retrain", len(data))
        sys.exit(1)

    logger.info("Loading GodLevelSignalGenerator …")
    from models.god_level_signal_generator import GodLevelSignalGenerator
    generator = GodLevelSignalGenerator(model_dir=str(model_dir))

    _backup_model(model_dir)

    logger.info("Starting train_models() on %d symbols …", len(data))
    generator.train_models(data)

    model_file = model_dir / "god_level_models.pkl"
    if model_file.exists():
        size_kb = model_file.stat().st_size / 1024
        logger.info("✅ Retrain complete — model saved: %s (%.1f KB)", model_file, size_kb)
    else:
        logger.error("❌ train_models() ran but model file not found at %s", model_file)
        sys.exit(1)


if __name__ == "__main__":
    main()
