"""Cron entry point for the APEX live trader."""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from scripts import live_trader

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LAST_RUN = DATA_DIR / "last_run.txt"
CRON_LINE = "35 9 * * 1-5 cd /Users/hakanyuzbasioglu/apex-trading && venv/bin/python scripts/run_scheduler.py"


def _logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("apex_scheduler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            LOG_DIR / "scheduler.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def main() -> None:
    logger = _logger()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    logger.info("scheduler start")
    try:
        live_trader.run_once()
    except Exception:
        duration = time.monotonic() - started
        logger.exception("scheduler failed after %.2fs", duration)
        raise
    duration = time.monotonic() - started
    LAST_RUN.write_text(f"{datetime.now(timezone.utc).isoformat()} duration_seconds={duration:.3f}\n")
    logger.info("scheduler complete started_at=%s duration_seconds=%.3f", started_at.isoformat(), duration)
    print(CRON_LINE)


if __name__ == "__main__":
    main()
