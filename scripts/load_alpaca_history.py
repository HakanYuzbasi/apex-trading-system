from __future__ import annotations

import hashlib
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from quant_system.data.stores.client import TimescaleDBClient
from quant_system.events import BarEvent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("load_alpaca_history")


SYMBOLS = ("AAPL", "SPY")
LOOKBACK_DAYS = 182
TIMEFRAME = TimeFrame.Hour
SOURCE_NAME = "alpaca.history"


def deterministic_event_id(symbol: str, timestamp: datetime) -> str:
    raw = f"{symbol}|{timestamp.astimezone(timezone.utc).isoformat()}|1H|{SOURCE_NAME}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"bar-{digest}"


def build_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_symbol_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start_ts: datetime,
    end_ts: datetime,
) -> list[BarEvent]:
    logger.info("Fetching %s hourly bars from %s to %s", symbol, start_ts.isoformat(), end_ts.isoformat())
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start_ts,
        end=end_ts,
    )
    response = client.get_stock_bars(request)
    bars = response[symbol]

    events: list[BarEvent] = []
    for bar in bars:
        timestamp = normalize_ts(bar.timestamp)
        event_id = deterministic_event_id(symbol, timestamp)
        events.append(
            BarEvent(
                instrument_id=symbol,
                exchange_ts=timestamp,
                received_ts=timestamp,
                processed_ts=timestamp,
                sequence_id=0,
                source=SOURCE_NAME,
                event_id=event_id,
                bar_id=event_id,
                payload_version="1.0",
                metadata={
                    "vendor": "alpaca",
                    "timeframe": "1H",
                    "symbol": symbol,
                },
                open_price=float(bar.open),
                high_price=float(bar.high),
                low_price=float(bar.low),
                close_price=float(bar.close),
                volume=float(bar.volume),
            )
        )
    logger.info("Fetched %d bars for %s", len(events), symbol)
    return events


def normalize_ts(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def chunked(events: list[BarEvent], size: int) -> Iterable[list[BarEvent]]:
    for offset in range(0, len(events), size):
        yield events[offset : offset + size]


def main() -> None:
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=LOOKBACK_DAYS)

    db_client = TimescaleDBClient()
    logger.info("Ensuring TimescaleDB schema exists")
    db_client.ensure_schema()

    alpaca_client = build_client()
    total_inserted = 0

    for symbol in SYMBOLS:
        events = fetch_symbol_bars(alpaca_client, symbol, start_ts, end_ts)
        if not events:
            logger.warning("No bars returned for %s", symbol)
            continue

        inserted_for_symbol = 0
        for batch in chunked(events, 1000):
            inserted = db_client.write_bar_events(batch)
            inserted_for_symbol += inserted
            logger.info(
                "Inserted batch for %s: batch_size=%d cumulative=%d",
                symbol,
                len(batch),
                inserted_for_symbol,
            )
        total_inserted += inserted_for_symbol
        logger.info("Completed %s load: inserted=%d", symbol, inserted_for_symbol)

    logger.info("Historical load complete. total_inserted=%d", total_inserted)


if __name__ == "__main__":
    main()
