from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

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
logger = logging.getLogger("load_universe_data")


PAIRS: tuple[tuple[str, str], ...] = (
    ("AAPL", "MSFT"),
    ("NVDA", "AMD"),
    ("KO", "PEP"),
    ("V", "MA"),
    ("JPM", "GS"),
    ("XOM", "CVX"),
    ("WMT", "TGT"),
)
LOOKBACK_DAYS = 365
TIMEFRAME = TimeFrame.Hour
SOURCE_NAME = "alpaca.history"
MAX_CONCURRENT_DOWNLOADS = 3
MAX_RETRIES = 5
BATCH_SIZE = 1000


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


def normalize_ts(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def chunked(events: Sequence[BarEvent], size: int) -> Iterable[Sequence[BarEvent]]:
    for offset in range(0, len(events), size):
        yield events[offset : offset + size]


def universe_symbols(pairs: Sequence[tuple[str, str]]) -> list[str]:
    unique_symbols = {symbol for pair in pairs for symbol in pair}
    return sorted(unique_symbols)


def has_hourly_data(
    db_client: TimescaleDBClient,
    symbol: str,
    start_ts: datetime,
    end_ts: datetime,
) -> bool:
    coverage_buffer = timedelta(days=2)
    with db_client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*), MIN(exchange_ts), MAX(exchange_ts)
                FROM bar_events
                WHERE instrument_id = %s
                  AND exchange_ts >= %s
                  AND exchange_ts < %s
                """,
                (symbol, start_ts, end_ts),
            )
            row = cur.fetchone()

    count = int(row[0] or 0)
    min_ts = row[1]
    max_ts = row[2]
    if count == 0 or min_ts is None or max_ts is None:
        return False
    return (
        normalize_ts(min_ts) <= start_ts + coverage_buffer
        and normalize_ts(max_ts) >= end_ts - coverage_buffer
    )


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


async def fetch_with_retry(
    symbol: str,
    start_ts: datetime,
    end_ts: datetime,
) -> list[BarEvent]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(fetch_symbol_bars, build_client(), symbol, start_ts, end_ts)
        except Exception as exc:
            is_last_attempt = attempt == MAX_RETRIES
            retry_delay = min(2 ** attempt, 30)
            logger.warning(
                "Fetch failed for %s on attempt %d/%d: %s",
                symbol,
                attempt,
                MAX_RETRIES,
                exc,
            )
            if is_last_attempt:
                raise
            await asyncio.sleep(retry_delay)
    return []


async def load_symbol(
    symbol: str,
    *,
    db_client: TimescaleDBClient,
    semaphore: asyncio.Semaphore,
    start_ts: datetime,
    end_ts: datetime,
) -> tuple[str, int, str]:
    if await asyncio.to_thread(has_hourly_data, db_client, symbol, start_ts, end_ts):
        logger.info("Skipping %s because sufficient hourly history already exists", symbol)
        return symbol, 0, "skipped"

    async with semaphore:
        events = await fetch_with_retry(symbol, start_ts, end_ts)

    if not events:
        logger.warning("No hourly bars returned for %s", symbol)
        return symbol, 0, "empty"

    inserted_total = 0
    for batch in chunked(events, BATCH_SIZE):
        inserted = await asyncio.to_thread(db_client.write_bar_events, batch)
        inserted_total += inserted
        logger.info(
            "Inserted batch for %s: batch_size=%d cumulative=%d",
            symbol,
            len(batch),
            inserted_total,
        )

    logger.info("Completed %s load: inserted=%d", symbol, inserted_total)
    return symbol, inserted_total, "loaded"


async def main() -> None:
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=LOOKBACK_DAYS)
    symbols = universe_symbols(PAIRS)

    db_client = TimescaleDBClient()
    logger.info("Ensuring TimescaleDB schema exists")
    await asyncio.to_thread(db_client.ensure_schema)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    tasks = [
        load_symbol(
            symbol,
            db_client=db_client,
            semaphore=semaphore,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        for symbol in symbols
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    inserted_total = 0
    failures: list[tuple[str, str]] = []

    for result in results:
        if isinstance(result, Exception):
            failures.append(("unknown", str(result)))
            logger.exception("Universe load task failed", exc_info=result)
            continue

        symbol, inserted_count, status = result
        inserted_total += inserted_count
        if status == "loaded":
            logger.info("Loaded %s successfully", symbol)

    logger.info("Universe load complete. total_symbols=%d total_inserted=%d", len(symbols), inserted_total)
    if failures:
        logger.warning("Failures encountered during universe load:")
        for symbol, error in failures:
            logger.warning("  %s -> %s", symbol, error)


if __name__ == "__main__":
    asyncio.run(main())
