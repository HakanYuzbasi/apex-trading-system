from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor, execute_batch
except ImportError:  # pragma: no cover
    psycopg2 = None
    Json = None
    RealDictCursor = None
    execute_batch = None

from quant_system.events import BarEvent, TradeTick


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    dsn: str | None = None
    host: str = "127.0.0.1"
    port: int = 5432
    database: str = "quant_system"
    user: str = "postgres"
    password: str = "postgres"
    connect_timeout: int = 5
    bar_chunk_interval: str = "7 days"
    tick_chunk_interval: str = "1 day"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            dsn=os.getenv("QUANT_DB_DSN"),
            host=os.getenv("QUANT_DB_HOST", "127.0.0.1"),
            port=int(os.getenv("QUANT_DB_PORT", "5432")),
            database=os.getenv("QUANT_DB_NAME", "quant_system"),
            user=os.getenv("QUANT_DB_USER", "postgres"),
            password=os.getenv("QUANT_DB_PASSWORD", "postgres"),
            connect_timeout=int(os.getenv("QUANT_DB_CONNECT_TIMEOUT", "5")),
            bar_chunk_interval=os.getenv("QUANT_DB_BAR_CHUNK_INTERVAL", "7 days"),
            tick_chunk_interval=os.getenv("QUANT_DB_TICK_CHUNK_INTERVAL", "1 day"),
        )


class TimescaleDBClient:
    """TimescaleDB client with streaming reads for deterministic replay."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig.from_env()

    @contextmanager
    def connection(self):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for TimescaleDBClient")

        if self.config.dsn:
            conn = psycopg2.connect(self.config.dsn, connect_timeout=self.config.connect_timeout)
        else:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.database,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connect_timeout,
            )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bar_events (
                        exchange_ts TIMESTAMPTZ NOT NULL,
                        instrument_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        bar_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        payload_version TEXT NOT NULL DEFAULT '1.0',
                        open_price DOUBLE PRECISION NOT NULL,
                        high_price DOUBLE PRECISION NOT NULL,
                        low_price DOUBLE PRECISION NOT NULL,
                        close_price DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (exchange_ts, instrument_id, event_id)
                    );
                    """
                )
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'bar_events',
                        'exchange_ts',
                        if_not_exists => TRUE,
                        migrate_data => TRUE,
                        chunk_time_interval => %s::interval
                    );
                    """,
                    (self.config.bar_chunk_interval,),
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bar_events_instrument_time ON bar_events (instrument_id, exchange_ts ASC);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bar_events_time_instrument ON bar_events (exchange_ts ASC, instrument_id ASC);"
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trade_ticks (
                        exchange_ts TIMESTAMPTZ NOT NULL,
                        instrument_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        trade_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        payload_version TEXT NOT NULL DEFAULT '1.0',
                        last_price DOUBLE PRECISION NOT NULL,
                        last_size DOUBLE PRECISION NOT NULL,
                        aggressor_side TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (exchange_ts, instrument_id, event_id)
                    );
                    """
                )
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'trade_ticks',
                        'exchange_ts',
                        if_not_exists => TRUE,
                        migrate_data => TRUE,
                        chunk_time_interval => %s::interval
                    );
                    """,
                    (self.config.tick_chunk_interval,),
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trade_ticks_instrument_time ON trade_ticks (instrument_id, exchange_ts ASC);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trade_ticks_time_instrument ON trade_ticks (exchange_ts ASC, instrument_id ASC);"
                )

    def write_bar_events(self, events: Sequence[BarEvent], *, page_size: int = 1000) -> int:
        if not events:
            return 0
        rows = [
            (
                event.exchange_ts,
                event.instrument_id,
                event.event_id,
                event.bar_id,
                event.source,
                event.payload_version,
                event.open_price,
                event.high_price,
                event.low_price,
                event.close_price,
                event.volume,
                Json(dict(event.metadata)),
            )
            for event in events
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """
                    INSERT INTO bar_events (
                        exchange_ts, instrument_id, event_id, bar_id, source, payload_version,
                        open_price, high_price, low_price, close_price, volume, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exchange_ts, instrument_id, event_id) DO NOTHING;
                    """,
                    rows,
                    page_size=page_size,
                )
        return len(rows)

    def write_trade_ticks(self, events: Sequence[TradeTick], *, page_size: int = 1000) -> int:
        if not events:
            return 0
        rows = [
            (
                event.exchange_ts,
                event.instrument_id,
                event.event_id,
                event.trade_id,
                event.source,
                event.payload_version,
                event.last_price,
                event.last_size,
                event.aggressor_side,
                Json(dict(event.metadata)),
            )
            for event in events
        ]
        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    """
                    INSERT INTO trade_ticks (
                        exchange_ts, instrument_id, event_id, trade_id, source, payload_version,
                        last_price, last_size, aggressor_side, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exchange_ts, instrument_id, event_id) DO NOTHING;
                    """,
                    rows,
                    page_size=page_size,
                )
        return len(rows)

    def read_bar_events(
        self,
        instrument_ids: Sequence[str],
        start_ts: datetime,
        end_ts: datetime,
        *,
        chunk_size: int = 10_000,
    ) -> Iterator[BarEvent]:
        query = """
            SELECT exchange_ts, instrument_id, event_id, bar_id, source, payload_version,
                   open_price, high_price, low_price, close_price, volume, metadata
            FROM bar_events
            WHERE instrument_id = ANY(%s)
              AND exchange_ts >= %s
              AND exchange_ts < %s
            ORDER BY exchange_ts ASC, instrument_id ASC, bar_id ASC;
        """
        with self.connection() as conn:
            with conn.cursor(name="bar_events_reader", cursor_factory=RealDictCursor) as cur:
                cur.itersize = chunk_size
                cur.execute(query, (list(instrument_ids), start_ts, end_ts))
                for row in cur:
                    yield self._row_to_bar_event(row)

    def read_trade_ticks(
        self,
        instrument_ids: Sequence[str],
        start_ts: datetime,
        end_ts: datetime,
        *,
        chunk_size: int = 10_000,
    ) -> Iterator[TradeTick]:
        query = """
            SELECT exchange_ts, instrument_id, event_id, trade_id, source, payload_version,
                   last_price, last_size, aggressor_side, metadata
            FROM trade_ticks
            WHERE instrument_id = ANY(%s)
              AND exchange_ts >= %s
              AND exchange_ts < %s
            ORDER BY exchange_ts ASC, instrument_id ASC, trade_id ASC;
        """
        with self.connection() as conn:
            with conn.cursor(name="trade_ticks_reader", cursor_factory=RealDictCursor) as cur:
                cur.itersize = chunk_size
                cur.execute(query, (list(instrument_ids), start_ts, end_ts))
                for row in cur:
                    yield self._row_to_trade_tick(row)

    def stream_event_rows(
        self,
        instrument_ids: Sequence[str],
        start_ts: datetime,
        end_ts: datetime,
        *,
        include_bars: bool = True,
        include_trade_ticks: bool = True,
        chunk_size: int = 10_000,
    ) -> Iterator[Mapping[str, Any]]:
        if not include_bars and not include_trade_ticks:
            return

        union_parts: list[str] = []
        params: list[Any] = []
        if include_bars:
            union_parts.append(
                """
                SELECT
                    exchange_ts, instrument_id, event_id, source, payload_version, metadata,
                    0 AS event_priority, 'bar' AS event_kind, bar_id,
                    open_price, high_price, low_price, close_price, volume,
                    NULL::DOUBLE PRECISION AS last_price, NULL::DOUBLE PRECISION AS last_size,
                    NULL::TEXT AS aggressor_side, NULL::TEXT AS trade_id
                FROM bar_events
                WHERE instrument_id = ANY(%s)
                  AND exchange_ts >= %s
                  AND exchange_ts < %s
                """
            )
            params.extend([list(instrument_ids), start_ts, end_ts])
        if include_trade_ticks:
            union_parts.append(
                """
                SELECT
                    exchange_ts, instrument_id, event_id, source, payload_version, metadata,
                    1 AS event_priority, 'trade_tick' AS event_kind, NULL::TEXT AS bar_id,
                    NULL::DOUBLE PRECISION AS open_price, NULL::DOUBLE PRECISION AS high_price,
                    NULL::DOUBLE PRECISION AS low_price, NULL::DOUBLE PRECISION AS close_price,
                    NULL::DOUBLE PRECISION AS volume,
                    last_price, last_size, aggressor_side, trade_id
                FROM trade_ticks
                WHERE instrument_id = ANY(%s)
                  AND exchange_ts >= %s
                  AND exchange_ts < %s
                """
            )
            params.extend([list(instrument_ids), start_ts, end_ts])

        query = f"""
            {' UNION ALL '.join(union_parts)}
            ORDER BY exchange_ts ASC, instrument_id ASC, event_priority ASC, COALESCE(bar_id, trade_id) ASC;
        """
        with self.connection() as conn:
            with conn.cursor(name="historical_events_reader", cursor_factory=RealDictCursor) as cur:
                cur.itersize = chunk_size
                cur.execute(query, tuple(params))
                for row in cur:
                    yield row

    @staticmethod
    def _row_to_bar_event(row: Mapping[str, Any]) -> BarEvent:
        exchange_ts = TimescaleDBClient._normalize_ts(row["exchange_ts"])
        return BarEvent(
            instrument_id=str(row["instrument_id"]),
            exchange_ts=exchange_ts,
            received_ts=exchange_ts,
            processed_ts=exchange_ts,
            sequence_id=0,
            source=str(row["source"]),
            event_id=str(row["event_id"]),
            payload_version=str(row.get("payload_version") or "1.0"),
            metadata=dict(row.get("metadata") or {}),
            bar_id=str(row["bar_id"]),
            open_price=float(row["open_price"]),
            high_price=float(row["high_price"]),
            low_price=float(row["low_price"]),
            close_price=float(row["close_price"]),
            volume=float(row["volume"]),
        )

    @staticmethod
    def _row_to_trade_tick(row: Mapping[str, Any]) -> TradeTick:
        exchange_ts = TimescaleDBClient._normalize_ts(row["exchange_ts"])
        return TradeTick(
            instrument_id=str(row["instrument_id"]),
            exchange_ts=exchange_ts,
            received_ts=exchange_ts,
            processed_ts=exchange_ts,
            sequence_id=0,
            source=str(row["source"]),
            event_id=str(row["event_id"]),
            payload_version=str(row.get("payload_version") or "1.0"),
            metadata=dict(row.get("metadata") or {}),
            trade_id=str(row["trade_id"]),
            last_price=float(row["last_price"]),
            last_size=float(row["last_size"]),
            aggressor_side=str(row.get("aggressor_side") or "unknown"),
        )

    @staticmethod
    def _normalize_ts(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
