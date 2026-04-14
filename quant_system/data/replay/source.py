from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from quant_system.events import BarEvent, TradeTick
from quant_system.events.base import BaseEvent


@dataclass(frozen=True, slots=True, order=True)
class _ReplayCursor:
    timestamp: datetime
    instrument_rank: int
    row_number: int
    instrument_id: str
    row: Mapping[str, Any]


class HistoricalReplaySource:
    """
    Historical replay source with two modes:
    - in-memory DataFrame mode for tests and synthetic research fixtures
    - database-backed streaming mode for production historical replay
    """

    def __init__(
        self,
        datasets: Mapping[str, pd.DataFrame] | None = None,
        *,
        client: Any | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        instrument_ids: Sequence[str] | None = None,
        include_bars: bool = True,
        include_trade_ticks: bool = True,
        chunk_size: int = 10_000,
        default_source: str = "historical_replay",
        payload_version: str = "1.0",
    ) -> None:
        if datasets is None and client is None:
            raise ValueError("provide either datasets or a database client")
        if client is not None and (start_ts is None or end_ts is None or instrument_ids is None):
            raise ValueError("database-backed replay requires client, start_ts, end_ts, and instrument_ids")

        self._datasets = dict(datasets or {})
        self._client = client
        self._start_ts = self._normalize_ts(start_ts) if start_ts is not None else None
        self._end_ts = self._normalize_ts(end_ts) if end_ts is not None else None
        self._instrument_ids = tuple(instrument_ids or ())
        self._include_bars = include_bars
        self._include_trade_ticks = include_trade_ticks
        self._chunk_size = chunk_size
        self._default_source = default_source
        self._payload_version = payload_version
        self._event_sequence = count()

    def __iter__(self) -> Iterator[BaseEvent]:
        return self.iter_events()

    def iter_events(self) -> Iterator[BaseEvent]:
        if self._client is not None:
            yield from self._iter_events_from_store()
            return
        yield from self._iter_events_from_datasets()

    def _iter_events_from_store(self) -> Iterator[BaseEvent]:
        assert self._client is not None
        assert self._start_ts is not None
        assert self._end_ts is not None
        rows = self._client.stream_event_rows(
            instrument_ids=self._instrument_ids,
            start_ts=self._start_ts,
            end_ts=self._end_ts,
            include_bars=self._include_bars,
            include_trade_ticks=self._include_trade_ticks,
            chunk_size=self._chunk_size,
        )
        for row in rows:
            yield self._row_to_event(row)

    def _iter_events_from_datasets(self) -> Iterator[BaseEvent]:
        heap: list[_ReplayCursor] = []
        prepared_frames: list[tuple[str, pd.DataFrame]] = []

        for instrument_rank, (instrument_id, frame) in enumerate(self._datasets.items()):
            normalized_frame = self._prepare_frame(frame)
            prepared_frames.append((instrument_id, normalized_frame))
            if normalized_frame.empty:
                continue
            first_row = normalized_frame.iloc[0]
            heapq.heappush(
                heap,
                _ReplayCursor(
                    timestamp=self._extract_timestamp(first_row),
                    instrument_rank=instrument_rank,
                    row_number=0,
                    instrument_id=instrument_id,
                    row=first_row.to_dict(),
                ),
            )

        while heap:
            cursor = heapq.heappop(heap)
            yield self._to_event(cursor.instrument_id, cursor.row)

            next_row_number = cursor.row_number + 1
            frame = prepared_frames[cursor.instrument_rank][1]
            if next_row_number >= len(frame):
                continue
            next_row = frame.iloc[next_row_number]
            heapq.heappush(
                heap,
                _ReplayCursor(
                    timestamp=self._extract_timestamp(next_row),
                    instrument_rank=cursor.instrument_rank,
                    row_number=next_row_number,
                    instrument_id=cursor.instrument_id,
                    row=next_row.to_dict(),
                ),
            )

    @staticmethod
    def _normalize_ts(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "timestamp" not in working.columns:
            working = working.reset_index()
            if "timestamp" not in working.columns:
                working = working.rename(columns={working.columns[0]: "timestamp"})
        if "timestamp" not in working.columns:
            raise ValueError("historical dataset must provide a DatetimeIndex or timestamp column")

        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
        working = working.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return working

    @staticmethod
    def _extract_timestamp(row: pd.Series) -> datetime:
        value = row["timestamp"]
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raise TypeError("timestamp must be datetime-like")

    def _to_event(self, instrument_id: str, row: Mapping[str, Any]) -> BaseEvent:
        exchange_ts = self._extract_timestamp(pd.Series(row))
        normalized_row = {
            "event_kind": "bar" if self._is_bar_row(row) else "trade_tick",
            "exchange_ts": exchange_ts,
            "instrument_id": instrument_id,
            "event_id": row.get("event_id") or f"{instrument_id}:{exchange_ts.isoformat()}:{next(self._event_sequence)}",
            "source": str(row.get("source") or self._default_source),
            "payload_version": str(row.get("payload_version") or self._payload_version),
            "metadata": {"replay_instrument_id": instrument_id},
            "bar_id": row.get("bar_id"),
            "trade_id": row.get("trade_id"),
            "open_price": row.get("Open", row.get("open")),
            "high_price": row.get("High", row.get("high")),
            "low_price": row.get("Low", row.get("low")),
            "close_price": row.get("Close", row.get("close")),
            "volume": row.get("Volume", row.get("volume")),
            "last_price": row.get("price", row.get("Price", row.get("last_price", row.get("Close", row.get("close"))))),
            "last_size": row.get("size", row.get("Size", row.get("volume", row.get("Volume", 1.0)))),
            "aggressor_side": row.get("aggressor_side", "unknown"),
        }
        return self._row_to_event(normalized_row)

    def _row_to_event(self, row: Mapping[str, Any]) -> BaseEvent:
        exchange_ts = self._coerce_datetime(row["exchange_ts"])
        exchange_ts = self._normalize_ts(exchange_ts)
        sequence_id = next(self._event_sequence)
        source = str(row.get("source") or self._default_source)
        payload_version = str(row.get("payload_version") or self._payload_version)
        metadata = dict(row.get("metadata") or {})
        event_id = str(row.get("event_id") or f"{row['instrument_id']}:{exchange_ts.isoformat()}:{sequence_id}")
        event_kind = str(row.get("event_kind") or "bar")

        if event_kind == "bar":
            return BarEvent(
                instrument_id=str(row["instrument_id"]),
                exchange_ts=exchange_ts,
                received_ts=exchange_ts,
                processed_ts=exchange_ts,
                sequence_id=sequence_id,
                source=source,
                event_id=event_id,
                payload_version=payload_version,
                metadata=metadata,
                bar_id=str(row.get("bar_id") or event_id),
                open_price=float(row["open_price"]),
                high_price=float(row["high_price"]),
                low_price=float(row["low_price"]),
                close_price=float(row["close_price"]),
                volume=float(row.get("volume", 0.0)),
            )

        return TradeTick(
            instrument_id=str(row["instrument_id"]),
            exchange_ts=exchange_ts,
            received_ts=exchange_ts,
            processed_ts=exchange_ts,
            sequence_id=sequence_id,
            source=source,
            event_id=event_id,
            payload_version=payload_version,
            metadata=metadata,
            last_price=float(row["last_price"]),
            last_size=float(row.get("last_size", 1.0)),
            aggressor_side=str(row.get("aggressor_side") or "unknown"),
            trade_id=str(row.get("trade_id") or event_id),
        )

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return datetime.fromisoformat(str(value))

    @staticmethod
    def _is_bar_row(row: Mapping[str, Any]) -> bool:
        lower_keys = {str(key).lower() for key in row.keys()}
        return {"open", "high", "low", "close"}.issubset(lower_keys)
