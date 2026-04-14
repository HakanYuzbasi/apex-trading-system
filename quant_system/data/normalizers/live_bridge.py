from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import count
from typing import Any, Iterable, Mapping, Sequence

from core.symbols import normalize_symbol
from quant_system.core.bus import InMemoryEventBus
from quant_system.events.base import BaseEvent
from quant_system.events.market import BarEvent, OrderBookTick, QuoteTick, TradeTick


class LiveDataBridge:
    """
    Normalize venue-native live payloads into canonical market events and publish
    them onto the shared event bus.

    Supported upstream shapes:
    - Alpaca websocket arrays containing `T=t` / `T=q` event dictionaries.
    - Internal Alpaca queue dictionaries produced by `live_websocket.py`.
    - IBKR dictionaries carrying bid/ask/last style market data snapshots.
    """

    def __init__(self, event_bus: InMemoryEventBus, *, source_prefix: str = "live_bridge") -> None:
        self._event_bus = event_bus
        self._source_prefix = source_prefix
        self._sequence = count()

    async def publish_alpaca(self, payload: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[BaseEvent]:
        events = self.normalize_alpaca(payload)
        for event in events:
            await self._event_bus.publish_async(event)
        return events

    async def publish_ibkr(self, payload: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[BaseEvent]:
        events = self.normalize_ibkr(payload)
        for event in events:
            await self._event_bus.publish_async(event)
        return events

    def normalize_alpaca(self, payload: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[BaseEvent]:
        decoded = self._decode_payload(payload)
        raw_events = self._coerce_sequence(decoded)
        events: list[BaseEvent] = []
        for raw_event in raw_events:
            event_type = str(raw_event.get("T") or raw_event.get("stream") or "").lower()
            if event_type in {"q", "quote"}:
                events.append(self._build_quote_tick(raw_event, venue="alpaca"))
            elif event_type in {"t", "trade"}:
                events.append(self._build_trade_tick(raw_event, venue="alpaca"))
            elif event_type in {"b", "bar", "u", "updated_bar"}:
                events.append(self._build_bar_event(raw_event, venue="alpaca"))
            elif event_type in {"o", "ob", "book", "l2", "l2_book"}:
                events.append(self._build_order_book_tick(raw_event, venue="alpaca"))
        return events

    def normalize_ibkr(self, payload: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[BaseEvent]:
        decoded = self._decode_payload(payload)
        raw_events = self._coerce_sequence(decoded)
        events: list[BaseEvent] = []
        for raw_event in raw_events:
            has_quote = self._has_any(raw_event, "bid", "bidPrice", "ask", "askPrice")
            has_trade = self._has_any(raw_event, "last", "lastPrice", "price")
            if has_quote:
                events.append(self._build_quote_tick(raw_event, venue="ibkr"))
            if has_trade:
                trade_event = self._build_trade_tick(raw_event, venue="ibkr", allow_missing_size=True)
                if trade_event is not None:
                    events.append(trade_event)
        return events

    def _build_quote_tick(self, payload: Mapping[str, Any], *, venue: str) -> QuoteTick:
        exchange_ts = self._parse_timestamp(
            self._first_value(payload, "t", "timestamp", "time", "updated_at"),
        )
        received_ts = self._parse_timestamp(
            self._first_value(payload, "received_ts", "receivedAt"),
            default=exchange_ts,
        )
        processed_ts = self._parse_timestamp(None, default=max(received_ts, self._utc_now()))
        instrument_id = self._normalize_instrument_id(payload, venue=venue)
        bid = self._as_float(self._first_value(payload, "bp", "bid", "bidPrice"))
        ask = self._as_float(self._first_value(payload, "ap", "ask", "askPrice"))
        bid_size = self._as_float(self._first_value(payload, "bs", "bid_size", "bidSize"), default=0.0)
        ask_size = self._as_float(self._first_value(payload, "as", "ask_size", "askSize"), default=0.0)
        return QuoteTick(
            instrument_id=instrument_id,
            exchange_ts=exchange_ts,
            received_ts=received_ts,
            processed_ts=processed_ts,
            sequence_id=next(self._sequence),
            source=f"{self._source_prefix}.{venue}",
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            metadata={"venue": venue},
        )

    def _build_trade_tick(
        self,
        payload: Mapping[str, Any],
        *,
        venue: str,
        allow_missing_size: bool = False,
    ) -> TradeTick | None:
        exchange_ts = self._parse_timestamp(
            self._first_value(payload, "t", "timestamp", "time", "updated_at"),
        )
        received_ts = self._parse_timestamp(
            self._first_value(payload, "received_ts", "receivedAt"),
            default=exchange_ts,
        )
        processed_ts = self._parse_timestamp(None, default=max(received_ts, self._utc_now()))
        instrument_id = self._normalize_instrument_id(payload, venue=venue)
        last_price = self._as_float(self._first_value(payload, "p", "price", "last", "lastPrice"))
        last_size = self._as_float(
            self._first_value(payload, "s", "size", "lastSize"),
            default=0.0 if allow_missing_size else None,
        )
        if allow_missing_size and last_size <= 0:
            return None
        aggressor_side = self._normalize_aggressor_side(
            self._first_value(payload, "tks", "aggressor_side", "side"),
        )
        trade_id = str(
            self._first_value(payload, "i", "trade_id", "tradeId", "execId")
            or f"{instrument_id}:{exchange_ts.isoformat()}:{next(self._sequence)}"
        )
        return TradeTick(
            instrument_id=instrument_id,
            exchange_ts=exchange_ts,
            received_ts=received_ts,
            processed_ts=processed_ts,
            sequence_id=next(self._sequence),
            source=f"{self._source_prefix}.{venue}",
            last_price=last_price,
            last_size=last_size,
            aggressor_side=aggressor_side,
            trade_id=trade_id,
            metadata={"venue": venue},
        )

    def _build_bar_event(self, payload: Mapping[str, Any], *, venue: str) -> BarEvent:
        exchange_ts = self._parse_timestamp(
            self._first_value(payload, "t", "timestamp", "time", "updated_at"),
        )
        received_ts = self._parse_timestamp(
            self._first_value(payload, "received_ts", "receivedAt"),
            default=exchange_ts,
        )
        processed_ts = self._parse_timestamp(None, default=max(received_ts, self._utc_now()))
        instrument_id = self._normalize_instrument_id(payload, venue=venue)
        return BarEvent(
            instrument_id=instrument_id,
            exchange_ts=exchange_ts,
            received_ts=received_ts,
            processed_ts=processed_ts,
            sequence_id=next(self._sequence),
            source=f"{self._source_prefix}.{venue}",
            open_price=self._as_float(self._first_value(payload, "o", "open")),
            high_price=self._as_float(self._first_value(payload, "h", "high")),
            low_price=self._as_float(self._first_value(payload, "l", "low")),
            close_price=self._as_float(self._first_value(payload, "c", "close")),
            volume=self._as_float(self._first_value(payload, "v", "volume"), default=0.0),
            metadata={"venue": venue},
        )

    def _build_order_book_tick(self, payload: Mapping[str, Any], *, venue: str) -> OrderBookTick:
        exchange_ts = self._parse_timestamp(
            self._first_value(payload, "t", "timestamp", "time", "updated_at"),
        )
        received_ts = self._parse_timestamp(
            self._first_value(payload, "received_ts", "receivedAt"),
            default=exchange_ts,
        )
        processed_ts = self._parse_timestamp(None, default=max(received_ts, self._utc_now()))
        instrument_id = self._normalize_instrument_id(payload, venue=venue)

        # Alpaca L2 shape: "b": list of dicts/lists, "a": list of dicts/lists
        raw_bids = payload.get("b") or payload.get("bids") or []
        raw_asks = payload.get("a") or payload.get("asks") or []

        bid_levels: list[tuple[float, float]] = []
        for level in raw_bids:
            if isinstance(level, dict):
                bid_levels.append((float(level["p"]), float(level["s"])))
            else:
                bid_levels.append((float(level[0]), float(level[1])))

        ask_levels: list[tuple[float, float]] = []
        for level in raw_asks:
            if isinstance(level, dict):
                ask_levels.append((float(level["p"]), float(level["s"])))
            else:
                ask_levels.append((float(level[0]), float(level[1])))

        return OrderBookTick(
            instrument_id=instrument_id,
            exchange_ts=exchange_ts,
            received_ts=received_ts,
            processed_ts=processed_ts,
            sequence_id=next(self._sequence),
            source=f"{self._source_prefix}.{venue}",
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            metadata={"venue": venue},
        )

    @staticmethod
    def _decode_payload(payload: str | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
        if isinstance(payload, str):
            return json.loads(payload)
        return payload

    @staticmethod
    def _coerce_sequence(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            return [payload]
        if isinstance(payload, Sequence):
            return [item for item in payload if isinstance(item, Mapping)]
        raise TypeError("payload must be a JSON string, mapping, or sequence of mappings")

    @staticmethod
    def _first_value(payload: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    @staticmethod
    def _has_any(payload: Mapping[str, Any], *keys: str) -> bool:
        return any(key in payload and payload[key] is not None for key in keys)

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float:
        if value is None:
            if default is None:
                raise ValueError("missing numeric field")
            return float(default)
        converted = float(value)
        if converted != converted or converted in (float("inf"), float("-inf")):
            raise ValueError("numeric field must be finite")
        return converted

    @classmethod
    def _parse_timestamp(cls, value: Any, *, default: datetime | None = None) -> datetime:
        if value is None:
            if default is not None:
                return default
            return cls._utc_now()
        if isinstance(value, datetime):
            dt = value
        elif hasattr(value, "seconds") and hasattr(value, "nanoseconds"):
            timestamp_sec = getattr(value, "seconds") + getattr(value, "nanoseconds") / 1e9
            dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        else:
            raw = str(value).strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(raw)
            except ValueError:
                if raw.startswith("Timestamp(seconds="):
                    import re
                    match = re.search(r"seconds=(\d+),\s*nanoseconds=(\d+)", raw)
                    if match:
                        timestamp_sec = int(match.group(1)) + int(match.group(2)) / 1e9
                        dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
                    else:
                        raise
                else:
                    raise
        if dt.tzinfo is None or dt.utcoffset() is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _normalize_instrument_id(cls, payload: Mapping[str, Any], *, venue: str) -> str:
        raw_symbol = str(
            cls._first_value(payload, "symbol", "S", "localSymbol", "ticker", "contract")
            or ""
        ).strip()
        if not raw_symbol:
            raise ValueError(f"missing symbol in {venue} payload")

        candidate = raw_symbol.upper().replace("-", "/") if venue == "alpaca" and "/" in raw_symbol else raw_symbol.upper()
        if venue == "alpaca" and "/" not in candidate and len(candidate) == 6:
            candidate = f"{candidate[:3]}/{candidate[3:]}"

        try:
            return normalize_symbol(candidate)
        except Exception:
            return raw_symbol.upper()

    @staticmethod
    def _normalize_aggressor_side(value: Any) -> str:
        if value is None:
            return "unknown"
        side = str(value).strip().lower()
        if side in {"b", "buy", "bid", "bot"}:
            return "buy"
        if side in {"s", "sell", "ask", "sold", "sld"}:
            return "sell"
        return "unknown"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
