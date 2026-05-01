from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping
from uuid import uuid4

EventScalar = str | int | float | bool | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_event_id(prefix: str = "evt") -> str:
    return f"{prefix}-{uuid4().hex}"


def _require_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_timezone_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class BaseEvent:
    instrument_id: str
    exchange_ts: datetime
    received_ts: datetime
    processed_ts: datetime
    sequence_id: int
    source: str
    event_id: str = field(default_factory=generate_event_id)
    payload_version: str = "1.0"
    metadata: Mapping[str, EventScalar] = field(default_factory=dict)
    event_type: str = field(init=False)

    def __post_init__(self) -> None:
        _require_non_empty("instrument_id", self.instrument_id)
        _require_non_empty("source", self.source)
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("payload_version", self.payload_version)
        _require_timezone_aware("exchange_ts", self.exchange_ts)
        _require_timezone_aware("received_ts", self.received_ts)
        _require_timezone_aware("processed_ts", self.processed_ts)
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.received_ts < self.exchange_ts:
            raise ValueError("received_ts cannot be earlier than exchange_ts")
        if self.processed_ts < self.received_ts:
            raise ValueError("processed_ts cannot be earlier than received_ts")
        object.__setattr__(self, "metadata", dict(self.metadata))
