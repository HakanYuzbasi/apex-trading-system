from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from quant_system.events.base import BaseEvent, generate_event_id

AggressorSide = Literal["buy", "sell", "unknown"]


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True, kw_only=True)
class QuoteTick(BaseEvent):
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    quote_id: str = field(default_factory=lambda: generate_event_id("qt"))
    event_type: Literal["quote_tick"] = field(init=False, default="quote_tick")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.quote_id.strip():
            raise ValueError("quote_id must be a non-empty string")
        _require_finite("bid", self.bid)
        _require_finite("ask", self.ask)
        _require_finite("bid_size", self.bid_size)
        _require_finite("ask_size", self.ask_size)
        if self.bid < 0 or self.ask < 0:
            raise ValueError("bid and ask must be non-negative")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("bid_size and ask_size must be non-negative")
        if self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")


@dataclass(frozen=True, slots=True, kw_only=True)
class TradeTick(BaseEvent):
    last_price: float
    last_size: float
    aggressor_side: AggressorSide
    trade_id: str
    event_type: Literal["trade_tick"] = field(init=False, default="trade_tick")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.trade_id.strip():
            raise ValueError("trade_id must be a non-empty string")
        _require_finite("last_price", self.last_price)
        _require_finite("last_size", self.last_size)
        if self.last_price <= 0:
            raise ValueError("last_price must be positive")
        if self.last_size <= 0:
            raise ValueError("last_size must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class BarEvent(BaseEvent):
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    bar_id: str = field(default_factory=lambda: generate_event_id("bar"))
    event_type: Literal["bar"] = field(init=False, default="bar")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.bar_id.strip():
            raise ValueError("bar_id must be a non-empty string")
        for name in ("open_price", "high_price", "low_price", "close_price", "volume"):
            _require_finite(name, getattr(self, name))
        if self.open_price <= 0 or self.high_price <= 0 or self.low_price <= 0 or self.close_price <= 0:
            raise ValueError("bar prices must be positive")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")
        if self.high_price < max(self.open_price, self.close_price, self.low_price):
            raise ValueError("high_price must be greater than or equal to open/close/low")
        if self.low_price > min(self.open_price, self.close_price, self.high_price):
            raise ValueError("low_price must be less than or equal to open/close/high")


@dataclass(frozen=True, slots=True, kw_only=True)
class GreeksSnapshot(BaseEvent):
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    snapshot_id: str = field(default_factory=lambda: generate_event_id("grk"))
    event_type: Literal["greeks_snapshot"] = field(init=False, default="greeks_snapshot")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        for name in ("iv", "delta", "gamma", "theta", "vega", "rho", "underlying_price"):
            _require_finite(name, getattr(self, name))
        if self.iv < 0:
            raise ValueError("iv must be non-negative")
        if not -1.0 <= self.delta <= 1.0:
            raise ValueError("delta must be between -1.0 and 1.0")
        if self.underlying_price <= 0:
            raise ValueError("underlying_price must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class FundingSnapshot(BaseEvent):
    funding_rate: float
    next_funding_ts: datetime
    snapshot_id: str = field(default_factory=lambda: generate_event_id("fund"))
    event_type: Literal["funding_snapshot"] = field(init=False, default="funding_snapshot")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        _require_finite("funding_rate", self.funding_rate)
        if self.next_funding_ts.tzinfo is None or self.next_funding_ts.utcoffset() is None:
            raise ValueError("next_funding_ts must be timezone-aware")
        if self.next_funding_ts < self.exchange_ts:
            raise ValueError("next_funding_ts cannot be earlier than exchange_ts")

@dataclass(frozen=True, slots=True, kw_only=True)
class OrderBookTick(BaseEvent):
    """
    Level 2 Order Book Snapshot.
    bid_levels: List of (price, size) tuples, sorted descending by price (best bid first).
    ask_levels: List of (price, size) tuples, sorted ascending by price (best ask first).
    """
    bid_levels: list[tuple[float, float]]
    ask_levels: list[tuple[float, float]]
    book_id: str = field(default_factory=lambda: generate_event_id("ob"))
    event_type: Literal["order_book_tick"] = field(init=False, default="order_book_tick")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.book_id.strip():
            raise ValueError("book_id must be a non-empty string")
        
        for price, size in self.bid_levels:
            _require_finite("bid_price", price)
            _require_finite("bid_size", size)
            if price < 0 or size < 0:
                raise ValueError("bid price and size must be non-negative")
                
        for price, size in self.ask_levels:
            _require_finite("ask_price", price)
            _require_finite("ask_size", size)
            if price < 0 or size < 0:
                raise ValueError("ask price and size must be non-negative")

    @property
    def best_bid(self) -> float:
        return self.bid_levels[0][0] if self.bid_levels else 0.0

    @property
    def best_ask(self) -> float:
        return self.ask_levels[0][0] if self.ask_levels else 0.0

    @property
    def best_bid_size(self) -> float:
        return self.bid_levels[0][1] if self.bid_levels else 0.0

    @property
    def best_ask_size(self) -> float:
        return self.ask_levels[0][1] if self.ask_levels else 0.0
