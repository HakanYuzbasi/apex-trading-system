from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from quant_system.events.base import BaseEvent, generate_event_id

OrderAction = Literal["submit", "modify", "cancel", "acknowledge", "reject"]
OrderScope = Literal["parent", "child"]
OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit", "market_on_close", "limit_on_close"]
TimeInForce = Literal["day", "gtc", "ioc", "fok", "opg", "cls"]
ExecutionAlgo = Literal["manual", "direct", "twap", "vwap", "pov", "iceberg", "sor"]


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderEvent(BaseEvent):
    order_id: str = field(default_factory=lambda: generate_event_id("ord"))
    order_action: OrderAction
    order_scope: OrderScope
    side: OrderSide
    order_type: OrderType
    quantity: float
    time_in_force: TimeInForce
    execution_algo: ExecutionAlgo
    parent_order_id: str | None = None
    strategy_id: str | None = None
    broker: str | None = None
    venue: str | None = None
    notional: float | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    event_type: Literal["order"] = field(init=False, default="order")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.order_action in {"submit", "modify"} and self.quantity <= 0:
            raise ValueError("submit/modify order events require quantity > 0")
        if self.order_scope == "child" and not self.parent_order_id:
            raise ValueError("child orders must reference parent_order_id")
        if self.order_scope == "parent" and self.parent_order_id is not None:
            raise ValueError("parent orders cannot define parent_order_id")
        if self.notional is not None and not math.isfinite(self.notional):
            raise ValueError("notional must be finite when provided")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided")
        if self.order_type in {"limit", "limit_on_close"} and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop orders require stop_price")
        if self.order_type == "stop_limit" and (self.stop_price is None or self.limit_price is None):
            raise ValueError("stop_limit orders require both stop_price and limit_price")
