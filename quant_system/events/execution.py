from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from quant_system.events.base import BaseEvent, generate_event_id

ExecutionStatus = Literal["partial_fill", "filled", "canceled", "rejected", "expired"]
LiquidityFlag = Literal["maker", "taker"]
ExecutionSide = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionEvent(BaseEvent):
    execution_id: str = field(default_factory=lambda: generate_event_id("exe"))
    order_id: str
    side: ExecutionSide
    execution_status: ExecutionStatus
    fill_qty: float
    fill_price: float
    fees: float
    slippage: float
    parent_order_id: str | None = None
    venue_order_id: str | None = None
    broker: str | None = None
    venue: str | None = None
    liquidity_flag: LiquidityFlag | None = None
    remaining_qty: float | None = None
    event_type: Literal["execution"] = field(init=False, default="execution")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if self.fill_qty < 0:
            raise ValueError("fill_qty must be non-negative")
        if self.execution_status in {"partial_fill", "filled"} and self.fill_qty <= 0:
            raise ValueError("filled execution events require fill_qty > 0")
        if self.fill_price < 0:
            raise ValueError("fill_price must be non-negative")
        if self.execution_status in {"partial_fill", "filled"} and self.fill_price <= 0:
            raise ValueError("filled execution events require fill_price > 0")
        if not math.isfinite(self.fees):
            raise ValueError("fees must be finite")
        if not math.isfinite(self.slippage):
            raise ValueError("slippage must be finite")
        if self.remaining_qty is not None and self.remaining_qty < 0:
            raise ValueError("remaining_qty must be non-negative when provided")
