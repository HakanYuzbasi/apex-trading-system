from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal, Mapping

from quant_system.events.base import BaseEvent, EventScalar, generate_event_id

SignalSide = Literal["buy", "sell", "short", "cover", "flatten", "rebalance"]
TargetType = Literal["units", "notional", "weight"]


@dataclass(frozen=True)
class SignalEvent(BaseEvent):
    _: KW_ONLY
    strategy_id: str
    side: SignalSide
    target_type: TargetType
    target_value: float
    confidence: float
    stop_model: str
    stop_params: Mapping[str, EventScalar] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: generate_event_id("sig"))
    metadata: Mapping[str, EventScalar] = field(default_factory=dict)
    event_type: Literal["signal"] = field(init=False, default="signal")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must be a non-empty string")
        if not math.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.stop_model.strip():
            raise ValueError("stop_model must be a non-empty string")
        if not self.signal_id.strip():
            raise ValueError("signal_id must be a non-empty string")
        object.__setattr__(self, "stop_params", dict(self.stop_params))
