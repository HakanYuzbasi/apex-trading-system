from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from quant_system.events.base import BaseEvent, generate_event_id


@dataclass(frozen=True, slots=True, kw_only=True)
class CorporateAction(BaseEvent):
    effective_date: date
    split_ratio: float | None = None
    dividend_cash: float | None = None
    action_id: str = field(default_factory=lambda: generate_event_id("corp"))
    event_type: Literal["corporate_action"] = field(init=False, default="corporate_action")

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        if not self.action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        if self.split_ratio is None and self.dividend_cash is None:
            raise ValueError("at least one of split_ratio or dividend_cash must be provided")
        if self.split_ratio is not None:
            if not math.isfinite(self.split_ratio):
                raise ValueError("split_ratio must be finite when provided")
            if self.split_ratio <= 0:
                raise ValueError("split_ratio must be positive when provided")
        if self.dividend_cash is not None:
            if not math.isfinite(self.dividend_cash):
                raise ValueError("dividend_cash must be finite when provided")
            if self.dividend_cash < 0:
                raise ValueError("dividend_cash must be non-negative when provided")
