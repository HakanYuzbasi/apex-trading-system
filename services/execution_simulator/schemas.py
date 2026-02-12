"""Pydantic schemas for Execution Simulator API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: int = Field(..., gt=0, le=10_000_000)
    urgency: str = Field(default="medium", description="low, medium, high, critical")
    time_horizon_minutes: Optional[int] = Field(default=None, ge=1, le=480)


class AlgoResult(BaseModel):
    algorithm: str
    estimated_cost_bps: float
    estimated_cost_usd: float
    time_to_complete_minutes: float
    market_impact_bps: float
    recommendation: Optional[str] = None


class SimulateResponse(BaseModel):
    symbol: str
    quantity: int
    side: str = "BUY"
    reference_price: float
    notional_usd: float
    results: List[AlgoResult]
    best_algorithm: str
