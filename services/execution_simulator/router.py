"""Execution Simulator API - POST /simulate."""

import logging

from fastapi import APIRouter, Depends

from services.execution_simulator.schemas import AlgoResult, SimulateRequest, SimulateResponse
from services.execution_simulator.service import simulate_execution
from services.common.subscription import require_feature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/execution-sim", tags=["execution-simulator"])


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(
    body: SimulateRequest,
    user=Depends(require_feature("execution_simulator")),
):
    """
    Simulate execution costs across VWAP, TWAP, POV, Iceberg, and Market.
    Requires Basic tier or higher. Rate limited by tier.
    """
    out = simulate_execution(
        symbol=body.symbol,
        quantity=body.quantity,
        urgency=body.urgency,
        time_horizon_minutes=body.time_horizon_minutes,
        reference_price=150.0,  # Could be fetched from market data in future
        daily_volume=2_000_000,
    )
    return SimulateResponse(
        symbol=out["symbol"],
        quantity=out["quantity"],
        side=out["side"],
        reference_price=out["reference_price"],
        notional_usd=out["notional_usd"],
        results=[AlgoResult(**r) for r in out["results"]],
        best_algorithm=out["best_algorithm"],
    )
