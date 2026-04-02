"""Portfolio Allocator API - GET /allocation, POST /update-pnl, GET /state."""

import logging

from fastapi import APIRouter

from services.portfolio_allocator.schemas import (
    AllocationResponse,
    AllocatorStateResponse,
    PnlUpdateRequest,
    PnlUpdateResponse,
)
from services.portfolio_allocator.service import PortfolioAllocatorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio-allocator", tags=["Portfolio Allocator"])

# Module-level singleton so state persists across requests.
_service = PortfolioAllocatorService()


@router.get("/allocation", response_model=AllocationResponse)
async def get_allocation():
    """Return the current Kelly-optimal allocation recommendation."""
    return _service.get_allocation()


@router.post("/update-pnl", response_model=PnlUpdateResponse)
async def update_pnl(body: PnlUpdateRequest):
    """Record one day's realized P&L for both legs and return updated allocation."""
    result = _service.update_pnl(
        equity_pnl_pct=body.equity_pnl_pct,
        crypto_pnl_pct=body.crypto_pnl_pct,
        equity_trades=body.equity_trades,
        crypto_trades=body.crypto_trades,
        trade_date=body.trade_date,
    )
    return PnlUpdateResponse(
        allocation=AllocationResponse(**result),
    )


@router.get("/state", response_model=AllocatorStateResponse)
async def get_state():
    """Return full allocator state including history and last result."""
    return _service.get_state()
