"""Pydantic schemas for Portfolio Allocator API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class PnlUpdateRequest(BaseModel):
    """Request body for POST /update-pnl."""

    equity_pnl_pct: float = Field(..., description="Daily realized equity return as fraction")
    crypto_pnl_pct: float = Field(..., description="Daily realized crypto return as fraction")
    equity_trades: int = Field(0, description="Number of equity trades")
    crypto_trades: int = Field(0, description="Number of crypto trades")
    trade_date: Optional[str] = Field(None, description="Trade date (YYYY-MM-DD). Defaults to today.")


class AllocationResponse(BaseModel):
    """Response from GET /allocation."""

    equity_frac: float
    crypto_frac: float
    equity_sharpe: float
    crypto_sharpe: float
    correlation: float
    rebalance_recommended: bool
    reason: str
    timestamp: str


class LegPerfItem(BaseModel):
    """Single day P&L snapshot for one leg."""

    date: str
    pnl_pct: float
    trades: int


class AllocatorStateResponse(BaseModel):
    """Response from GET /state."""

    current_equity_frac: float
    current_crypto_frac: float
    equity_history: List[LegPerfItem]
    crypto_history: List[LegPerfItem]
    last_result: Optional[AllocationResponse] = None


class PnlUpdateResponse(BaseModel):
    """Response from POST /update-pnl."""

    status: str = "ok"
    message: str = "P&L updated"
    allocation: AllocationResponse
