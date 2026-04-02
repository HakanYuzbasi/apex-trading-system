"""Pydantic schemas for Compliance Copilot API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PreTradeCheckRequest(BaseModel):
    """Request body for POST /compliance/check."""

    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="'BUY' or 'SELL'")
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: float = Field(..., gt=0, description="Order price")
    portfolio_value: float = Field(..., gt=0, description="Current portfolio value")
    current_positions: Dict[str, int] = Field(
        default_factory=dict, description="Current positions by symbol"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Trading configuration overrides"
    )


class PreTradeCheckResponse(BaseModel):
    """Response from POST /compliance/check."""

    approved: bool
    violations: List[str] = []
    warnings: List[str] = []
    check_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    notional: float
    timestamp: str


class DailyReportResponse(BaseModel):
    """Response from GET /compliance/report."""

    date: str
    report: str


class StatisticsResponse(BaseModel):
    """Response from GET /compliance/statistics."""

    total_checks: int = 0
    approved: int = 0
    rejected: int = 0
    approval_rate: float = 0.0
    total_violations: int = 0
    total_trades_logged: int = 0


class AuditTrailResponse(BaseModel):
    """Response from GET /compliance/audit-trail."""

    date: str
    verified: bool
    message: Optional[str] = None
    tampered_records: List[Dict[str, Any]] = []
