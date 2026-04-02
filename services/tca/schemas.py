"""Pydantic schemas for TCA (Transaction Cost Analysis) API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TCASymbolRow(BaseModel):
    """Per-symbol execution quality metrics."""

    symbol: str
    closed_trades: int = 0
    win_rate_pct: Optional[float] = None
    net_pnl: float = 0.0
    execution_drag: float = 0.0
    avg_entry_slip_bps: Optional[float] = None
    avg_exit_slip_bps: Optional[float] = None
    median_fill_ms: Optional[float] = None
    p95_fill_ms: Optional[float] = None
    fills: int = 0
    exit_reasons: Dict[str, int] = {}
    open_position: bool = False
    rejections: Dict[str, int] = {}


class TCASummary(BaseModel):
    """Aggregate execution summary."""

    closed_trades: int = 0
    win_rate_pct: float = 0.0
    total_net_pnl: float = 0.0
    total_execution_drag: float = 0.0
    alpha_before_costs: float = 0.0
    cost_ratio_pct: float = 0.0
    total_fills: int = 0
    total_rejections: int = 0
    rejection_breakdown: Dict[str, int] = {}
    execution_health_score: float = 0.0


class OpenPositionInfo(BaseModel):
    """Open position slippage info."""

    entry_slippage_bps: float = 0.0
    entry_signal: float = 0.0
    entry_time: Optional[str] = None
    notional: float = 0.0


class TCAReportResponse(BaseModel):
    """Full TCA report response."""

    generated_at: str
    summary: TCASummary
    per_symbol: List[TCASymbolRow]
    open_book: Dict[str, OpenPositionInfo] = {}


class TCAStatusResponse(BaseModel):
    """Condensed execution health status."""

    execution_health_score: float
    closed_trades: int
    win_rate_pct: float
    total_net_pnl: float
    total_rejections: int
