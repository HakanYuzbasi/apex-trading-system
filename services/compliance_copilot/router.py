"""Compliance Copilot API Router - endpoints for compliance monitoring and reporting."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from services.compliance_copilot.schemas import (
    AuditTrailResponse,
    DailyReportResponse,
    PreTradeCheckRequest,
    PreTradeCheckResponse,
    StatisticsResponse,
)
from services.compliance_copilot.service import ComplianceCopilotService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])

_service: Optional[ComplianceCopilotService] = None


def _get_service() -> ComplianceCopilotService:
    global _service
    if _service is None:
        _service = ComplianceCopilotService()
    return _service


@router.post("/check", response_model=PreTradeCheckResponse)
async def pre_trade_check(request: PreTradeCheckRequest):
    """Run a pre-trade compliance check against position, concentration, and regulatory limits."""
    svc = _get_service()
    try:
        result = svc.pre_trade_check(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            portfolio_value=request.portfolio_value,
            current_positions=request.current_positions,
            config=request.config,
        )
    except Exception as e:
        logger.exception("Pre-trade check failed")
        raise HTTPException(status_code=500, detail=str(e))

    return PreTradeCheckResponse(
        approved=result["approved"],
        violations=result["violations"],
        warnings=result["warnings"],
        check_id=result["check_id"],
        symbol=result["symbol"],
        side=result["side"],
        quantity=result["quantity"],
        price=result["price"],
        notional=result["notional"],
        timestamp=result["timestamp"],
    )


@router.get("/report", response_model=DailyReportResponse)
async def daily_report(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")):
    """Generate a daily compliance report. Defaults to today if no date is provided."""
    svc = _get_service()
    try:
        report_text = svc.generate_daily_report(date=date)
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    from datetime import datetime

    return DailyReportResponse(
        date=date or datetime.now().strftime("%Y-%m-%d"),
        report=report_text,
    )


@router.get("/statistics", response_model=StatisticsResponse)
async def compliance_statistics():
    """Return compliance check statistics (totals, approval rate, violations)."""
    svc = _get_service()
    try:
        stats = svc.get_statistics()
    except Exception as e:
        logger.exception("Statistics retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not stats:
        return StatisticsResponse()

    return StatisticsResponse(**stats)


@router.get("/audit-trail", response_model=AuditTrailResponse)
async def verify_audit_trail(
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format"),
):
    """Verify the integrity of the audit trail for a given date. Defaults to today."""
    svc = _get_service()
    try:
        result = svc.verify_audit_trail(date=date)
    except Exception as e:
        logger.exception("Audit trail verification failed")
        raise HTTPException(status_code=500, detail=str(e))

    return AuditTrailResponse(
        date=result.get("date", ""),
        verified=result["verified"],
        message=result.get("message"),
        tampered_records=result.get("tampered_records", []),
    )
