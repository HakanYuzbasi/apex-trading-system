"""Drift Monitor API - GET /status, GET /report, GET /features."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from services.common.subscription import require_feature
from services.drift_monitor.schemas import (
    DriftReportResponse,
    DriftStatusResponse,
    FeatureDriftSummaryResponse,
)
from services.drift_monitor.service import DriftMonitorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift-monitor", tags=["drift-monitor"])

_service: Optional[DriftMonitorService] = None


def _get_service() -> DriftMonitorService:
    global _service
    if _service is None:
        _service = DriftMonitorService()
    return _service


@router.get("/status", response_model=DriftStatusResponse)
async def get_drift_status(
    user=Depends(require_feature("drift_monitor")),
):
    """Return current model health status. Requires Pro tier or higher."""
    svc = _get_service()
    return DriftStatusResponse(**svc.get_status())


@router.get("/report", response_model=DriftReportResponse)
async def get_drift_report(
    user=Depends(require_feature("drift_monitor")),
):
    """Return detailed drift report with window history. Requires Pro tier or higher."""
    svc = _get_service()
    return DriftReportResponse(**svc.get_report())


@router.get("/features", response_model=FeatureDriftSummaryResponse)
async def get_feature_drift_summary(
    days: int = Query(default=7, ge=1, le=90, description="Look-back period in days"),
    user=Depends(require_feature("drift_monitor")),
):
    """Return feature drift summary for the requested look-back period. Requires Pro tier or higher."""
    svc = _get_service()
    return FeatureDriftSummaryResponse(**svc.get_feature_drift_summary(days=days))
