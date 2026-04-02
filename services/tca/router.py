"""TCA (Transaction Cost Analysis) API - GET /report, GET /status."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from services.tca.schemas import TCAReportResponse, TCAStatusResponse
from services.tca.service import TCAService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tca", tags=["Transaction Cost Analysis"])


@router.get("/report", response_model=TCAReportResponse)
async def get_tca_report(
    data_dir: Optional[str] = Query(None, description="Override data directory"),
):
    """
    Return the full TCA report: per-symbol execution quality,
    P&L attribution, rejection analysis, and overall health score.
    """
    try:
        svc = TCAService(data_dir=data_dir)
        return svc.get_report()
    except Exception as e:
        logger.exception("Failed to build TCA report")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TCAStatusResponse)
async def get_tca_status(
    data_dir: Optional[str] = Query(None, description="Override data directory"),
):
    """Return a condensed execution health score summary."""
    try:
        svc = TCAService(data_dir=data_dir)
        return svc.get_status()
    except Exception as e:
        logger.exception("Failed to build TCA status")
        raise HTTPException(status_code=500, detail=str(e))
