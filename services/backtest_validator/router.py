"""Backtest Validator API - POST /validate, GET /jobs, GET /jobs/{id}/report."""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth.models import ServiceJobModel
from services.backtest_validator.schemas import JobListResponse, JobListItem, ValidateResponse
from services.backtest_validator.service import BacktestValidatorService
from services.common.db import get_db
from services.common.file_upload import parse_upload
from services.common.schemas import JobStatus
from services.common.subscription import require_feature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/backtest-validator", tags=["backtest-validator"])


@router.post("/validate", response_model=ValidateResponse)
async def validate_backtest(
    file: UploadFile = File(...),
    user=Depends(require_feature("backtest_validator")),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a backtest result (CSV or JSON) and receive robustness validation.
    Requires Basic tier or higher. Rate limited by tier.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    content = await file.read()
    try:
        upload = parse_upload(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    svc = BacktestValidatorService()
    result = svc.validate(upload, str(user_id), db)
    return ValidateResponse(
        job_id=result["job_id"],
        status=result["status"],
        robustness_score=result.get("robustness_score"),
        monte_carlo=result.get("monte_carlo"),
        stress_test_summary=result.get("stress_test_summary"),
        pdf_url=result.get("pdf_url"),
        error=result.get("error"),
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    user=Depends(require_feature("backtest_validator")),
    db: AsyncSession = Depends(get_db),
    limit: int = 20,
    offset: int = 0,
):
    """List validation jobs for the current user."""
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await db.execute(
        select(ServiceJobModel)
        .where(
            ServiceJobModel.user_id == user_id,
            ServiceJobModel.feature_key == "backtest_validator",
        )
        .order_by(ServiceJobModel.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = list(result.scalars().all())
    count_result = await db.execute(
        select(ServiceJobModel).where(
            ServiceJobModel.user_id == user_id,
            ServiceJobModel.feature_key == "backtest_validator",
        )
    )
    total = len(count_result.scalars().all())

    return JobListResponse(
        jobs=[
            JobListItem(
                job_id=j.id,
                feature_key=j.feature_key,
                status=j.status,
                created_at=j.created_at.isoformat(),
                result_summary=j.result_summary,
            )
            for j in jobs
        ],
        total=total,
    )


@router.get("/jobs/{job_id}/report")
async def get_job_report(
    job_id: str,
    user=Depends(require_feature("backtest_validator")),
    db: AsyncSession = Depends(get_db),
):
    """Download the PDF report for a completed job."""
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = await db.execute(
        select(ServiceJobModel).where(
            ServiceJobModel.id == job_id,
            ServiceJobModel.user_id == user_id,
            ServiceJobModel.feature_key == "backtest_validator",
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_file_path:
        raise HTTPException(status_code=404, detail="Report not available")

    from fastapi.responses import FileResponse
    import os
    if not os.path.isfile(job.result_file_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(
        job.result_file_path,
        media_type="application/pdf",
        filename=f"backtest_validation_{job_id}.pdf",
    )
