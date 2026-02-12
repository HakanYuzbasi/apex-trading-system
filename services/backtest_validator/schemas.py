"""Pydantic schemas for Backtest Validator API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from services.common.schemas import JobStatus


class ValidateResponse(BaseModel):
    """Response from POST /validate."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    message: str = "Validation started"
    robustness_score: Optional[float] = None
    monte_carlo: Optional[Dict[str, float]] = None
    stress_test_summary: Optional[Dict[str, Any]] = None
    pdf_url: Optional[str] = None
    error: Optional[str] = None


class JobListItem(BaseModel):
    job_id: str
    feature_key: str
    status: JobStatus
    created_at: str
    result_summary: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    jobs: List[JobListItem]
    total: int
