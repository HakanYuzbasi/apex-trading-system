"""
services/common/schemas.py - Shared Pydantic models for all SaaS services.

Defines subscription tiers, feature access, job tracking, and common responses.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Subscription tiers
# ---------------------------------------------------------------------------

class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


TIER_RANK: Dict[SubscriptionTier, int] = {
    SubscriptionTier.FREE: 0,
    SubscriptionTier.BASIC: 1,
    SubscriptionTier.PRO: 2,
    SubscriptionTier.ENTERPRISE: 3,
}


# ---------------------------------------------------------------------------
# Feature access matrix  (feature_key → min tier + daily limit per tier)
# ---------------------------------------------------------------------------

class FeatureLimit(BaseModel):
    min_tier: SubscriptionTier
    daily_limits: Dict[SubscriptionTier, int] = Field(
        default_factory=dict,
        description="Daily request limit per tier. -1 = unlimited.",
    )


FEATURE_MATRIX: Dict[str, FeatureLimit] = {
    "mandate_copilot_preview": FeatureLimit(
        min_tier=SubscriptionTier.FREE,
        daily_limits={
            SubscriptionTier.FREE: 25,
            SubscriptionTier.BASIC: 150,
            SubscriptionTier.PRO: 500,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "mandate_workflow_pack": FeatureLimit(
        min_tier=SubscriptionTier.PRO,
        daily_limits={
            SubscriptionTier.PRO: 60,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "mandate_model_risk_report": FeatureLimit(
        min_tier=SubscriptionTier.PRO,
        daily_limits={
            SubscriptionTier.PRO: 120,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "backtest_validator": FeatureLimit(
        min_tier=SubscriptionTier.BASIC,
        daily_limits={
            SubscriptionTier.BASIC: 10,
            SubscriptionTier.PRO: 100,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "execution_simulator": FeatureLimit(
        min_tier=SubscriptionTier.BASIC,
        daily_limits={
            SubscriptionTier.BASIC: 10,
            SubscriptionTier.PRO: 100,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "drift_monitor": FeatureLimit(
        min_tier=SubscriptionTier.PRO,
        daily_limits={
            SubscriptionTier.PRO: -1,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "portfolio_allocator": FeatureLimit(
        min_tier=SubscriptionTier.PRO,
        daily_limits={
            SubscriptionTier.PRO: 50,
            SubscriptionTier.ENTERPRISE: -1,
        },
    ),
    "compliance_copilot": FeatureLimit(
        min_tier=SubscriptionTier.ENTERPRISE,
        daily_limits={
            SubscriptionTier.ENTERPRISE: 50,
        },
    ),
}


# ---------------------------------------------------------------------------
# Service job tracking
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ServiceJobBase(BaseModel):
    feature_key: str
    input_params: Dict[str, Any] = Field(default_factory=dict)


class ServiceJobCreate(ServiceJobBase):
    pass


class ServiceJobResponse(ServiceJobBase):
    job_id: str
    user_id: str
    status: JobStatus = JobStatus.PENDING
    result_summary: Optional[Dict[str, Any]] = None
    result_file_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# User / subscription schemas (API-facing, separate from ORM models)
# ---------------------------------------------------------------------------

class UserPublic(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=lambda: ["user"])
    tier: SubscriptionTier = SubscriptionTier.FREE

    model_config = {"from_attributes": True}


class SubscriptionInfo(BaseModel):
    tier: SubscriptionTier
    status: str = "active"
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    stripe_subscription_id: Optional[str] = None
    features: Dict[str, int] = Field(
        default_factory=dict,
        description="Feature key → daily limit for current tier",
    )


class PlanOffer(BaseModel):
    code: str
    name: str
    tier: SubscriptionTier
    monthly_usd: int
    annual_usd: int
    recommended: bool = False
    target_user: str
    usp: str
    feature_highlights: List[str] = Field(default_factory=list)
    feature_limits: Dict[str, int] = Field(default_factory=dict)


class RateLimitInfo(BaseModel):
    feature_key: str
    daily_limit: int
    used_today: int
    remaining: int


# ---------------------------------------------------------------------------
# Common API responses
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int = 1
    page_size: int = 20
    has_more: bool = False
