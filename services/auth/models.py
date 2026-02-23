"""
services/auth/models.py - SQLAlchemy ORM models for authentication and subscriptions.

Tables:
    - users: Core user accounts
    - subscriptions: Stripe-backed subscription tiers
    - user_roles: Role assignments
    - api_keys: API key management
    - feature_access: Tier → feature access matrix (seeded)
    - service_jobs: Async job tracking across all SaaS features
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship

from services.common.db import Base
from services.common.schemas import JobStatus, SubscriptionTier


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # null for OAuth-only users
    is_active = Column(Boolean, default=True, nullable=False)

    # MFA
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(64), nullable=True)

    # OAuth
    oauth_provider = Column(String(50), nullable=True)  # "google", "github"
    oauth_id = Column(String(255), nullable=True)

    # Stripe
    stripe_customer_id = Column(String(255), nullable=True, unique=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    subscription = relationship("SubscriptionModel", back_populates="user", uselist=False)
    roles = relationship("UserRoleModel", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKeyModel", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_oauth", "oauth_provider", "oauth_id", unique=True),
    )


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

class SubscriptionModel(Base):
    __tablename__ = "subscriptions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)

    tier = Column(
        Enum(SubscriptionTier, values_callable=lambda e: [t.value for t in e]),
        default=SubscriptionTier.FREE,
        nullable=False,
    )
    status = Column(String(50), default="active", nullable=False)  # active, past_due, canceled, trialing

    # Stripe
    stripe_subscription_id = Column(String(255), nullable=True, unique=True)
    stripe_price_id = Column(String(255), nullable=True)

    # Billing period
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("UserModel", back_populates="subscription")


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class UserRoleModel(Base):
    __tablename__ = "user_roles"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # "user", "admin", "trader"

    user = relationship("UserModel", back_populates="roles")

    __table_args__ = (
        Index("ix_user_roles_unique", "user_id", "role", unique=True),
    )


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

class ApiKeyModel(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(20), nullable=False)  # "apex-abc1" for display
    label = Column(String(100), nullable=True)
    permissions = Column(JSON, default=list)  # ["read", "trade"]
    rate_limit_rpm = Column(Integer, default=60)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("UserModel", back_populates="api_keys")


# ---------------------------------------------------------------------------
# Feature access matrix (seeded per tier)
# ---------------------------------------------------------------------------

class FeatureAccessModel(Base):
    __tablename__ = "feature_access"

    id = Column(String(36), primary_key=True, default=_uuid)
    tier = Column(
        Enum(SubscriptionTier, values_callable=lambda e: [t.value for t in e]),
        nullable=False,
    )
    feature_key = Column(String(100), nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    rate_limit_daily = Column(Integer, default=-1)  # -1 = unlimited

    __table_args__ = (
        Index("ix_feature_access_tier_key", "tier", "feature_key", unique=True),
    )


# ---------------------------------------------------------------------------
# Service jobs (async job tracking)
# ---------------------------------------------------------------------------

class ServiceJobModel(Base):
    __tablename__ = "service_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_key = Column(String(100), nullable=False, index=True)
    status = Column(
        Enum(JobStatus, values_callable=lambda e: [s.value for s in e]),
        default=JobStatus.PENDING,
        nullable=False,
    )
    input_params = Column(JSON, default=dict)
    result_summary = Column(JSON, nullable=True)
    result_file_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_service_jobs_user_feature", "user_id", "feature_key"),
    )
