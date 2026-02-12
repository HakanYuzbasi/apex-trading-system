"""World-Class Pydantic Schemas for Billing and Subscription Management.

Provides comprehensive data validation for subscription plans, user tiers,
usage tracking, and enterprise invoicing.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, condecimal, conint, constr


class PlanTier(str, Enum):
    """Available subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    INSTITUTIONAL = "institutional"
    ENTERPRISE = "enterprise"


class BillingCycle(str, Enum):
    """Supported billing frequencies."""
    MONTHLY = "monthly"
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class SubscriptionPlan(BaseModel):
    """Definition of a subscription plan and its features."""
    
    plan_id: str
    name: str
    tier: PlanTier
    price: Decimal = Field(..., ge=0)
    currency: str = "USD"
    interval: BillingCycle
    
    # Feature Flags
    max_strategies: conint(ge=1)
    max_api_calls_per_month: conint(ge=0)
    has_social_trading: bool = False
    has_advanced_analytics: bool = True
    has_priority_support: bool = False
    
    is_active: bool = True


class UserSubscription(BaseModel):
    """Active subscription record for a user."""
    
    subscription_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    plan_id: str
    tier: PlanTier
    
    status: str = "active"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    current_period_end: datetime
    cancel_at_period_end: bool = False
    
    # Stripe integration fields
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None


class UsageMetric(BaseModel):
    """Metered usage tracking for a user."""
    
    user_id: UUID
    metric_name: str # e.g., "api_calls", "trade_volume"
    current_value: Decimal = Field(default=Decimal("0"))
    limit: Optional[Decimal] = None
    
    reset_at: datetime


class Invoice(BaseModel):
    """Invoice record for historical billing."""
    
    invoice_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    amount: Decimal
    currency: str = "USD"
    
    status: str = "paid" # paid, pending, failed, void
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    paid_at: Optional[datetime] = None
    
    invoice_pdf_url: Optional[str] = None
