"""World-Class Billing and Subscription Management - Enterprise Monetization.

Handles subscription lifecycles, tiered feature access, usage-based billing,
and secure payment processing for the ApexTrader SaaS platform.

Features:
    - Multi-tier subscription plans (Free, Pro, Institutional)
    - Granular feature flags and access control
    - Usage tracking and metered billing (e.g., API calls, trade volume)
    - Invoice generation and automated billing cycles
    - Secure payment gateway integration (Stripe-ready)
    - Coupon and promotion management
    - Real-time revenue analytics and forecasting

Architecture:
    - Event-driven billing triggers with RabbitMQ
    - High-precision financial data in PostgreSQL
    - Feature flag management with Redis
    - Integration with external payment providers via secure webhooks

Author: ApexTrader Development Team
Version: 1.0.0
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import BillingService
    from .schemas import (
        SubscriptionPlan,
        UserSubscription,
        Invoice,
        UsageReport
    )

__version__ = "1.0.0"
__author__ = "ApexTrader Development Team"
__all__ = [
    "BillingService",
    "SubscriptionPlan",
    "UserSubscription",
    "Invoice",
    "UsageReport",
]
