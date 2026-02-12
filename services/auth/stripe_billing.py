"""
services/auth/stripe_billing.py - Stripe integration for subscription billing.

Handles checkout sessions, customer portal, and webhook processing.
Falls back gracefully when Stripe is not configured.
"""

import logging
import os
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth.models import SubscriptionModel, UserModel
from services.common.schemas import SubscriptionTier

logger = logging.getLogger(__name__)

_STRIPE_AVAILABLE = False
try:
    import stripe
    _STRIPE_AVAILABLE = True
except ImportError:
    logger.warning("stripe package not installed - billing disabled")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:3000/settings?session_id={CHECKOUT_SESSION_ID}")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "http://localhost:3000/settings")

# Map tier names to Stripe price IDs (set via env)
TIER_PRICE_MAP = {
    SubscriptionTier.BASIC: os.getenv("STRIPE_PRICE_BASIC", ""),
    SubscriptionTier.PRO: os.getenv("STRIPE_PRICE_PRO", ""),
    SubscriptionTier.ENTERPRISE: os.getenv("STRIPE_PRICE_ENTERPRISE", ""),
}

if _STRIPE_AVAILABLE and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def is_available() -> bool:
    return _STRIPE_AVAILABLE and bool(STRIPE_SECRET_KEY)


class StripeBilling:
    """Stripe billing operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_checkout_session(
        self, user_id: str, target_tier: SubscriptionTier,
    ) -> Optional[str]:
        """Create a Stripe checkout session. Returns the session URL."""
        if not is_available():
            raise RuntimeError("Stripe not configured")

        price_id = TIER_PRICE_MAP.get(target_tier)
        if not price_id:
            raise ValueError(f"No Stripe price configured for tier: {target_tier.value}")

        user = await self.db.get(UserModel, user_id)
        if not user:
            raise ValueError("User not found")

        # Get or create Stripe customer
        if not user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                metadata={"apex_user_id": user.id},
            )
            user.stripe_customer_id = customer.id
            await self.db.flush()

        session = stripe.checkout.Session.create(
            customer=user.stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=STRIPE_SUCCESS_URL,
            cancel_url=STRIPE_CANCEL_URL,
            metadata={"apex_user_id": user.id, "target_tier": target_tier.value},
        )
        return session.url

    async def create_portal_session(self, user_id: str) -> Optional[str]:
        """Create a Stripe customer portal session. Returns the URL."""
        if not is_available():
            raise RuntimeError("Stripe not configured")

        user = await self.db.get(UserModel, user_id)
        if not user or not user.stripe_customer_id:
            raise ValueError("No billing account found")

        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=STRIPE_CANCEL_URL,
        )
        return session.url

    async def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Process a Stripe webhook event. Returns summary dict."""
        if not is_available():
            raise RuntimeError("Stripe not configured")

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET,
            )
        except stripe.error.SignatureVerificationError:
            raise ValueError("Invalid Stripe signature")

        event_type = event["type"]
        data = event["data"]["object"]
        result = {"event_type": event_type, "handled": False}

        if event_type == "checkout.session.completed":
            await self._handle_checkout_completed(data)
            result["handled"] = True
        elif event_type == "customer.subscription.updated":
            await self._handle_subscription_updated(data)
            result["handled"] = True
        elif event_type == "customer.subscription.deleted":
            await self._handle_subscription_deleted(data)
            result["handled"] = True
        elif event_type == "invoice.payment_failed":
            await self._handle_payment_failed(data)
            result["handled"] = True

        return result

    # --- Webhook handlers ---

    async def _handle_checkout_completed(self, session_data: dict):
        user_id = session_data.get("metadata", {}).get("apex_user_id")
        target_tier = session_data.get("metadata", {}).get("target_tier", "basic")
        stripe_sub_id = session_data.get("subscription")

        if not user_id:
            logger.warning("Checkout session missing apex_user_id metadata")
            return

        result = await self.db.execute(
            select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
        )
        sub = result.scalar_one_or_none()
        if sub:
            sub.tier = SubscriptionTier(target_tier)
            sub.status = "active"
            sub.stripe_subscription_id = stripe_sub_id
        else:
            self.db.add(SubscriptionModel(
                user_id=user_id,
                tier=SubscriptionTier(target_tier),
                status="active",
                stripe_subscription_id=stripe_sub_id,
            ))
        logger.info("Subscription activated: user=%s tier=%s", user_id, target_tier)

    async def _handle_subscription_updated(self, sub_data: dict):
        stripe_sub_id = sub_data.get("id")
        status = sub_data.get("status", "active")

        result = await self.db.execute(
            select(SubscriptionModel).where(
                SubscriptionModel.stripe_subscription_id == stripe_sub_id
            )
        )
        sub = result.scalar_one_or_none()
        if sub:
            sub.status = status
            period = sub_data.get("current_period_end")
            if period:
                from datetime import datetime
                sub.current_period_end = datetime.fromtimestamp(period)
            sub.cancel_at_period_end = sub_data.get("cancel_at_period_end", False)

    async def _handle_subscription_deleted(self, sub_data: dict):
        stripe_sub_id = sub_data.get("id")
        result = await self.db.execute(
            select(SubscriptionModel).where(
                SubscriptionModel.stripe_subscription_id == stripe_sub_id
            )
        )
        sub = result.scalar_one_or_none()
        if sub:
            sub.tier = SubscriptionTier.FREE
            sub.status = "canceled"
            sub.stripe_subscription_id = None
            logger.info("Subscription canceled: user=%s", sub.user_id)

    async def _handle_payment_failed(self, invoice_data: dict):
        customer_id = invoice_data.get("customer")
        if customer_id:
            result = await self.db.execute(
                select(UserModel).where(UserModel.stripe_customer_id == customer_id)
            )
            user = result.scalar_one_or_none()
            if user:
                logger.warning("Payment failed for user=%s", user.id)
                # Update subscription status
                sub_result = await self.db.execute(
                    select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
                )
                sub = sub_result.scalar_one_or_none()
                if sub:
                    sub.status = "past_due"
