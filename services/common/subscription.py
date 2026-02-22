"""
services/common/subscription.py - Subscription tier gating for SaaS features.

Provides FastAPI dependencies that check whether the current user's subscription
tier grants access to a feature, and whether they are within their daily rate limit.

Usage:
    from services.common.subscription import require_feature

    @router.post("/validate")
    async def validate(
        user=Depends(require_feature("backtest_validator")),
    ):
        ...
"""

import logging
from typing import Optional, Sequence

from fastapi import Depends, HTTPException, Request

from services.common.schemas import (
    FEATURE_MATRIX,
    TIER_RANK,
    SubscriptionTier,
)
from services.common.redis_client import rate_check, rate_get_remaining

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to read user tier (bridged from the auth layer)
# ---------------------------------------------------------------------------

def _get_user_tier(user, fallback_roles: Optional[Sequence[str]] = None) -> SubscriptionTier:
    """Extract subscription tier from user object.

    Works with both the existing api.auth.User dataclass and future
    services.auth ORM models by checking for a ``tier`` attribute,
    falling back to FREE.
    """
    tier = getattr(user, "tier", None)
    if tier is None:
        # Legacy User dataclass doesn't have a tier field yet.
        # Admin users get enterprise-level access.
        roles: Sequence[str]
        try:
            roles = getattr(user, "roles", [])
        except Exception:
            roles = fallback_roles or []
        if "admin" in roles:
            return SubscriptionTier.ENTERPRISE
        return SubscriptionTier.FREE
    if isinstance(tier, SubscriptionTier):
        return tier
    return SubscriptionTier(tier)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def require_tier(min_tier: SubscriptionTier):
    """Dependency that requires the user to be on *min_tier* or higher."""

    async def _check(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")

        request_roles = getattr(request.state, "roles", None)
        user_tier = _get_user_tier(user, fallback_roles=request_roles)
        if TIER_RANK.get(user_tier, 0) < TIER_RANK[min_tier]:
            raise HTTPException(
                status_code=403,
                detail=f"Subscription tier '{min_tier.value}' or higher required (current: {user_tier.value})",
            )
        return user

    return _check


def require_feature(feature_key: str):
    """Dependency that checks tier access AND daily rate limit for a feature.

    Expects request.state.user to be set by SaaSAuthMiddleware.
    Returns the authenticated user if allowed; raises 403 or 429 otherwise.
    """

    async def _check(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")

        user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")

        # --- tier check ---
        feature = FEATURE_MATRIX.get(feature_key)
        if feature is None:
            return user

        request_roles = getattr(request.state, "roles", None)
        user_tier = _get_user_tier(user, fallback_roles=request_roles)
        if TIER_RANK.get(user_tier, 0) < TIER_RANK[feature.min_tier]:
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature_key}' requires '{feature.min_tier.value}' tier or higher",
            )

        # --- rate limit check ---
        daily_limit = feature.daily_limits.get(user_tier, 0)
        if daily_limit == 0:
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature_key}' not available on '{user_tier.value}' tier",
            )

        allowed = await rate_check(user_id, feature_key, daily_limit)
        if not allowed:
            remaining = await rate_get_remaining(user_id, feature_key, daily_limit)
            raise HTTPException(
                status_code=429,
                detail=f"Daily rate limit reached for '{feature_key}' ({daily_limit}/day)",
                headers={
                    "X-RateLimit-Feature": feature_key,
                    "X-RateLimit-Limit": str(daily_limit),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

        return user

    return _check


# ---------------------------------------------------------------------------
# Utility: get features available for a tier
# ---------------------------------------------------------------------------

def get_features_for_tier(tier: SubscriptionTier) -> dict[str, int]:
    """Return {feature_key: daily_limit} for the given tier."""
    result = {}
    for key, fl in FEATURE_MATRIX.items():
        if TIER_RANK.get(tier, 0) >= TIER_RANK[fl.min_tier]:
            limit = fl.daily_limits.get(tier, 0)
            if limit != 0:
                result[key] = limit
    return result
