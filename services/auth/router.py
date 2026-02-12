"""
services/auth/router.py - Authentication API endpoints.

Mounted at /auth in the main FastAPI app.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth.service import AuthService, decode_token
from services.common.db import get_db
from services.common.schemas import SubscriptionInfo, SubscriptionTier, UserPublic
from services.common.subscription import get_features_for_tier
from api.auth import RATE_LIMITER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _rate_limit_auth(request: Request):
    """Rate limit authentication endpoints: 10 requests per minute per IP."""
    forwarded = request.headers.get("X-Forwarded-For")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    key = f"auth:{request.url.path}:{ip}"
    if not await RATE_LIMITER.is_allowed(key, max_requests=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many requests — try again later")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str  # accepts username or email
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class RefreshRequest(BaseModel):
    refresh_token: str


class MFAEnableResponse(BaseModel):
    secret: str
    provisioning_uri: str


class MFAVerifyRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)


class ApiKeyCreateRequest(BaseModel):
    label: str = ""
    permissions: list[str] = Field(default_factory=lambda: ["read"])


class ApiKeyResponse(BaseModel):
    id: str
    key_prefix: str
    label: Optional[str]
    permissions: list
    created_at: str
    raw_key: Optional[str] = None  # Only included on creation


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, dependencies=[Depends(_rate_limit_auth)])
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Try DB-backed registration first
    try:
        svc = AuthService(db)
        user, access, refresh = await svc.register(
            username=body.username,
            email=body.email,
            password=body.password,
        )
        return TokenResponse(access_token=access, refresh_token=refresh)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as exc:
        logger.debug("DB register failed, trying legacy: %s", exc)

    # Fall back to legacy in-memory user creation
    try:
        from api.auth import USER_STORE, create_access_token, create_refresh_token
        new_user = USER_STORE.create_user(
            username=body.username,
            email=body.email,
            roles=["user"],
            password=body.password,
        )
        access = create_access_token(new_user)
        refresh = create_refresh_token(new_user)
        return TokenResponse(access_token=access, refresh_token=refresh)
    except Exception as exc:
        logger.error("Legacy register failed: %s", exc)
        raise HTTPException(status_code=500, detail="Registration unavailable")


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(_rate_limit_auth)])
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Try DB-backed auth first
    try:
        svc = AuthService(db)
        result = await svc.login(body.username, body.password)
        if result:
            user, access, refresh = result
            return TokenResponse(access_token=access, refresh_token=refresh)
    except Exception as exc:
        logger.debug("DB login failed, trying legacy auth: %s", exc)

    # Fall back to legacy in-memory auth (works without PostgreSQL)
    try:
        from api.auth import login as legacy_login
        legacy_result = await legacy_login(body.username, body.password)
        if legacy_result:
            return TokenResponse(
                access_token=legacy_result["access_token"],
                refresh_token=legacy_result["refresh_token"],
            )
    except Exception as exc:
        logger.debug("Legacy login also failed: %s", exc)

    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/refresh", response_model=TokenResponse, dependencies=[Depends(_rate_limit_auth)])
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    # Try DB first
    try:
        svc = AuthService(db)
        result = await svc.refresh_tokens(body.refresh_token)
        if result:
            access, new_refresh = result
            return TokenResponse(access_token=access, refresh_token=new_refresh)
    except Exception as exc:
        logger.debug("DB refresh failed, trying legacy: %s", exc)

    # Fall back to legacy token verification + re-issue
    try:
        from api.auth import verify_token, USER_STORE, create_access_token, create_refresh_token
        token_data = verify_token(body.refresh_token, expected_token_type="refresh")
        if token_data:
            user = USER_STORE.get_user(token_data.user_id)
            if user:
                return TokenResponse(
                    access_token=create_access_token(user),
                    refresh_token=create_refresh_token(user),
                )
    except Exception as exc:
        logger.debug("Legacy refresh also failed: %s", exc)

    raise HTTPException(status_code=401, detail="Invalid or expired refresh token")


@router.post("/logout")
async def logout(request: Request):
    """Revoke the caller's access token so it cannot be reused."""
    from api.auth import TOKEN_BLACKLIST
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await TOKEN_BLACKLIST.revoke(token)
    return {"detail": "Logged out"}


@router.get("/me", response_model=UserPublic)
async def get_me(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Try DB first
    try:
        svc = AuthService(db)
        info = await svc.get_user_with_subscription(user_id)
        if info:
            user = info["user"]
            return UserPublic(
                user_id=user.id,
                username=user.username,
                email=user.email,
                roles=info["roles"],
                tier=info["tier"],
            )
    except Exception as exc:
        logger.debug("DB /me lookup failed, trying legacy: %s", exc)

    # Fall back to legacy in-memory user store
    try:
        from api.auth import USER_STORE
        legacy_user = USER_STORE.get_user(user_id)
        if legacy_user:
            return UserPublic(
                user_id=legacy_user.user_id,
                username=legacy_user.username,
                email=legacy_user.email,
                roles=legacy_user.roles,
                tier="enterprise" if "admin" in legacy_user.roles else "free",
            )
    except Exception as exc:
        logger.debug("Legacy /me also failed: %s", exc)

    raise HTTPException(status_code=404, detail="User not found")


@router.get("/me/subscription", response_model=SubscriptionInfo)
async def get_subscription(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    info = await svc.get_user_with_subscription(user_id)
    if not info:
        raise HTTPException(status_code=404, detail="User not found")
    sub = info["subscription"]
    tier = info["tier"]
    return SubscriptionInfo(
        tier=tier,
        status=sub.status if sub else "active",
        current_period_start=sub.current_period_start if sub else None,
        current_period_end=sub.current_period_end if sub else None,
        stripe_subscription_id=sub.stripe_subscription_id if sub else None,
        features=get_features_for_tier(tier),
    )


# --- MFA ---

@router.post("/mfa/enable", response_model=MFAEnableResponse)
async def enable_mfa(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    try:
        secret = await svc.enable_mfa(user_id)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    if not secret:
        raise HTTPException(status_code=404, detail="User not found")

    import pyotp
    user = await svc.get_user(user_id)
    uri = pyotp.TOTP(secret).provisioning_uri(
        name=user.email or user.username,
        issuer_name="Apex Trading",
    )
    return MFAEnableResponse(secret=secret, provisioning_uri=uri)


@router.post("/mfa/verify")
async def verify_mfa(body: MFAVerifyRequest, request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    if not await svc.verify_mfa(user_id, body.code):
        raise HTTPException(status_code=403, detail="Invalid MFA code")
    return {"detail": "MFA verified"}


# --- API Keys ---

@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    body: ApiKeyCreateRequest, request: Request, db: AsyncSession = Depends(get_db),
):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    raw_key, model = await svc.create_api_key(
        user_id=user_id,
        label=body.label,
        permissions=body.permissions,
    )
    return ApiKeyResponse(
        id=model.id,
        key_prefix=model.key_prefix,
        label=model.label,
        permissions=model.permissions,
        created_at=model.created_at.isoformat(),
        raw_key=raw_key,
    )


@router.get("/api-keys")
async def list_api_keys(request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    keys = await svc.list_api_keys(user_id)
    return [
        ApiKeyResponse(
            id=k.id,
            key_prefix=k.key_prefix,
            label=k.label,
            permissions=k.permissions or [],
            created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    svc = AuthService(db)
    if not await svc.revoke_api_key(user_id, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"detail": "API key revoked"}


# ---------------------------------------------------------------------------
# Billing (Stripe)
# ---------------------------------------------------------------------------

class CheckoutRequest(BaseModel):
    tier: SubscriptionTier


@router.post("/billing/checkout")
async def create_checkout(body: CheckoutRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Create a Stripe checkout session for the given tier. Returns checkout URL."""
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from services.auth.stripe_billing import StripeBilling, is_available
    if not is_available():
        raise HTTPException(status_code=503, detail="Billing not configured")
    billing = StripeBilling(db)
    try:
        url = await billing.create_checkout_session(user_id, body.tier)
        return {"checkout_url": url}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/billing/portal")
async def create_portal(request: Request, db: AsyncSession = Depends(get_db)):
    """Create a Stripe customer portal session. Returns portal URL."""
    user_id = _get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from services.auth.stripe_billing import StripeBilling, is_available
    if not is_available():
        raise HTTPException(status_code=503, detail="Billing not configured")
    billing = StripeBilling(db)
    try:
        url = await billing.create_portal_session(user_id)
        return {"portal_url": url}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/billing/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Stripe webhook endpoint. No auth — verified via Stripe signature."""
    payload = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    from services.auth.stripe_billing import StripeBilling, is_available
    if not is_available():
        raise HTTPException(status_code=503, detail="Billing not configured")
    billing = StripeBilling(db)
    try:
        result = await billing.handle_webhook(payload, sig)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Helper: extract user_id from request (set by middleware)
# ---------------------------------------------------------------------------

def _get_user_id(request: Request) -> Optional[str]:
    """Get authenticated user_id from request state or Authorization header."""
    # Check if middleware already set it
    user = getattr(request.state, "user", None)
    if user:
        return getattr(user, "user_id", None) or getattr(user, "id", None)

    # Fall back to decoding the token directly
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        data = decode_token(auth_header[7:])
        if data:
            return data.get("sub")

    # Check API key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # For API key auth, we'd need a DB lookup — skip here,
        # the middleware should have handled it
        pass

    return None
