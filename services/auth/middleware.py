"""
services/auth/middleware.py - Auth middleware bridging services/auth/ to api/auth.py.

This middleware:
1. Intercepts requests and resolves the user from JWT or API key
2. Attaches user + tier info to request.state for downstream dependencies
3. Falls back to the existing api/auth.py in-memory auth when DB is unavailable
"""

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from core.request_context import set_user_id as set_request_context_user_id, user_id_var
from services.auth.service import AuthService, decode_token
from services.common.schemas import SubscriptionTier

logger = logging.getLogger(__name__)


class SaaSAuthMiddleware(BaseHTTPMiddleware):
    """Resolve authenticated user and attach to request.state.

    Checks (in order):
    1. Bearer JWT token → decode → load user from PostgreSQL
    2. X-API-Key header → verify hash against PostgreSQL
    3. Fall back to api/auth.py in-memory UserStore

    Sets on ``request.state``:
    - ``user``: User object (ORM or dataclass)
    - ``user_id``: str
    - ``tier``: SubscriptionTier
    - ``roles``: list[str]
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip auth for public endpoints
        if self._is_public(request.url.path):
            return await call_next(request)

        user = None
        tier = SubscriptionTier.FREE
        roles = ["user"]
        user_id = None

        try:
            user, tier, roles, user_id = await self._resolve_from_db(request)
        except Exception as e:
            logger.debug("DB auth resolution failed: %s", e)

        # Fallback to legacy in-memory auth
        if user is None:
            try:
                user, tier, roles, user_id = await self._resolve_from_legacy(request)
            except Exception as e:
                logger.debug("Legacy auth resolution failed: %s", e)

        # Attach to request state
        request.state.user = user
        request.state.user_id = user_id
        request.state.tier = tier
        request.state.roles = roles

        user_token = None
        if user_id is not None:
            user_token = set_request_context_user_id(str(user_id))
        try:
            return await call_next(request)
        finally:
            if user_token is not None:
                user_id_var.reset(user_token)

    async def _resolve_from_db(self, request: Request):
        """Try to resolve user from PostgreSQL via services/auth."""
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key")

        # Skip DB access when no auth material is present.
        if not auth_header.startswith("Bearer ") and not api_key:
            return None, SubscriptionTier.FREE, ["user"], None

        token_user_id: Optional[str] = None
        if auth_header.startswith("Bearer "):
            token_data = decode_token(auth_header[7:])
            # Access endpoints must reject refresh tokens.
            if not token_data or token_data.get("type") != "access":
                return None, SubscriptionTier.FREE, ["user"], None
            token_user_id = token_data.get("sub")

        from services.common.db import get_session_factory

        factory = get_session_factory()
        async with factory() as db:
            svc = AuthService(db)

            if token_user_id:
                info = await svc.get_user_with_subscription(token_user_id)
                if info:
                    user = info["user"]
                    return user, info["tier"], info["roles"], user.id

            if api_key:
                user = await svc.verify_api_key(api_key)
                if user:
                    info = await svc.get_user_with_subscription(user.id)
                    if info:
                        return user, info["tier"], info["roles"], user.id

        return None, SubscriptionTier.FREE, ["user"], None

    async def _resolve_from_legacy(self, request: Request):
        """Fall back to api/auth.py in-memory auth."""
        from api.auth import get_current_user, security, api_key_header, AUTH_CONFIG

        if not AUTH_CONFIG.enabled:
            # Auth disabled — return a least-privilege default user.
            from api.auth import User
            user = User(
                user_id="default",
                username="default",
                roles=["user"],
                permissions=["read"],
            )
            return user, SubscriptionTier.FREE, ["user"], "default"

        credentials = await security(request)
        api_key = request.headers.get("X-API-Key")
        user = await get_current_user(credentials=credentials, api_key=api_key, request=request)
        if user:
            roles = getattr(user, "roles", ["user"])
            tier = SubscriptionTier.ENTERPRISE if "admin" in roles else SubscriptionTier.FREE
            uid = getattr(user, "user_id", None)
            return user, tier, roles, uid

        return None, SubscriptionTier.FREE, ["user"], None

    def _is_public(self, path: str) -> bool:
        """Check if the path is a public (no-auth) endpoint."""
        public_prefixes = [
            "/auth/register",
            "/auth/login",
            "/auth/refresh",
            "/auth/oauth",
            "/auth/billing/webhook",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/metrics",
        ]
        return any(path.startswith(p) for p in public_prefixes)
