"""
services/auth/oauth_providers.py - OAuth2 provider integrations (Google, GitHub).

Uses authlib for standards-compliant OAuth2 flows.
Falls back gracefully when authlib is not installed.
"""

import logging
import os
import uuid
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth.models import SubscriptionModel, UserModel, UserRoleModel
from services.auth.service import create_access_token, create_refresh_token
from services.common.schemas import SubscriptionTier

logger = logging.getLogger(__name__)

_AUTHLIB_AVAILABLE = False
try:
    from authlib.integrations.starlette_client import OAuth
    _AUTHLIB_AVAILABLE = True
except ImportError:
    logger.warning("authlib not installed - OAuth disabled")


# ---------------------------------------------------------------------------
# Provider config (from env vars)
# ---------------------------------------------------------------------------

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", "http://localhost:8000")


def is_available() -> bool:
    return _AUTHLIB_AVAILABLE


def create_oauth_client() -> Optional["OAuth"]:
    """Create and configure the authlib OAuth client."""
    if not _AUTHLIB_AVAILABLE:
        return None

    oauth = OAuth()

    if GOOGLE_CLIENT_ID:
        oauth.register(
            name="google",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    if GITHUB_CLIENT_ID:
        oauth.register(
            name="github",
            client_id=GITHUB_CLIENT_ID,
            client_secret=GITHUB_CLIENT_SECRET,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )

    return oauth


# Singleton
_oauth = create_oauth_client()


def get_oauth() -> Optional["OAuth"]:
    return _oauth


class OAuthService:
    """Handle OAuth2 login/registration flows."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create_user(
        self, provider: str, oauth_id: str, email: str, username: Optional[str] = None,
    ) -> Tuple[UserModel, str, str]:
        """Find existing OAuth user or create a new one. Returns (user, access, refresh)."""
        # Check for existing OAuth link
        result = await self.db.execute(
            select(UserModel).where(
                UserModel.oauth_provider == provider,
                UserModel.oauth_id == oauth_id,
            )
        )
        user = result.scalar_one_or_none()

        if user:
            # Existing user — issue tokens
            roles = await self._get_roles(user.id)
            access = create_access_token(user.id, user.username, roles)
            refresh = create_refresh_token(user.id)
            return user, access, refresh

        # Check if email already exists (link OAuth to existing account)
        result = await self.db.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        if user:
            # Link OAuth to existing account
            user.oauth_provider = provider
            user.oauth_id = oauth_id
        else:
            # Create new user
            display_name = username or email.split("@")[0]
            # Ensure unique username
            base_name = display_name
            counter = 1
            while True:
                check = await self.db.execute(
                    select(UserModel).where(UserModel.username == display_name)
                )
                if not check.scalar_one_or_none():
                    break
                display_name = f"{base_name}{counter}"
                counter += 1

            user = UserModel(
                id=str(uuid.uuid4()),
                username=display_name,
                email=email,
                oauth_provider=provider,
                oauth_id=oauth_id,
            )
            self.db.add(user)
            self.db.add(UserRoleModel(user_id=user.id, role="user"))
            self.db.add(SubscriptionModel(
                user_id=user.id,
                tier=SubscriptionTier.FREE,
                status="active",
            ))

        await self.db.flush()
        roles = await self._get_roles(user.id)
        access = create_access_token(user.id, user.username, roles)
        refresh = create_refresh_token(user.id)
        return user, access, refresh

    async def _get_roles(self, user_id: str):
        result = await self.db.execute(
            select(UserRoleModel.role).where(UserRoleModel.user_id == user_id)
        )
        return [r for (r,) in result.all()] or ["user"]
