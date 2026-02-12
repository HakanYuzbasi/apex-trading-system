"""
services/auth/service.py - Core authentication service.

Handles user registration, login (bcrypt), JWT access+refresh tokens,
MFA (TOTP), and API key management.  Bridges to the in-memory
api/auth.py layer for backward compatibility.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth.models import (
    ApiKeyModel,
    SubscriptionModel,
    UserModel,
    UserRoleModel,
)
from services.common.schemas import SubscriptionTier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional bcrypt / jose / pyotp imports (graceful fallback)
# ---------------------------------------------------------------------------

_BCRYPT_AVAILABLE = False
try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    logger.warning("bcrypt not installed - using SHA-256 password hashing (dev only)")

_JOSE_AVAILABLE = False
try:
    from jose import JWTError, jwt as jose_jwt
    _JOSE_AVAILABLE = True
except ImportError:
    logger.warning("python-jose not installed - JWT disabled")

_PYOTP_AVAILABLE = False
try:
    import pyotp
    _PYOTP_AVAILABLE = True
except ImportError:
    logger.warning("pyotp not installed - MFA disabled")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os

def _get_shared_secret() -> str:
    """Share the same secret key with api/auth.py to ensure token compatibility."""
    try:
        from api.auth import AUTH_CONFIG
        return AUTH_CONFIG.secret_key
    except Exception:
        return os.getenv("APEX_SECRET_KEY", secrets.token_hex(32))

SECRET_KEY = _get_shared_secret()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("APEX_TOKEN_EXPIRE_MIN", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    if _BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    if _BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except (ValueError, TypeError):
            return False
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, username: str, roles: List[str]) -> str:
    if not _JOSE_AVAILABLE:
        return f"mock-token-{user_id}"
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "roles": roles,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    return jose_jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    if not _JOSE_AVAILABLE:
        return f"mock-refresh-{user_id}"
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    }
    return jose_jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if _JOSE_AVAILABLE:
        try:
            return jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            return None

    # Fall back to PyJWT (used by api/auth.py)
    try:
        import jwt
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        pass

    # Mock tokens (dev fallback)
    if token.startswith("mock-token-") or token.startswith("mock-refresh-"):
        uid = token.split("-", 2)[-1]
        return {"sub": uid, "type": "access", "roles": ["user"]}

    return None


# ---------------------------------------------------------------------------
# Auth Service
# ---------------------------------------------------------------------------

class AuthService:
    """High-level authentication operations backed by PostgreSQL."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # --- Registration ---

    async def register(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
    ) -> Tuple[UserModel, str, str]:
        """Register a new user. Returns (user, access_token, refresh_token)."""
        # Check uniqueness
        existing = await self.db.execute(
            select(UserModel).where(
                (UserModel.username == username) | (UserModel.email == email)
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Username or email already taken")

        user = UserModel(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=hash_password(password),
        )
        self.db.add(user)

        # Roles
        for role_name in (roles or ["user"]):
            self.db.add(UserRoleModel(user_id=user.id, role=role_name))

        # Free subscription
        self.db.add(SubscriptionModel(
            user_id=user.id,
            tier=SubscriptionTier.FREE,
            status="active",
        ))

        await self.db.flush()

        access = create_access_token(user.id, user.username, roles or ["user"])
        refresh = create_refresh_token(user.id)
        return user, access, refresh

    # --- Login ---

    async def login(self, username: str, password: str) -> Optional[Tuple[UserModel, str, str]]:
        """Authenticate with username/email + password."""
        result = await self.db.execute(
            select(UserModel).where(
                (UserModel.username == username) | (UserModel.email == username)
            )
        )
        user = result.scalar_one_or_none()
        if not user or not user.password_hash:
            return None
        if not verify_password(password, user.password_hash):
            return None
        if not user.is_active:
            return None

        user.last_login_at = datetime.utcnow()
        roles = await self._get_roles(user.id)
        access = create_access_token(user.id, user.username, roles)
        refresh = create_refresh_token(user.id)
        return user, access, refresh

    # --- Token refresh ---

    async def refresh_tokens(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Exchange a refresh token for new access + refresh tokens."""
        data = decode_token(refresh_token)
        if not data or data.get("type") != "refresh":
            return None
        user_id = data["sub"]
        user = await self.db.get(UserModel, user_id)
        if not user or not user.is_active:
            return None
        roles = await self._get_roles(user.id)
        return (
            create_access_token(user.id, user.username, roles),
            create_refresh_token(user.id),
        )

    # --- MFA ---

    async def enable_mfa(self, user_id: str) -> Optional[str]:
        """Generate a TOTP secret and enable MFA. Returns the secret (for QR code)."""
        if not _PYOTP_AVAILABLE:
            raise RuntimeError("pyotp not installed")
        user = await self.db.get(UserModel, user_id)
        if not user:
            return None
        secret = pyotp.random_base32()
        user.mfa_secret = secret
        user.mfa_enabled = True
        return secret

    async def verify_mfa(self, user_id: str, code: str) -> bool:
        """Verify a TOTP code."""
        if not _PYOTP_AVAILABLE:
            return False
        user = await self.db.get(UserModel, user_id)
        if not user or not user.mfa_secret:
            return False
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(code, valid_window=1)

    # --- API keys ---

    async def create_api_key(
        self, user_id: str, label: str = "", permissions: Optional[List[str]] = None,
    ) -> Tuple[str, ApiKeyModel]:
        """Create a new API key. Returns (raw_key, model)."""
        raw_key = f"apex-{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]

        model = ApiKeyModel(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            label=label,
            permissions=permissions or ["read"],
        )
        self.db.add(model)
        await self.db.flush()
        return raw_key, model

    async def verify_api_key(self, raw_key: str) -> Optional[UserModel]:
        """Verify an API key and return the owning user."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        result = await self.db.execute(
            select(ApiKeyModel).where(
                ApiKeyModel.key_hash == key_hash,
                ApiKeyModel.is_active == True,
            )
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            return None
        api_key.last_used_at = datetime.utcnow()
        user = await self.db.get(UserModel, api_key.user_id)
        return user if user and user.is_active else None

    async def list_api_keys(self, user_id: str) -> List[ApiKeyModel]:
        result = await self.db.execute(
            select(ApiKeyModel).where(ApiKeyModel.user_id == user_id)
        )
        return list(result.scalars().all())

    async def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        result = await self.db.execute(
            select(ApiKeyModel).where(
                ApiKeyModel.id == key_id,
                ApiKeyModel.user_id == user_id,
            )
        )
        key = result.scalar_one_or_none()
        if not key:
            return False
        key.is_active = False
        return True

    # --- User management ---

    async def get_user(self, user_id: str) -> Optional[UserModel]:
        return await self.db.get(UserModel, user_id)

    async def get_user_with_subscription(self, user_id: str) -> Optional[Dict]:
        """Get user info with subscription tier."""
        user = await self.db.get(UserModel, user_id)
        if not user:
            return None
        sub = await self.db.execute(
            select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
        )
        subscription = sub.scalar_one_or_none()
        roles = await self._get_roles(user_id)
        return {
            "user": user,
            "roles": roles,
            "tier": subscription.tier if subscription else SubscriptionTier.FREE,
            "subscription": subscription,
        }

    # --- Helpers ---

    async def _get_roles(self, user_id: str) -> List[str]:
        result = await self.db.execute(
            select(UserRoleModel.role).where(UserRoleModel.user_id == user_id)
        )
        return [r for (r,) in result.all()] or ["user"]
