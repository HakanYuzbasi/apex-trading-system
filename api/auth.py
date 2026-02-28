"""
api/auth.py - JWT Authentication and Rate Limiting

Provides security for the APEX Trading API:
- JWT token-based authentication
- API key authentication (for service-to-service)
- Rate limiting to prevent abuse
- Role-based access control

Usage:
    from api.auth import require_auth, require_role, rate_limit

    @app.get("/portfolio")
    @require_auth
    @rate_limit(requests=100, window=60)
    async def get_portfolio(user: User = Depends(get_current_user)):
        ...
"""

import os
import secrets
import hashlib
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any, Deque
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict, deque
import asyncio
from config import ApexConfig

from fastapi import HTTPException, Depends, Request, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy import select

logger = logging.getLogger(__name__)

# Alert aggregator for reducing auth warning noise
from core.alert_aggregator import get_alert_aggregator
alert_agg = get_alert_aggregator(logger)
from services.auth.models import ApiKeyModel, UserModel, UserRoleModel
from services.common.db import db_session
from services.common.redis_client import get_redis


def _runtime_env() -> str:
    """Return normalized runtime environment."""
    return (
        os.getenv("APEX_ENV")
        or os.getenv("APEX_ENVIRONMENT")
        or "development"
    ).strip().lower()


def _is_development_env() -> bool:
    """Return True when mock auth behavior is allowed."""
    return _runtime_env() == "development"

# Try to import JWT library
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available. Install with: pip install PyJWT")

# Optional bcrypt support (falls back to SHA-256)
try:
    import bcrypt as _bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not available — install with: pip install bcrypt")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (preferred) or SHA-256 fallback."""
    if _BCRYPT_AVAILABLE:
        return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    if not hashed_password:
        return False
    if hashed_password.startswith("$2a$") or hashed_password.startswith("$2b$") or hashed_password.startswith("$2y$"):
        if not _BCRYPT_AVAILABLE:
            return False
        try:
            return _bcrypt.checkpw(password.encode(), hashed_password.encode())
        except ValueError:
            return False
    return secrets.compare_digest(hashlib.sha256(password.encode()).hexdigest(), hashed_password)


# Configuration
@dataclass
class AuthConfig:
    """Authentication configuration."""
    secret_key: str = os.getenv("APEX_SECRET_KEY", secrets.token_hex(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = ApexConfig.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES
    refresh_token_expire_days: int = 7
    api_key_header: str = "X-API-Key"
    enabled: bool = os.getenv("APEX_AUTH_ENABLED", "true").lower() == "true"
    allowed_origins: List[str] = None

    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = [
                "http://localhost:3000",
                "http://127.0.0.1:3000"
            ]


# Global configuration
AUTH_CONFIG = AuthConfig()


@dataclass
class User:
    """User representation."""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    api_key: Optional[str] = None
    created_at: Optional[datetime] = None
    tier: Optional[Any] = None  # SubscriptionTier when set by SaaS middleware

    def __post_init__(self):
        if self.roles is None:
            self.roles = ["user"]
        if self.permissions is None:
            self.permissions = ["read"]

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or "admin" in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions or "admin" in self.roles

    def to_dict(self) -> Dict:
        """Convert user to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "api_key": self.api_key,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "tier": self.tier
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """Create user from dictionary."""
        created_at = data.get("created_at")
        if created_at:
            # Robust parsing: strip erroneous trailing Z when timezone offset already present
            created_at_str = str(created_at).replace("+00:00Z", "+00:00").rstrip("Z")
            created_at = datetime.fromisoformat(created_at_str)
        
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data.get("email"),
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            api_key=data.get("api_key"),
            created_at=created_at,
            tier=data.get("tier")
        )


@dataclass
class TokenData:
    """JWT token payload data."""
    user_id: str
    username: str
    roles: List[str]
    exp: datetime
    iat: datetime
    token_type: str = "access"


class DatabaseUserStore:
    """Database-backed user and API key access for auth dependencies."""

    @staticmethod
    def _legacy_users() -> List[Dict[str, Any]]:
        """Load legacy file-backed users for compatibility fallback paths."""
        users_file = ApexConfig.DATA_DIR / "users.json"
        try:
            if not users_file.exists():
                return []
            payload = json.loads(users_file.read_text(encoding="utf-8"))
            users = payload.get("users", []) if isinstance(payload, dict) else []
            return users if isinstance(users, list) else []
        except Exception as exc:
            logger.debug("Legacy users fallback unavailable: %s", exc)
            return []

    @staticmethod
    def _legacy_user_to_model(entry: Dict[str, Any]) -> User:
        """Convert legacy user record into auth User object."""
        created_at = entry.get("created_at")
        created_dt: Optional[datetime] = None
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at)
            except ValueError:
                created_dt = None

        roles_raw = entry.get("roles", [])
        roles = roles_raw if isinstance(roles_raw, list) and roles_raw else ["user"]
        permissions_raw = entry.get("permissions", [])
        if isinstance(permissions_raw, list) and permissions_raw:
            permissions = permissions_raw
        else:
            permissions = ["read", "write"] if "admin" not in roles else ["read", "write", "trade", "admin"]

        return User(
            user_id=str(entry.get("user_id") or entry.get("username") or "unknown"),
            username=str(entry.get("username") or "unknown"),
            email=entry.get("email"),
            roles=roles,
            permissions=permissions,
            api_key=entry.get("api_key"),
            created_at=created_dt,
            tier=entry.get("tier"),
        )

    @staticmethod
    async def _get_roles(user_id: str) -> List[str]:
        async with db_session() as session:
            result = await session.execute(
                select(UserRoleModel.role).where(UserRoleModel.user_id == user_id)
            )
            roles = [role for (role,) in result.all()]
            return roles or ["user"]

    @staticmethod
    async def get_user(user_id: str) -> Optional[User]:
        """Load a user by ID from the database."""
        try:
            async with db_session() as session:
                model = await session.get(UserModel, user_id)
                if model is not None and model.is_active:
                    roles = await DatabaseUserStore._get_roles(user_id)
                    return User(
                        user_id=model.id,
                        username=model.username,
                        email=model.email,
                        roles=roles,
                        permissions=["read", "write"] if "admin" not in roles else ["read", "write", "trade", "admin"],
                        created_at=model.created_at,
                    )
        except Exception as exc:
            logger.debug("DB get_user fallback for %s: %s", user_id, exc)

        for entry in DatabaseUserStore._legacy_users():
            if str(entry.get("user_id")) == user_id:
                return DatabaseUserStore._legacy_user_to_model(entry)
        return None

    @staticmethod
    async def get_user_by_api_key(api_key: str) -> Optional[User]:
        """Load a user from a raw API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        try:
            async with db_session() as session:
                key_result = await session.execute(
                    select(ApiKeyModel).where(
                        ApiKeyModel.key_hash == key_hash,
                        ApiKeyModel.is_active,
                    )
                )
                key = key_result.scalar_one_or_none()
                if key is not None:
                    user_model = await session.get(UserModel, key.user_id)
                    if user_model is not None and user_model.is_active:
                        key.last_used_at = datetime.utcnow()
                        roles = await DatabaseUserStore._get_roles(user_model.id)
                        return User(
                            user_id=user_model.id,
                            username=user_model.username,
                            email=user_model.email,
                            roles=roles,
                            permissions=["read", "write"] if "admin" not in roles else ["read", "write", "trade", "admin"],
                            created_at=user_model.created_at,
                        )
        except Exception as exc:
            logger.debug("DB api-key lookup fallback: %s", exc)

        for entry in DatabaseUserStore._legacy_users():
            if entry.get("api_key") == api_key:
                return DatabaseUserStore._legacy_user_to_model(entry)
        return None

    @staticmethod
    async def validate_credentials(username: str, password: str) -> Optional[User]:
        """Validate username/email + password against DB hash."""
        try:
            async with db_session() as session:
                result = await session.execute(
                    select(UserModel).where(
                        (UserModel.username == username) | (UserModel.email == username)
                    )
                )
                model = result.scalar_one_or_none()
                if (
                    model is not None
                    and model.is_active
                    and model.password_hash
                    and verify_password(password, model.password_hash)
                ):
                    roles = await DatabaseUserStore._get_roles(model.id)
                    return User(
                        user_id=model.id,
                        username=model.username,
                        email=model.email,
                        roles=roles,
                        permissions=["read", "write"] if "admin" not in roles else ["read", "write", "trade", "admin"],
                        created_at=model.created_at,
                    )
        except Exception as exc:
            logger.debug("DB credential lookup fallback for %s: %s", username, exc)

        username_norm = (username or "").strip().lower()
        for entry in DatabaseUserStore._legacy_users():
            entry_username = str(entry.get("username") or "").strip().lower()
            entry_email = str(entry.get("email") or "").strip().lower()
            if username_norm not in {entry_username, entry_email}:
                continue
            password_hash = entry.get("password_hash")
            if not isinstance(password_hash, str) or not verify_password(password, password_hash):
                break
            return DatabaseUserStore._legacy_user_to_model(entry)

        # Recovery fallback for local operations: keep admin login in sync with live env password
        # even if database/users.json hashes drift during resets or repair scripts.
        admin_password = (os.getenv("APEX_ADMIN_PASSWORD") or "").strip()
        if admin_password and username_norm in {"admin", "admin@apex.local"}:
            if secrets.compare_digest(password, admin_password):
                for entry in DatabaseUserStore._legacy_users():
                    if str(entry.get("username") or "").strip().lower() == "admin":
                        user = DatabaseUserStore._legacy_user_to_model(entry)
                        user.roles = sorted(set((user.roles or []) + ["admin", "user"]))
                        user.permissions = sorted(set((user.permissions or []) + ["admin", "read", "write", "trade"]))
                        return user
                return User(
                    user_id="admin",
                    username="admin",
                    email="admin@apex.local",
                    roles=["admin", "user"],
                    permissions=["admin", "read", "write", "trade"],
                )
        return None


USER_STORE = DatabaseUserStore()


class TokenBlacklist:
    """Redis-backed token blacklist for logout/revocation."""

    def __init__(self) -> None:
        self._fallback_revoked: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._prefix = "auth:blacklist:"
        self._strict_envs = {"staging", "production", "prod"}

    async def _require_redis_if_strict(self) -> None:
        env = _runtime_env()
        if env in self._strict_envs:
            redis_client = await get_redis()
            if redis_client is None:
                raise RuntimeError(
                    f"Redis is required for token revocation in {env} environment."
                )

    async def _get_redis_or_none(self):
        await self._require_redis_if_strict()
        return await get_redis()

    async def revoke(self, token: str, expires_at: Optional[datetime] = None):
        """Add a token to the blacklist."""
        exp_ts = (expires_at or datetime.utcnow() + timedelta(hours=24)).timestamp()
        try:
            redis_client = await self._get_redis_or_none()
            if redis_client is not None:
                ttl = max(1, int(exp_ts - time.time()))
                await redis_client.setex(f"{self._prefix}{token}", ttl, "1")
                return
        except Exception as e:
            logger.warning("Redis revoke failed, falling back to in-memory: %s", e)

        async with self._lock:
            self._fallback_revoked[token] = exp_ts
            now = time.time()
            self._fallback_revoked = {
                k: v for k, v in self._fallback_revoked.items() if v > now
            }

    async def is_revoked(self, token: str) -> bool:
        """Check if a token has been revoked."""
        try:
            redis_client = await self._get_redis_or_none()
            if redis_client is not None:
                return await redis_client.exists(f"{self._prefix}{token}") == 1
        except Exception as e:
            logger.debug("Redis is_revoked check failed, using in-memory: %s", e)

        async with self._lock:
            return token in self._fallback_revoked and self._fallback_revoked[token] > time.time()

    async def verify_runtime_requirements(self) -> None:
        """Ensure strict environments have Redis connectivity."""
        await self._require_redis_if_strict()


TOKEN_BLACKLIST = TokenBlacklist()


# JWT Token Functions
def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    if not JWT_AVAILABLE:
        if _is_development_env():
            return f"mock-token-{user.user_id}"
        raise RuntimeError("Mock access tokens are only allowed in development")

    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=AUTH_CONFIG.access_token_expire_minutes)
    )

    payload = {
        "sub": user.user_id,
        "username": user.username,
        "roles": user.roles,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    return jwt.encode(payload, AUTH_CONFIG.secret_key, algorithm=AUTH_CONFIG.algorithm)


def create_refresh_token(user: User) -> str:
    """Create a JWT refresh token."""
    if not JWT_AVAILABLE:
        if _is_development_env():
            return f"mock-refresh-{user.user_id}"
        raise RuntimeError("Mock refresh tokens are only allowed in development")

    expire = datetime.utcnow() + timedelta(days=AUTH_CONFIG.refresh_token_expire_days)

    payload = {
        "sub": user.user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    return jwt.encode(payload, AUTH_CONFIG.secret_key, algorithm=AUTH_CONFIG.algorithm)


async def verify_token_async(token: str, expected_token_type: Optional[str] = None) -> Optional[TokenData]:
    """Verify and decode a JWT token, optionally enforcing token type."""
    if await TOKEN_BLACKLIST.is_revoked(token):
        logger.warning("Rejected revoked token")
        return None

    if not JWT_AVAILABLE:
        # Mock verification — only allowed when auth is explicitly disabled
        if (
            _is_development_env()
            and not AUTH_CONFIG.enabled
            and token.startswith("mock-token-")
        ):
            user_id = token.replace("mock-token-", "")
            user = await USER_STORE.get_user(user_id)
            if user:
                token_data = TokenData(
                    user_id=user.user_id,
                    username=user.username,
                    roles=user.roles,
                    exp=datetime.utcnow() + timedelta(hours=1),
                    iat=datetime.utcnow()
                )
                if expected_token_type and token_data.token_type != expected_token_type:
                    return None
                return token_data
        raise RuntimeError(
            "JWT verification requested while PyJWT is unavailable outside development"
        )

    try:
        payload = jwt.decode(
            token,
            AUTH_CONFIG.secret_key,
            algorithms=[AUTH_CONFIG.algorithm]
        )

        token_data = TokenData(
            user_id=payload["sub"],
            username=payload.get("username", ""),
            roles=payload.get("roles", []),
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
            token_type=payload.get("type", "access")
        )
        if expected_token_type and token_data.token_type != expected_token_type:
            logger.warning(
                "Rejected token with invalid type: expected=%s actual=%s",
                expected_token_type,
                token_data.token_type,
            )
            return None
        return token_data
    except jwt.ExpiredSignatureError:
        alert_agg.add("token_expired", "Authentication token expired")
        return None
    except jwt.InvalidTokenError as e:
        alert_agg.add("token_invalid", "Invalid authentication token", data={"error": str(e)})
        return None


def verify_token(token: str, expected_token_type: Optional[str] = None) -> Optional[TokenData]:
    """Synchronous token verification wrapper for compatibility paths/tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("verify_token() cannot run inside an active event loop; use verify_token_async().")
    return asyncio.run(verify_token_async(token, expected_token_type=expected_token_type))


# FastAPI Dependencies
security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=AUTH_CONFIG.api_key_header, auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    api_key: str = Depends(api_key_header),
    request: Request = None
) -> Optional[User]:
    """
    Get current authenticated user.

    If SaaSAuthMiddleware set request.state.user (PostgreSQL/OAuth), use that.
    Otherwise supports JWT bearer tokens and API keys via DB-backed user store.
    """
    if not AUTH_CONFIG.enabled:
        # Return default user when auth is disabled
        return User(
            user_id="default",
            username="default",
            roles=["admin", "user"],
            permissions=["read", "write", "trade", "admin"]
        )

    # Bridge: use user set by SaaSAuthMiddleware (PostgreSQL/API key auth)
    if request is not None:
        state_user = getattr(request.state, "user", None)
        if state_user is not None:
            uid = getattr(state_user, "id", None) or getattr(state_user, "user_id", None)
            if uid:
                roles = getattr(request.state, "roles", None) or getattr(state_user, "roles", ["user"])
                tier = getattr(request.state, "tier", None)
                return User(
                    user_id=str(uid),
                    username=getattr(state_user, "username", "?"),
                    email=getattr(state_user, "email", None),
                    roles=roles if isinstance(roles, list) else list(roles),
                    permissions=["read", "write"],
                    tier=tier,
                )

    # Try API key first
    if api_key:
        try:
            user = await USER_STORE.get_user_by_api_key(api_key)
            if user:
                return user
        except Exception as exc:  # SWALLOW: lookup failure should degrade to anonymous user
            logger.exception("API key auth lookup failed: %s", exc)
            return None

    # Try JWT token
    if credentials:
        token_data = await verify_token_async(credentials.credentials, expected_token_type="access")
        if token_data:
            try:
                user = await USER_STORE.get_user(token_data.user_id)
            except Exception as exc:  # SWALLOW: lookup failure should degrade to anonymous user
                logger.exception("Token user lookup failed: %s", exc)
                return None
            if user:
                return user

    return None


async def require_user(user: User = Depends(get_current_user)) -> User:
    """Require authenticated user (raises 401 if not authenticated)."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


def require_role(role: str):
    """Dependency to require specific role."""
    async def check_role(user: User = Depends(require_user)) -> User:
        if not user.has_role(role):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required"
            )
        return user
    return check_role


def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def check_permission(user: User = Depends(require_user)) -> User:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user
    return check_permission


# Rate Limiting
class RateLimiter:
    """
    Token bucket rate limiter.

    Supports:
    - Per-IP rate limiting
    - Per-user rate limiting
    - Sliding window algorithm
    """

    def __init__(self):
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_seen: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._max_keys = 10_000
        self._clock = time.monotonic

    @staticmethod
    def _prune_window(window: Deque[float], window_start: float) -> None:
        while window and window[0] <= window_start:
            window.popleft()

    def _prune_stale_keys(self) -> None:
        if len(self._last_seen) <= self._max_keys:
            return
        overflow = len(self._last_seen) - self._max_keys
        for stale_key, _ in sorted(self._last_seen.items(), key=lambda item: item[1])[:overflow]:
            self.requests.pop(stale_key, None)
            self._last_seen.pop(stale_key, None)

    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            now = self._clock()
            window_start = now - window_seconds

            bucket = self.requests[key]
            self._prune_window(bucket, window_start)

            # Check limit
            if len(bucket) >= max_requests:
                self._last_seen[key] = now
                return False

            # Record request
            bucket.append(now)
            self._last_seen[key] = now
            self._prune_stale_keys()
            return True

    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests in window."""
        now = self._clock()
        window_start = now - window_seconds
        bucket = self.requests.get(key)
        if not bucket:
            return max_requests
        recent_count = 0
        for ts in bucket:
            if ts > window_start:
                recent_count += 1
        return max(0, max_requests - recent_count)

    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """Get seconds until rate limit resets."""
        bucket = self.requests.get(key)
        if not bucket:
            return 0
        now = self._clock()
        window_start = now - window_seconds
        for ts in bucket:
            if ts > window_start:
                return max(0, (ts + window_seconds) - now)
        return 0


# Global rate limiter
RATE_LIMITER = RateLimiter()


def rate_limit(
    requests: int = 100,
    window: int = 60,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    Rate limiting decorator.

    Args:
        requests: Maximum requests allowed
        window: Time window in seconds
        key_func: Function to extract rate limit key from request

    Usage:
        @app.get("/data")
        @rate_limit(requests=100, window=60)
        async def get_data():
            ...
    """
    def get_default_key(request: Request) -> str:
        # Use IP address as default key
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            if request is None:
                # Try to find request in kwargs
                request = kwargs.get('request')

            if request:
                key = (key_func or get_default_key)(request)
                key = f"rate:{func.__name__}:{key}"

                allowed = await RATE_LIMITER.is_allowed(key, requests, window)

                if not allowed:
                    remaining = RATE_LIMITER.get_remaining(key, requests, window)
                    reset_time = RATE_LIMITER.get_reset_time(key, window)

                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={
                            "X-RateLimit-Limit": str(requests),
                            "X-RateLimit-Remaining": str(remaining),
                            "X-RateLimit-Reset": str(int(reset_time))
                        }
                    )

            return await func(*args, request=request, **kwargs)

        return wrapper
    return decorator


# WebSocket Authentication
async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """
    Authenticate WebSocket connection.

    Checks for API key in query params or first message.
    """
    if not AUTH_CONFIG.enabled:
        return User(
            user_id="default",
            username="default",
            roles=["admin", "user"],
            permissions=["read", "write", "trade", "admin"]
        )

    # Check query params for API key
    api_key = websocket.query_params.get("api_key")
    if api_key:
        user = await USER_STORE.get_user_by_api_key(api_key)
        if user:
            return user

    # Check query params for token
    token = websocket.query_params.get("token")
    if token:
        token_data = await verify_token_async(token, expected_token_type="access")
        if token_data:
            return await USER_STORE.get_user(token_data.user_id)

    return None


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"apex-{secrets.token_hex(16)}"


def configure_auth(
    enabled: bool = None,
    secret_key: str = None,
    token_expire_minutes: int = None
):
    """Configure authentication settings."""
    global AUTH_CONFIG

    if enabled is not None:
        AUTH_CONFIG.enabled = enabled
    if secret_key is not None:
        AUTH_CONFIG.secret_key = secret_key
    if token_expire_minutes is not None:
        AUTH_CONFIG.access_token_expire_minutes = token_expire_minutes

    logger.info(f"Auth configured: enabled={AUTH_CONFIG.enabled}")


async def verify_auth_runtime_prerequisites() -> None:
    """Validate auth runtime dependencies for current environment."""
    if AUTH_CONFIG.enabled and not JWT_AVAILABLE and not _is_development_env():
        raise RuntimeError("PyJWT is required when authentication is enabled outside development")
    await TOKEN_BLACKLIST.verify_runtime_requirements()

    # Warn/fail if critical secrets are ephemeral
    is_dev = _is_development_env()
    if not os.getenv("APEX_SECRET_KEY"):
        msg = "APEX_SECRET_KEY not set — JWT tokens will not survive restarts or work across workers"
        if is_dev:
            logger.warning(msg)
        else:
            raise RuntimeError(msg + ". Set APEX_SECRET_KEY in your environment.")
    if not os.getenv("APEX_MASTER_KEY"):
        msg = "APEX_MASTER_KEY not set — encrypted broker credentials will be lost on restart"
        if is_dev:
            logger.warning(msg)
        else:
            raise RuntimeError(msg + ". Set APEX_MASTER_KEY in your environment.")


# Login endpoint helper
async def login(username: str, password: str) -> Optional[Dict[str, str]]:
    """
    Authenticate user and return tokens.

    Returns:
        Dict with access_token and refresh_token, or None if auth fails
    """
    user = await USER_STORE.validate_credentials(username, password)

    if not user:
        return None

    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
        "expires_in": AUTH_CONFIG.access_token_expire_minutes * 60
    }
