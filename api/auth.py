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
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict
import asyncio

from fastapi import HTTPException, Depends, Request, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

logger = logging.getLogger(__name__)

# Try to import JWT library
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available. Install with: pip install PyJWT")


# Configuration
@dataclass
class AuthConfig:
    """Authentication configuration."""
    secret_key: str = os.getenv("APEX_SECRET_KEY", secrets.token_hex(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    api_key_header: str = "X-API-Key"
    enabled: bool = os.getenv("APEX_AUTH_ENABLED", "false").lower() == "true"
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


@dataclass
class TokenData:
    """JWT token payload data."""
    user_id: str
    username: str
    roles: List[str]
    exp: datetime
    iat: datetime
    token_type: str = "access"


# In-memory user store (replace with database in production)
class UserStore:
    """Simple in-memory user storage."""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id

        # Create default admin user
        self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user if not exists."""
        admin_key = os.getenv("APEX_ADMIN_API_KEY", "apex-admin-key-change-me")
        admin = User(
            user_id="admin",
            username="admin",
            email="admin@apex.local",
            roles=["admin", "user"],
            permissions=["read", "write", "trade", "admin"],
            api_key=admin_key
        )
        self.users["admin"] = admin
        self.api_keys[admin_key] = "admin"

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.users.get(user_id)
        return None

    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        roles: List[str] = None
    ) -> User:
        """Create a new user."""
        user_id = secrets.token_hex(8)
        api_key = f"apex-{secrets.token_hex(16)}"

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or ["user"],
            permissions=["read"],
            api_key=api_key,
            created_at=datetime.utcnow()
        )

        self.users[user_id] = user
        self.api_keys[api_key] = user_id

        return user

    def validate_credentials(self, username: str, password_hash: str) -> Optional[User]:
        """Validate user credentials (simplified)."""
        # In production, check against database with proper password hashing
        user = next(
            (u for u in self.users.values() if u.username == username),
            None
        )
        return user


# Global user store
USER_STORE = UserStore()


# JWT Token Functions
def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    if not JWT_AVAILABLE:
        return f"mock-token-{user.user_id}"

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
        return f"mock-refresh-{user.user_id}"

    expire = datetime.utcnow() + timedelta(days=AUTH_CONFIG.refresh_token_expire_days)

    payload = {
        "sub": user.user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    return jwt.encode(payload, AUTH_CONFIG.secret_key, algorithm=AUTH_CONFIG.algorithm)


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    if not JWT_AVAILABLE:
        # Mock verification
        if token.startswith("mock-token-"):
            user_id = token.replace("mock-token-", "")
            user = USER_STORE.get_user(user_id)
            if user:
                return TokenData(
                    user_id=user.user_id,
                    username=user.username,
                    roles=user.roles,
                    exp=datetime.utcnow() + timedelta(hours=1),
                    iat=datetime.utcnow()
                )
        return None

    try:
        payload = jwt.decode(
            token,
            AUTH_CONFIG.secret_key,
            algorithms=[AUTH_CONFIG.algorithm]
        )

        return TokenData(
            user_id=payload["sub"],
            username=payload.get("username", ""),
            roles=payload.get("roles", []),
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
            token_type=payload.get("type", "access")
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None


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

    Supports both JWT bearer tokens and API keys.
    """
    if not AUTH_CONFIG.enabled:
        # Return default user when auth is disabled
        return User(
            user_id="default",
            username="default",
            roles=["admin"],
            permissions=["read", "write", "trade", "admin"]
        )

    # Try API key first
    if api_key:
        user = USER_STORE.get_user_by_api_key(api_key)
        if user:
            return user

    # Try JWT token
    if credentials:
        token_data = verify_token(credentials.credentials)
        if token_data:
            user = USER_STORE.get_user(token_data.user_id)
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
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds

            # Clean old requests
            self.requests[key] = [
                t for t in self.requests[key]
                if t > window_start
            ]

            # Check limit
            if len(self.requests[key]) >= max_requests:
                return False

            # Record request
            self.requests[key].append(now)
            return True

    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests in window."""
        now = time.time()
        window_start = now - window_seconds

        recent = [t for t in self.requests.get(key, []) if t > window_start]
        return max(0, max_requests - len(recent))

    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """Get seconds until rate limit resets."""
        if key not in self.requests or not self.requests[key]:
            return 0

        oldest = min(self.requests[key])
        reset_at = oldest + window_seconds
        return max(0, reset_at - time.time())


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
            roles=["admin"],
            permissions=["read", "write", "trade", "admin"]
        )

    # Check query params for API key
    api_key = websocket.query_params.get("api_key")
    if api_key:
        user = USER_STORE.get_user_by_api_key(api_key)
        if user:
            return user

    # Check query params for token
    token = websocket.query_params.get("token")
    if token:
        token_data = verify_token(token)
        if token_data:
            return USER_STORE.get_user(token_data.user_id)

    return None


# Utility functions
def hash_password(password: str) -> str:
    """Hash a password (use proper bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


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


# Login endpoint helper
async def login(username: str, password: str) -> Optional[Dict[str, str]]:
    """
    Authenticate user and return tokens.

    Returns:
        Dict with access_token and refresh_token, or None if auth fails
    """
    password_hash = hash_password(password)
    user = USER_STORE.validate_credentials(username, password_hash)

    if not user:
        return None

    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
        "expires_in": AUTH_CONFIG.access_token_expire_minutes * 60
    }
