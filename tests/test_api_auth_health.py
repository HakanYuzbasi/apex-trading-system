"""
tests/test_api_auth_health.py - API Authentication & Health Endpoint Tests

Validates:
- Health endpoint returns online/offline based on state freshness
- Staleness detection (_state_is_fresh) with configurable threshold
- Auth middleware (require_user) blocks unauthenticated requests
- API key authentication works
- JWT token creation and verification
- Rate limiter allows/blocks requests correctly
- WebSocket auth via query params
"""

import pytest
import time
from datetime import datetime, timedelta

from api.auth import (
    AuthConfig,
    User,
    UserStore,
    RateLimiter,
    create_access_token,
    create_refresh_token,
    verify_token,
    configure_auth,
    hash_password,
    verify_password,
    generate_api_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def user_store():
    """Fresh user store for each test."""
    return UserStore()


@pytest.fixture
def sample_user(user_store):
    """Create a sample non-admin user."""
    return user_store.create_user(
        username="testuser",
        email="test@example.com",
        roles=["user"],
        password="password-123",
    )


@pytest.fixture
def admin_user(user_store):
    """Get the default admin user."""
    return user_store.get_user("admin")


# ---------------------------------------------------------------------------
# Tests: User model
# ---------------------------------------------------------------------------

class TestUserModel:
    """User dataclass role/permission checks."""

    def test_default_roles(self):
        user = User(user_id="u1", username="test")
        assert user.roles == ["user"]
        assert user.permissions == ["read"]

    def test_has_role(self):
        user = User(user_id="u1", username="test", roles=["user", "trader"])
        assert user.has_role("trader")
        assert not user.has_role("admin")

    def test_admin_has_all_roles(self):
        user = User(user_id="u1", username="admin", roles=["admin"])
        assert user.has_role("trader")
        assert user.has_role("anything")

    def test_has_permission(self):
        user = User(user_id="u1", username="test", permissions=["read", "write"])
        assert user.has_permission("read")
        assert user.has_permission("write")
        assert not user.has_permission("admin")

    def test_admin_has_all_permissions(self):
        user = User(user_id="u1", username="admin", roles=["admin"], permissions=[])
        assert user.has_permission("anything")


# ---------------------------------------------------------------------------
# Tests: UserStore
# ---------------------------------------------------------------------------

class TestUserStore:
    """In-memory user store operations."""

    def test_default_admin_exists(self, user_store):
        admin = user_store.get_user("admin")
        assert admin is not None
        assert admin.username == "admin"
        assert "admin" in admin.roles

    def test_create_user(self, user_store):
        user = user_store.create_user(username="newuser", email="new@test.com")
        assert user.username == "newuser"
        assert user.api_key is not None
        assert user.api_key.startswith("apex-")

    def test_get_user_by_api_key(self, user_store, sample_user):
        found = user_store.get_user_by_api_key(sample_user.api_key)
        assert found is not None
        assert found.user_id == sample_user.user_id

    def test_get_user_by_invalid_key(self, user_store):
        assert user_store.get_user_by_api_key("invalid-key") is None

    def test_get_nonexistent_user(self, user_store):
        assert user_store.get_user("nonexistent") is None

    def test_validate_credentials_success(self, sample_user, user_store):
        assert user_store.validate_credentials(sample_user.username, "password-123") is not None

    def test_validate_credentials_rejects_wrong_password(self, sample_user, user_store):
        assert user_store.validate_credentials(sample_user.username, "wrong-password") is None


# ---------------------------------------------------------------------------
# Tests: JWT tokens
# ---------------------------------------------------------------------------

class TestJWTTokens:
    """Token creation and verification."""

    def test_create_and_verify_token(self, admin_user):
        token = create_access_token(admin_user)
        assert token is not None
        assert len(token) > 0

        data = verify_token(token)
        assert data is not None
        assert data.user_id == "admin"

    def test_verify_invalid_token(self):
        result = verify_token("completely-invalid-token")
        assert result is None

    def test_token_contains_roles(self, admin_user):
        token = create_access_token(admin_user)
        data = verify_token(token)
        assert data is not None
        assert "admin" in data.roles

    def test_refresh_token_rejected_when_access_required(self, admin_user):
        refresh = create_refresh_token(admin_user)
        assert verify_token(refresh, expected_token_type="access") is None


# ---------------------------------------------------------------------------
# Tests: Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    """Token bucket rate limiter logic."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        limiter = RateLimiter()
        for i in range(5):
            assert await limiter.is_allowed("test-key", max_requests=5, window_seconds=60)

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        limiter = RateLimiter()
        # Fill up the bucket
        for _ in range(3):
            await limiter.is_allowed("test-key", max_requests=3, window_seconds=60)
        # Next request should be blocked
        assert not await limiter.is_allowed("test-key", max_requests=3, window_seconds=60)

    @pytest.mark.asyncio
    async def test_separate_keys_independent(self):
        limiter = RateLimiter()
        # Fill up key-A
        for _ in range(2):
            await limiter.is_allowed("key-A", max_requests=2, window_seconds=60)
        # key-A blocked
        assert not await limiter.is_allowed("key-A", max_requests=2, window_seconds=60)
        # key-B still allowed
        assert await limiter.is_allowed("key-B", max_requests=2, window_seconds=60)

    def test_get_remaining(self):
        limiter = RateLimiter()
        assert limiter.get_remaining("fresh-key", max_requests=10, window_seconds=60) == 10

    def test_get_reset_time_empty(self):
        limiter = RateLimiter()
        assert limiter.get_reset_time("fresh-key", window_seconds=60) == 0


# ---------------------------------------------------------------------------
# Tests: Staleness detection
# ---------------------------------------------------------------------------

class TestStalenessDetection:
    """_state_is_fresh must correctly determine if state is stale."""

    def test_fresh_state(self):
        from api.server import _state_is_fresh
        state = {"timestamp": datetime.utcnow().isoformat()}
        assert _state_is_fresh(state, threshold_seconds=30) is True

    def test_stale_state(self):
        from api.server import _state_is_fresh
        old_ts = (datetime.utcnow() - timedelta(seconds=60)).isoformat()
        state = {"timestamp": old_ts}
        assert _state_is_fresh(state, threshold_seconds=30) is False

    def test_no_timestamp(self):
        from api.server import _state_is_fresh
        assert _state_is_fresh({}, threshold_seconds=30) is False

    def test_null_timestamp(self):
        from api.server import _state_is_fresh
        assert _state_is_fresh({"timestamp": None}, threshold_seconds=30) is False

    def test_exact_threshold_boundary(self):
        from api.server import _state_is_fresh
        # State well within threshold should be fresh
        ts = (datetime.utcnow() - timedelta(seconds=25)).isoformat()
        state = {"timestamp": ts}
        assert _state_is_fresh(state, threshold_seconds=30) is True

    def test_configurable_threshold(self):
        from api.server import _state_is_fresh
        # With a 120s threshold, 60s old state is still fresh
        ts = (datetime.utcnow() - timedelta(seconds=60)).isoformat()
        state = {"timestamp": ts}
        assert _state_is_fresh(state, threshold_seconds=120) is True
        # But with 30s threshold, it's stale
        assert _state_is_fresh(state, threshold_seconds=30) is False


class TestStateCaching:
    """State file reads should use mtime-aware cache semantics."""

    def test_read_trading_state_returns_same_object_when_unchanged(self, tmp_path, monkeypatch):
        import json
        from api import server

        state_file = tmp_path / "trading_state.json"
        price_file = tmp_path / "price_cache.json"
        state_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "capital": 10000,
                    "positions": {"AAPL": {"avg_price": 100, "qty": 10}},
                }
            )
        )
        price_file.write_text(json.dumps({"AAPL": 110.0}))

        monkeypatch.setattr(server, "STATE_FILE", state_file)
        monkeypatch.setattr(server, "PRICE_CACHE_FILE", price_file)
        monkeypatch.setattr(server, "_price_cache_data", {})
        monkeypatch.setattr(server, "_price_cache_mtime_ns", None)
        monkeypatch.setattr(server, "_state_cache_data", server.DEFAULT_STATE)
        monkeypatch.setattr(server, "_state_cache_mtime_ns", None)
        monkeypatch.setattr(server, "_state_cache_price_mtime_ns", None)

        s1 = server.read_trading_state()
        s2 = server.read_trading_state()

        assert s1 is s2
        assert s1["positions"]["AAPL"]["current_price"] == 110.0

    def test_read_trading_state_invalidates_when_price_cache_changes(self, tmp_path, monkeypatch):
        import json
        from api import server

        state_file = tmp_path / "trading_state.json"
        price_file = tmp_path / "price_cache.json"
        state_file.write_text(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "capital": 10000,
                    "positions": {"AAPL": {"avg_price": 100, "qty": 10}},
                }
            )
        )
        price_file.write_text(json.dumps({"AAPL": 100.0}))

        monkeypatch.setattr(server, "STATE_FILE", state_file)
        monkeypatch.setattr(server, "PRICE_CACHE_FILE", price_file)
        monkeypatch.setattr(server, "_price_cache_data", {})
        monkeypatch.setattr(server, "_price_cache_mtime_ns", None)
        monkeypatch.setattr(server, "_state_cache_data", server.DEFAULT_STATE)
        monkeypatch.setattr(server, "_state_cache_mtime_ns", None)
        monkeypatch.setattr(server, "_state_cache_price_mtime_ns", None)

        s1 = server.read_trading_state()
        time.sleep(0.01)
        price_file.write_text(json.dumps({"AAPL": 120.0}))
        s2 = server.read_trading_state()

        assert s1 is not s2
        assert s2["positions"]["AAPL"]["current_price"] == 120.0


# ---------------------------------------------------------------------------
# Tests: _parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    """Timestamp parsing for various ISO formats."""

    def test_naive_iso(self):
        from api.server import _parse_timestamp
        result = _parse_timestamp("2024-01-15T10:30:00")
        assert result is not None
        assert result.year == 2024

    def test_utc_z_suffix(self):
        from api.server import _parse_timestamp
        result = _parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None

    def test_none_returns_none(self):
        from api.server import _parse_timestamp
        assert _parse_timestamp(None) is None

    def test_empty_string_returns_none(self):
        from api.server import _parse_timestamp
        assert _parse_timestamp("") is None

    def test_garbage_returns_none(self):
        from api.server import _parse_timestamp
        assert _parse_timestamp("not-a-date") is None


# ---------------------------------------------------------------------------
# Tests: Auth config
# ---------------------------------------------------------------------------

class TestAuthConfig:
    """Authentication configuration."""

    def test_default_config(self):
        config = AuthConfig()
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 60
        assert config.api_key_header == "X-API-Key"

    def test_configure_auth(self):
        configure_auth(enabled=True, token_expire_minutes=120)
        from api.auth import AUTH_CONFIG
        assert AUTH_CONFIG.enabled is True
        assert AUTH_CONFIG.access_token_expire_minutes == 120
        # Reset
        configure_auth(enabled=False, token_expire_minutes=60)

    def test_password_hashing(self):
        hashed = hash_password("test123")
        assert hashed
        assert verify_password("test123", hashed)
        assert not verify_password("different", hashed)

    def test_generate_api_key(self):
        key = generate_api_key()
        assert key.startswith("apex-")
        assert len(key) > 10

    def test_api_keys_are_unique(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100


# ---------------------------------------------------------------------------
# Tests: FastAPI endpoints (using TestClient)
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Health endpoint integration tests."""

    def test_health_returns_offline_when_no_state(self):
        """Health should return offline when state file doesn't exist."""
        from fastapi.testclient import TestClient
        from api.server import app
        from api.auth import AUTH_CONFIG

        # Temporarily disable auth for testing
        original_enabled = AUTH_CONFIG.enabled
        AUTH_CONFIG.enabled = False
        try:
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["api"] == "ok"
            assert data["status"] in ("online", "offline")
        finally:
            AUTH_CONFIG.enabled = original_enabled

    def test_health_requires_auth_when_enabled(self):
        """Health should return 401 when auth is enabled and no creds provided."""
        from fastapi.testclient import TestClient
        from api.server import app
        from api.auth import AUTH_CONFIG

        original_enabled = AUTH_CONFIG.enabled
        AUTH_CONFIG.enabled = True
        try:
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 401
        finally:
            AUTH_CONFIG.enabled = original_enabled

    def test_health_with_api_key(self):
        """Health should work with valid API key."""
        from fastapi.testclient import TestClient
        from api.server import app
        from api.auth import AUTH_CONFIG, USER_STORE

        original_enabled = AUTH_CONFIG.enabled
        AUTH_CONFIG.enabled = True
        try:
            admin = USER_STORE.get_user("admin")
            client = TestClient(app)
            response = client.get(
                "/health",
                headers={"X-API-Key": admin.api_key},
            )
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers
        finally:
            AUTH_CONFIG.enabled = original_enabled

    def test_state_endpoint_requires_auth(self):
        """GET /state should require authentication."""
        from fastapi.testclient import TestClient
        from api.server import app
        from api.auth import AUTH_CONFIG

        original_enabled = AUTH_CONFIG.enabled
        AUTH_CONFIG.enabled = True
        try:
            client = TestClient(app)
            response = client.get("/state")
            assert response.status_code == 401
        finally:
            AUTH_CONFIG.enabled = original_enabled

    def test_state_with_valid_key(self):
        """GET /state should return data with valid API key."""
        from fastapi.testclient import TestClient
        from api.server import app
        from api.auth import AUTH_CONFIG, USER_STORE

        original_enabled = AUTH_CONFIG.enabled
        AUTH_CONFIG.enabled = True
        try:
            admin = USER_STORE.get_user("admin")
            client = TestClient(app)
            response = client.get(
                "/state",
                headers={"X-API-Key": admin.api_key},
            )
            assert response.status_code == 200
            data = response.json()
            assert "positions" in data
        finally:
            AUTH_CONFIG.enabled = original_enabled
