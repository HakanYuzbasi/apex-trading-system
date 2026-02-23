import pytest
from datetime import datetime, timedelta
from api.auth import (
    User, create_access_token, verify_token,
    RateLimiter
)

# -----------------------------------------------------------------------------
# User Serialization Tests
# -----------------------------------------------------------------------------

def test_user_serialization():
    user = User(
        user_id="test_id",
        username="test_user",
        email="test@example.com",
        roles=["user", "admin"],
        permissions=["read", "write"],
        api_key="test_key",
        created_at=datetime.utcnow(),
        tier="pro"
    )
    
    data = user.to_dict()
    assert data["user_id"] == "test_id"
    assert data["username"] == "test_user"
    assert "created_at" in data
    
    restored = User.from_dict(data)
    assert restored.user_id == user.user_id
    assert restored.username == user.username
    assert restored.roles == user.roles
    assert restored.created_at == user.created_at

# -----------------------------------------------------------------------------
# Persistent Store Tests
# -----------------------------------------------------------------------------

@pytest.mark.skip(reason="JSONFileUserStore removed — auth migrated to DatabaseUserStore")
def test_json_store_persistence():
    pass

# -----------------------------------------------------------------------------
# JWT Tests
# -----------------------------------------------------------------------------

def test_jwt_token_cycle():
    user = User(user_id="123", username="jwt_test", roles=["user"])
    
    token = create_access_token(user)
    assert isinstance(token, str)
    
    token_data = verify_token(token)
    assert token_data is not None
    assert token_data.user_id == "123"
    assert token_data.username == "jwt_test"

def test_jwt_expiry():
    user = User(user_id="123", username="expiry_test", roles=["user"])
    
    # Create expired token
    token = create_access_token(user, expires_delta=timedelta(seconds=-1))
    
    token_data = verify_token(token)
    assert token_data is None

# -----------------------------------------------------------------------------
# Rate Limiting Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limiter():
    limiter = RateLimiter()
    key = "test_ip"
    
    # Allow 2 requests per 1 second
    allowed1 = await limiter.is_allowed(key, max_requests=2, window_seconds=1)
    assert allowed1 is True
    
    allowed2 = await limiter.is_allowed(key, max_requests=2, window_seconds=1)
    assert allowed2 is True
    
    # Should be blocked
    allowed3 = await limiter.is_allowed(key, max_requests=2, window_seconds=1)
    assert allowed3 is False
    
    # Wait for window reset
    await asyncio.sleep(1.1)
    
    allowed4 = await limiter.is_allowed(key, max_requests=2, window_seconds=1)
    assert allowed4 is True

import asyncio
