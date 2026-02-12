import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from api.auth import (
    User, JSONFileUserStore, create_access_token, verify_token, 
    RateLimiter, hash_password, verify_password
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

@pytest.fixture
def temp_user_file(tmp_path):
    return tmp_path / "users.json"

def test_json_store_persistence(temp_user_file):
    store = JSONFileUserStore(temp_user_file)
    
    # Create user
    user = store.create_user(username="persistent_user", password="password123")
    assert user.user_id in store.users
    
    # Verify file content
    with open(temp_user_file, "r") as f:
        data = json.load(f)
        # Should be at least 1 (the created user), plus potentially default admin
        usernames = [u["username"] for u in data["users"]]
        assert "persistent_user" in usernames
        assert "password_hash" in next(u for u in data["users"] if u["username"] == "persistent_user")

    # Reload store
    new_store = JSONFileUserStore(temp_user_file)
    loaded_user = new_store.get_user(user.user_id)
    assert loaded_user is not None
    assert loaded_user.username == "persistent_user"
    
    # Check password validation on reloaded store
    validated = new_store.validate_credentials("persistent_user", "password123")
    assert validated is not None
    assert validated.user_id == user.user_id

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
