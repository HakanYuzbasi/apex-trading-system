"""Ensure mock token paths are blocked outside development."""

from __future__ import annotations

import pytest

from api.auth import User


@pytest.mark.asyncio
async def test_api_auth_mock_verify_blocked_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.auth as auth

    monkeypatch.setenv("APEX_ENV", "production")
    monkeypatch.setattr(auth, "JWT_AVAILABLE", False)
    async def _never_revoked(token: str) -> bool:
        return False

    monkeypatch.setattr(auth.TOKEN_BLACKLIST, "is_revoked", _never_revoked)

    with pytest.raises(RuntimeError, match="outside development"):
        await auth.verify_token_async("mock-token-user")


def test_api_auth_mock_create_blocked_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.auth as auth

    monkeypatch.setenv("APEX_ENV", "staging")
    monkeypatch.setattr(auth, "JWT_AVAILABLE", False)

    user = User(user_id="u-1", username="u")
    with pytest.raises(RuntimeError, match="development"):
        auth.create_access_token(user)
    with pytest.raises(RuntimeError, match="development"):
        auth.create_refresh_token(user)


def test_service_auth_mock_create_blocked_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.auth.service as service

    monkeypatch.setenv("APEX_ENV", "production")
    monkeypatch.setattr(service, "_JOSE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="development"):
        service.create_access_token("u-1", "user", ["user"])
    with pytest.raises(RuntimeError, match="development"):
        service.create_refresh_token("u-1")


def test_service_auth_mock_decode_blocked_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.auth.service as service

    monkeypatch.setenv("APEX_ENV", "prod")
    monkeypatch.setattr(service, "_JOSE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="development"):
        service.decode_token("mock-token-u-1")
