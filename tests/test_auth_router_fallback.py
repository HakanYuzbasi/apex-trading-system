from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.auth as legacy_auth
from services.auth.router import router as auth_router
from services.auth.service import AuthService
from services.common.db import get_db


class _DummyDB:
    def __init__(self) -> None:
        self.rollback_called = False

    async def rollback(self) -> None:
        self.rollback_called = True


def _build_app_with_db_override(db_obj: _DummyDB) -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)

    async def _fake_get_db():
        yield db_obj

    app.dependency_overrides[get_db] = _fake_get_db
    return app


def test_auth_login_falls_back_to_legacy_without_500(monkeypatch):
    db_obj = _DummyDB()
    app = _build_app_with_db_override(db_obj)

    async def _db_login_failure(self, username: str, password: str):
        _ = self, username, password
        raise RuntimeError("db unavailable")

    async def _legacy_login(username: str, password: str):
        _ = username, password
        return {"access_token": "tok-access", "refresh_token": "tok-refresh"}

    monkeypatch.setattr(AuthService, "login", _db_login_failure)
    monkeypatch.setattr(legacy_auth, "login", _legacy_login)

    client = TestClient(app)
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "irrelevant"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "tok-access"
    assert body["refresh_token"] == "tok-refresh"
    assert db_obj.rollback_called is True


def test_auth_login_returns_401_not_500_when_both_db_and_legacy_fail(monkeypatch):
    db_obj = _DummyDB()
    app = _build_app_with_db_override(db_obj)

    async def _db_login_failure(self, username: str, password: str):
        _ = self, username, password
        raise RuntimeError("db unavailable")

    async def _legacy_login(username: str, password: str):
        _ = username, password
        return None

    monkeypatch.setattr(AuthService, "login", _db_login_failure)
    monkeypatch.setattr(legacy_auth, "login", _legacy_login)

    client = TestClient(app)
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"
    assert db_obj.rollback_called is True
