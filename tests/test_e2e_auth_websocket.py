import uuid
import asyncio
import os
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.auth import AUTH_CONFIG, USER_STORE
from api.server import PROMETHEUS_AVAILABLE, app


@pytest.fixture
def client():
    original_enabled = AUTH_CONFIG.enabled
    AUTH_CONFIG.enabled = True
    try:
        yield TestClient(app)
    finally:
        AUTH_CONFIG.enabled = original_enabled


def _register_and_login(client: TestClient):
    unique = uuid.uuid4().hex[:8]
    username = f"e2e_{unique}"
    email = f"{username}@example.com"
    password = "e2e-password-123"

    register_resp = client.post(
        "/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert register_resp.status_code == 200
    register_data = register_resp.json()
    assert "access_token" in register_data
    assert "refresh_token" in register_data

    wrong_login = client.post(
        "/auth/login",
        json={"username": username, "password": "incorrect-password"},
    )
    assert wrong_login.status_code == 401

    login_resp = client.post(
        "/auth/login",
        json={"username": username, "password": password},
    )
    assert login_resp.status_code == 200
    return login_resp.json()


def test_e2e_auth_refresh_and_protected_health(client: TestClient):
    tokens = _register_and_login(client)
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    invalid_refresh = client.post("/auth/refresh", json={"refresh_token": access_token})
    assert invalid_refresh.status_code == 401

    valid_refresh = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert valid_refresh.status_code == 200
    assert "access_token" in valid_refresh.json()

    health_resp = client.get("/health", headers={"Authorization": f"Bearer {access_token}"})
    assert health_resp.status_code == 200
    assert "status" in health_resp.json()


def test_e2e_websocket_accepts_access_token_and_rejects_refresh_token(client: TestClient):
    tokens = _register_and_login(client)
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    with client.websocket_connect(f"/ws?token={access_token}") as ws:
        payload = ws.receive_json()
        assert payload["type"] == "state_update"

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws?token={refresh_token}") as ws:
            ws.receive_json()


def test_metrics_endpoint_available_when_prometheus_client_installed(client: TestClient, monkeypatch):
    monkeypatch.setenv("APEX_METRICS_TOKEN", "")
    resp = client.get("/metrics")
    if PROMETHEUS_AVAILABLE:
        assert resp.status_code == 200
        assert "apex_api_http_requests_total" in resp.text
    else:
        assert resp.status_code == 503


def test_kill_switch_reset_endpoint_requires_admin_and_queues_command(
    client: TestClient,
    tmp_path,
    monkeypatch,
):
    from api import server

    control_file = tmp_path / "trading_control_commands.json"
    monkeypatch.setattr(server, "CONTROL_COMMAND_FILE", control_file)

    tokens = _register_and_login(client)
    user_resp = client.post(
        "/ops/kill-switch/reset",
        json={"reason": "Reset after verified false-positive trigger"},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert user_resp.status_code == 403

    admin = asyncio.run(USER_STORE.get_user("admin"))
    assert admin is not None
    assert admin.api_key

    admin_resp = client.post(
        "/ops/kill-switch/reset",
        json={"reason": "Reset after verified false-positive trigger"},
        headers={"X-API-Key": admin.api_key},
    )
    assert admin_resp.status_code == 200
    body = admin_resp.json()
    assert body["status"] == "queued"
    assert body["command"]["kill_switch_reset_requested"] is True
    assert body["command"]["requested_by"] == "admin"
    assert body["command"]["request_id"].startswith("ksr-")

    duplicate_resp = client.post(
        "/ops/kill-switch/reset",
        json={"reason": "Second request should be blocked"},
        headers={"X-API-Key": admin.api_key},
    )
    assert duplicate_resp.status_code == 409

    status_resp = client.get("/ops/kill-switch", headers={"X-API-Key": admin.api_key})
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status["command"]["kill_switch_reset_requested"] is True


def test_governor_policy_approve_and_rollback_endpoints_require_admin_and_audit(
    client: TestClient,
    tmp_path,
    monkeypatch,
):
    from api import server
    from risk.governor_policy import (
        GovernorPolicy,
        GovernorPolicyRepository,
        PolicyPromotionService,
        PromotionStatus,
        TierControls,
    )

    def make_policy(version: str, sharpe: float, dd: float) -> GovernorPolicy:
        return GovernorPolicy(
            asset_class="EQUITY",
            regime="default",
            version=version,
            oos_sharpe=sharpe,
            oos_drawdown=dd,
            tier_controls={
                "green": TierControls(1.0, 0.0, 0.0, False),
                "yellow": TierControls(0.8, 0.02, 0.03, False),
                "orange": TierControls(0.6, 0.05, 0.06, False),
                "red": TierControls(0.3, 0.10, 0.12, True),
            },
        )

    control_file = tmp_path / "trading_control_commands.json"
    policy_dir = tmp_path / "governor_policies"
    monkeypatch.setattr(server, "CONTROL_COMMAND_FILE", control_file)
    monkeypatch.setattr(server, "GOVERNOR_POLICY_DIR", policy_dir)

    repo = GovernorPolicyRepository(policy_dir)
    repo.save_active([make_policy("v1", sharpe=1.0, dd=0.10)])
    service = PolicyPromotionService(repository=repo, environment="prod", live_trading=True)
    candidate = make_policy("v2", sharpe=1.2, dd=0.08)
    submit = service.submit_candidate(candidate)
    assert submit.status == PromotionStatus.STAGED

    tokens = _register_and_login(client)
    non_admin = client.post(
        "/ops/governor/policies/approve",
        json={
            "policy_id": candidate.policy_id(),
            "reason": "Requesting production promotion after paper checks",
        },
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert non_admin.status_code == 403

    admin = asyncio.run(USER_STORE.get_user("admin"))
    assert admin is not None
    assert admin.api_key

    approved = client.post(
        "/ops/governor/policies/approve",
        json={
            "policy_id": candidate.policy_id(),
            "reason": "Production promotion approved by risk committee",
        },
        headers={"X-API-Key": admin.api_key},
    )
    assert approved.status_code == 200
    approved_body = approved.json()
    assert approved_body["status"] == "approved"
    assert approved_body["reload_command"]["governor_policy_reload_requested"] is True

    active_after_approval = client.get(
        "/ops/governor/policies/active",
        headers={"X-API-Key": admin.api_key},
    )
    assert active_after_approval.status_code == 200
    versions = [p["version"] for p in active_after_approval.json()["policies"] if p["policy_key"] == "EQUITY:default"]
    assert "v2" in versions

    rolled_back = client.post(
        "/ops/governor/policies/rollback",
        json={
            "asset_class": "EQUITY",
            "regime": "default",
            "reason": "Live Sharpe degradation, revert to prior stable policy",
        },
        headers={"X-API-Key": admin.api_key},
    )
    assert rolled_back.status_code == 200
    rollback_body = rolled_back.json()
    assert rollback_body["status"] == "rolled_back"
    assert rollback_body["reload_command"]["governor_policy_reload_requested"] is True

    active_after_rollback = client.get(
        "/ops/governor/policies/active",
        headers={"X-API-Key": admin.api_key},
    )
    assert active_after_rollback.status_code == 200
    versions = [p["version"] for p in active_after_rollback.json()["policies"] if p["policy_key"] == "EQUITY:default"]
    assert "v1" in versions

    audit_resp = client.get(
        "/ops/governor/policies/audit?asset_class=EQUITY&regime=default&limit=100",
        headers={"X-API-Key": admin.api_key},
    )
    assert audit_resp.status_code == 200
    actions = [event.get("action") for event in audit_resp.json()["events"]]
    assert "manual_approved" in actions
    assert "rollback_activated" in actions


def test_social_governor_decision_audit_endpoint_requires_admin(
    client: TestClient,
    tmp_path,
    monkeypatch,
):
    from api import server
    from risk.social_decision_audit import SocialDecisionAuditRepository

    audit_file = tmp_path / "audit" / "social_governor_decisions.jsonl"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    repo = SocialDecisionAuditRepository(audit_file)
    
    monkeypatch.setattr(server, "_social_audit_repo_for_user", lambda uid: repo)
    monkeypatch.setattr(server, "SOCIAL_DECISION_AUDIT_FILE", audit_file)
    monkeypatch.setattr(server, "SOCIAL_DECISION_AUDIT_LEGACY_FILE", audit_file)
    repo.append_event(
        {
            "asset_class": "EQUITY",
            "regime": "risk_off",
            "policy_version": "sshock-test",
            "decision": {"block_new_entries": True},
        }
    )

    unauth = client.get(
        "/api/v1/social-governor/decisions?limit=10",
    )
    assert unauth.status_code in {401, 403}

    admin = asyncio.run(USER_STORE.get_user("admin"))
    assert admin is not None
    assert admin.api_key
    admin_resp = client.get(
        "/api/v1/social-governor/decisions?asset_class=EQUITY&regime=risk_off&limit=10",
        headers={"X-API-Key": admin.api_key},
    )
    assert admin_resp.status_code == 200
    body = admin_resp.json()
    assert body["count"] == 1
    assert body["events"][0]["asset_class"] == "EQUITY"
