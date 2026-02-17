from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from services.mandate_copilot import service as mandate_service_module
from services.mandate_copilot.router import router as mandate_router
from services.mandate_copilot.schemas import MandateEvaluationRequest, RecommendationMode
from services.mandate_copilot.service import MandateCopilotService


def _build_test_app(
    *,
    user_id: str = "user-1",
    roles: list[str] | None = None,
    tier: str = "free",
) -> FastAPI:
    app = FastAPI()
    user_roles = roles or ["user"]

    @app.middleware("http")
    async def attach_user(request: Request, call_next):
        request.state.user = SimpleNamespace(user_id=user_id, username=user_id, roles=user_roles, tier=tier)
        request.state.user_id = user_id
        request.state.roles = user_roles
        request.state.tier = tier
        return await call_next(request)

    app.include_router(mandate_router)
    return app


def test_service_parses_intent_and_returns_policy_only_response():
    service = MandateCopilotService()
    req = MandateEvaluationRequest(
        intent="I want to make 10% in the next two months by trading energy and tech sectors",
    )
    result = service.evaluate(req, user_id="u-test", tier="free")
    assert result.recommendation_mode == RecommendationMode.POLICY_ONLY
    assert result.parsed_mandate["target_return_pct"] == 10.0
    assert result.parsed_mandate["horizon_days"] == 60
    assert "technology" in result.parsed_mandate["sectors"]
    assert "energy" in result.parsed_mandate["sectors"]
    assert "order_generation=disabled" in result.policy.constraints


def test_service_rejects_non_policy_mode():
    service = MandateCopilotService()
    req = MandateEvaluationRequest(
        target_return_pct=8.0,
        horizon_days=45,
        recommendation_mode=RecommendationMode.ORDERS_REQUIRE_APPROVAL,
    )
    try:
        service.evaluate(req, user_id="u-test", tier="free")
        assert False, "Expected ValueError for non-policy recommendation mode"
    except ValueError as exc:
        assert "POLICY_ONLY" in str(exc)


def test_service_appends_audit_event(tmp_path, monkeypatch):
    audit_file = tmp_path / "mandate_audit.jsonl"
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", audit_file)

    service = MandateCopilotService()
    req = MandateEvaluationRequest(target_return_pct=7.5, horizon_days=75)
    _ = service.evaluate(req, user_id="u-audit", tier="pro")

    lines = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "mandate_evaluation"
    assert event["user_id"] == "u-audit"
    assert event["model_version"] == mandate_service_module.MODEL_VERSION
    assert len(event["output_hash"]) == 64


def test_router_enforces_policy_only_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", tmp_path / "router_audit.jsonl")
    client = TestClient(_build_test_app())
    payload = {
        "target_return_pct": 12.0,
        "horizon_days": 50,
        "recommendation_mode": "ORDERS_REQUIRE_APPROVAL",
    }
    response = client.post("/api/v1/mandate-copilot/evaluate", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert "POLICY_ONLY" in body["detail"]


def test_router_evaluates_mandate(tmp_path, monkeypatch):
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", tmp_path / "router_audit.jsonl")
    client = TestClient(_build_test_app())
    payload = {
        "intent": "10% in two months across energy and tech",
        "max_drawdown_pct": 15.0,
    }
    response = client.post("/api/v1/mandate-copilot/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_mode"] == "POLICY_ONLY"
    assert "probability_target_hit" in body
    assert "policy" in body
    assert "risk_limits" in body["policy"]


def test_calibration_snapshot_uses_audit_and_state(tmp_path, monkeypatch):
    audit_file = tmp_path / "audit.jsonl"
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(mandate_service_module, "STATE_FILE", state_file)

    events = [
        {
            "timestamp": "2026-02-13T00:00:00Z",
            "event": "mandate_evaluation",
            "model_version": "mandate-copilot-policy-v1",
            "user_id": "u-1",
            "tier": "free",
            "recommendation_mode": "POLICY_ONLY",
            "request": {"sectors": ["energy", "technology"]},
            "response_summary": {"probability_target_hit": 0.70},
            "output_hash": "a" * 64,
        }
    ]
    audit_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    state_file.write_text(
        json.dumps(
            {
                "performance_attribution": {
                    "by_sleeve": {
                        "energy_sleeve": {"trades": 25, "net_pnl": 1200},
                        "technology_sleeve": {"trades": 25, "net_pnl": -300},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    service = MandateCopilotService()
    snapshot = service.build_calibration_snapshot(limit=10)
    assert snapshot.lookback_events == 1
    rows = {row.sleeve: row for row in snapshot.rows}
    assert "energy_sleeve" in rows
    assert rows["energy_sleeve"].predicted_hit_rate == 0.7


def test_workflow_pack_requires_paid_or_admin(tmp_path, monkeypatch):
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", tmp_path / "audit.jsonl")
    monkeypatch.setattr(mandate_service_module, "WORKFLOW_FILE", tmp_path / "workflows.json")
    monkeypatch.setattr(mandate_service_module, "POLICY_CHANGE_FILE", tmp_path / "policy.jsonl")
    client = TestClient(_build_test_app(tier="free", roles=["user"]))

    evaluate_response = client.post(
        "/api/v1/mandate-copilot/evaluate",
        json={"intent": "10% in two months energy and tech"},
    )
    assert evaluate_response.status_code == 200
    request_id = evaluate_response.json()["request_id"]

    workflow_response = client.post(
        "/api/v1/mandate-copilot/workflows/initiate",
        json={"request_id": request_id, "note": "test"},
    )
    assert workflow_response.status_code == 403


def test_admin_can_run_workflow_lifecycle_and_monthly_report(tmp_path, monkeypatch):
    monkeypatch.setattr(mandate_service_module, "AUDIT_FILE", tmp_path / "audit.jsonl")
    monkeypatch.setattr(mandate_service_module, "WORKFLOW_FILE", tmp_path / "workflows.json")
    monkeypatch.setattr(mandate_service_module, "POLICY_CHANGE_FILE", tmp_path / "policy.jsonl")
    monkeypatch.setattr(mandate_service_module, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text('{"performance_attribution":{"by_sleeve":{}}}', encoding="utf-8")

    client = TestClient(_build_test_app(tier="free", roles=["admin", "user"], user_id="admin-1"))

    evaluate_response = client.post(
        "/api/v1/mandate-copilot/evaluate",
        json={"intent": "10% in two months energy and tech"},
    )
    assert evaluate_response.status_code == 200
    request_id = evaluate_response.json()["request_id"]

    initiate_response = client.post(
        "/api/v1/mandate-copilot/workflows/initiate",
        json={"request_id": request_id, "note": "init"},
    )
    assert initiate_response.status_code == 200
    workflow_id = initiate_response.json()["workflow_id"]

    pm_signoff = client.post(
        f"/api/v1/mandate-copilot/workflows/{workflow_id}/signoff",
        json={"role": "pm", "note": "pm ok"},
    )
    assert pm_signoff.status_code == 200

    compliance_signoff = client.post(
        f"/api/v1/mandate-copilot/workflows/{workflow_id}/signoff",
        json={"role": "compliance", "note": "compliance ok"},
    )
    assert compliance_signoff.status_code == 200
    assert compliance_signoff.json()["status"] == "approved"

    promote_response = client.post(
        f"/api/v1/mandate-copilot/workflows/{workflow_id}/status",
        json={"status": "paper_live", "note": "paper launch"},
    )
    assert promote_response.status_code == 200
    assert promote_response.json()["status"] == "paper_live"

    report_response = client.get("/api/v1/mandate-copilot/reports/monthly?lookback=100")
    assert report_response.status_code == 200
    assert "drift_rows" in report_response.json()
