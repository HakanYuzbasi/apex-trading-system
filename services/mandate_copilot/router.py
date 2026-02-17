"""API router for policy-first mandate copilot and workflow pack."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.common.subscription import require_feature
from services.mandate_copilot.schemas import (
    MandateCalibrationResponse,
    MandateEvaluationRequest,
    MandateEvaluationResponse,
    MandateLifecycleStatus,
    MandateWorkflowPack,
    MonthlyModelRiskReport,
    RecommendationMode,
    WorkflowInitiateRequest,
    WorkflowSignoffRequest,
    WorkflowStatusUpdateRequest,
)
from services.mandate_copilot.service import MandateCopilotService

router = APIRouter(prefix="/api/v1/mandate-copilot", tags=["mandate-copilot"])
_SERVICE = MandateCopilotService()


def _normalized_roles(request: Request, user) -> list[str]:
    request_roles = getattr(request.state, "roles", None) or []
    user_roles = getattr(user, "roles", None) or []
    source = request_roles if request_roles else user_roles
    return [str(role).strip().lower() for role in source]


def require_feature_or_admin(feature_key: str):
    """Allow admins unconditionally; enforce feature gating for non-admin users."""
    feature_gate = require_feature(feature_key)

    async def _check(request: Request):
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if "admin" in _normalized_roles(request, user):
            return user
        return await feature_gate(request)

    return _check


@router.post("/evaluate", response_model=MandateEvaluationResponse)
async def evaluate_mandate(
    body: MandateEvaluationRequest,
    request: Request,
    user=Depends(require_feature("mandate_copilot_preview")),
):
    """Evaluate mandate feasibility and return policy-only recommendations."""
    if body.recommendation_mode != RecommendationMode.POLICY_ONLY:
        raise HTTPException(
            status_code=400,
            detail="POLICY_ONLY mode is enforced for this MVP. Order generation is disabled.",
        )

    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    tier = str(getattr(request.state, "tier", "free"))

    try:
        return _SERVICE.evaluate(body, user_id=str(user_id), tier=tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/audit")
async def get_mandate_audit(
    request: Request,
    user=Depends(require_feature("mandate_copilot_preview")),
    limit: int = Query(25, ge=1, le=200),
):
    """Get recent mandate copilot audit events."""
    roles = _normalized_roles(request, user)
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    events = _SERVICE.load_recent_audit(limit=limit)
    if "admin" in roles:
        return {"events": events}
    scoped = [e for e in events if str(e.get("user_id")) == str(user_id)]
    return {"events": scoped}


@router.get("/calibration", response_model=MandateCalibrationResponse)
async def get_mandate_calibration(
    request: Request,
    user=Depends(require_feature("mandate_copilot_preview")),
    limit: int = Query(200, ge=10, le=1000),
):
    """Get lightweight predicted-vs-realized calibration snapshot by sleeve."""
    _ = request, user
    return _SERVICE.build_calibration_snapshot(limit=limit)


@router.post("/workflows/initiate", response_model=MandateWorkflowPack)
async def initiate_mandate_workflow(
    body: WorkflowInitiateRequest,
    request: Request,
    user=Depends(require_feature_or_admin("mandate_workflow_pack")),
):
    """Initiate a mandate workflow pack (paywalled, with explicit admin override)."""
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    tier = str(getattr(request.state, "tier", "free"))
    try:
        return _SERVICE.initiate_workflow(
            request_id=body.request_id,
            user_id=str(user_id),
            tier=tier,
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/workflows", response_model=list[MandateWorkflowPack])
async def list_mandate_workflows(
    request: Request,
    user=Depends(require_feature_or_admin("mandate_workflow_pack")),
    limit: int = Query(50, ge=1, le=500),
):
    roles = _normalized_roles(request, user)
    is_admin = "admin" in roles
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None) or ""
    return _SERVICE.list_workflows(limit=limit, user_id=str(user_id), is_admin=is_admin)


@router.post("/workflows/{workflow_id}/signoff", response_model=MandateWorkflowPack)
async def signoff_mandate_workflow(
    workflow_id: str,
    body: WorkflowSignoffRequest,
    request: Request,
    user=Depends(require_feature_or_admin("mandate_workflow_pack")),
):
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    username = getattr(user, "username", None) or str(user_id or "unknown")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    roles = _normalized_roles(request, user)
    role = body.role.strip().lower()
    if role not in {"pm", "compliance"}:
        raise HTTPException(status_code=400, detail="Role must be 'pm' or 'compliance'.")
    if role == "compliance" and "admin" not in roles:
        raise HTTPException(status_code=403, detail="Compliance sign-off requires admin/compliance role.")
    try:
        return _SERVICE.signoff_workflow(
            workflow_id=workflow_id,
            role=role,
            actor=str(username),
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/workflows/{workflow_id}/status", response_model=MandateWorkflowPack)
async def update_mandate_workflow_status(
    workflow_id: str,
    body: WorkflowStatusUpdateRequest,
    request: Request,
    user=Depends(require_feature_or_admin("mandate_workflow_pack")),
):
    _ = request
    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    username = getattr(user, "username", None) or str(user_id or "unknown")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if body.status == MandateLifecycleStatus.PAPER_LIVE and body.note.strip() == "":
        raise HTTPException(status_code=400, detail="Provide an explicit note before moving to paper_live.")
    try:
        return _SERVICE.update_workflow_status(
            workflow_id=workflow_id,
            status=body.status,
            actor=str(username),
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/reports/monthly", response_model=MonthlyModelRiskReport)
async def get_monthly_mandate_model_risk_report(
    request: Request,
    user=Depends(require_feature_or_admin("mandate_model_risk_report")),
    month: str | None = Query(None),
    lookback: int = Query(1000, ge=50, le=10000),
):
    """Monthly model risk report with drift, miss reasons, and policy changes."""
    _ = request, user
    return _SERVICE.build_monthly_model_risk_report(month=month, lookback=lookback)
