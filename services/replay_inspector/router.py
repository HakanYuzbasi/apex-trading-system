"""API router for symbol and liquidation-plan replay inspection."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_user
from config import ApexConfig
from services.replay_inspector.schemas import ReplayInspectionResponse
from services.replay_inspector.service import ReplayInspectorService

router = APIRouter(prefix="/api/v1/replay-inspector", tags=["replay-inspector"])


def _tenant_root(user_id: str) -> str:
    tenant = str(user_id or "").strip()
    if not tenant or tenant == "default":
        return str(ApexConfig.DATA_DIR)
    return str(ApexConfig.DATA_DIR / "users" / tenant)


@router.get("/plan/{plan_id}", response_model=ReplayInspectionResponse)
async def inspect_plan_replay(
    plan_id: str,
    limit: int = Query(500, ge=1, le=5000),
    days: int = Query(7, ge=1, le=60),
    include_raw: bool = Query(False),
    user=Depends(require_user),
):
    """Inspect a liquidation plan across symbols for the authenticated tenant."""
    normalized_plan_id = str(plan_id or "").strip()
    if not normalized_plan_id:
        raise HTTPException(status_code=400, detail="plan_id is required")

    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    service = ReplayInspectorService(_tenant_root(str(user_id)))
    response = service.inspect_plan(
        plan_id=normalized_plan_id,
        limit=limit,
        days=days,
        include_raw=include_raw,
    )
    if response is None:
        raise HTTPException(status_code=404, detail="liquidation plan not found")
    return response


@router.get("/{symbol:path}", response_model=ReplayInspectionResponse)
async def inspect_symbol_replay(
    symbol: str,
    limit: int = Query(500, ge=1, le=5000),
    days: int = Query(7, ge=1, le=60),
    include_raw: bool = Query(False),
    user=Depends(require_user),
):
    """Inspect a symbol's journaled decision chain for the authenticated tenant."""
    normalized_symbol = str(symbol or "").strip()
    if not normalized_symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    user_id = getattr(user, "user_id", None) or getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    service = ReplayInspectorService(_tenant_root(str(user_id)))
    return service.inspect_symbol(
        symbol=normalized_symbol,
        limit=limit,
        days=days,
        include_raw=include_raw,
    )
