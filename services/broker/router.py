from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from datetime import datetime, timezone
import logging

from api.auth import require_user, User
from core.exceptions import ApexBrokerError
from models.broker import BrokerConnection, BrokerType
from services.broker.service import broker_service

router = APIRouter(prefix="/brokers", tags=["brokers"])
portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])

logger = logging.getLogger(__name__)

class CreateConnectionRequest(BaseModel):
    broker_type: BrokerType
    name: str
    environment: str = "paper"
    credentials: dict
    client_id: Optional[int] = None

class BrokerConnectionResponse(BaseModel):
    id: str
    user_id: str
    broker_type: BrokerType
    name: str
    environment: str
    client_id: Optional[int]
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True

class PortfolioBalanceResponse(BaseModel):
    total_equity: float
    last_updated: datetime
    breakdown: List[dict] = []

class AggregatedPositionItem(BaseModel):
    source: str
    source_id: str
    broker_type: str
    symbol: str
    qty: float
    side: str
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float

class PortfolioSourceItem(BaseModel):
    id: str
    name: str
    broker_type: str
    environment: str

@router.post("/connect", response_model=BrokerConnectionResponse, status_code=status.HTTP_201_CREATED)
async def connect_broker(
    request: CreateConnectionRequest,
    user: User = Depends(require_user)
):
    """Connect a new broker account. Validates credentials before saving."""
    try:
        connection = await broker_service.create_connection(
            user_id=user.user_id,
            broker_type=request.broker_type,
            name=request.name,
            credentials=request.credentials,
            environment=request.environment,
            client_id=request.client_id
        )
        return connection
    except ApexBrokerError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to connect broker: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("", response_model=List[BrokerConnectionResponse])
async def list_brokers(user: User = Depends(require_user)):
    """List all configured broker connections."""
    return await broker_service.list_connections(user.user_id)

@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_broker(
    connection_id: str,
    user: User = Depends(require_user)
):
    """Delete a broker connection."""
    conn = await broker_service.get_connection(connection_id, user.user_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    deleted = await broker_service.delete_connection(connection_id, user.user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete connection")

@router.patch("/{connection_id}/toggle", response_model=BrokerConnectionResponse)
async def toggle_broker(
    connection_id: str,
    user: User = Depends(require_user)
):
    """Toggle a broker connection active/inactive (persisted to disk)."""
    conn = await broker_service.get_connection(connection_id, user.user_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    updated = await broker_service.toggle_connection(connection_id, user.user_id)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Toggle failed")
    return updated

@portfolio_router.get("/balance", response_model=PortfolioBalanceResponse)
async def get_portfolio_balance(user: User = Depends(require_user)):
    """Get aggregated portfolio balance with per-source breakdown."""
    try:
        snapshot = await broker_service.get_tenant_equity_snapshot(user.user_id)

        return {
            "total_equity": snapshot["total_equity"],
            "last_updated": datetime.now(timezone.utc),
            "breakdown": snapshot["breakdown"],
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio balance: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch balance")

@portfolio_router.get("/positions", response_model=List[AggregatedPositionItem])
async def get_aggregated_positions(
    source_id: Optional[str] = None,
    user: User = Depends(require_user)
):
    """
    Get aggregated positions across all active broker connections.
    Pass ?source_id=<id> to filter to a single account.
    """
    try:
        positions = await broker_service.get_positions(user.user_id)
        if source_id:
            positions = [p for p in positions if p["source_id"] == source_id]
        return positions
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch positions")

@portfolio_router.get("/sources", response_model=List[PortfolioSourceItem])
async def get_portfolio_sources(user: User = Depends(require_user)):
    """Returns active broker connections for the account selector dropdown."""
    try:
        return await broker_service.list_connection_sources(user.user_id)
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch sources")
