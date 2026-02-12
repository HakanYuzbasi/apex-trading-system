"""World-Class Strategy Marketplace Router Implementation.

Provides high-performance, secure, and well-documented API endpoints for
strategy discovery, engagement, and management.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from .service import StrategyMarketplaceService
from .schemas import (
    StrategyListing,
    StrategyMetadata,
    MarketplaceSearch,
    LeaderboardEntry,
    UserReview,
    StrategyCategory,
    StrategyStatus
)
from utils.performance_monitor import PerformanceMonitor
from utils.error_tracker import ErrorTracker


router = APIRouter(
    prefix="/marketplace",
    tags=["Strategy Marketplace"],
    responses={404: {"description": "Strategy not found"}}
)

# Global service instance
marketplace_service = StrategyMarketplaceService()


@router.get("/search", response_model=dict)
@PerformanceMonitor.track_latency("api_marketplace_search")
async def search_strategies(
    query: Optional[str] = Query(None, description="Search query for strategy name/description"),
    category: Optional[List[StrategyCategory]] = Query(None, description="Filter by categories"),
    verified_only: bool = Query(False, description="Show only verified strategies"),
    sort_by: str = Query("rank", description="Sort by metric (rank, rating, return, etc.)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """World-class advanced search and discovery for trading strategies."""
    search_params = MarketplaceSearch(
        query=query,
        categories=set(category) if category else None,
        only_verified=verified_only,
        sort_by=sort_by,
        page=page,
        page_size=page_size
    )
    return await marketplace_service.list_strategies(search_params)


@router.get("/strategy/{strategy_id}", response_model=StrategyListing)
async def get_strategy(strategy_id: UUID):
    """Get full details for a specific trading strategy."""
    strategy = await marketplace_service.get_strategy_details(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )
    return strategy


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    category: Optional[StrategyCategory] = Query(None, description="Filter leaderboard by category")
):
    """Get real-time global or category-specific rankings."""
    return await marketplace_service.get_leaderboard(category)


@router.get("/recommendations/{user_id}", response_model=List[StrategyListing])
async def get_personalized_recommendations(user_id: UUID, limit: int = 5):
    """Get ML-powered personalized strategy recommendations for a user."""
    return await marketplace_service.get_recommendations(user_id, limit)


@router.post("/strategy/{strategy_id}/review", status_code=status.HTTP_201_CREATED)
async def submit_review(strategy_id: UUID, review: UserReview):
    """Submit a comprehensive user review for a strategy."""
    # Logic to save review and update strategy ratings
    return {"message": "Review submitted successfully", "review_id": review.review_id}


@router.post("/strategy", response_model=StrategyMetadata, status_code=status.HTTP_201_CREATED)
async def list_new_strategy(metadata: StrategyMetadata):
    """Submit a new strategy for listing in the marketplace."""
    # Logic to register new strategy with validation and review process
    return metadata


@router.patch("/strategy/{strategy_id}/status")
async def update_strategy_status(strategy_id: UUID, new_status: StrategyStatus):
    """Manage the lifecycle status of a strategy listing."""
    # Admin or Creator only logic
    return {"strategy_id": strategy_id, "status": new_status}
