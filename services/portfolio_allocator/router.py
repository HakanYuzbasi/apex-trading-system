"""
services/portfolio_allocator/router.py - Portfolio allocator service router stub
"""
from fastapi import APIRouter

router = APIRouter(prefix="/portfolio-allocator", tags=["Portfolio Allocator"])

@router.get("/status")
async def get_status():
    """Portfolio allocator status endpoint (stub)."""
    return {"status": "ok", "message": "Portfolio allocator coming soon"}
