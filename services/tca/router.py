"""
services/tca/router.py - Transaction Cost Analysis service router stub
"""
from fastapi import APIRouter

router = APIRouter(prefix="/tca", tags=["Transaction Cost Analysis"])

@router.get("/status")
async def get_status():
    """TCA status endpoint (stub)."""
    return {"status": "ok", "message": "TCA service coming soon"}
