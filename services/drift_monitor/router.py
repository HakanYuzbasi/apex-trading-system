"""
services/drift_monitor/router.py - Drift Monitor API Router (Stub)

This module will contain endpoints for feature drift monitoring.
Currently a placeholder to prevent import warnings.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/drift-monitor", tags=["drift-monitor"])

# Future endpoints:
# - GET /drift-monitor/features - Feature drift metrics
# - GET /drift-monitor/alerts - Drift alerts
# - POST /drift-monitor/baseline - Set baseline distribution
