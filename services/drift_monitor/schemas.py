"""Pydantic schemas for Drift Monitor API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class WindowStatsResponse(BaseModel):
    """A single evaluation window snapshot."""

    window_id: int
    n_obs: int
    ic: float
    hit_rate: float
    med_confidence: float
    health: str
    timestamp: str


class DriftStatusResponse(BaseModel):
    """Current model health status (from ModelDriftMonitor.get_status)."""

    health: str
    should_retrain: bool
    ic_current: float
    ic_trend: float
    hit_rate_current: float
    med_confidence: float
    consecutive_degraded: int
    total_windows: int
    pending_obs: int
    last_updated: str


class DriftReportResponse(BaseModel):
    """Detailed drift report including window history (from ModelDriftMonitor.get_report)."""

    health: str
    should_retrain: bool
    ic_current: float
    ic_trend: float
    hit_rate_current: float
    med_confidence: float
    consecutive_degraded: int
    total_windows: int
    pending_obs: int
    last_updated: str
    window_history: List[WindowStatsResponse] = []


class FeatureDriftSummaryResponse(BaseModel):
    """Summary of recent feature drift events (from FeatureDriftDetector.get_drift_summary)."""

    total_drift_events: int
    unique_features_drifted: int
    drift_by_feature: Dict[str, int] = {}
    days: int
    threshold_sigma: float
