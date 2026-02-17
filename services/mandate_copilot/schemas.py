"""Schemas for mandate copilot policy recommendations and workflow pack."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationMode(str, Enum):
    POLICY_ONLY = "POLICY_ONLY"
    ORDERS_REQUIRE_APPROVAL = "ORDERS_REQUIRE_APPROVAL"
    ORDERS_AUTO_EXECUTE = "ORDERS_AUTO_EXECUTE"


class FeasibilityBand(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SuitabilityProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class MandateLifecycleStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    PAPER_LIVE = "paper_live"
    RETIRED = "retired"


class MandateEvaluationRequest(BaseModel):
    """User mandate request for AI feasibility assessment."""

    intent: str = Field(
        default="",
        max_length=500,
        description="Natural language mandate request.",
    )
    target_return_pct: Optional[float] = Field(default=None, ge=0.1, le=200.0)
    horizon_days: Optional[int] = Field(default=None, ge=5, le=365)
    sectors: List[str] = Field(default_factory=list)
    asset_classes: List[str] = Field(default_factory=lambda: ["equity"])
    suitability_profile: SuitabilityProfile = SuitabilityProfile.BALANCED
    max_drawdown_pct: Optional[float] = Field(default=None, ge=5.0, le=50.0)
    include_options: bool = False
    recommendation_mode: RecommendationMode = RecommendationMode.POLICY_ONLY


class RegimeConfidenceInterval(BaseModel):
    regime: str
    lower: float
    upper: float


class StressNarrative(BaseModel):
    scenario: str
    quant_shock: str
    projected_impact_pct: float
    mitigation: str


class MandatePolicyRecommendation(BaseModel):
    sleeve_allocations: Dict[str, float] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict)
    execution_constraints: Dict[str, float] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class MandateEvaluationResponse(BaseModel):
    request_id: str
    recommendation_mode: RecommendationMode
    feasible: bool
    feasibility_band: FeasibilityBand
    probability_target_hit: float
    confidence: float
    expected_max_drawdown_pct: float
    expected_cvar95_pct: float
    expected_sortino: float
    rationale: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parsed_mandate: Dict[str, object] = Field(default_factory=dict)
    policy: MandatePolicyRecommendation
    stress_narratives: List[StressNarrative] = Field(default_factory=list)
    confidence_intervals_by_regime: List[RegimeConfidenceInterval] = Field(default_factory=list)
    disclaimer: str


class MandateAuditEvent(BaseModel):
    timestamp: str
    event: str
    model_version: str
    user_id: str
    tier: str
    recommendation_mode: str
    request: Dict[str, object] = Field(default_factory=dict)
    response_summary: Dict[str, object] = Field(default_factory=dict)
    output_hash: str


class SleeveCalibrationRow(BaseModel):
    sleeve: str
    predictions: int
    predicted_hit_rate: float
    realized_hit_rate: float
    calibration_gap: float
    threshold_abs_gap: float
    within_threshold: bool
    data_quality: str


class MandateCalibrationResponse(BaseModel):
    generated_at: str
    lookback_events: int
    rows: List[SleeveCalibrationRow] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class WorkflowSignoff(BaseModel):
    role: str
    approved_by: str = ""
    approved_at: str = ""
    note: str = ""
    approved: bool = False


class MandateWorkflowPack(BaseModel):
    workflow_id: str
    request_id: str
    created_at: str
    created_by: str
    tier: str
    status: MandateLifecycleStatus
    recommendation_mode: RecommendationMode
    feasibility_band: FeasibilityBand
    probability_target_hit: float
    expected_max_drawdown_pct: float
    required_signoffs: List[str] = Field(default_factory=lambda: ["pm", "compliance"])
    signoffs: Dict[str, WorkflowSignoff] = Field(default_factory=dict)
    policy_snapshot: Dict[str, object] = Field(default_factory=dict)
    risk_disclosure: str
    execution_enabled: bool = False
    notes: List[str] = Field(default_factory=list)


class WorkflowInitiateRequest(BaseModel):
    request_id: str
    note: str = ""


class WorkflowStatusUpdateRequest(BaseModel):
    status: MandateLifecycleStatus
    note: str = ""


class WorkflowSignoffRequest(BaseModel):
    role: str
    note: str = ""


class MonthlyDriftRow(BaseModel):
    sleeve: str
    mean_abs_gap: float
    max_abs_gap: float
    threshold_abs_gap: float
    breach_count: int


class PolicyChangeEvent(BaseModel):
    timestamp: str
    workflow_id: str
    actor: str
    from_status: str
    to_status: str
    note: str = ""


class MonthlyModelRiskReport(BaseModel):
    month: str
    generated_at: str
    drift_rows: List[MonthlyDriftRow] = Field(default_factory=list)
    miss_reasons: Dict[str, int] = Field(default_factory=dict)
    policy_changes: List[PolicyChangeEvent] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

