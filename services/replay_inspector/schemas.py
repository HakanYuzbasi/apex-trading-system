"""Schemas for replay inspection responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReplayTimelineEvent(BaseModel):
    timestamp: str
    event_type: str
    symbol: str
    asset_class: str
    hash: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReplayGovernorPolicySnapshot(BaseModel):
    policy_key: str
    policy_id: str
    version: str
    asset_class: str
    regime: str
    created_at: str = ""
    observed_tier: str = ""
    tier_controls: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class ReplayLiquidationProgress(BaseModel):
    symbol: str
    status: str
    plan_id: str = ""
    plan_epoch: int = 0
    planned_reduction_qty: float = 0.0
    executed_reduction_qty: float = 0.0
    remaining_qty: float = 0.0
    progress_pct: float = 0.0
    target_reduction_pct: float = 0.0
    initial_position_qty: float = 0.0
    expected_stress_pnl: float = 0.0
    remaining_stress_pnl: Optional[float] = None
    remaining_stress_return: Optional[float] = None
    worst_scenario_id: str = ""
    worst_scenario_name: str = ""
    breach_event: Optional[ReplayTimelineEvent] = None
    plan_event: Optional[ReplayTimelineEvent] = None


class ReplayPlanAudit(BaseModel):
    plan_id: str
    plan_epoch: int = 0
    started_at: Optional[str] = None
    worst_scenario_id: str = ""
    worst_scenario_name: str = ""
    candidate_symbols: List[str] = Field(default_factory=list)
    completed_symbols: int = 0
    in_progress_symbols: int = 0
    planned_symbols: int = 0
    breach_event: Optional[ReplayTimelineEvent] = None
    plan_event: Optional[ReplayTimelineEvent] = None


class ReplayChain(BaseModel):
    chain_id: str
    symbol: str
    asset_class: str
    chain_kind: str
    started_at: str
    completed_at: Optional[str] = None
    final_status: str
    terminal_reason: str = ""
    signal_event: Optional[ReplayTimelineEvent] = None
    risk_events: List[ReplayTimelineEvent] = Field(default_factory=list)
    order_events: List[ReplayTimelineEvent] = Field(default_factory=list)
    position_events: List[ReplayTimelineEvent] = Field(default_factory=list)
    stress_events: List[ReplayTimelineEvent] = Field(default_factory=list)
    liquidation_progress: Optional[ReplayLiquidationProgress] = None
    governor_policy: Optional[ReplayGovernorPolicySnapshot] = None


class ReplayInspectionSummary(BaseModel):
    symbol: str
    asset_class: Optional[str] = None
    total_events: int
    total_chains: int
    blocked_chains: int
    filled_chains: int
    open_chains: int
    stress_liquidation_chains: int = 0
    latest_event_at: Optional[str] = None


class ReplayInspectionResponse(BaseModel):
    mode: str = "symbol"
    symbol: str
    days: int
    limit: int
    summary: ReplayInspectionSummary
    latest_chain: Optional[ReplayChain] = None
    chains: List[ReplayChain] = Field(default_factory=list)
    raw_events: List[ReplayTimelineEvent] = Field(default_factory=list)
    plan_audit: Optional[ReplayPlanAudit] = None
