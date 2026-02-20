"""Policy-first mandate copilot service with isolated workflow/audit storage."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

from config import ApexConfig
from services.mandate_copilot.schemas import (
    FeasibilityBand,
    MandateCalibrationResponse,
    MandateEvaluationRequest,
    MandateEvaluationResponse,
    MandateLifecycleStatus,
    MandatePolicyRecommendation,
    MandateWorkflowPack,
    MonthlyDriftRow,
    MonthlyModelRiskReport,
    PolicyChangeEvent,
    RecommendationMode,
    RegimeConfidenceInterval,
    SleeveCalibrationRow,
    StressNarrative,
    SuitabilityProfile,
    WorkflowSignoff,
)

AUDIT_FILE = ApexConfig.DATA_DIR / "mandate_copilot_audit.jsonl"
WORKFLOW_FILE = ApexConfig.DATA_DIR / "mandate_workflow_packs.json"
POLICY_CHANGE_FILE = ApexConfig.DATA_DIR / "mandate_policy_changes.jsonl"
STATE_FILE = ApexConfig.DATA_DIR / "trading_state.json"
MODEL_VERSION = "mandate-copilot-policy-v2"

_AUDIT_LOCK = threading.Lock()
_WORKFLOW_LOCK = threading.Lock()
_POLICY_CHANGE_LOCK = threading.Lock()

_SECTOR_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "technology": ("technology", "tech", "software", "semiconductor"),
    "energy": ("energy", "oil", "gas", "xle"),
    "healthcare": ("healthcare", "biotech", "pharma"),
    "financials": ("financial", "bank", "insurance"),
    "industrials": ("industrial", "manufacturing", "transport"),
    "consumer": ("consumer", "retail"),
}

_ASSET_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "equity": ("equity", "stock", "stocks", "etf"),
    "fx": ("fx", "forex", "currency"),
    "crypto": ("crypto", "btc", "eth"),
}

_SUITABILITY_DEFAULT_MAX_DD: Dict[SuitabilityProfile, float] = {
    SuitabilityProfile.CONSERVATIVE: 10.0,
    SuitabilityProfile.BALANCED: 15.0,
    SuitabilityProfile.AGGRESSIVE: 25.0,
}

_VALID_TRANSITIONS: Dict[MandateLifecycleStatus, set[MandateLifecycleStatus]] = {
    MandateLifecycleStatus.DRAFT: {MandateLifecycleStatus.APPROVED, MandateLifecycleStatus.RETIRED},
    MandateLifecycleStatus.APPROVED: {MandateLifecycleStatus.PAPER_LIVE, MandateLifecycleStatus.RETIRED},
    MandateLifecycleStatus.PAPER_LIVE: {MandateLifecycleStatus.RETIRED},
    MandateLifecycleStatus.RETIRED: set(),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_horizon_days(intent: str) -> int | None:
    lower = intent.lower()
    month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:month|months|mo)\b", lower)
    if month_match:
        return max(5, int(round(float(month_match.group(1)) * 30)))
    week_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:week|weeks|wk)\b", lower)
    if week_match:
        return max(5, int(round(float(week_match.group(1)) * 7)))
    day_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:day|days|d)\b", lower)
    if day_match:
        return max(5, int(round(float(day_match.group(1)))))
    return None


def _parse_target_return(intent: str) -> float | None:
    lower = intent.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
    if not match:
        return None
    return float(match.group(1))


def _extract_keywords(intent: str, mapping: Dict[str, Tuple[str, ...]]) -> List[str]:
    lower = intent.lower()
    hits: List[str] = []
    for canonical, words in mapping.items():
        if any(word in lower for word in words):
            hits.append(canonical)
    return hits


def _normalize_text_list(values: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for value in values:
        token = str(value).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _asset_class_for_sleeve(sleeve: str) -> str:
    lower = sleeve.lower()
    if "crypto" in lower:
        return "crypto"
    if "fx" in lower or "forex" in lower:
        return "fx"
    return "equity"


def _sleeve_threshold_abs_gap(sleeve: str) -> float:
    asset = _asset_class_for_sleeve(sleeve)
    if asset == "equity":
        return 0.08
    if asset == "fx":
        return 0.12
    if asset == "crypto":
        return 0.18
    return 0.10


class MandateCopilotService:
    """Deterministic mandate assessment engine with policy-only outputs."""

    def evaluate(self, req: MandateEvaluationRequest, user_id: str, tier: str) -> MandateEvaluationResponse:
        if req.recommendation_mode != RecommendationMode.POLICY_ONLY:
            raise ValueError("Only POLICY_ONLY mode is enabled in this deployment.")

        parsed = self._parse_request(req)
        response = self._score_and_recommend(parsed)
        self._append_audit_event(
            user_id=user_id,
            tier=tier,
            request_payload=req.model_dump(mode="json"),
            response_payload=response.model_dump(mode="json"),
        )
        return response

    def initiate_workflow(self, request_id: str, user_id: str, tier: str, note: str = "") -> MandateWorkflowPack:
        source_event = self._find_audit_event_by_request_id(request_id)
        if source_event is None:
            raise ValueError(f"Mandate assessment '{request_id}' was not found in audit history.")

        existing = self._find_workflow_by_request_id(request_id)
        if existing is not None:
            return existing

        summary = source_event.get("response_summary", {}) or {}
        response_band = FeasibilityBand(str(summary.get("feasibility_band", "red")))
        pack = MandateWorkflowPack(
            workflow_id=f"mwf-{uuid4().hex[:12]}",
            request_id=request_id,
            created_at=_utc_now_iso(),
            created_by=str(user_id),
            tier=str(tier),
            status=MandateLifecycleStatus.DRAFT,
            recommendation_mode=RecommendationMode.POLICY_ONLY,
            feasibility_band=response_band,
            probability_target_hit=float(summary.get("probability_target_hit", 0.0) or 0.0),
            expected_max_drawdown_pct=float(summary.get("expected_max_drawdown_pct", 0.0) or 0.0),
            signoffs={
                "pm": WorkflowSignoff(role="pm"),
                "compliance": WorkflowSignoff(role="compliance"),
            },
            policy_snapshot=dict(source_event.get("request", {}) or {}),
            risk_disclosure=(
                "Policy recommendation only. No order generation or live capital routing is enabled "
                "by this workflow pack."
            ),
            execution_enabled=False,
            notes=[note] if note else [],
        )
        workflows = self._load_workflows()
        workflows.append(pack.model_dump(mode="json"))
        self._save_workflows(workflows)
        self._append_policy_change_event(
            workflow_id=pack.workflow_id,
            actor=str(user_id),
            from_status="none",
            to_status=pack.status.value,
            note=note,
        )
        return pack

    def list_workflows(self, limit: int = 50, user_id: str = "", is_admin: bool = False) -> List[MandateWorkflowPack]:
        workflows = self._load_workflows()
        reversed_rows = list(reversed(workflows))
        scoped: List[dict] = []
        for row in reversed_rows:
            if is_admin or str(row.get("created_by", "")) == str(user_id):
                scoped.append(row)
            if len(scoped) >= limit:
                break
        result: List[MandateWorkflowPack] = []
        for row in scoped:
            try:
                result.append(MandateWorkflowPack.model_validate(row))
            except Exception as exc:
                logger.warning("Skipping malformed workflow row: %s", exc)
                continue
        return result

    def signoff_workflow(self, workflow_id: str, role: str, actor: str, note: str = "") -> MandateWorkflowPack:
        role_key = str(role).strip().lower()
        workflows = self._load_workflows()
        idx = self._find_workflow_index(workflows, workflow_id)
        if idx < 0:
            raise ValueError(f"Workflow '{workflow_id}' was not found.")

        pack = MandateWorkflowPack.model_validate(workflows[idx])
        if role_key not in pack.required_signoffs:
            raise ValueError(f"Unsupported sign-off role '{role_key}'.")

        signoff = WorkflowSignoff(
            role=role_key,
            approved_by=actor,
            approved_at=_utc_now_iso(),
            note=note,
            approved=True,
        )
        pack.signoffs[role_key] = signoff

        # Auto-promote draft -> approved once required sign-offs are complete.
        if (
            pack.status == MandateLifecycleStatus.DRAFT
            and all(pack.signoffs.get(req_role, WorkflowSignoff(role=req_role)).approved for req_role in pack.required_signoffs)
        ):
            old_status = pack.status
            pack.status = MandateLifecycleStatus.APPROVED
            self._append_policy_change_event(
                workflow_id=pack.workflow_id,
                actor=actor,
                from_status=old_status.value,
                to_status=pack.status.value,
                note="Auto-approved after required PM + compliance sign-offs.",
            )

        workflows[idx] = pack.model_dump(mode="json")
        self._save_workflows(workflows)
        return pack

    def update_workflow_status(self, workflow_id: str, status: MandateLifecycleStatus, actor: str, note: str = "") -> MandateWorkflowPack:
        workflows = self._load_workflows()
        idx = self._find_workflow_index(workflows, workflow_id)
        if idx < 0:
            raise ValueError(f"Workflow '{workflow_id}' was not found.")

        pack = MandateWorkflowPack.model_validate(workflows[idx])
        if status == pack.status:
            return pack
        allowed = _VALID_TRANSITIONS.get(pack.status, set())
        if status not in allowed:
            raise ValueError(f"Invalid lifecycle transition: {pack.status.value} -> {status.value}.")
        if status == MandateLifecycleStatus.PAPER_LIVE:
            if not all(pack.signoffs.get(req_role, WorkflowSignoff(role=req_role)).approved for req_role in pack.required_signoffs):
                raise ValueError("All required PM + compliance sign-offs must be completed before paper_live.")

        old_status = pack.status
        pack.status = status
        if note:
            pack.notes.append(note)

        workflows[idx] = pack.model_dump(mode="json")
        self._save_workflows(workflows)
        self._append_policy_change_event(
            workflow_id=pack.workflow_id,
            actor=actor,
            from_status=old_status.value,
            to_status=status.value,
            note=note,
        )
        return pack

    def load_recent_audit(self, limit: int = 50) -> List[dict]:
        if limit <= 0 or not Path(AUDIT_FILE).exists():
            return []
        with _AUDIT_LOCK:
            lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        events: List[dict] = []
        for line in reversed(lines[-limit:]):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def build_calibration_snapshot(self, limit: int = 200) -> MandateCalibrationResponse:
        events = self.load_recent_audit(limit=limit)
        by_sleeve_prediction_sum: Dict[str, float] = {}
        by_sleeve_prediction_count: Dict[str, int] = {}

        for event in events:
            response = event.get("response_summary", {}) or {}
            prob = float(response.get("probability_target_hit", 0.0) or 0.0)
            request_payload = event.get("request", {}) or {}
            sectors = request_payload.get("sectors", [])
            if not isinstance(sectors, list):
                sectors = []
            if not sectors:
                sectors = ["equities"]
            mapped_sleeves = [f"{str(sector).strip().lower()}_sleeve" for sector in sectors]
            for sleeve in mapped_sleeves:
                by_sleeve_prediction_sum[sleeve] = by_sleeve_prediction_sum.get(sleeve, 0.0) + prob
                by_sleeve_prediction_count[sleeve] = by_sleeve_prediction_count.get(sleeve, 0) + 1

        realized_hit_by_sleeve: Dict[str, float] = {}
        data_quality_by_sleeve: Dict[str, str] = {}
        try:
            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                attribution = state.get("performance_attribution", {}) if isinstance(state, dict) else {}
                by_sleeve = attribution.get("by_sleeve", {}) if isinstance(attribution, dict) else {}
                if isinstance(by_sleeve, dict):
                    for sleeve, metrics in by_sleeve.items():
                        if not isinstance(metrics, dict):
                            continue
                        trades = float(metrics.get("trades", 0) or 0)
                        net_pnl = float(metrics.get("net_pnl", 0) or 0)
                        realized_hit_by_sleeve[str(sleeve)] = 1.0 if net_pnl >= 0 else 0.0
                        data_quality_by_sleeve[str(sleeve)] = "limited_sample" if trades < 20 else "ok"
        except Exception as exc:
            logger.warning("Failed to load sleeve calibration data: %s", exc)

        sleeves = sorted(set(by_sleeve_prediction_count) | set(realized_hit_by_sleeve))
        rows: List[SleeveCalibrationRow] = []
        for sleeve in sleeves:
            count = by_sleeve_prediction_count.get(sleeve, 0)
            pred = (
                by_sleeve_prediction_sum[sleeve] / float(count)
                if count > 0 and sleeve in by_sleeve_prediction_sum
                else 0.0
            )
            realized = realized_hit_by_sleeve.get(sleeve, 0.0)
            gap = pred - realized
            threshold = _sleeve_threshold_abs_gap(sleeve)
            rows.append(
                SleeveCalibrationRow(
                    sleeve=sleeve,
                    predictions=count,
                    predicted_hit_rate=round(pred, 4),
                    realized_hit_rate=round(realized, 4),
                    calibration_gap=round(gap, 4),
                    threshold_abs_gap=round(threshold, 4),
                    within_threshold=abs(gap) <= threshold,
                    data_quality=data_quality_by_sleeve.get(
                        sleeve,
                        "proxy_realized_from_latest_attribution",
                    ),
                )
            )

        notes = [
            "Sleeve-specific calibration thresholds are tighter for equities and wider for crypto.",
            "Realized hit-rate is currently a proxy from latest sleeve net PnL sign until trade-level labels are stored.",
        ]
        return MandateCalibrationResponse(
            generated_at=_utc_now_iso(),
            lookback_events=len(events),
            rows=rows,
            notes=notes,
        )

    def build_monthly_model_risk_report(self, month: str | None = None, lookback: int = 1000) -> MonthlyModelRiskReport:
        target_month = month or datetime.now(timezone.utc).strftime("%Y-%m")
        calibration = self.build_calibration_snapshot(limit=lookback)

        drift_rows: List[MonthlyDriftRow] = []
        for row in calibration.rows:
            gap = abs(float(row.calibration_gap))
            drift_rows.append(
                MonthlyDriftRow(
                    sleeve=row.sleeve,
                    mean_abs_gap=round(gap, 4),
                    max_abs_gap=round(gap, 4),
                    threshold_abs_gap=row.threshold_abs_gap,
                    breach_count=1 if gap > row.threshold_abs_gap else 0,
                )
            )

        miss_reasons: Dict[str, int] = defaultdict(int)
        for pack in self.list_workflows(limit=lookback, is_admin=True):
            if not pack.created_at.startswith(target_month):
                continue
            if pack.status != MandateLifecycleStatus.RETIRED:
                continue
            reason = "unspecified"
            for note in reversed(pack.notes):
                if note:
                    reason = note.strip().split(";", 1)[0][:80]
                    break
            miss_reasons[reason] += 1

        policy_changes = self.load_policy_changes(limit=lookback, month=target_month)
        report_notes = [
            "Drift values currently reflect latest predicted-vs-proxy-realized gaps by sleeve.",
            "Miss reasons come from retired workflow notes and should be standardized over time.",
            "Workflow packs are isolated from live execution and do not route orders.",
        ]
        return MonthlyModelRiskReport(
            month=target_month,
            generated_at=_utc_now_iso(),
            drift_rows=drift_rows,
            miss_reasons=dict(miss_reasons),
            policy_changes=policy_changes,
            notes=report_notes,
        )

    def load_policy_changes(self, limit: int = 200, month: str | None = None) -> List[PolicyChangeEvent]:
        if limit <= 0 or not Path(POLICY_CHANGE_FILE).exists():
            return []
        with _POLICY_CHANGE_LOCK:
            lines = POLICY_CHANGE_FILE.read_text(encoding="utf-8").splitlines()
        events: List[PolicyChangeEvent] = []
        for line in reversed(lines[-limit:]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if month and not str(payload.get("timestamp", "")).startswith(month):
                continue
            try:
                events.append(PolicyChangeEvent.model_validate(payload))
            except Exception as exc:
                logger.warning("Skipping malformed policy change event: %s", exc)
                continue
        return events

    def _parse_request(self, req: MandateEvaluationRequest) -> dict:
        intent = req.intent.strip()
        intent_sectors = _extract_keywords(intent, _SECTOR_KEYWORDS)
        intent_assets = _extract_keywords(intent, _ASSET_KEYWORDS)

        target_return_pct = req.target_return_pct or _parse_target_return(intent) or 10.0
        horizon_days = req.horizon_days or _parse_horizon_days(intent) or 60
        sectors = _normalize_text_list(list(req.sectors) + intent_sectors)
        asset_classes = _normalize_text_list(list(req.asset_classes) + intent_assets)
        if not sectors:
            sectors = ["technology", "energy"]
        if not asset_classes:
            asset_classes = ["equity"]

        dd_from_profile = _SUITABILITY_DEFAULT_MAX_DD[req.suitability_profile]
        if req.max_drawdown_pct is None:
            max_drawdown_pct = dd_from_profile
            dd_source = "suitability_profile_default"
        else:
            max_drawdown_pct = float(req.max_drawdown_pct)
            dd_source = "explicit_request"

        return {
            "intent": intent,
            "target_return_pct": float(target_return_pct),
            "horizon_days": int(horizon_days),
            "sectors": sectors,
            "asset_classes": asset_classes,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_source": dd_source,
            "suitability_profile": req.suitability_profile.value,
            "include_options": bool(req.include_options),
        }

    def _score_and_recommend(self, parsed: dict) -> MandateEvaluationResponse:
        target = float(parsed["target_return_pct"])
        horizon = int(parsed["horizon_days"])
        sectors: List[str] = list(parsed["sectors"])
        assets: List[str] = list(parsed["asset_classes"])
        max_dd = float(parsed["max_drawdown_pct"])
        include_options = bool(parsed["include_options"])

        required_daily_bps = (target / max(horizon, 1)) * 100.0
        required_monthly_return = target / (max(horizon, 1) / 30.0)

        probability = 0.83
        probability -= max(0.0, required_daily_bps - 8.0) * 0.012
        probability -= max(0.0, required_monthly_return - 6.0) * 0.03
        probability -= 0.05 if horizon < 30 else 0.0
        probability -= 0.04 if len(sectors) <= 1 else 0.0
        probability -= 0.03 if "crypto" in assets else 0.0
        probability -= 0.05 if "fx" in assets else 0.0
        probability -= 0.08 if include_options else 0.0
        probability = _clamp(probability, 0.05, 0.95)

        expected_max_dd = _clamp(4.5 + target * 0.75 + max(0, 60 - horizon) * 0.04, 4.0, 40.0)
        expected_cvar95 = -_clamp(expected_max_dd * 1.30, 6.0, 50.0)
        expected_sortino = _clamp(0.9 + probability * 1.6 - (expected_max_dd / 30), 0.2, 3.5)

        confidence = 0.56
        confidence += 0.10 if parsed["intent"] else 0.0
        confidence += 0.08 if len(sectors) >= 2 else 0.0
        confidence += 0.08 if horizon >= 45 else 0.0
        confidence -= 0.08 if include_options else 0.0
        confidence = _clamp(confidence, 0.35, 0.92)

        cvar_policy_ok = abs(expected_cvar95) <= 20.0
        feasible = probability >= 0.60 and expected_max_dd <= max_dd and cvar_policy_ok
        if feasible and probability >= 0.70:
            band = FeasibilityBand.GREEN
        elif feasible:
            band = FeasibilityBand.YELLOW
        else:
            band = FeasibilityBand.RED

        policy = self._build_policy(sectors=sectors, assets=assets, max_dd=max_dd)
        stress_narratives = self._build_stress_narratives(sectors=sectors, assets=assets, horizon=horizon)
        confidence_intervals = self._build_regime_confidence_intervals(probability, confidence)

        rationale = [
            f"Required return pace is {required_daily_bps:.1f} bps/day over {horizon} days.",
            f"Target-hit probability calibrated to {probability:.2f} under policy-only constraints.",
            f"Expected max drawdown {expected_max_dd:.1f}% vs mandate cap {max_dd:.1f}%.",
            "Workflow pack supports PM + compliance sign-off while preserving execution isolation.",
        ]

        warnings: List[str] = []
        if expected_max_dd > max_dd:
            warnings.append("Expected drawdown exceeds requested hard cap.")
        if abs(expected_cvar95) > 20.0:
            warnings.append("Estimated CVaR(95%) breaches institutional -20% safeguard.")
        if include_options:
            warnings.append("Options requested, but MVP policy model is calibrated for linear instruments first.")
        if "crypto" in assets and horizon > 90:
            warnings.append("Crypto horizon assumptions may be unstable; daily retuning cadence is recommended.")

        request_id = f"mdt-{uuid4().hex[:12]}"
        return MandateEvaluationResponse(
            request_id=request_id,
            recommendation_mode=RecommendationMode.POLICY_ONLY,
            feasible=feasible,
            feasibility_band=band,
            probability_target_hit=round(probability, 4),
            confidence=round(confidence, 4),
            expected_max_drawdown_pct=round(expected_max_dd, 2),
            expected_cvar95_pct=round(expected_cvar95, 2),
            expected_sortino=round(expected_sortino, 2),
            rationale=rationale,
            warnings=warnings,
            parsed_mandate=parsed,
            policy=policy,
            stress_narratives=stress_narratives,
            confidence_intervals_by_regime=confidence_intervals,
            disclaimer=(
                "Assessment is probabilistic and policy-only. It is not an execution instruction, "
                "not investment advice, and does not guarantee returns."
            ),
        )

    def _build_policy(self, sectors: List[str], assets: List[str], max_dd: float) -> MandatePolicyRecommendation:
        sleeves: Dict[str, float] = {}
        sector_budget = 0.80
        if sectors:
            per_sector = sector_budget / float(len(sectors))
            for sector in sectors:
                sleeves[f"{sector}_sleeve"] = round(per_sector, 4)
        sleeves["defensive_overlay"] = round(1.0 - sum(sleeves.values()), 4)

        risk_limits = {
            "max_drawdown_pct": round(max_dd, 2),
            "cvar95_floor_pct": -20.0,
            "max_position_pct": 5.0,
            "max_sector_exposure_pct": 30.0,
            "turnover_cap_daily_pct": 200.0,
            "overnight_risk_buffer_pct": 2.0,
        }

        base_spread = 12.0 if "equity" in assets else 20.0
        if "crypto" in assets:
            base_spread = max(base_spread, 26.0)
        execution_constraints = {
            "max_spread_bps": base_spread,
            "slippage_budget_bps": 180.0,
            "min_edge_over_cost_bps": 8.0 if "equity" in assets else 10.0,
            "signal_to_edge_bps": 80.0,
        }

        constraints = [
            "recommendation_mode=POLICY_ONLY",
            "order_generation=disabled",
            "workflow_pack_requires_pm_and_compliance_signoff=true",
            "execution_isolation_enabled=true",
        ]
        notes = [
            "Promote only after OOS Sharpe and drawdown gates pass in paper/staging.",
            "Tune edge/spread/slippage thresholds weekly from fill + attribution telemetry.",
            "Escalate to daily retuning if regime instability exceeds alert thresholds.",
        ]
        return MandatePolicyRecommendation(
            sleeve_allocations=sleeves,
            risk_limits=risk_limits,
            execution_constraints=execution_constraints,
            constraints=constraints,
            notes=notes,
        )

    def _build_stress_narratives(self, sectors: List[str], assets: List[str], horizon: int) -> List[StressNarrative]:
        narratives: List[StressNarrative] = [
            StressNarrative(
                scenario="Rates shock",
                quant_shock="UST +150bps parallel shift",
                projected_impact_pct=round(-_clamp(2.5 + horizon / 60.0, 1.0, 8.0), 2),
                mitigation="Reduce gross, tighten max spread/slippage gates, raise quality threshold.",
            ),
            StressNarrative(
                scenario="Volatility spike",
                quant_shock="VIX +15 points in 3 sessions",
                projected_impact_pct=round(-_clamp(3.0 + len(assets), 1.5, 9.0), 2),
                mitigation="Activate defensive overlay and enforce lower position cap.",
            ),
        ]
        if "energy" in sectors:
            narratives.append(
                StressNarrative(
                    scenario="Commodity dislocation",
                    quant_shock="Brent gap +/-20%",
                    projected_impact_pct=-3.8,
                    mitigation="Limit single-sector concentration and add intraday stop policy.",
                )
            )
        if "technology" in sectors:
            narratives.append(
                StressNarrative(
                    scenario="Growth de-rating",
                    quant_shock="Nasdaq factor beta -2 sigma",
                    projected_impact_pct=-4.2,
                    mitigation="Pair momentum sleeve with market-neutral hedge overlay.",
                )
            )
        return narratives[:4]

    def _build_regime_confidence_intervals(self, probability: float, confidence: float) -> List[RegimeConfidenceInterval]:
        uncertainty = 0.10 + (1.0 - confidence) * 0.20
        regime_adjustments = [
            ("risk_on", 0.05),
            ("neutral", 0.0),
            ("risk_off", -0.12),
            ("carry_crash", -0.10),
            ("high_vol", -0.18),
        ]
        intervals: List[RegimeConfidenceInterval] = []
        for regime, adjust in regime_adjustments:
            center = _clamp(probability + adjust, 0.01, 0.99)
            lower = _clamp(center - uncertainty, 0.0, 1.0)
            upper = _clamp(center + uncertainty, 0.0, 1.0)
            intervals.append(
                RegimeConfidenceInterval(
                    regime=regime,
                    lower=round(lower, 4),
                    upper=round(upper, 4),
                )
            )
        return intervals

    def _append_audit_event(
        self,
        user_id: str,
        tier: str,
        request_payload: dict,
        response_payload: dict,
    ) -> None:
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        output_hash = hashlib.sha256(
            json.dumps(response_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        event = {
            "timestamp": _utc_now_iso(),
            "event": "mandate_evaluation",
            "model_version": MODEL_VERSION,
            "user_id": user_id,
            "tier": tier,
            "recommendation_mode": response_payload.get("recommendation_mode"),
            "request": request_payload,
            "response_summary": {
                "request_id": response_payload.get("request_id"),
                "feasible": response_payload.get("feasible"),
                "feasibility_band": response_payload.get("feasibility_band"),
                "probability_target_hit": response_payload.get("probability_target_hit"),
                "confidence": response_payload.get("confidence"),
                "expected_max_drawdown_pct": response_payload.get("expected_max_drawdown_pct"),
                "expected_cvar95_pct": response_payload.get("expected_cvar95_pct"),
            },
            "output_hash": output_hash,
        }
        with _AUDIT_LOCK:
            with AUDIT_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=True) + "\n")

    def _append_policy_change_event(self, workflow_id: str, actor: str, from_status: str, to_status: str, note: str = "") -> None:
        POLICY_CHANGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        event = PolicyChangeEvent(
            timestamp=_utc_now_iso(),
            workflow_id=workflow_id,
            actor=actor,
            from_status=from_status,
            to_status=to_status,
            note=note,
        ).model_dump(mode="json")
        with _POLICY_CHANGE_LOCK:
            with POLICY_CHANGE_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=True) + "\n")

    def _load_workflows(self) -> List[dict]:
        if not WORKFLOW_FILE.exists():
            return []
        with _WORKFLOW_LOCK:
            try:
                payload = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Failed to load workflow file %s: %s", WORKFLOW_FILE, exc)
                return []
        return payload if isinstance(payload, list) else []

    def _save_workflows(self, rows: List[dict]) -> None:
        WORKFLOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _WORKFLOW_LOCK:
            WORKFLOW_FILE.write_text(
                json.dumps(rows, separators=(",", ":"), ensure_ascii=True),
                encoding="utf-8",
            )

    def _find_audit_event_by_request_id(self, request_id: str) -> dict | None:
        events = self.load_recent_audit(limit=5000)
        for event in events:
            summary = event.get("response_summary", {}) or {}
            if str(summary.get("request_id", "")) == str(request_id):
                return event
        return None

    def _find_workflow_by_request_id(self, request_id: str) -> MandateWorkflowPack | None:
        workflows = self._load_workflows()
        for row in workflows:
            if str(row.get("request_id", "")) == str(request_id):
                try:
                    return MandateWorkflowPack.model_validate(row)
                except Exception as exc:
                    logger.warning("Failed to validate workflow for request %s: %s", request_id, exc)
                    return None
        return None

    @staticmethod
    def _find_workflow_index(rows: List[dict], workflow_id: str) -> int:
        for idx, row in enumerate(rows):
            if str(row.get("workflow_id", "")) == str(workflow_id):
                return idx
        return -1

