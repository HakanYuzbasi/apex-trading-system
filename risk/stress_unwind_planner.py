"""
Stress unwind planner.

Builds a staged de-risking plan from the active stress-control state so the
execution loop can unwind the most vulnerable positions through the normal exit
path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Mapping, Optional

from risk.intraday_stress_engine import StressControlState, StressScenarioSummary


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StressUnwindCandidate:
    symbol: str
    current_qty: float
    side: str
    expected_stress_pnl: float
    position_value: float
    concentration_weight: float
    scenario_loss_share: float
    liquidity_regime: str
    liquidity_score: float
    liquidity_size_multiplier: float
    volume_ratio: float
    avg_daily_volume: float
    avg_daily_dollar_volume: float
    max_liquidation_qty: float
    target_qty: float
    target_reduction_pct: float
    priority_score: float
    action: str = "partial_reduce"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressUnwindPlan:
    active: bool
    created_at: str
    action: str
    reason: str
    plan_id: str = ""
    plan_epoch: int = 0
    worst_scenario_id: str = ""
    worst_scenario_name: str = ""
    candidates: List[StressUnwindCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


class StressUnwindPlanner:
    """Selects the next positions to cut when the stress engine trips hard."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_positions_per_cycle: int = 2,
        max_participation_rate: float = 0.05,
        min_reduction_pct: float = 0.10,
        fallback_reduction_pct: float = 0.25,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_positions_per_cycle = max(1, int(max_positions_per_cycle))
        self.max_participation_rate = max(0.01, float(max_participation_rate))
        self.min_reduction_pct = min(1.0, max(0.01, float(min_reduction_pct)))
        self.fallback_reduction_pct = min(1.0, max(0.05, float(fallback_reduction_pct)))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _normalize_qty(current_qty: float, target_qty: float) -> float:
        current_abs = abs(float(current_qty))
        bounded = max(0.0, min(current_abs, float(target_qty)))
        if current_abs >= 1.0 and math.isclose(current_abs, round(current_abs), abs_tol=1e-6):
            return float(max(1, min(int(round(current_abs)), int(math.floor(bounded)))))
        return round(max(0.000001, bounded), 6)

    def _liquidity_score(
        self,
        *,
        position_value: float,
        avg_daily_dollar_volume: float,
        volume_ratio: float,
        liquidity_size_multiplier: float,
    ) -> float:
        if avg_daily_dollar_volume > 0:
            participation_load = self._clamp(position_value / avg_daily_dollar_volume, 0.0, 2.0)
            adv_component = 1.0 - min(1.0, participation_load)
        else:
            adv_component = 0.35
        return self._clamp(
            (0.50 * adv_component)
            + (0.30 * self._clamp(liquidity_size_multiplier, 0.0, 1.0))
            + (0.20 * self._clamp(volume_ratio, 0.0, 1.5) / 1.5),
            0.05,
            1.0,
        )

    def idle_plan(self, *, reason: str = "not_evaluated") -> StressUnwindPlan:
        return StressUnwindPlan(
            active=False,
            created_at=_utc_now_iso(),
            action="none",
            reason=reason,
        )

    def build_plan(
        self,
        *,
        stress_state: StressControlState,
        positions: Dict[str, float],
        prices: Optional[Mapping[str, float]] = None,
        portfolio_value: float = 0.0,
        liquidity_snapshot: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> StressUnwindPlan:
        if not self.enabled:
            return self.idle_plan(reason="disabled")
        if not stress_state.active or not stress_state.halt_new_entries:
            return self.idle_plan(reason="stress_normalized")

        worst_scenario: StressScenarioSummary | None = next(
            (
                scenario
                for scenario in stress_state.scenarios
                if scenario.scenario_id == stress_state.worst_scenario_id
            ),
            None,
        )
        if worst_scenario is None:
            return self.idle_plan(reason="missing_worst_scenario")

        prices = prices or {}
        liquidity_snapshot = liquidity_snapshot or {}
        gross_exposure = sum(
            abs(float(qty)) * max(0.0, float(prices.get(symbol, 0.0) or 0.0))
            for symbol, qty in positions.items()
        )
        gross_exposure = max(gross_exposure, float(portfolio_value or 0.0), 1.0)
        total_scenario_loss = sum(
            abs(float(row.get("pnl", 0.0) or 0.0))
            for row in worst_scenario.top_position_losses
            if float(row.get("pnl", 0.0) or 0.0) < 0.0
        )
        severity = self._clamp(abs(float(stress_state.worst_portfolio_return)) / 0.15, 0.0, 1.0)

        candidates: List[StressUnwindCandidate] = []
        for row in worst_scenario.top_position_losses:
            symbol = str(row.get("symbol", "")).strip()
            if not symbol:
                continue
            current_qty = float(positions.get(symbol, 0.0) or 0.0)
            if current_qty == 0.0:
                continue
            price = float(prices.get(symbol, 0.0) or 0.0)
            if price <= 0.0:
                continue
            expected_stress_pnl = float(row.get("pnl", 0.0) or 0.0)
            if expected_stress_pnl >= 0.0:
                continue
            position_value = abs(current_qty) * price
            concentration_weight = self._clamp(position_value / gross_exposure, 0.0, 1.0)
            scenario_loss_share = self._clamp(
                abs(expected_stress_pnl) / max(total_scenario_loss, 1.0),
                0.0,
                1.0,
            )
            liquidity = dict(liquidity_snapshot.get(symbol, {}) or {})
            volume_ratio = max(0.0, float(liquidity.get("volume_ratio", 1.0) or 1.0))
            avg_daily_volume = max(0.0, float(liquidity.get("avg_daily_volume", 0.0) or 0.0))
            avg_daily_dollar_volume = max(
                0.0,
                float(liquidity.get("avg_daily_dollar_volume", avg_daily_volume * price) or 0.0),
            )
            liquidity_size_multiplier = self._clamp(
                float(liquidity.get("position_size_multiplier", 1.0) or 1.0),
                0.0,
                1.0,
            )
            liquidity_regime = str(liquidity.get("liquidity_regime", "NORMAL") or "NORMAL").upper()
            liquidity_score = self._liquidity_score(
                position_value=position_value,
                avg_daily_dollar_volume=avg_daily_dollar_volume,
                volume_ratio=volume_ratio,
                liquidity_size_multiplier=liquidity_size_multiplier,
            )
            priority_score = self._clamp(
                (0.55 * scenario_loss_share)
                + (0.30 * concentration_weight)
                + (0.15 * liquidity_score),
                0.0,
                1.0,
            )
            desired_reduction_pct = self._clamp(
                self.min_reduction_pct
                + (
                    (0.55 * scenario_loss_share)
                    + (0.35 * concentration_weight)
                    + (0.10 * severity)
                )
                * (1.0 - self.min_reduction_pct),
                self.min_reduction_pct,
                1.0,
            )
            liquidity_adjustment = self._clamp(0.35 + (0.65 * liquidity_score), 0.20, 1.0)
            desired_qty = abs(current_qty) * desired_reduction_pct * liquidity_adjustment
            liquidity_cap_qty = (
                avg_daily_volume
                * self.max_participation_rate
                * max(0.25, liquidity_size_multiplier)
            )
            if liquidity_cap_qty <= 0.0:
                liquidity_cap_qty = abs(current_qty) * self.fallback_reduction_pct
            target_qty = self._normalize_qty(current_qty, min(abs(current_qty), min(desired_qty, liquidity_cap_qty)))
            if target_qty <= 0.0:
                continue
            target_reduction_pct = self._clamp(target_qty / abs(current_qty), 0.0, 1.0)
            action = "partial_reduce"
            if (
                target_reduction_pct >= 0.95
                or (
                    scenario_loss_share >= 0.45
                    and concentration_weight >= 0.25
                    and liquidity_score >= 0.60
                )
            ):
                target_qty = abs(current_qty)
                target_reduction_pct = 1.0
                action = "full_exit"
            candidates.append(
                StressUnwindCandidate(
                    symbol=symbol,
                    current_qty=current_qty,
                    side="LONG" if current_qty > 0 else "SHORT",
                    expected_stress_pnl=expected_stress_pnl,
                    position_value=position_value,
                    concentration_weight=concentration_weight,
                    scenario_loss_share=scenario_loss_share,
                    liquidity_regime=liquidity_regime,
                    liquidity_score=liquidity_score,
                    liquidity_size_multiplier=liquidity_size_multiplier,
                    volume_ratio=volume_ratio,
                    avg_daily_volume=avg_daily_volume,
                    avg_daily_dollar_volume=avg_daily_dollar_volume,
                    max_liquidation_qty=float(liquidity_cap_qty),
                    target_qty=target_qty,
                    target_reduction_pct=target_reduction_pct,
                    priority_score=priority_score,
                    action=action,
                ),
            )

        if not candidates:
            return self.idle_plan(reason="no_candidates")

        selected = sorted(
            candidates,
            key=lambda candidate: (-candidate.priority_score, candidate.expected_stress_pnl),
        )[: self.max_positions_per_cycle]
        return StressUnwindPlan(
            active=True,
            created_at=_utc_now_iso(),
            action="liquidation_plan",
            reason=stress_state.reason,
            worst_scenario_id=stress_state.worst_scenario_id,
            worst_scenario_name=stress_state.worst_scenario_name,
            candidates=selected,
        )
