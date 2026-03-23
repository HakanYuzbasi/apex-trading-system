"""
Intraday portfolio stress engine for live control-loop decisions.

Evaluates current positions against a bounded set of predefined scenarios and
returns a control posture that the trading loop can apply immediately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

from risk.portfolio_stress_test import PortfolioStressTest, StressTestResult


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StressScenarioSummary:
    scenario_id: str
    scenario_name: str
    portfolio_pnl: float
    portfolio_return: float
    max_drawdown: float
    top_position_losses: List[Dict[str, Any]] = field(default_factory=list)
    breached_limits: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_result(cls, scenario_id: str, result: StressTestResult) -> "StressScenarioSummary":
        return cls(
            scenario_id=scenario_id,
            scenario_name=result.scenario_name,
            portfolio_pnl=float(result.portfolio_pnl),
            portfolio_return=float(result.portfolio_return),
            max_drawdown=float(result.max_drawdown),
            top_position_losses=[
                {"symbol": str(symbol), "pnl": float(pnl)}
                for symbol, pnl in result.worst_positions
            ],
            breached_limits=list(result.breached_limits),
            recommendations=list(result.recommendations),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressControlState:
    enabled: bool
    active: bool
    evaluated_at: str
    scenario_count: int
    action: str
    halt_new_entries: bool
    size_multiplier: float
    reason: str
    worst_scenario_id: str = ""
    worst_scenario_name: str = ""
    worst_portfolio_return: float = 0.0
    worst_portfolio_pnl: float = 0.0
    worst_drawdown: float = 0.0
    breached_limits: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scenarios: List[StressScenarioSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["scenarios"] = [scenario.to_dict() for scenario in self.scenarios]
        return payload


class IntradayStressEngine:
    """Periodic live portfolio stress evaluator."""

    DEFAULT_SCENARIOS = (
        "2020_covid_crash",
        "vix_spike",
        "correlation_breakdown",
        "rate_shock",
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        scenario_ids: Sequence[str] | None = None,
        warning_return_threshold: float = -0.04,
        halt_return_threshold: float = -0.08,
        warning_drawdown_threshold: float = 0.06,
        halt_drawdown_threshold: float = 0.10,
        warning_size_multiplier: float = 0.60,
        halt_size_multiplier: float = 0.25,
    ) -> None:
        self.enabled = bool(enabled)
        self.scenario_ids = list(scenario_ids or self.DEFAULT_SCENARIOS)
        self.warning_return_threshold = float(warning_return_threshold)
        self.halt_return_threshold = float(halt_return_threshold)
        self.warning_drawdown_threshold = float(warning_drawdown_threshold)
        self.halt_drawdown_threshold = float(halt_drawdown_threshold)
        self.warning_size_multiplier = float(warning_size_multiplier)
        self.halt_size_multiplier = float(halt_size_multiplier)

    def idle_state(self, *, reason: str = "not_evaluated") -> StressControlState:
        return StressControlState(
            enabled=self.enabled,
            active=False,
            evaluated_at=_utc_now_iso(),
            scenario_count=0,
            action="normal",
            halt_new_entries=False,
            size_multiplier=1.0,
            reason=reason,
        )

    def evaluate(
        self,
        *,
        positions: Dict[str, float],
        prices: Dict[str, float],
        historical_data: Dict[str, Any] | None,
        capital: float,
    ) -> StressControlState:
        if not self.enabled:
            return self.idle_state(reason="disabled")

        normalized_positions = {
            str(symbol): float(qty)
            for symbol, qty in positions.items()
            if float(qty) != 0.0 and float(prices.get(symbol, 0.0) or 0.0) > 0.0
        }
        normalized_prices = {
            str(symbol): float(prices[symbol])
            for symbol in normalized_positions
        }
        if not normalized_positions:
            return self.idle_state(reason="no_positions")

        tester = PortfolioStressTest(
            positions=normalized_positions,
            prices=normalized_prices,
            historical_data=historical_data or {},
            capital=max(float(capital), 1.0),
        )
        scenario_summaries: List[StressScenarioSummary] = []
        for scenario_id in self.scenario_ids:
            scenario = tester.PREDEFINED_SCENARIOS.get(str(scenario_id))
            if scenario is None:
                continue
            scenario_summaries.append(
                StressScenarioSummary.from_result(scenario_id, tester.run_scenario(scenario))
            )

        if not scenario_summaries:
            return self.idle_state(reason="no_scenarios")

        worst = min(
            scenario_summaries,
            key=lambda summary: (summary.portfolio_return, summary.portfolio_pnl),
        )

        breach_reasons: List[str] = []
        if worst.portfolio_return <= self.halt_return_threshold:
            breach_reasons.append(
                f"worst_return={worst.portfolio_return:.2%} <= {self.halt_return_threshold:.2%}"
            )
        if worst.max_drawdown >= self.halt_drawdown_threshold:
            breach_reasons.append(
                f"worst_drawdown={worst.max_drawdown:.2%} >= {self.halt_drawdown_threshold:.2%}"
            )
        if worst.breached_limits:
            breach_reasons.append("risk_limits=" + ",".join(worst.breached_limits[:3]))

        if breach_reasons:
            action = "halt_entries"
            halt_new_entries = True
            size_multiplier = self.halt_size_multiplier
            reason = "; ".join(breach_reasons)
        elif (
            worst.portfolio_return <= self.warning_return_threshold
            or worst.max_drawdown >= self.warning_drawdown_threshold
        ):
            action = "size_down"
            halt_new_entries = False
            size_multiplier = self.warning_size_multiplier
            if worst.portfolio_return <= self.warning_return_threshold:
                reason = (
                    f"worst_return={worst.portfolio_return:.2%} <= "
                    f"{self.warning_return_threshold:.2%}"
                )
            else:
                reason = (
                    f"worst_drawdown={worst.max_drawdown:.2%} >= "
                    f"{self.warning_drawdown_threshold:.2%}"
                )
        else:
            action = "normal"
            halt_new_entries = False
            size_multiplier = 1.0
            reason = "within_limits"

        return StressControlState(
            enabled=self.enabled,
            active=action != "normal",
            evaluated_at=_utc_now_iso(),
            scenario_count=len(scenario_summaries),
            action=action,
            halt_new_entries=halt_new_entries,
            size_multiplier=size_multiplier,
            reason=reason,
            worst_scenario_id=worst.scenario_id,
            worst_scenario_name=worst.scenario_name,
            worst_portfolio_return=float(worst.portfolio_return),
            worst_portfolio_pnl=float(worst.portfolio_pnl),
            worst_drawdown=float(worst.max_drawdown),
            breached_limits=list(worst.breached_limits),
            recommendations=list(worst.recommendations),
            scenarios=scenario_summaries,
        )
