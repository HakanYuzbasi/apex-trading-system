from __future__ import annotations

from risk.intraday_stress_engine import StressControlState, StressScenarioSummary
from risk.stress_unwind_planner import StressUnwindPlanner


def test_stress_unwind_planner_selects_worst_positions_from_active_halt_state():
    planner = StressUnwindPlanner(
        enabled=True,
        max_positions_per_cycle=2,
        max_participation_rate=0.05,
        min_reduction_pct=0.10,
        fallback_reduction_pct=0.25,
    )
    stress_state = StressControlState(
        enabled=True,
        active=True,
        evaluated_at="2099-01-01T00:00:00+00:00",
        scenario_count=1,
        action="halt_entries",
        halt_new_entries=True,
        size_multiplier=0.25,
        reason="worst_return=-12.00% <= -8.00%",
        worst_scenario_id="2020_covid_crash",
        worst_scenario_name="2020 COVID Crash",
        worst_portfolio_return=-0.12,
        worst_portfolio_pnl=-12000.0,
        worst_drawdown=0.12,
        scenarios=[
            StressScenarioSummary(
                scenario_id="2020_covid_crash",
                scenario_name="2020 COVID Crash",
                portfolio_pnl=-12000.0,
                portfolio_return=-0.12,
                max_drawdown=0.12,
                top_position_losses=[
                    {"symbol": "AAPL", "pnl": -8000.0},
                    {"symbol": "MSFT", "pnl": -3000.0},
                    {"symbol": "NVDA", "pnl": -1000.0},
                ],
            )
        ],
    )

    plan = planner.build_plan(
        stress_state=stress_state,
        positions={"AAPL": 100.0, "MSFT": 50.0, "NVDA": 25.0},
        prices={"AAPL": 200.0, "MSFT": 300.0, "NVDA": 100.0},
        portfolio_value=37_500.0,
        liquidity_snapshot={
            "AAPL": {
                "avg_daily_volume": 50_000.0,
                "avg_daily_dollar_volume": 10_000_000.0,
                "volume_ratio": 1.2,
                "position_size_multiplier": 1.0,
                "liquidity_regime": "NORMAL",
            },
            "MSFT": {
                "avg_daily_volume": 600.0,
                "avg_daily_dollar_volume": 180_000.0,
                "volume_ratio": 0.7,
                "position_size_multiplier": 0.5,
                "liquidity_regime": "STRESSED",
            },
            "NVDA": {
                "avg_daily_volume": 1_000.0,
                "avg_daily_dollar_volume": 100_000.0,
                "volume_ratio": 0.8,
                "position_size_multiplier": 0.75,
                "liquidity_regime": "THIN",
            },
        },
    )

    assert plan.active is True
    assert [candidate.symbol for candidate in plan.candidates] == ["AAPL", "MSFT"]
    assert plan.candidates[0].action == "full_exit"
    assert plan.candidates[0].target_reduction_pct == 1.0
    assert plan.candidates[1].action == "partial_reduce"
    assert plan.candidates[1].target_qty < abs(plan.candidates[1].current_qty)
    assert plan.candidates[1].max_liquidation_qty == 15.0
    assert plan.candidates[1].liquidity_regime == "STRESSED"


def test_stress_unwind_planner_caps_partial_cut_by_liquidity():
    planner = StressUnwindPlanner(
        enabled=True,
        max_positions_per_cycle=1,
        max_participation_rate=0.05,
        min_reduction_pct=0.10,
        fallback_reduction_pct=0.25,
    )
    stress_state = StressControlState(
        enabled=True,
        active=True,
        evaluated_at="2099-01-01T00:00:00+00:00",
        scenario_count=1,
        action="halt_entries",
        halt_new_entries=True,
        size_multiplier=0.25,
        reason="worst_return=-9.00% <= -8.00%",
        worst_scenario_id="liquidity_crisis",
        worst_scenario_name="Liquidity Crisis",
        worst_portfolio_return=-0.09,
        worst_portfolio_pnl=-9000.0,
        worst_drawdown=0.11,
        scenarios=[
            StressScenarioSummary(
                scenario_id="liquidity_crisis",
                scenario_name="Liquidity Crisis",
                portfolio_pnl=-9000.0,
                portfolio_return=-0.09,
                max_drawdown=0.11,
                top_position_losses=[{"symbol": "TSLA", "pnl": -9000.0}],
            )
        ],
    )

    plan = planner.build_plan(
        stress_state=stress_state,
        positions={"TSLA": 400.0},
        prices={"TSLA": 100.0},
        portfolio_value=40_000.0,
        liquidity_snapshot={
            "TSLA": {
                "avg_daily_volume": 1_000.0,
                "avg_daily_dollar_volume": 100_000.0,
                "volume_ratio": 0.4,
                "position_size_multiplier": 0.5,
                "liquidity_regime": "STRESSED",
            }
        },
    )

    assert plan.active is True
    assert len(plan.candidates) == 1
    candidate = plan.candidates[0]
    assert candidate.action == "partial_reduce"
    assert candidate.target_qty == 25.0
    assert candidate.target_reduction_pct == 25.0 / 400.0


def test_stress_unwind_planner_returns_idle_plan_when_stress_not_active():
    planner = StressUnwindPlanner(enabled=True)
    stress_state = StressControlState(
        enabled=True,
        active=False,
        evaluated_at="2099-01-01T00:00:00+00:00",
        scenario_count=0,
        action="normal",
        halt_new_entries=False,
        size_multiplier=1.0,
        reason="within_limits",
    )

    plan = planner.build_plan(stress_state=stress_state, positions={"AAPL": 100.0})

    assert plan.active is False
    assert plan.reason == "stress_normalized"
