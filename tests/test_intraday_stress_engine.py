from __future__ import annotations

from risk.intraday_stress_engine import IntradayStressEngine


def test_intraday_stress_engine_halts_entries_on_severe_portfolio_loss():
    engine = IntradayStressEngine(
        scenario_ids=["2020_covid_crash"],
        warning_return_threshold=-0.04,
        halt_return_threshold=-0.08,
    )

    state = engine.evaluate(
        positions={"AAPL": 1000},
        prices={"AAPL": 100.0},
        historical_data={},
        capital=100_000.0,
    )

    assert state.action == "halt_entries"
    assert state.halt_new_entries is True
    assert state.size_multiplier == engine.halt_size_multiplier
    assert state.worst_scenario_id == "2020_covid_crash"
    assert state.worst_portfolio_return <= -0.08


def test_intraday_stress_engine_returns_idle_state_when_no_positions():
    engine = IntradayStressEngine()

    state = engine.evaluate(
        positions={},
        prices={},
        historical_data={},
        capital=100_000.0,
    )

    assert state.action == "normal"
    assert state.active is False
    assert state.reason == "no_positions"
    assert state.scenario_count == 0
