"""Tests for portfolio kill-switch behavior."""

from datetime import datetime, timedelta

from risk.kill_switch import KillSwitchConfig, RiskKillSwitch


def _curve(values):
    start = datetime(2026, 1, 1, 9, 30)
    return [(start + timedelta(hours=i), v) for i, v in enumerate(values)]


def test_kill_switch_triggers_on_drawdown_breach():
    ks = RiskKillSwitch(
        config=KillSwitchConfig(
            dd_multiplier=1.5,
            sharpe_window_days=63,
            sharpe_floor=-10.0,  # neutralize sharpe trigger
            logic="OR",
            min_points=10,
        ),
        historical_mdd_baseline=0.08,
    )
    state = ks.update(_curve([100, 101, 102, 101, 99, 97, 95, 93, 91, 88, 85]))
    assert state.active is True
    assert state.drawdown > 0.12  # 1.5 * 8%


def test_kill_switch_latches_until_manual_reset():
    ks = RiskKillSwitch(
        config=KillSwitchConfig(dd_multiplier=1.5, sharpe_window_days=63, sharpe_floor=0.0, logic="OR", min_points=10),
        historical_mdd_baseline=0.08,
    )
    state = ks.update(_curve([100, 99, 98, 97, 96, 94, 92, 90, 89, 87, 85]))
    assert state.active is True

    recovered = ks.update(_curve([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]))
    assert recovered.active is True  # latched

    ks.reset()
    after_reset = ks.update(_curve([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]))
    assert after_reset.active is False
