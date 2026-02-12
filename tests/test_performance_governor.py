"""Tests for adaptive performance governor controls."""

from datetime import datetime, timedelta

from risk.performance_governor import GovernorTier, PerformanceGovernor


def _feed(governor: PerformanceGovernor, values):
    start = datetime(2026, 1, 1, 9, 30)
    for i, value in enumerate(values):
        governor.update(value, start + timedelta(minutes=i))
    return governor.get_snapshot()


def test_performance_governor_warmup_state():
    governor = PerformanceGovernor(
        min_samples=8,
        sample_interval_minutes=1,
        lookback_points=20,
    )
    snapshot = _feed(governor, [100.0, 100.2, 100.1, 100.3])

    assert snapshot.tier == GovernorTier.GREEN
    assert snapshot.sample_count == 4
    assert snapshot.halt_new_entries is False
    assert snapshot.size_multiplier == 1.0


def test_performance_governor_blocks_new_entries_on_deep_drawdown():
    governor = PerformanceGovernor(
        target_sharpe=1.5,
        target_sortino=2.0,
        max_drawdown=0.08,
        min_samples=6,
        sample_interval_minutes=1,
        lookback_points=32,
    )
    snapshot = _feed(governor, [100, 99, 98, 97, 96, 95, 94, 93, 92])

    assert snapshot.tier == GovernorTier.RED
    assert snapshot.halt_new_entries is True
    assert snapshot.size_multiplier == 0.25
    assert snapshot.drawdown >= 0.08


def test_performance_governor_stays_green_on_healthy_series():
    governor = PerformanceGovernor(
        target_sharpe=0.5,
        target_sortino=0.7,
        max_drawdown=0.10,
        min_samples=8,
        sample_interval_minutes=1,
        lookback_points=32,
    )
    snapshot = _feed(
        governor,
        [100.0, 100.3, 100.2, 100.6, 100.55, 100.9, 100.85, 101.2, 101.1, 101.5, 101.4, 101.8],
    )

    assert snapshot.tier == GovernorTier.GREEN
    assert snapshot.halt_new_entries is False
    assert snapshot.size_multiplier == 1.0
