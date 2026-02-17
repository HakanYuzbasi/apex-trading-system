from datetime import datetime

from reconciliation.equity_reconciler import EquityReconciler


def test_equity_reconciler_no_breach_when_gap_within_limits():
    reconciler = EquityReconciler(
        max_gap_dollars=50_000.0,
        max_gap_pct=0.05,
        breach_confirmations=2,
        heal_confirmations=2,
    )
    snapshot = reconciler.evaluate(
        broker_equity=1_000_000.0,
        modeled_equity=980_000.0,
        timestamp=datetime(2026, 2, 13, 10, 0, 0),
    )
    assert snapshot.breached is False
    assert snapshot.block_entries is False
    assert snapshot.reason == "ok"
    assert snapshot.healthy_streak == 1


def test_equity_reconciler_latches_and_clears_with_hysteresis():
    reconciler = EquityReconciler(
        max_gap_dollars=10_000.0,
        max_gap_pct=0.01,
        breach_confirmations=2,
        heal_confirmations=3,
    )
    first = reconciler.evaluate(
        broker_equity=1_000_000.0,
        modeled_equity=950_000.0,
        timestamp=datetime(2026, 2, 13, 10, 0, 0),
    )
    assert first.breached is True
    assert first.block_entries is False

    second = reconciler.evaluate(
        broker_equity=1_000_000.0,
        modeled_equity=949_500.0,
        timestamp=datetime(2026, 2, 13, 10, 1, 0),
    )
    assert second.breached is True
    assert second.block_entries is True
    assert second.breach_streak == 2

    for i in range(2):
        healed = reconciler.evaluate(
            broker_equity=1_000_000.0,
            modeled_equity=999_500.0,
            timestamp=datetime(2026, 2, 13, 10, 2 + i, 0),
        )
        assert healed.breached is False
        assert healed.block_entries is True

    clear_snapshot = reconciler.evaluate(
        broker_equity=1_000_000.0,
        modeled_equity=999_700.0,
        timestamp=datetime(2026, 2, 13, 10, 4, 0),
    )
    assert clear_snapshot.breached is False
    assert clear_snapshot.block_entries is False
    assert clear_snapshot.healthy_streak == 3


def test_equity_reconciler_fail_closed_on_unavailable_data():
    reconciler = EquityReconciler(
        fail_closed_on_unavailable=True,
        breach_confirmations=1,
        heal_confirmations=1,
    )
    snapshot = reconciler.evaluate(
        broker_equity=1_000_000.0,
        modeled_equity=None,
        timestamp=datetime(2026, 2, 13, 10, 0, 0),
    )
    assert snapshot.breached is True
    assert snapshot.block_entries is True
    assert snapshot.reason == "data_unavailable"
