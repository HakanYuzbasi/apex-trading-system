from risk.equity_outlier_guard import EquityOutlierGuard


def test_guard_seeds_and_accepts_normal_moves():
    guard = EquityOutlierGuard(
        enabled=True,
        max_step_move_pct=0.25,
        confirmations_required=3,
        suspect_match_tolerance_pct=0.02,
    )

    first = guard.evaluate(1_100_000.0)
    assert first.accepted is True
    assert first.reason == "seeded"
    assert first.filtered_value == 1_100_000.0

    normal = guard.evaluate(1_150_000.0)
    assert normal.accepted is True
    assert normal.reason == "within_threshold"
    assert normal.filtered_value == 1_150_000.0


def test_guard_rejects_one_off_large_drop_and_uses_last_good():
    guard = EquityOutlierGuard(
        enabled=True,
        max_step_move_pct=0.25,
        confirmations_required=3,
        suspect_match_tolerance_pct=0.02,
    )
    guard.seed(1_280_000.0)

    rejected = guard.evaluate(100_000.0)
    assert rejected.accepted is False
    assert rejected.reason == "outlier_rejected"
    assert rejected.filtered_value == 1_280_000.0
    assert rejected.suspect_count == 1


def test_guard_accepts_large_move_after_confirmations():
    guard = EquityOutlierGuard(
        enabled=True,
        max_step_move_pct=0.20,
        confirmations_required=3,
        suspect_match_tolerance_pct=0.05,
    )
    guard.seed(1_000_000.0)

    d1 = guard.evaluate(600_000.0)
    d2 = guard.evaluate(603_000.0)
    d3 = guard.evaluate(598_000.0)

    assert d1.accepted is False
    assert d2.accepted is False
    assert d3.accepted is True
    assert d3.reason == "confirmed_large_move"
    assert d3.filtered_value == 598_000.0


def test_guard_rejects_non_positive_values():
    guard = EquityOutlierGuard(enabled=True)
    guard.seed(1_200_000.0)

    decision = guard.evaluate(0.0)
    assert decision.accepted is False
    assert decision.reason == "non_positive"
    assert decision.filtered_value == 1_200_000.0
