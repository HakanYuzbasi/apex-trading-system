import pytest
from pathlib import Path
from monitoring.performance_tracker import PerformanceTracker
from core.execution_loop import ApexTradingSystem


@pytest.mark.asyncio
async def test_performance_tracker_reset_history(tmp_path):
    tracker = PerformanceTracker(data_dir=tmp_path)
    tracker.trades = [{"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 100.0}]
    tracker.equity_curve = [("2026-01-01T00:00:00", 1_250_000.0)]
    tracker.starting_capital = 1_250_000.0

    await tracker.reset_history(starting_capital=100_000.0, reason="unit_test")

    assert tracker.trades == []
    assert len(tracker.equity_curve) == 1
    assert float(tracker.equity_curve[0][1]) == 100_000.0
    assert tracker.starting_capital == 100_000.0
    assert tracker.history_file.exists()


@pytest.mark.asyncio
async def test_performance_tracker_rebase_baseline_preserves_trade_context(tmp_path):
    tracker = PerformanceTracker(data_dir=tmp_path)
    tracker.trades.append({"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 100.0})
    tracker.equity_curve.append(("2026-01-01T00:00:00", 125_000.0))
    tracker.equity_curve.append(("2026-01-02T00:00:00", 124_500.0))
    tracker.starting_capital = 125_000.0

    await tracker.rebase_baseline(starting_capital=100_000.0, reason="unit_test_rebase")

    assert len(tracker.trades) == 1
    assert len(tracker.equity_curve) == 3
    assert float(tracker.equity_curve[-1][1]) == 100_000.0
    assert tracker.starting_capital == 100_000.0
    assert tracker.history_file.exists()


def test_execution_loop_session_scoped_state_path():
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.user_data_dir = Path("/tmp/apex-test-session")
    system._session_state_prefix = "crypto_"

    assert system._session_scoped_state_path("performance_governor_state.json") == (
        system.user_data_dir / "crypto_performance_governor_state.json"
    )


def test_build_confidence_audit_flags_near_miss(monkeypatch):
    from config import ApexConfig

    monkeypatch.setattr(ApexConfig, "CONFIDENCE_AUDIT_NEAR_MISS_GAP", 0.02, raising=False)
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.session_type = "crypto"
    system._current_regime = "neutral"

    payload = system._build_confidence_audit(
        symbol="CRYPTO:BTC/USD",
        asset_class="CRYPTO",
        confidence=0.398,
        effective_confidence_threshold=0.400,
        base_confidence=0.40,
        governor_confidence_boost=0.0,
        crypto_confidence_multiplier=1.0,
        rotation_discount=0.0,
        signal=0.31,
    )

    assert payload["near_miss"] is True
    assert payload["confidence_gap"] == pytest.approx(0.002)
