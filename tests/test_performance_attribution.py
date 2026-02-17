from datetime import datetime

import pytest

from monitoring.performance_attribution import PerformanceAttributionTracker


def test_entry_exit_lifecycle_persists_and_computes_drag(tmp_path):
    tracker = PerformanceAttributionTracker(data_dir=tmp_path, max_closed_trades=100)
    entry_time = datetime(2026, 1, 1, 10, 0, 0)
    exit_time = datetime(2026, 1, 1, 13, 0, 0)

    tracker.record_entry(
        symbol="AAPL",
        asset_class="EQUITY",
        sleeve="equities_sleeve",
        side="LONG",
        quantity=10,
        entry_price=100.0,
        entry_signal=0.62,
        entry_confidence=0.78,
        governor_tier="green",
        governor_regime="risk_on",
        risk_multiplier=1.0,
        vix_multiplier=0.9,
        governor_size_multiplier=0.8,
        entry_slippage_bps=2.0,
        entry_time=entry_time,
        source="unit_test_entry",
    )
    assert "AAPL" in tracker.open_positions

    tracker.record_exit(
        symbol="AAPL",
        quantity=10,
        exit_price=110.0,
        gross_pnl=100.0,
        net_pnl=95.0,
        commissions=5.0,
        exit_reason="target_hit",
        exit_slippage_bps=4.0,
        exit_time=exit_time,
        source="unit_test_exit",
    )

    assert "AAPL" not in tracker.open_positions
    assert len(tracker.closed_trades) == 1
    closed = tracker.closed_trades[0]
    assert closed["symbol"] == "AAPL"
    assert closed["asset_class"] == "EQUITY"
    assert closed["sleeve"] == "equities_sleeve"
    assert closed["holding_hours"] == pytest.approx(3.0)
    assert closed["modeled_entry_slippage_cost"] == pytest.approx(0.2)
    assert closed["modeled_exit_slippage_cost"] == pytest.approx(0.44)
    assert closed["modeled_execution_drag"] == pytest.approx(5.64)
    assert closed["source"] == "unit_test_exit"

    restored = PerformanceAttributionTracker(data_dir=tmp_path, max_closed_trades=100)
    assert len(restored.closed_trades) == 1
    assert restored.closed_trades[0]["symbol"] == "AAPL"


def test_exit_without_entry_uses_fallback_context(tmp_path):
    tracker = PerformanceAttributionTracker(data_dir=tmp_path, max_closed_trades=100)
    entry_time = datetime(2026, 1, 2, 9, 30, 0)
    exit_time = datetime(2026, 1, 2, 15, 30, 0)

    tracker.record_exit(
        symbol="FX:EUR/USD",
        quantity=10_000,
        exit_price=1.12,
        gross_pnl=200.0,
        net_pnl=198.0,
        commissions=2.0,
        exit_reason="risk_exit",
        exit_slippage_bps=1.5,
        asset_class_fallback="FOREX",
        sleeve_fallback="fx_sleeve",
        side_fallback="LONG",
        entry_price_fallback=1.10,
        entry_signal_fallback=0.4,
        entry_confidence_fallback=0.7,
        governor_tier_fallback="yellow",
        governor_regime_fallback="carry",
        entry_time_fallback=entry_time,
        exit_time=exit_time,
        source="unit_test_fallback_exit",
    )

    assert len(tracker.closed_trades) == 1
    closed = tracker.closed_trades[0]
    assert closed["asset_class"] == "FOREX"
    assert closed["sleeve"] == "fx_sleeve"
    assert closed["entry_price"] == pytest.approx(1.10)
    assert closed["governor_regime"] == "carry"
    assert closed["source"] == "unit_test_fallback_exit"


def test_summary_aggregates_by_sleeve_and_asset_class(tmp_path):
    tracker = PerformanceAttributionTracker(data_dir=tmp_path, max_closed_trades=100)
    entry_time = datetime(2026, 1, 3, 10, 0, 0)
    exit_time = datetime(2026, 1, 3, 14, 0, 0)

    tracker.record_entry(
        symbol="MSFT",
        asset_class="EQUITY",
        sleeve="equities_sleeve",
        side="LONG",
        quantity=5,
        entry_price=200.0,
        entry_signal=0.5,
        entry_confidence=0.8,
        governor_tier="green",
        governor_regime="risk_on",
        entry_time=entry_time,
    )
    tracker.record_exit(
        symbol="MSFT",
        quantity=5,
        exit_price=210.0,
        gross_pnl=50.0,
        net_pnl=49.0,
        commissions=1.0,
        exit_reason="tp",
        exit_time=exit_time,
    )

    tracker.record_entry(
        symbol="CRYPTO:BTC/USDT",
        asset_class="CRYPTO",
        sleeve="crypto_sleeve",
        side="LONG",
        quantity=1,
        entry_price=50_000.0,
        entry_signal=0.7,
        entry_confidence=0.9,
        governor_tier="yellow",
        governor_regime="high_vol",
        entry_time=entry_time,
    )
    tracker.record_exit(
        symbol="CRYPTO:BTC/USDT",
        quantity=1,
        exit_price=49_000.0,
        gross_pnl=-1000.0,
        net_pnl=-1005.0,
        commissions=5.0,
        exit_reason="stop",
        exit_time=exit_time,
    )
    tracker.record_social_governor_impact(
        asset_class="EQUITY",
        regime="risk_off",
        blocked_alpha_opportunity=120.0,
        avoided_drawdown_estimate=260.0,
        hedge_cost_drag=18.0,
        policy_version="sshock-test",
        reason="verified_event_risk",
        event_id="macro_event",
    )

    summary = tracker.get_summary(lookback_days=365)
    assert summary["closed_trades"] == 2
    assert summary["net_pnl"] == pytest.approx(-956.0)
    assert summary["by_sleeve"]["equities_sleeve"]["trades"] == 1
    assert summary["by_sleeve"]["crypto_sleeve"]["trades"] == 1
    assert summary["by_sleeve"]["equities_sleeve"]["modeled_slippage_drag"] >= 0.0
    assert summary["by_sleeve"]["crypto_sleeve"]["modeled_slippage_drag"] >= 0.0
    assert summary["by_asset_class"]["EQUITY"]["trades"] == 1
    assert summary["by_asset_class"]["CRYPTO"]["trades"] == 1
    assert summary["social_governor"]["events"] == 1
    assert summary["social_governor"]["blocked_alpha_opportunity"] == pytest.approx(120.0)
    assert summary["social_governor"]["avoided_drawdown_estimate"] == pytest.approx(260.0)
