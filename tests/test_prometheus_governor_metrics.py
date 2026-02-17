"""Tests for governor-related Prometheus metric exports."""

import pytest

from monitoring import prometheus_metrics as pm


@pytest.mark.skipif(not pm.PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
def test_governor_metrics_emit_without_error():
    from prometheus_client import CollectorRegistry

    registry = CollectorRegistry()
    metrics = pm.PrometheusMetrics(port=0, registry=registry)

    metrics.update_governor_state(
        asset_class="EQUITY",
        regime="risk_off",
        tier_level=2,
        size_multiplier=0.5,
        signal_threshold_boost=0.05,
        confidence_boost=0.08,
        halt_entries=False,
        policy_version="v-test",
    )
    metrics.record_governor_transition(
        asset_class="EQUITY",
        regime="risk_off",
        from_tier="yellow",
        to_tier="orange",
    )
    metrics.record_governor_blocked_entry(
        asset_class="EQUITY",
        regime="risk_off",
        reason="threshold",
    )
    metrics.update_kill_switch(active=True, reason="dd_sharpe_breach")
    metrics.record_pretrade_gate_decision(
        asset_class="EQUITY",
        allowed=False,
        reason="max_order_notional",
    )
    metrics.record_execution_spread_gate_block(
        asset_class="EQUITY",
        regime="risk_off",
    )
    metrics.record_execution_slippage_budget_block(
        asset_class="CRYPTO",
        regime="high_vol",
    )
    metrics.record_execution_edge_gate_block(
        asset_class="EQUITY",
        regime="risk_off",
    )
    metrics.update_attribution_summary(
        {
            "lookback_days": 30,
            "by_sleeve": {
                "equities_sleeve": {
                    "trades": 12,
                    "net_pnl": 2500.0,
                    "modeled_execution_drag": 320.0,
                    "modeled_slippage_drag": 180.0,
                }
            },
        }
    )
    metrics.record_attribution_trade(
        sleeve="equities_sleeve",
        net_alpha=75.0,
        execution_drag=9.5,
        slippage_drag=4.0,
    )
    metrics.update_equity_validation(
        accepted=False,
        reason="outlier_rejected",
        raw_value=100_000.0,
        filtered_value=1_250_000.0,
        deviation_pct=0.92,
        suspect_count=1,
    )
    metrics.update_equity_reconciliation(
        broker_equity=1_255_000.0,
        modeled_equity=1_220_000.0,
        gap_dollars=35_000.0,
        gap_pct=0.0279,
        block_entries=True,
        breach_streak=2,
        healthy_streak=0,
        reason="gap_threshold_exceeded",
        breached=True,
        breach_event=True,
    )
    metrics.record_equity_reconciliation_entry_block(reason="gap_threshold_exceeded")
    metrics.update_social_risk_state(
        asset_class="EQUITY",
        regime="risk_off",
        social_risk_score=0.81,
        attention_z=2.7,
        sentiment_score=-0.74,
        gross_exposure_multiplier=0.45,
        block_new_entries=True,
    )
    metrics.record_social_shock_block(
        asset_class="EQUITY",
        regime="risk_off",
        reason="verified_event_risk",
    )
    metrics.record_social_decision(
        asset_class="EQUITY",
        regime="risk_off",
        result="block",
        policy_version="sshock-test",
    )
    metrics.update_prediction_verification(
        asset_class="EQUITY",
        regime="risk_off",
        event="fomc_shock",
        verified_probability=0.72,
        verified=True,
    )
    metrics.update_prediction_verification(
        asset_class="EQUITY",
        regime="risk_off",
        event="cpi_shock",
        verified_probability=0.0,
        verified=False,
        failure_reason="insufficient_independent_sources",
    )
    metrics.update_attribution_summary(
        {
            "lookback_days": 30,
            "by_sleeve": {},
            "social_governor": {
                "by_asset_class_regime": {
                    "EQUITY:risk_off": {
                        "asset_class": "EQUITY",
                        "regime": "risk_off",
                        "blocked_alpha_opportunity": 150.0,
                        "avoided_drawdown_estimate": 240.0,
                        "hedge_cost_drag": 22.0,
                    }
                }
            },
        }
    )
