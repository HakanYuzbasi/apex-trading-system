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
