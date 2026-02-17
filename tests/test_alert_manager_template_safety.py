import pytest

from monitoring.alert_manager import AlertCategory, AlertManager, AlertSeverity


@pytest.mark.asyncio
async def test_alert_template_missing_fields_do_not_drop_alert():
    manager = AlertManager(max_alerts=10)
    manager.set_state("vix", 31.2)
    manager.register_rule(
        rule_id="missing_field_template",
        name="Template Safety",
        condition=lambda: True,
        category=AlertCategory.MARKET,
        severity=AlertSeverity.WARNING,
        message_template="VIX={vix:.1f}, note={missing_field}",
        cooldown_seconds=0,
        metadata_provider=lambda: {"vix": manager.get_state("vix")},
    )

    triggered = await manager.check_rules()

    assert len(triggered) == 1
    assert "VIX=31.2" in triggered[0].message
    assert "{missing_field}" in triggered[0].message
