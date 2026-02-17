from monitoring.health_dashboard import HealthCheck, HealthDashboard, HealthStatus


def test_health_dashboard_dedupes_repeated_critical_alerts(tmp_path):
    dashboard = HealthDashboard(data_dir=str(tmp_path))
    check = HealthCheck(
        name="drawdown",
        status=HealthStatus.CRITICAL,
        message="Drawdown critical: 90.9%",
    )

    dashboard._process_alerts([check])
    dashboard._process_alerts([check])

    assert len(dashboard.alerts) == 1
