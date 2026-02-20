"""Tests for monitoring.alert_aggregator channel routing."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from monitoring.alert_aggregator import AlertAggregator, AlertSeverity


class _DummyResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_info_routes_to_slack_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], timeout: int):
        calls.append({"url": url, "payload": json, "timeout": timeout})
        return _DummyResponse(200)

    monkeypatch.setattr("monitoring.alert_aggregator.requests.post", fake_post)

    aggregator = AlertAggregator(
        cooldown_seconds=0,
        slack_webhook_url="https://hooks.slack.test/abc",
        pagerduty_routing_key="pd-routing",
    )
    aggregator.send_alert("system_ok", "System healthy", AlertSeverity.INFO)

    assert len(calls) == 1
    assert calls[0]["url"] == "https://hooks.slack.test/abc"


def test_warning_routes_to_slack_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], timeout: int):
        calls.append({"url": url, "payload": json, "timeout": timeout})
        return _DummyResponse(200)

    monkeypatch.setattr("monitoring.alert_aggregator.requests.post", fake_post)

    aggregator = AlertAggregator(
        cooldown_seconds=0,
        slack_webhook_url="https://hooks.slack.test/abc",
        pagerduty_routing_key="pd-routing",
    )
    aggregator.send_alert("drawdown_warn", "Drawdown near threshold", AlertSeverity.WARNING)

    assert len(calls) == 1
    assert calls[0]["url"] == "https://hooks.slack.test/abc"


def test_critical_routes_to_slack_and_pagerduty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_post(url: str, json: Dict[str, Any], timeout: int):
        calls.append(url)
        return _DummyResponse(200)

    monkeypatch.setattr("monitoring.alert_aggregator.requests.post", fake_post)

    aggregator = AlertAggregator(
        cooldown_seconds=0,
        slack_webhook_url="https://hooks.slack.test/abc",
        pagerduty_routing_key="pd-routing",
    )
    aggregator.send_alert("kill_switch", "Kill switch triggered", AlertSeverity.CRITICAL)

    assert "https://hooks.slack.test/abc" in calls
    assert "https://events.pagerduty.com/v2/enqueue" in calls
    assert len(calls) == 2


def test_slack_delivery_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: Dict[str, Any], timeout: int):
        return _DummyResponse(500)

    monkeypatch.setattr("monitoring.alert_aggregator.requests.post", fake_post)

    aggregator = AlertAggregator(cooldown_seconds=0, slack_webhook_url="https://hooks.slack.test/abc")

    with pytest.raises(RuntimeError, match="Slack delivery failed"):
        aggregator.send_alert("risk", "Bad state", AlertSeverity.WARNING)
