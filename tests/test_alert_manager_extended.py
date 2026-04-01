"""
Tests for extended core/alert_manager.py:
  - Slack webhook support
  - Alert history buffer (get_recent_alerts)
  - New event methods: send_model_drift_alert, send_regime_alert, send_execution_quality_alert
  - channel property with all combinations
  - AlertRecord.to_dict()
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from core.alert_manager import AlertManager, AlertRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(
    token: str = "",
    chat_id: str = "",
    slack_url: str = "",
) -> AlertManager:
    return AlertManager(bot_token=token, chat_id=chat_id, slack_webhook_url=slack_url)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# AlertRecord
# ---------------------------------------------------------------------------

class TestAlertRecord:
    def test_to_dict_has_required_keys(self):
        rec = AlertRecord(event_type="KILL_SWITCH", text="halted", channel="telegram")
        d = rec.to_dict()
        assert "event_type" in d
        assert "message" in d
        assert "ts" in d
        assert "channel" in d

    def test_message_truncated_to_200(self):
        long_text = "A" * 500
        rec = AlertRecord(event_type="TEST", text=long_text)
        assert len(rec.to_dict()["message"]) <= 200

    def test_markdown_stripped_from_message(self):
        rec = AlertRecord(event_type="TEST", text="*bold* `code` _italic_")
        msg = rec.to_dict()["message"]
        assert "*" not in msg
        assert "`" not in msg

    def test_channel_preserved(self):
        rec = AlertRecord(event_type="X", text="t", channel="telegram+slack")
        assert rec.to_dict()["channel"] == "telegram+slack"


# ---------------------------------------------------------------------------
# channel property
# ---------------------------------------------------------------------------

class TestChannelProperty:
    def test_log_only_when_no_credentials(self):
        m = _make_manager()
        assert m.channel == "log_only"

    def test_telegram_when_only_tg_configured(self):
        m = _make_manager(token="tok", chat_id="123")
        assert m.channel == "telegram"

    def test_slack_when_only_slack_configured(self):
        m = _make_manager(slack_url="https://hooks.slack.com/x")
        assert m.channel == "slack"

    def test_telegram_plus_slack_when_both(self):
        m = _make_manager(token="tok", chat_id="123", slack_url="https://hooks.slack.com/x")
        assert m.channel == "telegram+slack"


# ---------------------------------------------------------------------------
# Alert history buffer
# ---------------------------------------------------------------------------

class TestAlertHistory:
    def test_empty_before_any_sends(self):
        m = _make_manager()
        assert m.get_recent_alerts() == []

    def test_alert_appended_after_send(self):
        m = _make_manager()
        _run(m._send("TEST_EVENT", "hello world"))
        hist = m.get_recent_alerts()
        assert len(hist) == 1
        assert hist[0]["event_type"] == "TEST_EVENT"

    def test_respects_n_limit(self):
        m = _make_manager()
        for i in range(10):
            _run(m._send(f"EV{i}", f"msg{i}", rate_limit_sec=0))
        assert len(m.get_recent_alerts(5)) == 5

    def test_most_recent_first(self):
        m = _make_manager()
        for i in range(3):
            _run(m._send(f"EV{i}", f"msg{i}", rate_limit_sec=0))
        hist = m.get_recent_alerts()
        # get_recent_alerts returns reversed (newest first)
        assert hist[0]["event_type"] == "EV2"

    def test_buffer_capped_at_100(self):
        m = _make_manager()
        for i in range(120):
            m._history.append(AlertRecord(event_type=f"EV{i}", text="t"))
        assert len(m._history) == 100  # deque maxlen

    def test_rate_limited_event_not_added(self):
        m = _make_manager()
        _run(m._send("RL_EVENT", "first"))
        _run(m._send("RL_EVENT", "second (should be suppressed)"))
        # Only 1 record in history
        assert sum(1 for h in m._history if h.event_type == "RL_EVENT") == 1


# ---------------------------------------------------------------------------
# send_model_drift_alert
# ---------------------------------------------------------------------------

class TestModelDriftAlert:
    def test_sends_without_error(self):
        m = _make_manager()
        _run(m.send_model_drift_alert(ic_current=0.005, hit_rate=0.48, consecutive_degraded=3))
        hist = m.get_recent_alerts()
        assert any(h["event_type"] == "MODEL_DRIFT" for h in hist)

    def test_text_contains_ic(self):
        m = _make_manager()
        _run(m.send_model_drift_alert(ic_current=0.0042, hit_rate=0.49, consecutive_degraded=2))
        assert any("0.0042" in h.text for h in m._history)

    def test_rate_limited_after_first(self):
        m = _make_manager()
        _run(m.send_model_drift_alert(ic_current=0.001, hit_rate=0.47, consecutive_degraded=2))
        _run(m.send_model_drift_alert(ic_current=0.001, hit_rate=0.47, consecutive_degraded=2))
        assert sum(1 for h in m._history if h.event_type == "MODEL_DRIFT") == 1


# ---------------------------------------------------------------------------
# send_regime_alert
# ---------------------------------------------------------------------------

class TestRegimeAlert:
    def test_sends_without_error(self):
        m = _make_manager()
        _run(m.send_regime_alert(from_regime="bull", to_regime="bear", severity="critical", confidence=0.82))
        assert any(h.event_type == "REGIME_ALERT" for h in m._history)

    def test_text_contains_regime_names(self):
        m = _make_manager()
        _run(m.send_regime_alert(from_regime="neutral", to_regime="volatile", severity="warning", confidence=0.70))
        assert any("volatile" in h.text.lower() for h in m._history)

    def test_rate_limited(self):
        m = _make_manager()
        _run(m.send_regime_alert("bull", "bear", "critical"))
        _run(m.send_regime_alert("bull", "bear", "critical"))
        assert sum(1 for h in m._history if h.event_type == "REGIME_ALERT") == 1


# ---------------------------------------------------------------------------
# send_execution_quality_alert
# ---------------------------------------------------------------------------

class TestExecutionQualityAlert:
    def test_sends_without_error(self):
        m = _make_manager()
        _run(m.send_execution_quality_alert(
            worst_symbol="AAPL", slippage_p95_bps=45.0, degraded_count=3
        ))
        assert any(h.event_type == "EXECUTION_QUALITY" for h in m._history)

    def test_text_contains_symbol_and_bps(self):
        m = _make_manager()
        _run(m.send_execution_quality_alert("TSLA", 52.3, 2))
        assert any("TSLA" in h.text for h in m._history)

    def test_rate_limited_at_30_min(self):
        m = _make_manager()
        _run(m.send_execution_quality_alert("X", 40.0, 1))
        _run(m.send_execution_quality_alert("X", 40.0, 1))
        assert sum(1 for h in m._history if h.event_type == "EXECUTION_QUALITY") == 1


# ---------------------------------------------------------------------------
# Slack webhook (mocked)
# ---------------------------------------------------------------------------

class TestSlackWebhook:
    def test_slack_post_called_when_url_configured(self):
        m = _make_manager(slack_url="https://hooks.slack.com/test")
        with patch.object(m, "_slack_post", new_callable=AsyncMock) as mock_sl:
            _run(m._send("SLACK_TEST", "hello slack"))
            mock_sl.assert_called_once()

    def test_slack_not_called_when_url_empty(self):
        m = _make_manager()
        with patch.object(m, "_slack_post", new_callable=AsyncMock) as mock_sl:
            _run(m._send("NO_SLACK", "hi"))
            mock_sl.assert_not_called()

    def test_slack_failure_does_not_raise(self):
        m = _make_manager(slack_url="https://hooks.slack.com/test")
        with patch.object(m, "_slack_post", side_effect=Exception("connection error")):
            _run(m._send("SLACK_FAIL", "test"))  # should not raise
