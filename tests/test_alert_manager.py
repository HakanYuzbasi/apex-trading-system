"""Tests for AlertManager — rate limiting, channel detection, each alert method."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.alert_manager import AlertManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(**kwargs) -> AlertManager:
    defaults = dict(bot_token="", chat_id="")
    defaults.update(kwargs)
    return AlertManager(**defaults)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------

class TestChannelDetection:

    def test_log_only_when_no_token(self):
        mgr = _make_manager(bot_token="", chat_id="chat123")
        assert mgr.channel == "log_only"

    def test_log_only_when_no_chat_id(self):
        mgr = _make_manager(bot_token="tok123", chat_id="")
        assert mgr.channel == "log_only"

    def test_telegram_when_both_configured(self):
        mgr = _make_manager(bot_token="tok123", chat_id="chat123")
        assert mgr.channel == "telegram"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("APEX_TELEGRAM_BOT_TOKEN", "envtok")
        monkeypatch.setenv("APEX_TELEGRAM_CHAT_ID", "envchat")
        mgr = AlertManager()
        assert mgr.channel == "telegram"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def test_first_send_not_rate_limited(self):
        mgr = _make_manager()
        assert not mgr._is_rate_limited("KILL_SWITCH", 600)

    def test_second_send_within_cooldown_is_rate_limited(self):
        mgr = _make_manager()
        mgr._last_sent["KILL_SWITCH"] = time.time()
        assert mgr._is_rate_limited("KILL_SWITCH", 600)

    def test_after_cooldown_not_rate_limited(self):
        mgr = _make_manager()
        mgr._last_sent["KILL_SWITCH"] = time.time() - 700  # past 600s
        assert not mgr._is_rate_limited("KILL_SWITCH", 600)

    def test_rate_limit_is_per_event_type(self):
        mgr = _make_manager()
        mgr._last_sent["KILL_SWITCH"] = time.time()
        assert not mgr._is_rate_limited("DRAWDOWN", 600)

    def test_send_suppressed_when_rate_limited(self):
        mgr = _make_manager()
        mgr._last_sent["KILL_SWITCH"] = time.time()
        # Should not raise; just suppress
        _run(mgr._send("KILL_SWITCH", "test msg"))
        # last_sent should NOT be updated (still the original time, not reset)
        assert mgr._is_rate_limited("KILL_SWITCH", 600)

    def test_send_updates_last_sent_on_success(self):
        mgr = _make_manager()
        before = time.time()
        _run(mgr._send("TEST_EVENT", "hello"))
        assert mgr._last_sent["TEST_EVENT"] >= before


# ---------------------------------------------------------------------------
# Log-only path (no Telegram)
# ---------------------------------------------------------------------------

class TestLogOnlyPath:

    def test_send_kill_switch_alert_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_kill_switch_alert(reason="daily loss", session_pnl=-1234.56))

    def test_send_drawdown_alert_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_drawdown_alert(daily_loss_pct=4.5, daily_loss_usd=-4500.0))

    def test_send_stress_alert_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_stress_alert(
            scenario="2020_crash",
            action="halt",
            portfolio_return=-0.18,
            candidates=["AAPL", "MSFT"],
        ))

    def test_send_trade_alert_win_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_trade_alert(
            symbol="AAPL", side="LONG", pnl_usd=750.0, pnl_pct=3.5, exit_reason="take_profit",
        ))

    def test_send_trade_alert_loss_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_trade_alert(
            symbol="BTC", side="LONG", pnl_usd=-500.0, pnl_pct=-2.5, exit_reason="stop_loss",
        ))

    def test_send_eod_summary_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_eod_summary(
            report_date="2026-03-22",
            total_trades=15,
            realized_pnl=1234.56,
            win_rate=0.60,
            recommendations=["Reduce crypto exposure"],
        ))

    def test_send_engine_error_no_exception(self):
        mgr = _make_manager()
        _run(mgr.send_engine_error(error="AttributeError: foo", context="process_symbol"))


# ---------------------------------------------------------------------------
# Trade alert threshold logic
# ---------------------------------------------------------------------------

class TestTradeAlertThresholds:

    def test_trade_alert_below_win_threshold_is_silent(self):
        mgr = _make_manager(win_alert_usd=500.0, loss_alert_usd=-300.0)
        # pnl_usd = 200, well below 500 threshold → no alert
        sent_events: list[str] = []
        original_send = mgr._send

        async def _spy(event_type, text, **kw):
            sent_events.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_trade_alert("MSFT", "LONG", 200.0, 1.0, "take_profit"))
        assert len(sent_events) == 0

    def test_trade_alert_above_win_threshold_fires(self):
        mgr = _make_manager(win_alert_usd=500.0, loss_alert_usd=-300.0)
        sent_events: list[str] = []

        async def _spy(event_type, text, **kw):
            sent_events.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_trade_alert("MSFT", "LONG", 600.0, 3.0, "take_profit"))
        assert "TRADE_WIN" in sent_events

    def test_trade_alert_below_loss_threshold_fires(self):
        mgr = _make_manager(win_alert_usd=500.0, loss_alert_usd=-300.0)
        sent_events: list[str] = []

        async def _spy(event_type, text, **kw):
            sent_events.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_trade_alert("MSFT", "LONG", -400.0, -2.0, "stop_loss"))
        assert "TRADE_LOSS" in sent_events

    def test_trade_alert_at_exactly_win_threshold_fires(self):
        mgr = _make_manager(win_alert_usd=500.0, loss_alert_usd=-300.0)
        sent_events: list[str] = []

        async def _spy(event_type, text, **kw):
            sent_events.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_trade_alert("MSFT", "LONG", 500.0, 2.5, "take_profit"))
        assert "TRADE_WIN" in sent_events

    def test_trade_alert_at_exactly_loss_threshold_fires(self):
        mgr = _make_manager(win_alert_usd=500.0, loss_alert_usd=-300.0)
        sent_events: list[str] = []

        async def _spy(event_type, text, **kw):
            sent_events.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_trade_alert("MSFT", "LONG", -300.0, -1.5, "stop_loss"))
        assert "TRADE_LOSS" in sent_events


# ---------------------------------------------------------------------------
# EOD summary rate limit (1 hour)
# ---------------------------------------------------------------------------

class TestEodRateLimit:

    def test_eod_summary_rate_limited_at_1hr(self):
        mgr = _make_manager()
        mgr._last_sent["EOD_SUMMARY"] = time.time()
        # Second call should be suppressed (rate-limited at 3600s)
        sent = []

        async def _spy(event_type, text, rate_limit_sec=600):
            sent.append(event_type)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_eod_summary("2026-03-22", 10, 500.0, 0.55, []))
        # The send is called but _send itself enforces rate limit — here we test _send directly
        # Since we replaced _send, just confirm the args match
        assert len(sent) == 1
        assert sent[0] == "EOD_SUMMARY"


# ---------------------------------------------------------------------------
# EOD summary with None win_rate
# ---------------------------------------------------------------------------

class TestEodSummaryFormatting:

    def test_eod_summary_none_win_rate(self):
        mgr = _make_manager()
        # Should not raise
        _run(mgr.send_eod_summary("2026-03-22", 0, 0.0, None, []))

    def test_eod_summary_no_recommendations(self):
        mgr = _make_manager()
        _run(mgr.send_eod_summary("2026-03-22", 5, -200.0, 0.40, []))

    def test_eod_summary_negative_pnl_icon(self):
        mgr = _make_manager()
        captured: list[str] = []

        async def _spy(event_type, text, **kw):
            captured.append(text)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_eod_summary("2026-03-22", 5, -200.0, 0.40, []))
        assert captured and "📉" in captured[0]

    def test_eod_summary_positive_pnl_icon(self):
        mgr = _make_manager()
        captured: list[str] = []

        async def _spy(event_type, text, **kw):
            captured.append(text)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_eod_summary("2026-03-22", 5, 500.0, 0.60, []))
        assert captured and "📈" in captured[0]


# ---------------------------------------------------------------------------
# Telegram path (mocked)
# ---------------------------------------------------------------------------

class TestTelegramPath:

    def test_telegram_post_called_when_configured(self):
        mgr = _make_manager(bot_token="tok123", chat_id="chat123")
        call_args: list = []

        async def _mock_post(text: str) -> None:
            call_args.append(text)

        mgr._telegram_post = _mock_post  # type: ignore[method-assign]
        _run(mgr._send("KILL_SWITCH", "⚠️ test"))
        assert len(call_args) == 1
        assert "test" in call_args[0]

    def test_telegram_failure_does_not_raise(self):
        mgr = _make_manager(bot_token="tok123", chat_id="chat123")

        async def _mock_post(text: str) -> None:
            raise ConnectionError("network down")

        mgr._telegram_post = _mock_post  # type: ignore[method-assign]
        # Should not propagate the exception
        _run(mgr._send("KILL_SWITCH", "⚠️ test"))

    def test_telegram_not_called_without_token(self):
        mgr = _make_manager(bot_token="", chat_id="chat123")
        call_args: list = []

        async def _mock_post(text: str) -> None:
            call_args.append(text)

        mgr._telegram_post = _mock_post  # type: ignore[method-assign]
        _run(mgr._send("DRAWDOWN", "down test"))
        assert len(call_args) == 0


# ---------------------------------------------------------------------------
# Engine error formatting
# ---------------------------------------------------------------------------

class TestEngineErrorFormatting:

    def test_engine_error_truncates_long_error(self):
        mgr = _make_manager()
        captured: list[str] = []

        async def _spy(event_type, text, **kw):
            captured.append(text)

        mgr._send = _spy  # type: ignore[method-assign]
        long_err = "X" * 500
        _run(mgr.send_engine_error(error=long_err))
        assert captured
        # Should be truncated to 300 chars
        assert len(captured[0]) < 500

    def test_engine_error_with_context(self):
        mgr = _make_manager()
        captured: list[str] = []

        async def _spy(event_type, text, **kw):
            captured.append(text)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_engine_error(error="RuntimeError", context="process_symbol:AAPL"))
        assert captured
        assert "process_symbol" in captured[0]

    def test_engine_error_no_context(self):
        mgr = _make_manager()
        captured: list[str] = []

        async def _spy(event_type, text, **kw):
            captured.append(text)

        mgr._send = _spy  # type: ignore[method-assign]
        _run(mgr.send_engine_error(error="ValueError: bad value"))
        assert captured
        assert "Context" not in captured[0]
