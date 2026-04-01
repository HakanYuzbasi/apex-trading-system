"""
core/alert_manager.py — Real-time Telegram + Slack Alert Manager

Sends operational alerts to Telegram (primary) and/or Slack (secondary)
when critical events occur, without tailing log files.

Alert events:
    KILL_SWITCH         — kill switch has been tripped (trading halted)
    DRAWDOWN            — daily / session loss exceeds alert threshold
    STRESS_HALT         — intraday stress engine halted new entries
    TRADE_WIN           — exceptional single trade win  (≥ +$500 or ≥ +2%)
    TRADE_LOSS          — exceptional single trade loss (≤ -$300 or ≤ -1.5%)
    EOD_SUMMARY         — end-of-day digest summary
    ENGINE_ERROR        — unhandled exception in main engine loop
    MODEL_DRIFT         — ML model IC / hit-rate has degraded below threshold
    REGIME_ALERT        — critical market regime transition detected
    EXECUTION_QUALITY   — fill slippage P95 has spiked above threshold

Configuration (env vars, all optional):
    APEX_TELEGRAM_BOT_TOKEN  — Telegram bot token (create via @BotFather)
    APEX_TELEGRAM_CHAT_ID    — Target chat / channel ID (negative for groups)
    APEX_SLACK_WEBHOOK_URL   — Slack incoming webhook URL (optional secondary)
    APEX_ALERT_DRAWDOWN_PCT  — Alert when daily loss ≥ this % (default 3.0)
    APEX_ALERT_WIN_USD       — Alert on wins ≥ this $ amount (default 500)
    APEX_ALERT_LOSS_USD      — Alert on losses ≤ this $ amount (default -300)

Rate limiting: at most 1 alert per event_type per 10 minutes (prevents spam).
Alert history: last 100 alerts retained in memory for dashboard display.
Falls back to logging-only when no Telegram token is configured.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_SEC = 600       # 10 min cooldown per alert type
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_HISTORY_MAX = 100          # max alerts kept in memory


@dataclass
class AlertRecord:
    """Single alert entry stored in the history buffer."""
    event_type: str
    text: str
    ts: float = field(default_factory=time.time)
    channel: str = "log"    # "telegram" | "slack" | "telegram+slack" | "log"

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "message": self.text.replace("*", "").replace("`", "").replace("\n", " ")[:200],
            "ts": self.ts,
            "channel": self.channel,
        }


class AlertManager:
    """
    Async alert dispatcher.  Telegram is the primary channel;
    Slack is an optional secondary; falls back to structured logging.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        drawdown_alert_pct: float = 3.0,
        win_alert_usd: float = 500.0,
        loss_alert_usd: float = -300.0,
    ) -> None:
        self._token = bot_token or os.getenv("APEX_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("APEX_TELEGRAM_CHAT_ID", "")
        self._slack_url = slack_webhook_url or os.getenv("APEX_SLACK_WEBHOOK_URL", "")
        self.drawdown_alert_pct = float(
            os.getenv("APEX_ALERT_DRAWDOWN_PCT", str(drawdown_alert_pct))
        )
        self.win_alert_usd = float(
            os.getenv("APEX_ALERT_WIN_USD", str(win_alert_usd))
        )
        self.loss_alert_usd = float(
            os.getenv("APEX_ALERT_LOSS_USD", str(loss_alert_usd))
        )
        self._last_sent: Dict[str, float] = {}   # event_type → epoch
        self._history: Deque[AlertRecord] = deque(maxlen=_HISTORY_MAX)

    # ------------------------------------------------------------------
    # Channel label + history
    # ------------------------------------------------------------------

    @property
    def channel(self) -> str:
        has_tg = bool(self._token and self._chat_id)
        has_sl = bool(self._slack_url)
        if has_tg and has_sl:
            return "telegram+slack"
        if has_tg:
            return "telegram"
        if has_sl:
            return "slack"
        return "log_only"

    def get_recent_alerts(self, n: int = 50) -> List[Dict]:
        """Return last n alerts from the in-memory history buffer."""
        items = list(self._history)
        return [a.to_dict() for a in reversed(items[-n:])]

    # ------------------------------------------------------------------
    # High-level event methods (call from execution_loop)
    # ------------------------------------------------------------------

    async def send_kill_switch_alert(self, reason: str, session_pnl: float) -> None:
        text = (
            "🛑 *KILL SWITCH TRIPPED*\n"
            f"Reason: `{reason}`\n"
            f"Session P&L: `{session_pnl:+.2f} USD`\n"
            "_Trading is now halted. Manual reset required._"
        )
        await self._send("KILL_SWITCH", text)

    async def send_drawdown_alert(self, daily_loss_pct: float, daily_loss_usd: float) -> None:
        text = (
            f"⚠️ *DRAWDOWN ALERT*\n"
            f"Daily loss: `{daily_loss_pct:.2f}%` ({daily_loss_usd:+.0f} USD)\n"
            f"Threshold: `{self.drawdown_alert_pct:.1f}%`"
        )
        await self._send("DRAWDOWN", text)

    async def send_stress_alert(
        self,
        scenario: str,
        action: str,
        portfolio_return: float,
        candidates: List[str],
    ) -> None:
        cand_str = ", ".join(candidates[:5]) or "none"
        text = (
            f"🚨 *STRESS ENGINE: {action.upper()}*\n"
            f"Worst scenario: `{scenario}`\n"
            f"Portfolio stress return: `{portfolio_return:.1%}`\n"
            f"Unwind candidates: `{cand_str}`"
        )
        await self._send("STRESS_HALT", text)

    async def send_trade_alert(
        self,
        symbol: str,
        side: str,
        pnl_usd: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        if pnl_usd >= self.win_alert_usd:
            icon, event_type = "🏆", "TRADE_WIN"
            label = "EXCEPTIONAL WIN"
        elif pnl_usd <= self.loss_alert_usd:
            icon, event_type = "💸", "TRADE_LOSS"
            label = "EXCEPTIONAL LOSS"
        else:
            return   # not noteworthy
        text = (
            f"{icon} *{label}*\n"
            f"Symbol: `{symbol}` ({side})\n"
            f"P&L: `{pnl_usd:+.2f} USD` ({pnl_pct:+.2f}%)\n"
            f"Reason: `{exit_reason}`"
        )
        await self._send(event_type, text)

    async def send_eod_summary(
        self,
        report_date: str,
        total_trades: int,
        realized_pnl: float,
        win_rate: Optional[float],
        recommendations: List[str],
    ) -> None:
        wr_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
        rec_str = "\n".join(f"• {r}" for r in recommendations) if recommendations else "• None"
        pnl_icon = "📈" if realized_pnl >= 0 else "📉"
        text = (
            f"{pnl_icon} *EOD Summary — {report_date}*\n"
            f"Trades: `{total_trades}` | Win rate: `{wr_str}`\n"
            f"Realized P&L: `{realized_pnl:+.2f} USD`\n"
            f"*Top recommendations:*\n{rec_str}"
        )
        await self._send("EOD_SUMMARY", text, rate_limit_sec=3600)  # max once per hour

    async def send_model_drift_alert(
        self,
        ic_current: float,
        hit_rate: float,
        consecutive_degraded: int,
        regime: str = "unknown",
    ) -> None:
        """Alert when ML model IC / hit-rate has degraded below healthy thresholds."""
        status = "CRITICAL" if consecutive_degraded >= 3 else "WARNING"
        text = (
            f"🧠 *MODEL DRIFT {status}*\n"
            f"IC current: `{ic_current:.4f}` | Hit rate: `{hit_rate:.1%}`\n"
            f"Consecutive degraded windows: `{consecutive_degraded}`\n"
            f"Regime: `{regime}`\n"
            "_Auto-retrain has been triggered._"
        )
        await self._send("MODEL_DRIFT", text, rate_limit_sec=_RATE_LIMIT_SEC)

    async def send_regime_alert(
        self,
        from_regime: str,
        to_regime: str,
        severity: str,
        confidence: float = 0.0,
    ) -> None:
        """Alert on a critical market regime transition."""
        icon = "🔴" if severity in ("critical", "emergency") else "🟡"
        text = (
            f"{icon} *REGIME ALERT [{severity.upper()}]*\n"
            f"Transition: `{from_regime}` → `{to_regime}`\n"
            f"Confidence: `{confidence:.0%}`\n"
            "_Position sizing automatically reduced._"
        )
        await self._send("REGIME_ALERT", text)

    async def send_execution_quality_alert(
        self,
        worst_symbol: str,
        slippage_p95_bps: float,
        degraded_count: int,
        threshold_bps: float = 30.0,
    ) -> None:
        """Alert when fill slippage P95 spikes above threshold."""
        text = (
            f"⚡ *EXECUTION QUALITY DEGRADED*\n"
            f"Worst symbol: `{worst_symbol}` — P95 slippage `{slippage_p95_bps:.1f} bps`\n"
            f"Symbols above {threshold_bps:.0f} bps threshold: `{degraded_count}`\n"
            "_Sizing penalty applied automatically._"
        )
        await self._send("EXECUTION_QUALITY", text, rate_limit_sec=1800)  # 30 min

    async def send_engine_error(self, error: str, context: str = "") -> None:
        text = (
            f"🔴 *ENGINE ERROR*\n"
            f"```\n{str(error)[:300]}\n```"
            + (f"\nContext: `{context[:100]}`" if context else "")
        )
        await self._send("ENGINE_ERROR", text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_rate_limited(self, event_type: str, rate_limit_sec: float) -> bool:
        last = self._last_sent.get(event_type, 0.0)
        return (time.time() - last) < rate_limit_sec

    async def _send(
        self, event_type: str, text: str, rate_limit_sec: float = _RATE_LIMIT_SEC
    ) -> None:
        if self._is_rate_limited(event_type, rate_limit_sec):
            logger.debug("AlertManager: %s suppressed (rate-limited)", event_type)
            return

        self._last_sent[event_type] = time.time()

        # Always log
        logger.info("ALERT [%s]: %s", event_type, text.replace("\n", " ").replace("*", "").replace("`", ""))

        sent_channels: List[str] = []

        # Send to Telegram if configured
        if self._token and self._chat_id:
            try:
                await self._telegram_post(text)
                sent_channels.append("telegram")
            except Exception as exc:
                logger.warning("AlertManager: Telegram send failed: %s", exc)

        # Send to Slack if configured
        if self._slack_url:
            try:
                await self._slack_post(text)
                sent_channels.append("slack")
            except Exception as exc:
                logger.warning("AlertManager: Slack send failed: %s", exc)

        # Record in history
        channel_label = "+".join(sent_channels) if sent_channels else "log"
        self._history.append(AlertRecord(event_type=event_type, text=text, channel=channel_label))

    async def _slack_post(self, text: str) -> None:
        import json as _json
        import urllib.request
        import urllib.error

        # Strip Telegram Markdown; Slack uses plain text in payload
        plain = text.replace("*", "").replace("`", "").replace("_", "")
        payload = _json.dumps({"text": plain}).encode()

        def _post() -> int:
            req = urllib.request.Request(
                self._slack_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status

        status = await asyncio.to_thread(_post)
        if status not in (200, 204):
            logger.warning("AlertManager: Slack returned HTTP %d", status)

    async def _telegram_post(self, text: str) -> None:
        import urllib.request
        import urllib.parse
        import urllib.error

        url = _TELEGRAM_API.format(token=self._token)
        payload = urllib.parse.urlencode({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }).encode()

        # Run blocking urllib in thread to keep async-safe
        def _post() -> int:
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status

        status = await asyncio.to_thread(_post)
        if status != 200:
            logger.warning("AlertManager: Telegram returned HTTP %d", status)
