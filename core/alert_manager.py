"""
core/alert_manager.py — Real-time Telegram Alert Manager

Sends operational alerts to a Telegram chat so operators know immediately
when critical events occur, without tailing log files.

Alert events:
    KILL_SWITCH   — kill switch has been tripped (trading halted)
    DRAWDOWN      — daily / session loss exceeds alert threshold
    STRESS_HALT   — intraday stress engine halted new entries
    TRADE_WIN     — exceptional single trade win  (≥ +$500 or ≥ +2%)
    TRADE_LOSS    — exceptional single trade loss (≤ -$300 or ≤ -1.5%)
    EOD_SUMMARY   — end-of-day digest summary
    ENGINE_ERROR  — unhandled exception in main engine loop

Configuration (env vars, all optional):
    APEX_TELEGRAM_BOT_TOKEN  — Telegram bot token (create via @BotFather)
    APEX_TELEGRAM_CHAT_ID    — Target chat / channel ID (negative for groups)
    APEX_ALERT_DRAWDOWN_PCT  — Alert when daily loss ≥ this % (default 3.0)
    APEX_ALERT_WIN_USD       — Alert on wins ≥ this $ amount (default 500)
    APEX_ALERT_LOSS_USD      — Alert on losses ≤ this $ amount (default -300)

Rate limiting: at most 1 alert per event_type per 10 minutes (prevents spam).
Falls back to logging-only when no Telegram token is configured.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_SEC = 600       # 10 min cooldown per alert type
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class AlertManager:
    """
    Async alert dispatcher.  Telegram is the primary channel;
    falls back to structured logging when unconfigured.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        drawdown_alert_pct: float = 3.0,
        win_alert_usd: float = 500.0,
        loss_alert_usd: float = -300.0,
    ) -> None:
        self._token = bot_token or os.getenv("APEX_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("APEX_TELEGRAM_CHAT_ID", "")
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

    # ------------------------------------------------------------------
    # Channel label for logging
    # ------------------------------------------------------------------

    @property
    def channel(self) -> str:
        return "telegram" if (self._token and self._chat_id) else "log_only"

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

        # Send to Telegram if configured
        if self._token and self._chat_id:
            try:
                await self._telegram_post(text)
            except Exception as exc:
                logger.warning("AlertManager: Telegram send failed: %s", exc)

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
