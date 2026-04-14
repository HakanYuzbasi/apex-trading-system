from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BaseEvent, ExecutionEvent

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Async Telegram Bot API notifier.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._event_bus = event_bus
        self._bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self._timeout = aiohttp.ClientTimeout(total=max(timeout_seconds, 0.1))
        self._session: aiohttp.ClientSession | None = None
        self._subscriptions: tuple[Subscription, ...] = (
            self._event_bus.subscribe("execution", self._on_execution, is_async=True),
            self._event_bus.subscribe("risk_limit_breach", self._on_risk_limit_breach, is_async=True),
        )

        if not self._bot_token or not self._chat_id:
            logger.warning("TelegramNotifier initialized without TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID; notifications are disabled")

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    async def close(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription.token)
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _on_execution(self, event: BaseEvent) -> None:
        if not isinstance(event, ExecutionEvent):
            return
        if event.execution_status != "filled":
            return
        await self._post_message(self._format_execution_message(event))

    async def _on_risk_limit_breach(self, event: BaseEvent) -> None:
        await self._post_message(self._format_risk_message(event))

    async def notify_system_event(self, title: str, detail: str) -> None:
        await self._post_message(self._format_system_message(title, detail))

    async def notify_text(self, text: str) -> None:
        await self._post_message(text)

    async def _post_message(self, content: str) -> None:
        if not self._bot_token or not self._chat_id:
            return
        session = await self._get_session()
        endpoint = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": content}
        try:
            async with session.post(endpoint, json=payload) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning("Telegram notification failed status=%s body=%s", response.status, body)
        except Exception:
            logger.exception("Failed to send Telegram notification")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    @staticmethod
    def _format_execution_message(event: ExecutionEvent) -> str:
        action = "BOUGHT" if event.side == "buy" else "SOLD"
        quantity = TelegramNotifier._format_quantity(event.fill_qty)
        return (
            f"\N{POLICE CARS REVOLVING LIGHT} {action} {quantity} {event.instrument_id} "
            f"@ ${event.fill_price:,.2f} | Fees: ${event.fees:,.2f}"
        )

    @staticmethod
    def _format_risk_message(event: BaseEvent) -> str:
        instrument = getattr(event, "instrument_id", "PORTFOLIO")
        reason = TelegramNotifier._metadata_value(event.metadata, "reason", "Risk limit breach")
        limit_name = TelegramNotifier._metadata_value(event.metadata, "limit_name", "unspecified_limit")
        return (
            f"\N{WARNING SIGN} RISK LIMIT BREACH | {instrument} | "
            f"{limit_name} | {reason}"
        )

    @staticmethod
    def _format_system_message(title: str, detail: str) -> str:
        return f"\N{ANTENNA WITH BARS} {title}\n{detail}"

    @staticmethod
    def _metadata_value(metadata: dict[str, Any], key: str, default: str) -> str:
        value = metadata.get(key, default)
        return str(value)

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        rounded = round(quantity)
        if abs(quantity - rounded) <= 1e-9:
            return str(int(rounded))
        return f"{quantity:,.6f}".rstrip("0").rstrip(".")
