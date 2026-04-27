"""
IBKRBroker — routes FOREX (and optionally equity) order events to IBKR TWS.

Listens to "order" events on the event bus, filters for FOREX instruments,
delegates execution to IBKRConnector, and publishes ExecutionEvents back.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from itertools import count
from typing import Any, Optional

from core.symbols import AssetClass, parse_symbol
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events.execution import ExecutionEvent
from quant_system.events.order import OrderEvent

logger = logging.getLogger(__name__)


class IBKRBroker:
    """
    Execution adapter that processes OrderEvents for FOREX instruments and
    routes them to an IBKRConnector instance.

    Designed to run alongside AlpacaBroker: Alpaca handles crypto + equities,
    IBKR handles forex (and optionally equity when TWS is live).
    """

    HANDLED_ASSET_CLASSES = {AssetClass.FOREX}

    def __init__(
        self,
        ibkr_connector: Any,
        event_bus: InMemoryEventBus,
    ) -> None:
        self._ibkr = ibkr_connector
        self._event_bus = event_bus
        self._sequence = count()
        self._subscription: Optional[Subscription] = None

    def start(self) -> None:
        self._subscription = self._event_bus.subscribe(
            "order", self._on_order_event, is_async=True
        )
        logger.info("IBKRBroker started — handling FOREX order events via IBKR TWS")

    def close(self) -> None:
        if self._subscription is not None:
            self._event_bus.unsubscribe(self._subscription.token)
            self._subscription = None

    async def _on_order_event(self, event: Any) -> None:
        if not isinstance(event, OrderEvent):
            return
        if event.order_action != "submit":
            return

        try:
            parsed = parse_symbol(event.instrument_id)
        except Exception:
            return

        if parsed.asset_class not in self.HANDLED_ASSET_CLASSES:
            return

        symbol = parsed.normalized
        side = event.side.upper()
        quantity = float(event.quantity)
        confidence = float(getattr(event, "confidence", 0.5) or 0.5)

        logger.info(
            "IBKRBroker: routing %s %s qty=%.6f via IBKR TWS",
            side, symbol, quantity,
        )

        result = await self._ibkr.execute_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            confidence=confidence,
        )

        now = datetime.now(timezone.utc)
        if result:
            fill_price = float(result.get("price", 0.0) or 0.0)
            fill_qty = quantity if result.get("status") == "FILLED" else 0.0
            execution_status = "filled" if result.get("status") == "FILLED" else "new"
            venue_order_id = str(result.get("order_id", ""))
        else:
            fill_price = 0.0
            fill_qty = 0.0
            execution_status = "rejected"
            venue_order_id = None

        execution_event = ExecutionEvent(
            instrument_id=symbol,
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=next(self._sequence),
            source="ibkr.broker",
            order_id=event.order_id,
            parent_order_id=None,
            venue_order_id=venue_order_id or None,
            broker="ibkr",
            venue="ibkr",
            side=event.side,
            execution_status=execution_status,
            fill_qty=fill_qty,
            fill_price=fill_price,
            fees=0.0,
            slippage=0.0,
            remaining_qty=quantity - fill_qty,
            metadata={"ibkr_result": str(result)},
        )
        await self._event_bus.publish_async(execution_event)

        if execution_status == "rejected":
            logger.warning(
                "IBKRBroker: order rejected for %s side=%s qty=%.6f",
                symbol, side, quantity,
            )
        else:
            logger.info(
                "IBKRBroker: %s %s qty=%.6f fill_price=%.5f status=%s",
                side, symbol, quantity, fill_price, execution_status,
            )
