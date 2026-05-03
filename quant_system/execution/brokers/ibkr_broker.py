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
from quant_system.events.market import BarEvent

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
        forex_symbols_fn: Optional[Callable[[], tuple[str, ...]]] = None,
    ) -> None:
        self._ibkr = ibkr_connector
        self._event_bus = event_bus
        self._forex_symbols_fn = forex_symbols_fn
        self._sequence = count()
        self._subscription: Optional[Subscription] = None
        self._poll_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._subscription = self._event_bus.subscribe(
            "order", self._on_order_event, is_async=True
        )
        # Start background polling as a non-blocking asyncio task
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("IBKRBroker started — handling FOREX order events and background polling")

    def stop(self) -> None:
        if self._subscription is not None:
            self._event_bus.unsubscribe(self._subscription.token)
            self._subscription = None
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Background polling loop for IBKR data (e.g. Forex)."""
        _poll_seq = 0
        poll_interval = 60.0  # As requested: 60s loop
        
        while True:
            try:
                if not self._forex_symbols_fn or not self._ibkr.is_connected():
                    await asyncio.sleep(10.0)
                    continue

                symbols = self._forex_symbols_fn()
                if not symbols:
                    await asyncio.sleep(10.0)
                    continue

                for symbol in symbols:
                    try:
                        # Use a timeout to prevent hanging the task if IBKR is slow
                        quote = await asyncio.wait_for(self._ibkr.get_quote(symbol), timeout=10.0)
                        if not quote or not quote.get("mid"):
                            continue
                        
                        mid = float(quote["mid"])
                        now = datetime.now(timezone.utc)
                        bar = BarEvent(
                            instrument_id=symbol,
                            exchange_ts=now,
                            received_ts=now,
                            processed_ts=now,
                            sequence_id=_poll_seq,
                            source="ibkr.forex.poll",
                            open_price=mid,
                            high_price=mid,
                            low_price=mid,
                            close_price=mid,
                            volume=1.0,
                            metadata={
                                "venue": "ibkr",
                                "bid": quote.get("bid", mid),
                                "ask": quote.get("ask", mid),
                            },
                        )
                        _poll_seq += 1
                        await self._event_bus.publish_async(bar)
                        logger.debug("Forex bar published: %s mid=%.5f", symbol, mid)
                    except asyncio.TimeoutError:
                        logger.warning(f"Forex poll timeout for {symbol}")
                    except Exception as e:
                        logger.debug(f"Forex poll error for {symbol}: {e}")

                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"IBKRBroker poll loop error: {e}")
                await asyncio.sleep(10.0)

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
