from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from itertools import count
from typing import Any, Mapping, Set

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from requests.adapters import HTTPAdapter
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, ReplaceOrderRequest
from alpaca.trading.stream import TradingStream

from core.symbols import ParsedSymbol, parse_symbol
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events.execution import ExecutionEvent
from quant_system.events.market import BarEvent, QuoteTick, TradeTick
from quant_system.events.order import OrderEvent
from quant_system.execution.sniper import ChasedOrderState, LimitOrderChaser

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Bridge canonical order flow into Alpaca paper trading and translate trade
    updates back into canonical execution events.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        event_bus: InMemoryEventBus,
        *,
        trading_stream: TradingStream | None = None,
        enable_limit_chaser: bool = True,
        chaser_pricing_mode: str = "join",
        limit_chaser: LimitOrderChaser | None = None,
    ) -> None:
        self._trading_client = trading_client
        # Raise urllib3 pool size to avoid connection-pool exhaustion under rapid order flow.
        _adapter = HTTPAdapter(pool_connections=1, pool_maxsize=25, max_retries=3)
        if hasattr(trading_client, "_session"):
            trading_client._session.mount("https://", _adapter)
        self._event_bus = event_bus
        self._sequence = count()
        self._latest_market: dict[str, QuoteTick | TradeTick | BarEvent] = {}
        self._orders_by_client_order_id: dict[str, OrderEvent] = {}
        self._orders_by_venue_order_id: dict[str, OrderEvent] = {}
        self._filled_qty_by_client_order_id: dict[str, float] = {}
        self._quote_sequence = count()
        self._trading_stream = trading_stream or TradingStream(
            trading_client._api_key,  # noqa: SLF001 - reuse existing authenticated client config
            trading_client._secret_key,  # noqa: SLF001
            paper=bool(getattr(trading_client, "_sandbox", True)),
            raw_data=True,
        )

        if limit_chaser is not None:
            self._limit_chaser = limit_chaser
            # Wire broker's market data and order callbacks into the injected chaser
            if hasattr(self._limit_chaser, "_latest_market"):
                self._limit_chaser._latest_market = self._latest_market
            if hasattr(self._limit_chaser, "_submit_limit_order"):
                self._limit_chaser._submit_limit_order = self._submit_chased_limit_order
            if hasattr(self._limit_chaser, "_replace_limit_order"):
                self._limit_chaser._replace_limit_order = self._replace_chased_limit_order
        elif enable_limit_chaser:
            self._limit_chaser = LimitOrderChaser(
                latest_market=self._latest_market,
                submit_limit_order=self._submit_chased_limit_order,
                replace_limit_order=self._replace_chased_limit_order,
                pricing_mode=chaser_pricing_mode,
            )
        else:
            self._limit_chaser = None

        # Wash-trade deduplication state.
        # _pending_cancels: order_id → monotonic timestamp when cancel was sent.
        # Prevents the same conflicting order from being cancelled twice within
        # the 90-second in-flight window and stops the per-cycle duplicate storm.
        self._pending_cancels: dict[str, float] = {}
        # Per-cycle set: cleared at the start of every _submit_direct_order batch
        # to catch same-cycle duplicates (same order_id cancelled twice in <1s).
        self._cycle_cancel_ids: Set[str] = set()

        self._subscriptions: tuple[Subscription, ...] = (
            self._event_bus.subscribe("bar", self._on_bar),
            self._event_bus.subscribe("quote_tick", self._on_quote_tick, is_async=True),
            self._event_bus.subscribe("trade_tick", self._on_trade_tick),
            self._event_bus.subscribe("order", self._on_order_event),
        )
        self._trading_stream.subscribe_trade_updates(self._on_trade_update)

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    async def run(self) -> None:
        await self._trading_stream._run_forever()  # noqa: SLF001 - keep a single asyncio loop for the bus

    async def stop(self) -> None:
        try:
            if self._limit_chaser is not None:
                await self._limit_chaser.close()
            await self._trading_stream.stop_ws()
        finally:
            await self._trading_stream.close()

    def close(self) -> None:
        for subscription in self._subscriptions:
            self._event_bus.unsubscribe(subscription.token)

    def _on_bar(self, event: BarEvent) -> None:
        self._latest_market[event.instrument_id] = event

    async def _on_quote_tick(self, event: QuoteTick) -> None:
        self._latest_market[event.instrument_id] = event
        if self._limit_chaser is not None:
            await self._limit_chaser.update_quote(event)
            await self._limit_chaser.reprice_if_favorable(event)

    def _on_trade_tick(self, event: TradeTick) -> None:
        self._latest_market[event.instrument_id] = event

    async def _on_order_event(self, event: OrderEvent) -> None:
        if event.order_action != "submit":
            return
        # New cycle starting — reset same-cycle cancel dedup set
        self._cycle_cancel_ids = set()

        if self._limit_chaser is not None and event.order_type == "market":
            venue_order_id = await self._limit_chaser.submit(event)
            if venue_order_id is not None:
                logger.info(
                    "Submitted chased limit order for client_order_id=%s venue_order_id=%s symbol=%s side=%s qty=%.6f",
                    event.order_id,
                    venue_order_id,
                    self._alpaca_symbol(event.instrument_id),
                    event.side,
                    event.quantity,
                )
                return
            # Limit chaser had no market ref — fall through to direct submit

        await self._submit_direct_order(event)

    async def _on_trade_update(self, update: Any) -> None:
        payload = self._coerce_update_payload(update)
        raw_event = str(payload.get("event") or "").strip().lower()
        if not raw_event:
            return
        if raw_event not in {"fill", "partial_fill", "canceled", "rejected", "expired"}:
            return

        order_payload = payload.get("order")
        if not isinstance(order_payload, Mapping):
            return

        client_order_id = str(order_payload.get("client_order_id") or "").strip()
        venue_order_id = str(order_payload.get("id") or "").strip()
        original_order = self._resolve_original_order(client_order_id, venue_order_id)
        instrument_id = self._normalize_instrument_id(str(order_payload.get("symbol") or ""))
        event_side = self._normalize_side(order_payload.get("side"))
        execution_status = self._normalize_execution_status(raw_event)
        exchange_ts = self._parse_timestamp(payload.get("timestamp"))
        fill_qty = self._extract_fill_qty(payload, order_payload, client_order_id)
        fill_price = self._extract_fill_price(payload, order_payload)
        fees = self._extract_fees(payload)
        slippage = self._compute_slippage(
            instrument_id=instrument_id,
            side=event_side,
            fill_price=fill_price,
            execution_status=execution_status,
        )
        order_id = original_order.order_id if original_order is not None else client_order_id or venue_order_id
        parent_order_id = self._parent_order_id(original_order, order_id)
        remaining_qty = self._extract_remaining_qty(order_payload, fill_qty)

        execution_event = ExecutionEvent(
            instrument_id=instrument_id,
            exchange_ts=exchange_ts,
            received_ts=exchange_ts,
            processed_ts=max(exchange_ts, self._utc_now()),
            sequence_id=next(self._sequence),
            source="alpaca.trade_updates",
            order_id=order_id,
            parent_order_id=parent_order_id,
            venue_order_id=venue_order_id or None,
            broker="alpaca",
            venue="alpaca",
            side=event_side,
            execution_status=execution_status,
            fill_qty=fill_qty,
            fill_price=fill_price,
            fees=fees,
            slippage=slippage,
            remaining_qty=remaining_qty,
            metadata={"alpaca_event": raw_event},
        )
        # A fill or cancel confirmation means the order is gone — remove from the
        # pending-cancel registry so the 90s guard doesn't block future conflicts.
        if execution_status in {"filled", "canceled"} and venue_order_id:
            self._pending_cancels.pop(venue_order_id, None)

        if self._limit_chaser is not None:
            await self._limit_chaser.mark_execution(
                order_id=order_id,
                execution_status=execution_status,
                remaining_qty=remaining_qty,
                venue_order_id=venue_order_id or None,
            )
        await self._event_bus.publish_async(execution_event)

    def _build_order_request(self, event: OrderEvent) -> MarketOrderRequest | LimitOrderRequest | None:
        symbol = self._alpaca_symbol(event.instrument_id)
        side = AlpacaOrderSide.BUY if event.side == "buy" else AlpacaOrderSide.SELL
        tif = self._map_time_in_force(event.time_in_force)
        # Alpaca crypto restrictions: GTC/IOC only (no DAY), and no short selling.
        parsed = self._parse_symbol(event.instrument_id)
        is_crypto = parsed is not None and parsed.asset_class.value == "CRYPTO"
        is_equity = parsed is not None and parsed.asset_class.value in ("US_EQUITY", "EQUITY")
        if is_crypto:
            if tif not in {AlpacaTimeInForce.GTC, AlpacaTimeInForce.IOC}:
                tif = AlpacaTimeInForce.GTC
            # Alpaca crypto paper/live does NOT support short selling.
            # Silently skip SELL orders when we have no position to close
            # (short opens). Sell-to-close orders on existing longs are fine.

        # Alpaca paper does NOT support fractional short-sells on equities.
        # Round SELL quantities down to whole shares so Kalman pairs can short
        # equity legs without hitting "fractional orders cannot be sold short".
        qty = event.quantity
        if is_equity and side == AlpacaOrderSide.SELL:
            whole = int(qty)
            if whole < qty and whole > 0:
                logger.debug(
                    "Rounding equity SELL qty %.6f → %d whole shares for %s",
                    qty, whole, symbol,
                )
                qty = float(whole)
            elif whole == 0:
                logger.debug("Equity SELL qty %.6f < 1 share — skipping %s", qty, symbol)
                return None

        if event.order_type in {"market", "market_on_close"}:
            if event.order_type == "market_on_close":
                tif = AlpacaTimeInForce.CLS
            return MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                type=AlpacaOrderType.MARKET,
                time_in_force=tif,
                client_order_id=event.order_id,
            )
        if event.order_type in {"limit", "limit_on_close"}:
            if event.limit_price is None:
                logger.warning("Rejecting limit order without limit_price for %s", event.order_id)
                return None
            if event.order_type == "limit_on_close":
                tif = AlpacaTimeInForce.CLS
            return LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                type=AlpacaOrderType.LIMIT,
                time_in_force=tif,
                limit_price=event.limit_price,
                client_order_id=event.order_id,
            )

        logger.warning("Unsupported Alpaca order type '%s' for %s", event.order_type, event.order_id)
        return None

    async def _submit_direct_order(self, event: OrderEvent) -> str | None:
        alpaca_request = self._build_order_request(event)
        if alpaca_request is None:
            return None

        self._orders_by_client_order_id[event.order_id] = event
        try:
            submitted_order = await asyncio.to_thread(self._trading_client.submit_order, alpaca_request)
        except Exception as exc:
            err_str = str(exc).lower()
            # Wash-trade rejection: Alpaca refuses to place an order on the same
            # side as an already-resting limit at the same price.  Extract the
            # conflicting order ID and cancel it — but only if we haven't already
            # sent a cancel for that ID within the last 90 seconds (prevents the
            # cancel→"pending cancel"→retry infinite loop).
            if "wash trade" in err_str or (
                "40310000" in err_str and "existing_order_id" in err_str
            ):
                # Purge stale entries (> 120s) before checking
                _now = time.monotonic()
                self._pending_cancels = {
                    oid: ts for oid, ts in self._pending_cancels.items()
                    if _now - ts < 120.0
                }
                try:
                    raw_json = str(exc)
                    start = raw_json.find("{")
                    body = json.loads(raw_json[start:]) if start != -1 else {}
                    conflict_id = body.get("existing_order_id", "")

                    if not conflict_id:
                        logger.warning(
                            "Wash-trade rejection on %s (no conflict ID in body): %s",
                            self._alpaca_symbol(event.instrument_id), exc,
                        )
                    elif conflict_id in self._cycle_cancel_ids:
                        # Same cycle duplicate — skip silently
                        logger.debug(
                            "Wash-trade: cancel already sent this cycle for %s — skipping",
                            conflict_id,
                        )
                    elif (
                        conflict_id in self._pending_cancels
                        and _now - self._pending_cancels[conflict_id] < 90.0
                    ):
                        logger.debug(
                            "Wash-trade: cancel already in-flight for %s (%.0fs ago) — skipping",
                            conflict_id, _now - self._pending_cancels[conflict_id],
                        )
                    else:
                        await asyncio.to_thread(
                            self._trading_client.cancel_order_by_id, conflict_id
                        )
                        self._pending_cancels[conflict_id] = _now
                        self._cycle_cancel_ids.add(conflict_id)
                        logger.warning(
                            "Wash-trade conflict on %s — cancelled conflicting order %s; will retry next cycle.",
                            self._alpaca_symbol(event.instrument_id),
                            conflict_id,
                        )
                except Exception as cancel_exc:
                    cancel_err = str(cancel_exc).lower()
                    if "42210000" in cancel_err or "already filled" in cancel_err:
                        # The conflicting order was filled before we could cancel it.
                        # Remove from pending (the fill means the position exists).
                        conflict_id_safe = locals().get("conflict_id", "")
                        if conflict_id_safe:
                            self._pending_cancels.pop(conflict_id_safe, None)
                        logger.info(
                            "Wash-trade: conflicting order already filled — no retry needed (%s)",
                            cancel_exc,
                        )
                    else:
                        logger.warning(
                            "Wash-trade: failed to cancel conflicting order for %s: %s",
                            self._alpaca_symbol(event.instrument_id), cancel_exc,
                        )
            elif "insufficient balance" in err_str:
                # Parse the available qty from Alpaca's error JSON and retry the
                # sell with that qty so stale ledger positions don't loop-reject.
                retried = False
                if event.side == "sell":
                    try:
                        start = str(exc).find("{")
                        body = json.loads(str(exc)[start:]) if start != -1 else {}
                        available = float(body.get("available", 0.0))
                        min_qty = 1e-6
                        if available > min_qty and available < event.quantity:
                            capped_qty = round(available * 0.999, 8)
                            if capped_qty > min_qty:
                                logger.warning(
                                    "Insufficient balance for %s sell: requested %.8f, available %.8f — "
                                    "retrying with %.8f",
                                    self._alpaca_symbol(event.instrument_id),
                                    event.quantity, available, capped_qty,
                                )
                                capped_event = OrderEvent(
                                    instrument_id=event.instrument_id,
                                    exchange_ts=event.exchange_ts,
                                    received_ts=event.received_ts,
                                    processed_ts=event.processed_ts,
                                    sequence_id=event.sequence_id,
                                    source=event.source,
                                    strategy_id=event.strategy_id,
                                    order_action=event.order_action,
                                    order_scope=event.order_scope,
                                    side=event.side,
                                    order_type=event.order_type,
                                    quantity=capped_qty,
                                    time_in_force=event.time_in_force,
                                    execution_algo=event.execution_algo,
                                    limit_price=event.limit_price,
                                    metadata=event.metadata,
                                )
                                retry_result = await self._submit_direct_order(capped_event)
                                retried = True
                                return retry_result
                    except Exception as parse_exc:
                        logger.debug("Could not parse available qty for retry: %s", parse_exc)
                if not retried:
                    logger.warning(
                        "Order %s rejected by Alpaca (%s): %s",
                        event.order_id,
                        self._alpaca_symbol(event.instrument_id),
                        exc,
                    )
            elif any(k in err_str for k in (
                "position_limit_exceeded", "forbidden",
                "qty must be", "42210000", "unprocessable", "422",
            )):
                logger.warning(
                    "Order %s rejected by Alpaca (%s): %s",
                    event.order_id,
                    self._alpaca_symbol(event.instrument_id),
                    exc,
                )
            else:
                logger.exception("Failed to submit order %s to Alpaca", event.order_id)
            await self._publish_rejection(event, str(exc))
            return None

        venue_order_id = str(getattr(submitted_order, "id", "") or "")
        if venue_order_id:
            self._orders_by_venue_order_id[venue_order_id] = event
        logger.info(
            "Submitted Alpaca order client_order_id=%s venue_order_id=%s symbol=%s side=%s qty=%.6f",
            event.order_id,
            venue_order_id or "pending",
            self._alpaca_symbol(event.instrument_id),
            event.side,
            event.quantity,
        )
        return venue_order_id or None

    async def _submit_chased_limit_order(self, event: OrderEvent, quantity: float, limit_price: float) -> str | None:
        limit_order_event = OrderEvent(
            instrument_id=event.instrument_id,
            exchange_ts=event.exchange_ts,
            received_ts=event.received_ts,
            processed_ts=event.processed_ts,
            sequence_id=event.sequence_id,
            source=event.source,
            event_id=event.event_id,
            order_id=event.order_id,
            order_action=event.order_action,
            order_scope=event.order_scope,
            side=event.side,
            order_type="limit",
            quantity=quantity,
            time_in_force=event.time_in_force,
            execution_algo=event.execution_algo,
            parent_order_id=event.parent_order_id,
            strategy_id=event.strategy_id,
            broker=event.broker,
            venue=event.venue,
            notional=event.notional,
            limit_price=limit_price,
            stop_price=event.stop_price,
            payload_version=event.payload_version,
            metadata=event.metadata,
        )
        return await self._submit_direct_order(limit_order_event)

    async def _replace_chased_limit_order(self, state: ChasedOrderState, new_limit_price: float) -> str | None:
        if not state.venue_order_id:
            return None
        replace_request = ReplaceOrderRequest(
            qty=int(state.remaining_qty) if float(state.remaining_qty).is_integer() else None,
            time_in_force=self._map_time_in_force(state.order_event.time_in_force),
            limit_price=new_limit_price,
        )
        try:
            replaced_order = await asyncio.to_thread(
                self._trading_client.replace_order_by_id,
                state.venue_order_id,
                replace_request,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            # Suppress terminal errors: order is gone from Alpaca's system (filled,
            # cancelled, or purged). "not found" / 40410000 means the venue_order_id
            # no longer exists and retrying will never succeed.
            if any(k in err_str for k in ("filled", "422", "unprocessable", "already", "not found", "40410000")):
                logger.debug(
                    "Chased order %s already filled/closed or not found — abandoning chase.",
                    state.venue_order_id,
                )
                # Signal the chaser to stop retrying this state by marking it done.
                state.done = True
            else:
                logger.warning(
                    "Failed to replace chased order client_order_id=%s venue_order_id=%s: %s",
                    state.order_event.order_id, state.venue_order_id, exc,
                )
            return None

        new_venue_order_id = str(getattr(replaced_order, "id", "") or state.venue_order_id)
        self._orders_by_client_order_id[state.order_event.order_id] = state.order_event
        self._orders_by_venue_order_id[new_venue_order_id] = state.order_event
        logger.info(
            "Repriced chased order client_order_id=%s old_venue_order_id=%s new_venue_order_id=%s limit=%.4f",
            state.order_event.order_id,
            state.venue_order_id,
            new_venue_order_id,
            new_limit_price,
        )
        return new_venue_order_id

    async def _publish_rejection(self, event: OrderEvent, reason: str) -> None:
        now = self._utc_now()
        rejection = ExecutionEvent(
            instrument_id=event.instrument_id,
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=next(self._sequence),
            source="alpaca.order_submit",
            order_id=event.order_id,
            parent_order_id=event.parent_order_id if event.order_scope == "child" else event.order_id,
            broker="alpaca",
            venue="alpaca",
            side=event.side,
            execution_status="rejected",
            fill_qty=0.0,
            fill_price=0.0,
            fees=0.0,
            slippage=0.0,
            remaining_qty=event.quantity,
            metadata={"reason": reason},
        )
        await self._event_bus.publish_async(rejection)

    def _resolve_original_order(self, client_order_id: str, venue_order_id: str) -> OrderEvent | None:
        original_order = self._orders_by_client_order_id.get(client_order_id)
        if original_order is not None:
            return original_order
        if venue_order_id:
            return self._orders_by_venue_order_id.get(venue_order_id)
        return None

    def _extract_fill_qty(
        self,
        payload: Mapping[str, Any],
        order_payload: Mapping[str, Any],
        client_order_id: str,
    ) -> float:
        event_qty = self._as_float(payload.get("qty"), default=None)
        if event_qty is not None:
            previous_filled = self._filled_qty_by_client_order_id.get(client_order_id, 0.0)
            cumulative = previous_filled + event_qty
            self._filled_qty_by_client_order_id[client_order_id] = cumulative
            return event_qty

        cumulative_filled = self._as_float(order_payload.get("filled_qty"), default=0.0) or 0.0
        previous_filled = self._filled_qty_by_client_order_id.get(client_order_id, 0.0)
        incremental = max(0.0, cumulative_filled - previous_filled)
        self._filled_qty_by_client_order_id[client_order_id] = max(previous_filled, cumulative_filled)
        return incremental

    @staticmethod
    def _extract_fill_price(payload: Mapping[str, Any], order_payload: Mapping[str, Any]) -> float:
        return (
            AlpacaBroker._as_float(payload.get("price"), default=None)
            or AlpacaBroker._as_float(order_payload.get("filled_avg_price"), default=0.0)
            or 0.0
        )

    @staticmethod
    def _extract_fees(payload: Mapping[str, Any]) -> float:
        return (
            AlpacaBroker._as_float(payload.get("commission"), default=None)
            or AlpacaBroker._as_float(payload.get("fees"), default=0.0)
            or 0.0
        )

    @staticmethod
    def _extract_remaining_qty(order_payload: Mapping[str, Any], last_fill_qty: float) -> float | None:
        total_qty = AlpacaBroker._as_float(order_payload.get("qty"), default=None)
        cumulative_filled = AlpacaBroker._as_float(order_payload.get("filled_qty"), default=None)
        if total_qty is None:
            return None
        if cumulative_filled is None:
            return max(0.0, total_qty - last_fill_qty)
        return max(0.0, total_qty - cumulative_filled)

    def _compute_slippage(
        self,
        *,
        instrument_id: str,
        side: str,
        fill_price: float,
        execution_status: str,
    ) -> float:
        if execution_status not in {"partial_fill", "filled"} or fill_price <= 0:
            return 0.0

        market_event = self._latest_market.get(instrument_id)
        if market_event is None:
            return 0.0

        if isinstance(market_event, QuoteTick):
            reference_price = market_event.ask if side == "buy" else market_event.bid
        elif isinstance(market_event, TradeTick):
            reference_price = market_event.last_price
        else:
            reference_price = market_event.close_price

        if side == "buy":
            return fill_price - reference_price
        return reference_price - fill_price

    @staticmethod
    def _normalize_execution_status(raw_event: str) -> str:
        if raw_event == "fill":
            return "filled"
        if raw_event == "partial_fill":
            return "partial_fill"
        if raw_event == "canceled":
            return "canceled"
        if raw_event == "rejected":
            return "rejected"
        if raw_event == "expired":
            return "expired"
        raise ValueError(f"Unsupported Alpaca execution status: {raw_event}")

    @staticmethod
    def _normalize_side(value: Any) -> str:
        side = str(value or "").strip().lower()
        return "buy" if side == "buy" else "sell"

    @staticmethod
    def _parent_order_id(original_order: OrderEvent | None, fallback_order_id: str) -> str:
        if original_order is None:
            return fallback_order_id
        if original_order.order_scope == "child" and original_order.parent_order_id is not None:
            return original_order.parent_order_id
        return original_order.order_id

    @classmethod
    def _coerce_update_payload(cls, update: Any) -> Mapping[str, Any]:
        if hasattr(update, "model_dump"):
            model_payload = update.model_dump()
            if isinstance(model_payload, Mapping):
                return model_payload
        if isinstance(update, Mapping):
            data_payload = update.get("data")
            if isinstance(data_payload, Mapping):
                return data_payload
            return update
        raise TypeError("Unsupported Alpaca trade update payload type")

    @classmethod
    def _alpaca_symbol(cls, instrument_id: str) -> str:
        parsed = cls._parse_symbol(instrument_id)
        if parsed is None:
            return instrument_id
        if parsed.asset_class.value == "CRYPTO":
            return f"{parsed.base}/{parsed.quote}"
        if parsed.asset_class.value == "FOREX":
            return f"{parsed.base}{parsed.quote}"
        return parsed.base

    @classmethod
    def _normalize_instrument_id(cls, raw_symbol: str) -> str:
        parsed = cls._parse_symbol(raw_symbol)
        if parsed is None:
            return raw_symbol.strip().upper()
        return parsed.normalized

    @staticmethod
    def _parse_symbol(raw_symbol: str) -> ParsedSymbol | None:
        try:
            return parse_symbol(raw_symbol)
        except Exception:
            return None

    @staticmethod
    def _map_time_in_force(time_in_force: str) -> AlpacaTimeInForce:
        mapping = {
            "day": AlpacaTimeInForce.DAY,
            "gtc": AlpacaTimeInForce.GTC,
            "ioc": AlpacaTimeInForce.IOC,
            "fok": AlpacaTimeInForce.FOK,
            "opg": AlpacaTimeInForce.OPG,
            "cls": AlpacaTimeInForce.CLS,
        }
        return mapping.get(time_in_force, AlpacaTimeInForce.DAY)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            raw = str(value or "").strip()
            if not raw:
                return AlpacaBroker._utc_now()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None or dt.utcoffset() is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _as_float(value: Any, *, default: float | None) -> float | None:
        if value is None:
            return default
        converted = float(value)
        if converted != converted or converted in (float("inf"), float("-inf")):
            raise ValueError("Numeric Alpaca payload field must be finite")
        return converted

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
