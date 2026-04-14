from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from quant_system.events.market import QuoteTick, TradeTick
from quant_system.events.order import OrderEvent

logger = logging.getLogger(__name__)

PricingMode = str


@dataclass(slots=True)
class ChasedOrderState:
    order_event: OrderEvent
    venue_order_id: str | None
    current_limit_price: float
    remaining_qty: float
    last_reprice_monotonic: float
    last_quote_ts: datetime | None = None
    done: bool = False
    task: asyncio.Task[None] | None = None
    lock: asyncio.Lock | None = None


class LimitOrderChaser:
    """
    Limit order executor that joins or improves the current spread and replaces
    resting orders when the quote moves.
    """

    def __init__(
        self,
        *,
        latest_market: dict[str, QuoteTick | TradeTick],
        submit_limit_order: Callable[[OrderEvent, float, float], Awaitable[str | None]],
        replace_limit_order: Callable[[ChasedOrderState, float], Awaitable[str | None]],
        pricing_mode: PricingMode = "join",
        chase_interval_seconds: float = 2.0,
        max_rest_seconds: float = 5.0,
        min_price_increment: float = 0.01,
    ) -> None:
        if chase_interval_seconds <= 0 or max_rest_seconds <= 0:
            raise ValueError("chase timing parameters must be positive")
        if pricing_mode not in {"join", "mid"}:
            raise ValueError("pricing_mode must be 'join' or 'mid'")
        self._latest_market = latest_market
        self._submit_limit_order = submit_limit_order
        self._replace_limit_order = replace_limit_order
        self._pricing_mode = pricing_mode
        self._chase_interval_seconds = float(chase_interval_seconds)
        self._max_rest_seconds = float(max_rest_seconds)
        self._min_price_increment = float(min_price_increment)
        self._states: dict[str, ChasedOrderState] = {}

    @property
    def states(self) -> dict[str, ChasedOrderState]:
        return dict(self._states)

    async def submit(self, order_event: OrderEvent) -> str | None:
        initial_price = self._target_limit_price(order_event)
        if initial_price is None:
            logger.warning("LimitOrderChaser has no market reference for %s", order_event.instrument_id)
            return None

        venue_order_id = await self._submit_limit_order(order_event, order_event.quantity, initial_price)
        now_monotonic = asyncio.get_running_loop().time()
        state = ChasedOrderState(
            order_event=order_event,
            venue_order_id=venue_order_id,
            current_limit_price=initial_price,
            remaining_qty=order_event.quantity,
            last_reprice_monotonic=now_monotonic,
            lock=asyncio.Lock(),
        )
        state.task = asyncio.create_task(self._chase_loop(order_event.order_id), name=f"chase-{order_event.order_id}")
        self._states[order_event.order_id] = state
        return venue_order_id

    async def update_quote(self, event: QuoteTick) -> None:
        for state in self._states.values():
            if state.order_event.instrument_id == event.instrument_id and not state.done:
                state.last_quote_ts = event.exchange_ts

    async def reprice_if_favorable(self, event: QuoteTick) -> None:
        """
        Immediately check and reprice if a new quote offers a better technical
        basis for the resting order (higher bid for buy, lower ask for sell).
        This eliminates 'Execution Lag' from fixed-interval chase loops.
        """
        for state in list(self._states.values()):
            if state.order_event.instrument_id != event.instrument_id or state.done:
                continue

            async with state.lock:
                if state.done or state.remaining_qty <= 1e-12:
                    continue

                # Throttle: don't reprice more often than chase_interval_seconds
                now_mono = asyncio.get_running_loop().time()
                if now_mono - state.last_reprice_monotonic < self._chase_interval_seconds:
                    continue

                new_limit_price = self._target_limit_price(state.order_event)
                if new_limit_price is None:
                    continue

                # Reprice if the new target price is BETTER than the current limit price
                # For BUY: new target is HIGHER than current limit
                # For SELL: new target is LOWER than current limit
                is_better = False
                if state.order_event.side == "buy":
                    is_better = new_limit_price > state.current_limit_price
                else:
                    is_better = new_limit_price < state.current_limit_price

                if is_better:
                    new_venue_order_id = await self._replace_limit_order(state, new_limit_price)
                    if new_venue_order_id:
                        state.venue_order_id = new_venue_order_id
                    state.current_limit_price = new_limit_price
                    state.last_reprice_monotonic = now_mono
                    logger.info(
                        "Favorable move detected for %s: Repriced to %.4f",
                        state.order_event.order_id,
                        new_limit_price,
                    )

    async def mark_execution(
        self,
        *,
        order_id: str,
        execution_status: str,
        remaining_qty: float | None,
        venue_order_id: str | None,
    ) -> None:
        state = self._states.get(order_id)
        if state is None:
            return
        if venue_order_id:
            state.venue_order_id = venue_order_id
        if remaining_qty is not None:
            state.remaining_qty = remaining_qty
        if execution_status in {"filled", "canceled", "rejected", "expired"} or state.remaining_qty <= 1e-12:
            await self._close_state(order_id)

    async def close(self) -> None:
        for order_id in list(self._states):
            await self._close_state(order_id)

    async def _close_state(self, order_id: str) -> None:
        state = self._states.pop(order_id, None)
        if state is None:
            return
        state.done = True
        if state.task is not None:
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)

    async def _chase_loop(self, order_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self._chase_interval_seconds)
                state = self._states.get(order_id)
                if state is None or state.done:
                    return
                async with state.lock:
                    if state.done or state.remaining_qty <= 1e-12:
                        return
                    new_limit_price = self._target_limit_price(state.order_event)
                    if new_limit_price is None:
                        continue
                    loop_now = asyncio.get_running_loop().time()
                    price_changed = self._should_replace(state.current_limit_price, new_limit_price)
                    stale_resting_order = (loop_now - state.last_reprice_monotonic) >= self._max_rest_seconds
                    if not price_changed and not stale_resting_order:
                        continue
                    new_venue_order_id = await self._replace_limit_order(state, new_limit_price)
                    if new_venue_order_id:
                        state.venue_order_id = new_venue_order_id
                    state.current_limit_price = new_limit_price
                    state.last_reprice_monotonic = loop_now
        except asyncio.CancelledError:
            return

    def _target_limit_price(self, order_event: OrderEvent) -> float | None:
        market_event = self._latest_market.get(order_event.instrument_id)
        if market_event is None:
            return None
        if isinstance(market_event, QuoteTick):
            if self._pricing_mode == "mid":
                raw_price = (market_event.bid + market_event.ask) / 2.0
            else:
                raw_price = market_event.bid if order_event.side == "buy" else market_event.ask
            return max(self._min_price_increment, round(raw_price / self._min_price_increment) * self._min_price_increment)
        
        # Safe attribute extraction for TradeTick, BarEvent, etc.
        raw_price = (
            getattr(market_event, "last_price", None) or 
            getattr(market_event, "close_price", None) or
            getattr(market_event, "price", 0.0)
        )
        return max(self._min_price_increment, round(raw_price / self._min_price_increment) * self._min_price_increment)

    def _should_replace(self, current_limit_price: float, new_limit_price: float) -> bool:
        return abs(new_limit_price - current_limit_price) >= self._min_price_increment / 2.0
