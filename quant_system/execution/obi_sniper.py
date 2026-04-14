from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

import numexpr as ne
import numpy as np

from quant_system.core.bus import InMemoryEventBus
from quant_system.events.market import OrderBookTick, QuoteTick, TradeTick
from quant_system.events.order import OrderEvent
from quant_system.execution.sniper import ChasedOrderState, LimitOrderChaser
from quant_system.execution.fast_math import calculate_obi_cython
from quant_system.execution.microstructure import VPINCalculator

logger = logging.getLogger(__name__)


async def _noop_submit(*_: Any) -> str | None:
    return None


async def _noop_replace(*_: Any) -> str | None:
    return None


class OBISniper(LimitOrderChaser):
    """
    Advanced Limit Order Chaser with Order Book Imbalance (OBI) awareness.
    Uses OBI to switch between 'Aggressive' and 'Patient' execution modes.

    OBI = (BidSize - AskSize) / (BidSize + AskSize)
    - OBI > 0.7: Buying Pressure -> Aggressive (Join Ask)
    - OBI < -0.7: Selling Pressure -> Patient (Stay at Bid)
    """

    def __init__(
        self,
        *,
        event_bus: InMemoryEventBus | None = None,
        latest_market: dict[str, QuoteTick | TradeTick | OrderBookTick] | None = None,
        submit_limit_order: Callable[[OrderEvent, float, float], Awaitable[str | None]] | None = None,
        replace_limit_order: Callable[[ChasedOrderState, float], Awaitable[str | None]] | None = None,
        pricing_mode: str = "join",
        chase_interval_seconds: float = 2.0,
        max_rest_seconds: float = 5.0,
        min_price_increment: float = 0.01,
        obi_threshold: float = 0.7,
    ) -> None:
        super().__init__(
            latest_market=latest_market if latest_market is not None else {},
            submit_limit_order=submit_limit_order if submit_limit_order is not None else _noop_submit,
            replace_limit_order=replace_limit_order if replace_limit_order is not None else _noop_replace,
            pricing_mode=pricing_mode,
            chase_interval_seconds=chase_interval_seconds,
            max_rest_seconds=max_rest_seconds,
            min_price_increment=min_price_increment,
        )
        self._bus = event_bus
        self._obi_threshold = obi_threshold
        self._vpin_calculators: dict[str, VPINCalculator] = {}
        self._latest_obi: dict[str, float] = {}
        self.known_icebergs: dict[str, dict[str, float]] = {}
        self.last_latency_micros: float = 0.0
        if event_bus is not None:
            self._subscription = event_bus.subscribe("market", self._on_market_wrapper)
            self._iceberg_sub = event_bus.subscribe("hidden_liquidity", self._on_hidden_liquidity)
            logger.info("OBISniper Subscribed to market and hidden_liquidity channels.")

    async def _on_hidden_liquidity(self, event: dict) -> None:
        """Listener for IcebergDetector payloads."""
        instrument = event.get("instrument")
        side = event.get("side")
        price = event.get("price")
        if instrument and side and price:
            if instrument not in self.known_icebergs:
                self.known_icebergs[instrument] = {}
            self.known_icebergs[instrument][side] = float(price)

    async def _on_market_wrapper(self, event: Any) -> None:
        if isinstance(event, OrderBookTick):
            await self.update_book(event)

    async def update_book(self, event: OrderBookTick) -> None:
        """Calculate OBI using Cython for microsecond speed (<50us requirement)."""
        import time
        t0 = time.perf_counter_ns()
        
        obi = calculate_obi_cython(event.best_bid_size, event.best_ask_size)
        
        t1 = time.perf_counter_ns()
        self.last_latency_micros = (t1 - t0) / 1000.0
        
        self._latest_obi[event.instrument_id] = obi

    async def update_trades(self, event: TradeTick) -> None:
        if event.instrument_id not in self._vpin_calculators:
            self._vpin_calculators[event.instrument_id] = VPINCalculator(bucket_volume=50000.0)
            
        self._vpin_calculators[event.instrument_id].update(event)

    def _target_limit_price(self, order_event: OrderEvent) -> float | None:
        market_event = self._latest_market.get(order_event.instrument_id)
        if market_event is None:
            return None
            
        # 1. vPIN Deep Shadow Veto Check
        vpin_calc = self._vpin_calculators.get(order_event.instrument_id)
        if vpin_calc and vpin_calc.is_toxic:
            logger.warning(f"DEEP SHADOW VETO: vPIN for {order_event.instrument_id} is extremely toxic ({vpin_calc.vpin:.2f} > 0.8). Canceling limit orders in Sniper.")
            return None # A returning None causes LimitOrderChaser to cancel the resting order

        obi = self._latest_obi.get(order_event.instrument_id, 0.0)
        
        # Determine base prices
        if isinstance(market_event, OrderBookTick):
            bid = market_event.best_bid
            ask = market_event.best_ask
        elif isinstance(market_event, QuoteTick):
            bid = market_event.bid
            ask = market_event.ask
        else:
            # Fallback to TradeTick last price
            price = getattr(market_event, "last_price", None)
            if price is None: return None
            return max(self._min_price_increment, round(price / self._min_price_increment) * self._min_price_increment)

        # OBI Logic
        side = order_event.side
        mode = "normal"
        
        if obi > self._obi_threshold:
            mode = "aggressive"
        elif obi < -self._obi_threshold:
            mode = "patient"
            
        if side == "buy":
            if self._pricing_mode == "passive_rebate":
                mid = (bid + ask) / 2.0
                spread = ask - bid
                vpin_val = vpin_calc.vpin if vpin_calc else 0.0
                offset = spread * 0.5 * (1 + vpin_val) * (1 - obi)
                raw_price = max(bid, mid - offset)
            elif mode == "aggressive":
                raw_price = ask
            elif mode == "patient":
                raw_price = bid if self._pricing_mode == "join" else (bid + ask) / 2.0
            else:
                raw_price = bid if self._pricing_mode == "join" else (bid + ask) / 2.0
                
            # PENNY JUMP LOGIC (Buy)
            iceberg_buy_price = self.known_icebergs.get(order_event.instrument_id, {}).get("buy")
            if iceberg_buy_price is not None and abs(raw_price - iceberg_buy_price) < 1e-4:
                raw_price += 0.01
                
        else: # sell
            if self._pricing_mode == "passive_rebate":
                mid = (bid + ask) / 2.0
                spread = ask - bid
                vpin_val = vpin_calc.vpin if vpin_calc else 0.0
                offset = spread * 0.5 * (1 + vpin_val) * (1 + obi)
                raw_price = min(ask, mid + offset)
            elif mode == "aggressive":
                raw_price = bid
            elif mode == "patient":
                raw_price = ask
            else:
                raw_price = ask if self._pricing_mode == "join" else (bid + ask) / 2.0
                
            # PENNY JUMP LOGIC (Sell)
            iceberg_sell_price = self.known_icebergs.get(order_event.instrument_id, {}).get("sell")
            if iceberg_sell_price is not None and abs(raw_price - iceberg_sell_price) < 1e-4:
                raw_price -= 0.01

        return max(self._min_price_increment, round(raw_price / self._min_price_increment) * self._min_price_increment)
