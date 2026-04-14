import logging
import time
from typing import Dict, Any

from quant_system.core.bus import InMemoryEventBus
from quant_system.events.market import OrderBookTick, TradeTick

logger = logging.getLogger("iceberg_detector")

class IcebergState:
    def __init__(self):
        self.last_trade_time = 0.0
        self.last_trade_qty = 0.0
        self.last_trade_side = ""
        self.last_trade_price = 0.0
        
        self.consecutive_reloads = 0
        self.last_known_depth = 0.0

class IcebergDetector:
    """
    Subscribes to Market ticks. Maps taker sweeps against immediate book replenishment.
    If a level regenerates instantly after being swept 3 times, emits a HIDDEN_LIQUIDITY_EVENT.
    """
    def __init__(self, event_bus: InMemoryEventBus):
        self._bus = event_bus
        # map: instrument_id -> side -> IcebergState
        self._states: Dict[str, Dict[str, IcebergState]] = {}
        
    def start(self):
        self._bus.subscribe("market", self._on_market_event)
        logger.info("IcebergDetector online. Monitoring for limit replenishments.")
        
    async def _on_market_event(self, event: Any):
        if isinstance(event, TradeTick):
            await self._handle_trade(event)
        elif isinstance(event, OrderBookTick):
            await self._handle_book(event)

    async def _handle_trade(self, event: TradeTick):
        # A taker trade occurred
        instrument = event.instrument_id
        if instrument not in self._states:
            self._states[instrument] = {"buy": IcebergState(), "sell": IcebergState()}
            
        side = event.aggressor_side # "buy" means taker bought (hit the ask)
        # We track the maker side that got hit. If aggressor is "buy", the maker is "sell" (ask).
        maker_side = "sell" if side == "buy" else "buy"
        
        state = self._states[instrument][maker_side]
        state.last_trade_time = time.time()
        state.last_trade_qty = event.quantity
        state.last_trade_price = event.price
        
    async def _handle_book(self, event: OrderBookTick):
        instrument = event.instrument_id
        if instrument not in self._states:
            return
            
        now = time.time()
        
        # Check Ask Icebergs (where takers previously bought)
        ask_state = self._states[instrument]["sell"]
        if ask_state.last_trade_time > 0 and (now - ask_state.last_trade_time) < 0.050: # within 50ms
            if event.best_ask == ask_state.last_trade_price:
                # If size hasn't decreased significantly despite the trade, it regenerated
                if event.best_ask_size >= (ask_state.last_known_depth * 0.9): 
                    ask_state.consecutive_reloads += 1
                    
                    if ask_state.consecutive_reloads >= 3:
                        logger.warning(f"🧊 ICEBERG DETECTED [ASK]: {instrument} @ {event.best_ask}. Emitting HIDDEN_LIQUIDITY_EVENT.")
                        await self._bus.publish("hidden_liquidity", {
                            "instrument": instrument,
                            "side": "sell",
                            "price": event.best_ask,
                            "estimated_hidden_qty": ask_state.last_trade_qty * 3 # rough heuristic
                        })
                        ask_state.consecutive_reloads = 0 # reset after emit
                else:
                    ask_state.consecutive_reloads = 0
                    
        ask_state.last_known_depth = event.best_ask_size
        
        # Check Bid Icebergs (where takers previously sold)
        bid_state = self._states[instrument]["buy"]
        if bid_state.last_trade_time > 0 and (now - bid_state.last_trade_time) < 0.050: # within 50ms
            if event.best_bid == bid_state.last_trade_price:
                if event.best_bid_size >= (bid_state.last_known_depth * 0.9):
                    bid_state.consecutive_reloads += 1
                    
                    if bid_state.consecutive_reloads >= 3:
                        logger.warning(f"🧊 ICEBERG DETECTED [BID]: {instrument} @ {event.best_bid}. Emitting HIDDEN_LIQUIDITY_EVENT.")
                        await self._bus.publish("hidden_liquidity", {
                            "instrument": instrument,
                            "side": "buy",
                            "price": event.best_bid,
                            "estimated_hidden_qty": bid_state.last_trade_qty * 3
                        })
                        bid_state.consecutive_reloads = 0
                else:
                    bid_state.consecutive_reloads = 0
                    
        bid_state.last_known_depth = event.best_bid_size
