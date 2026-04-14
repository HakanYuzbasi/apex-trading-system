import logging
from typing import Dict, Any, Optional

from quant_system.core.bus import InMemoryEventBus
from quant_system.events.order import OrderEvent
from quant_system.events.execution import ExecutionEvent
from quant_system.events.market import QuoteTick, OrderBookTick

logger = logging.getLogger("execution_alpha")

class ExecutionAlphaTracker:
    """
    Tracks the execution alpha.
    Execution Alpha = Profit generated strictly from the Sniper's
    ability to capture maker rebates and minimize slippage against
    the 'Intended Mid-Price' at signal generation.
    """
    def __init__(self, event_bus: InMemoryEventBus):
        self._bus = event_bus
        
        # order_id -> Intended Mid Price
        self._intended_mids: Dict[str, float] = {}
        
        # order_id -> Initial order quantity
        self._order_qty: Dict[str, float] = {}
        
        # Latest market data to grab mid prices
        self._latest_mids: Dict[str, float] = {}
        
        self.total_rebates_collected = 0.0
        self.total_fees_paid = 0.0
        self.cumulative_execution_alpha = 0.0

    def start(self):
        self._bus.subscribe("order", self._on_order)
        self._bus.subscribe("execution", self._on_execution)
        self._bus.subscribe("market", self._on_market)
        logger.info("Alpha Tracker listening for Order/Execution events.")

    async def _on_market(self, event: Any):
        # Extract mid price from incoming market ticks
        if isinstance(event, OrderBookTick):
            self._latest_mids[event.instrument_id] = (event.best_bid + event.best_ask) / 2.0
        elif isinstance(event, QuoteTick):
            self._latest_mids[event.instrument_id] = (event.bid + event.ask) / 2.0

    async def _on_order(self, event: OrderEvent):
        """When an order is requested, record the strict mid price intention."""
        instrument = event.instrument_id
        current_mid = self._latest_mids.get(instrument)
        if current_mid is None:
            return  # Can't properly assess alpha without mid-price context
            
        self._intended_mids[event.order_id] = current_mid
        self._order_qty[event.order_id] = event.quantity
        
    async def _on_execution(self, event: ExecutionEvent):
        """When a fill occurs, compare actual price (incl fees) vs intended mid."""
        # Only process actual fills
        if event.execution_status not in ["partial_fill", "filled"]:
            return
            
        order_id = event.order_id
        if order_id not in self._intended_mids:
            return
            
        intended_mid = self._intended_mids[order_id]
        
        # Extract actual fill components
        fill_price = event.fill_price
        fill_qty = event.fill_qty
        
        # Often fees/rebates are embedded in standard OMS. Let's mock extraction.
        # Negative fee = strict rebate. Positive = fee paid.
        fee = event.fee if hasattr(event, "fee") and event.fee is not None else 0.0
        if fee < 0:
            self.total_rebates_collected += abs(fee)
        else:
            self.total_fees_paid += fee

        # Calculate True Net Fill Price
        # Net fill = fill_price + (fee / fill_qty) for Buy
        # Net fill = fill_price - (fee / fill_qty) for Sell
        if event.side == "buy":
            net_fill_price = fill_price + (fee / fill_qty) if fill_qty > 0 else fill_price
            slippage_impact = intended_mid - net_fill_price # Positive is good (bought lower)
        else:
            net_fill_price = fill_price - (fee / fill_qty) if fill_qty > 0 else fill_price
            slippage_impact = net_fill_price - intended_mid # Positive is good (sold higher)
            
        alpha_dollars = slippage_impact * fill_qty
        self.cumulative_execution_alpha += alpha_dollars
        
        logger.debug(f"Execution Alpha Update [{event.instrument_id}]: Trade captured {alpha_dollars:.4f} USD edge. Cumulative: {self.cumulative_execution_alpha:.2f}")
        
        # Broadcast the metrics update
        metrics_payload = {
            "type": "execution_alpha_update",
            "instrument": event.instrument_id,
            "trade_alpha": alpha_dollars,
            "cumulative_alpha": self.cumulative_execution_alpha,
            "total_rebates": self.total_rebates_collected,
            "total_fees": self.total_fees_paid
        }
        await self._bus.publish("metrics", metrics_payload)
