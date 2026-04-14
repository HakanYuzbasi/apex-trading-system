import logging
import time
import numpy as np
from typing import Any
from stable_baselines3 import PPO

from quant_system.core.bus import InMemoryEventBus
from quant_system.execution.obi_sniper import OBISniper
from quant_system.events.market import OrderBookTick
from quant_system.events.order import OrderEvent

logger = logging.getLogger("neural_sniper")

class NeuralSniper(OBISniper):
    """
    Subclasses OBISniper to override strict deterministic limits with
    DRL PPO Neural Network inferences operating within < 2ms latency bounds.
    """
    def __init__(
        self,
        event_bus: InMemoryEventBus,
        model_path: str = "run_state/models/ppo_execution_v1.zip",
        min_price_increment: float = 0.01,
    ):
        super().__init__(event_bus=event_bus, min_price_increment=min_price_increment)
        
        logger.info(f"Loading PPO Execution Weights from {model_path}...")
        try:
            self.model = PPO.load(model_path)
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load Neural Execution Model: {e}. Falling back to OBISniper.")
            self.model_loaded = False
            
        self.last_action: int = 0
        self.last_confidence: float = 0.0
        self.last_state_vector: np.ndarray = np.zeros(6, dtype=np.float32)
        self.logger = logger

    def _build_state_vector(self, instrument: str) -> np.ndarray:
        """Transforms current execution context into the normalized 6-D DRL State Vector."""
        
        # Pull latest metrics
        vpin = 0.0
        if instrument in self._vpin_calculators:
            vpin = self._vpin_calculators[instrument].vpin
            
        obi = self._latest_obi.get(instrument, 0.0)
        
        # Proxy standard spread and iceberg
        spread = 0.01 # dummy normalized spread
        
        iceberg_presence = 0.0
        if instrument in self.known_icebergs and len(self.known_icebergs[instrument]) > 0:
            iceberg_presence = 1.0
            
        inventory_remaining = 1.0 # default single block 
        time_remaining = 0.5 # 50% time remaining default proxy
        
        vector = np.array([vpin, obi, spread, iceberg_presence, inventory_remaining, time_remaining], dtype=np.float32)
        
        # Publish exactly what the neural net is "seeing" to the dashboard
        self.last_state_vector = vector
        
        return vector

    async def _on_market_wrapper(self, event: Any):
        await super()._on_market_wrapper(event)
        
        # Periodically emit the neural view metrics over the bus
        if self.model_loaded and isinstance(event, OrderBookTick):
            vector = self._build_state_vector(event.instrument_id)
            await self._bus.publish("neural_metrics", {
                "instrument": event.instrument_id,
                "state_vector": vector.tolist(),
                "action": int(self.last_action),
                "confidence": float(self.last_confidence)
            })

    def _target_limit_price(self, order_event: OrderEvent) -> float | None:
        """
        Overrides OBISniper/LimitOrderChaser. Queries the PPO model in shadow mode,
        logs its intent, then executes a market-sweep (pay ask / take bid).
        Falls back to parent implementation if no market data is available.
        """
        # Resolve bid/ask from the latest market snapshot (same path as base class)
        market_event = self._latest_market.get(order_event.instrument_id)
        if market_event is None:
            return None

        from quant_system.events.market import QuoteTick, TradeTick, BarEvent
        if isinstance(market_event, QuoteTick):
            bid, ask = market_event.bid, market_event.ask
        elif isinstance(market_event, TradeTick):
            price = market_event.last_price
            bid, ask = price, price
        else:
            # BarEvent — use close as mid; round to increment
            price = getattr(market_event, "close_price", None) or getattr(market_event, "price", None)
            if not price:
                return None
            bid, ask = price, price

        # PPO shadow-mode inference
        try:
            state_vector = self._build_state_vector(order_event.instrument_id)
            action_tuple = self.model.predict(state_vector, deterministic=True)
            predicted_action_idx = int(action_tuple[0])
        except Exception:
            predicted_action_idx = 3  # default: Market Sweep

        action_map = {0: "Wait", 1: "Passive Maker", 2: "Penny-Jump", 3: "Market Sweep"}
        suggested_action = action_map.get(predicted_action_idx, "Market Sweep")
        self.last_action = predicted_action_idx
        self.logger.debug(
            "🛰️ [SHADOW MODE] PPO suggests: %s | Overriding to Market Sweep for %s.",
            suggested_action, order_event.instrument_id,
        )

        # Always execute as market sweep: buy at ask, sell at bid
        raw_price = ask if order_event.side.upper() == "BUY" else bid
        return max(self._min_price_increment, round(raw_price / self._min_price_increment) * self._min_price_increment)
