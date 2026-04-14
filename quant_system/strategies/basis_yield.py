import logging
import asyncio
from typing import Dict, Any

from quant_system.core.bus import InMemoryEventBus
from quant_system.strategies.base import StrategyBase
from quant_system.events.order import OrderEvent

logger = logging.getLogger("basis_yield")

class BasisYieldStrategy(StrategyBase):
    """
    Delta-Neutral Cash-and-Carry Strategy.
    Monitors Perpetual Funding Rates (mocked). If APY > 15% and idle USD > 30%, 
    synthesizes a Long Spot + Short Perpetual position.
    """
    def __init__(self, event_bus: InMemoryEventBus):
        super().__init__(strategy_id="BasisYield_01")
        self._bus = event_bus
        
        # Idle capital tracker (would normally be bound closely to PortfolioLedger)
        self.idle_usd = 0.0
        self.total_usd = 0.0
        
        self.active_positions: Dict[str, float] = {} # instrument -> size
        self.mock_apy_streams: Dict[str, float] = {
            "BTC/USD": 0.08,
            "ETH/USD": 0.12
        }

    def start(self):
        # We assume a periodic polling mechanism is managed here
        asyncio.create_task(self._poll_funding_rates())
        logger.info(f"[{self.strategy_id}] Cash and Carry Strategy Online. Polling Funding Rates.")

    async def _poll_funding_rates(self):
        """Simulates fetching from Coinbase/Kraken perpetual swap APIs."""
        while True:
            await asyncio.sleep(10) # 10 second polling
            
            # Simulate shifting yields
            # Frequently crossing the 15% (0.15) threshold for demonstration
            import random
            for inst in self.mock_apy_streams:
                current_apy = self.mock_apy_streams[inst]
                shift = (random.random() - 0.4) * 0.05 # slight upward bias to reach 15% quickly
                new_apy = max(0.01, min(0.40, current_apy + shift))
                self.mock_apy_streams[inst] = new_apy
                
                logger.debug(f"[{self.strategy_id}] Current APY for {inst}: {new_apy*100:.2f}%")
                
                await self._evaluate_yields(inst, new_apy)

    async def update_capital_state(self, total_usd: float, idle_usd: float):
        """Hook called by the Portfolio Manager or Rebalancer."""
        self.total_usd = total_usd
        self.idle_usd = idle_usd

    async def _evaluate_yields(self, instrument: str, apy: float):
        # Trigger Condition: APY > 15% and Idle USD > 30%
        if apy > 0.15:
            idle_ratio = self.idle_usd / self.total_usd if self.total_usd > 0 else 0.0
            
            # Use 30% ratio as a proxy. For mock, if total=0 (not initialized), assume safe.
            if self.total_usd > 0 and idle_ratio < 0.30:
                return # Not enough idle capital to engage basis trade
                
            if self.active_positions.get(instrument, 0.0) == 0.0:
                logger.info(f"[{self.strategy_id}] 💰 HIGH YIELD DETECTED ({apy*100:.2f}%). Synthesizing Cash-and-Carry on {instrument}")
                await self._execute_delta_neutral(instrument)

    async def _execute_delta_neutral(self, instrument: str):
        """
        Emits pairs of orders to capture the yield.
        Buy Spot, Sell Short the Perpetual contract of exact same size.
        """
        # A real system calculates the exact lot size based on available USD
        target_lots = 1.0 
        
        # Emit Long Spot Leg
        long_spot = OrderEvent(
            order_id=f"BY_SPOT_buy_{instrument}",
            instrument_id=instrument,
            quantity=target_lots,
            side="buy",
            order_type="market"
        )
        await self._bus.publish("order", long_spot)
        
        # Emit Short Perp Leg
        short_perp = OrderEvent(
            order_id=f"BY_PERP_sell_{instrument}",
            instrument_id=f"{instrument}-PERP",
            quantity=target_lots,
            side="sell",
            order_type="market"
        )
        await self._bus.publish("order", short_perp)
        
        self.active_positions[instrument] = target_lots
        logger.info(f"[{self.strategy_id}] Delta-Neutral legs dispatched for {instrument}")
