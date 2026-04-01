"""
quant_system/data/mock_websocket.py
================================================================================
ASYNC TICK GENERATOR (ALPACA/IBKR SIMULATION)
================================================================================
"""

import asyncio
import numpy as np
from datetime import datetime

class TickReplayer:
    def __init__(self, num_ticks: int = 3000):
        self.num_ticks = num_ticks
        
    async def stream_ticks(self):
        """
        Asynchronous Generator strictly mirroring live JSON dictionaries from Alpaca / IBKR structures.
        """
        # Mock geometric Brownian motion for simulation
        prices = np.exp(np.cumsum(np.random.normal(0, 0.001, self.num_ticks))) * 100
        volumes = np.random.randint(50, 500, self.num_ticks)
        
        for i in range(self.num_ticks):
            tick = {
                "stream": "trade",
                "symbol": "BTCUSD", # Target symbol payload
                "price": float(prices[i]),
                "size": int(volumes[i]),
                "timestamp": datetime.now().isoformat()
            }
            yield tick
            
            # Non-blocking micro-latency isolating synchronous deadlocks from network
            await asyncio.sleep(0.001)
