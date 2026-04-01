"""
quant_system/data/streaming_buffers.py
================================================================================
O(1) CIRCULAR TICK BUFFERS
================================================================================
"""

import collections
import statistics

class TickBuffer:
    def __init__(self, maxlen: int = 200):
        # Collections deque provides strictly mapped O(1) append arrays replacing pandas .rolling() lag
        self.maxlen = maxlen
        self.prices = collections.deque(maxlen=maxlen)
        self.volumes = collections.deque(maxlen=maxlen)
        self.returns = collections.deque(maxlen=maxlen)
        self._last_price = None

    def update(self, tick_data: dict):
        price = tick_data.get('price', 0.0)
        volume = tick_data.get('size', 0.0)
        
        self.prices.append(price)
        self.volumes.append(volume)
        
        if self._last_price is not None and self._last_price > 0:
            ret = (price / self._last_price) - 1.0
            self.returns.append(ret)
        else:
            self.returns.append(0.0)
            
        self._last_price = price

    @property
    def current_volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        return statistics.stdev(self.returns)
        
    @property
    def moving_average(self) -> float:
        if not self.prices:
            return 0.0
        return sum(self.prices) / len(self.prices)
        
    @property
    def momentum(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        # Price magnitude mapping vs T-N lag
        return self.prices[-1] / self.prices[0]
