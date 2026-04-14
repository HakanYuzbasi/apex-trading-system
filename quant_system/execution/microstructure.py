import numpy as np
from collections import deque
from quant_system.events.market import TradeTick
from quant_system.execution.fast_math import calculate_vpin_cython

class VPINCalculator:
    """
    Computes real-time Volume-synchronized Probability of Informed Trading (vPIN).
    Normalizes time blocks into volume buckets (size V).
    A high vPIN (>0.8) indicates highly toxic unbalanced flow.
    """
    def __init__(self, bucket_volume: float = 50000.0, num_buckets: int = 50):
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets
        
        self.buy_buckets = deque(maxlen=num_buckets)
        self.sell_buckets = deque(maxlen=num_buckets)
        
        # Current active bucket
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._current_vol = 0.0
        
        self.vpin = 0.0

    def update(self, event: TradeTick) -> float:
        """
        Updates the bucket with new trade volume and recalculates vPIN if full.
        Returns the instantaneous vPIN value.
        """
        vol = event.last_size
        side = event.aggressor_side
        
        # If a massive trade comes in, it might spill over the bucket
        remaining_vol = vol
        while remaining_vol > 0:
            space_left = self.bucket_volume - self._current_vol
            fill = min(space_left, remaining_vol)
            
            if side == "buy":
                self._current_buy += fill
            elif side == "sell":
                self._current_sell += fill
            
            self._current_vol += fill
            remaining_vol -= fill
            
            # Bucket is full, roll it over
            if self._current_vol >= self.bucket_volume:
                self.buy_buckets.append(self._current_buy)
                self.sell_buckets.append(self._current_sell)
                
                self._current_buy = 0.0
                self._current_sell = 0.0
                self._current_vol = 0.0
                
                # Re-calculate vPIN using Fast Math if we have full history
                self._recalculate()
                
        return self.vpin

    def _recalculate(self) -> None:
        if len(self.buy_buckets) == 0:
            return
            
        buy_arr = np.array(self.buy_buckets, dtype=np.float64)
        sell_arr = np.array(self.sell_buckets, dtype=np.float64)
        
        self.vpin = calculate_vpin_cython(buy_arr, sell_arr, self.bucket_volume)

    @property
    def is_toxic(self) -> bool:
        return self.vpin > 0.8
