import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger("canary_allocator")

class CanaryAllocator:
    """
    Volume-synchronized risk gating for NeuralSniper.
    Tracks live vs shadow slippage to dynamically cap execution size.
    """
    def __init__(self, window_size: int = 50, low_threshold: float = 0.005, high_threshold: float = 0.015):
        self.window_size = window_size
        self.low_threshold = low_threshold    # 0.5%
        self.high_threshold = high_threshold  # 1.5%
        
        # ring buffers for trailing slippage delta: {instrument: deque([delta_0, delta_1, ...])}
        self._history: Dict[str, deque] = {}
        
    def record_execution(self, instrument: str, intended_slip: float, actual_slip: float):
        """
        Records a completed execution's slippage metrics.
        intended_slip: theoretical slippage from shadow model.
        actual_slip: true realized slippage from live fill.
        """
        if instrument not in self._history:
            self._history[instrument] = deque(maxlen=self.window_size)
            
        # delta between shadow prediction and live reality
        delta = abs(actual_slip - intended_slip)
        self._history[instrument].append(delta)
        
        logger.info(f"[Canary] Recorded {instrument} | Delta: {delta*10000:.2f} bps | Total Window: {len(self._history[instrument])}")

    def get_allocation_multiplier(self, instrument: str) -> float:
        """
        Returns the risk-adjusted multiplier for position sizing.
        1.0: Full confidence.
        0.0: Kill live execution, revert to shadow/manual.
        """
        if instrument not in self._history or len(self._history[instrument]) < 10:
            # Seed period: Start small (canary phase)
            return 0.25 
            
        avg_delta = sum(self._history[instrument]) / len(self._history[instrument])
        
        if avg_delta > self.high_threshold:
            logger.warning(f"[Canary] CRITICAL DIVERGENCE on {instrument}: {avg_delta*10000:.1f} bps > {self.high_threshold*10000:.1f} bps. Kiling allocation.")
            return 0.0
            
        if avg_delta < self.low_threshold:
            # High performance: Full allocation
            return 1.0
            
        # Intermediate: Linear scaling between low and high threshold
        # multiplier = 1 - (avg_delta - low) / (high - low)
        multiplier = 1.0 - (avg_delta - self.low_threshold) / (self.high_threshold - self.low_threshold)
        
        return max(0.0, min(1.0, multiplier))

    def get_stats(self, instrument: str) -> Dict[str, float]:
        if instrument not in self._history:
            return {"avg_delta_bps": 0.0, "count": 0}
            
        avg_delta = sum(self._history[instrument]) / len(self._history[instrument])
        return {
            "avg_delta_bps": avg_delta * 10000,
            "count": len(self._history[instrument]),
            "multiplier": self.get_allocation_multiplier(instrument)
        }
