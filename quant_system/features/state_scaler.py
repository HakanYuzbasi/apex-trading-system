"""
quant_system/features/state_scaler.py
================================================================================
Production Online State Scaler (STRICT PIPELINE)
================================================================================
"""

import numpy as np

class OnlineStateScaler:
    def __init__(self, state_dim: int, exclude_dims: list = None):
        self.state_dim = state_dim
        self.count = 0
        self.mean = np.zeros(state_dim)
        self.M2 = np.zeros(state_dim)
        self.exclude_dims = exclude_dims if exclude_dims else []
        
        # UPGRADE 5: Feature Weighting (State Importance Scaling)
        self.feature_weights = np.ones(state_dim)
        self.feature_weights[4] = 1.2 # meta_prob
        self.feature_weights[5] = 1.3 # confidence
        self.feature_weights[6] = 1.1 # disagreement
        self.feature_weights[7] = 0.8 # expected impact
        self.feature_weights[8] = 0.8 # recent cost spike
        self.feature_weights[14]= 1.0 # previous action

    def _update_stats(self, x: np.ndarray):
        """Welford's online variance algorithm. Strictly no look-ahead."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def scale(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        # UPGRADE 1: Strictly lagged processing
        if update:
            self._update_stats(x)
            
        if self.count < 2:
            return x * self.feature_weights
            
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance)
        std = np.maximum(std, 1e-6) # Fallback stability
        
        scaled_state = np.copy(x)
        for i in range(self.state_dim):
            if i not in self.exclude_dims:
                scaled_state[i] = (x[i] - self.mean[i]) / std[i]
                
        # UPGRADE 2: State Value Clipping
        scaled_state = np.clip(scaled_state, -5.0, 5.0)
        
        # UPGRADE 5: Feature Weighting
        weighted_state = scaled_state * self.feature_weights
        
        return weighted_state
