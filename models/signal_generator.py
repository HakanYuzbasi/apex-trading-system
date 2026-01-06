"""
models/signal_generator.py - ML Signal Generation
"""

import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals using multiple strategies."""
    
    def __init__(self):
        self.lookback = 20
        
    def generate_momentum_signal(self, prices: pd.Series) -> float:
        """Momentum strategy signal."""
        if len(prices) < self.lookback:
            return 0.0
        returns = prices.pct_change(self.lookback).iloc[-1]
        return np.tanh(returns * 10)
    
    def generate_mean_reversion_signal(self, prices: pd.Series) -> float:
        """Mean reversion signal."""
        if len(prices) < self.lookback:
            return 0.0
        mean = prices.rolling(self.lookback).mean().iloc[-1]
        std = prices.rolling(self.lookback).std().iloc[-1]
        if std == 0:
            return 0.0
        z_score = (prices.iloc[-1] - mean) / std
        return -np.tanh(z_score)
    
    def generate_ml_signal(self, symbol: str) -> Dict:
        """Generate combined ML signal."""
        # Mock price data - replace with real data fetch
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        
        momentum = self.generate_momentum_signal(prices)
        mean_rev = self.generate_mean_reversion_signal(prices)
        
        # Ensemble: 70% momentum, 30% mean reversion
        signal = 0.7 * momentum + 0.3 * mean_rev
        confidence = abs(signal)
        
        return {
            'signal': float(np.clip(signal, -1, 1)),
            'confidence': float(min(confidence, 1.0)),
            'momentum': float(momentum),
            'mean_reversion': float(mean_rev),
            'timestamp': datetime.now()
        }
