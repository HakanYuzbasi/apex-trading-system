"""
models/regime_common.py
================================================================================
SHARED REGIME DETECTION UTILITIES
================================================================================

Provides a unified interface for regime detection using the sophisticated
AdaptiveRegimeDetector, replacing ad-hoc implementations across the codebase.
"""

import pandas as pd
from typing import Optional
from .adaptive_regime_detector import AdaptiveRegimeDetector, RegimeAssessment

def get_regime(prices: pd.Series, lookback: int = 60) -> str:
    """
    Detect market regime using AdaptiveRegimeDetector.
    
    Args:
        prices: Historical price series
        lookback: Minimum lookback period (handled by detector, but kept for API compatibility)
        
    Returns:
        Regime string: 'bull', 'bear', 'neutral', 'volatile'
    """
    # Instantiate a fresh detector for stateless assessment
    # Note: In a live running system, maintaining state is better, 
    # but this replaces static ad-hoc methods.
    detector = AdaptiveRegimeDetector()
    
    # Ensure we have enough data
    if len(prices) < 60:
        return "neutral"
        
    assessment = detector.assess_regime(prices)
    return assessment.primary_regime
