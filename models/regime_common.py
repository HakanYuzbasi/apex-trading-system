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

# Global singleton to prevent re-initialization overhead (logs, CPU)
_GLOBAL_DETECTOR = None

def get_regime(prices: pd.Series, lookback: int = 60) -> str:
    """
    Detect market regime using AdaptiveRegimeDetector.
    
    Args:
        prices: Historical price series
        lookback: Minimum lookback period (handled by detector, but kept for API compatibility)
        
    Returns:
        Regime string: 'bull', 'bear', 'neutral', 'volatile'
    """
    global _GLOBAL_DETECTOR
    
    # Initialize singleton if needed
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = AdaptiveRegimeDetector()
    
    # Ensure we have enough data
    if len(prices) < 60:
        return "neutral"
        
    assessment = _GLOBAL_DETECTOR.assess_regime(prices)
    return assessment.primary_regime
