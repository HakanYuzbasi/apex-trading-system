"""
models/portfolio_optimizer.py

Calculates Half-Kelly Allocation multipliers for execution blocks based on:
1. Real-time GodLevelSignalGenerator confidence scores
2. Rolling empirical win-rate from the OutcomeFeedbackLoop

This shifts execution from static dollar blocks into dynamic compounding risk units.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class KellyCriterionOptimizer:
    def __init__(self, target_rr: float = 1.5, half_kelly_modifier: float = 0.5):
        """
        target_rr (b): The average profit/loss ratio. If the setup aims for 1.5% profit
                       with a 1.0% stop loss, RR is 1.5.
        half_kelly_modifier: To prevent catastrophic drawdowns due to variance,
                             we employ Fractional Kelly (usually 0.5x).
        """
        self.b = target_rr
        self.fractional_modifier = half_kelly_modifier
        self.MAX_LEVERAGE = 3.0
        self.MIN_LEVERAGE = 0.1

    def calculate_sizing_multiplier(self, ml_confidence: float, historical_win_rate: float, is_high_vix: bool = False) -> float:
        """
        Dynamically scales the order block quantity. 
        Returns a scalar e.g., 0.5 (half size) up to 2.5 (back up the truck).
        """
        
        # We blend the instantaneous ML setup confidence with the system's actual
        # reality-tested 30-day win rate to formulate true probability 'p'.
        # If the historical loop isn't mature yet, default it safely towards 0.5.
        if historical_win_rate <= 0.0 or historical_win_rate >= 1.0:
            historical_win_rate = 0.5
            
        p = (ml_confidence + historical_win_rate) / 2.0
        q = 1.0 - p

        # Kelly Formula: f* = p - (q / b)
        full_kelly = p - (q / self.b)

        if full_kelly <= 0:
            # The odds are mathematically against us, don't take the trade
            return 0.0

        fractional_kelly = full_kelly * self.fractional_modifier

        # Normalize the raw Kelly percentage back into a block-scaling multiplier for the execution loop.
        # Generally: Kelly 10% -> 1.0x standard block. Kelly 25% -> 2.5x block.
        # Assume our standard portfolio sizing is 10% of equity per trade.
        standard_block = 0.10
        scalar = fractional_kelly / standard_block

        # Aggressive risk mitigation during highly unstable VIX environments
        if is_high_vix:
            scalar *= 0.6  # Chop position size by 40%

        # Clip bounds to prevent overflow or micro-dust
        return float(np.clip(scalar, self.MIN_LEVERAGE, self.MAX_LEVERAGE))

# Global Optimizer
_optimizer = KellyCriterionOptimizer()

def get_kelly_multiplier(ml_confidence: float, historical_win_rate: float, is_high_vix: bool = False) -> float:
    return _optimizer.calculate_sizing_multiplier(ml_confidence, historical_win_rate, is_high_vix)
