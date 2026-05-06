import logging

logger = logging.getLogger(__name__)

class KellySizer:
    """
    Implements fractional Kelly Criterion for position sizing.
    Kelly % = (Win_Rate * Payout_Ratio - Loss_Rate) / Payout_Ratio
    """
    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction
        # Defaults based on typical ORB/Pairs performance
        self.win_rate = 0.55
        self.payout_ratio = 2.0 # 2:1 Reward:Risk

    def calculate_multiplier(self, confidence: float) -> float:
        """
        Adjusts the Kelly multiplier based on Meta-Labeler confidence.
        If confidence is high, we lean more into the Kelly ideal.
        """
        # Adjust win rate by confidence (confidence of 0.5 means a coin flip)
        adj_win_rate = confidence
        loss_rate = 1.0 - adj_win_rate
        
        if self.payout_ratio <= 0:
            return 1.0
            
        kelly_pct = (adj_win_rate * self.payout_ratio - loss_rate) / self.payout_ratio
        
        # Clip Kelly % between 0 and 1
        kelly_pct = max(0.0, min(1.0, kelly_pct))
        
        # Apply fractional Kelly (e.g., quarter-Kelly) for safety
        multiplier = kelly_pct * self.fraction
        
        # Normalization: We want a multiplier that scales the 'base' notional.
        # If confidence is 0.72 (base), multiplier should be ~1.0.
        # Let's use a simple normalization relative to a "standard" 0.65 confidence.
        base_kelly = (0.65 * 2.0 - 0.35) / 2.0 # 0.475
        normalized_multiplier = kelly_pct / base_kelly if base_kelly > 0 else 1.0
        
        return max(0.2, min(2.0, normalized_multiplier * self.fraction * 4)) # Scale up to 2x for high confidence
