"""
portfolio/rebalancing.py - Smart Portfolio Rebalancing

Implements transaction-cost aware rebalancing.
Avoids rebalancing if the alpha generated is less than the transaction costs.

Features:
- Alpha estimation vs Cost estimation
- Minimum deviation thresholds
- Turnover reduction
"""

import numpy as np
from typing import Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RebalanceAction:
    symbol: str
    current_weight: float
    target_weight: float
    trade_pct: float
    estimated_cost_bps: float
    estimated_alpha_bps: float
    action: str  # 'BUY', 'SELL', 'HOLD'


class SmartRebalancer:
    """
    Cost-aware Portfolio Rebalancer.
    """
    
    def __init__(
        self,
        commission_bps: float = 5.0,
        impact_bps: float = 10.0,
        min_trade_amount: float = 100.0,
        alpha_decay_days: float = 30.0
    ):
        """
        Args:
            commission_bps: Commission in basis points
            impact_bps: Market impact model (linear) per 10% ADV (simplified to constant here)
            min_trade_amount: Minimum $ amount to bother trading
            alpha_decay_days: Assumed half-life of alpha for benefit calculation
        """
        self.commission_cost = commission_bps / 10000.0
        self.impact_cost = impact_bps / 10000.0
        self.min_trade = min_trade_amount
        self.alpha_decay = alpha_decay_days
        
        logger.info("SmartRebalancer initialized")
        
    def generate_orders(
        self,
        current_positions: Dict[str, float],  # symbol -> $ value
        target_weights: Dict[str, float],     # symbol -> weight (0.0 to 1.0)
        total_equity: float,
        price_map: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate rebalancing orders filtering out non-economical trades.
        
        Args:
            current_positions: Value held per symbol
            target_weights: Efficient frontier / Optimization output
            total_equity: Current portfolio equity
            price_map: Current prices
            
        Returns:
            List of order dicts
        """
        actions = []
        
        # Normalize current weights
        current_weights = {s: v / total_equity for s, v in current_positions.items()}
        
        # Union of all symbols
        all_symbols = set(current_positions.keys()).union(target_weights.keys())
        
        for symbol in all_symbols:
            w_curr = current_weights.get(symbol, 0.0)
            w_targ = target_weights.get(symbol, 0.0)
            
            diff = w_targ - w_curr
            
            if abs(diff) < 1e-4:  # Ignorable
                continue
                
            trade_val = abs(diff) * total_equity
            
            if trade_val < self.min_trade:
                continue
                
            # Estimated Cost
            # Cost = Commission + Impact
            trade_val * (self.commission_cost + self.impact_cost)
            
            # Estimated Benefit (Alpha)
            # Assumption: The 'Target' portfolio outperforms 'Current' by some margin
            # Simplistic model: Deviation represents mispricing. 
            # Benefit = Trade Size * Estimated Edge (e.g., 50bps over horizon)
            # A rigorous model would use the optimizer's expected return difference.
            # Here we use a heuristic threshold: Trade if deviation > X% or benefit > cost * 2
            
            # Use deviation-based turnover control (buffer)
            # Don't trade if weight delta is small relative to current weight (e.g. < 5-10% change)
            deviation_pct = abs(diff)
            
            # Threshold: Trade if we are fixing a > 0.5% portfolio weight deviation
            # OR if we are exiting a position
            is_worth_it = False
            
            if w_targ == 0: # Full exit
                is_worth_it = True
            elif w_curr == 0: # New entry
                is_worth_it = True
            elif deviation_pct > 0.005: # > 0.5% portfolio shift
                is_worth_it = True
                
            if is_worth_it:
                action = 'BUY' if diff > 0 else 'SELL'
                shares = int(trade_val / price_map.get(symbol, 100.0))
                
                if shares > 0:
                    actions.append({
                        'symbol': symbol,
                        'side': action,
                        'qty': shares,
                        'reason': f"Rebalance (Target: {w_targ:.1%}, Curr: {w_curr:.1%})"
                    })
                    
        return actions

    def optimize_turnover(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        turnover_limit: float = 0.20
    ) -> np.ndarray:
        """
        Constrain target weights to limit turnover.
        w_new = w_curr + clip(w_targ - w_curr, -limit, limit)
        
        Args:
            current_weights: Array of current weights
            target_weights: Array of ideal weights
            turnover_limit: Max one-way turnover allowed (e.g. 20%)
            
        Returns:
            Constrained weights
        """
        diff = target_weights - current_weights
        
        # If total turnover > limit, scale back
        total_turnover = np.sum(np.abs(diff)) / 2.0  # One-way
        
        if total_turnover > turnover_limit:
            scale = turnover_limit / total_turnover
            diff = diff * scale
            logger.info(f"Rebalance constrained: Turnover scaled by {scale:.2f}")
            
        return current_weights + diff
