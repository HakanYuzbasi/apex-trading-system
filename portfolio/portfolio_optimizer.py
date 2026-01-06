"""
portfolio/portfolio_optimizer.py - Portfolio Optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Optimize portfolio allocation."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - self.risk_free_rate / 252
        return float(excess_returns / returns.std() * np.sqrt(252))
    
    def optimize_weights(self, symbols: List[str], returns_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Optimize portfolio weights."""
        n = len(symbols)
        if n == 0:
            return {}
        
        # Equal weight for simplicity - can be enhanced with mean-variance optimization
        weights = {symbol: 1.0 / n for symbol in symbols}
        
        logger.info(f"📊 Optimized weights: {weights}")
        return weights
    
    def rebalance_portfolio(self, current_positions: Dict[str, int], 
                          target_weights: Dict[str, float],
                          total_value: float) -> Dict[str, int]:
        """Calculate rebalancing trades."""
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            target_shares = int(target_value / 100)  # Assume $100/share
            current_shares = current_positions.get(symbol, 0)
            
            trade = target_shares - current_shares
            if abs(trade) > 0:
                trades[symbol] = trade
        
        return trades
