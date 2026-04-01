"""
quant_system/execution/cost_model.py
================================================================================
Non-Linear Regime-Aware Transaction Cost Model (UPGRADED)
================================================================================
"""

import numpy as np

class TransactionCostModel:
    def __init__(self, base_spread_bps: float = 1.0, slippage_vol_factor: float = 0.5, impact_factor: float = 0.1):
        self.base_spread_bps = base_spread_bps
        self.slippage_vol_factor = slippage_vol_factor
        self.impact_factor = impact_factor

    def calculate_cost(self, price: float, order_size_usd: float, adv_usd: float, current_volatility: float, regime: int) -> dict:
        """Regimes: 1="normal", 2="high_vol", >=3="crash"."""
        if order_size_usd == 0 or adv_usd == 0:
            return {'total_fractional_cost': 0.0, 'spread': 0.0, 'slippage': 0.0, 'impact': 0.0, 'multiplier': 1.0}
            
        regime_mult = 1.0
        if regime == 2: regime_mult = 1.5
        elif regime >= 3: regime_mult = 3.0
            
        spread_cost = (self.base_spread_bps / 10000.0) / 2.0
        slippage_cost = current_volatility * self.slippage_vol_factor
        
        participation_rate = order_size_usd / adv_usd
        if participation_rate < 0.01:
            impact_cost = self.impact_factor * np.sqrt(participation_rate)
        else:
            impact_cost = self.impact_factor * participation_rate * 10.0 # Linear escalation
            
        total = (spread_cost + slippage_cost + impact_cost) * regime_mult
        
        return {
            'total_fractional_cost': total,
            'spread': spread_cost * regime_mult,
            'slippage': slippage_cost * regime_mult,
            'impact': impact_cost * regime_mult,
            'multiplier': regime_mult
        }

    def estimate_expected_impact(self, order_size_usd: float, adv_usd: float, current_volatility: float, regime: int) -> float:
        """Proactive Estimation: Predicts execution cost BEFORE action selection."""
        if order_size_usd == 0: return 0.0
        cost_dict = self.calculate_cost(100.0, order_size_usd, adv_usd, current_volatility, regime)
        return cost_dict['total_fractional_cost']
