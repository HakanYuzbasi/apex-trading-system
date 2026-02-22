"""
execution/transaction_cost_optimizer.py
Calculates execution impact and optimal chunk sizes for Parent Orders.
"""
import math
import logging

logger = logging.getLogger(__name__)

class TransactionCostOptimizer:
    def __init__(self, default_spread_bps: float = 2.0):
        self.default_spread_bps = default_spread_bps

    def optimize_child_order_size(self, parent_qty: float, adv: float, max_impact_bps: float, volatility: float) -> float:
        """
        Almgren-Chriss variant: Determine the optimal child order size to stay under max_impact_bps.
        """
        if adv <= 0 or volatility <= 0:
            return max(1.0, parent_qty * 0.1) # Default to 10% chunks
            
        target_participation = (max_impact_bps / 10000.0 / (0.1 * volatility)) ** 2
        optimal_qty = target_participation * adv
        
        # Constrain to not exceed parent quantity, but guarantee at least 1 unit
        return max(1.0, min(parent_qty, optimal_qty))

    def calculate_dynamic_reprice(self, current_price: float, side: str, spread: float, seconds_unfilled: int, max_budget_bps: float) -> float:
        """
        Queue position aware repricing. Aggressively step up/down based on time un-filled in the order book.
        """
        urgency_factor = min(1.0, seconds_unfilled / 60.0) # Max urgency reached at 60s
        tick_adjustment = spread * urgency_factor * 0.5
        
        # Constrain by max slippage budget
        max_price_move = current_price * (max_budget_bps / 10000.0)
        actual_adjustment = min(tick_adjustment, max_price_move)
        
        if side == "BUY":
            return current_price + actual_adjustment # Pay more to get filled
        else:
            return current_price - actual_adjustment # Ask for less to get filled
