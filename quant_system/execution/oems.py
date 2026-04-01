"""
quant_system/execution/oems.py
================================================================================
OFFLINE ORDER EXECUTION MANAGEMENT SYSTEM
================================================================================
"""

import os
import logging
from datetime import datetime

os.makedirs('quant_system/logs', exist_ok=True)
oems_logger = logging.getLogger('OEMS')
oems_logger.setLevel(logging.INFO)

class OrderManagementSystem:
    def __init__(self, log_path: str = "quant_system/logs/shadow_oems_blotter.csv"):
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                # Institutional physical fields tracking
                f.write("timestamp,side,quantity,intended_price,estimated_cost_usd\n")

    def resolve_action(self, target_allocation_fraction: float, current_physical_quantity: float, current_price: float, total_portfolio_usd: float) -> dict:
        """
        Translates fractional RL Neural inferences (action ∈ [0, 1]) directly into physical exchange volumes.
        Under the baseline Agent configuration, the Action dictates the percentage of the PRE-EXISTING 
        stance to violently liquidate down towards 0. 
        """
        # Calculate exactly how many digital/physical tokens form the intended closure footprint.
        fraction_to_sell = target_allocation_fraction
        physical_qty_to_execute = fraction_to_sell * current_physical_quantity
        
        # Directional mapping bounds
        side = "SELL" if physical_qty_to_execute > 0 else "HOLD"
        
        return {
            "side": side,
            "quantity": physical_qty_to_execute,
            "price": current_price
        }

    def route_order(self, side: str, quantity: float, current_price: float, slippage_bps: float = 2.0):
        """
        Bypasses live exchange gateways directly into the mathematical Shadow Tracker 
        preserving pure Implementation Shortfall boundaries.
        """
        if side == "HOLD" or quantity <= 0:
            return
            
        # Hard-calculate institutional impact variables
        notional_usd = quantity * current_price
        estimated_slippage_cost = notional_usd * (slippage_bps / 10000.0)
        
        ts = datetime.now().isoformat()
        
        with open(self.log_path, "a") as f:
            f.write(f"{ts},{side},{quantity:.8f},{current_price:.4f},{estimated_slippage_cost:.4f}\n")
            
        oems_logger.info(f"OEMS SHADOW ROUTED: {side} {quantity:.4f} units @ ${current_price:.2f} | Est Slippage: ${estimated_slippage_cost:.2f}")
