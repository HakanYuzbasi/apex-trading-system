"""
quant_system/infrastructure/persistence.py
================================================================================
CRASH-RESILIENT STATE MANAGER (HOT RELOADING)
================================================================================
"""

import json
import os
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, scaler=None, risk_manager=None, env=None):
        self.scaler = scaler
        self.risk_manager = risk_manager
        self.env = env
        
    def save_checkpoint(self, filepath: str = "quant_system/state/production_checkpoint.json"):
        """
        Physically locks Welford Variances, Hysteresis bounds, and physical asset trackers dynamically mapping to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state_dump = {}
        
        if self.scaler:
            state_dump["scaler"] = {
                "count": int(self.scaler.count),
                "mean": self.scaler.mean.tolist(),
                "M2": self.scaler.M2.tolist()
            }
            
        if self.risk_manager:
            state_dump["risk_manager"] = {
                "condition": self.risk_manager.condition,
                "ticks_in_recovery": int(self.risk_manager.ticks_in_recovery),
                "kl_ema": float(self.risk_manager.kl_ema) if self.risk_manager.kl_ema is not None else None
            }
            
        if self.env:
            state_dump["portfolio"] = {
                "position_sz": float(self.env.position_sz),
                "physical_quantity": float(self.env.asset_quantity),
                "entry_price": float(self.env.entry_price),
                "peak_pnl": float(self.env.peak_pnl),
                "recent_cost_spike": float(self.env.recent_cost_spike),
                "previous_action": float(self.env.previous_action)
            }
            
        with open(filepath, "w") as f:
            json.dump(state_dump, f, indent=4)
            
        logger.info(f"System Brain Safely Serialized -> {filepath}")

    def load_checkpoint(self, filepath: str = "quant_system/state/production_checkpoint.json") -> bool:
        """
        Instantiates hot swaps directly from node loss arrays restoring memory footprint instantly.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Production Checkpoint missing at {filepath}. Initializing from Baseline (Zero State).")
            return False
            
        try:
            with open(filepath, "r") as f:
                state_dump = json.load(f)
                
            if self.scaler and "scaler" in state_dump:
                s_dict = state_dump["scaler"]
                if s_dict["count"] > 0:
                    self.scaler.count = s_dict["count"]
                    self.scaler.mean = np.array(s_dict["mean"])
                    self.scaler.M2 = np.array(s_dict["M2"])
                    
            if self.risk_manager and "risk_manager" in state_dump:
                r_dict = state_dump["risk_manager"]
                self.risk_manager.condition = r_dict.get("condition", "GREEN")
                self.risk_manager.ticks_in_recovery = r_dict.get("ticks_in_recovery", 0)
                self.risk_manager.kl_ema = r_dict.get("kl_ema")
                
            if self.env and "portfolio" in state_dump:
                p_dict = state_dump["portfolio"]
                self.env.position_sz = p_dict.get("position_sz", 1.0)
                self.env.asset_quantity = p_dict.get("physical_quantity", 0.0)
                self.env.entry_price = p_dict.get("entry_price", 0.0)
                self.env.peak_pnl = p_dict.get("peak_pnl", 0.0)
                self.env.recent_cost_spike = p_dict.get("recent_cost_spike", 0.0)
                self.env.previous_action = p_dict.get("previous_action", 0.0)
                
            logger.info(f"System Memory Restored Continuously from -> {filepath}")
            return True
        except Exception as e:
            logger.error(f"FATAL: Hot Reload Corrupted -> {e}")
            return False
