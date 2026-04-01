"""
quant_system/governance/risk_manager.py
================================================================================
LIVE-MARKET FAIL-SAFE ARCHITECTURE (WITH ANTI-THRASHING HYSTERESIS)
================================================================================
"""

import numpy as np
import logging
import os

os.makedirs('quant_system/logs', exist_ok=True)
risk_logger = logging.getLogger('RiskManager')
risk_logger.setLevel(logging.INFO)
if not risk_logger.handlers:
    fh = logging.FileHandler('quant_system/logs/risk_alerts.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    risk_logger.addHandler(fh)

class SystemHealthMonitor:
    def __init__(self, drawdown_limit: float = -0.05, cooldown_ticks: int = 50, kl_alpha: float = 0.1):
        self.vol_history = []
        self.kl_history = []
        self.drawdown_limit = drawdown_limit 
        self.condition = "GREEN"
        
        # UPGRADE 2 & 3: Anti-Thrashing Memory Elements
        self.kl_ema = None
        self.kl_alpha = kl_alpha
        self.cooldown_ticks = cooldown_ticks
        self.ticks_in_recovery = 0
        
    def evaluate_health(self, current_vol: float, current_kl: float, current_drawdown: float) -> str:
        self.vol_history.append(current_vol)
        
        # UPGRADE 3: KL-Divergence Smoothing (EMA)
        if self.kl_ema is None:
            self.kl_ema = current_kl
        else:
            self.kl_ema = (self.kl_alpha * current_kl) + ((1.0 - self.kl_alpha) * self.kl_ema)
            
        self.kl_history.append(self.kl_ema)
        
        # Determine Volatility Sigma
        vol_z = 0.0
        if len(self.vol_history) > 50:
            mean_vol = np.mean(self.vol_history[:-1])
            std_vol = np.std(self.vol_history[:-1]) + 1e-8
            vol_z = (current_vol - mean_vol) / std_vol
            
        # Determine EMA Divergence Percentile
        kl_alert = False
        if len(self.kl_history) > 50:
            pct_99 = np.percentile(self.kl_history[:-1], 99)
            if self.kl_ema > pct_99:
                kl_alert = True
                
        # UPGRADE 1: Asymmetric Escalation Thresholds
        red_dd_trigger = (current_drawdown <= -0.050)
        red_vol_trigger = (vol_z >= 5.0)
        
        yellow_dd_trigger = (current_drawdown <= -0.030)
        yellow_kl_trigger = kl_alert
        
        # UPGRADE 1: Asymmetric Recovery (De-escalation) Thresholds
        red_dd_recovered = (current_drawdown >= -0.040)
        vol_recovered = (vol_z <= 3.5)
        
        yellow_dd_recovered = (current_drawdown >= -0.020)
        kl_recovered = (not kl_alert)
        
        next_condition = self.condition
        escalated = False
        
        # --- STATE TRANSITION LOGIC ---
        
        if self.condition == "GREEN":
            # Escalations are INSTANT
            if red_dd_trigger or red_vol_trigger:
                next_condition = "RED"
                escalated = True
            elif yellow_dd_trigger or yellow_kl_trigger:
                next_condition = "YELLOW"
                escalated = True
                
        elif self.condition == "YELLOW":
            # Escalate instantly to RED if limits hit
            if red_dd_trigger or red_vol_trigger:
                next_condition = "RED"
                escalated = True
            else:
                # De-escalation to GREEN requires stable recovery
                if yellow_dd_recovered and kl_recovered and vol_recovered:
                    self.ticks_in_recovery += 1
                else:
                    self.ticks_in_recovery = 0
                    
                if self.ticks_in_recovery >= self.cooldown_ticks:
                    next_condition = "GREEN"
                    self.ticks_in_recovery = 0

        elif self.condition == "RED":
            # De-escalation from RED requires stable recovery across multiple fronts
            if red_dd_recovered and vol_recovered:
                self.ticks_in_recovery += 1
            else:
                self.ticks_in_recovery = 0
                
            if self.ticks_in_recovery >= self.cooldown_ticks:
                # Check if we naturally step down entirely to GREEN or simply YELLOW
                if yellow_dd_recovered and kl_recovered:
                    next_condition = "GREEN"
                else:
                    next_condition = "YELLOW"
                self.ticks_in_recovery = 0

        # Reset consecutive counter if escalated
        if escalated:
            self.ticks_in_recovery = 0

        if next_condition != self.condition:
            self._log_and_update(next_condition, vol_z, kl_alert, current_drawdown)
            
        return self.condition

    def _log_and_update(self, new_condition: str, vol_z: float, kl_alert: bool, dd: float):
        risk_logger.info(
            f"STATUS ALERT: {self.condition} -> {new_condition} "
            f"| Vol Z: {vol_z:.2f} | KL EMA: {self.kl_ema:.4f} (Alert={kl_alert}) | DD: {dd:.2%}"
        )
        self.condition = new_condition

    def apply_graceful_degradation(self, action: float, condition: str) -> float:
        """
        Graceful Degradation overrides standard capital distributions linearly based on governance flags.
        """
        if condition == "GREEN":
            return action
        elif condition == "YELLOW":
            return min(action, 0.5)
        elif condition == "RED":
            return 0.10
