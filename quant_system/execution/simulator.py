"""
quant_system/execution/simulator.py
================================================================================
Multi-Phase Environment Simulator (WITH OEMS DEPLOYMENT)
================================================================================
"""

import numpy as np
import logging
from typing import Tuple, Dict
import os
import pandas as pd
from quant_system.execution.cost_model import TransactionCostModel
from quant_system.features.state_scaler import OnlineStateScaler
from quant_system.execution.oems import OrderManagementSystem

logger = logging.getLogger(__name__)

class RLSimulatorEnv:
    def __init__(
        self, price_array: np.ndarray, meta_probs: np.ndarray, confidences: np.ndarray,
        disagreements: np.ndarray, features: np.ndarray, regimes: np.ndarray,
        lambda_dd: float = 2.0, lambda_vol: float = 1.0, lambda_cost_ratio: float = 0.5
    ):
        self.prices, self.meta_probs, self.confidences = price_array, meta_probs, confidences
        self.disagreements, self.features, self.regimes = disagreements, features, regimes
        
        self.lambda_dd, self.lambda_vol, self.lambda_cost_ratio = lambda_dd, lambda_vol, lambda_cost_ratio * 1.2 
        self.cost_model = TransactionCostModel(base_spread_bps=2.0)
        self.scaler = OnlineStateScaler(15, exclude_dims=[13]) 
        self.oems = OrderManagementSystem()
        
        self.STATE_INDEX = {
            "feat0": 0, "feat1": 1, "feat2": 2, "feat3": 3,
            "meta_prob": 4, "confidence": 5, "disagreement": 6,
            "expected_impact": 7, "recent_cost_spike": 8, "prob_ma": 9,
            "position": 10, "pnl": 11, "drawdown": 12, "regime": 13, "previous_action": 14
        }
        
        self.recent_cost_spike = 0.0
        self.alpha_cost = 0.2
        self.previous_action = 0.0 
        self.current_step = 0
        self.entry_price = 0.0
        self.position_sz = 1.0
        self.peak_pnl = 0.0
        
        self.total_portfolio_usd = 1000000.0
        self.asset_quantity = 0.0 # Physical Asset
        
        self.state_distribution = []
        self.action_distribution = []
        self.regime_behaviors = {1: [], 2: [], 3: []}
        self.metrics = {
            'total_steps': 0, 'traded_steps': 0, 'blocked_steps': 0,
            'cost_dominant_trades': 0, 'total_reward': 0.0, 'sum_action_size': 0.0,
            'filtered_micro_steps': 0, 'effective_trade_steps': 0, 'capital_utilized': 0.0
        }

    def reset(self, start_idx: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        self.current_step = start_idx
        self.entry_price = self.prices[start_idx]
        self.position_sz = 1.0
        self.asset_quantity = (self.position_sz * self.total_portfolio_usd) / self.prices[start_idx]
        self.peak_pnl = 0.0
        self.recent_cost_spike = 0.0
        self.previous_action = 0.0
        for k in self.metrics: self.metrics[k] = 0.0 if 'sz' in k or 'reward' in k or 'cap' in k else 0
        return self._build_state()

    def _build_state(self) -> Tuple[np.ndarray, np.ndarray]:
        t = self.current_step
        t_lag = max(0, t - 1)
        
        open_pnl = (self.prices[t] / self.entry_price - 1.0) if self.position_sz > 0 else 0.0
        if open_pnl > self.peak_pnl: self.peak_pnl = open_pnl
        drawdown = min(0.0, open_pnl - self.peak_pnl)
        
        raw = np.zeros(15)
        raw[0:4] = self.features[t_lag, :4]
        raw[4] = self.meta_probs[t_lag]
        raw[5] = self.confidences[t_lag]
        raw[6] = self.disagreements[t_lag]
        raw[7] = self.cost_model.estimate_expected_impact(100000*self.position_sz, 1e6, np.std(self.prices[max(0,t-20):t+1])/self.prices[t], self.regimes[t])
        raw[8] = self.recent_cost_spike
        raw[9] = np.mean(self.meta_probs[max(0,t-10):t+1])
        raw[10] = self.position_sz
        raw[11] = open_pnl
        raw[12] = drawdown
        raw[13] = self.regimes[t_lag]
        raw[14] = self.previous_action
        
        scaled = self.scaler.scale(raw, update=True)
        self.state_distribution.append(raw.copy()) 
        
        return scaled, raw

    def step(self, action_dict: dict, shadow_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        self.current_step += 1
        self.metrics['total_steps'] += 1
        done = False
        t = self.current_step
        
        action_fraction = action_dict['action']
        self.previous_action = action_fraction 
        current_price = self.prices[t]
        
        # UPGRADE: Physical OEMS Translation Hook
        order_intent = self.oems.resolve_action(action_fraction, self.asset_quantity, current_price, self.total_portfolio_usd)
        
        if shadow_mode:
            self.oems.route_order(order_intent['side'], order_intent['quantity'], order_intent['price'])
            action_fraction = 0.0 
            
        self.action_distribution.append(action_fraction)
        regime = int(self.regimes[max(0, t-1)])
        self.regime_behaviors[regime].append(action_fraction)
        
        if t >= len(self.prices) - 1:
            done = True
            action_fraction = 1.0 
            
        if action_dict.get('blocked_by_ntz', False): self.metrics['blocked_steps'] += 1
        if action_dict.get('filtered_micro_trade', False): self.metrics['filtered_micro_steps'] += 1
        if action_fraction >= 0.01: self.metrics['effective_trade_steps'] += 1
            
        if action_fraction > 0:
            self.metrics['traded_steps'] += 1
            self.metrics['sum_action_size'] += action_fraction
            self.metrics['capital_utilized'] += action_fraction
            
        prev_price = self.prices[t-1]
        bar_pnl = (current_price / prev_price - 1.0)
        total_pnl = (current_price / self.entry_price - 1.0) if self.entry_price > 0 else 0.0
        volatility = np.std(self.prices[max(0, t-20):t+1]) / current_price
        
        confidence = self.confidences[max(0, t-1)]
        costs = {'total_fractional_cost': 0.0, 'regime_multiplier': 1.0}
        executed_volume = action_fraction * self.position_sz
        
        cost_ratio = 0.0
        realized_pnl = 0.0
        base_reward = 0.0
        
        if executed_volume > 0.01: 
            # Physical reduction applying OEMS properties
            qty_liquidated = fraction_to_sell = action_fraction * self.asset_quantity
            self.asset_quantity -= qty_liquidated
            
            costs = self.cost_model.calculate_cost(current_price, 100000 * executed_volume, 1000000, volatility, regime)
            current_cost = costs['total_fractional_cost']
            
            self.recent_cost_spike = self.alpha_cost * current_cost + (1 - self.alpha_cost) * self.recent_cost_spike
            realized_pnl = (total_pnl * executed_volume) - current_cost
            base_reward = realized_pnl * 5.0
            self.position_sz -= executed_volume
            
            cost_ratio = current_cost / (abs(total_pnl * executed_volume) + 1e-6)
            cost_ratio = min(cost_ratio, 5.0) 
            
            if current_cost > abs(total_pnl * executed_volume): self.metrics['cost_dominant_trades'] += 1
            
            if self.position_sz <= 0.01:
                done = True
                self.position_sz = 0.0
        else:
            base_reward = bar_pnl * self.position_sz 
            
        penalty_dd = self.lambda_dd * abs(min(0, total_pnl - self.peak_pnl))
        penalty_vol = self.lambda_vol * volatility
        penalty_cost = self.lambda_cost_ratio * np.log(1 + cost_ratio)
        
        reward = (base_reward - penalty_dd - penalty_vol) * confidence - penalty_cost
        self.metrics['total_reward'] += reward
        
        s_scaled, s_raw = self._build_state()
        return s_scaled, s_raw, float(reward), done, {'cost_breakdown': costs, 'bar_pnl': bar_pnl * self.position_sz}

    def export_analytics(self, directory: str = "quant_system/logs/"):
        os.makedirs(directory, exist_ok=True)
        np.savetxt(os.path.join(directory, "state_distribution.csv"), np.array(self.state_distribution), delimiter=",")
        acts = np.array(self.action_distribution)
        df_acts = pd.DataFrame([{
            'mean': np.mean(acts), 'zero_ratio': np.mean(acts == 0.0), 'large_alloc_ratio': np.mean(acts > 0.05)
        }])
        df_acts.to_csv(os.path.join(directory, "action_distribution.csv"), index=False)
        reg_df = []
        for reg, r_acts in self.regime_behaviors.items():
            if len(r_acts) > 0: reg_df.append({'regime': reg, 'avg_action': np.mean(r_acts), 'trade_freq': np.mean(np.array(r_acts) > 0)})
        pd.DataFrame(reg_df).to_csv(os.path.join(directory, "regime_behavior.csv"), index=False)
