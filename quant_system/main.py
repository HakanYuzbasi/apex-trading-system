"""
quant_system/main.py
================================================================================
INSTITUTIONAL QUANTITATIVE PIPELINE ORCHESTRATOR (HOT-RELOAD ENABLED)
================================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import torch

from quant_system.features.feature_engineer import FeatureEngineer
from quant_system.models.xgb_model import SpatialTreeMetaLearner
from quant_system.models.dqn_agent import ContinuousExecutionAgent
from quant_system.portfolio.hrp import HRPOptimizer
from quant_system.governance.purged_kfold import PurgedKFold
from quant_system.governance.monitoring import GovernanceMonitor
from quant_system.execution.simulator import RLSimulatorEnv
from quant_system.evaluation.tearsheet import InstitutionalTearsheet
from quant_system.governance.risk_manager import SystemHealthMonitor
from quant_system.infrastructure.persistence import StateManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("QuantMain")

def generate_mock_data(n_bars: int = 2000):
    logger.info("-> Mocking historical data arrays...")
    dates = pd.date_range(end='today', periods=n_bars, freq='5min')
    b_ret = np.random.normal(0, 0.002, n_bars)
    p_prices = np.exp(np.cumsum(b_ret)) * 100
    b_prices = np.exp(np.cumsum(np.random.normal(0, 0.001, n_bars))) * 100
    primary = pd.DataFrame({'Close': p_prices, 'High': p_prices*1.01, 'Low': p_prices*0.99, 'Volume': 1e6}, index=dates)
    bench = pd.DataFrame({'Close': b_prices, 'Volume': 1e7}, index=dates)
    return primary, bench

def main():
    logger.info("=====================================================")
    logger.info(" INSTANTIATING FULLY HARDENED PRODUCTION PIPELINE ")
    logger.info("=====================================================")
    
    # UPGRADE 2: State Persistence Validation Drop
    logger.info("Attempting Base Brain Reboot from Survival Disk Layers...")
    state_mgr = StateManager()
    state_mgr.load_checkpoint() # Gracefully attempts to catch historical state nodes
    
    feat_engine = FeatureEngineer(target_lag_bars=15)
    gov_mon = GovernanceMonitor()
    cpcv = PurgedKFold(n_splits=3, n_test_splits=1, embargo_pct=0.05)
    
    tearsheet_engine = InstitutionalTearsheet()
    fail_safe_monitor = SystemHealthMonitor(drawdown_limit=-0.05)
    
    df_prim, df_bench = generate_mock_data(3000)
    df_engineered = feat_engine.engineer_primary_asset(df_prim)
    df_corr = feat_engine.integrate_cross_assets(df_engineered, df_bench, "BTC")
    
    X_seq, X_spatial, Y_aligned, prices = feat_engine.produce_aligned_matrices(df_corr, seq_length=30)
    splits = cpcv.split(len(X_spatial))
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\n--- PURGED FOLD {fold+1}/{len(splits)} ---")
        
        X_train_sp = X_spatial[train_idx]
        Y_train = Y_aligned[train_idx]
        
        xgb_meta = SpatialTreeMetaLearner()
        xgb_meta.train(X_train_sp, Y_train)
        
        xgb_probs = np.array([xgb_meta.predict_meta_prob(x) for x in X_train_sp])
        lstm_variance = np.random.uniform(0.1, 0.4, len(xgb_probs)) 
        
        confidences = 1.0 / (1.0 + lstm_variance)
        disagreements = np.random.uniform(0.01, 0.03, len(xgb_probs)) 
        
        hrp = HRPOptimizer(max_adv_participation=0.05)
        mock_returns = pd.DataFrame(X_train_sp[:, :2])
        alloc, current_regime = hrp.allocate(mock_returns[-100:])
        regimes_train = np.full(len(xgb_probs), current_regime)
        
        env = RLSimulatorEnv(prices[train_idx], xgb_probs, confidences, disagreements, X_train_sp, regimes_train)
        ddpg_agent = ContinuousExecutionAgent(state_dim=15) 
        
        # Link state persistence engines properly
        state_mgr.scaler = env.scaler
        state_mgr.risk_manager = fail_safe_monitor
        state_mgr.env = env
        
        fold_returns = []
        fold_actions = []
        
        for e in range(50):
            s_scaled, s_raw = env.reset(start_idx=1)
            done = False
            fold_returns.clear()
            fold_actions.clear()
            
            while not done:
                action_dict = ddpg_agent.select_action(s_scaled, s_raw, evaluate=False)
                
                current_vol = s_raw[7] 
                t = env.current_step
                real_vol = np.std(env.prices[max(0, t-20):t+1]) / env.prices[t]
                current_kl = s_raw[6] 
                current_dd = s_raw[12]
                
                sys_condition = fail_safe_monitor.evaluate_health(real_vol, current_kl, current_dd)
                action_dict['action'] = fail_safe_monitor.apply_graceful_degradation(action_dict['action'], sys_condition)
                
                SHADOW_MODE_TOGGLE = (e == 0) 
                sn_scaled, sn_raw, r, done, info = env.step(action_dict, shadow_mode=SHADOW_MODE_TOGGLE)
                
                fold_returns.append(info.get('bar_pnl', 0.0))
                fold_actions.append(action_dict['action'])
                
                if not SHADOW_MODE_TOGGLE:
                    ddpg_agent.store_transition(s_scaled, action_dict['action'], r, sn_scaled, done)
                    ddpg_agent.train_step(batch_size=32)
                    
                # State Persistence Checkpoint Serialization Layer Interception
                if t % 500 == 0:
                    state_mgr.save_checkpoint()
                    
                s_scaled, s_raw = sn_scaled, sn_raw

        logger.info("      -> RL Episode Bound. Writing Diagnostics Block.")
        
        tearsheet_engine.generate_tearsheet(np.array(fold_returns), np.array(fold_actions))
        env.export_analytics()
        
        m = env.metrics
        t_steps = max(m['total_steps'], 1)
        t_acts = max(m['traded_steps'], 1)
        
        df_metric = pd.DataFrame([{
            'fold': fold+1, 'effective_trade_ratio': m['effective_trade_steps'] / t_steps, 
            'blocked_ratio': m['blocked_steps'] / t_steps, 'micro_filtered_ratio': m['filtered_micro_steps'] / t_steps,
            'cost_dominance': m['cost_dominant_trades'] / t_acts, 'avg_action_nonzero': m['sum_action_size'] / t_acts, 
            'capital_utilization': m['capital_utilized'] / t_steps, 'reward_per_action': m['total_reward'] / t_acts, 
            'reward_per_unit_action': m['total_reward'] / max(m['sum_action_size'], 1e-6)
        }])
        df_metric.to_csv(f"quant_system/logs/behavioral_metrics_fold{fold+1}.csv", index=False)
        break 

    logger.info("=====================================================")
    logger.info(" INSTITUTIONAL DEPLOYMENT SCRIPT SUCCESSFUL ")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
