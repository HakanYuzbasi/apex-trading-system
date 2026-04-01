"""
scripts/run_full_validation.py
================================================================================
END-TO-END QUANTITATIVE ARCHITECTURE VALIDATION (PHASES 1-5)
================================================================================

Executes the comprehensive automated backtesting sequence for the 5-Phase
Advanced Quantitative Pipeline:

[Phase 1]: Alternative Data & Micro-Structure Extraction (L2, Cross-Asset)
[Phase 2]: Hybrid Meta-Learning (LSTM + LightGBM/XGBoost)
[Phase 3]: DQN Trade Management (Dynamic Exits via RL)
[Phase 4]: Hierarchical Risk Parity & Liquidity Sizing
[Phase 5]: Purged Cross-Validation & Tail-Risk Governance
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Connect to core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_meta_learner import HybridMetaLearner
from models.rl_trade_manager import DynamicTradeManagerDQN, DynamicTradeEnvironment
from models.portfolio_optimizer import DynamicPortfolioManager
from models.governance_engine import PurgedKFold, StressTestEngine, GovernanceMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase5-Validator")


# ==============================================================================
# A. DATA PIPELINE
# ==============================================================================
def construct_multivariate_universe(n_bars: int = 5000) -> Dict[str, pd.DataFrame]:
    """Generates a synthetic universe reflecting Phase 1 Data Extractions."""
    logger.info("A. Loading Alternative Data Universe (L2 Imbalance, Cross-Asset Betas)...")
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'SPY', 'QQQ']
    universe = {}
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    
    # Base SPY/BTC returns for correlation
    base_returns = {
        'SPY': np.random.normal(0.0001, 0.001, n_bars),
        'BTC/USD': np.random.normal(0.0002, 0.002, n_bars)
    }
    
    for sym in symbols:
        # Correlate returns
        ret = np.random.normal(0.0001, 0.002, n_bars)
        if 'USD' in sym:
            ret += base_returns['BTC/USD'] * 0.7
        else:
            ret += base_returns['SPY'] * 0.8
            
        prices = np.exp(np.cumsum(ret)) * 100.0
        
        df = pd.DataFrame({
            'Close': prices,
            'l2_imbalance': np.random.normal(0, 0.5, n_bars),
            'l2_imbalance_surge': np.random.normal(0, 1.0, n_bars),
            'cross_asset_corr_btc': np.clip(np.random.normal(0.7 if 'USD' in sym else 0.1, 0.2, n_bars), -1, 1),
            'cross_asset_corr_spy': np.clip(np.random.normal(0.8 if sym in ['SPY','QQQ'] else 0.2, 0.2, n_bars), -1, 1),
            'Volume': np.random.lognormal(10, 1, n_bars)
        }, index=dates)
        
        # Target Generation
        df['target_return'] = df['Close'].pct_change(10).shift(-10)
        df['label'] = (df['target_return'] > 0.002).astype(int)
        
        universe[sym] = df.dropna()
        
    return universe


# ==============================================================================
# E. GOVERNANCE & PURGED CV INTEGRATION (Orchestrates B, C, D)
# ==============================================================================
def execute_purged_cv_pipeline(universe: Dict[str, pd.DataFrame]):
    """
    Executes the full 5-Phase pipeline wrapped securely inside a 
    Combinatorially Purged Cross-Validation loop.
    """
    logger.info("E. Initializing Phase 5 Combinatorially Purged CV Automation...")
    
    # Initialize Core Engines
    gov_monitor = GovernanceMonitor(ic_threshold=0.02, dd_threshold=-0.15)
    cpcv = PurgedKFold(n_splits=5, purge_pct=0.05, n_test_splits=2)
    portfolio_mgr = DynamicPortfolioManager(max_capital_usd=100000.0)
    
    # We will run the pipeline on the most volatile asset for the deep RL tracking
    target_sym = 'BTC/USD'
    df = universe[target_sym]
    
    features = ['l2_imbalance', 'l2_imbalance_surge', 'cross_asset_corr_btc', 'cross_asset_corr_spy']
    X = df[features].values
    y = df['label'].values
    prices = df['Close'].values
    
    splits = cpcv.split(X)
    
    fold_metrics = []
    
    logger.info(f"   Executing {len(splits)} Purged Combinatorial Folds...")
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"--- FOLD {fold+1}/{len(splits)} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        price_test = prices[test_idx]
        
        # ==================================================================
        # B. PHASE 2: META-LEARNER EVALUATION
        # ==================================================================
        logger.info("   [Phase 2] Training Hybrid Meta-Learner (LSTM + XGBoost)...")
        learner = HybridMetaLearner(seq_length=30)
        learner.train_pipeline(X_train, y_train)
        
        # Hash & Audit
        gov_monitor.hash_model(f"HybridMeta_Fold{fold}", {"seq": 30})
        
        # Predict Test Set
        hybrid_probs = np.zeros(len(X_test))
        for i in range(30, len(X_test)):
            hybrid_probs[i] = learner.predict_meta_label(X_test[i-30:i])
            
        # ==================================================================
        # C. PHASE 3: DQN TRADE MANAGER
        # ==================================================================
        logger.info("   [Phase 3] Training DQN Trade Management Agent...")
        # Train on a slice of the train set to learn dynamics
        env_train = DynamicTradeEnvironment(prices[train_idx], y_train, X_train[:, 0], X_train[:, 2]) # Mocking inputs
        agent = DynamicTradeManagerDQN()
        
        for _ in range(200): # Light train for CV speed
            s = env_train.reset()
            done = False
            while not done:
                a = agent.select_action(s)
                s_next, r, done, _ = env_train.step(a)
                agent.push_memory(s, a, r, s_next, done)
                agent.optimize_model()
                s = s_next
                
        gov_monitor.hash_model(f"DQN_Fold{fold}", {"gamma": 0.99})
        
        # Evaluate RL on Test Set
        env_test = DynamicTradeEnvironment(price_test, hybrid_probs, X_test[:, 0], X_test[:, 2])
        test_pnl = 0.0
        
        for _ in range(50):
            s = env_test.reset(np.random.randint(0, len(price_test)-100))
            done = False
            while not done:
                a = agent.select_action(s, evaluate=True)
                s, r, done, info = env_test.step(a)
                if done: test_pnl += info['pnl']
                
        # ==================================================================
        # D. PHASE 4: HRP PORTFOLIO ALLOCATION
        # ==================================================================
        logger.info("   [Phase 4] Computing Hierarchical Risk Parity Allocations...")
        # Build out-of-sample return matrix across all assets for the test indices
        oos_returns = pd.DataFrame({s: universe[s]['Close'].iloc[test_idx].pct_change() for s in universe.keys()}).fillna(0)
        
        meta_sigs = {s: np.random.uniform(0.5, 0.9) for s in universe.keys()}
        meta_sigs[target_sym] = np.mean(hybrid_probs[30:]) # Use actual pred for target
        
        dqn_acts = {s: 0 for s in universe.keys()}
        liq = {s: {'adv': 1e7, 'spread_bps': 2.0} for s in universe.keys()}
        
        # Live Tick Array test
        final_allocs = portfolio_mgr.evaluate_live_portfolio_tick(oos_returns, meta_sigs, dqn_acts, liq)
        
        gov_monitor.hash_model(f"HRP_Fold{fold}", {"cap": 100000})
        
        # ==================================================================
        # E. GOVERNANCE STRESS TESTING
        # ==================================================================
        logger.info("   [Phase 5] Executing Tail-Risk Stress Simulations...")
        stressed_ret = StressTestEngine.simulate_flash_crash(oos_returns, crash_severity=-0.25)
        cvar = StressTestEngine.calculate_cvar(stressed_ret[target_sym].values, alpha=0.05)
        
        # Generate IC metrics
        from scipy.stats import spearmanr
        ic, _ = spearmanr(hybrid_probs[30:], (price_test[30:]/price_test[:-30])-1)
        
        fold_metrics.append({
            'Fold': fold + 1,
            'Information_Coefficient': ic,
            'RL_Test_PnL': test_pnl,
            'HRP_Capital_Deployed': sum(final_allocs.values()),
            'Tail_Risk_CVaR_5%': cvar
        })
        
        logger.info(f"   -> Fold {fold+1} IC: {ic:.4f} | RL PnL: {test_pnl:.2%} | CVaR: {cvar:.2%}")

    # Aggregation & Governance Audit
    metrics_df = pd.DataFrame(fold_metrics)
    os.makedirs('logs', exist_ok=True)
    metrics_df.to_csv('logs/purged_cv_metrics.csv', index=False)
    gov_monitor.generate_audit_report('logs/institutional_audit.csv')
    
    generate_visualizations(metrics_df)


# ==============================================================================
# F. OUTPUT & VISUALIZATION
# ==============================================================================
def generate_visualizations(metrics_df: pd.DataFrame):
    """Produces institutional visual artifacts for the Phase 5 pipeline."""
    logger.info("F. Generating Phase 5 Quantitative Visualizations...")
    
    plt.figure(figsize=(12, 5))
    
    # 1. IC Stability Across Purged Folds
    plt.subplot(1, 2, 1)
    plt.bar(metrics_df['Fold'], metrics_df['Information_Coefficient'], color='teal')
    plt.axhline(0.02, color='red', linestyle='--', label='Edge Threshold')
    plt.title('Meta-Learner IC Stability (Purged CV)')
    plt.xlabel('Combinatorial Fold')
    plt.ylabel('Information Coefficient (Rank Corr)')
    plt.legend()
    
    # 2. Tail Risk vs RL Return
    plt.subplot(1, 2, 2)
    plt.scatter(abs(metrics_df['Tail_Risk_CVaR_5%']), metrics_df['RL_Test_PnL'], color='purple', s=100)
    plt.title('Tail Risk (CVaR) vs RL PnL Profile')
    plt.xlabel('Absolute CVaR (Stress Test)')
    plt.ylabel('DQN Out-of-Sample PnL')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join("logs", "phase5_governance_report.png")
    plt.savefig(plot_path)
    logger.info(f"   -> Saved Institutional Teardown Plot: {plot_path}")
    logger.info("=" * 80)
    logger.info("END-TO-END 5-PHASE VALIDATION COMPLETE.")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("INITIATING APEX QUANTITATIVE PIPELINE (PHASES 1-5)")
    logger.info("=" * 80)
    
    # Run the full orchestrated sequence
    market_universe = construct_multivariate_universe(n_bars=2000)
    execute_purged_cv_pipeline(market_universe)
