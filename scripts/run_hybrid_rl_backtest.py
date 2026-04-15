"""
scripts/run_hybrid_rl_backtest.py
================================================================================
HYBRID ML + RL BACKTESTING & INTEGRATION SCRIPT
================================================================================

Automates the integration of the Phase 2 Hybrid Meta-Learner (LSTM + Trees)
with the Phase 3 DQN Trade Manager for path-dependent execution.

Capabilities:
1. Data Pipeline: Aligns historical OHLCV, L2 Imbalance, and Cross-Asset Beta.
2. Phase 2: Generates Hybrid Meta-Learner probability tensors.
3. Phase 3: Feeds the RL Agent to optimize partial/full exits dynamically.
4. Backtesting: Compares RL-enhanced trajectory vs Static Baseline.
5. Live Hooks: Exposes `evaluate_live_tick()` for realtime ExecutionManager use.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List

# Explicitly add the root project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_meta_learner import HybridMetaLearner
from models.rl_trade_manager import DynamicTradeManagerDQN, DynamicTradeEnvironment
from core.config import ApexConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hybrid-RL-Backtester")


# ==============================================================================
# A. DATA PIPELINE
# ==============================================================================
def construct_mock_data(n_bars: int = 5000) -> pd.DataFrame:
    """
    Constructs a synthetic dataframe mimicking aligned Phase 1 extracted features.
    In production, this is replaced by data fetched from `FeatureEngine`.
    """
    logger.info("A. Constructing synthetic OHLCV + Alternative Data Pipeline...")
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    
    # Synthetic correlated random walk
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = np.exp(np.cumsum(returns)) * 100.0
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
        'Volume': np.random.lognormal(10, 1, n_bars),
        'l2_imbalance': np.random.normal(0, 0.5, n_bars),                # Phase 1
        'l2_imbalance_surge': np.random.normal(0, 1.0, n_bars),          # Phase 1
        'cross_asset_corr_btc': np.clip(np.random.normal(0.6, 0.2, n_bars), -1, 1),
        'cross_asset_corr_spy': np.clip(np.random.normal(0.3, 0.2, n_bars), -1, 1)
    }, index=dates)
    
    # Target: 1 if return over next 10 bars > 0.2%, else 0
    df['target_return'] = df['Close'].pct_change(10).shift(-10)
    df['label'] = (df['target_return'] > 0.002).astype(int)
    
    return df.dropna()


# ==============================================================================
# B. PHASE 2: META-LEARNER EVALUATION
# ==============================================================================
def train_and_eval_meta_learner(df: pd.DataFrame, seq_length: int = 30) -> np.ndarray:
    """
    Initializes and trains the Phase 2 LSTM+Tree architecture, then outputs
    probabilistic trade signals used as State inputs for the RL Agent.
    """
    logger.info("B. Initializing Phase 2 Hybrid Meta-Learner (LSTM + XGBoost/LGBM)")
    
    features = ['l2_imbalance', 'l2_imbalance_surge', 'cross_asset_corr_btc', 'cross_asset_corr_spy']
    X_raw = df[features].values
    y = df['label'].values
    
    # Train/Test Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X_raw[:split_idx], y[:split_idx]
    X_test, y_test = X_raw[split_idx:], y[split_idx:]
    
    learner = HybridMetaLearner(seq_length=seq_length)
    learner.train_pipeline(X_train, y_train)
    
    # Generate probabilities for the entire dataset
    logger.info("   -> Generating probabilistic Meta-Signals over entire timeline...")
    hybrid_probs = np.zeros(len(df))
    
    for i in range(seq_length, len(df)):
        window = X_raw[i - seq_length : i]
        hybrid_probs[i] = learner.predict_meta_label(window)
        
    df['meta_prob'] = hybrid_probs
    return hybrid_probs


# ==============================================================================
# C. & D. PHASE 3: DQN INTEGRATION & BACKTESTING
# ==============================================================================
def run_rl_backtest(df: pd.DataFrame):
    """
    Initializes the DQN Agent and Environment, trains via episodic replay,
    and calculates final backtest metrics vs a static baseline.
    """
    logger.info("C. Initializing Phase 3 DQN Trade Manager Integration")
    
    prices = df['Close'].values
    meta_probs = df['meta_prob'].values
    l2_imb = df['l2_imbalance'].values
    cross_spy = df['cross_asset_corr_spy'].values
    
    # Init Env & Agent
    env = DynamicTradeEnvironment(prices, meta_probs, l2_imb, cross_spy, slippage_bps=2.0)
    agent = DynamicTradeManagerDQN(state_dim=8, action_dim=3, epsilon_decay=2000)
    
    logger.info("   -> Training DQN Agent via Experience Replay (500 episodes)...")
    episodes = 500
    for e in range(episodes):
        start_idx = np.random.randint(0, len(prices) - 100)
        state = env.reset(start_idx)
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.push_memory(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            
        if (e + 1) % 100 == 0:
            logger.info(f"      [Episode {e+1}/{episodes}] Epsilon: {agent.epsilon:.3f}")
            
    # D. BACKTEST METRICS (Out of sample emulation)
    logger.info("D. Executing Full Backtest & Evaluating Metrics")
    evaluate_agent_performance(agent, env, df)


def evaluate_agent_performance(agent: DynamicTradeManagerDQN, env: DynamicTradeEnvironment, df: pd.DataFrame):
    """Compares the active DQN management against a baseline HOLD strategy."""
    trades = 200
    rl_returns = []
    baseline_returns = []
    action_counts = {0: 0, 1: 0, 2: 0} # Hold, Partial, Full
    
    for _ in range(trades):
        start_idx = np.random.randint(int(len(df)*0.8), len(df) - 100)
        
        # 1. RL Agent Run
        state = env.reset(start_idx)
        done = False
        rl_pnl = 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            action_counts[action] += 1
            state, reward, done, info = env.step(action)
            if done:
                rl_pnl = info['pnl']
        rl_returns.append(rl_pnl)
        
        # 2. Static Baseline (Hold for exactly 20 bars, or until -1.5% SL)
        entry_price = env.price_data[start_idx]
        static_pnl = 0.0
        for offset in range(1, 21):
            curr_price = env.price_data[start_idx + offset]
            static_pnl = (curr_price / entry_price) - 1.0
            if static_pnl < -0.015: # Hard stop
                break
        baseline_returns.append(static_pnl)
        
    # Calculate Metrics
    rl_ret_arr = np.array(rl_returns)
    base_ret_arr = np.array(baseline_returns)
    
    rl_win_rate = np.mean(rl_ret_arr > 0)
    base_win_rate = np.mean(base_ret_arr > 0)
    
    rl_sharpe = np.sqrt(252*288) * np.mean(rl_ret_arr) / (np.std(rl_ret_arr) + 1e-8)
    base_sharpe = np.sqrt(252*288) * np.mean(base_ret_arr) / (np.std(base_ret_arr) + 1e-8)
    
    logger.info("="*60)
    logger.info("BACKTEST RESULTS: HYBRID ML + DQN vs STATIC BASELINE")
    logger.info("="*60)
    logger.info(f"DQN RL Win Rate    : {rl_win_rate:.2%}")
    logger.info(f"Baseline Win Rate  : {base_win_rate:.2%}")
    logger.info(f"DQN RL Sharpe      : {rl_sharpe:.2f}")
    logger.info(f"Baseline Sharpe    : {base_sharpe:.2f}")
    logger.info(f"DQN Agent Action Distributions:")
    logger.info(f"   Holds            : {action_counts[0]}")
    logger.info(f"   Parts (50% scale): {action_counts[1]}")
    logger.info(f"   Full Closes      : {action_counts[2]}")
    
    # Save dummy metrics
    save_logs(rl_returns, base_returns=baseline_returns)


# ==============================================================================
# E. LIVE FORWARD SIMULATION HOOKS
# ==============================================================================
def evaluate_live_tick(
    agent: DynamicTradeManagerDQN, 
    learner: HybridMetaLearner, 
    recent_window_df: pd.DataFrame, 
    entry_price: float, 
    current_size: float
) -> int:
    """
    MODULAR HOOK: Designed to be imported directly into execution_loop.py.
    Accepts rolling OHLCV + Alt Data, outputs 0, 1, or 2 for execution.
    """
    features = ['l2_imbalance', 'l2_imbalance_surge', 'cross_asset_corr_btc', 'cross_asset_corr_spy']
    X_recent = recent_window_df[features].values
    
    # 1. Ask Phase 2 for Confidence
    meta_prob = learner.predict_meta_label(X_recent)
    
    # 2. Construct Phase 3 RL State Vector
    current_price = recent_window_df['Close'].iloc[-1]
    pnl_pct = (current_price / entry_price) - 1.0
    l2_imb = recent_window_df['l2_imbalance'].iloc[-1]
    cross = recent_window_df['cross_asset_corr_spy'].iloc[-1]
    vol = np.std(recent_window_df['Close'].tail(20)) / current_price
    
    state = np.array([
        pnl_pct * 10.0,
        max(0, entry_price - current_price) / entry_price * 10.0,
        meta_prob,
        l2_imb,
        cross,
        current_size,
        0.5, # Time decay proxy
        vol
    ])
    
    # 3. Query DQN Agent
    action_cmd = agent.select_action(np.nan_to_num(state), evaluate=True)
    return action_cmd


# ==============================================================================
# F. LOGGING & VISUALIZATION
# ==============================================================================
def save_logs(rl_returns: List[float], base_returns: List[float]):
    """Save backtest metrics to disk and generate plot artifacts."""
    logger.info("F. Generating Logs and Visualization Artifacts...")
    os.makedirs('logs', exist_ok=True)
    
    pd.DataFrame({
        'DQN_Returns': rl_returns,
        'Baseline_Returns': base_returns
    }).to_csv('logs/hybrid_rl_backtest_results.csv', index=False)
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(rl_returns), label="DQN Trade Manager PnL", color='blue')
    plt.plot(np.cumsum(base_returns), label="Static Meta-Learner (No DQN)", color='gray', alpha=0.7)
    plt.title("Phase 2/3 Hybrid Evaluation: DQN RL Agent vs Baseline")
    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join("logs", "hybrid_rl_backtest_curve.png")
    plt.savefig(plot_path)
    logger.info(f"   -> Saved performance chart to {plot_path}")


if __name__ == "__main__":
    logger.info("Starting End-to-End Hybrid ML + RL Pipeline Integration script...")
    
    # A. Execute Data Pipeline
    timeline_df = construct_mock_data(3000)
    
    # B. Train Phase 2 LSTM+Trees
    train_and_eval_meta_learner(timeline_df, seq_length=30)
    
    # C & D. Train Phase 3 DQN & Backtest
    run_rl_backtest(timeline_df)
    
    logger.info("Pipeline Execution Complete.")
