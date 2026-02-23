"""
models/rl_environment.py - Reinforcement Learning Environment

OpenAI Gym-compatible environment wrapper for the Apex Backtest Engine.
Allows training RL agents (PPO, DQN, SAC) to trade using the simulation engine.

State Space:
- Technicals (price, returns, volatility)
- Microstructure (spread, volume profile)
- Portfolio (holdings, current pnl, cash)
- Market Regime (VIX state)

Action Space:
- Discrete: [Flat, Long, Short] or [Buy, Sell, Hold]
- Continuous: [Size of trade (-1.0 to 1.0)]
"""

import numpy as np
import pandas as pd
import logging

# Try to import gym
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    class gym:
        class Env: pass
    class spaces:
        class Box: 
            def __init__(self, *args, **kwargs): pass
        class Discrete: 
            def __init__(self, *args, **kwargs): pass

from backtesting.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Trading Environment for RL Agents.
    """
    
    def __init__(self, engine: BacktestEngine, symbol: str, features: pd.DataFrame, window_size: int = 10):
        """
        Initialize Environment.
        
        Args:
            engine: BacktestEngine instance
            symbol: Symbol to trade
            features: Pre-computed features DataFrame (index=timestamp)
            window_size: Number of past steps to include in observation
        """
        if not GYM_AVAILABLE:
            logger.warning("gym not available - RL Environment disabled")
            return
            
        super(TradingEnvironment, self).__init__()
        
        self.engine = engine
        self.symbol = symbol
        self.features = features
        self.window_size = window_size
        
        # Align features with engine data if possible
        # (Assuming engine data loaded)
        
        self.timestamps = self.features.index.tolist()
        self.current_step = window_size
        
        # Define Action Space: Continuous [-1, 1] representing target position size
        # -1 = Max Short, 0 = Flat, 1 = Max Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Define Observation Space
        # Window of features + Account State (Equity/Capital, CurrPos)
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * n_features + 2,), 
            dtype=np.float32
        )
        
        self.initial_balance = engine.initial_capital
        self.prev_equity = self.initial_balance
        
    def reset(self):
        """Reset environment to start."""
        self.current_step = self.window_size
        # Reset engine logic manually or re-init engine if needed
        # Since engine tracks history, we might need a fresh engine for clean episodes
        self.engine.cash = self.initial_balance
        self.engine.positions.clear()
        self.engine.trades.clear()
        self.engine.history.clear()
        self.prev_equity = self.initial_balance
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute action and return next state.
        
        Args:
            action: float [-1, 1] target position
        """
        target_pct = np.clip(action[0], -1, 1)
        
        # Get current timestamp
        if self.current_step >= len(self.timestamps) - 1:
            return self._get_observation(), 0, True, {}
            
        current_ts = self.timestamps[self.current_step]
        self.engine.current_time = current_ts
        self.engine._process_step(current_ts) # Update prices
        
        # Execute Trade
        # Calculate target shares
        current_equity = self.engine.total_equity()
        if self.symbol in self.engine.data:
            price = self.engine.data[self.symbol].loc[current_ts]['Close']
            
            # Target value
            target_value = current_equity * target_pct
            target_shares = int(target_value / price)
            
            # Current shares
            current_shares = 0
            if self.symbol in self.engine.positions:
                current_shares = self.engine.positions[self.symbol].quantity
                
            # Diff
            diff_shares = target_shares - current_shares
            
            if diff_shares != 0:
                side = 'BUY' if diff_shares > 0 else 'SELL'
                self.engine.execute_order(self.symbol, side, abs(diff_shares), price)
                
        # Calculate Reward
        # Reward = Change in Log Equity - Risk Penalty?
        # Simple Reward = Returns
        new_equity = self.engine.total_equity()
        try:
            reward = np.log(new_equity / self.prev_equity)
        except:
            reward = -1 # Bust
            
        self.prev_equity = new_equity
        
        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.timestamps) - 1
        
        # Early Stopping if broke
        if new_equity < self.initial_balance * 0.5:
            done = True
            reward = -10 # Penalty for blowing up
            
        obs = self._get_observation()
        
        return obs, reward, done, {'equity': new_equity}
        
    def _get_observation(self):
        """Construct observation vector."""
        # 1. Market Features Window
        start = self.current_step - self.window_size
        end = self.current_step
        mkt_features = self.features.iloc[start:end].values.flatten()
        
        # 2. Account State
        equity_ratio = self.engine.total_equity() / self.initial_capital
        
        current_pos_ratio = 0.0
        if self.symbol in self.engine.positions:
            pos = self.engine.positions[self.symbol]
            current_pos_ratio = (pos.quantity * pos.current_price) / self.engine.total_equity()
            
        state = np.concatenate([mkt_features, [equity_ratio, current_pos_ratio]])
        return state.astype(np.float32)
