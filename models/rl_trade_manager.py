"""
models/rl_trade_manager.py
================================================================================
PHASE 3: PATH-DEPENDENT REINFORCEMENT LEARNING EXITS
================================================================================

Implements a Deep Q-Network (DQN) Trade Management Agent to replace static
Take-Profit (TP) / Stop-Loss (SL) heuristics entirely.

Key Capabilities:
1. State Space: Incorporates Meta-Learner Probability, Open PnL, Drawdown, 
   L2 Imbalance, and Cross-Asset Beta vectors.
2. Action Space: [0: HOLD, 1: PARTIAL_CLOSE_50%, 2: FULL_CLOSE]
3. Reward Shaping: Penalizes deep drawdowns while rewarding optimal R:R capture 
   minus execution slippage/transaction costs.
"""

import numpy as np
import pandas as pd
import random
import logging
from collections import deque
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# 1. DQN ARCHITECTURE
# ------------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """Standard Feed-Forward Neural Net estimating Q-Values for Actions."""
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim * 2)
            self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.out = nn.Linear(hidden_dim, action_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.out(x) # Outputs Q-value for each action [Hold, Partial, Close]


# ------------------------------------------------------------------------------
# 2. RL AGENT & EXPERIENCE REPLAY
# ------------------------------------------------------------------------------
class DynamicTradeManagerDQN:
    """
    DQN Agent capable of partial sizing and dynamic path-aware exiting.
    Future hook: Hierarchical Risk Parity inputs as part of the state.
    """
    def __init__(
        self, 
        state_dim: int = 8, 
        action_dim: int = 3, 
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        memory_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval() # Target net does not train
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-Greedy Action Selection."""
        if not TORCH_AVAILABLE:
            return 0 # Default HOLD dummy
            
        # Exploit (Greedy)
        if evaluate or random.random() > self.epsilon:
            with torch.no_grad():
                state_ten = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_ten)
                return q_values.max(1)[1].item()
        # Explore (Random)
        else:
            return random.randrange(self.action_dim)
            
    def push_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experiences in Replay Buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def optimize_model(self):
        """Perform one step of Minibatch Gradient Descent on the Policy Net."""
        if not TORCH_AVAILABLE or len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        # Transpose the batch
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Compute max Q(s_{t+1}, a) for next states using Target Network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            # Zero out Q values for terminal states
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
            
        # Huber Loss (Smooth L1) protects against exploding gradients
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize Policy Network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Epsilon Decay
        self.steps_done += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1.0 - self.epsilon_end) / self.epsilon_decay
            
        # Target Network Sync
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# ------------------------------------------------------------------------------
# 3. RL ENVIRONMENT / BACKTEST SIMULATOR
# ------------------------------------------------------------------------------
class DynamicTradeEnvironment:
    """
    OpenAI Gym-style Environment to train the RL Agent on historical paths.
    """
    def __init__(self, price_data: np.ndarray, meta_signals: np.ndarray, l2_imbalances: np.ndarray, cross_assets: np.ndarray, slippage_bps: float = 2.0):
        self.price_data = price_data      # [T, 4] (OHLC proxy)
        self.meta_signals = meta_signals  # [T] (Hybrid Learner outputs)
        self.l2_imb = l2_imbalances       # [T]
        self.cross = cross_assets         # [T]
        self.slippage = slippage_bps / 10000.0
        
        self.current_step = 0
        self.entry_price = 0.0
        self.current_position_size = 0.0  # 1.0 = Full, 0.5 = Half, 0.0 = Flat
        self.trade_active = False
        
    def reset(self, start_step: int = 0) -> np.ndarray:
        self.current_step = start_step
        self.entry_price = self.price_data[self.current_step]
        self.current_position_size = 1.0  # Assume we enter full size
        self.trade_active = True
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        """
        State Vector: 
        1. Current PnL (%)
        2. Drawdown from Peak (%)
        3. Meta-Signal Probability
        4. OBI (L2 Imbalance)
        5. Cross-Asset Beta
        6. Position Size (1.0 or 0.5)
        7. Bar Duration counter
        8. Volatility proxy
        """
        current_price = self.price_data[self.current_step]
        pnl_pct = (current_price / self.entry_price) - 1.0
        # Simplistic assumptions for state dimension requirements
        state = np.array([
            pnl_pct * 10.0,                                    # PnL (Scaled)
            max(0, self.entry_price - current_price) / self.entry_price * 10.0, # Unrealized Drawdown
            self.meta_signals[self.current_step],              # Meta prob
            self.l2_imb[self.current_step],                    # OBI
            self.cross[self.current_step],                     # Cross-asset
            self.current_position_size,                        # Scale
            min(self.current_step / 100.0, 1.0),               # Time decay
            np.std(self.price_data[max(0, self.current_step-20):self.current_step+1]) / current_price # Vol
        ])
        return np.nan_to_num(state)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Actions: 0 (Hold), 1 (Partial 50%), 2 (Full Close)
        """
        self.current_step += 1
        done = False
        reward = 0.0
        
        if self.current_step >= len(self.price_data) - 1:
            done = True
            action = 2 # Force close at end
            
        current_price = self.price_data[self.current_step]
        bar_pnl = (current_price / self.price_data[self.current_step-1]) - 1.0
        total_pnl = (current_price / self.entry_price) - 1.0
        
        # 1. Reward Shaping Structure
        # -----------------------------
        # In a generic HOLD, we accumulate the tick PnL.
        if action == 0:  # HOLD
            reward = bar_pnl * self.current_position_size
            # Small penalty to encourage faster exits if PnL is stagnant
            reward -= 0.0001
            
        elif action == 1: # PARTIAL CLOSE (50% scale-out)
            if self.current_position_size > 0.6: # If we haven't scaled yet
                realized = total_pnl - self.slippage
                reward = (realized * 0.5) * 2.0 # Explicit reward for locking in profit
                self.current_position_size = 0.5
            else:
                reward = -0.01 # Penalty for invalid action (trying to scale out twice)
                
        elif action == 2: # FULL CLOSE
            realized = total_pnl - self.slippage
            reward = realized * self.current_position_size * 2.0
            self.current_position_size = 0.0
            done = True
            
        # Drawdown Penalty (asymmetric risk shaping)
        if total_pnl < -0.015: # -1.5% hard stop
            reward -= 0.05
            done = True
            self.current_position_size = 0.0

        next_state = self._get_state()
        info = {
            "pnl": total_pnl,
            "action": action,
            "size": self.current_position_size
        }
        
        return next_state, reward, done, info


# ------------------------------------------------------------------------------
# 4. BACKTESTING HOOKS
# ------------------------------------------------------------------------------
def evaluate_rl_agent(agent: DynamicTradeManagerDQN, env: DynamicTradeEnvironment, num_trades: int = 100) -> Dict:
    """Runs out-of-sample evaluation on the RL agent."""
    cumulative_pnl = 0.0
    wins = 0
    drawdowns = []
    
    for i in range(num_trades):
        # Pick a random starting point
        start_idx = random.randint(0, len(env.price_data) - 100)
        state = env.reset(start_step=start_idx)
        done = False
        peak_pnl = 0.0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            
            if info['pnl'] > peak_pnl:
                peak_pnl = info['pnl']
            
            if done:
                cumulative_pnl += info['pnl']
                if info['pnl'] > 0:
                    wins += 1
                drawdowns.append(peak_pnl - info['pnl']) # PnL left on table
                
    win_rate = wins / num_trades
    avg_dd = np.mean(drawdowns)
    
    return {
        "Total Return (%)": round(cumulative_pnl * 100, 2),
        "RL Win Rate (%)": round(win_rate * 100, 1),
        "Avg Giveback/DD (%)": round(avg_dd * 100, 2),
        "RL Exit Alpha": "STRONG" if cumulative_pnl > 0.05 else "NEUTRAL"
    }

if __name__ == "__main__":
    logger.info("Deep Q-Network Trade Management Engine Initialized.")
