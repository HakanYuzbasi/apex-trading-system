import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import torch


class L2ExecutionEnv(gym.Env):
    """
    Gymnasium environment designed for L2 microstructural execution.
    The agent learns to sweep, wait, or passively join limits across continuous ticks.
    """
    def __init__(self, historical_ticks):
        super(L2ExecutionEnv, self).__init__()
        if historical_ticks is None:
            raise ValueError("L2ExecutionEnv must be initialized with live or historical ticks in production.")
        
        self.historical_ticks = historical_ticks
        
        # State Space: [vPIN, OBI, Spread, Iceberg, Inventory Remaining, Time Remaining]
        # Normalized values between -1.0 and 1.0 (or 0.0 to 1.0)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action Space: 4 options
        # 0: Wait (Hold)
        # 1: Passive Maker (Join bid/ask)
        # 2: Penny-Jump (+0.01 beyond limit)
        # 3: Market Sweep (Take liquidity, pay fee)
        self.action_space = spaces.Discrete(4)

        self.current_step = 0
        self.max_steps = len(self.historical_ticks)
        
        # Episode state
        self.inventory_remaining = 1.0
        self.time_limit_ms = 60000 # 60 seconds
        self.start_time = 0.0
        
        self.base_fee = 0.002
        self.maker_rebate = 0.0005


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pure Live: No re-generation on reset
        self.current_step = 0
        self.inventory_remaining = 1.0
        self.time_spent = 0.0
        
        # Calculate arrival price to judge slippage later
        self.arrival_mid = self.historical_ticks[0]["mid_price"]
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        tick = self.historical_ticks[self.current_step]
        time_rem = max(0.0, 1.0 - (self.time_spent / self.time_limit_ms))
        
        obs = np.array([
            tick["vpin"], 
            tick["obi"], 
            tick["spread"], 
            tick["iceberg"], 
            self.inventory_remaining, 
            time_rem
        ], dtype=np.float32)
        return obs

    def step(self, action):
        tick = self.historical_ticks[self.current_step]
        reward = 0.0
        done = False
        
        time_penalty = 0.0001 # Small penalty per tick to encourage execution
        self.time_spent += 100 # Assume every tick advances clock by 100ms
        
        vpin = tick["vpin"]
        current_mid = tick["mid_price"]
        
        fill_occurred = False
        edge_captured = 0.0
        
        if action == 0:
            # Wait
            reward -= time_penalty
            if vpin > 0.8:
                reward -= 0.001 # Penalty for sitting exposed to toxic flow
                
        elif action == 1:
            # Passive Maker
            # Only filled if probability holds (e.g. 70% chance if not toxic)
            fill_prob = 0.7 if vpin < 0.6 else 0.1
            if np.random.random() < fill_prob:
                fill_occurred = True
                edge_captured = (self.arrival_mid - current_mid) / self.arrival_mid
                reward += self.maker_rebate # Earned rebate
            else:
                reward -= time_penalty # Missed the fill
                
        elif action == 2:
            # Penny-Jump
            # Always fills if there's an iceberg, highly probable otherwise
            fill_prob = 0.95 if tick["iceberg"] == 1.0 else 0.8
            if np.random.random() < fill_prob:
                fill_occurred = True
                edge_captured = (self.arrival_mid - current_mid) / self.arrival_mid
                # Paid no fee, paid no rebate, essentially just jumped spread
            else:
                reward -= time_penalty
                
        elif action == 3:
            # Market Sweep
            fill_occurred = True
            edge_captured = (self.arrival_mid - current_mid) / self.arrival_mid
            reward -= self.base_fee # Paid taker fee
            
            # Massive reward chunk if we avoided adverse selection by sweeping right before a crash
            if vpin > 0.8:
                reward += 0.005 # Specifically reward the network for bailing on toxic flow
            
        if fill_occurred:
            self.inventory_remaining = 0.0
            done = True
            # Weight edge heavy
            reward += (edge_captured * 100.0) 
            
        self.current_step += 1
        
        if self.current_step >= self.max_steps - 1 or self.time_spent >= self.time_limit_ms:
            done = True
            if self.inventory_remaining > 0:
                reward -= 0.01 # Heavy penalty for failing to execute inventory
                
        return self._get_obs(), reward, done, False, {}
