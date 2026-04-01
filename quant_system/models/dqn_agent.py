"""
quant_system/models/dqn_agent.py
================================================================================
Continuous Execution Agent (PRODUCTION GRADE)
================================================================================
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict

class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class ContinuousExecutionAgent:
    def __init__(self, state_dim: int = 15, lr_actor: float = 1e-4, lr_critic: float = 1e-3, gamma: float = 0.99, tau: float=0.005):
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        
        self.memory = deque(maxlen=50000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim).to(self.device)
        self.actor_target = Actor(state_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim).to(self.device)
        self.critic_target = Critic(state_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state: np.ndarray, raw_context: np.ndarray, evaluate: bool = False) -> Dict:
        """
        Receives BOTH strictly Normalized State (for inference) and Raw Context (for physical boundaries).
        """
        meta_prob = raw_context[4]
        confidence = raw_context[5]
        regime = raw_context[13] 
        previous_action = raw_context[14]
        
        epsilon = 0.04 + 0.08 * (1.0 - confidence)
        blocked_by_ntz = bool(abs(meta_prob - 0.5) < epsilon)
        
        edge_strength = (abs(meta_prob - 0.5) * 2.0) ** 1.2
        edge_strength = np.clip(edge_strength, 0.0, 1.0)

        with torch.no_grad():
            self.actor.eval()
            state_ten = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            raw_action = self.actor(state_ten).item()
            self.actor.train()
            
        if not evaluate:
            raw_action += np.random.normal(0, 0.1)
            raw_action = np.clip(raw_action, 0.0, 1.0)
            
        current_action = 0.5 * previous_action + 0.5 * raw_action
        inertia_adjusted_action = current_action
        current_action = current_action * edge_strength * confidence
        
        boosted = False
        if confidence > 0.85 and edge_strength > 0.7:
            current_action *= 1.2
            boosted = True
            
        current_action *= 1.5
        global_scaling_applied = 1.5
        
        if blocked_by_ntz:
            current_action = 0.0
            
        conviction_floor_applied = False
        if not blocked_by_ntz and confidence > 0.85 and edge_strength > 0.6:
            if current_action < 0.05:
                current_action = max(current_action, 0.05)
                conviction_floor_applied = True
                
        action_before_threshold = current_action
                
        filtered_micro_trade = False
        if current_action < 0.01:
            current_action = 0.0
            filtered_micro_trade = True
            
        action_after_threshold = current_action
            
        max_action_map = {1: 1.0, 2: 0.6, 3: 0.2}
        max_act = max_action_map.get(int(regime), 1.0)
        
        final_action = min(current_action, max_act)
        
        # UPGRADE 6: Action Sanity Check Layer
        if int(regime) == 3 and final_action > 0.2:
            final_action *= 0.5
            
        final_action = max(0.0, min(final_action, 1.0))
            
        return {
            'action': float(final_action),
            'raw_action': float(raw_action),
            'action_after_inertia': float(inertia_adjusted_action),
            'epsilon': float(epsilon),
            'edge_strength': float(edge_strength),
            'max_action': float(max_act),
            'blocked_by_ntz': blocked_by_ntz,
            'boosted_action_flag': boosted,
            'confidence': float(confidence),
            'filtered_micro_trade': filtered_micro_trade,
            'action_before_threshold': float(action_before_threshold),
            'action_after_threshold': float(action_after_threshold),
            'conviction_floor_applied': conviction_floor_applied,
            'global_scaling_applied': float(global_scaling_applied)
        }

    # Standard Memory loops
    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self, batch_size: int = 64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.FloatTensor(np.array(actions)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_q = self.critic_target(next_state_batch, next_actions)
            expected_q = reward_batch + self.gamma * target_q * (1 - done_batch)
        current_q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q, expected_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
