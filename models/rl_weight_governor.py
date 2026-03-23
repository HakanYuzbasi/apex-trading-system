"""
models/rl_weight_governor.py

A native Q-Learning Reinforcement Learning agent that continuously optimizes
the constituent ML weights across market regimes, instead of using static matrices.
"""

import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Base static bounds representing the default (Action 0) state
BASE_WEIGHTS = {
    'momentum': -0.40,
    'mean_reversion': 0.20,
    'trend': -0.40,
    'volatility': 0.05,
    'breakout': 0.05,
    'senate_sentiment': 0.25,
    'smart_money_flow': 0.15,
    'news_sentiment': 0.15,
    'ml_rf': 0.30,
    'ml_gb': 0.30,
    'ml_xgb': 0.00,
    'ml_lgb': 0.00
}

# Define the Discrete Action Space (which weight matrix to apply)
# 0: Standard Baseline (Balanced)
# 1: Aggressive Institutional Alpha (Senate + Flow + News dominate)
# 2: Defensive Mean Reversion (Reversion dominates, Momentum/Trend heavily inverted)
# 3: Pure Machine Learning (RF / GB / XGB take over)

ACTIONS = {
    0: BASE_WEIGHTS.copy(),
    1: {**BASE_WEIGHTS, 'senate_sentiment': 0.50, 'smart_money_flow': 0.40, 'news_sentiment': 0.30, 'momentum': -0.10, 'trend': -0.10},
    2: {**BASE_WEIGHTS, 'mean_reversion': 0.60, 'momentum': -0.60, 'trend': -0.60, 'breakout': -0.30, 'ml_rf': 0.10, 'ml_gb': 0.10},
    3: {**BASE_WEIGHTS, 'ml_rf': 0.50, 'ml_gb': 0.50, 'ml_xgb': 0.25, 'senate_sentiment': 0.05, 'smart_money_flow': 0.05, 'news_sentiment': 0.05}
}

class RLWeightGovernor:
    """
    Q-Learning Agent observing the environment state (Regime + Volatility)
    to select the optimal weight matrix dynamically.
    """
    def __init__(self, model_dir: str = "models/saved"):
        self.q_table = {}  # state -> {action: q_value}
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.2 # Exploration rate (Epsilon-Greedy)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.q_table_path = self.model_dir / "q_table_governor.json"
        
        self._load_q_table()

    def _load_q_table(self):
        if self.q_table_path.exists():
            try:
                with open(self.q_table_path, "r") as f:
                    self.q_table = json.load(f)
                logger.info("✅ RL Governor Q-Table Loaded Successfully.")
            except Exception as e:
                logger.warning(f"Failed to load Q-Table: {e}. Starting fresh.")
                
    def _save_q_table(self):
        try:
            with open(self.q_table_path, "w") as f:
                json.dump(self.q_table, f)
        except Exception as e:
            logger.warning(f"Failed to save RL Q-Table: {e}")

    def _get_state_key(self, regime: str, is_high_volatility: bool) -> str:
        """Discretize the current environment into a finite Markov State."""
        vol_state = "HIGH_VOL" if is_high_volatility else "NORM_VOL"
        return f"{str(regime).upper()}_{vol_state}"

    def _ensure_state(self, state: str):
        if state not in self.q_table:
            # Initialize with Neutral expectations (0.0) for all 4 actions
            self.q_table[state] = {str(a): 0.0 for a in ACTIONS.keys()}

    def get_optimal_weights(self, regime: str, is_high_volatility: bool = False, training: bool = True) -> Dict[str, float]:
        """
        Choose an action via Epsilon-Greedy policy to map current regime to a weight matrix.
        Returns the specific Dict of constituent weights to use.
        """
        state = self._get_state_key(regime, is_high_volatility)
        self._ensure_state(state)

        # Exploration vs Exploitation
        if training and np.random.uniform(0, 1) < self.epsilon:
            # Explore
            action = str(np.random.choice(list(ACTIONS.keys())))
        else:
            # Exploit (Choose action with highest Q-Value for this state)
            action = max(self.q_table[state].items(), key=lambda x: x[1])[0]

        weight_map = ACTIONS[int(action)].copy()
        
        # USER MANDATE: Knife-Catching Override for Mega-Bull Exhaustion
        if regime.upper() == "STRONG_BULL":
            # When the 50/200 MA gap is > 15%, the market is fundamentally overextended.
            # Force massive priority onto Micro-Pullbacks and Options Flow Gamma limits
            # to structurally catch the knife when the gap collapses!
            weight_map['mean_reversion'] = max(weight_map.get('mean_reversion', 0.0), 0.45)
            weight_map['smart_money_flow'] = max(weight_map.get('smart_money_flow', 0.0), 0.35)
            
            # Dampen trend following to avoid buying the extreme top
            weight_map['trend'] = min(weight_map.get('trend', 0.0), -0.20)
            weight_map['momentum'] = min(weight_map.get('momentum', 0.0), -0.10)

        weight_map['__action__'] = float(action)
        return weight_map

    def update_q_value(self, regime: str, is_high_volatility: bool, action_taken: int, reward: float, next_regime: str = None):
        """
        Bellman Equation update based on the rolling reward (trade PnL delta).
        """
        state = self._get_state_key(regime, is_high_volatility)
        self._ensure_state(state)
        
        action_str = str(action_taken)
        current_q = self.q_table[state][action_str]
        
        # If we calculate a next state (for future reward discounting)
        max_future_q = 0.0
        if next_regime:
            next_state = self._get_state_key(next_regime, is_high_volatility)
            self._ensure_state(next_state)
            max_future_q = max(self.q_table[next_state].values())

        # Q(s, a) = Q(s, a) + α * [R(s, a) + γ * max Q(s', a') - Q(s, a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action_str] = float(new_q)
        self._total_updates = getattr(self, '_total_updates', 0) + 1

        # Epsilon decay: reduce exploration as Q-table matures
        # Decays from 0.20 → 0.05 over ~200 updates
        min_eps = 0.05
        decay = 0.995
        self.epsilon = max(min_eps, self.epsilon * decay)

        self._save_q_table()

    def get_best_action(self, regime: str, is_high_volatility: bool = False) -> int:
        """Return the greedy best action (no exploration) — for production confidence overlay."""
        state = self._get_state_key(regime, is_high_volatility)
        self._ensure_state(state)
        return int(max(self.q_table[state].items(), key=lambda x: float(x[1]))[0])

    def get_action_confidence(self, regime: str, is_high_volatility: bool = False) -> float:
        """
        Return a confidence multiplier [0.85, 1.10] based on Q-value spread.
        High spread → the governor is confident about one action → boost confidence.
        Low spread → uncertain → stay neutral (1.0).
        """
        state = self._get_state_key(regime, is_high_volatility)
        self._ensure_state(state)
        q_vals = list(self.q_table[state].values())
        if not q_vals:
            return 1.0
        q_max = max(q_vals)
        q_min = min(q_vals)
        spread = q_max - q_min
        # Spread 0 = uninformed = 1.0; spread 0.10+ = confident = 1.10
        mult = 1.0 + min(0.10, spread * 0.5)
        # Also penalise if best action is negative Q (regime is generically unprofitable)
        if q_max < 0:
            mult = max(0.85, 1.0 + q_max * 0.5)
        return round(mult, 4)

    def get_governor_report(self) -> dict:
        """Full Q-table diagnostic for the walk-forward dashboard."""
        report = {}
        for state, actions in self.q_table.items():
            best = max(actions.items(), key=lambda x: float(x[1]))
            report[state] = {
                "best_action": int(best[0]),
                "best_q": round(float(best[1]), 4),
                "q_spread": round(max(actions.values()) - min(actions.values()), 4),
                "all_q": {k: round(float(v), 4) for k, v in actions.items()},
            }
        return {
            "states": report,
            "epsilon": round(self.epsilon, 4),
            "total_updates": getattr(self, '_total_updates', 0),
        }


# Global Singleton Integration Ready
_governor = RLWeightGovernor()

def get_rl_weights(regime: str, is_high_volatility: bool = False) -> Dict[str, float]:
    return _governor.get_optimal_weights(regime, is_high_volatility)

def feedback_rl_reward(regime: str, is_high_volatility: bool, action_taken: int, reward: float):
    _governor.update_q_value(regime, is_high_volatility, action_taken, reward)

def get_rl_confidence_mult(regime: str, is_high_volatility: bool = False) -> float:
    """Production confidence multiplier from Q-table certainty. Call at signal blend time."""
    return _governor.get_action_confidence(regime, is_high_volatility)

def get_rl_governor_report() -> dict:
    return _governor.get_governor_report()
