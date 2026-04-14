import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from quant_system.execution.drl_env import L2ExecutionEnv

logger = logging.getLogger("ppo_trainer")
logging.basicConfig(level=logging.INFO)

import random

class MockTickGenerator:
    """Helper to generate simulation ticks for PPO training baseline."""
    @staticmethod
    def generate(n_ticks=1000):
        ticks = []
        base_price = 50000.0
        for i in range(n_ticks):
            base_price += random.gauss(0, 5)
            ticks.append({
                "vpin": min(1.0, max(0.0, random.gauss(0.5, 0.2))),
                "obi": min(1.0, max(-1.0, random.gauss(0.0, 0.4))),
                "spread": min(1.0, max(0.0, random.gauss(0.1, 0.05))),
                "iceberg": random.choice([0.0, 1.0]),
                "mid_price": base_price
            })
        return ticks

class ExecutionTrainer:
    """
    Offline DRL Training hook for execution optimization.
    Takes the L2ExecutionEnv and trains a PPO network to map (6-D state -> 4-action).
    """
    def __init__(self, model_save_path: str = "run_state/models/ppo_execution_v1"):
        self.model_save_path = model_save_path
        
        # Pure Live Guard: Generate training ticks locally within the trainer
        # to ensure drl_env.py remains clean for production WebSocket feeds.
        training_ticks = MockTickGenerator.generate(n_ticks=2000)
        
        # Instantiate environment and wrap for stable/baselines vectorization
        self.env = Monitor(L2ExecutionEnv(historical_ticks=training_ticks))
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # We increase entropy_coef massively higher than default to force the agent to explore 
        # penny-jumping and market sweeping across diverse conditions instead of converging immediately.
        self.model = PPO(
            "MlpPolicy", 
            self.vec_env, 
            verbose=1,
            ent_coef=0.05, 
            learning_rate=3e-4,
            batch_size=64,
            n_steps=1024,
            tensorboard_log="./run_state/tensorboard/"
        )

    def train(self, timesteps: int = 1000000):
        logger.info(f"Starting PPO Execution training for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        self.model.save(self.model_save_path)
        logger.info(f"Training complete. Weights saved to {self.model_save_path}.zip")

if __name__ == "__main__":
    trainer = ExecutionTrainer()
    # In a real environment we would train for 1M+ steps. 
    # Increased for production baseline training
    trainer.train(timesteps=50000)
