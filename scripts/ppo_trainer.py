#!/usr/bin/env python3
import os
import sys

# Ensure the root of the repository is in the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.ml.ppo_trainer import ExecutionTrainer

def main():
    # Instantiate the trainer with its default model save path
    trainer = ExecutionTrainer(model_save_path="run_state/models/ppo_execution_v1")
    
    # Train for 1,000,000 timesteps to fulfill the "millions of ticks" objective.
    # Note: L2ExecutionEnv generates training ticks locally.
    trainer.train(timesteps=1000000)

if __name__ == "__main__":
    main()
