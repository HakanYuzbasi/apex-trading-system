import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("adversarial_mc")

class MarketGenerator(nn.Module):
    """RNN-based generator of return sequences."""
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1, seq_len=20):
        super(MarketGenerator, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h)
        return out

def run_adversarial_mc(pairs: list[dict[str, Any]], n_sims: int = 100, horizon: int = 20):
    """
    Search for adversarial return sequences that crash the strategy for each pair.
    """
    pair_results = {}
    
    for pair_cfg in pairs:
        pair_label = f"{pair_cfg['instrument_a']}/{pair_cfg['instrument_b']}"
        logger.info(f"Running Adversarial MC for {pair_label}...")
        
        generator = MarketGenerator(input_dim=1, seq_len=horizon)
        optimizer = optim.Adam(generator.parameters(), lr=0.01)
        
        max_dd_observed = 0.0
        breaking_points = []
        
        for epoch in range(100):
            z = torch.randn(1, horizon, 1) 
            output_rets = generator(z).squeeze()
            
            # Simple cumulative return simulation for the pair
            cum_ret = torch.cumprod(1 + output_rets / 100.0, dim=0) - 1
            max_dd = -torch.min(cum_ret)
            
            vol_penalty = torch.pow(torch.std(output_rets) - 1.25, 2)
            mean_penalty = torch.pow(torch.mean(output_rets), 2)
            
            loss = -max_dd + 2.0 * vol_penalty + 5.0 * mean_penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            max_dd_observed = max(max_dd_observed, max_dd.item())
            if max_dd > 0.15:
                breaking_points.append(output_rets.detach().numpy().tolist())

        pair_results[pair_label] = {
            "max_adversarial_drawdown": max_dd_observed,
            "n_breaking_points": len(breaking_points)
        }

    # Save results
    results_file = PROJECT_ROOT / "run_state" / "adversarial_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pair_results": pair_results
        }, f, indent=2)
        
    logger.info("Adversarial MC complete for all pairs.")
    return pair_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from scripts.run_global_harness_v3 import DEFAULT_WINNER_DICTIONARY
    all_pairs = DEFAULT_WINNER_DICTIONARY["crypto"] + DEFAULT_WINNER_DICTIONARY["equity"]
    run_adversarial_mc(all_pairs)
