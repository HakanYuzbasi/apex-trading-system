import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("monte_carlo")

def run_monte_carlo(n_sims: int = 10000, confidence_level: float = 0.95):
    """
    Run Monte Carlo simulations to find Value at Risk (VaR).
    If VaR > 10% of equity, recommend reducing leverage.
    """
    # 1. Load historical daily returns
    # In a real system, we'd pull from TimescaleDB or trades.csv
    trades_path = PROJECT_ROOT / "data" / "trades.csv"
    
    if not trades_path.exists():
        logger.warning("No historical trade data found at %s. Using synthetic data for stress test.", trades_path)
        # Synthetic daily returns for a 1-year period
        daily_returns = np.random.normal(0.001, 0.015, 252)
    else:
        df = pd.read_csv(trades_path)
        if df.empty or 'pnl' not in df.columns:
            daily_returns = np.random.normal(0.001, 0.02, 252)
        else:
            # Aggregate pnl by day (mocked timestamp parsing)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_pnl = df.groupby('date')['pnl'].sum()
            # For returns, we need equity. Let's assume $100k starting.
            daily_returns = daily_pnl / 100000.0
            
    # 2. Add "Bad Days" and "Gap Downs" to historical distribution
    # Stressing the distribution with extreme tails
    bad_days = np.array([-0.04, -0.05, -0.07, -0.10])
    stressed_returns = np.concatenate([daily_returns, bad_days])
    
    # 3. Simulations
    # We simulate a 20-day path (typical risk horizon)
    horizon = 20
    sim_paths = np.random.choice(stressed_returns, size=(n_sims, horizon))
    
    # Calculate cumulative returns over the horizon for each path
    path_returns = np.prod(1 + sim_paths, axis=1) - 1
    
    # 4. Calculate VaR
    var_95 = -np.percentile(path_returns, (1 - confidence_level) * 100)
    
    # 5. Calculate Survival Probability (Prob that drawdown < 20%)
    survival_prob = np.mean(path_returns > -0.20)
    
    logger.info("Monte Carlo Results: VaR_95=%.2f%%, Survival_Prob=%.2f%%", var_95 * 100, survival_prob * 100)
    
    # 6. Feedback Loop: Auto-adjust leverage
    # If VaR > 10%, we scale leverage down linearly (e.g. if VaR is 20%, leverage = 0.5)
    leverage_limit = 1.0
    if var_95 > 0.10:
        leverage_limit = 0.10 / var_95
        logger.warning("VaR exceeds 10%%! Recommending leverage limit: %.2f", leverage_limit)
        
    # Write adjustment to a shared state file
    adjustment_file = PROJECT_ROOT / "run_state" / "risk_adjustments.json"
    adjustment_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(adjustment_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "var_95": var_95,
            "survival_probability": survival_prob,
            "recommended_leverage": max(0.1, round(leverage_limit, 2))
        }, f, indent=2)
        
    return var_95, survival_prob, leverage_limit

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_monte_carlo()
