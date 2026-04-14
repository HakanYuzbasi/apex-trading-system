#!/usr/bin/env python3
"""
scripts/train_meta_labeler.py
================================================================================
APEX TRADING - ML Supervisor Training Loop

Ingests performance attribution data, extracts ML features, and trains the 
LightGBM Meta-Labeler to predict the probability of trade success.
"""

import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from quant_system.ml.meta_labeler import MetaLabeler
from config import ApexConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainMetaLabeler")

def generate_mock_data(n=200):
    """Generate mock attribution data for cold-start testing."""
    data = []
    for i in range(n):
        # Features
        k_resid = np.random.normal(0, 0.05)
        b_prob = np.random.uniform(0.3, 0.7)
        vix = np.random.uniform(15, 35)
        conc = np.random.uniform(0.1, 0.4)
        
        # Outcome logic: success depends on low residual and moderate VIX
        success_prob = 0.5 + 0.2 * (1.0 - abs(k_resid)*10) - 0.1 * (vix/40)
        success_prob = np.clip(success_prob, 0.1, 0.9)
        outcome = 1 if np.random.random() < success_prob else 0
        
        data.append({
            "kalman_residual": k_resid,
            "bayesian_prob": b_prob,
            "vix_level": vix,
            "sector_concentration": conc,
            "net_pnl": 100.0 if outcome == 1 else -100.0
        })
    return data

def main():
    logger.info("🚀 Starting Meta-Labeler Training Loop")
    
    # 1. Locate Data
    data_dir = Path("run_state")
    attr_file = data_dir / "performance_attribution.json"
    
    closed_trades = []
    
    if attr_file.exists():
        try:
            with open(attr_file, 'r') as f:
                state = json.load(f)
                closed_trades = state.get("closed_trades", [])
            logger.info(f"📂 Loaded {len(closed_trades)} closed trades from attribution file.")
        except Exception as e:
            logger.error(f"❌ Failed to load attribution file: {e}")
    else:
        logger.warning("⚠️ No attribution file found. Generating mock data for cold-start.")
        closed_trades = generate_mock_data()

    if not closed_trades:
        logger.error("❌ No data available for training. Exiting.")
        return

    # 2. Extract Features and Labels
    df = pd.DataFrame(closed_trades)
    
    # Ensure columns exist
    required_cols = ["kalman_residual", "bayesian_prob", "vix_level", "sector_concentration", "net_pnl"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 # Default if missing
            
    # X = features, y = label (Success = Net PnL > 0)
    X = df[["kalman_residual", "bayesian_prob", "vix_level", "sector_concentration"]]
    y = (df["net_pnl"] > 0).astype(int)
    
    logger.info(f"📊 Training on {len(df)} samples. Win Rate: {y.mean():.1%}")

    # 3. Train Model
    labeler = MetaLabeler()
    labeler.train(X, y)
    
    model_path = Path("run_state/meta_labeler.lgb")
    logger.info(f"✅ Training complete. Model saved to {model_path}")
    
    # Print feature importance as diagnostic
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        importances = model.feature_importance(importance_type='gain')
        names = ["kalman_residual", "bayesian_prob", "vix_level", "sector_concentration"]
        logger.info("📈 Feature Importance (Gain):")
        for n, i in zip(names, importances):
            logger.info(f"  - {n}: {i:.2f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
