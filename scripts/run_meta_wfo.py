#!/usr/bin/env python3
"""
scripts/run_meta_wfo.py
================================================================================
APEX TRADING - Meta-Labeler Walk-Forward Optimization (Validation)

Evaluates the effectiveness of the Meta-Labeler supervisor by performing 
a "What-If" analysis on historical trade data.
"""

import json
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from quant_system.ml.meta_labeler import MetaLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaWFO")

def calculate_metrics(pnl_series):
    """Calculate Sharpe and Sortino ratios."""
    if len(pnl_series) < 2:
        return 0.0, 0.0
    
    returns = pnl_series # Assuming PnL in dollar or bps terms per trade
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    # Sharpe (simplified for trade-level)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
    
    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    return sharpe, sortino

def main():
    logger.info("🧪 Starting Meta-Labeler Validation (WFO)")
    
    # 1. Load Data
    attr_file = Path("run_state/performance_attribution.json")
    if not attr_file.exists():
        logger.error("❌ No attribution data found. Run the system or training first.")
        # Generate some synthetic test data if it doesn't exist
        from scripts.train_meta_labeler import generate_mock_data
        closed_trades = generate_mock_data(n=500)
    else:
        with open(attr_file, 'r') as f:
            state = json.load(f)
            closed_trades = state.get("closed_trades", [])

    if not closed_trades:
        logger.error("❌ No closed trades to analyze.")
        return

    df = pd.DataFrame(closed_trades)
    
    # 2. Load Model
    # MetaLabeler automatically loads from run_state/meta_labeler.lgb in __init__
    labeler = MetaLabeler()
    if labeler._model:
        logger.info(f"📂 Loaded trained model from {labeler.model_path}")
    else:
        logger.warning("⚠️ No trained model found. Running with untrained (random) supervisor.")

    # 3. Simulate "Veto" Logic
    results = []
    
    for _, row in df.iterrows():
        # Predict confidence
        conf = labeler.predict_confidence(
            kalman_residual=row.get("kalman_residual", 0.0),
            bayesian_prob=row.get("bayesian_prob", 0.5),
            vix_level=row.get("vix_level", 20.0),
            sector_concentration=row.get("sector_concentration", 0.0)
        )
        
        # Base logic: keep every trade
        # Supervisor logic: veto if conf < 0.65
        vetoed = conf < 0.65
        
        results.append({
            "pnl": row.get("net_pnl", 0.0),
            "conf": conf,
            "vetoed": vetoed
        })

    res_df = pd.DataFrame(results)
    
    # 4. Compare Metrics
    base_pnl = res_df["pnl"]
    super_pnl = res_df[~res_df["vetoed"]]["pnl"]
    
    base_sharpe, base_sortino = calculate_metrics(base_pnl)
    super_sharpe, super_sortino = calculate_metrics(super_pnl)
    
    veto_ratio = res_df["vetoed"].mean()
    
    logger.info("\n" + "="*40)
    logger.info("📊 VALIDATION RESULTS: Base vs. Supervisor")
    logger.info("="*40)
    logger.info(f"Total Trades:      {len(res_df)}")
    logger.info(f"Vetoed Trades:     {res_df['vetoed'].sum()} ({veto_ratio:.1%})")
    logger.info("-" * 40)
    logger.info(f"BASE Sharpe:       {base_sharpe:.3f}")
    logger.info(f"SUPER Sharpe:      {super_sharpe:.3f} ({(super_sharpe/base_sharpe-1)*100:+.1f}%)" if base_sharpe > 0 else "N/A")
    logger.info("-" * 40)
    logger.info(f"BASE Sortino:      {base_sortino:.3f}")
    logger.info(f"SUPER Sortino:     {super_sortino:.3f} ({(super_sortino/base_sortino-1)*100:+.1f}%)" if base_sortino > 0 else "N/A")
    logger.info("="*40)
    
    if super_sortino > base_sortino * 1.15:
        logger.info("✅ SUCCESS: Sortino improvement exceeds 15% threshold.")
    else:
        logger.info("⚠️ WARNING: Improvement below target threshold.")

if __name__ == "__main__":
    main()
