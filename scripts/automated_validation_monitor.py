"""
scripts/automated_validation_monitor.py
================================================================================
PHASE 5 AUTOMATED VALIDATION MONITOR
================================================================================

Programmatically asserts the structural integrity and operational health of the 
5-Phase Quantitative Trading Architecture.

Validates:
A. Pre-Execution Checks (Dependencies, Files, Data)
B. Output Generation (CVaR, IC, PnL, HRP Metrics)
C. Automated Post-Execution Data Assertions
D. Live Hook Stress-Tests (Emergency Halts & IC Drift Retraining)
"""

import os
import sys
import importlib
import pandas as pd
import numpy as np
import logging
from typing import List, Dict

# Explicitly add the root project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='\033[96m[MONITOR] %(asctime)s - %(message)s\033[0m')
logger = logging.getLogger("ValidationMonitor")


def check_pre_execution() -> bool:
    """A. Validates environment and critical infrastructure."""
    logger.info("--- A. PRE-EXECUTION CHECKS ---")
    
    # Check Python Packages
    required_pkgs = ['torch', 'lightgbm', 'xgboost', 'scipy', 'pandas']
    missing_pkgs = []
    for pkg in required_pkgs:
        try:
            importlib.import_module(pkg)
            logger.info(f" [PASS] Package '{pkg}' is available.")
        except ImportError:
            missing_pkgs.append(pkg)
            
    if missing_pkgs:
        logger.error(f" [FAIL] Missing critical packages: {missing_pkgs}")
        return False
        
    # Check Model Stubs & Paths
    required_files = [
        'models/hybrid_meta_learner.py',
        'models/rl_trade_manager.py',
        'models/portfolio_optimizer.py',
        'models/governance_engine.py',
        'scripts/run_full_validation.py'
    ]
    
    for f in required_files:
        path = os.path.join(os.path.dirname(__file__), '..', f)
        if os.path.exists(path):
            logger.info(f" [PASS] Model endpoint localized: {f}")
        else:
            logger.error(f" [FAIL] Missing structural requirement: {f}")
            return False
            
    # Check Logs Dir
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if os.access(log_dir, os.W_OK):
        logger.info(f" [PASS] Directory {log_dir} exists and is writable.")
    else:
        logger.error(f" [FAIL] Logging directory is missing or read-only.")
        return False
        
    return True

    
def check_validation_outputs() -> bool:
    """C & D. Validates the existence and metric boundaries of the CP-CV Reports."""
    logger.info("\n--- C. & D. OUTPUT VALIDATION & POST-ANALYSIS ---")
    
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    
    metrics_path = os.path.join(log_dir, 'purged_cv_metrics.csv')
    audit_path = os.path.join(log_dir, 'institutional_audit.csv')
    plot_path = os.path.join(log_dir, 'phase5_governance_report.png')
    
    all_passed = True
    
    # 1. Image Plots
    if os.path.exists(plot_path):
        logger.info(" [PASS] Visual Analytics Artifact: phase5_governance_report.png generated.")
    else:
        logger.error(" [FAIL] Visual analytics artifact missing (Has run_full_validation.py completed?).")
        all_passed = False
        
    # 2. Institutional Audit
    if os.path.exists(audit_path):
        df_audit = pd.read_csv(audit_path)
        logger.info(f" [PASS] Institutional Audit loaded. Detected {len(df_audit)} Versioned Models.")
        if 'hash' in df_audit.columns and df_audit['hash'].nunique() == len(df_audit):
            logger.info(" [PASS] Cryptographic SHA-256 Hashes verified for uniqueness.")
        else:
            logger.error(" [FAIL] Cryptographic Hash collison or missing column in Audit CSV.")
            all_passed = False
    else:
        logger.error(" [FAIL] Institutional Audit CSV missing.")
        all_passed = False
        
    # 3. Purged CV Metrics
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        req_cols = ['Fold', 'Information_Coefficient', 'RL_Test_PnL', 'HRP_Capital_Deployed', 'Tail_Risk_CVaR_5%']
        
        # Check integrity
        if all(c in df_metrics.columns for c in req_cols):
            logger.info(" [PASS] Combinatorial Metrics schema validated.")
            
            # Post-Execution Analysis (Drawdowns & Overfitting)
            ic_mean = df_metrics['Information_Coefficient'].mean()
            ic_std = df_metrics['Information_Coefficient'].std()
            cvar_mean = df_metrics['Tail_Risk_CVaR_5%'].mean()
            
            logger.info(f"   -> Average Information Coefficient: {ic_mean:.4f}")
            logger.info(f"   -> Fold Varinace (Overfitting Risk): {ic_std:.4f}")
            logger.info(f"   -> Average Absolute CVaR: {cvar_mean:.2%}")
            
            if ic_std > 0.05:
                # Warning for wide variation across purged folds
                logger.warning(" [DIAGNOSTIC] High IC Variance across Folds detected. Possible temporal overfitting/Regime Dependency.")
        else:
            logger.error(" [FAIL] Missing required metrics columns in Purged CV output.")
            all_passed = False
    else:
        logger.warning(" [WARN] CV Metrics currently missing. Ensure run_full_validation.py is executed.")
        all_passed = False
        
    return all_passed


def test_live_simulation_hooks():
    """E. Stresses the Live Hooks by dynamically calling Python methods."""
    logger.info("\n--- E. LIVE / FORWARD SIMULATION HOOK STRESS-TESTS ---")
    
    try:
        from models.governance_engine import GovernanceMonitor
        from models.portfolio_optimizer import DynamicPortfolioManager
        
        # 1. Test Drift & Drawdown Governance
        gov = GovernanceMonitor(ic_threshold=0.02, dd_threshold=-0.15)
        
        # Artificial Drawdown feeding
        logger.info(" [TEST] Injecting artificial -16% PnL into evaluate_live_governance_tick()...")
        alerts = gov.evaluate_live_governance_tick(-0.16, np.zeros(0), np.zeros(0))
        
        if alerts.get("emergency_halt", False):
            logger.info(" [PASS] Governance Engine successfully emitted emergency_halt=True at -16% PnL.")
        else:
            logger.error(" [FAIL] Governance Engine FAILED to halt at -16% PnL.")
            
        # 2. Test Live Portfolio HRP
        mgr = DynamicPortfolioManager(max_capital_usd=100000)
        returns = pd.DataFrame(np.random.normal(0,0.01,(100,2)), columns=['BTC', 'ETH'])
        meta = {'BTC': 0.45, 'ETH': 0.85} # BTC Confidence < 50%
        actions = {'BTC': 0, 'ETH': 1}
        liq = {'BTC': {'adv': 1e7, 'spread_bps': 2.0}, 'ETH': {'adv': 1e7, 'spread_bps': 2.0}}
        
        logger.info(" [TEST] Requesting live portfolio allocation via HRP with sub-0.5 edge scalars...")
        allocs = mgr.evaluate_live_portfolio_tick(returns, meta, actions, liq)
        
        if allocs.get('BTC', 0.0) == 0.0:
            logger.info(" [PASS] HRP Optimizer correctly suppressed $0.00 allocation to BTC (Meta-Prob: 0.45 < 0.50).")
        else:
            logger.error(f" [FAIL] HRP Optimizer failed clipping. Allocated ${allocs.get('BTC')} to BTC despite weak edge.")
            
    except ImportError as e:
        logger.error(f" [FAIL] Unable to import Core Hooks for testing: {str(e)}")


if __name__ == "__main__":
    logger.info("==========================================")
    logger.info("INITIATING PROGRAMMATIC VALIDATION TRACER")
    logger.info("==========================================")
    
    check_pre_execution()
    check_validation_outputs()
    test_live_simulation_hooks()
    
    logger.info("\nValidation Monitoring Sequence Complete.")
