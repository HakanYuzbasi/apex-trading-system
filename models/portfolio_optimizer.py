"""
models/portfolio_optimizer.py
================================================================================
PHASE 4: DYNAMIC LIQUIDITY & HIERARCHICAL RISK PARITY (HRP) OPTIMIZATION
================================================================================

Implements a Portfolio Optimizer that ingests:
1. Signal Probs from the Phase 2 Hybrid Meta-Learner (LSTM + Trees).
2. Dynamic Exits/Risk States from the Phase 3 DQN Trade Manager.
3. Live L2 Imbalance and Cross-Asset Beta correlations.

Algorithms used:
- Hierarchical Risk Parity (HRP): Clusters assets by correlation distance and
  allocates inversely to cluster variance.
- Liquidity Constraints: Scales HRP weights proportionally downward if ADV
  (Average Daily Volume) or Bid-Ask spreads indicate execution danger.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

from config import ApexConfig

logger = logging.getLogger(__name__)

try:
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - exercised indirectly in lightweight tests
    sch = None
    squareform = None


def _require_scipy() -> None:
    if sch is None or squareform is None:
        raise ImportError("SciPy is required for HRP allocation")


def get_kelly_multiplier(
    ml_confidence: float,
    historical_win_rate: float,
    is_high_vix: bool = False,
) -> float:
    """
    Convert model confidence and realized win-rate into a bounded sizing multiplier.

    Convert Kelly edge into a leverage-style multiplier for the execution loop.
    A truly negative Kelly edge still returns 0.0; cold-start protection is
    handled by the caller via the win-rate floor before this helper is invoked.
    """
    conf = float(np.clip(ml_confidence, 0.0, 1.0))
    win_rate = float(historical_win_rate)
    if win_rate <= 0.0 or win_rate >= 1.0 or not np.isfinite(win_rate):
        win_rate = 0.5
    else:
        win_rate = float(np.clip(win_rate, 0.01, 0.99))

    implied_edge_prob = (conf + win_rate) / 2.0
    loss_prob = 1.0 - implied_edge_prob
    odds_ratio = float(getattr(ApexConfig, "KELLY_ODDS_RATIO", 1.5))
    fractional_modifier = float(getattr(ApexConfig, "KELLY_FRACTION", 0.5))
    base_risk_pct = float(getattr(ApexConfig, "KELLY_BASE_RISK_PCT", 0.10))
    min_leverage = float(getattr(ApexConfig, "KELLY_MIN_LEVERAGE", 0.10))
    max_leverage = float(getattr(ApexConfig, "KELLY_MAX_LEVERAGE", 3.0))

    full_kelly = implied_edge_prob - (loss_prob / max(odds_ratio, 1e-9))
    if full_kelly <= 0.0:
        return 0.0

    leverage = (full_kelly * max(fractional_modifier, 0.0)) / max(base_risk_pct, 1e-9)

    if is_high_vix:
        leverage *= 0.6

    return float(np.clip(leverage, min_leverage, max_leverage))


# ------------------------------------------------------------------------------
# 1. HIERARCHICAL RISK PARITY (HRP) ENGINE
# ------------------------------------------------------------------------------
class HRPOptimizer:
    """Computes Hierarchical Risk Parity weights for a correlation matrix."""
    
    @staticmethod
    def get_inverse_variance_weight(cov_matrix: pd.DataFrame, indices: List[int]) -> np.ndarray:
        """Calculate Inverse Variance weights for a subset of assets."""
        sub_cov = cov_matrix.iloc[indices, indices].values
        inv_diag = 1.0 / np.diag(sub_cov)
        weights = inv_diag / np.sum(inv_diag)
        return weights.reshape(-1, 1)

    @staticmethod
    def get_cluster_variance(cov_matrix: pd.DataFrame, indices: List[int]) -> float:
        """Calculate variance of a cluster."""
        w = HRPOptimizer.get_inverse_variance_weight(cov_matrix, indices)
        sub_cov = cov_matrix.iloc[indices, indices].values
        var = np.dot(np.dot(w.T, sub_cov), w)[0, 0]
        return var

    @staticmethod
    def get_quasi_diag_sort(linkage_matrix: np.ndarray, num_items: int) -> List[int]:
        """Sort clustered items by distance."""
        sort_idx = [linkage_matrix[-1, 0], linkage_matrix[-1, 1]]
        sort_idx = [int(i) for i in sort_idx]
        while max(sort_idx) >= num_items:
            sort_idx_new = []
            for i in sort_idx:
                if i < num_items:
                    sort_idx_new.append(i)
                else:
                    sort_idx_new.append(int(linkage_matrix[i - num_items, 0]))
                    sort_idx_new.append(int(linkage_matrix[i - num_items, 1]))
            sort_idx = sort_idx_new
        return sort_idx

    @staticmethod
    def get_recursive_bisection(cov_matrix: pd.DataFrame, sort_idx: List[int]) -> pd.Series:
        """Compute HRP allocation weights recursively."""
        weights = pd.Series(1.0, index=sort_idx)
        clusters = [sort_idx]
        
        while len(clusters) > 0:
            clusters_idx = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = int(len(cluster) / 2)
                    clusters_idx.append(cluster[:mid])
                    clusters_idx.append(cluster[mid:])
            
            for i in range(0, len(clusters_idx), 2):
                cluster_first = clusters_idx[i]
                cluster_second = clusters_idx[i+1]
                
                var_first = HRPOptimizer.get_cluster_variance(cov_matrix, cluster_first)
                var_second = HRPOptimizer.get_cluster_variance(cov_matrix, cluster_second)
                
                # Allocation factor
                alpha = 1 - var_first / (var_first + var_second)
                
                weights[cluster_first] *= alpha
                weights[cluster_second] *= (1 - alpha)
                
            clusters = clusters_idx
            
        weights.index = cov_matrix.columns[weights.index] # Map back to symbol names
        return weights.sort_index()

    def allocate(self, returns_df: pd.DataFrame) -> pd.Series:
        """Generate full HRP weights for the given returns history."""
        _require_scipy()
        if returns_df.shape[1] < 2:
            # Degenerate case (1 asset)
            return pd.Series(1.0, index=returns_df.columns)
            
        # 1. Compute Correlation and Covariance
        corr = returns_df.corr().fillna(0)
        cov = returns_df.cov().fillna(0)
        
        # 2. Distance Matrix: D = sqrt(0.5 * (1 - Cor))
        # Clip values slightly inside [-1, 1] to avoid float precision issues with sqrt
        dist = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, 1.0))
        
        # 3. Hierarchical Clustering
        condensed_dist = squareform(dist.values, checks=False)
        linkage_matrix = sch.linkage(condensed_dist, method='single')
        
        # 4. Quasi-Diagonalization & Recursive Bisection
        sort_idx = self.get_quasi_diag_sort(linkage_matrix, returns_df.shape[1])
        hrp_weights = self.get_recursive_bisection(cov, sort_idx)
        
        return hrp_weights


# ------------------------------------------------------------------------------
# 2. LIQUIDITY & META-SIGNAL ADJUSTMENTS
# ------------------------------------------------------------------------------
class DynamicPortfolioManager:
    """
    Combines HRP optimal baseline weights with Meta-Learner Edge and 
    Liquidity Constraints to emit real trade sizes.
    """
    def __init__(self, max_capital_usd: float = 100000.0, max_adv_participation: float = 0.01):
        self.hrp_engine = HRPOptimizer()
        self.max_capital = max_capital_usd
        self.max_adv_participation = max_adv_participation # Cap trade size to 1% of ADV

    def compute_liquidity_discount(self, symbol: str, adv_usd: float, estimated_spread_bps: float) -> float:
        """
        Penalizes allocations if the asset is illiquid. 
        Returns multiplier [0.0 - 1.0]
        """
        # 1. Spread drag
        spread_discount = max(0.0, 1.0 - (estimated_spread_bps / 50.0)) # 50 bps spread = 0 weight
        
        # 2. Capital vs ADV check (Is our HRP weight demanding too much volume?)
        # Let's say we have an idealized $10K allocation. Is $10K > 1% of ADV?
        # Handled in the final absolute sizing phase, but we can emit a risk scaler here.
        if adv_usd < 100000:
            vol_discount = 0.2
        elif adv_usd < 1000000:
            vol_discount = 0.5
        else:
            vol_discount = 1.0
            
        return float(np.clip(spread_discount * vol_discount, 0.0, 1.0))

    def evaluate_live_portfolio_tick(
        self,
        returns_history_df: pd.DataFrame,
        meta_signals: Dict[str, float],
        dqn_actions: Dict[str, int],
        liquidity_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        LIVE HOOK: Produces actual USD allocation sizes for the ExecutionManager.
        
        Args:
            returns_history_df: Recent returns [Time, Asset] for Covariance/HRP.
            meta_signals: {symbol: prob} from Phase 2 Hybrid Learner.
            dqn_actions: {symbol: action} from Phase 3 DQN (0: Hold, 1: 50%, 2: Flat).
            liquidity_metrics: {symbol: {'adv': float, 'spread_bps': float}}.
            
        Returns:
            Dict[str, float] mapping execution symbol to Target USD position size.
        """
        # 1. Generate Baseline Risk-Parity allocation assuming all are active
        active_symbols = [s for s in returns_history_df.columns if meta_signals.get(s, 0) > 0.10]
        
        if not active_symbols:
            return {}
            
        # Filter histories to active symbols
        df_active = returns_history_df[active_symbols]
        hrp_weights = self.hrp_engine.allocate(df_active) # Sums to ~1.0
        
        final_allocations_usd = {}
        
        # 2. Apply Meta-Signal, DQN, and Liquidity Constraints
        for symbol, weight in hrp_weights.items():
            base_alloc_usd = weight * self.max_capital
            
            # Phase 2: Meta-Learner Signal Edge Scaling
            prob = meta_signals.get(symbol, 0.5)
            # Kelly-esque scaler: (prob - 0.5) * 2. If prob=0.6, scaler=0.2. If prob=0.9, scaler=0.8
            edge_scaler = max(0.0, (prob - 0.5) * 2.0)
            
            # Phase 3: DQN Trade Action Scaling
            action = dqn_actions.get(symbol, 0)
            if action == 2:   # Full Close
                dqn_scaler = 0.0
            elif action == 1: # Partial Close (Hold 50%)
                dqn_scaler = 0.5
            else:             # Hold (Maintain)
                dqn_scaler = 1.0
                
            # Phase 4: Liquidity Scaling
            liq_data = liquidity_metrics.get(symbol, {'adv': 1e7, 'spread_bps': 2.0})
            liq_scaler = self.compute_liquidity_discount(symbol, liq_data['adv'], liq_data['spread_bps'])
            
            # Combine scalers
            raw_target_usd = base_alloc_usd * edge_scaler * dqn_scaler * liq_scaler
            
            # Final Volume Constraint Hard-Cap
            max_trade_usd = liq_data['adv'] * self.max_adv_participation
            final_target_usd = min(raw_target_usd, max_trade_usd)
            
            final_allocations_usd[symbol] = round(final_target_usd, 2)
            
            logger.debug(f"{symbol} | HRP Wght: {weight:.2f} | Edge: {edge_scaler:.2f} | DQN: {dqn_scaler:.2f} | Liq: {liq_scaler:.2f} | Trgt: ${final_target_usd}")
            
        return final_allocations_usd


# ------------------------------------------------------------------------------
# 3. BACKTESTING & EVALUATION 
# ------------------------------------------------------------------------------
def run_hrp_backtest():
    """Simulates Phase 4 Portfolio Engine over a mock temporal sequence."""
    logger.info("Initializing Phase 4 HRP + Liquidity Backtester...")
    
    # Mock Universe
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'SPY', 'QQQ']
    n_bars = 500
    
    # Mock Returns & Covariance
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.01, (n_bars, len(symbols))), 
        columns=symbols
    )
    
    # Induce artificial correlation clustering (Crypto vs Equities)
    returns['ETH/USD'] += returns['BTC/USD'] * 0.8  
    returns['SOL/USD'] += returns['BTC/USD'] * 0.6
    returns['QQQ'] += returns['SPY'] * 0.9
    
    manager = DynamicPortfolioManager(max_capital_usd=100000.0)
    
    hrp_history = []
    eq_history = []
    
    logger.info("Running rolling portfolio allocations...")
    # Rolling Walk-Forward
    for i in range(100, n_bars):
        window_returns = returns.iloc[i-100:i]
        
        # Mock signals (Phase 2 & 3 outputs)
        meta_signals = {s: np.random.uniform(0.5, 0.9) for s in symbols}
        dqn_actions = {s: np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) for s in symbols}
        liquidity = {s: {'adv': 10000000.0, 'spread_bps': 2.0} for s in symbols}
        
        # Phase 4 Live Tick
        target_usd = manager.evaluate_live_portfolio_tick(
            window_returns, meta_signals, dqn_actions, liquidity
        )
        
        # Calculate Forward PnL
        forward_return = returns.iloc[i]
        
        # HRP Realized
        hrp_pnl = sum([target_usd.get(s, 0.0) * forward_return[s] for s in symbols])
        hrp_history.append(hrp_pnl)
        
        # Naive Equal Weight (No Edge Scaler, No HRP, No RL Close)
        eq_alloc = 100000.0 / len(symbols)
        eq_pnl = sum([eq_alloc * forward_return[s] for s in symbols])
        eq_history.append(eq_pnl)

    # Metrics
    hrp_cum = np.cumsum(hrp_history)
    eq_cum = np.cumsum(eq_history)
    hrp_sharpe = np.sqrt(252*288) * np.mean(hrp_history) / (np.std(hrp_history) + 1e-8)
    eq_sharpe = np.sqrt(252*288) * np.mean(eq_history) / (np.std(eq_history) + 1e-8)
    
    logger.info("="*60)
    logger.info("PHASE 4 BACKTEST RESULTS (HRP vs EQUAL WEIGHT)")
    logger.info("="*60)
    logger.info(f"HRP + Phase 2/3 Sharpe : {hrp_sharpe:.2f}")
    logger.info(f"Naive Equal Wght Sharpe: {eq_sharpe:.2f}")
    logger.info(f"HRP Total Net PnL      : ${hrp_cum[-1]:.2f}")
    logger.info(f"Naive Total Net PnL    : ${eq_cum[-1]:.2f}")
    
    # Save Artifacts
    os.makedirs('logs', exist_ok=True)
    pd.DataFrame({'HRP_PnL': hrp_history, 'Equal_PnL': eq_history}).to_csv('logs/phase4_hrp_backtest.csv')

if __name__ == "__main__":
    run_hrp_backtest()
