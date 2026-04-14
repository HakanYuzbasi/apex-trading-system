from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional

class MeanVarianceSizer:
    """
    Portfolio sizer implementing Mean-Variance Optimization (MVO)
    with Ledoit-Wolf style linear shrinkage of the covariance matrix.
    """
    def __init__(
        self,
        shrinkage_limit: float = 0.5,
        target_vol: float = 0.02,  # Daily target vol
        is_market_neutral: bool = True,
        base_kelly_multiplier: float = 0.5  # Half-Kelly for safety
    ) -> None:
        self.shrinkage_limit = shrinkage_limit
        self.target_vol = target_vol
        self.is_market_neutral = is_market_neutral
        self.base_kelly_multiplier = base_kelly_multiplier

    def compute_weights(
        self,
        expected_returns: np.ndarray,
        returns_history: np.ndarray,
        max_weight: float = 0.2
    ) -> np.ndarray:
        """
        Compute optimal weights using MVO with shrinkage.
        
        Args:
            expected_returns: Vector of expected alpha/returns for each asset [N]
            returns_history: Matrix of historical returns [T x N]
            max_weight: Maximum absolute weight for a single asset
            
        Returns:
            Optimal weight vector [N]
        """
        n_assets = expected_returns.shape[0]
        if n_assets == 0:
            return np.array([])

        # 1. Shrinkage-Adjusted Covariance
        sample_cov = np.cov(returns_history, rowvar=False)
        # Linear shrinkage towards identity (or average variance)
        prior = np.eye(n_assets) * np.mean(np.diag(sample_cov))
        
        # Simple heuristic shrinkage factor based on T/N if not provided
        T, N = returns_history.shape
        gamma = min(self.shrinkage_limit, N / T if T > 0 else self.shrinkage_limit)
        
        shrunk_cov = (1 - gamma) * sample_cov + gamma * prior

        # 2. Optimization
        # Objective: Maximize returns - Risk Aversion * Variance
        # For simplicity, we'll solve: min -w'r subject to w'Qw <= target_var
        
        def objective(w):
            return -np.dot(w, expected_returns)

        def constraint_vol(w):
            return self.target_vol**2 - np.dot(w.T, np.dot(shrunk_cov, w))

        constraints = [{'type': 'ineq', 'fun': constraint_vol}]
        
        if self.is_market_neutral:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w)})

        bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        w0 = np.zeros(n_assets)
        
        res = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        if not res.success:
            # Fallback to simple risk-parity or scaled expected returns
            return self._fallback_weights(expected_returns, shrunk_cov)

        return res.x

    def _fallback_weights(self, returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Simple inverse-volatility weighting as fallback."""
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / np.clip(vols, 1e-6, None)
        weights = inv_vols * np.sign(returns)
        # Normalize to target vol crudely
        current_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        if current_vol > 0:
            weights *= (self.target_vol / current_vol)
        return weights

    def allocate_strategies(self, prob_high_vol: float) -> Dict[str, float]:
        """
        Ensemble Logic: Allocates capital using a Sigmoid function based on 
        the Bayesian high-volatility probability, scaled by a Kelly criterion base.
        
        Args:
            prob_high_vol: Floating probability (0.0 to 1.0) of high vol state.
            
        Returns:
            Dict mapping strategy string names to allocation weight percentage.
        """
        # Smooth Sigmoid transition
        # k = 10 steepness, shifting center at P=0.5
        # This yields w_breakout ~ 0.95 at P=0.8, ~ 0.05 at P=0.2
        k = 10.0
        w_breakout = 1.0 / (1.0 + np.exp(-k * (prob_high_vol - 0.5)))
        w_pairs = 1.0 - w_breakout
        
        # Apply Kelly base multiplier
        pair_weight = w_pairs * self.base_kelly_multiplier
        breakout_weight = w_breakout * self.base_kelly_multiplier
        
        return {"KalmanPairs": float(pair_weight), "BreakoutPod": float(breakout_weight)}

