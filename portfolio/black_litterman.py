"""
portfolio/black_litterman.py - Black-Litterman Portfolio Optimization

Combines:
- Market equilibrium (CAPM) as prior
- ML signal views as active bets
- Posterior blended portfolio

Benefits over mean-variance:
- More stable allocations
- Incorporates uncertainty in views
- Naturally Bayesian framework

Free implementation - no paid data required.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlackLittermanResult:
    """Result of Black-Litterman optimization."""
    posterior_returns: Dict[str, float]
    posterior_weights: Dict[str, float]
    prior_returns: Dict[str, float]
    view_contribution: Dict[str, float]
    confidence_adjusted: bool
    
    def to_dict(self) -> Dict:
        return {
            'posterior_returns': self.posterior_returns,
            'posterior_weights': self.posterior_weights,
            'prior_returns': self.prior_returns,
            'view_contribution': self.view_contribution,
            'confidence_adjusted': self.confidence_adjusted
        }


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer.
    
    Combines market equilibrium returns (CAPM-implied)
    with active views from ML signals to produce
    stable, diversified portfolios.
    
    Math:
    - Prior: π = δ * Σ * w_mkt (equilibrium returns)
    - Views: P * μ = Q + ε (active views with uncertainty)
    - Posterior: μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1π + P'Ω^-1Q]
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion coefficient (delta)
            tau: Uncertainty in prior (typically 0.01-0.10)
            risk_free_rate: Annual risk-free rate
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        
        logger.info("BlackLittermanOptimizer initialized")
        logger.info(f"  Risk aversion: {risk_aversion}, Tau: {tau}")
    
    def calculate_equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate equilibrium returns using reverse optimization.
        
        π = δ * Σ * w_mkt
        
        Args:
            cov_matrix: Covariance matrix (annualized)
            market_weights: Market cap weights
        
        Returns:
            Dict of equilibrium returns
        """
        symbols = list(cov_matrix.columns)
        weights = np.array([market_weights.get(s, 1/len(symbols)) for s in symbols])
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate equilibrium returns
        cov = cov_matrix.values
        pi = self.risk_aversion * cov @ weights
        
        return {s: float(pi[i]) for i, s in enumerate(symbols)}
    
    def optimize(
        self,
        returns_data: Dict[str, pd.Series],
        views: Dict[str, float],
        view_confidences: Optional[Dict[str, float]] = None,
        market_weights: Optional[Dict[str, float]] = None
    ) -> BlackLittermanResult:
        """
        Run Black-Litterman optimization.
        
        Args:
            returns_data: Historical returns for each symbol
            views: Dict of {symbol: expected_return} from ML signals
            view_confidences: Dict of {symbol: confidence 0-1}
            market_weights: Market cap weights (defaults to equal)
        
        Returns:
            BlackLittermanResult
        """
        # Build returns dataframe
        symbols = list(returns_data.keys())
        n = len(symbols)
        
        if n < 2:
            return self._fallback_result(symbols, views)
        
        # Align returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 60:
            return self._fallback_result(symbols, views)
        
        # Covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252
        
        # Market weights (default: equal)
        if market_weights is None:
            market_weights = {s: 1/n for s in symbols}
        
        # Equilibrium returns (prior)
        prior_returns = self.calculate_equilibrium_returns(cov_matrix, market_weights)
        
        # If no views, return equilibrium
        if not views:
            weights = self._optimize_weights(cov_matrix, prior_returns)
            return BlackLittermanResult(
                posterior_returns=prior_returns,
                posterior_weights=weights,
                prior_returns=prior_returns,
                view_contribution={s: 0.0 for s in symbols},
                confidence_adjusted=False
            )
        
        # Build view matrices
        P, Q, omega = self._build_view_matrices(
            symbols, views, view_confidences, cov_matrix
        )
        
        # Posterior returns (Black-Litterman formula)
        posterior_returns = self._calculate_posterior(
            prior_returns, cov_matrix, P, Q, omega, symbols
        )
        
        # Optimize weights based on posterior
        weights = self._optimize_weights(cov_matrix, posterior_returns)
        
        # Calculate view contribution
        view_contrib = {
            s: posterior_returns[s] - prior_returns[s]
            for s in symbols
        }
        
        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_weights=weights,
            prior_returns=prior_returns,
            view_contribution=view_contrib,
            confidence_adjusted=view_confidences is not None
        )
    
    def _build_view_matrices(
        self,
        symbols: List[str],
        views: Dict[str, float],
        confidences: Optional[Dict[str, float]],
        cov_matrix: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build view matrices (P, Q, Ω).
        
        P: Pick matrix (which assets in each view)
        Q: View returns
        Ω: View uncertainty (diagonal)
        """
        n = len(symbols)
        symbol_idx = {s: i for i, s in enumerate(symbols)}
        
        # Filter views to symbols we have
        valid_views = {s: v for s, v in views.items() if s in symbol_idx}
        k = len(valid_views)
        
        if k == 0:
            return np.zeros((0, n)), np.zeros(0), np.eye(1)
        
        # Build P matrix (one row per view)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        
        for i, (symbol, view_return) in enumerate(valid_views.items()):
            idx = symbol_idx[symbol]
            P[i, idx] = 1.0
            Q[i] = view_return
            
            # View uncertainty based on confidence
            conf = 1.0
            if confidences and symbol in confidences:
                conf = max(0.1, min(1.0, confidences[symbol]))
            
            # Uncertainty inversely proportional to confidence
            # Tied to asset variance
            asset_var = cov_matrix.iloc[idx, idx]
            omega_diag[i] = (1 / conf) * self.tau * asset_var
        
        omega = np.diag(omega_diag)
        
        return P, Q, omega
    
    def _calculate_posterior(
        self,
        prior_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Calculate posterior returns using Black-Litterman formula."""
        len(symbols)
        
        # Convert to arrays
        pi = np.array([prior_returns.get(s, 0) for s in symbols])
        sigma = cov_matrix.values
        tau_sigma = self.tau * sigma
        
        if len(Q) == 0:
            return prior_returns
        
        try:
            # Black-Litterman posterior
            # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1π + P'Ω^-1Q]
            
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)
            
            # Left side: inverse of sum
            left_inv = tau_sigma_inv + P.T @ omega_inv @ P
            left = np.linalg.inv(left_inv)
            
            # Right side
            right = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
            
            # Posterior
            mu_bl = left @ right
            
            return {s: float(mu_bl[i]) for i, s in enumerate(symbols)}
            
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, returning prior")
            return prior_returns
    
    def _optimize_weights(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance.
        
        Simplified: w* = (δΣ)^-1 * μ (unconstrained)
        Then normalized and constrained.
        """
        symbols = list(cov_matrix.columns)
        n = len(symbols)
        
        mu = np.array([expected_returns.get(s, 0) for s in symbols])
        sigma = cov_matrix.values
        
        try:
            # Unconstrained optimal weights
            sigma_inv = np.linalg.inv(self.risk_aversion * sigma)
            w_star = sigma_inv @ mu
            
            # Normalize to sum to 1
            w_star = w_star / np.abs(w_star).sum()
            
            # Constrain to [-0.3, 0.3] per position
            w_star = np.clip(w_star, -0.3, 0.3)
            w_star = w_star / np.abs(w_star).sum()  # Re-normalize
            
            return {s: float(w_star[i]) for i, s in enumerate(symbols)}
            
        except np.linalg.LinAlgError:
            # Fallback to equal weight
            return {s: 1/n for s in symbols}
    
    def _fallback_result(
        self,
        symbols: List[str],
        views: Dict[str, float]
    ) -> BlackLittermanResult:
        """Return fallback result when optimization fails."""
        n = len(symbols)
        equal_weight = 1/n if n > 0 else 0
        
        return BlackLittermanResult(
            posterior_returns={s: 0.10 for s in symbols},  # 10% default
            posterior_weights={s: equal_weight for s in symbols},
            prior_returns={s: 0.10 for s in symbols},
            view_contribution={s: 0.0 for s in symbols},
            confidence_adjusted=False
        )
    
    def combine_with_ml_signals(
        self,
        returns_data: Dict[str, pd.Series],
        ml_signals: Dict[str, Dict],
        signal_scale: float = 0.20
    ) -> BlackLittermanResult:
        """
        Convenience method to combine with ML signal outputs.
        
        Args:
            returns_data: Historical returns
            ml_signals: Dict of {symbol: {'signal': float, 'confidence': float}}
            signal_scale: Scale factor for signal -> return conversion
        
        Returns:
            BlackLittermanResult
        """
        # Convert signals to views
        views = {}
        confidences = {}
        
        for symbol, signal_data in ml_signals.items():
            signal = signal_data.get('signal', 0)
            conf = signal_data.get('confidence', 0.5)
            
            # Convert signal (-1 to 1) to expected return
            # signal * scale = expected excess return
            views[symbol] = signal * signal_scale
            confidences[symbol] = conf
        
        return self.optimize(
            returns_data=returns_data,
            views=views,
            view_confidences=confidences
        )
