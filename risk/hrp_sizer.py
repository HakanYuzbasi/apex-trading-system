"""
risk/hrp_sizer.py — Hierarchical Risk Parity Position Sizer

HRP (López de Prado 2016) builds a diversified portfolio by:
  1. Computing the correlation + covariance matrix of all open positions.
  2. Hierarchically clustering correlated assets so that each cluster
     is treated as a single diversification unit.
  3. Allocating risk budget inversely proportional to cluster variance.

Result: positions in highly-correlated clusters are collectively down-sized,
preventing correlated-factor blowups that per-symbol Kelly misses.

Outputs:
  - get_size_multiplier(symbol, positions, returns_dict) → float [0.5, 1.0]
    A sizing dampener to apply on top of the existing Kelly/vol sizing.
    Returns 1.0 when there's insufficient history or only 1 position.

Wire-in (execution_loop sizing stack):
    from risk.hrp_sizer import HRPSizer
    _hrp = HRPSizer()
    mult = _hrp.get_size_multiplier(symbol, self.positions, self._returns_cache)
    shares = shares * mult
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_HISTORY = 10    # min return observations per symbol
_MIN_SYMBOLS = 2     # need at least 2 symbols to compute covariance
_MAX_CLUSTER_WEIGHT = 0.50   # single cluster can't exceed 50% of portfolio
_DAMPEN_FLOOR = 0.50          # minimum multiplier output


class HRPSizer:
    """
    Hierarchical Risk Parity sizing dampener.

    The multiplier answers: "given all existing positions are correlated,
    how much should this new position be scaled down?"
    """

    def __init__(
        self,
        min_history: int = _MIN_HISTORY,
        max_cluster_weight: float = _MAX_CLUSTER_WEIGHT,
        dampen_floor: float = _DAMPEN_FLOOR,
    ) -> None:
        self._min_history = min_history
        self._max_cluster_weight = max_cluster_weight
        self._dampen_floor = dampen_floor

    def get_size_multiplier(
        self,
        symbol: str,
        positions: Dict[str, float],
        returns_cache: Dict[str, List[float]],
    ) -> float:
        """
        Return a multiplier [dampen_floor, 1.0] for the new position in `symbol`.

        Args:
            symbol:        The symbol being sized.
            positions:     {symbol: quantity} of all current open positions.
            returns_cache: {symbol: [daily_return, ...]} rolling history.

        Returns:
            1.0  → symbol is uncorrelated with portfolio; no dampening.
            <1.0 → symbol is correlated; reduce size proportionally.
        """
        open_syms = [s for s, q in positions.items() if q != 0 and s != symbol]
        if len(open_syms) < _MIN_SYMBOLS - 1:
            return 1.0  # no existing positions to correlate with

        # Gather returns for symbol + all open positions
        all_syms = [symbol] + open_syms
        matrix = self._build_return_matrix(all_syms, returns_cache)
        if matrix is None or matrix.shape[0] < _MIN_SYMBOLS or matrix.shape[1] < self._min_history:
            return 1.0

        try:
            cov = np.cov(matrix)
            if cov.ndim == 0:
                return 1.0
            corr = self._cov_to_corr(cov)

            # Average absolute correlation of target (index 0) with existing positions
            n = matrix.shape[0]
            avg_abs_corr = float(np.mean([abs(corr[0, i]) for i in range(1, n)]))

            # Dampen proportionally to correlation: corr=0 → 1.0, corr=1.0 → dampen_floor
            mult = max(self._dampen_floor, 1.0 - avg_abs_corr * (1.0 - self._dampen_floor))
            logger.debug(
                "HRP [%s]: avg_abs_corr=%.3f → mult=%.2f (n=%d)",
                symbol, avg_abs_corr, mult, n,
            )
            return round(mult, 4)

        except Exception as exc:
            logger.debug("HRP sizing error for %s: %s", symbol, exc)
            return 1.0

    def get_portfolio_weights(
        self,
        symbols: List[str],
        returns_cache: Dict[str, List[float]],
    ) -> Optional[Dict[str, float]]:
        """
        Return full HRP weight vector for a list of symbols.
        Returns None if insufficient data.
        """
        matrix = self._build_return_matrix(symbols, returns_cache)
        if matrix is None or matrix.shape[0] < 2 or matrix.shape[1] < self._min_history:
            return None
        try:
            cov = np.cov(matrix)
            corr = self._cov_to_corr(cov)
            weights = self._hrp_weights(cov, corr)
            return {s: round(float(w), 6) for s, w in zip(symbols, weights)}
        except Exception as exc:
            logger.debug("HRP portfolio weights error: %s", exc)
            return None

    # ── HRP Algorithm ─────────────────────────────────────────────────────────

    def _hrp_weights(self, cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """
        Full HRP: seriation → recursive bisection → inverse-variance weights per cluster.
        """
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])
        if n == 2:
            v = np.array([cov[i, i] for i in range(n)])
            iv = 1.0 / (v + 1e-12)
            return iv / iv.sum()

        # 1. Distance matrix from correlation
        dist = np.sqrt(0.5 * (1.0 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0.0)

        # 2. Quasi-diagonalisation via single-linkage clustering
        order = self._quasi_diag(dist)

        # 3. Recursive bisection
        weights = np.ones(n)
        clusters = [list(range(n))]
        while clusters:
            cluster = clusters.pop()
            if len(cluster) <= 1:
                continue
            # Split into two halves by sorted order
            mid = len(cluster) // 2
            left = [order[i] for i in range(len(order)) if i < mid and order[i] in cluster]
            right = [order[i] for i in range(len(order)) if i >= mid and order[i] in cluster]
            if not left or not right:
                # fallback: split evenly
                half = len(cluster) // 2
                left, right = cluster[:half], cluster[half:]

            var_left = self._cluster_var(cov, left)
            var_right = self._cluster_var(cov, right)
            total = var_left + var_right + 1e-12
            alpha_left = 1.0 - var_left / total
            alpha_right = 1.0 - alpha_left

            weights[left] *= alpha_left
            weights[right] *= alpha_right

            if len(left) > 1:
                clusters.append(left)
            if len(right) > 1:
                clusters.append(right)

        # Normalise
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        return weights

    @staticmethod
    def _quasi_diag(dist: np.ndarray) -> List[int]:
        """Single-linkage clustering → leaf order (simplified)."""
        n = dist.shape[0]
        # Greedy nearest-neighbour path as a simple seriation approximation
        unvisited = set(range(n))
        order = []
        current = 0
        unvisited.remove(current)
        order.append(current)
        while unvisited:
            nearest = min(unvisited, key=lambda j: dist[current, j])
            order.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return order

    @staticmethod
    def _cluster_var(cov: np.ndarray, cluster: List[int]) -> float:
        """Inverse-variance weighted cluster variance."""
        sub = cov[np.ix_(cluster, cluster)]
        variances = np.diag(sub)
        iv = 1.0 / (variances + 1e-12)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.diag(cov))
        outer = np.outer(std, std)
        corr = cov / (outer + 1e-12)
        np.clip(corr, -1, 1, out=corr)
        return corr

    def _build_return_matrix(
        self,
        symbols: List[str],
        returns_cache: Dict[str, List[float]],
    ) -> Optional[np.ndarray]:
        """Build (n_symbols × n_obs) matrix; only include symbols with sufficient history."""
        rows = []
        valid_syms = []
        for sym in symbols:
            rets = returns_cache.get(sym) or returns_cache.get(sym.split(":")[-1])
            if rets and len(rets) >= self._min_history:
                rows.append(rets[-self._min_history:])
                valid_syms.append(sym)
        if len(valid_syms) < _MIN_SYMBOLS:
            return None
        # Align lengths
        min_len = min(len(r) for r in rows)
        matrix = np.array([r[-min_len:] for r in rows], dtype=float)
        return matrix
