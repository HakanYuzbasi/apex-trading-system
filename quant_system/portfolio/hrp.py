"""
quant_system/portfolio/hrp.py
================================================================================
Hierarchical Risk Parity (HRP) & Regime Clustering
================================================================================
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import List, Tuple

class HRPOptimizer:
    def __init__(self, max_adv_participation: float = 0.05):
        self.max_adv_participation = max_adv_participation
        self.current_regime = 1
        
    def _get_inverse_variance(self, cov: pd.DataFrame, idx: List[int]) -> np.ndarray:
        sub_cov = cov.iloc[idx, idx].values
        inv_diag = 1.0 / np.diag(sub_cov)
        return (inv_diag / np.sum(inv_diag)).reshape(-1, 1)

    def _get_cluster_var(self, cov: pd.DataFrame, idx: List[int]) -> float:
        w = self._get_inverse_variance(cov, idx)
        sub_cov = cov.iloc[idx, idx].values
        return np.dot(np.dot(w.T, sub_cov), w)[0, 0]

    def _quasi_diag(self, link: np.ndarray, num: int) -> List[int]:
        sort_idx = [int(link[-1, 0]), int(link[-1, 1])]
        while max(sort_idx) >= num:
            new_idx = []
            for i in sort_idx:
                if i < num: new_idx.append(i)
                else:
                    new_idx.append(int(link[i - num, 0]))
                    new_idx.append(int(link[i - num, 1]))
            sort_idx = new_idx
        return sort_idx

    def _rec_bipartite(self, cov: pd.DataFrame, sort_idx: List[int]) -> pd.Series:
        weights = pd.Series(1.0, index=sort_idx)
        clusters = [sort_idx]
        while len(clusters) > 0:
            c_new = []
            for c in clusters:
                if len(c) > 1:
                    m = int(len(c) / 2)
                    c_new.extend([c[:m], c[m:]])
            for i in range(0, len(c_new), 2):
                c1, c2 = c_new[i], c_new[i+1]
                v1 = self._get_cluster_var(cov, c1)
                v2 = self._get_cluster_var(cov, c2)
                alpha = 1 - v1 / (v1 + v2)
                weights[c1] *= alpha
                weights[c2] *= (1 - alpha)
            clusters = c_new
        return weights

    def allocate(self, returns_df: pd.DataFrame) -> Tuple[pd.Series, int]:
        """Returns bounds and distinct macro-regime state."""
        if returns_df.shape[1] < 2:
            return pd.Series(1.0, index=returns_df.columns), 1
            
        corr = returns_df.corr().fillna(0)
        cov = returns_df.cov().fillna(0)
        
        dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))
        condensed = squareform(dist.values, checks=False)
        link = sch.linkage(condensed, method='single')
        
        # UPGRADE 5: HRP REGIME INTEGRATION
        # Few clusters (assets highly correlated moving together) = CRASH/HIGH VOL
        # Many clusters (assets disjoint) = NORMAL
        flat_clusters = sch.fcluster(link, t=0.5, criterion='distance')
        num_clusters = len(np.unique(flat_clusters))
        
        if num_clusters == 1:
            self.current_regime = 3 # Crash Regime
        elif num_clusters <= 3:
            self.current_regime = 2 # High Vol Regime
        else:
            self.current_regime = 1 # Normal Regime
        
        sort_idx = self._quasi_diag(link, returns_df.shape[1])
        weights = self._rec_bipartite(cov, sort_idx)
        weights.index = returns_df.columns[weights.index]
        return weights.sort_index(), self.current_regime
