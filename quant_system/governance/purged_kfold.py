"""
quant_system/governance/purged_kfold.py
================================================================================
Combinatorially Purged Cross-Validation (MANDATORY REQUIREMENT 9)
================================================================================
Eliminates look-ahead bias and temporal leakage between test and train folds.
"""

import numpy as np
from typing import List, Tuple
from itertools import combinations
from sklearn.model_selection import KFold

class PurgedKFold:
    def __init__(self, n_splits: int = 5, n_test_splits: int = 2, embargo_pct: float = 0.05):
        """
        Args:
            n_splits: Total number of temporal blocks.
            n_test_splits: Number of blocks to group as the test set in each permutation.
            embargo_pct: Fractional size of the embargo added after test sets.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Yields (train_idx, test_idx) strictly preventing temporal leakage."""
        indices = np.arange(n_samples)
        
        # Divide into sequential blocks
        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        blocks = [test_idx for _, test_idx in kfold.split(indices)]
        
        # All combinatorial choices of test blocks
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        splits = []
        
        purge_size = int(n_samples * self.embargo_pct)
        
        for test_groups in test_combinations:
            test_indices = []
            train_indices = []
            
            for i in test_groups:
                test_indices.extend(blocks[i])
            
            for i in range(self.n_splits):
                if i not in test_groups:
                    block_indices = blocks[i]
                    # Apply Purge/Embargo if adjacent to test blocks
                    if (i - 1) in test_groups:
                        block_indices = block_indices[purge_size:] # Embargo right of test
                    if (i + 1) in test_groups:
                        block_indices = block_indices[:-purge_size] # Purge left of test
                        
                    train_indices.extend(block_indices)
            
            splits.append((np.array(train_indices), np.array(test_indices)))
            
        return splits
