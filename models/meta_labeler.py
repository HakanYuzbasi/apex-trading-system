"""
models/meta_labeler.py - Meta-Labeling & Bet Sizing

Implements the "Meta-Labeling" technique from Lopez de Prado (Advances in Financial Machine Learning).

Concept:
1. Primary Model: Determines the SIDE (Long/Short) - High recall, low precision.
2. Secondary Model (Meta): Determines the SIZE (Bet/No Bet) - Filters false positives.

This separates the problem of "direction" from "confidence/sizing".
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelResult:
    """Result of meta-labeling prediction."""
    decision: bool  # True = Take trade, False = Pass
    probability: float  # Probability of success
    bet_size: float  # Recommended size multiplier (0.0 to 1.0)


class MetaLabeler:
    """
    Meta-Labeling Model.
    
    Attributes:
        primary_threshold: Threshold for primary model to trigger a "potential trade".
        meta_threshold: Probability threshold for meta model to confirm the trade.
    """
    
    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        meta_threshold: float = 0.6,
        use_proba_sizing: bool = True
    ):
        """
        Initialize Meta-Labeler.
        
        Args:
            base_estimator: Classifier for the meta-model (default: RandomForest)
            meta_threshold: Minimum probability to take the trade
            use_proba_sizing: If True, uses probability curve for specific bet sizing
        """
        self.learner = base_estimator if base_estimator else RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        self.meta_threshold = meta_threshold
        self.use_proba_sizing = use_proba_sizing
        self.is_fitted = False
        
        logger.info("MetaLabeler initialized")
        
    def generate_labels(
        self,
        primary_signals: pd.Series,
        future_returns: pd.Series,
        expiration_horizon: int = 5
    ) -> pd.Series:
        """
        Generate target labels for the meta-model (1 if primary was correct, 0 otherwise).
        
        Args:
            primary_signals: Series of -1, 0, 1 from primary model
            future_returns: Series of realized returns over horizon
            expiration_horizon: (unused in simplified logic, inferred from returns)
            
        Returns:
            pd.Series: 1 (Trade profitable), 0 (Trade unprofitable/flat)
        """
        # Align indices
        common_idx = primary_signals.index.intersection(future_returns.index)
        signals = primary_signals.loc[common_idx]
        returns = future_returns.loc[common_idx]
        
        # Where did primary model get it right?
        # Signal 1 and Return > 0 -> Correct
        # Signal -1 and Return < 0 -> Correct
        # Signal 0 -> Ignored (no event)
        
        # Sign of return matches signal?
        correct_direction = np.sign(signals) == np.sign(returns)
        
        # Only consider instances where primary signal was non-zero
        meta_labels = correct_direction.astype(int)
        
        # If signal was 0, label is irrelevant (we filter these out during training)
        meta_labels[signals == 0] = 0
        
        return meta_labels

    def train(
        self,
        X_features: pd.DataFrame,
        primary_signals: pd.Series,
        future_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Train the meta-model.
        
        Args:
            X_features: Feature matrix (Volatility, Momentum, etc.)
            primary_signals: Output from primary strategy (-1, 0, 1)
            future_returns: Realized returns
            
        Returns:
            Metrics dict
        """
        # 1. Generate Meta-Labels
        y_meta = self.generate_labels(primary_signals, future_returns)
        
        # 2. Filter: We only train on times when Primary Model had an opinion (signal != 0)
        # Because Meta-model behaves as a filter for the Primary model
        mask = primary_signals != 0
        
        if mask.sum() < 50:
            logger.warning("Insufficient primary signals to train meta-model")
            return {}
            
        X_train = X_features.loc[mask]
        y_train = y_meta.loc[mask]
        
        # Align X with y (intersection)
        common = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common]
        y_train = y_train.loc[common]
        
        # 3. Fit Model
        self.learner.fit(X_train, y_train)
        self.is_fitted = True
        
        # 4. Calc In-Sample Metrics
        preds = self.learner.predict(X_train)
        probs = self.learner.predict_proba(X_train)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_train, preds),
            'precision': precision_score(y_train, preds),
            'auc': roc_auc_score(y_train, probs)
        }
        
        logger.info(f"Meta-Model Trained: AUC={metrics['auc']:.3f}, Precision={metrics['precision']:.3f}")
        return metrics
        
    def predict(self, features: pd.DataFrame) -> MetaLabelResult:
        """
        Evaluate a potential trade.
        
        Args:
            features: Feature vector for current timestamp (DataFrame 1 row)
            
        Returns:
            MetaLabelResult
        """
        if not self.is_fitted:
            # Pass-through if not fitted (neutral)
            return MetaLabelResult(decision=True, probability=0.5, bet_size=1.0)
            
        # Predict probability that Primary signal is CORRECT
        try:
            prob_success = self.learner.predict_proba(features)[0, 1]
        except:
            prob_success = 0.5
            
        # Decision
        decision = prob_success >= self.meta_threshold
        
        # Bet Sizing (Discretized sigmoid-like or linear scaling above threshold)
        bet_size = 0.0
        if decision:
            if self.use_proba_sizing:
                # Scale from threshold (min size) to 1.0 (max size)
                # e.g., if thresh=0.6, prob=0.8 -> size ~ 0.5?
                # De Prado suggests utilizing CDF of probabilities, but simpler:
                bet_size = (prob_success - self.meta_threshold) / (1.0 - self.meta_threshold)
                bet_size = max(0.1, min(1.0, bet_size * 2)) # Boost curve
            else:
                bet_size = 1.0
        
        return MetaLabelResult(
            decision=decision,
            probability=prob_success,
            bet_size=bet_size
        )
