"""
models/online_learner.py - Online Machine Learning & Drift Detection

Enables the system to adapt to changing market conditions in real-time
without requiring expensive full model retraining.

Features:
- Incremental learning (partial_fit)
- Concept drift detection (DDM - Drift Detection Method)
- Adaptive window sizing
- Model health monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
from datetime import datetime
import logging
import pickle
from dataclasses import dataclass

# Try to import sklearn components
try:
    from sklearn.linear_model import SGDRegressor, SGDClassifier
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class SGDRegressor: pass
    class SGDClassifier: pass
    class BaseEstimator: pass
    class StandardScaler: pass

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    """Status of concept drift detection."""
    detected: bool
    warning: bool
    drift_level: float
    error_rate: float
    sample_count: int


class DriftDetector:
    """
    Drift Detection Method (DDM).
    
    Monitors the error rate of the model. If the error rate increases
    significantly, it signals that the data distribution has changed (drift).
    """
    
    def __init__(self, min_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        """
        Initialize Drift Detector.
        
        Args:
            min_instances: Minimum samples before detection starts
            warning_level: Standard deviations for warning zone
            drift_level: Standard deviations for drift detection
        """
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        self.reset()
        
    def reset(self):
        """Reset detector statistics."""
        self.sample_count = 0
        self.errors = 0
        self.min_p_plus_s = float('inf')
        self.p_min = float('inf')
        self.s_min = float('inf')
        
    def update(self, error: bool) -> DriftStatus:
        """
        Update detector with new prediction result.
        
        Args:
            error: True if prediction was wrong, False if correct
            
        Returns:
            DriftStatus object
        """
        self.sample_count += 1
        
        if error:
            self.errors += 1
            
        # Error rate (probability of error)
        p = self.errors / self.sample_count
        
        # Standard deviation of error rate
        s = np.sqrt(p * (1 - p) / self.sample_count)
        
        # Track minimum statistics
        if p + s < self.min_p_plus_s:
            self.min_p_plus_s = p + s
            self.p_min = p
            self.s_min = s
            
        # Check for drift
        detected = False
        warning = False
        level = 0.0
        
        if self.sample_count >= self.min_instances:
            # Drift score = how many std devs away from min error
            drift_score = (p + s) - (self.p_min + self.warning_level * self.s_min)
            level = max(0, drift_score)
            
            if p + s > self.p_min + self.drift_level * self.s_min:
                detected = True
                level = 3.0 + (p + s - (self.p_min + self.drift_level * self.s_min))
            elif p + s > self.p_min + self.warning_level * self.s_min:
                warning = True
                level = 2.0 + (p + s - (self.p_min + self.warning_level * self.s_min))
                
        return DriftStatus(
            detected=detected,
            warning=warning,
            drift_level=level,
            error_rate=p,
            sample_count=self.sample_count
        )


class OnlineLearner:
    """
    Online learning model wrapper.
    
    Wraps standard models to provide:
    - Incremental updates
    - Drift monitoring
    - Auto-retraining triggers
    """
    
    def __init__(
        self,
        model_type: str = 'regressor',
        learning_rate: str = 'optimal',
        drift_sensitivity: float = 3.0
    ):
        """
        Initialize Online Learner.
        
        Args:
            model_type: 'regressor' or 'classifier'
            learning_rate: SGD learning rate schedule
            drift_sensitivity: Sensitivity for drift detection
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - OnlineLearner disabled")
            self.model = None
            return

        self.model_type = model_type
        self.drift_detector = DriftDetector(drift_level=drift_sensitivity)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if model_type == 'classifier':
            self.model = SGDClassifier(
                loss='log_loss',  # Logistic regression
                learning_rate=learning_rate,
                penalty='l2',
                alpha=0.0001,
                warm_start=True,
                random_state=42
            )
        else:
            self.model = SGDRegressor(
                loss='squared_error',
                learning_rate=learning_rate,
                penalty='l2',
                alpha=0.0001,
                warm_start=True,
                random_state=42
            )
            
        logger.info(f"OnlineLearner ({model_type}) initialized")
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """
        Update model with new batch of data.
        
        Args:
            X: Features matrix
            y: Target vector
            classes: Component classes (for classifier first fit)
        """
        if not SKLEARN_AVAILABLE:
            return
            
        # Update scaler
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Update model
        if self.model_type == 'classifier' and not self.is_fitted and classes is not None:
            self.model.partial_fit(X_scaled, y, classes=classes)
        else:
            self.model.partial_fit(X_scaled, y)
            
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            if self.model_type == 'classifier':
                return np.zeros(X.shape[0])
            else:
                return np.zeros(X.shape[0])
                
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def update_and_monitor(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Update model and check for drift based on prediction error.
        
        Args:
            X: Features
            y_true: True targets (received after prediction)
            
        Returns:
            Dict with drift status
        """
        if not self.is_fitted:
            self.partial_fit(X, y_true, classes=np.unique(y_true) if self.model_type == 'classifier' else None)
            return {'status': 'initialized', 'drift': False}
            
        # 1. Make prediction before update (to check current performance)
        y_pred = self.predict(X)
        
        # 2. Check for drift
        has_drift = False
        drift_info = None
        
        for i in range(len(y_true)):
            # Calculate error
            if self.model_type == 'classifier':
                is_error = (y_pred[i] != y_true[i])
            else:
                # For regression, check if error exceeds threshold (e.g., 1 std dev)
                # Simplified: treat large error as "incorrect"
                error_mag = abs(y_pred[i] - y_true[i])
                is_error = error_mag > 0.02  # 2% return diff threshold
                
            status = self.drift_detector.update(is_error)
            
            if status.detected:
                has_drift = True
                drift_info = status
                
        # 3. Update model with new data (learn from mistake)
        self.partial_fit(X, y_true)
        
        result = {
            'drift_detected': has_drift,
            'warning': drift_info.warning if drift_info else False,
            'error_rate': drift_info.error_rate if drift_info else 0.0
        }
        
        if has_drift:
            logger.warning(f"🚨 Concept drift detected! Error rate: {result['error_rate']:.1%}")
            # Reset detector to adapt to new regime
            self.drift_detector.reset()
            
        elif result['warning']:
            logger.info(f"⚠️ Drift warning level. Error rate: {result['error_rate']:.1%}")
            
        return result
    
    def save(self, path: str):
        """Save learner state."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.error(f"Failed to save OnlineLearner: {e}")
            
    @classmethod
    def load(cls, path: str) -> 'OnlineLearner':
        """Load learner state."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load OnlineLearner: {e}")
            return cls()
