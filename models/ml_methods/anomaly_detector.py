"""Anomaly-aware regressor using Isolation Forest for input screening."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor


class AnomalyAwareRegressor(BaseEstimator, RegressorMixin):
    """Wraps a base regressor with an Isolation Forest anomaly detector.

    During inference, anomalous inputs are flagged and predictions are
    attenuated toward zero (conservative). This acts as an automatic
    circuit breaker for unusual market conditions.
    """

    def __init__(self, contamination: float = 0.05,
                 n_estimators_iso: int = 100,
                 n_estimators_base: int = 150,
                 max_depth: int = 3, learning_rate: float = 0.03,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators_iso = n_estimators_iso
        self.n_estimators_base = n_estimators_base
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._iso = IsolationForest(
            n_estimators=self.n_estimators_iso,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._base = GradientBoostingRegressor(
            n_estimators=self.n_estimators_base,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.feature_importances_ = None
        self.anomaly_scores_ = None

    def fit(self, X, y):
        self._iso.fit(X)
        self._base.fit(X, y)
        self.feature_importances_ = self._base.feature_importances_
        return self

    def predict(self, X):
        preds = self._base.predict(X)
        # Anomaly scores: -1 = anomaly, 1 = normal
        labels = self._iso.predict(X)
        # decision_function: lower = more anomalous
        scores = self._iso.decision_function(X)
        self.anomaly_scores_ = scores

        # Attenuate predictions for anomalous inputs
        for i in range(len(preds)):
            if labels[i] == -1:
                # Scale prediction toward zero based on anomaly severity
                attenuation = max(0.1, min(1.0, 0.5 + scores[i]))
                preds[i] *= attenuation

        return preds

    def is_anomalous(self, X) -> np.ndarray:
        """Return boolean mask: True for anomalous inputs."""
        return self._iso.predict(X) == -1
