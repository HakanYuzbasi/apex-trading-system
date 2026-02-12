"""ElasticNet regressor with regime-specific regularization."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class ElasticNetRegressor(BaseEstimator, RegressorMixin):
    """Wrapper around sklearn ElasticNet with internal standardization.

    L1+L2 regularization provides built-in feature selection and
    shrinkage. Internal StandardScaler prevents coefficient explosion
    on features with very different scales.
    """

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5,
                 max_iter: int = 2000, random_state: int = 42,
                 clip_predictions: float = 0.15):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.clip_predictions = clip_predictions
        self._scaler = StandardScaler()
        self._model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self.feature_importances_ = np.abs(self._model.coef_)
        return self

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if self.clip_predictions > 0:
            pred = np.clip(pred, -self.clip_predictions, self.clip_predictions)
        return pred
