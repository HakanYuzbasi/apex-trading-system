"""Support Vector Regression with RBF kernel."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR


class SVRRegressor(BaseEstimator, RegressorMixin):
    """SVR wrapper with automatic sub-sampling for large datasets.

    SVR scales O(n^2)-O(n^3), so we subsample when n > max_samples
    to keep training tractable while still capturing the support vectors.
    """

    def __init__(self, C: float = 1.0, epsilon: float = 0.01,
                 kernel: str = "rbf", gamma: str = "scale",
                 max_samples: int = 5000, random_state: int = 42):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.max_samples = max_samples
        self.random_state = random_state
        self._model = SVR(
            C=self.C, epsilon=self.epsilon,
            kernel=self.kernel, gamma=self.gamma,
        )
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        if len(X) > self.max_samples:
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        self._model.fit(X_sub, y_sub)
        # SVR doesn't provide feature importances natively; use uniform
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self

    def predict(self, X):
        return self._model.predict(X)
