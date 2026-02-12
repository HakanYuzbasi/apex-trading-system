"""Gaussian Process regression with uncertainty quantification."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


class GPRegressor(BaseEstimator, RegressorMixin):
    """GP regression wrapper with sub-sampling and internal scaling.

    Provides posterior uncertainty (std) on every prediction.
    Key fixes:
    - Internal StandardScaler for stable kernel optimization
    - Configurable alpha (noise regularization in GP fit) prevents overfitting
    - Adaptive length_scale_bounds let the optimizer find the right scale
    - Clipped predictions prevent outlier contamination
    """

    def __init__(self, max_samples: int = 2000, random_state: int = 42,
                 length_scale: float = 1.0, noise_level: float = 0.1,
                 alpha: float = 0.1, n_restarts: int = 3,
                 clip_predictions: float = 0.15):
        self.max_samples = max_samples
        self.random_state = random_state
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.clip_predictions = clip_predictions
        self.prediction_std_ = None
        self.feature_importances_ = None
        self._scaler = StandardScaler()

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            RBF(length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=self.noise_level,
                        noise_level_bounds=(1e-5, 1e1))
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
            normalize_y=True,
        )

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        if len(X) > self.max_samples:
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        X_scaled = self._scaler.fit_transform(X_sub)
        self._model.fit(X_scaled, y_sub)
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        pred, std = self._model.predict(X_scaled, return_std=True)
        self.prediction_std_ = std
        if self.clip_predictions > 0:
            pred = np.clip(pred, -self.clip_predictions, self.clip_predictions)
        return pred

    def predict_with_uncertainty(self, X):
        """Return (predictions, standard_deviations)."""
        X_scaled = self._scaler.transform(X)
        pred, std = self._model.predict(X_scaled, return_std=True)
        self.prediction_std_ = std
        if self.clip_predictions > 0:
            pred = np.clip(pred, -self.clip_predictions, self.clip_predictions)
        return pred, std
