"""Bayesian Ridge regression with uncertainty quantification."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


class BayesianRidgeRegressor(BaseEstimator, RegressorMixin):
    """Wrapper around sklearn BayesianRidge with internal standardization.

    Key fixes for production use:
    - Internal StandardScaler prevents coefficient explosion on unscaled features
    - Stronger default priors (1e-4) encourage regularization
    - Clipped predictions prevent extreme outliers from contaminating the ensemble
    """

    def __init__(self, n_iter: int = 300, alpha_1: float = 1e-4,
                 alpha_2: float = 1e-4, lambda_1: float = 1e-4,
                 lambda_2: float = 1e-4, clip_predictions: float = 0.15):
        self.n_iter = n_iter
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.clip_predictions = clip_predictions
        self._scaler = StandardScaler()
        self._model = BayesianRidge(
            n_iter=self.n_iter,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
        )
        self.prediction_std_ = None

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self.feature_importances_ = np.abs(self._model.coef_)
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
