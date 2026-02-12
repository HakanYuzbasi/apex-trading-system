"""CatBoost regressor with ordered boosting."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from catboost import CatBoostRegressor as _CatBoost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper around CatBoost with regime-specific defaults.

    CatBoost uses ordered boosting which inherently prevents target
    leakage — particularly valuable for time-series financial data.
    Falls back to GradientBoosting if catboost is not installed.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 3,
                 learning_rate: float = 0.03, l2_leaf_reg: float = 3.0,
                 subsample: float = 0.7, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_state = random_state
        self.feature_importances_ = None

        if CATBOOST_AVAILABLE:
            self._model = _CatBoost(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                subsample=self.subsample,
                random_seed=self.random_state,
                verbose=0,
                allow_writing_files=False,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )

    def fit(self, X, y):
        self._model.fit(X, y)
        if hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = self._model.feature_importances_
        else:
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self

    def predict(self, X):
        return self._model.predict(X)
