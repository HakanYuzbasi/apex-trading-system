"""Stacking meta-learner that trains on out-of-fold base model predictions."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False


class StackingMetaLearner(BaseEstimator, RegressorMixin):
    """Two-level stacking: base tree models → Ridge meta-learner.

    Uses internal cross-validation to generate out-of-fold predictions
    for the meta-learner, preventing information leakage.
    """

    def __init__(self, n_folds: int = 3, meta_alpha: float = 1.0,
                 n_estimators: int = 100, max_depth: int = 3,
                 random_state: int = 42):
        self.n_folds = n_folds
        self.meta_alpha = meta_alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self._base_models = []
        self._meta = Ridge(alpha=self.meta_alpha)
        self.feature_importances_ = None

    def _make_base_models(self):
        models = [
            ("rf", RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.random_state, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.03, random_state=self.random_state)),
        ]
        if _XGB:
            models.append(("xgb", xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.03, random_state=self.random_state,
                verbosity=0)))
        if _LGB:
            models.append(("lgb", lgb.LGBMRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.03, random_state=self.random_state,
                verbose=-1)))
        return models

    def fit(self, X, y):
        n = len(X)
        base_models_list = self._make_base_models()
        n_base = len(base_models_list)

        # Generate out-of-fold predictions
        oof_preds = np.zeros((n, n_base))
        fold_size = n // self.n_folds

        # Store fitted base models (re-fit on full data at end)
        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n
            mask = np.ones(n, dtype=bool)
            mask[start:end] = False

            X_train, y_train = X[mask], y[mask]
            X_val = X[~mask]

            for j, (name, model_template) in enumerate(base_models_list):
                from sklearn.base import clone
                model = clone(model_template)
                model.fit(X_train, y_train)
                oof_preds[~mask, j] = model.predict(X_val)

        # Train meta-learner on OOF predictions
        self._meta.fit(oof_preds, y)

        # Re-fit base models on full training data for inference
        self._base_models = []
        for name, model_template in base_models_list:
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X, y)
            self._base_models.append((name, model))

        # Feature importances: average across base models
        importances = []
        for name, model in self._base_models:
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)
        else:
            self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]

        return self

    def predict(self, X):
        base_preds = np.column_stack([
            model.predict(X) for _, model in self._base_models
        ])
        return self._meta.predict(base_preds)
