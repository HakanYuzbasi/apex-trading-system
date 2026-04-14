"""
quant_system/ml/meta_labeler.py - Meta-Labeling & Bet Sizing

Implements the "Meta-Labeling" technique from Lopez de Prado (Advances in Financial ML).
sklearn is imported lazily so the module can be loaded in containers that don't have it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelResult:
    """Result of meta-labeling prediction."""
    decision: bool
    probability: float
    bet_size: float


class MetaLabeler:
    """Meta-Labeling Model — separates direction from confidence/sizing."""

    def __init__(
        self,
        base_estimator: "Optional[BaseEstimator]" = None,
        meta_threshold: float = 0.6,
        use_proba_sizing: bool = True,
    ):
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._model = base_estimator or RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
        except ModuleNotFoundError:
            logger.warning("sklearn not available — MetaLabeler will pass all signals through at full size")
            self._model = None
        self.meta_threshold = meta_threshold
        self.use_proba_sizing = use_proba_sizing
        self.is_fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaLabeler":
        if self._model is None:
            self.is_fitted = False
            return self
        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> MetaLabelResult:
        if not self.is_fitted or self._model is None:
            return MetaLabelResult(decision=True, probability=1.0, bet_size=1.0)
        try:
            prob = float(self._model.predict_proba(X[self._feature_names])[:, 1][0])
            decision = prob >= self.meta_threshold
            bet_size = prob if self.use_proba_sizing else (1.0 if decision else 0.0)
            return MetaLabelResult(decision=decision, probability=prob, bet_size=bet_size)
        except Exception as exc:
            logger.warning("MetaLabeler.predict failed: %s — passing through", exc)
            return MetaLabelResult(decision=True, probability=1.0, bet_size=1.0)

    def predict_confidence(self, **kwargs) -> float:
        """Alias for predict() to maintain API compatibility."""
        res = self.predict(pd.DataFrame([kwargs]))
        return res.probability

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if not self.is_fitted or self._model is None:
            return {"accuracy": 0.0, "precision": 0.0, "auc": 0.0}
        try:
            from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
            preds = self._model.predict(X[self._feature_names])
            proba = self._model.predict_proba(X[self._feature_names])[:, 1]
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "precision": float(precision_score(y, preds, zero_division=0)),
                "auc": float(roc_auc_score(y, proba)),
            }
        except Exception as exc:
            logger.warning("MetaLabeler.score failed: %s", exc)
            return {"accuracy": 0.0, "precision": 0.0, "auc": 0.0}
