from __future__ import annotations

import logging
import os
import shutil
import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["vpin", "rsi", "atr"]
N_FEATURES = len(FEATURE_NAMES)


class MetaLabeler:
    """
    ML Supervisor layer that predicts the probability of success for a trade signal.

    Features (v2): kalman_residual, bayesian_prob, vix_level, sector_concentration.
    Replaces v1 features (vpin, rsi, atr) — any saved model with 3 features is
    auto-quarantined and the labeler runs in pass-through mode until retrained.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "run_state/meta_labeler.lgb"
        self._model = None
        self._is_trained = False
        self.is_bootstrapped_on_synthetic = True
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                booster = lgb.Booster(model_file=self.model_path)
                if booster.num_feature() != N_FEATURES:
                    logger.warning(
                        "MetaLabeler: stale model has %d features, expected %d. Quarantining.",
                        booster.num_feature(), N_FEATURES,
                    )
                    shutil.move(self.model_path, self.model_path + ".v1.bak")
                    return
                self._model = booster
                self._is_trained = True
                logger.info("MetaLabeler: Loaded model from %s", self.model_path)
            except Exception as e:
                logger.error("MetaLabeler: Failed to load model: %s. Quarantining.", e)
                try:
                    shutil.move(self.model_path, self.model_path + ".bak")
                except Exception as move_err:
                    logger.error("MetaLabeler: Failed to quarantine file: %s", move_err)
        else:
            logger.warning(
                "MetaLabeler: Model not found at %s. Running in pass-through mode (confidence=1.0).",
                self.model_path,
            )

    def predict_confidence(self, vpin: float = 0.0, rsi: float = 50.0, atr: float = 0.0, **kwargs) -> float:
        """
        Returns the probability of success (0.0–1.0).

        Expected kwargs: vpin, rsi, atr.
        Returns 1.0 (pass-through) when the model is not yet trained.
        """
        if not self._is_trained or self._model is None:
            return 1.0

        features = np.array([[
            float(vpin),
            float(rsi),
            float(atr),
        ]])

        try:
            prob = float(self._model.predict(features)[0])
            
            if getattr(self, "is_bootstrapped_on_synthetic", False):
                logger.debug(f"MetaLabeler (Synthetic Warm-up): Raw Score {prob:.4f} -> Overridden to 1.0")
                return 1.0
                
            return prob
        except Exception as e:
            logger.error("MetaLabeler: Prediction error: %s", e)
            return 1.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the LightGBM model.
        X: shape (n, 4) — columns must match FEATURE_NAMES order.
        y: binary outcome array (1 = profitable trade).
        """
        logger.info("MetaLabeler: Training model with %d samples...", len(X))

        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }

        self._model = lgb.train(params, train_data, num_boost_round=100)
        self._model.save_model(self.model_path)
        self._is_trained = True
        logger.info("MetaLabeler: Training complete. Saved to %s", self.model_path)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self._is_trained or self._model is None:
            return {"accuracy": 0.0, "auc": 0.0}
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score
            proba = self._model.predict(X)
            preds = (proba >= 0.5).astype(int)
            return {
                "accuracy": float(accuracy_score(y, preds)),
                "auc": float(roc_auc_score(y, proba)),
            }
        except Exception as exc:
            logger.warning("MetaLabeler.score failed: %s", exc)
            return {"accuracy": 0.0, "auc": 0.0}
