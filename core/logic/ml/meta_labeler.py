from __future__ import annotations

import logging
import os
import shutil
import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "kalman_residual", 
    "bayesian_prob", 
    "vix_level", 
    "sector_concentration",
    "relative_volume",
    "market_impact"
]
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
                import concurrent.futures
                from config import ApexConfig
                timeout = getattr(ApexConfig, "MODEL_LOAD_TIMEOUT_SECONDS", 15.0)

                def _load_booster():
                    return lgb.Booster(model_file=self.model_path)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load_booster)
                    booster = future.result(timeout=timeout)

                if booster.num_feature() != N_FEATURES:
                    logger.warning(
                        "MetaLabeler: stale model has %d features, expected %d. Quarantining.",
                        booster.num_feature(), N_FEATURES,
                    )
                    shutil.move(self.model_path, self.model_path + ".v1.bak")
                    return
                self._model = booster
                self._is_trained = True
                self.is_bootstrapped_on_synthetic = False
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

    def predict_confidence(
        self, 
        kalman_residual: float = 0.0, 
        bayesian_prob: float = 0.5, 
        vix_level: float = 20.0, 
        sector_concentration: float = 0.0,
        relative_volume: float = 1.0,
        market_impact: float = 0.0,
        **kwargs
    ) -> float:
        """
        Returns the probability of success (0.0–1.0).

        Expected kwargs: kalman_residual, bayesian_prob, vix_level, sector_concentration.
        Returns 1.0 (pass-through) when the model is not yet trained.
        """
        if not self._is_trained or self._model is None:
            return 1.0

        features = np.array([[
            float(kalman_residual),
            float(bayesian_prob),
            float(vix_level),
            float(sector_concentration),
            float(relative_volume),
            float(market_impact),
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

        if X.shape[1] != N_FEATURES:
            logger.error("MetaLabeler: X has %d features, expected %d. Aborting train.", X.shape[1], N_FEATURES)
            return

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


    def auto_train_from_attribution(
        self,
        attribution_json_path: str = "data/performance_attribution.json",
        min_samples: int = 50,
    ) -> bool:
        """
        #6: Auto-train at startup from historical closed trades stored in
        performance_attribution.json.  Called right after __init__ when the
        model file is missing.  Returns True if training succeeded.

        Each closed trade already carries kalman_residual, bayesian_prob,
        vix_level, sector_concentration — no feature reconstruction needed.
        Label = 1 if net_pnl > 0, else 0.
        """
        if self._is_trained:
            return False  # already have a model — skip

        import json, os
        if not os.path.exists(attribution_json_path):
            logger.info(
                "MetaLabeler auto-train: attribution file not found at %s — skipping",
                attribution_json_path,
            )
            return False

        try:
            with open(attribution_json_path) as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("MetaLabeler auto-train: failed to read attribution file: %s", exc)
            return False

        closed = payload.get("closed_trades", [])
        if len(closed) < min_samples:
            logger.info(
                "MetaLabeler auto-train: only %d closed trades found (need ≥%d) — skipping",
                len(closed), min_samples,
            )
            return False

        rows_X, rows_y = [], []
        for t in closed:
            try:
                x_row = [
                    float(t.get("kalman_residual", 0.0) or 0.0),
                    float(t.get("bayesian_prob",   0.5) or 0.5),
                    float(t.get("vix_level",       20.0) or 20.0),
                    float(t.get("sector_concentration", 0.0) or 0.0),
                    float(t.get("relative_volume", 1.0) or 1.0),
                    float(t.get("market_impact", 0.0) or 0.0),
                ]
                label = 1 if float(t.get("net_pnl", 0.0) or 0.0) > 0 else 0
                rows_X.append(x_row)
                rows_y.append(label)
            except Exception:
                continue

        if len(rows_X) < min_samples:
            logger.info(
                "MetaLabeler auto-train: only %d usable rows after filtering — skipping",
                len(rows_X),
            )
            return False

        X = np.array(rows_X, dtype=float)
        y = np.array(rows_y, dtype=int)
        positive_rate = float(y.mean())
        logger.info(
            "MetaLabeler auto-train: %d samples, %.1f%% profitable — training now",
            len(y), positive_rate * 100,
        )
        self.train(X, y)
        self.is_bootstrapped_on_synthetic = False
        return True

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
