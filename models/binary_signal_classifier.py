"""
models/binary_signal_classifier.py — Binary Direction Classifier
=================================================================
Predicts next-bar price direction (up/down) using the same feature matrix
as the regression regressors in AdvancedSignalGenerator.

Signal = prob_up * 2 - 1  → range [-1, 1]

Usage:
    classifier = BinarySignalClassifier()
    # Training (called from AdvancedSignalGenerator.train_walk_forward):
    classifier.train(X_train, y_train_binary, X_test, y_test_binary, regime, asset_class)
    # Inference:
    signal, conf = classifier.predict(X_preprocessed, regime, asset_class)
"""

import os
import pickle
import logging
from typing import Dict, Optional, Tuple

import numpy as np

from config import ApexConfig
from core.symbols import AssetClass

logger = logging.getLogger(__name__)

# --- Optional ML imports (same pattern as advanced_signal_generator) ---
CLF_AVAILABLE = False
XGBOOST_CLF_AVAILABLE = False
LIGHTGBM_CLF_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from joblib import dump, load
    CLF_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available — BinarySignalClassifier disabled")

try:
    import xgboost as xgb
    XGBOOST_CLF_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_CLF_AVAILABLE = True
except ImportError:
    pass


_REGIMES = ("bull", "bear", "neutral", "volatile")
_ASSET_CLASSES = (
    AssetClass.EQUITY.value,
    AssetClass.FOREX.value,
    AssetClass.CRYPTO.value,
)


class BinarySignalClassifier:
    """
    4-model ensemble binary direction classifier.

    Trained alongside the regression ensemble.  Each
    (asset_class, regime) cell gets its own set of classifiers
    so the model learns regime-specific direction patterns.
    """

    def __init__(self, model_dir: str = "models/saved_advanced/binary"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # {asset_class: {regime: {model_name: clf}}}
        self.models: Dict[str, Dict[str, Dict]] = {
            ac: {r: {} for r in _REGIMES} for ac in _ASSET_CLASSES
        }

        self.is_trained: bool = False
        self._label_horizon: int = int(
            getattr(ApexConfig, "BINARY_LABEL_HORIZON_DAYS", 1)
        )

        self._load_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_labels(prices: "pd.Series", horizon: int = 1) -> "pd.Series":  # noqa: F821
        """Create binary direction labels: 1 = price higher in `horizon` bars."""
        import pandas as pd  # local to avoid hard dep at module level
        future = prices.shift(-horizon)
        return (future > prices).astype(int)

    def train(
        self,
        X_train: np.ndarray,
        y_train_binary: np.ndarray,
        X_test: np.ndarray,
        y_test_binary: np.ndarray,
        regime: str,
        asset_class: str,
    ) -> Dict:
        """
        Train classifier ensemble for a (regime, asset_class) cell.

        Returns dict with 'accuracy', 'n_train', 'n_test'.
        """
        if not CLF_AVAILABLE:
            return {}

        if len(np.unique(y_train_binary)) < 2:
            logger.debug(
                "BinaryClassifier: skipping %s/%s — only one class in training labels",
                asset_class, regime,
            )
            return {}

        clfs: Dict = {}
        probas = []

        # --- Random Forest ---
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )
            rf.fit(X_train, y_train_binary)
            clfs["rf"] = rf
            probas.append(rf.predict_proba(X_test)[:, 1])
        except Exception as exc:
            logger.debug("Binary RF train error: %s", exc)

        # --- Gradient Boosting ---
        try:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
            gb.fit(X_train, y_train_binary)
            clfs["gb"] = gb
            probas.append(gb.predict_proba(X_test)[:, 1])
        except Exception as exc:
            logger.debug("Binary GB train error: %s", exc)

        # --- XGBoost ---
        if XGBOOST_CLF_AVAILABLE:
            try:
                scale_pos_weight = (
                    float(np.sum(y_train_binary == 0))
                    / max(1, float(np.sum(y_train_binary == 1)))
                )
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.03,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
                xgb_clf.fit(X_train, y_train_binary)
                clfs["xgb"] = xgb_clf
                probas.append(xgb_clf.predict_proba(X_test)[:, 1])
            except Exception as exc:
                logger.debug("Binary XGB train error: %s", exc)

        # --- LightGBM ---
        if LIGHTGBM_CLF_AVAILABLE:
            try:
                lgb_clf = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.03,
                    random_state=42,
                    verbose=-1,
                    is_unbalance=True,
                )
                lgb_clf.fit(X_train, y_train_binary)
                clfs["lgb"] = lgb_clf
                probas.append(lgb_clf.predict_proba(X_test)[:, 1])
            except Exception as exc:
                logger.debug("Binary LGB train error: %s", exc)

        if not clfs:
            return {}

        # Store models
        if asset_class in self.models and regime in self.models[asset_class]:
            self.models[asset_class][regime] = clfs
        else:
            self.models.setdefault(asset_class, {})[regime] = clfs

        # Compute ensemble accuracy on test set
        accuracy = 0.0
        if probas:
            ensemble_proba = np.mean(probas, axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            accuracy = float(np.mean(ensemble_pred == y_test_binary))

        self.is_trained = True

        logger.info(
            "BinaryClassifier %s/%s: acc=%.3f n_train=%d n_test=%d",
            asset_class,
            regime,
            accuracy,
            len(y_train_binary),
            len(y_test_binary),
        )

        return {
            "accuracy": accuracy,
            "n_train": int(len(y_train_binary)),
            "n_test": int(len(y_test_binary)),
        }

    def predict(
        self,
        X: np.ndarray,
        regime: str,
        asset_class: str,
    ) -> Tuple[float, float]:
        """
        Return (binary_signal, confidence).

        binary_signal  = mean(prob_up) * 2 - 1  ∈ [-1, 1]
        confidence     = |binary_signal| clipped to [0, 1]
        """
        if not self.is_trained or not CLF_AVAILABLE:
            return 0.0, 0.0

        cell = (
            self.models.get(asset_class, {}).get(regime)
            or self.models.get(asset_class, {}).get("neutral")
            or {}
        )
        if not cell:
            return 0.0, 0.0

        probas = []
        for name in ("rf", "gb", "xgb", "lgb"):
            clf = cell.get(name)
            if clf is None:
                continue
            try:
                probas.append(clf.predict_proba(X)[:, 1])
            except Exception:
                pass

        if not probas:
            return 0.0, 0.0

        mean_proba = float(np.mean([p[0] for p in probas]))  # single sample
        binary_signal = float(np.clip(mean_proba * 2.0 - 1.0, -1.0, 1.0))
        confidence = float(min(abs(binary_signal) * 1.5, 1.0))  # stronger proba → more confident
        return binary_signal, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self) -> None:
        if not CLF_AVAILABLE:
            return
        try:
            for ac, regimes in self.models.items():
                for regime, clfs in regimes.items():
                    if not clfs:
                        continue
                    cell_dir = os.path.join(self.model_dir, ac, regime)
                    os.makedirs(cell_dir, exist_ok=True)
                    for name, clf in clfs.items():
                        path = os.path.join(cell_dir, f"{name}_clf.pkl")
                        try:
                            dump(clf, path)
                        except Exception as exc:
                            logger.debug("Binary save %s: %s", path, exc)

            meta = {"is_trained": self.is_trained, "label_horizon": self._label_horizon}
            with open(os.path.join(self.model_dir, "meta.pkl"), "wb") as fh:
                pickle.dump(meta, fh)

            logger.info("✅ BinarySignalClassifier models saved to %s", self.model_dir)
        except Exception as exc:
            logger.error("BinarySignalClassifier save error: %s", exc)

    def _load_models(self) -> bool:
        if not CLF_AVAILABLE:
            return False
        meta_path = os.path.join(self.model_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            self.is_trained = meta.get("is_trained", False)
            self._label_horizon = meta.get("label_horizon", 1)

            for ac in _ASSET_CLASSES:
                for regime in _REGIMES:
                    cell_dir = os.path.join(self.model_dir, ac, regime)
                    if not os.path.isdir(cell_dir):
                        continue
                    clfs: Dict = {}
                    for name in ("rf", "gb", "lgb"):
                        path = os.path.join(cell_dir, f"{name}_clf.pkl")
                        if os.path.exists(path):
                            try:
                                clfs[name] = load(path)
                            except Exception:
                                pass
                    xgb_path = os.path.join(cell_dir, "xgb_clf.pkl")
                    if os.path.exists(xgb_path) and XGBOOST_CLF_AVAILABLE:
                        try:
                            clfs["xgb"] = load(xgb_path)
                        except Exception:
                            pass
                    if clfs:
                        self.models[ac][regime] = clfs

            if self.is_trained:
                logger.info("BinarySignalClassifier loaded from %s", self.model_dir)
            return self.is_trained
        except Exception as exc:
            logger.debug("BinarySignalClassifier load error: %s", exc)
            return False
