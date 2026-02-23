"""
models/hyperparameter_tuner.py - Hyperparameter Optimization with Optuna

Automated hyperparameter tuning for ML models using:
- Optuna for Bayesian optimization
- Walk-forward validation for time series
- Multi-objective optimization (Sharpe + Win Rate)
- Pruning of unpromising trials

Usage:
    tuner = HyperparameterTuner(X_train, y_train, X_val, y_val)
    best_params = tuner.optimize(n_trials=100)
    model = tuner.get_best_model()
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("optuna not installed. Install with: pip install optuna")

# Check for ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    n_trials: int
    study_name: str
    optimization_history: List[Dict]
    duration_seconds: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'study_name': self.study_name,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp.isoformat()
        }


class HyperparameterTuner:
    """
    Automated hyperparameter tuning using Optuna.

    Supports:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM

    Features:
    - Bayesian optimization (TPE sampler)
    - Early stopping/pruning
    - Walk-forward cross-validation
    - Multi-objective optimization
    """

    # Default parameter search spaces
    PARAM_SPACES = {
        'random_forest': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 15),
            'min_samples_split': ('int', 10, 100),
            'min_samples_leaf': ('int', 5, 50),
            'max_features': ('categorical', ['sqrt', 'log2', 0.5, 0.7])
        },
        'gradient_boosting': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 10),
            'learning_rate': ('float_log', 0.01, 0.3),
            'min_samples_split': ('int', 10, 100),
            'min_samples_leaf': ('int', 5, 50),
            'subsample': ('float', 0.6, 1.0)
        },
        'xgboost': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float_log', 0.01, 0.3),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.5, 1.0),
            'reg_alpha': ('float_log', 1e-8, 10.0),
            'reg_lambda': ('float_log', 1e-8, 10.0),
            'min_child_weight': ('int', 1, 10)
        },
        'lightgbm': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float_log', 0.01, 0.3),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.5, 1.0),
            'reg_alpha': ('float_log', 1e-8, 10.0),
            'reg_lambda': ('float_log', 1e-8, 10.0),
            'min_child_samples': ('int', 5, 100),
            'num_leaves': ('int', 20, 150)
        }
    }

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        model_type: str = 'xgboost',
        scoring: str = 'sharpe',
        cv_folds: int = 5,
        results_dir: Path = Path('models/tuning')
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            X_train: Training features
            y_train: Training targets (returns)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_type: Type of model to tune
            scoring: Scoring metric ('sharpe', 'mse', 'directional')
            cv_folds: Number of CV folds for walk-forward validation
            results_dir: Directory to save results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train[-len(X_train)//5:]
        self.y_val = y_val if y_val is not None else y_train[-len(y_train)//5:]
        self.model_type = model_type
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.results_dir = results_dir

        # Results
        self.study: Optional[optuna.Study] = None
        self.best_model: Any = None
        self.tuning_history: List[TuningResult] = []

        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🔧 Hyperparameter Tuner initialized")
        logger.info(f"   Model: {model_type}")
        logger.info(f"   Scoring: {scoring}")
        logger.info(f"   Train samples: {len(X_train)}, Val samples: {len(self.X_val)}")

    def _suggest_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial."""
        param_space = self.PARAM_SPACES.get(model_type, {})
        params = {}

        for param_name, (param_type, *args) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif param_type == 'float_log':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1], log=True)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])

        return params

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create model with given parameters."""
        if self.model_type == 'random_forest':
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn not available")
            return RandomForestRegressor(**params, n_jobs=-1, random_state=42)

        elif self.model_type == 'gradient_boosting':
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn not available")
            return GradientBoostingRegressor(**params, random_state=42)

        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost not available")
            return xgb.XGBRegressor(**params, random_state=42, verbosity=0)

        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("lightgbm not available")
            return lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _calculate_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate score based on scoring metric."""
        if self.scoring == 'sharpe':
            # Treat predictions as signals, calculate Sharpe of signal * actual return
            signal_returns = np.sign(y_pred) * y_true
            if signal_returns.std() == 0:
                return 0.0
            return np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252)

        elif self.scoring == 'directional':
            # Directional accuracy
            correct = np.sum((y_pred > 0) == (y_true > 0))
            return correct / len(y_true)

        elif self.scoring == 'mse':
            # Negative MSE (Optuna maximizes)
            return -mean_squared_error(y_true, y_pred)

        elif self.scoring == 'r2':
            return r2_score(y_true, y_pred)

        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

    def _walk_forward_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[float]:
        """
        Walk-forward cross-validation for time series.

        Splits data into sequential folds, always training on past
        and validating on future data.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.cv_folds + 1)
        scores = []

        for i in range(self.cv_folds):
            # Train on data up to fold i
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = val_start + fold_size

            if val_end > n_samples:
                break

            X_fold_train = X[:train_end]
            y_fold_train = y[:train_end]
            X_fold_val = X[val_start:val_end]
            y_fold_val = y[val_start:val_end]

            try:
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                score = self._calculate_score(y_fold_val, y_pred)
                scores.append(score)
            except Exception as e:
                logger.debug(f"CV fold {i} failed: {e}")
                scores.append(-999)

        return scores

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Suggest parameters
        params = self._suggest_params(trial, self.model_type)

        # Create model
        model = self._create_model(params)

        # Walk-forward CV
        cv_scores = self._walk_forward_cv(model, self.X_train, self.y_train)

        if not cv_scores or all(s == -999 for s in cv_scores):
            return float('-inf')

        # Mean CV score
        mean_cv_score = np.mean([s for s in cv_scores if s != -999])

        # Also validate on held-out validation set
        try:
            model.fit(self.X_train, self.y_train)
            y_val_pred = model.predict(self.X_val)
            val_score = self._calculate_score(self.y_val, y_val_pred)
        except Exception:
            val_score = mean_cv_score

        # Combined score (weighted average)
        final_score = 0.7 * mean_cv_score + 0.3 * val_score

        # Report intermediate value for pruning
        trial.report(final_score, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return final_score

    def optimize(
        self,
        n_trials: int = 100,
        timeout: int = None,
        study_name: str = None,
        show_progress: bool = True
    ) -> TuningResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            study_name: Name for the study
            show_progress: Show progress bar

        Returns:
            TuningResult with best parameters and model
        """
        start_time = datetime.now()
        study_name = study_name or f"{self.model_type}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info("🔍 Starting hyperparameter optimization")
        logger.info(f"   Trials: {n_trials}")
        logger.info(f"   Study: {study_name}")

        # Create study with TPE sampler and median pruner
        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        )

        # Suppress Optuna logging if not showing progress
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            gc_after_trial=True
        )

        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info("✅ Optimization complete")
        logger.info(f"   Best score: {best_score:.4f}")
        logger.info(f"   Best params: {best_params}")

        # Train final model with best parameters
        self.best_model = self._create_model(best_params)
        self.best_model.fit(self.X_train, self.y_train)

        # Build optimization history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'score': trial.value
                })

        duration = (datetime.now() - start_time).total_seconds()

        # Create result
        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_model=self.best_model,
            n_trials=len(self.study.trials),
            study_name=study_name,
            optimization_history=history,
            duration_seconds=duration,
            timestamp=start_time
        )

        # Save result
        self._save_result(result)
        self.tuning_history.append(result)

        return result

    def _save_result(self, result: TuningResult):
        """Save tuning result to disk."""
        try:
            # Save metadata
            result_file = self.results_dir / f"{result.study_name}.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            # Save history plot if possible
            try:
                fig = optuna.visualization.plot_optimization_history(self.study)
                fig.write_html(str(self.results_dir / f"{result.study_name}_history.html"))
            except Exception:
                pass

            # Save parameter importance
            try:
                fig = optuna.visualization.plot_param_importances(self.study)
                fig.write_html(str(self.results_dir / f"{result.study_name}_importance.html"))
            except Exception:
                pass

            logger.info(f"Saved tuning results to {result_file}")

        except Exception as e:
            logger.error(f"Failed to save tuning result: {e}")

    def get_best_model(self) -> Any:
        """Get the best model from optimization."""
        if self.best_model is None:
            raise ValueError("No model available. Run optimize() first.")
        return self.best_model

    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance from the study."""
        if self.study is None:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception as e:
            logger.error(f"Failed to get param importance: {e}")
            return {}

    def get_status(self) -> Dict:
        """Get tuner status for dashboard."""
        return {
            'model_type': self.model_type,
            'scoring': self.scoring,
            'n_completed_studies': len(self.tuning_history),
            'best_score': self.study.best_value if self.study else None,
            'best_params': self.study.best_params if self.study else None,
            'optuna_available': OPTUNA_AVAILABLE
        }


def tune_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials_per_model: int = 50
) -> Dict[str, TuningResult]:
    """
    Tune all available model types and return best of each.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials_per_model: Trials per model type

    Returns:
        Dict of {model_type: TuningResult}
    """
    results = {}
    model_types = []

    if SKLEARN_AVAILABLE:
        model_types.extend(['random_forest', 'gradient_boosting'])
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')

    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Tuning {model_type}...")
        logger.info(f"{'='*50}")

        try:
            tuner = HyperparameterTuner(
                X_train, y_train, X_val, y_val,
                model_type=model_type,
                scoring='sharpe'
            )
            result = tuner.optimize(n_trials=n_trials_per_model)
            results[model_type] = result

        except Exception as e:
            logger.error(f"Failed to tune {model_type}: {e}")

    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info("TUNING SUMMARY")
    logger.info(f"{'='*50}")
    for model_type, result in sorted(results.items(), key=lambda x: x[1].best_score, reverse=True):
        logger.info(f"{model_type}: {result.best_score:.4f}")

    return results
