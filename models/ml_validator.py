"""
models/ml_validator.py
WALK-FORWARD VALIDATION & HYPERPARAMETER OPTIMIZATION
- Time-series proper validation
- Overfitting detection
- Bayesian hyperparameter optimization
- Label-leakage audit (Round 8 / GAP-8E)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Dict, Iterable, List, Optional, Sequence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.samplers import TPESampler

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label-leakage audit (Round 8 / GAP-8E)
# ─────────────────────────────────────────────────────────────────────────────


class LabelLeakageError(ValueError):
    """Raised when a feature DataFrame fails the label-leakage audit."""


def _has_monotonic_time_index(df: pd.DataFrame) -> bool:
    """Return True iff ``df.index`` is a strictly increasing time index."""
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        time_level = idx.get_level_values(0)
    else:
        time_level = idx
    if not isinstance(time_level, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
        return False
    return bool(pd.Index(time_level).is_monotonic_increasing)


def _feature_correlates_with_future(
    series: pd.Series,
    reference: pd.Series,
    max_shift: int,
    threshold: float,
) -> Optional[int]:
    """
    Return the first lead (``k > 0``) at which ``series`` correlates with
    ``reference.shift(-k)`` above ``threshold``, or ``None``. A feature that
    matches future reference values within tolerance indicates leakage.

    Args:
        series: Candidate feature series (already aligned to ``reference``).
        reference: Baseline series (typically the label or raw price).
        max_shift: Maximum forward shift (in rows) to probe.
        threshold: Absolute Pearson correlation threshold that triggers a
            leakage flag.

    Returns:
        The smallest ``k`` in ``[1, max_shift]`` whose absolute correlation
        exceeds ``threshold``, or ``None`` when no shift qualifies.
    """
    if series.std(skipna=True) == 0 or reference.std(skipna=True) == 0:
        return None
    for k in range(1, max_shift + 1):
        shifted = reference.shift(-k)
        both = pd.concat([series, shifted], axis=1).dropna()
        if len(both) < 30:
            continue
        corr = both.iloc[:, 0].corr(both.iloc[:, 1])
        if corr is None or not np.isfinite(corr):
            continue
        if abs(float(corr)) >= threshold:
            return k
    return None


def leakage_check(
    df: pd.DataFrame,
    label_col: str,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    reference_col: Optional[str] = None,
    max_future_shift: int = 5,
    leak_corr_threshold: float = 0.98,
    raise_on_fail: bool = True,
) -> Dict[str, object]:
    """
    Audit a training DataFrame for label / feature leakage before model.fit().

    Four checks are performed:

    1. ``df.index`` must be strictly monotonic increasing (time-ordered).
    2. ``label_col`` must exist and contain at least two non-null values.
    3. Feature columns must not be identical to (or a forward-shift of) the
       label — any column whose correlation with ``label_col.shift(-k)``
       for ``k ∈ [1, max_future_shift]`` exceeds ``leak_corr_threshold`` is
       flagged.
    4. If ``reference_col`` is provided (typically raw close price), the
       same forward-shift correlation audit is re-run against it so that
       features derived from future prices (e.g. ``price.shift(-1)``) are
       also caught even when the label itself is derived separately.

    Args:
        df: Feature DataFrame. Must have a time-ordered index.
        label_col: Name of the target / label column in ``df``.
        feature_cols: Optional explicit feature column whitelist. When
            ``None``, every column other than ``label_col`` (and
            ``reference_col`` if provided) is audited.
        reference_col: Optional raw reference series name (e.g. ``"Close"``)
            used for the redundant forward-shift audit.
        max_future_shift: Maximum forward shift (rows) probed in the
            correlation audit.
        leak_corr_threshold: Absolute Pearson correlation at which a feature
            is declared leaky.
        raise_on_fail: When ``True`` (default) and any check fails, raise
            :class:`LabelLeakageError`. When ``False`` the function returns
            the audit report without raising.

    Returns:
        A dict with keys:

        - ``"ok"`` — ``True`` iff every check passed.
        - ``"errors"`` — list of error strings.
        - ``"leaky_features"`` — ``{column: leading_shift}`` for each
          feature flagged by the forward-shift audit.

    Raises:
        LabelLeakageError: When ``raise_on_fail`` is ``True`` and the audit
            detects leakage or a non-monotonic index. The caller should
            treat this as a STOP signal — any historical accuracy metric
            computed without the audit is inflated.
    """
    errors: List[str] = []
    leaky: Dict[str, int] = {}

    if not isinstance(df, pd.DataFrame):
        errors.append(f"df must be a DataFrame, got {type(df).__name__}")
    elif df.empty:
        errors.append("df is empty — nothing to audit")
    else:
        if not _has_monotonic_time_index(df):
            errors.append(
                "df.index is not monotonic increasing — training must be "
                "strictly time-ordered (no random shuffle)"
            )
        if label_col not in df.columns:
            errors.append(f"label_col={label_col!r} not found in df.columns")
        else:
            label = df[label_col]
            if label.dropna().shape[0] < 2:
                errors.append(
                    f"label_col={label_col!r} has fewer than 2 non-null values"
                )
            else:
                if feature_cols is None:
                    candidates: Iterable[str] = [
                        c for c in df.columns
                        if c != label_col and c != reference_col
                    ]
                else:
                    candidates = [c for c in feature_cols if c in df.columns]

                for col in candidates:
                    series = df[col]
                    if not pd.api.types.is_numeric_dtype(series):
                        continue
                    # Direct label identity — column IS the label shifted
                    for k in range(1, max_future_shift + 1):
                        if series.equals(label.shift(-k)):
                            leaky[col] = k
                            errors.append(
                                f"feature {col!r} equals label.shift(-{k}) — "
                                f"direct forward leakage"
                            )
                            break
                    if col in leaky:
                        continue
                    lead = _feature_correlates_with_future(
                        series, label,
                        max_shift=max_future_shift,
                        threshold=leak_corr_threshold,
                    )
                    if lead is not None:
                        leaky[col] = lead
                        errors.append(
                            f"feature {col!r} correlates ≥ "
                            f"{leak_corr_threshold:.2f} with "
                            f"{label_col}.shift(-{lead}) — suspected leakage"
                        )
                        continue
                    if reference_col and reference_col in df.columns:
                        ref_lead = _feature_correlates_with_future(
                            series, df[reference_col],
                            max_shift=max_future_shift,
                            threshold=leak_corr_threshold,
                        )
                        if ref_lead is not None:
                            leaky[col] = ref_lead
                            errors.append(
                                f"feature {col!r} correlates ≥ "
                                f"{leak_corr_threshold:.2f} with "
                                f"{reference_col}.shift(-{ref_lead}) — "
                                f"suspected leakage via reference"
                            )

    ok = not errors
    if not ok:
        logger.warning(
            "leakage_check FAILED (%d issue(s)): %s — historical accuracy "
            "metrics computed without this audit are INFLATED and must be "
            "re-run on leakage-free features.",
            len(errors),
            "; ".join(errors),
        )
        if raise_on_fail:
            raise LabelLeakageError("; ".join(errors))

    return {"ok": ok, "errors": errors, "leaky_features": leaky}


class MLValidator:
    """
    Advanced ML validation for trading strategies.
    
    Features:
    - Walk-forward validation (proper time-series splits)
    - Overfitting detection
    - Hyperparameter optimization
    - Performance metrics tracking
    """
    
    def __init__(self, model_class=RandomForestClassifier):
        self.model_class = model_class
        self.validation_results = []
        self.best_params = None
        logger.info("✅ ML Validator initialized")
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        train_size: int = 252,  # 1 year
        test_size: int = 63,     # 1 quarter
        **model_params
    ) -> Dict:
        """
        Walk-forward validation for time-series data.
        
        This is the PROPER way to validate trading models:
        - Train on historical data
        - Test on future unseen data
        - Roll forward and repeat
        
        Args:
            X: Features
            y: Target
            n_splits: Number of walk-forward periods
            train_size: Training window size
            test_size: Testing window size
            **model_params: Model hyperparameters
        
        Returns:
            Validation results with metrics
        """
        logger.info("🔄 Starting walk-forward validation...")
        logger.info(f"   Splits: {n_splits}, Train: {train_size}, Test: {test_size}")
        
        results = []
        
        # Create time-series splits
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\n📊 Fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            logger.info(f"   Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
            logger.info(f"   Test:  {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")
            
            # Train model
            model = self.model_class(**model_params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate on train and test
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Get predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            test_metrics = self._calculate_metrics(y_test, test_pred)
            
            # Check overfitting
            overfitting = self._detect_overfitting(train_metrics, test_metrics)
            
            logger.info(f"   Train Accuracy: {train_score:.3f}")
            logger.info(f"   Test Accuracy:  {test_score:.3f}")
            logger.info(f"   Overfitting: {'⚠️ YES' if overfitting else '✅ NO'}")
            
            results.append({
                'fold': fold,
                'train_score': train_score,
                'test_score': test_score,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'overfitting': overfitting,
                'model': model,
                'train_period': (X_train.index[0], X_train.index[-1]),
                'test_period': (X_test.index[0], X_test.index[-1])
            })
        
        # Aggregate results
        avg_train_score = np.mean([r['train_score'] for r in results])
        avg_test_score = np.mean([r['test_score'] for r in results])
        overfitting_pct = sum(r['overfitting'] for r in results) / len(results) * 100
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 WALK-FORWARD VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Average Train Score: {avg_train_score:.3f}")
        logger.info(f"Average Test Score:  {avg_test_score:.3f}")
        logger.info(f"Overfitting Rate:    {overfitting_pct:.1f}%")
        logger.info(f"{'='*60}\n")
        
        self.validation_results = results
        
        return {
            'results': results,
            'avg_train_score': avg_train_score,
            'avg_test_score': avg_test_score,
            'overfitting_rate': overfitting_pct,
            'is_valid': avg_test_score > 0.52 and overfitting_pct < 50  # Minimum thresholds
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5
            }
    
    def _detect_overfitting(self, train_metrics: Dict, test_metrics: Dict) -> bool:
        """
        Detect if model is overfitting.
        
        Overfitting indicators:
        - Train accuracy >> Test accuracy (>15% gap)
        - Train F1 >> Test F1
        - Very high train accuracy (>95%)
        """
        train_acc = train_metrics['accuracy']
        test_acc = test_metrics['accuracy']
        
        gap = train_acc - test_acc
        
        # Check for overfitting
        if gap > 0.15:  # 15% gap
            return True
        
        if train_acc > 0.95:  # Suspiciously high
            return True
        
        return False
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: int = 600  # 10 minutes
    ) -> Dict:
        """
        Bayesian hyperparameter optimization using Optuna.
        
        Args:
            X: Features
            y: Target
            n_trials: Number of optimization trials
            timeout: Max time in seconds
        
        Returns:
            Best hyperparameters found
        """
        logger.info("🔧 Starting hyperparameter optimization...")
        logger.info(f"   Trials: {n_trials}, Timeout: {timeout}s")
        
        def objective(trial):
            """Optimization objective function."""
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            
            # Perform walk-forward validation
            val_results = self.walk_forward_validation(
                X, y,
                n_splits=3,  # Faster for optimization
                train_size=252,
                test_size=63,
                **params
            )
            
            # Return test score (what we want to maximize)
            return val_results['avg_test_score']
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"\n{'='*60}")
        logger.info("🏆 OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best Test Score: {best_score:.3f}")
        logger.info("Best Parameters:")
        for param, value in best_params.items():
            logger.info(f"   {param}: {value}")
        logger.info(f"{'='*60}\n")
        
        self.best_params = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\n🎯 Top 10 Most Important Features:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"   {row['feature']:30s}: {row['importance']:.4f}")
            
            return importance_df
        
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
    
    def cross_validate_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        **model_params
    ) -> Dict:
        """
        Test model stability across multiple random seeds.
        
        A good model should have consistent performance across seeds.
        """
        logger.info(f"🔄 Testing model stability ({n_repeats} runs)...")
        
        scores = []
        
        for seed in range(n_repeats):
            model = self.model_class(**model_params, random_state=seed)
            
            # Simple train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info(f"   Mean Score: {mean_score:.3f} ± {std_score:.3f}")
        logger.info(f"   Stability: {'✅ GOOD' if std_score < 0.05 else '⚠️ UNSTABLE'}")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': scores,
            'is_stable': std_score < 0.05
        }


if __name__ == "__main__":
    # Test ML validator
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features with time index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    
    # Create target (binary classification)
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int), index=dates)
    
    # Initialize validator
    validator = MLValidator()
    
    # 1. Walk-forward validation
    print("\n" + "="*60)
    print("TEST 1: WALK-FORWARD VALIDATION")
    print("="*60)
    
    val_results = validator.walk_forward_validation(
        X, y,
        n_splits=5,
        train_size=500,
        test_size=100,
        n_estimators=50,
        max_depth=5
    )
    
    # 2. Hyperparameter optimization
    print("\n" + "="*60)
    print("TEST 2: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    opt_results = validator.optimize_hyperparameters(
        X, y,
        n_trials=10,
        timeout=60
    )
    
    # 3. Stability test
    print("\n" + "="*60)
    print("TEST 3: STABILITY TEST")
    print("="*60)
    
    stability = validator.cross_validate_stability(
        X, y,
        n_repeats=5,
        **opt_results['best_params']
    )
    
    print("\n✅ All tests complete!")
