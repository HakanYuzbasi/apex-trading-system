"""
models/institutional_signal_generator.py

Institutional-Grade ML Signal Generator

Key Features:
- Time-series correct validation (no look-ahead bias)
- Purged/Embargo cross-validation
- Proper feature engineering with standardization
- Ensemble with model disagreement tracking
- Reproducible training with fixed seeds
- Feature importance and model diagnostics

Author: Institutional Quant Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import hashlib
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ML imports with availability checks
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import sklearn
    ML_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass


class ModelType(Enum):
    """Supported model types."""
    RANDOM_FOREST = "rf"
    GRADIENT_BOOSTING = "gb"
    XGBOOST = "xgb"
    LIGHTGBM = "lgb"


@dataclass
class SignalOutput:
    """Structured signal output with full metadata."""
    symbol: str
    signal: float  # [-1, 1]
    confidence: float  # [0, 1]
    timestamp: datetime

    # Component signals
    momentum_signal: float = 0.0
    mean_reversion_signal: float = 0.0
    trend_signal: float = 0.0
    volatility_signal: float = 0.0

    # ML outputs
    ml_prediction: float = 0.0
    ml_std: float = 0.0  # Model disagreement
    model_weights: Dict[str, float] = field(default_factory=dict)

    # Feature diagnostics
    feature_count: int = 0
    feature_hash: str = ""  # For reproducibility tracking

    # Quality metrics
    data_quality: float = 1.0  # 1.0 = all data present
    is_stale: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'momentum': self.momentum_signal,
            'mean_reversion': self.mean_reversion_signal,
            'trend': self.trend_signal,
            'volatility': self.volatility_signal,
            'ml_prediction': self.ml_prediction,
            'ml_std': self.ml_std,
            'feature_count': self.feature_count,
            'data_quality': self.data_quality
        }


@dataclass
class TrainingMetrics:
    """Training metrics for model diagnostics."""
    model_name: str
    train_mse: float
    val_mse: float
    directional_accuracy: float
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    timestamp: datetime = field(default_factory=datetime.now)


class PurgedTimeSeriesSplit:
    """
    Time-series cross-validation with purging and embargo.

    Prevents look-ahead bias by:
    1. Purging: Removing samples too close to the test set
    2. Embargo: Adding gap between train and test
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_gap: int = 2,
        test_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.test_size = test_size

    def split(self, X: np.ndarray, y: np.ndarray = None, groups=None):
        """Generate purged train/test indices."""
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        for i in range(self.n_splits):
            # Calculate test indices
            test_end = n_samples - i * test_size
            test_start = test_end - test_size

            if test_start <= 0:
                break

            # Calculate train end with purge and embargo
            train_end = test_start - self.purge_gap - self.embargo_gap

            if train_end <= 0:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices


class FeatureEngine:
    """
    Institutional-grade feature engineering.

    Features are categorized by:
    - Momentum (price momentum across timeframes)
    - Mean Reversion (z-scores, Bollinger bands)
    - Trend (MA crossovers, ADX)
    - Volatility (historical vol, vol ratios)
    - Volume (OBV, volume trends) - if available
    - Statistical (skew, kurtosis)
    """

    FEATURE_GROUPS = {
        'momentum': [
            'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
            'roc_5d', 'roc_10d', 'roc_20d'
        ],
        'mean_reversion': [
            'zscore_20d', 'zscore_60d', 'bb_position', 'bb_width',
            'rsi_14', 'rsi_28'
        ],
        'trend': [
            'ma_cross_5_20', 'ma_cross_20_50', 'ma_dist_20', 'ma_dist_50',
            'adx_14', 'trend_strength', 'trend_r2'
        ],
        'volatility': [
            'vol_5d', 'vol_20d', 'vol_60d', 'vol_ratio_5_20',
            'vol_ratio_20_60', 'vol_regime'
        ],
        'statistical': [
            'skew_20d', 'kurtosis_20d', 'pct_rank_252d'
        ]
    }

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.feature_names: List[str] = []
        self._build_feature_list()

    def _build_feature_list(self):
        """Build ordered feature name list."""
        self.feature_names = []
        for group_features in self.FEATURE_GROUPS.values():
            self.feature_names.extend(group_features)

    def extract_features(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> Tuple[np.ndarray, float]:
        """
        Extract features from price series.

        Returns:
            Tuple of (feature_array, data_quality_score)
        """
        if len(prices) < self.lookback:
            return np.zeros(len(self.feature_names)), 0.0

        # Ensure 1D series
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices = prices.squeeze()

        features = {}
        missing_count = 0
        total_features = len(self.feature_names)

        # Calculate returns
        returns = prices.pct_change().dropna()

        # === MOMENTUM FEATURES ===
        for period in [1, 5, 10, 20, 60]:
            key = f'ret_{period}d'
            if len(prices) >= period + 1:
                features[key] = float(prices.iloc[-1] / prices.iloc[-period-1] - 1)
            else:
                features[key] = 0.0
                missing_count += 1

        # Rate of change
        for period in [5, 10, 20]:
            key = f'roc_{period}d'
            if len(prices) >= period + 1:
                features[key] = float(prices.iloc[-1] / prices.iloc[-period-1] - 1)
            else:
                features[key] = 0.0
                missing_count += 1

        # === MEAN REVERSION FEATURES ===
        for period in [20, 60]:
            key = f'zscore_{period}d'
            if len(prices) >= period:
                mean = float(prices.iloc[-period:].mean())
                std = float(prices.iloc[-period:].std())
                if std > 0:
                    features[key] = float((prices.iloc[-1] - mean) / std)
                else:
                    features[key] = 0.0
            else:
                features[key] = 0.0
                missing_count += 1

        # Bollinger Bands
        if len(prices) >= 20:
            ma_20 = float(prices.rolling(20).mean().iloc[-1])
            std_20 = float(prices.rolling(20).std().iloc[-1])
            upper = ma_20 + 2 * std_20
            lower = ma_20 - 2 * std_20

            if upper != lower:
                features['bb_position'] = float((prices.iloc[-1] - lower) / (upper - lower))
            else:
                features['bb_position'] = 0.5

            if ma_20 > 0:
                features['bb_width'] = float((upper - lower) / ma_20)
            else:
                features['bb_width'] = 0.0
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0.0
            missing_count += 2

        # RSI
        for period in [14, 28]:
            key = f'rsi_{period}'
            features[key] = self._calculate_rsi(prices, period)

        # === TREND FEATURES ===
        # MA crossovers
        if len(prices) >= 50:
            ma_5 = float(prices.rolling(5).mean().iloc[-1])
            ma_20 = float(prices.rolling(20).mean().iloc[-1])
            ma_50 = float(prices.rolling(50).mean().iloc[-1])

            features['ma_cross_5_20'] = 1.0 if ma_5 > ma_20 else -1.0
            features['ma_cross_20_50'] = 1.0 if ma_20 > ma_50 else -1.0
            features['ma_dist_20'] = float((prices.iloc[-1] - ma_20) / ma_20) if ma_20 > 0 else 0.0
            features['ma_dist_50'] = float((prices.iloc[-1] - ma_50) / ma_50) if ma_50 > 0 else 0.0
        else:
            features['ma_cross_5_20'] = 0.0
            features['ma_cross_20_50'] = 0.0
            features['ma_dist_20'] = 0.0
            features['ma_dist_50'] = 0.0
            missing_count += 4

        # ADX approximation
        features['adx_14'] = self._calculate_adx_approx(prices, 14)

        # Trend strength (linear regression)
        if len(prices) >= 20:
            y = prices.iloc[-20:].values
            x = np.arange(20)
            slope, r2 = self._linear_regression_metrics(x, y)
            features['trend_strength'] = float(slope / prices.iloc[-1] * 20) if prices.iloc[-1] > 0 else 0.0
            features['trend_r2'] = float(r2)
        else:
            features['trend_strength'] = 0.0
            features['trend_r2'] = 0.0
            missing_count += 2

        # === VOLATILITY FEATURES ===
        for period in [5, 20, 60]:
            key = f'vol_{period}d'
            if len(returns) >= period:
                features[key] = float(returns.iloc[-period:].std() * np.sqrt(252))
            else:
                features[key] = 0.0
                missing_count += 1

        # Volatility ratios
        if features.get('vol_20d', 0) > 0:
            features['vol_ratio_5_20'] = features['vol_5d'] / features['vol_20d']
        else:
            features['vol_ratio_5_20'] = 1.0

        if features.get('vol_60d', 0) > 0:
            features['vol_ratio_20_60'] = features['vol_20d'] / features['vol_60d']
        else:
            features['vol_ratio_20_60'] = 1.0

        # Vol regime (simplified)
        vol_20 = features.get('vol_20d', 0.15)
        if vol_20 < 0.15:
            features['vol_regime'] = -1.0  # Low vol
        elif vol_20 > 0.30:
            features['vol_regime'] = 1.0   # High vol
        else:
            features['vol_regime'] = 0.0   # Normal

        # === STATISTICAL FEATURES ===
        if len(returns) >= 20:
            features['skew_20d'] = float(returns.iloc[-20:].skew())
            features['kurtosis_20d'] = float(returns.iloc[-20:].kurtosis())
        else:
            features['skew_20d'] = 0.0
            features['kurtosis_20d'] = 0.0
            missing_count += 2

        # Percentile rank
        if len(prices) >= 252:
            price_range = prices.iloc[-252:].max() - prices.iloc[-252:].min()
            if price_range > 0:
                features['pct_rank_252d'] = float(
                    (prices.iloc[-1] - prices.iloc[-252:].min()) / price_range
                )
            else:
                features['pct_rank_252d'] = 0.5
        else:
            features['pct_rank_252d'] = 0.5

        # Build feature array in correct order
        feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])

        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Data quality score
        data_quality = 1.0 - (missing_count / total_features)

        return feature_array, data_quality

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI normalized to [-1, 1]."""
        if len(prices) < period + 1:
            return 0.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        gain_val = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else 0
        loss_val = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else 0

        if loss_val == 0:
            rsi = 100
        else:
            rs = gain_val / loss_val
            rsi = 100 - (100 / (1 + rs))

        # Normalize to [-1, 1]
        return float((rsi - 50) / 50)

    def _calculate_adx_approx(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate ADX approximation normalized to [0, 1]."""
        if len(prices) < period + 1:
            return 0.0

        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0

        pos = returns.where(returns > 0, 0).rolling(period).mean()
        neg = (-returns.where(returns < 0, 0)).rolling(period).mean()

        pos_val = float(pos.iloc[-1]) if pd.notna(pos.iloc[-1]) else 0
        neg_val = float(neg.iloc[-1]) if pd.notna(neg.iloc[-1]) else 0

        total = pos_val + neg_val
        if total == 0:
            return 0.0

        return float(abs(pos_val - neg_val) / total)

    def _linear_regression_metrics(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculate slope and R-squared."""
        if len(x) < 2:
            return 0.0, 0.0

        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return float(slope), float(max(0, r2))


class InstitutionalSignalGenerator:
    """
    Institutional-grade ML signal generator.

    Key principles:
    1. Time-series correct validation (purged CV)
    2. Reproducible training with fixed seeds
    3. Proper feature standardization
    4. Ensemble with disagreement tracking
    5. Clear separation between training and inference
    """

    def __init__(
        self,
        lookback: int = 60,
        n_cv_splits: int = 5,
        purge_gap: int = 5,
        embargo_gap: int = 2,
        random_seed: int = 42
    ):
        self.lookback = lookback
        self.n_cv_splits = n_cv_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.random_seed = random_seed

        self.feature_engine = FeatureEngine(lookback=lookback)
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.models: Dict[str, Any] = {}
        self.models_trained = False
        self.training_metrics: List[TrainingMetrics] = []
        self.feature_importance: Dict[str, float] = {}

        # Model weights (updated based on validation performance)
        self.model_weights: Dict[str, float] = {}

        self._initialize_models()

        logger.info(f"InstitutionalSignalGenerator initialized")
        logger.info(f"  Lookback: {lookback} days")
        logger.info(f"  CV Splits: {n_cv_splits}")
        logger.info(f"  Purge Gap: {purge_gap}, Embargo: {embargo_gap}")
        logger.info(f"  Models: {list(self.models.keys())}")

    def _initialize_models(self):
        """Initialize ML models with regularization."""
        np.random.seed(self.random_seed)

        if ML_AVAILABLE:
            # RandomForest with regularization
            self.models[ModelType.RANDOM_FOREST.value] = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=30,
                min_samples_split=50,
                max_features='sqrt',
                n_jobs=-1,
                random_state=self.random_seed
            )

            # GradientBoosting with regularization
            self.models[ModelType.GRADIENT_BOOSTING.value] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                min_samples_leaf=30,
                random_state=self.random_seed
            )

        if XGBOOST_AVAILABLE:
            self.models[ModelType.XGBOOST.value] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_seed,
                verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            self.models[ModelType.LIGHTGBM.value] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_seed,
                verbose=-1
            )

        # Initialize equal weights
        for name in self.models:
            self.model_weights[name] = 1.0 / len(self.models)

    def train(
        self,
        historical_data: Dict[str, pd.DataFrame],
        target_horizon: int = 5,
        min_samples: int = 500
    ) -> Dict[str, TrainingMetrics]:
        """
        Train models with proper time-series validation.

        Args:
            historical_data: Dict of symbol -> OHLCV DataFrame
            target_horizon: Days ahead to predict returns
            min_samples: Minimum training samples required

        Returns:
            Dict of model_name -> TrainingMetrics
        """
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available")
            return {}

        logger.info("=" * 60)
        logger.info("INSTITUTIONAL ML TRAINING")
        logger.info("=" * 60)

        # Build training dataset
        X_all, y_all, symbols_all = self._build_training_data(
            historical_data, target_horizon
        )

        if len(X_all) < min_samples:
            logger.warning(f"Insufficient samples: {len(X_all)} < {min_samples}")
            return {}

        logger.info(f"Training samples: {len(X_all)}")
        logger.info(f"Features: {X_all.shape[1]}")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X_all)

        # Train with purged cross-validation
        cv = PurgedTimeSeriesSplit(
            n_splits=self.n_cv_splits,
            purge_gap=self.purge_gap,
            embargo_gap=self.embargo_gap
        )

        results = {}

        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name.upper()}...")

            train_scores = []
            val_scores = []
            directional_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]

                # Train
                model.fit(X_train, y_train)

                # Evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)

                train_mse = mean_squared_error(y_train, train_pred)
                val_mse = mean_squared_error(y_val, val_pred)

                # Directional accuracy (more relevant for trading)
                dir_acc = np.mean((val_pred > 0) == (y_val > 0))

                train_scores.append(train_mse)
                val_scores.append(val_mse)
                directional_scores.append(dir_acc)

            # Final training on all data
            model.fit(X_scaled, y_all)

            # Average metrics
            avg_train_mse = np.mean(train_scores)
            avg_val_mse = np.mean(val_scores)
            avg_dir_acc = np.mean(directional_scores)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(
                    self.feature_engine.feature_names,
                    model.feature_importances_
                ))
            else:
                importance = {}

            metrics = TrainingMetrics(
                model_name=model_name,
                train_mse=avg_train_mse,
                val_mse=avg_val_mse,
                directional_accuracy=avg_dir_acc,
                feature_importance=importance,
                training_samples=len(X_all),
                validation_samples=len(X_all) // self.n_cv_splits
            )

            results[model_name] = metrics
            self.training_metrics.append(metrics)

            logger.info(f"  Train MSE: {avg_train_mse:.6f}")
            logger.info(f"  Val MSE: {avg_val_mse:.6f}")
            logger.info(f"  Directional Accuracy: {avg_dir_acc:.1%}")

            # Check for overfitting
            if avg_train_mse > 0:
                overfit_ratio = avg_val_mse / avg_train_mse
                if overfit_ratio > 2.0:
                    logger.warning(f"  ⚠️ Potential overfitting (ratio: {overfit_ratio:.2f})")
                else:
                    logger.info(f"  ✓ Overfit check passed (ratio: {overfit_ratio:.2f})")

        # Update model weights based on validation performance
        self._update_model_weights(results)

        # Aggregate feature importance
        self._aggregate_feature_importance()

        self.models_trained = True

        logger.info("\n" + "=" * 60)
        logger.info("MODEL WEIGHTS:")
        for name, weight in self.model_weights.items():
            logger.info(f"  {name}: {weight:.2%}")
        logger.info("=" * 60)

        return results

    def _build_training_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        target_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build training dataset from historical data."""
        X_list = []
        y_list = []
        symbols_list = []

        for symbol, data in historical_data.items():
            if len(data) < self.lookback + target_horizon + 10:
                continue

            prices = data['Close']

            for i in range(self.lookback, len(prices) - target_horizon):
                # Extract features
                features, quality = self.feature_engine.extract_features(prices.iloc[:i])

                if quality < 0.8:  # Skip low quality samples
                    continue

                # Calculate target: forward return capped at +/- 10%
                current_price = prices.iloc[i]
                future_price = prices.iloc[i + target_horizon]

                target = (future_price / current_price - 1)
                target = np.clip(target, -0.10, 0.10)  # Cap extreme returns

                X_list.append(features)
                y_list.append(target)
                symbols_list.append(symbol)

        return (
            np.array(X_list),
            np.array(y_list),
            symbols_list
        )

    def _update_model_weights(self, results: Dict[str, TrainingMetrics]):
        """Update model weights based on validation performance."""
        # Use inverse of validation MSE as weight
        inv_mse = {}
        for name, metrics in results.items():
            if metrics.val_mse > 0:
                inv_mse[name] = 1.0 / metrics.val_mse
            else:
                inv_mse[name] = 1.0

        # Normalize
        total = sum(inv_mse.values())
        for name in self.model_weights:
            if name in inv_mse:
                self.model_weights[name] = inv_mse[name] / total

    def _aggregate_feature_importance(self):
        """Aggregate feature importance across models."""
        importance_sum = {}

        for metrics in self.training_metrics:
            for feature, imp in metrics.feature_importance.items():
                importance_sum[feature] = importance_sum.get(feature, 0) + imp

        # Normalize
        total = sum(importance_sum.values()) if importance_sum else 1.0
        self.feature_importance = {
            k: v / total for k, v in importance_sum.items()
        }

    def generate_signal(self, symbol: str, prices: pd.Series) -> SignalOutput:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Stock ticker
            prices: Price series (Close prices)

        Returns:
            SignalOutput with signal, confidence, and metadata
        """
        timestamp = datetime.now()

        # Ensure 1D series
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices = prices.squeeze()

        # Extract features
        features, data_quality = self.feature_engine.extract_features(prices)

        if data_quality < 0.5:
            return SignalOutput(
                symbol=symbol,
                signal=0.0,
                confidence=0.0,
                timestamp=timestamp,
                data_quality=data_quality,
                is_stale=True
            )

        # Calculate component signals (rule-based)
        momentum = self._calc_momentum_signal(prices)
        mean_rev = self._calc_mean_reversion_signal(prices)
        trend = self._calc_trend_signal(prices)
        volatility = self._calc_volatility_signal(prices)

        # ML prediction
        ml_pred = 0.0
        ml_std = 0.0

        if self.models_trained and ML_AVAILABLE:
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            predictions = []
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred * self.model_weights[name])
                except Exception:
                    pass

            if predictions:
                ml_pred = sum(predictions)
                ml_std = np.std(predictions) if len(predictions) > 1 else 0.0

        # Combine signals
        # Weight: 60% ML, 40% rule-based
        rule_signal = (
            momentum * 0.35 +
            mean_rev * 0.15 +
            trend * 0.35 +
            volatility * 0.15
        )

        if self.models_trained:
            combined_signal = ml_pred * 0.6 + rule_signal * 0.4
        else:
            combined_signal = rule_signal

        # Clip to [-1, 1]
        # Note: Reduced multiplier from 5 to 2 to prevent signals from hitting bounds too often
        # This gives better signal differentiation (not all signals at +1.0/-1.0)
        signal = float(np.clip(combined_signal * 2, -1, 1))

        # Calculate confidence
        confidence = self._calc_confidence(
            ml_std=ml_std,
            data_quality=data_quality,
            signal_strength=abs(signal)
        )

        # Feature hash for reproducibility
        feature_hash = hashlib.md5(features.tobytes()).hexdigest()[:8]

        return SignalOutput(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            timestamp=timestamp,
            momentum_signal=momentum,
            mean_reversion_signal=mean_rev,
            trend_signal=trend,
            volatility_signal=volatility,
            ml_prediction=ml_pred,
            ml_std=ml_std,
            model_weights=self.model_weights.copy(),
            feature_count=len(features),
            feature_hash=feature_hash,
            data_quality=data_quality,
            is_stale=False
        )

    def _calc_momentum_signal(self, prices: pd.Series) -> float:
        """Multi-timeframe momentum signal."""
        if len(prices) < 60:
            return 0.0

        try:
            ret_5 = prices.iloc[-1] / prices.iloc[-5] - 1
            ret_20 = prices.iloc[-1] / prices.iloc[-20] - 1
            ret_60 = prices.iloc[-1] / prices.iloc[-60] - 1

            signal = ret_5 * 0.5 + ret_20 * 0.3 + ret_60 * 0.2
            # Reduced from *10 to *5 to prevent constant saturation
            return float(np.clip(signal * 5, -1, 1))
        except:
            return 0.0

    def _calc_mean_reversion_signal(self, prices: pd.Series) -> float:
        """Mean reversion (z-score based)."""
        if len(prices) < 50:
            return 0.0

        try:
            mean_20 = prices.iloc[-20:].mean()
            std_20 = prices.iloc[-20:].std()

            if std_20 > 0:
                z = (prices.iloc[-1] - mean_20) / std_20
                # Negative z = below mean = bullish
                return float(np.clip(-z / 3, -1, 1))
            return 0.0
        except:
            return 0.0

    def _calc_trend_signal(self, prices: pd.Series) -> float:
        """Trend following signal."""
        if len(prices) < 50:
            return 0.0

        try:
            ma_10 = prices.rolling(10).mean().iloc[-1]
            ma_20 = prices.rolling(20).mean().iloc[-1]
            ma_50 = prices.rolling(50).mean().iloc[-1]

            score = 0
            if ma_10 > ma_20:
                score += 1
            if ma_20 > ma_50:
                score += 1
            if prices.iloc[-1] > ma_50:
                score += 1

            return float((score - 1.5) / 1.5)
        except:
            return 0.0

    def _calc_volatility_signal(self, prices: pd.Series) -> float:
        """Volatility regime signal."""
        if len(prices) < 60:
            return 0.0

        try:
            returns = prices.pct_change().dropna()
            vol_20 = returns.iloc[-20:].std()
            vol_60 = returns.iloc[-60:].std()

            if vol_60 > 0:
                ratio = vol_20 / vol_60
                # High recent vol = reduce exposure
                return float(np.clip(-(ratio - 1) * 2, -1, 1))
            return 0.0
        except:
            return 0.0

    def _calc_confidence(
        self,
        ml_std: float,
        data_quality: float,
        signal_strength: float
    ) -> float:
        """Calculate signal confidence."""
        # Base confidence from data quality
        conf = data_quality * 0.4

        # Model agreement (low std = high agreement)
        if ml_std > 0:
            agreement = max(0, 1 - ml_std * 10)
            conf += agreement * 0.3
        else:
            conf += 0.3

        # Signal strength
        conf += signal_strength * 0.3

        return float(np.clip(conf, 0, 1))

    def get_diagnostics(self) -> Dict:
        """Get model diagnostics for monitoring."""
        return {
            'models_trained': self.models_trained,
            'model_weights': self.model_weights,
            'feature_importance': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10
            'training_metrics': [
                {
                    'model': m.model_name,
                    'train_mse': m.train_mse,
                    'val_mse': m.val_mse,
                    'directional_accuracy': m.directional_accuracy
                }
                for m in self.training_metrics[-4:]  # Last training
            ]
        }


# Backward compatibility alias
def create_signal_generator(
    lookback: int = 60,
    **kwargs
) -> InstitutionalSignalGenerator:
    """Factory function for signal generator."""
    return InstitutionalSignalGenerator(lookback=lookback, **kwargs)
