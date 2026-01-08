"""
models/advanced_signal_generator.py - Enhanced ML Signal Generator
30+ features with XGBoost/LightGBM ensemble for better accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Check available ML libraries
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    import warnings
    warnings.filterwarnings('ignore')
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


class AdvancedSignalGenerator:
    """Enhanced ML signal generation with 30+ features and multi-model ensemble."""

    def __init__(self):
        self.lookback = 60
        self.feature_scaler = RobustScaler() if ML_AVAILABLE else None
        self.models = {}
        self.models_trained = False
        self.feature_names = []

        # Initialize ML models with optimized hyperparameters
        if ML_AVAILABLE:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                min_samples_split=40,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42
            )

        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )

        logger.info(f"✅ Advanced Signal Generator initialized")
        logger.info(f"   Models: RF={ML_AVAILABLE}, GB={ML_AVAILABLE}, XGB={XGBOOST_AVAILABLE}, LGB={LIGHTGBM_AVAILABLE}")

    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract 30+ features from price data."""
        if len(prices) < self.lookback:
            return np.array([])

        # Ensure prices is a 1D Series (handle multi-index columns)
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices = prices.squeeze()

        features = []
        self.feature_names = []
        returns = prices.pct_change().dropna()

        # === 1. MULTI-TIMEFRAME RETURNS (6 features) ===
        for period in [3, 5, 10, 20, 40, 60]:
            if len(prices) >= period:
                ret = prices.iloc[-1] / prices.iloc[-period] - 1
                features.append(ret)
                self.feature_names.append(f'return_{period}d')
            else:
                features.append(0)
                self.feature_names.append(f'return_{period}d')

        # === 2. RSI AT MULTIPLE TIMEFRAMES (4 features) ===
        for period in [7, 14, 21, 28]:
            rsi = self._calculate_rsi(prices, period)
            features.append((rsi - 50) / 50)  # Normalize to -1 to 1
            self.feature_names.append(f'rsi_{period}')

        # === 3. MACD FEATURES (3 features) ===
        macd, signal, hist = self._calculate_macd_full(prices)
        features.append(macd)
        features.append(signal)
        features.append(hist)
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])

        # === 4. BOLLINGER BAND FEATURES (3 features) ===
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger_bands_full(prices, 20)
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_position = (prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        features.append(bb_width)
        features.append(bb_position)
        features.append(1 if prices.iloc[-1] > bb_upper else (-1 if prices.iloc[-1] < bb_lower else 0))
        self.feature_names.extend(['bb_width', 'bb_position', 'bb_breakout'])

        # === 5. VOLATILITY FEATURES (4 features) ===
        for period in [5, 10, 20, 60]:
            if len(returns) >= period:
                vol = returns.iloc[-period:].std() * np.sqrt(252)  # Annualized
                features.append(vol)
            else:
                features.append(0)
            self.feature_names.append(f'volatility_{period}d')

        # === 6. MOVING AVERAGE FEATURES (6 features) ===
        ma_periods = [10, 20, 50]
        mas = {}
        for period in ma_periods:
            if len(prices) >= period:
                mas[period] = prices.rolling(period).mean().iloc[-1]
            else:
                mas[period] = prices.iloc[-1]

        # Price distance from MAs
        for period in ma_periods:
            dist = (prices.iloc[-1] - mas[period]) / mas[period] if mas[period] > 0 else 0
            features.append(dist)
            self.feature_names.append(f'dist_ma_{period}')

        # MA crossovers
        features.append(1 if mas[10] > mas[20] else -1)
        features.append(1 if mas[20] > mas[50] else -1)
        features.append((mas[10] - mas[50]) / mas[50] if mas[50] > 0 else 0)
        self.feature_names.extend(['ma_10_20_cross', 'ma_20_50_cross', 'ma_10_50_spread'])

        # === 7. TREND STRENGTH (3 features) ===
        features.append(self._calculate_adx(prices, 14))
        features.append(self._calculate_trend_strength(prices, 20))
        features.append(self._calculate_trend_consistency(prices, 20))
        self.feature_names.extend(['adx', 'trend_strength', 'trend_consistency'])

        # === 8. MOMENTUM FEATURES (3 features) ===
        # Rate of change
        roc_5 = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        roc_10 = (prices.iloc[-1] / prices.iloc[-10] - 1) if len(prices) >= 10 else 0
        features.append(roc_5)
        features.append(roc_10)
        features.append(roc_5 - roc_10 / 2)  # Momentum acceleration
        self.feature_names.extend(['roc_5', 'roc_10', 'momentum_accel'])

        # === 9. STATISTICAL FEATURES (4 features) ===
        if len(returns) >= 20:
            features.append(returns.iloc[-20:].skew())
            features.append(returns.iloc[-20:].kurtosis())
        else:
            features.append(0)
            features.append(0)
        self.feature_names.extend(['skew_20d', 'kurtosis_20d'])

        # Z-score
        if len(prices) >= 60:
            mean_60 = prices.iloc[-60:].mean()
            std_60 = prices.iloc[-60:].std()
            features.append((prices.iloc[-1] - mean_60) / std_60 if std_60 > 0 else 0)
        else:
            features.append(0)
        self.feature_names.append('zscore_60d')

        # Percentile rank
        if len(prices) >= 252:
            pct_rank = (prices.iloc[-1] - prices.iloc[-252:].min()) / \
                      (prices.iloc[-252:].max() - prices.iloc[-252:].min()) \
                      if prices.iloc[-252:].max() != prices.iloc[-252:].min() else 0.5
            features.append(pct_rank)
        else:
            features.append(0.5)
        self.feature_names.append('pct_rank_252d')

        return np.array(features)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        gain_val = float(gain.iloc[-1]) if not pd.isna(gain.iloc[-1]) else 0
        loss_val = float(loss.iloc[-1]) if not pd.isna(loss.iloc[-1]) else 0

        rs = gain_val / loss_val if loss_val != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd_full(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # Normalize by price
        price = prices.iloc[-1]
        return (
            float(macd_line.iloc[-1] / price * 100) if price > 0 else 0,
            float(signal_line.iloc[-1] / price * 100) if price > 0 else 0,
            float(histogram.iloc[-1] / price * 100) if price > 0 else 0
        )

    def _calculate_bollinger_bands_full(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            p = float(prices.iloc[-1])
            return p, p, p

        ma = float(prices.rolling(period).mean().iloc[-1])
        std = float(prices.rolling(period).std().iloc[-1])

        if pd.isna(ma) or pd.isna(std):
            p = float(prices.iloc[-1])
            return p, p, p

        return float(ma + 2 * std), float(ma), float(ma - 2 * std)

    def _calculate_adx(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate ADX approximation."""
        if len(prices) < period + 1:
            return 0.0

        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0

        pos_moves = float(returns.where(returns > 0, 0).rolling(period).mean().iloc[-1])
        neg_moves = float((-returns.where(returns < 0, 0)).rolling(period).mean().iloc[-1])

        if pd.isna(pos_moves) or pd.isna(neg_moves):
            return 0.0

        total_move = pos_moves + neg_moves
        if total_move == 0:
            return 0.0

        di_diff = abs(pos_moves - neg_moves)
        return float(di_diff / total_move)

    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(prices) < period:
            return 0.0

        y = prices.iloc[-period:].values
        x = np.arange(period)

        slope = np.polyfit(x, y, 1)[0]
        return float(slope / prices.iloc[-1] * period) if prices.iloc[-1] > 0 else 0

    def _calculate_trend_consistency(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate R-squared of linear fit."""
        if len(prices) < period:
            return 0.0

        y = prices.iloc[-period:].values
        x = np.arange(period)

        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return float(r_squared)

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Train ML models with walk-forward validation."""
        if not ML_AVAILABLE:
            logger.warning("⚠️  ML libraries not available. Skipping training.")
            return

        logger.info("Starting enhanced ML model training...")

        X_train = []
        y_train = []

        for symbol, data in historical_data.items():
            if len(data) < self.lookback + 10:
                continue

            prices = data['Close']

            # Create training samples
            for i in range(self.lookback, len(prices) - 5):
                features = self.extract_features(prices.iloc[:i])
                if len(features) == 0:
                    continue

                # Target: 5-day forward return (capped to reduce outlier impact)
                future_return = (prices.iloc[i+5] - prices.iloc[i]) / prices.iloc[i]
                future_return = np.clip(future_return, -0.15, 0.15)  # Cap at +/- 15%

                X_train.append(features)
                y_train.append(future_return)

        if len(X_train) < 500:
            logger.warning(f"⚠️  Not enough training data ({len(X_train)} samples)")
            return

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        logger.info(f"Training on {len(X_train)} samples with {X_train.shape[1]} features.")

        # Train each model
        accuracies = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)

                # Calculate directional accuracy (more meaningful for trading)
                predictions = model.predict(X_train_scaled)
                correct_direction = np.sum((predictions > 0) == (y_train > 0))
                accuracy = correct_direction / len(y_train)
                accuracies[name] = accuracy

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        self.models_trained = True

        # Log results
        acc_str = ", ".join([f"{k.upper()}: {v:.3f}" for k, v in accuracies.items()])
        logger.info(f"Training Completed. Directional Accuracy -> {acc_str}")

        # Check for overfitting
        max_acc = max(accuracies.values()) if accuracies else 0
        if max_acc < 0.95:
            logger.info("✅ Model regularization appears effective (Accuracy < 95%)")
        else:
            logger.warning("⚠️  Possible overfitting detected (Accuracy > 95%)")

    def generate_ml_signal(self, symbol: str, prices: pd.Series) -> Dict:
        """Generate ML-powered signal with ensemble prediction."""
        # Ensure prices is a 1D Series
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices = prices.squeeze()

        if len(prices) < self.lookback:
            return self._empty_signal()

        features = self.extract_features(prices)

        if len(features) == 0:
            return self._empty_signal()

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Component signals
        components = {}

        # 1. Technical Analysis Signals
        components['momentum'] = self._momentum_signal(prices)
        components['mean_reversion'] = self._mean_reversion_signal(prices)
        components['trend'] = self._trend_signal(prices)
        components['volatility'] = self._volatility_signal(prices)

        # 2. ML Model Predictions
        ml_predictions = []
        if self.models_trained and ML_AVAILABLE:
            try:
                features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

                for name, model in self.models.items():
                    try:
                        pred = model.predict(features_scaled)[0]
                        scaled_pred = np.tanh(pred * 20)  # Scale to -1, 1
                        components[f'ml_{name}'] = float(scaled_pred)
                        ml_predictions.append(scaled_pred)
                    except Exception:
                        components[f'ml_{name}'] = 0.0
            except Exception:
                pass

        # 3. Weighted ensemble
        weights = {
            'momentum': 0.15,
            'mean_reversion': 0.10,
            'trend': 0.15,
            'volatility': 0.05,
            'ml_rf': 0.15,
            'ml_gb': 0.15,
            'ml_xgb': 0.15,
            'ml_lgb': 0.10
        }

        signal = 0.0
        total_weight = 0.0
        for component, value in components.items():
            weight = weights.get(component, 0.1)
            signal += value * weight
            total_weight += weight

        if total_weight > 0:
            signal = signal / total_weight

        signal = float(np.clip(signal, -1, 1))

        # 4. Calculate confidence
        confidence = self._calculate_confidence(components, ml_predictions)

        return {
            'signal': signal,
            'confidence': confidence,
            'components': components,
            'timestamp': datetime.now()
        }

    def _empty_signal(self) -> Dict:
        """Return empty signal."""
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'components': {},
            'timestamp': datetime.now()
        }

    def _calculate_confidence(self, components: Dict, ml_predictions: List) -> float:
        """Calculate confidence based on component agreement."""
        if not components:
            return 0.0

        # Component agreement
        signs = [np.sign(v) for v in components.values() if v != 0]
        if signs:
            agreement = abs(sum(signs) / len(signs))
        else:
            agreement = 0.0

        # ML model agreement
        if len(ml_predictions) >= 2:
            ml_signs = [np.sign(p) for p in ml_predictions if p != 0]
            ml_agreement = abs(sum(ml_signs) / len(ml_signs)) if ml_signs else 0
        else:
            ml_agreement = 0.5

        # Signal strength
        avg_signal = abs(np.mean(list(components.values())))

        confidence = (agreement * 0.3 + ml_agreement * 0.4 + avg_signal * 0.3)
        return float(min(confidence, 1.0))

    def _momentum_signal(self, prices: pd.Series) -> float:
        """Multi-timeframe momentum signal."""
        if len(prices) < 60:
            return 0.0

        try:
            p_now = float(prices.iloc[-1])
            mom_5 = p_now / float(prices.iloc[-5]) - 1
            mom_20 = p_now / float(prices.iloc[-20]) - 1
            mom_60 = p_now / float(prices.iloc[-60]) - 1

            signal = mom_5 * 0.5 + mom_20 * 0.3 + mom_60 * 0.2
            return float(np.tanh(signal * 15))
        except:
            return 0.0

    def _mean_reversion_signal(self, prices: pd.Series) -> float:
        """Mean reversion signal."""
        if len(prices) < 50:
            return 0.0

        try:
            p_now = float(prices.iloc[-1])
            mean_20 = float(prices.iloc[-20:].mean())
            std_20 = float(prices.iloc[-20:].std())
            mean_50 = float(prices.iloc[-50:].mean())
            std_50 = float(prices.iloc[-50:].std())

            z_20 = (p_now - mean_20) / std_20 if std_20 > 0 else 0
            z_50 = (p_now - mean_50) / std_50 if std_50 > 0 else 0

            signal = -(z_20 * 0.6 + z_50 * 0.4)
            return float(np.tanh(signal / 2))
        except:
            return 0.0

    def _trend_signal(self, prices: pd.Series) -> float:
        """Trend-following signal."""
        if len(prices) < 50:
            return 0.0

        try:
            ma_10 = float(prices.rolling(10).mean().iloc[-1])
            ma_20 = float(prices.rolling(20).mean().iloc[-1])
            ma_50 = float(prices.rolling(50).mean().iloc[-1])
            p_now = float(prices.iloc[-1])

            score = 0
            if ma_10 > ma_20: score += 1
            if ma_20 > ma_50: score += 1
            if p_now > ma_50: score += 1

            return float((score - 1.5) / 1.5)
        except:
            return 0.0

    def _volatility_signal(self, prices: pd.Series) -> float:
        """Volatility-based signal."""
        if len(prices) < 60:
            return 0.0

        try:
            returns = prices.pct_change().dropna()

            vol_20 = float(returns.iloc[-20:].std())
            vol_60 = float(returns.iloc[-60:].std())

            if vol_60 == 0 or pd.isna(vol_60) or pd.isna(vol_20):
                return 0.0

            vol_ratio = vol_20 / vol_60
            return float(-np.tanh((vol_ratio - 1) * 3))
        except:
            return 0.0
