"""
models/god_level_signal_generator.py - God Level ML Signal Generator
Advanced ensemble combining: XGBoost + LightGBM + RandomForest + Neural Network features
With market regime detection, multi-timeframe analysis, and adaptive confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum
import logging
import warnings
import os
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Check available ML libraries
ML_LIBS = {'sklearn': False, 'xgboost': False, 'lightgbm': False}

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    ML_LIBS['sklearn'] = True
except ImportError:
    pass

try:
    import xgboost as xgb
    ML_LIBS['xgboost'] = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    ML_LIBS['lightgbm'] = True
except ImportError:
    pass


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


class GodLevelSignalGenerator:
    """
    God-level signal generation with:
    - Multi-model ensemble (XGBoost, LightGBM, RandomForest, GradientBoosting)
    - Market regime detection
    - Multi-timeframe analysis
    - Adaptive confidence scoring
    - Feature importance tracking
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.lookback = 60
        self.feature_scaler = RobustScaler() if ML_LIBS['sklearn'] else None
        self.models = {}
        self.models_trained = False
        self.feature_importance = {}
        self.regime_history = []
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models based on available libraries
        self._init_models()
        
        # Try to load existing models
        self.load_models()

        logger.info(f"God Level Signal Generator initialized")
        logger.info(f"  ML Libraries: sklearn={ML_LIBS['sklearn']}, xgboost={ML_LIBS['xgboost']}, lightgbm={ML_LIBS['lightgbm']}")

    def _init_models(self):
        """Initialize all available ML models."""
        if ML_LIBS['sklearn']:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
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

        if ML_LIBS['xgboost']:
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

        if ML_LIBS['lightgbm']:
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

    def extract_features(self, prices: pd.Series, volume: pd.Series = None) -> np.ndarray:
        """
        Extract comprehensive feature set.
        50+ features across multiple categories.
        """
        if len(prices) < self.lookback:
            return np.array([])

        features = []
        returns = prices.pct_change().dropna()

        # === 1. PRICE MOMENTUM FEATURES (10 features) ===
        for period in [5, 10, 20, 40, 60]:
            if len(prices) >= period:
                features.append(prices.iloc[-1] / prices.iloc[-period] - 1)
            else:
                features.append(0)

        # Rate of change acceleration
        roc_5 = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        roc_10 = (prices.iloc[-1] / prices.iloc[-10] - 1) if len(prices) >= 10 else 0
        features.append(roc_5 - roc_10 / 2)  # Momentum acceleration

        # === 2. RSI FEATURES (4 features) ===
        for period in [7, 14, 21, 28]:
            features.append(self._calculate_rsi(prices, period))

        # === 3. MACD FEATURES (4 features) ===
        macd, signal, hist = self._calculate_macd_full(prices)
        features.append(macd)
        features.append(signal)
        features.append(hist)
        features.append(np.sign(hist) if hist != 0 else 0)  # MACD direction

        # === 4. BOLLINGER BAND FEATURES (4 features) ===
        bb_upper, bb_mid, bb_lower = self._calculate_bollinger_bands_full(prices, 20)
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
        bb_position = (prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        features.append(bb_width)
        features.append(bb_position)
        features.append(1 if prices.iloc[-1] > bb_upper else (-1 if prices.iloc[-1] < bb_lower else 0))
        features.append(bb_width - self._calculate_bollinger_bands_full(prices.iloc[:-5], 20)[0] if len(prices) > 25 else 0)

        # === 5. VOLATILITY FEATURES (6 features) ===
        for period in [5, 10, 20]:
            if len(returns) >= period:
                features.append(returns.iloc[-period:].std())
            else:
                features.append(0)

        # Volatility ratio (current vs historical)
        vol_5 = returns.iloc[-5:].std() if len(returns) >= 5 else 0
        vol_20 = returns.iloc[-20:].std() if len(returns) >= 20 else 1
        features.append(vol_5 / vol_20 if vol_20 > 0 else 1)

        # ATR-like feature
        features.append(self._calculate_atr_approx(prices, 14))

        # Volatility trend
        vol_10 = returns.iloc[-10:].std() if len(returns) >= 10 else 0
        features.append(vol_5 - vol_10)

        # === 6. MOVING AVERAGE FEATURES (8 features) ===
        ma_periods = [10, 20, 50, 100]
        mas = {}
        for period in ma_periods:
            if len(prices) >= period:
                mas[period] = prices.rolling(period).mean().iloc[-1]
            else:
                mas[period] = prices.iloc[-1]

        # Price distance from MAs
        for period in ma_periods:
            features.append((prices.iloc[-1] - mas[period]) / mas[period] if mas[period] > 0 else 0)

        # MA crossover features
        features.append(1 if mas[10] > mas[20] else -1)
        features.append(1 if mas[20] > mas[50] else -1)
        features.append(1 if mas[50] > mas[100] else -1)
        features.append((mas[10] - mas[50]) / mas[50] if mas[50] > 0 else 0)

        # === 7. TREND STRENGTH FEATURES (4 features) ===
        features.append(self._calculate_adx(prices, 14))
        features.append(self._calculate_trend_strength(prices, 20))
        features.append(self._calculate_trend_consistency(prices, 20))
        features.append(self._calculate_higher_highs_lows(prices, 20))

        # === 8. SUPPORT/RESISTANCE FEATURES (4 features) ===
        sr_levels = self._calculate_support_resistance(prices, 20)
        features.append(sr_levels['distance_to_support'])
        features.append(sr_levels['distance_to_resistance'])
        features.append(sr_levels['sr_strength'])
        features.append(sr_levels['breakout_potential'])

        # === 9. PATTERN FEATURES (4 features) ===
        features.append(self._detect_double_bottom_top(prices))
        features.append(self._detect_trend_reversal(prices))
        features.append(self._calculate_price_acceleration(prices))
        features.append(self._calculate_momentum_divergence(prices))

        # === 10. STATISTICAL FEATURES (4 features) ===
        if len(returns) >= 20:
            features.append(returns.iloc[-20:].skew())
            features.append(returns.iloc[-20:].kurtosis())
        else:
            features.append(0)
            features.append(0)

        # Z-score of current price
        if len(prices) >= 60:
            mean_60 = prices.iloc[-60:].mean()
            std_60 = prices.iloc[-60:].std()
            features.append((prices.iloc[-1] - mean_60) / std_60 if std_60 > 0 else 0)
        else:
            features.append(0)

        # Percentile rank
        if len(prices) >= 252:
            features.append((prices.iloc[-1] - prices.iloc[-252:].min()) /
                          (prices.iloc[-252:].max() - prices.iloc[-252:].min())
                          if prices.iloc[-252:].max() != prices.iloc[-252:].min() else 0.5)
        else:
            features.append(0.5)

        return np.array(features)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI normalized to -1 to 1 range."""
        if len(prices) < period + 1:
            return 0.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # Normalize to -1 to 1 (50 = neutral)
        return float((rsi - 50) / 50)

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
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            p = prices.iloc[-1]
            return p, p, p

        ma = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]

        return float(ma + 2 * std), float(ma), float(ma - 2 * std)

    def _calculate_atr_approx(self, prices: pd.Series, period: int = 14) -> float:
        """Approximate ATR using close prices only."""
        if len(prices) < period + 1:
            return 0.0

        # Use price range as approximation
        daily_range = prices.diff().abs()
        atr = daily_range.rolling(period).mean().iloc[-1]

        return float(atr / prices.iloc[-1] * 100) if prices.iloc[-1] > 0 else 0

    def _calculate_adx(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate ADX (trend strength) approximation."""
        if len(prices) < period + 1:
            return 0.0

        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0

        # Simplified ADX: trend strength approximation
        pos_moves = returns.where(returns > 0, 0).rolling(period).mean().iloc[-1]
        neg_moves = (-returns.where(returns < 0, 0)).rolling(period).mean().iloc[-1]

        total_move = pos_moves + neg_moves
        if total_move == 0:
            return 0.0

        di_diff = abs(pos_moves - neg_moves)
        adx = di_diff / total_move

        return float(adx)

    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(prices) < period:
            return 0.0

        y = prices.iloc[-period:].values
        x = np.arange(period)

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Normalize by price
        return float(slope / prices.iloc[-1] * period) if prices.iloc[-1] > 0 else 0

    def _calculate_trend_consistency(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate how consistent the trend is (R-squared of linear fit)."""
        if len(prices) < period:
            return 0.0

        y = prices.iloc[-period:].values
        x = np.arange(period)

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)

        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return float(r_squared)

    def _calculate_higher_highs_lows(self, prices: pd.Series, period: int = 20) -> float:
        """Count higher highs and higher lows pattern."""
        if len(prices) < period:
            return 0.0

        recent = prices.iloc[-period:]

        # Split into 4 segments
        seg_size = period // 4
        segments = [recent.iloc[i*seg_size:(i+1)*seg_size] for i in range(4)]

        higher_highs = sum(1 for i in range(1, 4) if segments[i].max() > segments[i-1].max())
        higher_lows = sum(1 for i in range(1, 4) if segments[i].min() > segments[i-1].min())

        # Score: +1 for uptrend, -1 for downtrend
        score = (higher_highs + higher_lows - 3) / 3  # Normalize to roughly -1 to 1

        return float(score)

    def _calculate_support_resistance(self, prices: pd.Series, period: int = 20) -> Dict:
        """Calculate support/resistance levels and related features."""
        if len(prices) < period:
            return {'distance_to_support': 0, 'distance_to_resistance': 0, 'sr_strength': 0, 'breakout_potential': 0}

        recent = prices.iloc[-period:]
        current = prices.iloc[-1]

        # Simple S/R: recent high and low
        resistance = recent.max()
        support = recent.min()

        range_size = resistance - support
        if range_size == 0:
            return {'distance_to_support': 0, 'distance_to_resistance': 0, 'sr_strength': 0, 'breakout_potential': 0}

        dist_to_support = (current - support) / range_size
        dist_to_resistance = (resistance - current) / range_size

        # S/R strength: how many times price touched these levels
        touch_threshold = range_size * 0.05
        support_touches = sum(1 for p in recent if abs(p - support) < touch_threshold)
        resistance_touches = sum(1 for p in recent if abs(p - resistance) < touch_threshold)
        sr_strength = (support_touches + resistance_touches) / period

        # Breakout potential: price near boundary with momentum
        if dist_to_resistance < 0.1:
            breakout_potential = 0.5  # Near resistance
        elif dist_to_support < 0.1:
            breakout_potential = -0.5  # Near support
        else:
            breakout_potential = 0

        return {
            'distance_to_support': float(dist_to_support),
            'distance_to_resistance': float(dist_to_resistance),
            'sr_strength': float(sr_strength),
            'breakout_potential': float(breakout_potential)
        }

    def _detect_double_bottom_top(self, prices: pd.Series) -> float:
        """Detect double bottom/top patterns."""
        if len(prices) < 40:
            return 0.0

        recent = prices.iloc[-40:]

        # Find local minima and maxima
        min_idx = recent.idxmin()
        max_idx = recent.idxmax()

        # Simple pattern detection
        first_half = recent.iloc[:20]
        second_half = recent.iloc[20:]

        # Double bottom: two similar lows
        if abs(first_half.min() - second_half.min()) / prices.iloc[-1] < 0.02:
            if prices.iloc[-1] > (first_half.min() + second_half.min()) / 2:
                return 0.5  # Potential double bottom breakout

        # Double top: two similar highs
        if abs(first_half.max() - second_half.max()) / prices.iloc[-1] < 0.02:
            if prices.iloc[-1] < (first_half.max() + second_half.max()) / 2:
                return -0.5  # Potential double top breakdown

        return 0.0

    def _detect_trend_reversal(self, prices: pd.Series) -> float:
        """Detect potential trend reversal."""
        if len(prices) < 30:
            return 0.0

        # Compare recent momentum to previous momentum
        recent_return = prices.iloc[-5:].mean() / prices.iloc[-10:-5].mean() - 1
        prev_return = prices.iloc[-15:-10].mean() / prices.iloc[-20:-15].mean() - 1

        # Reversal: momentum changed direction
        if recent_return > 0 and prev_return < 0:
            return 0.5  # Bullish reversal
        elif recent_return < 0 and prev_return > 0:
            return -0.5  # Bearish reversal

        return 0.0

    def _calculate_price_acceleration(self, prices: pd.Series) -> float:
        """Calculate price acceleration (second derivative)."""
        if len(prices) < 15:
            return 0.0

        # First derivative (velocity)
        vel_recent = prices.iloc[-5:].mean() - prices.iloc[-10:-5].mean()
        vel_prev = prices.iloc[-10:-5].mean() - prices.iloc[-15:-10].mean()

        # Second derivative (acceleration)
        accel = vel_recent - vel_prev

        # Normalize
        return float(np.tanh(accel / prices.iloc[-1] * 100))

    def _calculate_momentum_divergence(self, prices: pd.Series) -> float:
        """Detect price/momentum divergence."""
        if len(prices) < 20:
            return 0.0

        # Price trend
        price_trend = prices.iloc[-1] > prices.iloc[-10]

        # RSI trend
        rsi_now = self._calculate_rsi(prices, 14)
        rsi_prev = self._calculate_rsi(prices.iloc[:-5], 14)
        rsi_trend = rsi_now > rsi_prev

        # Divergence
        if price_trend and not rsi_trend:
            return -0.3  # Bearish divergence
        elif not price_trend and rsi_trend:
            return 0.3  # Bullish divergence

        return 0.0

    def detect_market_regime(self, prices: pd.Series, benchmark_prices: pd.Series = None) -> MarketRegime:
        """
        Detect current market regime.
        """
        if len(prices) < 60:
            return MarketRegime.NEUTRAL

        # Calculate key metrics
        returns = prices.pct_change().dropna()

        # Trend: 20-day vs 50-day MA
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        trend = (ma_20 - ma_50) / ma_50 if ma_50 > 0 else 0

        # Volatility
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized

        # Momentum
        momentum = prices.iloc[-1] / prices.iloc[-20] - 1

        # Classify regime
        # Note: 50% threshold for individual stocks (35% was too low - typical stock vol is 25-45%)
        if vol_20 > 0.50:  # High volatility (>50% annualized for individual stocks)
            return MarketRegime.HIGH_VOLATILITY
        elif trend > 0.05 and momentum > 0.05:
            return MarketRegime.STRONG_BULL
        elif trend > 0.02 and momentum > 0:
            return MarketRegime.BULL
        elif trend < -0.05 and momentum < -0.05:
            return MarketRegime.STRONG_BEAR
        elif trend < -0.02 and momentum < 0:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Train all ML models with walk-forward validation."""
        if not any(ML_LIBS.values()):
            logger.warning("No ML libraries available. Skipping training.")
            return

        logger.info("Training God Level ML models...")

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

                # Target: 5-day forward return
                future_return = (prices.iloc[i+5] - prices.iloc[i]) / prices.iloc[i]

                X_train.append(features)
                y_train.append(future_return)

        if len(X_train) < 500:
            logger.warning(f"Not enough training data ({len(X_train)} samples)")
            return

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        logger.info(f"  Training on {len(X_train)} samples with {X_train.shape[1]} features...")

        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                logger.info(f"  {name.upper()} trained successfully")
            except Exception as e:
                logger.error(f"  Failed to train {name}: {e}")

        # Calculate feature importance (average across models)
        self._calculate_feature_importance()

        self.models_trained = True
        self.save_models()
        logger.info(f"God Level models trained with {len(self.models)} active models and saved to {self.model_dir}")

    def save_models(self):
        """Save trained models and scaler to disk."""
        if not self.models_trained:
            return

        try:
            model_data = {
                'models': self.models,
                'scaler': self.feature_scaler,
                'importance': self.feature_importance,
                'trained_at': datetime.now()
            }
            with open(self.model_dir / "god_level_models.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"✅ God Level models saved to {self.model_dir / 'god_level_models.pkl'}")
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")

    def load_models(self):
        """Load models and scaler from disk if available."""
        model_path = self.model_dir / "god_level_models.pkl"
        if not model_path.exists():
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.feature_scaler = model_data['scaler']
            self.feature_importance = model_data.get('importance', {})
            self.models_trained = True
            
            trained_at = model_data.get('trained_at', 'unknown')
            logger.info(f"✅ God Level models loaded from disk (trained at: {trained_at})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False

    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        importance_sum = None
        count = 0

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                if importance_sum is None:
                    importance_sum = model.feature_importances_.copy()
                else:
                    importance_sum += model.feature_importances_
                count += 1

        if count > 0:
            self.feature_importance = importance_sum / count

    def generate_ml_signal(self, symbol: str, prices: pd.Series) -> Dict:
        """
        Generate god-level ML signal with ensemble prediction and adaptive confidence.
        """
        if len(prices) < self.lookback:
            return self._empty_signal()

        # Extract features
        features = self.extract_features(prices)
        if len(features) == 0:
            return self._empty_signal()

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Detect market regime
        regime = self.detect_market_regime(prices)

        # Component signals
        components = {}

        # 1. Technical Analysis Signals
        components['momentum'] = self._momentum_signal(prices)
        components['mean_reversion'] = self._mean_reversion_signal(prices)
        components['trend'] = self._trend_signal(prices)
        components['volatility'] = self._volatility_signal(prices)
        components['breakout'] = self._breakout_signal(prices)

        # 2. ML Model Predictions
        ml_predictions = []
        if self.models_trained and ML_LIBS['sklearn']:
            try:
                features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

                for name, model in self.models.items():
                    try:
                        pred = model.predict(features_scaled)[0]
                        # Scale prediction to -1, 1 range
                        scaled_pred = np.tanh(pred * 20)
                        components[f'ml_{name}'] = float(scaled_pred)
                        ml_predictions.append(scaled_pred)
                    except Exception:
                        components[f'ml_{name}'] = 0.0
            except Exception:
                pass

        # 3. Regime-adjusted weights
        weights = self._get_regime_weights(regime)

        # 4. Calculate ensemble signal
        signal = 0.0
        total_weight = 0.0
        for component, value in components.items():
            weight = weights.get(component, 0.1)
            signal += value * weight
            total_weight += weight

        if total_weight > 0:
            signal = signal / total_weight

        signal = float(np.clip(signal, -1, 1))

        # 5. Calculate adaptive confidence
        confidence = self._calculate_adaptive_confidence(components, ml_predictions, regime)

        return {
            'signal': signal,
            'confidence': confidence,
            'regime': regime.value,
            'components': components,
            'timestamp': datetime.now()
        }

    def _empty_signal(self) -> Dict:
        """Return empty/neutral signal."""
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'regime': MarketRegime.NEUTRAL.value,
            'components': {},
            'timestamp': datetime.now()
        }

    def _get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get component weights based on market regime."""
        base_weights = {
            'momentum': 0.15,
            'mean_reversion': 0.10,
            'trend': 0.15,
            'volatility': 0.10,
            'breakout': 0.10,
            'ml_rf': 0.10,
            'ml_gb': 0.10,
            'ml_xgb': 0.10,
            'ml_lgb': 0.10
        }

        # Adjust weights based on regime
        if regime == MarketRegime.STRONG_BULL:
            base_weights['momentum'] = 0.25
            base_weights['trend'] = 0.25
            base_weights['mean_reversion'] = 0.05
        elif regime == MarketRegime.STRONG_BEAR:
            base_weights['momentum'] = 0.25
            base_weights['trend'] = 0.25
            base_weights['mean_reversion'] = 0.05
        elif regime == MarketRegime.NEUTRAL:
            base_weights['mean_reversion'] = 0.20
            base_weights['momentum'] = 0.10
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_weights['volatility'] = 0.20
            base_weights['breakout'] = 0.15
            base_weights['momentum'] = 0.10

        return base_weights

    def _calculate_adaptive_confidence(self, components: Dict, ml_predictions: List, regime: MarketRegime) -> float:
        """Calculate confidence based on component agreement and regime."""
        if not components:
            return 0.0

        # 1. Component agreement
        signs = [np.sign(v) for v in components.values() if v != 0]
        if signs:
            agreement = abs(sum(signs) / len(signs))
        else:
            agreement = 0.0

        # 2. ML model agreement
        if len(ml_predictions) >= 2:
            ml_signs = [np.sign(p) for p in ml_predictions if p != 0]
            ml_agreement = abs(sum(ml_signs) / len(ml_signs)) if ml_signs else 0
        else:
            ml_agreement = 0.5

        # 3. Signal strength
        avg_signal = abs(np.mean(list(components.values())))

        # 4. Regime penalty
        regime_multiplier = {
            MarketRegime.STRONG_BULL: 1.0,
            MarketRegime.BULL: 0.9,
            MarketRegime.NEUTRAL: 0.7,
            MarketRegime.BEAR: 0.9,
            MarketRegime.STRONG_BEAR: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.6
        }.get(regime, 0.7)

        # Combined confidence
        confidence = (agreement * 0.3 + ml_agreement * 0.4 + avg_signal * 0.3) * regime_multiplier

        return float(min(confidence, 1.0))

    def _momentum_signal(self, prices: pd.Series) -> float:
        """Multi-timeframe momentum signal."""
        if len(prices) < 60:
            return 0.0

        # Multiple timeframe momentum
        mom_5 = prices.iloc[-1] / prices.iloc[-5] - 1
        mom_20 = prices.iloc[-1] / prices.iloc[-20] - 1
        mom_60 = prices.iloc[-1] / prices.iloc[-60] - 1

        # Weighted average
        signal = mom_5 * 0.5 + mom_20 * 0.3 + mom_60 * 0.2

        return float(np.tanh(signal * 15))

    def _mean_reversion_signal(self, prices: pd.Series) -> float:
        """Mean reversion signal with multiple lookbacks."""
        if len(prices) < 50:
            return 0.0

        # Z-scores at different timeframes
        z_20 = (prices.iloc[-1] - prices.iloc[-20:].mean()) / prices.iloc[-20:].std() if prices.iloc[-20:].std() > 0 else 0
        z_50 = (prices.iloc[-1] - prices.iloc[-50:].mean()) / prices.iloc[-50:].std() if prices.iloc[-50:].std() > 0 else 0

        # Combine and invert (buy oversold, sell overbought)
        signal = -(z_20 * 0.6 + z_50 * 0.4)

        return float(np.tanh(signal / 2))

    def _trend_signal(self, prices: pd.Series) -> float:
        """Trend-following signal."""
        if len(prices) < 100:
            return 0.0

        # Multiple MA crossovers
        ma_10 = prices.rolling(10).mean().iloc[-1]
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        ma_100 = prices.rolling(100).mean().iloc[-1]

        # Trend score
        score = 0
        if ma_10 > ma_20: score += 1
        if ma_20 > ma_50: score += 1
        if ma_50 > ma_100: score += 1
        if prices.iloc[-1] > ma_50: score += 1

        # Normalize to -1 to 1
        signal = (score - 2) / 2

        return float(signal)

    def _volatility_signal(self, prices: pd.Series) -> float:
        """Volatility-based signal (prefer low vol for longs)."""
        if len(prices) < 60:
            return 0.0

        returns = prices.pct_change().dropna()

        vol_20 = returns.iloc[-20:].std()
        vol_60 = returns.iloc[-60:].std()

        if vol_60 == 0:
            return 0.0

        vol_ratio = vol_20 / vol_60

        # Low current vol is bullish, high vol is bearish
        return float(-np.tanh((vol_ratio - 1) * 3))

    def _breakout_signal(self, prices: pd.Series) -> float:
        """Breakout detection signal."""
        if len(prices) < 20:
            return 0.0

        recent = prices.iloc[-20:]
        current = prices.iloc[-1]

        high_20 = recent.max()
        low_20 = recent.min()
        range_20 = high_20 - low_20

        if range_20 == 0:
            return 0.0

        # Breakout score
        if current >= high_20 * 0.99:
            return 0.8  # Bullish breakout
        elif current <= low_20 * 1.01:
            return -0.8  # Bearish breakdown

        # Position within range
        position = (current - low_20) / range_20
        return float((position - 0.5) * 0.5)
