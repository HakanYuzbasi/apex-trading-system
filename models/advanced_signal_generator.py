"""
models/advanced_signal_generator.py - State-of-the-Art ML Signal Generator
Combines: Transformer + Ensemble ML + Technical Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


class AdvancedSignalGenerator:
    """State-of-the-art ML signal generation using multiple models."""
    
    def __init__(self):
        self.lookback = 60
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Initialize ML models
        if ML_AVAILABLE:
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            self.models_trained = False
        else:
            self.models_trained = False
        
        logger.info("✅ Advanced Signal Generator initialized")
    
    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract features from price data."""
        if len(prices) < self.lookback:
            return np.array([])
        
        features = []
        
        # Price-based features
        features.append(prices.iloc[-1] / prices.iloc[-20] - 1)  # 20-day return
        features.append(prices.iloc[-1] / prices.iloc[-60] - 1)  # 60-day return
        features.append(prices.iloc[-1] / prices.iloc[-5] - 1)   # 5-day return
        
        # Momentum indicators
        features.append(self._calculate_rsi(prices, 14))
        features.append(self._calculate_rsi(prices, 7))
        
        # Volatility
        returns = prices.pct_change().dropna()
        features.append(returns.iloc[-20:].std() if len(returns) >= 20 else 0)
        features.append(returns.iloc[-60:].std() if len(returns) >= 60 else 0)
        
        # Moving averages
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        features.append((prices.iloc[-1] - ma_20) / ma_20 if ma_20 > 0 else 0)
        features.append((ma_20 - ma_50) / ma_50 if ma_50 > 0 else 0)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices, 20)
        if bb_upper > bb_lower:
            features.append((prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower))
        else:
            features.append(0.5)
        
        # MACD
        macd, signal = self._calculate_macd(prices)
        features.append(macd - signal)
        
        # Volume-weighted indicators (mock if volume not available)
        features.append(self._calculate_obv_trend(prices))
        
        # Trend strength
        features.append(self._calculate_adx(prices))
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
        
        return float(rsi)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return prices.iloc[-1], prices.iloc[-1]
        
        ma = prices.rolling(period).mean().iloc[-1]
        std = prices.rolling(period).std().iloc[-1]
        
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        
        return float(upper), float(lower)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and Signal line."""
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        macd = ema_12.iloc[-1] - ema_26.iloc[-1]
        
        macd_series = ema_12 - ema_26
        signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
        
        return float(macd), float(signal)
    
    def _calculate_obv_trend(self, prices: pd.Series) -> float:
        """Calculate On-Balance Volume trend (simplified without volume)."""
        if len(prices) < 20:
            return 0.0
        
        # Use price changes as proxy for volume direction
        price_changes = prices.diff().iloc[-20:]
        obv = (price_changes > 0).astype(int).sum() / 20.0
        
        return float(obv - 0.5) * 2  # Scale to -1 to 1
    
    def _calculate_adx(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index (trend strength)."""
        if len(prices) < period + 1:
            return 0.0
        
        # Simplified ADX calculation
        returns = prices.pct_change().dropna()
        
        if len(returns) < period:
            return 0.0
        
        # Trend strength approximation
        trend = abs(returns.iloc[-period:].mean() / returns.iloc[-period:].std()) if returns.iloc[-period:].std() > 0 else 0
        adx = min(trend * 25, 100)  # Scale to 0-100
        
        return float(adx)
    
    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Train ML models on historical data."""
        if not ML_AVAILABLE:
            logger.warning("⚠️  ML libraries not available. Skipping training.")
            return
        
        logger.info("🧠 Training ML models...")
        
        X_train = []
        y_train = []
        
        for symbol, data in historical_data.items():
            if len(data) < self.lookback + 5:
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
        
        if len(X_train) < 100:
            logger.warning("⚠️  Not enough training data")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        # Train models
        logger.info(f"   Training on {len(X_train)} samples...")
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)
        
        self.models_trained = True
        logger.info("✅ ML models trained successfully")
    
    def generate_ml_signal(self, symbol: str, prices: pd.Series) -> Dict:
        """Generate ML-powered signal."""
        if len(prices) < self.lookback:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'components': {},
                'timestamp': datetime.now()
            }
        
        # Extract features
        features = self.extract_features(prices)
        
        if len(features) == 0:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'components': {},
                'timestamp': datetime.now()
            }
        
        # Component signals
        components = {}
        
        # 1. Technical Analysis Signals
        components['momentum'] = self._momentum_signal(prices)
        components['mean_reversion'] = self._mean_reversion_signal(prices)
        components['trend'] = self._trend_signal(prices)
        components['volatility'] = self._volatility_signal(prices)
        
        # 2. ML Model Predictions
        if ML_AVAILABLE and self.models_trained:
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            rf_pred = self.rf_model.predict(features_scaled)[0]
            gb_pred = self.gb_model.predict(features_scaled)[0]
            
            components['ml_rf'] = np.tanh(rf_pred * 20)  # Scale to -1, 1
            components['ml_gb'] = np.tanh(gb_pred * 20)
        else:
            components['ml_rf'] = 0.0
            components['ml_gb'] = 0.0
        
        # 3. Ensemble: Weighted combination
        weights = {
            'momentum': 0.20,
            'mean_reversion': 0.15,
            'trend': 0.20,
            'volatility': 0.10,
            'ml_rf': 0.20,
            'ml_gb': 0.15
        }
        
        signal = sum(components[k] * weights[k] for k in weights.keys())
        signal = float(np.clip(signal, -1, 1))
        
        # Confidence: How much components agree
        component_signs = [np.sign(v) for v in components.values() if v != 0]
        if component_signs:
            agreement = abs(sum(component_signs) / len(component_signs))
            confidence = float(min(abs(signal) * agreement, 1.0))
        else:
            confidence = 0.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'components': components,
            'timestamp': datetime.now()
        }
    
    def _momentum_signal(self, prices: pd.Series) -> float:
        """Momentum-based signal."""
        if len(prices) < 20:
            return 0.0
        
        returns_20 = (prices.iloc[-1] / prices.iloc[-20]) - 1
        return float(np.tanh(returns_20 * 10))
    
    def _mean_reversion_signal(self, prices: pd.Series) -> float:
        """Mean reversion signal."""
        if len(prices) < 20:
            return 0.0
        
        mean = prices.rolling(20).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]
        
        if std == 0:
            return 0.0
        
        z_score = (prices.iloc[-1] - mean) / std
        return float(-np.tanh(z_score))  # Negative: buy when oversold
    
    def _trend_signal(self, prices: pd.Series) -> float:
        """Trend-following signal."""
        if len(prices) < 50:
            return 0.0
        
        ma_20 = prices.rolling(20).mean().iloc[-1]
        ma_50 = prices.rolling(50).mean().iloc[-1]
        
        if ma_50 == 0:
            return 0.0
        
        trend = (ma_20 - ma_50) / ma_50
        return float(np.tanh(trend * 20))
    
    def _volatility_signal(self, prices: pd.Series) -> float:
        """Volatility-adjusted signal."""
        if len(prices) < 20:
            return 0.0
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < 20:
            return 0.0
        
        current_vol = returns.iloc[-5:].std()
        historical_vol = returns.iloc[-20:].std()
        
        if historical_vol == 0:
            return 0.0
        
        # Prefer low volatility
        vol_ratio = current_vol / historical_vol
        return float(-np.tanh((vol_ratio - 1) * 2))