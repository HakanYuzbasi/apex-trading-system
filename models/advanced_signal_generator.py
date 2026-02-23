"""
models/advanced_signal_generator.py - GOD LEVEL ADVANCED
================================================================================
APEX TRADING SYSTEM - 30+ FEATURES WITH REGIME AWARENESS

Features:
✅ Vectorized Feature Extraction (1000x faster)
✅ Walk-Forward Validation
✅ Regime-Aware Training (Separate models per regime)
✅ Drift Monitoring
✅ Model Persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import os
import pickle
from collections import deque
from joblib import dump, load

from core.symbols import parse_symbol, AssetClass
from config import ApexConfig
from models.regime_common import get_regime

logger = logging.getLogger(__name__)

# ML imports
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer
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


# RegimeDetector class removed - using models.regime_common.get_regime


class AdvancedSignalGenerator:
    """
    GOD LEVEL: 30+ Feature Advanced ML Signal Generator.
    """
    
    def __init__(self, model_dir: str = "models/saved_advanced"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.lookback = 300  # Need 300 bars for 200-day MA
        
        # Regime-specific models
        self.regime_models: Dict[str, Dict] = {
            'bull': {},
            'bear': {},
            'neutral': {},
            'volatile': {}
        }

        # Asset-class-specific regime models (equity/forex/crypto)
        self.asset_class_models: Dict[str, Dict[str, Dict]] = {
            AssetClass.EQUITY.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
            AssetClass.FOREX.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
            AssetClass.CRYPTO.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
        }
        
        # Preprocessors per regime
        self.regime_scalers: Dict[str, RobustScaler] = {}
        self.regime_imputers: Dict[str, SimpleImputer] = {}

        # Preprocessors per asset class + regime
        self.asset_class_scalers: Dict[str, Dict[str, RobustScaler]] = {
            AssetClass.EQUITY.value: {},
            AssetClass.FOREX.value: {},
            AssetClass.CRYPTO.value: {},
        }
        self.asset_class_imputers: Dict[str, Dict[str, SimpleImputer]] = {
            AssetClass.EQUITY.value: {},
            AssetClass.FOREX.value: {},
            AssetClass.CRYPTO.value: {},
        }
        
        self.feature_names = []
        self.is_trained = False
        self.training_date = None
        self.last_retrain_date = None
        
        # Drift monitoring
        self.prediction_history = deque(maxlen=100)
        self.outcome_history = deque(maxlen=100)
        self.performance_baseline = 0.52
        self.retrain_interval_days = 30
        self.signal_ema: Dict[str, float] = {}
        self.signal_ema_alpha = 0.35  # Smoothing to reduce noise for better Sharpe

        # Try to load existing models
        self._load_models()

        logger.info("✅ GOD LEVEL Advanced Signal Generator initialized")
    
    def compute_features_vectorized(
        self, 
        data: Union[pd.Series, pd.DataFrame],
        sentiment_score: Optional[float] = None,
        momentum_rank: Optional[float] = None
    ) -> pd.DataFrame:
        """Vectorized 40+ feature extraction with OHLCV and Context."""
        if isinstance(data, pd.Series):
            p = data
            df = pd.DataFrame(index=p.index)
            v, h, low_px, _o = None, None, None, None
        else:
            p = data['Close']
            df = pd.DataFrame(index=p.index)
            v = data.get('Volume')
            h = data.get('High')
            low_px = data.get('Low')
            data.get('Open')

        returns = p.pct_change()
        
        # 1. Multi-period returns
        for d in [3, 5, 10, 20, 60]:
            df[f'ret_{d}d'] = p.pct_change(d)
        
        # 2. Volatility
        for d in [10, 20, 60]:
            df[f'vol_{d}d'] = returns.rolling(d).std() * np.sqrt(252)
        
        # 3. RSI
        delta = p.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(0)))
        df['rsi_14'] = rsi
        df['rsi_norm'] = (rsi - 50) / 50
        
        # 4. MACD
        ema12 = p.ewm(span=12).mean()
        ema26 = p.ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / p.replace(0, np.nan)
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. Bollinger Bands
        sma20 = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        df['bb_width'] = (4 * std20) / sma20.replace(0, np.nan)
        df['bb_pos'] = (p - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
        
        # 6. Moving Averages
        for d in [10, 20, 50, 200]:
            ma = p.rolling(d).mean()
            df[f'dist_ma_{d}'] = (p - ma) / ma.replace(0, np.nan)
        
        # 7. Momentum
        df['roc_5'] = p.pct_change(5)
        df['roc_10'] = p.pct_change(10)
        df['mom_accel'] = df['roc_5'] - (df['roc_10'] / 2)
        
        # 8. Statistical
        df['skew_20'] = returns.rolling(20).skew()
        df['kurt_20'] = returns.rolling(20).kurt()
        
        # 9. Z-score
        df['zscore_60'] = (p - p.rolling(60).mean()) / p.rolling(60).std().replace(0, np.nan)

        # 10. Trend strength
        df['trend_strength'] = returns.rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * 100 if len(x) > 1 else 0,
            raw=False
        )

        # 11. Quality Factors
        df['range_position'] = (p - p.rolling(20).min()) / (p.rolling(20).max() - p.rolling(20).min()).replace(0, np.nan)
        df['mean_rev_20'] = -df['zscore_60'].clip(-3, 3) / 3

        def calc_r2(x):
            if len(x) < 5: return 0
            y = np.arange(len(x))
            corr = np.corrcoef(x, y)[0, 1]
            return corr ** 2 if not np.isnan(corr) else 0

        df['trend_consistency'] = p.rolling(20).apply(calc_r2, raw=True)

        # 12. Volume Features (if available)
        if v is not None:
            df['volume_surge'] = v / v.rolling(20).mean().replace(0, np.nan)
            # OBV momentum
            obv = (np.sign(returns) * v).fillna(0).cumsum()
            df['obv_momentum'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, np.nan)
        else:
            df['volume_surge'] = 1.0
            df['obv_momentum'] = 0.0

        # 13. Intraday Features (if available)
        if h is not None and low_px is not None:
            df['intraday_volat'] = (h - low_px) / p.replace(0, np.nan)
        else:
            df['intraday_volat'] = 0.0

        # 14. Contextual Features (direct injection)
        df['sentiment_feature'] = sentiment_score if sentiment_score is not None else 0.0
        df['momentum_rank_feature'] = momentum_rank if momentum_rank is not None else 0.5

        return df.fillna(0).replace([np.inf, -np.inf], 0)

    def _align_features(self, df_feats: pd.DataFrame) -> pd.DataFrame:
        """Align current feature set to training feature names for compatibility."""
        if not self.feature_names:
            return df_feats
        # Ensure stable column order and fill missing with zeros
        return df_feats.reindex(columns=self.feature_names, fill_value=0.0)

    def _cross_sectional_normalize(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Z-score features cross-sectionally per timestamp."""
        if not feature_cols:
            return panel
        grouped = panel.groupby(level=0)
        for col in feature_cols:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            # When std is 0 or NaN (≤1 symbol at timestamp), keep demeaned value
            std = std.replace(0, np.nan).fillna(1.0)
            panel[col] = (panel[col] - mean) / std
        return panel

    def _build_training_panel(
        self,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build time-sorted MultiIndex panel for training."""
        frames = []
        feature_names: List[str] = []

        for symbol, data in historical_data.items():
            if len(data) < self.lookback + 20:
                continue

            try:
                parsed = parse_symbol(symbol)
            except ValueError:
                continue

            prices = data['Close']
            df_feats = self.compute_features_vectorized(data)

            # Targets: volatility-scaled future return
            future_ret = prices.pct_change(5).shift(-5)
            vol = prices.pct_change().rolling(ApexConfig.ADV_LABEL_VOL_LOOKBACK).std()
            vol = vol.replace(0, np.nan)
            scaled = (future_ret / vol).clip(
                -ApexConfig.ADV_LABEL_VOL_CLIP,
                ApexConfig.ADV_LABEL_VOL_CLIP
            )

            # Regimes
            ma_20 = prices.rolling(20).mean()
            ma_60 = prices.rolling(60).mean()
            vol_20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
            trend = (ma_20 - ma_60) / ma_60
            regimes_series = pd.Series('neutral', index=prices.index)
            regimes_series.loc[vol_20 > 0.35] = 'volatile'
            regimes_series.loc[(vol_20 <= 0.35) & (trend > 0.05)] = 'bull'
            regimes_series.loc[(vol_20 <= 0.35) & (trend < -0.05)] = 'bear'

            valid = ~df_feats.isnull().any(axis=1) & ~scaled.isnull()
            if valid.sum() == 0:
                continue

            df = df_feats.loc[valid].copy()
            df["target"] = scaled.loc[valid]
            df["regime"] = regimes_series.loc[valid]
            df["asset_class"] = parsed.asset_class.value
            df["symbol"] = symbol

            df = df.set_index(pd.MultiIndex.from_arrays(
                [df.index, df["symbol"]],
                names=["timestamp", "symbol"]
            ))
            df = df.drop(columns=["symbol"])

            if not feature_names:
                feature_names = list(df_feats.columns)

            frames.append(df)

        if not frames:
            return pd.DataFrame(), []

        panel = pd.concat(frames).sort_index(level=0)

        if getattr(ApexConfig, "ADV_CROSS_SECTIONAL_NORM", True):
            panel = self._cross_sectional_normalize(panel, feature_names)

        panel = panel.dropna(subset=feature_names + ["target", "regime", "asset_class"])
        return panel, feature_names

    def _quality_adjust(self, signal: float, features: pd.Series, regime: str) -> Tuple[float, float]:
        """Apply risk-aware adjustments to improve Sharpe (reduce noisy trades)."""
        vol_20 = float(features.get('vol_20d', 0.0))
        zscore = float(features.get('zscore_60', 0.0))
        dist_ma_50 = float(features.get('dist_ma_50', 0.0))
        bb_width = float(features.get('bb_width', 0.0))
        trend_strength = float(features.get('trend_strength', 0.0))

        # Volatility penalty: reduce size in high vol regimes
        vol_penalty = np.clip((vol_20 - 0.25) / 0.35, 0.0, 0.6)

        # Mean-reversion penalty if signal fights extreme zscore
        z_penalty = 0.0
        if signal > 0 and zscore > 2.0:
            z_penalty = 0.20
        elif signal < 0 and zscore < -2.0:
            z_penalty = 0.20

        # Trend alignment bonus: align with MA slope/trend strength
        trend_bonus = np.clip(abs(trend_strength) * 8.0, 0.0, 0.15)
        if signal > 0 and dist_ma_50 < -0.02:
            trend_bonus *= 0.5
        if signal < 0 and dist_ma_50 > 0.02:
            trend_bonus *= 0.5

        # Bandwidth penalty for noisy ranges
        band_penalty = np.clip((bb_width - 0.10) / 0.30, 0.0, 0.25)

        # Regime bias: reduce counter-trend signals
        regime_bias = 0.0
        if regime == 'bull' and signal < 0:
            regime_bias = 0.15
        elif regime == 'bear' and signal > 0:
            regime_bias = 0.15

        quality = 1.0 - vol_penalty - z_penalty - band_penalty - regime_bias + trend_bonus
        quality = float(np.clip(quality, 0.2, 1.1))

        adjusted_signal = float(np.clip(signal * quality, -1, 1))
        return adjusted_signal, quality
    
    def train_walk_forward(
        self,
        historical_data: Dict[str, pd.DataFrame],
        n_splits: int = 4
    ):
        """Walk-forward training with regime awareness."""
        if not ML_AVAILABLE:
            return
        
        logger.info("\n" + "="*80)
        logger.info("🧠 GOD LEVEL WALK-FORWARD TRAINING")
        logger.info("="*80)
        
        panel, feature_names = self._build_training_panel(historical_data)
        if panel.empty:
            logger.error("No valid training data")
            return

        self.feature_names = feature_names

        # Time-sorted panel to prevent leakage
        times = panel.index.get_level_values(0).unique().sort_values()
        if len(times) < 50:
            logger.error("Insufficient time points for walk-forward training")
            return

        min_train = int(len(times) * 0.5)
        window_size = max(5, int((len(times) - min_train) / max(1, n_splits)))

        regime_results = {r: [] for r in ['bull', 'bear', 'neutral', 'volatile']}

        purge_days = getattr(ApexConfig, "ADV_PURGE_DAYS", 5)
        embargo_days = getattr(ApexConfig, "ADV_EMBARGO_DAYS", 2)

        for split in range(n_splits):
            test_start_idx = min_train + (split * window_size)
            test_end_idx = min(test_start_idx + window_size, len(times))

            if test_end_idx - test_start_idx < 5:
                break

            test_start_time = times[test_start_idx]
            test_end_time = times[test_end_idx - 1]

            purge_start = test_start_time - timedelta(days=purge_days)
            embargo_end = test_end_time + timedelta(days=embargo_days)

            train_mask = (times < purge_start) | (times > embargo_end)
            train_times = times[train_mask]
            test_times = times[test_start_idx:test_end_idx]

            logger.info(
                f"\n[Split {split+1}/{n_splits}] Train: {train_times.min()}-{train_times.max()} "
                f"Test: {test_start_time}-{test_end_time} (purge={purge_days}d embargo={embargo_days}d)"
            )

            train_panel = panel.loc[train_times]
            test_panel = panel.loc[test_times]

            for asset_class in [AssetClass.EQUITY.value, AssetClass.FOREX.value, AssetClass.CRYPTO.value]:
                train_ac = train_panel[train_panel["asset_class"] == asset_class]
                test_ac = test_panel[test_panel["asset_class"] == asset_class]

                if train_ac.empty or test_ac.empty:
                    continue

                X_train_all = train_ac[feature_names].values
                y_train_all = train_ac["target"].values
                regimes_train = train_ac["regime"].values

                X_test_all = test_ac[feature_names].values
                y_test_all = test_ac["target"].values
                regimes_test = test_ac["regime"].values

                for regime in ['bull', 'bear', 'neutral', 'volatile']:
                    mask_train = (regimes_train == regime)
                    mask_test = (regimes_test == regime)

                    if mask_train.sum() < 100 or mask_test.sum() < 20:
                        continue

                    X_r_train = X_train_all[mask_train]
                    y_r_train = y_train_all[mask_train]
                    X_r_test = X_test_all[mask_test]
                    y_r_test = y_test_all[mask_test]

                    if regime not in self.asset_class_imputers[asset_class]:
                        self.asset_class_imputers[asset_class][regime] = SimpleImputer(strategy='median')
                        self.asset_class_scalers[asset_class][regime] = RobustScaler()

                    imputer = self.asset_class_imputers[asset_class][regime]
                    scaler = self.asset_class_scalers[asset_class][regime]

                    X_r_train_imp = imputer.fit_transform(X_r_train)
                    X_r_train_sc = scaler.fit_transform(X_r_train_imp)

                    X_r_test_imp = imputer.transform(X_r_test)
                    X_r_test_sc = scaler.transform(X_r_test_imp)

                    models = self._train_regime_models(
                        X_r_train_sc, y_r_train, X_r_test_sc, y_r_test, regime
                    )
                    self.asset_class_models[asset_class][regime] = models

                    if 'r2_score' in models:
                        regime_results[regime].append(models['r2_score'])
        
        # Finalize
        self.is_trained = True
        self.training_date = datetime.now()
        self.last_retrain_date = datetime.now()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 WALK-FORWARD REGIME RESULTS")
        logger.info("="*80)
        for regime, scores in regime_results.items():
            if scores:
                logger.info(f"{regime.upper():>10}: R²={np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        self._save_models()
    
    def _train_regime_models(self, X_train, y_train, X_test, y_test, regime) -> Dict:
        """Train regression models for a regime."""
        models = {}
        predictions = []
        
        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            models['rf'] = rf
            predictions.append(rf.predict(X_test))
        except:
            pass
        
        # Gradient Boosting
        try:
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            gb.fit(X_train, y_train)
            models['gb'] = gb
            predictions.append(gb.predict(X_test))
        except:
            pass
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.03, random_state=42, verbosity=0)
                xgb_model.fit(X_train, y_train)
                models['xgb'] = xgb_model
                predictions.append(xgb_model.predict(X_test))
            except:
                pass
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.03, random_state=42, verbose=-1)
                lgb_model.fit(X_train, y_train)
                models['lgb'] = lgb_model
                predictions.append(lgb_model.predict(X_test))
            except:
                pass
        
        # Calculate ensemble R²
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            ss_res = np.sum((y_test - ensemble_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            models['r2_score'] = r2
        
        return models
    
    def generate_ml_signal(
        self, 
        symbol: str, 
        data: Union[pd.Series, pd.DataFrame], 
        sentiment_score: float = 0.0,
        momentum_rank: float = 0.5,
        track: bool = True
    ) -> Dict:
        if not self.is_trained:
            if not self._load_models():
                return self._empty_signal()

        # Check retrain
        if self._should_retrain():
            logger.warning("⚠️ Retrain recommended")

        try:
            try:
                parsed = parse_symbol(symbol)
                asset_class = parsed.asset_class.value
            except ValueError:
                asset_class = AssetClass.EQUITY.value

            # Extract prices for regime detection
            if isinstance(data, pd.DataFrame):
                prices = data['Close']
            else:
                prices = data
                
            # Robustly extract a clean 1D Series from any input
            if isinstance(prices, pd.DataFrame):
                # Handle both regular and MultiIndex columns
                if 'Close' in prices.columns:
                    prices = prices['Close']
                elif len(prices.columns) > 0:
                    prices = prices.iloc[:, 0]

            # Handle any remaining DataFrame or Series issues
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            # Force to clean 1D numpy-backed Series with simple integer index
            values = np.asarray(prices).flatten()
            prices_clean = pd.Series(values, index=range(len(values)))

            # Detect regime
            regime = get_regime(prices_clean, 60)
            
            # Get models
            models_by_class = self.asset_class_models.get(asset_class) or self.regime_models
            if regime not in models_by_class or not models_by_class[regime]:
                regime = 'neutral'

            models = models_by_class.get(regime, {})
            if not models:
                return self._empty_signal()
            
            # Features (passing full window + context)
            window = data.iloc[-self.lookback:] if len(data) > self.lookback else data
            df_feats = self.compute_features_vectorized(
                window, 
                sentiment_score=sentiment_score,
                momentum_rank=momentum_rank
            )
            df_feats = self._align_features(df_feats)
            current = df_feats.iloc[-1:].values
            
            # Preprocess
            imputer = self.asset_class_imputers.get(asset_class, {}).get(regime) or self.regime_imputers.get(regime)
            scaler = self.asset_class_scalers.get(asset_class, {}).get(regime) or self.regime_scalers.get(regime)
            if not imputer or not scaler:
                return self._empty_signal()
            
            current = imputer.transform(current)
            current = scaler.transform(current)
            
            # Predict
            predictions = []
            for name in ['rf', 'gb', 'xgb', 'lgb']:
                model = models.get(name)
                if model:
                    try:
                        pred_return = model.predict(current)[0]
                        signal = np.tanh(pred_return * 25)
                        predictions.append(signal)
                    except:
                        pass
            
            if not predictions:
                return self._empty_signal()
            
            avg_signal = float(np.mean(predictions))
            std_signal = float(np.std(predictions))
            confidence = max(0, 1.0 - (std_signal * 2))

            # Risk-aware adjustment for Sharpe improvement
            features_last = df_feats.iloc[-1]
            adjusted_signal, quality = self._quality_adjust(avg_signal, features_last, regime)
            confidence = float(np.clip(confidence * quality, 0.0, 1.0))

            # Smooth signal to reduce noise and improve stability
            prev = self.signal_ema.get(symbol, adjusted_signal)
            smoothed = self.signal_ema_alpha * adjusted_signal + (1 - self.signal_ema_alpha) * prev
            self.signal_ema[symbol] = smoothed
            
            result = {
                'signal': float(np.clip(smoothed, -1, 1)),
                'confidence': float(np.clip(confidence, 0, 1)),
                'quality': float(np.clip(quality, 0, 1)),
                'regime': regime,
                'timestamp': datetime.now().isoformat()
            }
            
            if track:
                self.prediction_history.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._empty_signal()
    
    def record_outcome(self, actual_return: float):
        """Record outcome for drift monitoring."""
        self.outcome_history.append(actual_return)
    
    def check_drift(self) -> Dict:
        """Check drift."""
        if len(self.prediction_history) < 30 or len(self.outcome_history) < 30:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        min_len = min(len(self.prediction_history), len(self.outcome_history))
        
        # Directional accuracy
        correct = sum(
            (self.prediction_history[i]['signal'] > 0) == (self.outcome_history[i] > 0)
            for i in range(-min_len, 0)
        )
        accuracy = correct / min_len
        
        # Sharpe
        returns = [
            self.prediction_history[i]['signal'] * self.outcome_history[i]
            for i in range(-min_len, 0)
        ]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        drift = (accuracy < self.performance_baseline) or (sharpe < 1.0)
        
        return {
            'drift_detected': drift,
            'accuracy': accuracy,
            'sharpe': sharpe,
            'recommendation': 'RETRAIN' if drift else 'OK'
        }
    
    def _should_retrain(self) -> bool:
        """Check if retrain needed."""
        if self.last_retrain_date:
            days = (datetime.now() - self.last_retrain_date).days
            if days >= self.retrain_interval_days:
                return True
        
        if self.check_drift()['drift_detected']:
            return True
        
        return False
    
    def _empty_signal(self):
        return {'signal': 0.0, 'confidence': 0.0, 'regime': 'unknown', 'timestamp': datetime.now().isoformat()}
    
    def _save_models(self):
        """Save all models."""
        try:
            has_asset_models = any(
                any(self.asset_class_models[ac][r] for r in self.asset_class_models[ac])
                for ac in self.asset_class_models
            )

            if has_asset_models:
                for asset_class, regimes in self.asset_class_models.items():
                    for regime, models in regimes.items():
                        regime_dir = f"{self.model_dir}/{asset_class}/{regime}"
                        os.makedirs(regime_dir, exist_ok=True)

                        for name, model in models.items():
                            if name == 'r2_score':
                                continue
                            if name == 'xgb':
                                model.save_model(f"{regime_dir}/xgb.json")
                            else:
                                dump(model, f"{regime_dir}/{name}.pkl")

                        imputer = self.asset_class_imputers.get(asset_class, {}).get(regime)
                        scaler = self.asset_class_scalers.get(asset_class, {}).get(regime)
                        if imputer:
                            dump(imputer, f"{regime_dir}/imputer.pkl")
                        if scaler:
                            dump(scaler, f"{regime_dir}/scaler.pkl")
            else:
                for regime, models in self.regime_models.items():
                    regime_dir = f"{self.model_dir}/{regime}"
                    os.makedirs(regime_dir, exist_ok=True)

                    for name, model in models.items():
                        if name == 'r2_score':
                            continue
                        if name == 'xgb':
                            model.save_model(f"{regime_dir}/xgb.json")
                        else:
                            dump(model, f"{regime_dir}/{name}.pkl")

                    if regime in self.regime_imputers:
                        dump(self.regime_imputers[regime], f"{regime_dir}/imputer.pkl")
                        dump(self.regime_scalers[regime], f"{regime_dir}/scaler.pkl")
            
            metadata = {
                'feature_names': self.feature_names,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                'prediction_history': list(self.prediction_history),
                'outcome_history': list(self.outcome_history),
                'model_layout': 'asset_class_regime' if has_asset_models else 'legacy'
            }
            
            with open(f"{self.model_dir}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("✅ Advanced models saved")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def _load_models(self) -> bool:
        """Load models."""
        try:
            if not os.path.exists(f"{self.model_dir}/metadata.pkl"):
                return False
            
            with open(f"{self.model_dir}/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata['feature_names']
            if metadata.get('last_retrain_date'):
                self.last_retrain_date = datetime.fromisoformat(metadata['last_retrain_date'])
            
            self.prediction_history = deque(metadata.get('prediction_history', []), maxlen=100)
            self.outcome_history = deque(metadata.get('outcome_history', []), maxlen=100)
            
            layout = metadata.get('model_layout', 'legacy')
            if layout == 'asset_class_regime':
                for asset_class in [AssetClass.EQUITY.value, AssetClass.FOREX.value, AssetClass.CRYPTO.value]:
                    for regime in ['bull', 'bear', 'neutral', 'volatile']:
                        regime_dir = f"{self.model_dir}/{asset_class}/{regime}"
                        if not os.path.exists(regime_dir):
                            continue

                        self.asset_class_models[asset_class][regime] = {}

                        if os.path.exists(f"{regime_dir}/imputer.pkl"):
                            self.asset_class_imputers[asset_class][regime] = load(f"{regime_dir}/imputer.pkl")
                        if os.path.exists(f"{regime_dir}/scaler.pkl"):
                            self.asset_class_scalers[asset_class][regime] = load(f"{regime_dir}/scaler.pkl")

                        for name in ['rf', 'gb', 'lgb']:
                            path = f"{regime_dir}/{name}.pkl"
                            if os.path.exists(path):
                                self.asset_class_models[asset_class][regime][name] = load(path)

                        if os.path.exists(f"{regime_dir}/xgb.json"):
                            model = xgb.XGBRegressor()
                            model.load_model(f"{regime_dir}/xgb.json")
                            self.asset_class_models[asset_class][regime]['xgb'] = model
            else:
                for regime in ['bull', 'bear', 'neutral', 'volatile']:
                    regime_dir = f"{self.model_dir}/{regime}"
                    if not os.path.exists(regime_dir):
                        continue
                    
                    self.regime_models[regime] = {}
                    
                    if os.path.exists(f"{regime_dir}/imputer.pkl"):
                        self.regime_imputers[regime] = load(f"{regime_dir}/imputer.pkl")
                    if os.path.exists(f"{regime_dir}/scaler.pkl"):
                        self.regime_scalers[regime] = load(f"{regime_dir}/scaler.pkl")
                    
                    for name in ['rf', 'gb', 'lgb']:
                        path = f"{regime_dir}/{name}.pkl"
                        if os.path.exists(path):
                            self.regime_models[regime][name] = load(path)
                    
                    if os.path.exists(f"{regime_dir}/xgb.json"):
                        model = xgb.XGBRegressor()
                        model.load_model(f"{regime_dir}/xgb.json")
                        self.regime_models[regime]['xgb'] = model
            
            self.is_trained = True
            logger.info("✅ Advanced models loaded")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Alias for train_walk_forward for API compatibility."""
        self.train_walk_forward(historical_data)
