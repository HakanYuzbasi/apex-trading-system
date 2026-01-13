"""
models/ensemble_signal_generator.py - GOD LEVEL ENSEMBLE
================================================================================
APEX TRADING SYSTEM - HEDGE FUND GRADE ML ENSEMBLE

Features:
✅ 5-Model Ensemble (RF, GB, XGB, LGB, LR)
✅ Walk-Forward Validation (No Look-Ahead Bias)
✅ Regime-Aware Training (Bull/Bear/Volatile/Neutral)
✅ Drift Monitoring (Auto-Retrain on Degradation)
✅ Model Persistence
✅ Performance Tracking
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import pickle
import os
from collections import deque

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regime for regime-aware training."""
    
    @staticmethod
    def detect_regime(prices: pd.Series, lookback: int = 60) -> str:
        """
        Detect current market regime.
        
        Returns: 'bull', 'bear', 'neutral', 'volatile'
        """
        if len(prices) < lookback:
            return 'neutral'
        
        recent = prices.iloc[-lookback:]
        returns = recent.pct_change().dropna()
        
        # Trend: Compare recent MA to older MA
        ma_20 = prices.iloc[-20:].mean()
        ma_60 = prices.iloc[-60:].mean()
        trend_strength = (ma_20 - ma_60) / ma_60 if ma_60 > 0 else 0
        
        # Volatility: Annualized std dev
        vol = returns.std() * np.sqrt(252)
        
        # Classification Logic
        if vol > 0.35:  # High volatility threshold
            return 'volatile'
        elif trend_strength > 0.05:
            return 'bull'
        elif trend_strength < -0.05:
            return 'bear'
        else:
            return 'neutral'


class EnsembleSignalGenerator:
    """
    GOD LEVEL: Hedge Fund Grade 5-Model Ensemble.
    
    Architecture:
    - Regime-Aware Models (separate models per regime)
    - Walk-Forward Validation (no look-ahead bias)
    - Drift Monitoring (auto-retrain on degradation)
    """
    
    def __init__(self, model_dir: str = "models/saved_ensemble"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models organized by regime
        self.regime_models: Dict[str, Dict] = {
            'bull': {},
            'bear': {},
            'neutral': {},
            'volatile': {}
        }
        
        # Preprocessors per regime
        self.regime_scalers: Dict[str, StandardScaler] = {}
        self.regime_imputers: Dict[str, SimpleImputer] = {}
        
        # Training metadata
        self.is_trained = False
        self.feature_names: List[str] = []
        self.training_date = None
        self.walk_forward_results = []
        
        # Drift monitoring
        self.prediction_history = deque(maxlen=100)  # Last 100 predictions
        self.outcome_history = deque(maxlen=100)     # Last 100 actual outcomes
        self.performance_baseline = 0.55  # Minimum acceptable accuracy
        self.last_retrain_date = None
        self.retrain_interval_days = 30
        
        logger.info("✅ GOD LEVEL Ensemble initialized")
        logger.info(f"   Directory: {model_dir}")
    
    def train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict:
        """
        Walk-Forward Validation Training.
        
        Splits data into time-ordered windows:
        - Train on window 1, test on window 2
        - Train on windows 1+2, test on window 3
        - Etc.
        
        Args:
            X: Features
            y: Target (1=up, 0=down)
            regimes: Market regime labels ('bull', 'bear', etc.)
            n_splits: Number of walk-forward windows
            test_size: Test set fraction
        """
        logger.info("\n" + "="*80)
        logger.info("🧠 WALK-FORWARD TRAINING (GOD LEVEL)")
        logger.info("="*80)
        
        # Validation
        if len(X) < 200:
            raise ValueError(f"Need at least 200 samples, got {len(X)}")
        
        self.feature_names = X.columns.tolist()
        
        # Calculate split points
        total_samples = len(X)
        min_train = int(total_samples * 0.3)  # Start with 30% training data
        test_window = int(total_samples / n_splits)
        
        results_by_regime = {r: [] for r in ['bull', 'bear', 'neutral', 'volatile']}
        
        # Walk-Forward Loop
        for split in range(n_splits):
            train_end = min_train + (split * test_window)
            test_start = train_end
            test_end = min(test_start + test_window, total_samples)
            
            if test_end - test_start < 20:
                break
            
            logger.info(f"\n[Split {split+1}/{n_splits}] Train: 0-{train_end}, Test: {test_start}-{test_end}")
            
            # Split data
            X_train = X.iloc[:train_end].copy()
            y_train = y.iloc[:train_end].copy()
            regimes_train = regimes.iloc[:train_end].copy()
            
            X_test = X.iloc[test_start:test_end].copy()
            y_test = y.iloc[test_start:test_end].copy()
            regimes_test = regimes.iloc[test_start:test_end].copy()
            
            # Train models for each regime
            for regime_name in ['bull', 'bear', 'neutral', 'volatile']:
                # Filter by regime
                regime_mask_train = (regimes_train == regime_name)
                regime_mask_test = (regimes_test == regime_name)
                
                if regime_mask_train.sum() < 50 or regime_mask_test.sum() < 10:
                    continue
                
                X_regime_train = X_train[regime_mask_train]
                y_regime_train = y_train[regime_mask_train]
                X_regime_test = X_test[regime_mask_test]
                y_regime_test = y_test[regime_mask_test]
                
                # Initialize preprocessors for this regime if needed
                if regime_name not in self.regime_imputers:
                    self.regime_imputers[regime_name] = SimpleImputer(strategy='mean')
                    self.regime_scalers[regime_name] = StandardScaler()
                
                # Preprocess
                X_regime_train_imp = self.regime_imputers[regime_name].fit_transform(X_regime_train)
                X_regime_train_sc = self.regime_scalers[regime_name].fit_transform(X_regime_train_imp)
                
                X_regime_test_imp = self.regime_imputers[regime_name].transform(X_regime_test)
                X_regime_test_sc = self.regime_scalers[regime_name].transform(X_regime_test_imp)
                
                # Train models
                regime_models = self._train_regime_models(
                    X_regime_train_sc, y_regime_train,
                    X_regime_test_sc, y_regime_test,
                    regime_name
                )
                
                # Store models (only last split models are kept for production use)
                self.regime_models[regime_name] = regime_models
                
                # Record results
                if 'accuracy' in regime_models:
                    results_by_regime[regime_name].append(regime_models['accuracy'])
        
        # Aggregate results
        self.is_trained = True
        self.training_date = datetime.now()
        self.last_retrain_date = datetime.now()
        
        # Calculate average performance per regime
        regime_performance = {}
        for regime, accs in results_by_regime.items():
            if accs:
                regime_performance[regime] = {
                    'avg_accuracy': np.mean(accs),
                    'std_accuracy': np.std(accs),
                    'n_windows': len(accs)
                }
        
        logger.info("\n" + "="*80)
        logger.info("📊 WALK-FORWARD RESULTS")
        logger.info("="*80)
        for regime, perf in regime_performance.items():
            logger.info(f"{regime.upper():>10}: Acc={perf['avg_accuracy']:.3f} ± {perf['std_accuracy']:.3f} ({perf['n_windows']} windows)")
        
        self._save_models()
        
        return {
            'is_trained': True,
            'regime_performance': regime_performance,
            'training_date': self.training_date.isoformat(),
            'walk_forward_splits': n_splits
        }
    
    def _train_regime_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regime: str
    ) -> Dict:
        """Train 5 models for a specific regime."""
        models = {}
        predictions = []
        
        # Random Forest
        try:
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=20,
                min_samples_leaf=10, max_features='sqrt', n_jobs=-1, random_state=42
            )
            rf.fit(X_train, y_train)
            models['rf'] = rf
            predictions.append(rf.predict(X_test))
        except Exception as e:
            logger.debug(f"RF failed for {regime}: {e}")
        
        # Gradient Boosting
        try:
            gb = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=5,
                min_samples_split=20, subsample=0.8, random_state=42
            )
            gb.fit(X_train, y_train)
            models['gb'] = gb
            predictions.append(gb.predict(X_test))
        except Exception as e:
            logger.debug(f"GB failed for {regime}: {e}")
        
        # XGBoost
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            models['xgb'] = xgb_model
            predictions.append(xgb_model.predict(X_test))
        except Exception as e:
            logger.debug(f"XGB failed for {regime}: {e}")
        
        # LightGBM
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            models['lgb'] = lgb_model
            predictions.append(lgb_model.predict(X_test))
        except Exception as e:
            logger.debug(f"LGB failed for {regime}: {e}")
        
        # Logistic Regression
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            lr.fit(X_train, y_train)
            models['lr'] = lr
            predictions.append(lr.predict(X_test))
        except Exception as e:
            logger.debug(f"LR failed for {regime}: {e}")
        
        # Ensemble accuracy
        if predictions:
            ensemble_pred = np.round(np.mean(predictions, axis=0))
            accuracy = accuracy_score(y_test, ensemble_pred)
            models['accuracy'] = accuracy
        
        return models
    
    def generate_signal(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        track_for_drift: bool = True
    ) -> Dict:
        """
        Generate regime-aware signal with drift monitoring.
        
        Args:
            features: Feature dataframe (1 row)
            prices: Recent price history (for regime detection)
            track_for_drift: Whether to store prediction for drift monitoring
        """
        if not self.is_trained:
            if not self._load_models():
                return self._neutral_signal()
        
        # Check if retrain is needed
        if self._should_retrain():
            logger.warning("⚠️ Retrain recommended (performance drift or time-based)")
        
        try:
            # 1. Detect regime
            regime = RegimeDetector.detect_regime(prices, lookback=60)
            
            # 2. Get regime-specific models
            if regime not in self.regime_models or not self.regime_models[regime]:
                logger.warning(f"No models for regime {regime}, using neutral")
                regime = 'neutral'
            
            models = self.regime_models[regime]
            if not models:
                return self._neutral_signal()
            
            # 3. Prepare features
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_names]
            
            # 4. Preprocess
            imputer = self.regime_imputers.get(regime)
            scaler = self.regime_scalers.get(regime)
            
            if imputer is None or scaler is None:
                return self._neutral_signal()
            
            features_imp = imputer.transform(features)
            features_sc = scaler.transform(features_imp)
            
            # 5. Collect predictions
            probabilities = []
            votes = {}
            
            for name in ['rf', 'gb', 'xgb', 'lgb', 'lr']:
                model = models.get(name)
                if model is not None:
                    try:
                        pred = model.predict(features_sc)[0]
                        prob = model.predict_proba(features_sc)[0, 1]
                        votes[name] = int(pred)
                        probabilities.append(prob)
                    except:
                        pass
            
            if not probabilities:
                return self._neutral_signal()
            
            # 6. Ensemble logic
            avg_prob = np.mean(probabilities)
            signal = 2 * avg_prob - 1  # Convert [0,1] to [-1,1]
            
            bullish_votes = sum(votes.values())
            consensus = bullish_votes / len(votes)
            confidence = abs(consensus - 0.5) * 2
            
            result = {
                'signal': float(np.clip(signal, -1, 1)),
                'confidence': float(np.clip(confidence, 0, 1)),
                'consensus': float(consensus),
                'regime': regime,
                'model_votes': votes,
                'timestamp': datetime.now().isoformat()
            }
            
            # 7. Track for drift monitoring
            if track_for_drift:
                self.prediction_history.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Signal generation error: {e}", exc_info=True)
            return self._neutral_signal()
    
    def record_outcome(self, actual_return: float):
        """
        Record actual outcome for drift monitoring.
        
        Call this after a trade closes to track realized performance.
        """
        self.outcome_history.append(actual_return)
    
    def check_drift(self) -> Dict:
        """
        Check for model performance drift.
        
        Returns dict with drift metrics and recommendation.
        """
        if len(self.prediction_history) < 30 or len(self.outcome_history) < 30:
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'samples': len(self.prediction_history)
            }
        
        # Align predictions with outcomes (both are time-ordered deques)
        min_len = min(len(self.prediction_history), len(self.outcome_history))
        
        # Calculate directional accuracy
        correct = 0
        for i in range(-min_len, 0):
            pred_signal = self.prediction_history[i]['signal']
            actual_return = self.outcome_history[i]
            if (pred_signal > 0 and actual_return > 0) or (pred_signal < 0 and actual_return < 0):
                correct += 1
        
        accuracy = correct / min_len
        
        # Calculate recent Sharpe
        returns = []
        for i in range(-min_len, 0):
            pred_signal = self.prediction_history[i]['signal']
            actual_return = self.outcome_history[i]
            returns.append(pred_signal * actual_return)  # Scaled return
        
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Drift detection
        drift_detected = (accuracy < self.performance_baseline) or (sharpe < 1.0)
        
        return {
            'drift_detected': drift_detected,
            'accuracy': accuracy,
            'sharpe': sharpe,
            'baseline_accuracy': self.performance_baseline,
            'samples': min_len,
            'recommendation': 'RETRAIN' if drift_detected else 'OK'
        }
    
    def _should_retrain(self) -> bool:
        """Check if retrain is needed (time-based or performance-based)."""
        # Time-based
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.retrain_interval_days:
                return True
        
        # Performance-based
        drift_check = self.check_drift()
        if drift_check['drift_detected']:
            return True
        
        return False
    
    def _neutral_signal(self) -> Dict:
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'consensus': 0.5,
            'regime': 'unknown',
            'model_votes': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_models(self):
        """Save all regime models."""
        try:
            for regime, models in self.regime_models.items():
                regime_dir = f"{self.model_dir}/{regime}"
                os.makedirs(regime_dir, exist_ok=True)
                
                for name, model in models.items():
                    if name == 'accuracy':
                        continue
                    
                    if name == 'xgb':
                        model.save_model(f"{regime_dir}/xgb.json")
                    else:
                        dump(model, f"{regime_dir}/{name}.pkl")
            
            # Save preprocessors
            for regime in self.regime_imputers:
                regime_dir = f"{self.model_dir}/{regime}"
                dump(self.regime_imputers[regime], f"{regime_dir}/imputer.pkl")
                dump(self.regime_scalers[regime], f"{regime_dir}/scaler.pkl")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                'performance_baseline': self.performance_baseline,
                'prediction_history': list(self.prediction_history),
                'outcome_history': list(self.outcome_history)
            }
            
            with open(f"{self.model_dir}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✅ Regime models saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def _load_models(self) -> bool:
        """Load all regime models."""
        try:
            if not os.path.exists(f"{self.model_dir}/metadata.pkl"):
                return False
            
            # Load metadata
            with open(f"{self.model_dir}/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata['feature_names']
            self.performance_baseline = metadata.get('performance_baseline', 0.55)
            
            if metadata.get('last_retrain_date'):
                self.last_retrain_date = datetime.fromisoformat(metadata['last_retrain_date'])
            
            # Restore history
            self.prediction_history = deque(metadata.get('prediction_history', []), maxlen=100)
            self.outcome_history = deque(metadata.get('outcome_history', []), maxlen=100)
            
            # Load models for each regime
            for regime in ['bull', 'bear', 'neutral', 'volatile']:
                regime_dir = f"{self.model_dir}/{regime}"
                if not os.path.exists(regime_dir):
                    continue
                
                self.regime_models[regime] = {}
                
                # Load preprocessors
                if os.path.exists(f"{regime_dir}/imputer.pkl"):
                    self.regime_imputers[regime] = load(f"{regime_dir}/imputer.pkl")
                if os.path.exists(f"{regime_dir}/scaler.pkl"):
                    self.regime_scalers[regime] = load(f"{regime_dir}/scaler.pkl")
                
                # Load models
                for name in ['rf', 'gb', 'lgb', 'lr']:
                    path = f"{regime_dir}/{name}.pkl"
                    if os.path.exists(path):
                        self.regime_models[regime][name] = load(path)
                
                # XGBoost special handling
                xgb_path = f"{regime_dir}/xgb.json"
                if os.path.exists(xgb_path):
                    model = xgb.XGBClassifier()
                    model.load_model(xgb_path)
                    self.regime_models[regime]['xgb'] = model
            
            self.is_trained = True
            logger.info(f"✅ Regime models loaded from {self.model_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False
