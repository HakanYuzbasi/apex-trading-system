"""
models/ensemble_signal_generator.py
5-MODEL ENSEMBLE FOR ROBUST SIGNALS
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Logistic Regression
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


class EnsembleSignalGenerator:
    """
    Ensemble of 5 ML models for robust trading signals.
    
    Models:
    1. Random Forest (baseline)
    2. Gradient Boosting (boosting)
    3. XGBoost (advanced boosting)
    4. LightGBM (fast boosting)
    5. Logistic Regression (linear baseline)
    
    Combines signals using weighted voting based on:
    - Historical performance
    - Model confidence
    - Model agreement (consensus)
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.is_trained = False
        
        self._initialize_models()
        logger.info("✅ Ensemble Signal Generator initialized (5 models)")
    
    def _initialize_models(self):
        """Initialize all models with default parameters."""
        # 1. Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # 3. XGBoost (if available)
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            logger.warning("⚠️ XGBoost not available, skipping")
        
        # 4. LightGBM (if available)
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
        else:
            logger.warning("⚠️ LightGBM not available, skipping")
        
        # 5. Logistic Regression (needs scaling)
        self.models['logistic'] = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scalers['logistic'] = StandardScaler()
        
        # Initialize weights (equal initially, will adjust based on performance)
        total_models = len(self.models)
        self.weights = {name: 1.0/total_models for name in self.models.keys()}
        
        logger.info(f"   Models: {list(self.models.keys())}")
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train all models in the ensemble.
        
        Args:
            X: Features
            y: Target (binary or multiclass)
        """
        logger.info(f"🧠 Training ensemble on {len(X)} samples...")
        
        trained_models = []
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"   Training {model_name}...")
                
                # Scale features for logistic regression
                if model_name == 'logistic':
                    X_scaled = self.scalers['logistic'].fit_transform(X)
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                # Calculate training accuracy
                if model_name == 'logistic':
                    train_score = model.score(X_scaled, y)
                else:
                    train_score = model.score(X, y)
                
                logger.info(f"      ✅ {model_name}: {train_score:.3f} accuracy")
                trained_models.append(model_name)
                
            except Exception as e:
                logger.error(f"      ❌ {model_name} failed: {e}")
                # Remove failed model
                del self.models[model_name]
                if model_name in self.weights:
                    del self.weights[model_name]
        
        # Renormalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        logger.info(f"✅ Ensemble training complete ({len(trained_models)}/{len(self.models)} models)")
    
    def generate_signal(self, X: pd.DataFrame) -> Dict:
        """
        Generate trading signal from ensemble.
        
        Args:
            X: Features (single row or multiple)
        
        Returns:
            {
                'signal': float (-1 to 1),
                'confidence': float (0 to 1),
                'consensus': float (0 to 1),
                'individual_signals': dict,
                'individual_confidences': dict
            }
        """
        if not self.is_trained:
            logger.warning("⚠️ Ensemble not trained yet!")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'consensus': 0.0,
                'individual_signals': {},
                'individual_confidences': {}
            }
        
        signals = {}
        confidences = {}
        probabilities = {}
        
        # Get prediction from each model
        for model_name, model in self.models.items():
            try:
                # Scale features for logistic regression
                if model_name == 'logistic':
                    X_input = self.scalers['logistic'].transform(X)
                else:
                    X_input = X
                
                # Get probabilities
                proba = model.predict_proba(X_input)
                
                # Extract probabilities for last sample
                if len(proba.shape) > 1:
                    proba_last = proba[-1] if len(proba.shape) == 2 else proba
                else:
                    proba_last = proba
                
                # Calculate signal: -1 (sell) to +1 (buy)
                if len(proba_last) == 2:  # Binary classification
                    signal = proba_last[1] - proba_last[0]  # P(buy) - P(sell)
                else:  # Multiclass
                    # Weight by class (-2, -1, 0, 1, 2)
                    classes = np.arange(len(proba_last)) - len(proba_last)//2
                    signal = np.dot(proba_last, classes) / len(proba_last)
                
                signals[model_name] = float(signal)
                
                # Confidence: how sure is the model?
                confidence = float(np.max(proba_last) - 0.5) * 2  # 0 to 1 scale
                confidences[model_name] = max(0, confidence)
                
                probabilities[model_name] = proba_last
                
            except Exception as e:
                logger.warning(f"   {model_name} prediction failed: {e}")
                signals[model_name] = 0.0
                confidences[model_name] = 0.0
        
        # Calculate weighted ensemble signal
        ensemble_signal = sum(
            signals[m] * self.weights[m]
            for m in signals.keys()
        )
        
        # Calculate weighted ensemble confidence
        ensemble_confidence = sum(
            confidences[m] * self.weights[m]
            for m in confidences.keys()
        )
        
        # Calculate consensus (what % of models agree on direction)
        positive_signals = sum(1 for s in signals.values() if s > 0.1)
        negative_signals = sum(1 for s in signals.values() if s < -0.1)
        neutral_signals = len(signals) - positive_signals - negative_signals
        
        consensus = max(positive_signals, negative_signals) / len(signals)
        
        # Normalize signal to -1..1
        ensemble_signal = np.tanh(ensemble_signal)
        
        return {
            'signal': ensemble_signal,
            'confidence': ensemble_confidence,
            'consensus': consensus,
            'individual_signals': signals,
            'individual_confidences': confidences,
            'num_bullish': positive_signals,
            'num_bearish': negative_signals,
            'num_neutral': neutral_signals,
            'probabilities': probabilities
        }
    
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Update model weights based on recent performance.
        
        Args:
            performance_metrics: {model_name: accuracy/sharpe/etc}
        """
        logger.info("🔄 Updating ensemble weights based on performance...")
        
        # Normalize metrics to weights
        total = sum(performance_metrics.values())
        if total > 0:
            new_weights = {
                model: metric / total
                for model, metric in performance_metrics.items()
                if model in self.models
            }
            
            # Smooth transition (70% old, 30% new)
            for model in new_weights:
                if model in self.weights:
                    self.weights[model] = 0.7 * self.weights[model] + 0.3 * new_weights[model]
            
            # Renormalize
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            logger.info("   New weights:")
            for model, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"      {model:20s}: {weight:.3f}")
    
    def get_model_importance(self, X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from all models."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            importance_df['mean'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('mean', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()


if __name__ == "__main__":
    # Test ensemble
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Binary target
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int))
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train ensemble
    ensemble = EnsembleSignalGenerator()
    ensemble.train(X_train, y_train)
    
    # Generate signals
    print("\n" + "="*60)
    print("TESTING SIGNAL GENERATION")
    print("="*60)
    
    for i in range(5):
        signal_data = ensemble.generate_signal(X_test.iloc[[i]])
        
        print(f"\nSample {i+1}:")
        print(f"  Signal: {signal_data['signal']:+.3f}")
        print(f"  Confidence: {signal_data['confidence']:.3f}")
        print(f"  Consensus: {signal_data['consensus']:.3f}")
        print(f"  Bullish: {signal_data['num_bullish']}, Bearish: {signal_data['num_bearish']}")
    
    # Test weight update
    print("\n" + "="*60)
    print("TESTING WEIGHT UPDATE")
    print("="*60)
    
    performance = {
        'random_forest': 0.65,
        'gradient_boosting': 0.70,
        'xgboost': 0.68,
        'lightgbm': 0.72,
        'logistic': 0.55
    }
    
    ensemble.update_weights(performance)
    
    print("\n✅ Ensemble tests complete!")
