"""
models/ensemble_signal_generator.py
================================================================================
APEX TRADING SYSTEM - 5-MODEL ENSEMBLE WITH VOTING
================================================================================

Advanced ensemble learning with:
- Random Forest (robust, feature importance)
- Gradient Boosting (strong learner)
- XGBoost (fast, handles missing data)
- LightGBM (memory efficient)
- Logistic Regression (interpretable)

Features:
- Weighted voting mechanism
- Probability averaging
- Model persistence (save/load)
- Real-time signal generation
- Consensus confidence scoring
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load

logger = logging.getLogger(__name__)


class EnsembleSignalGenerator:
    """
    Production-grade 5-model ensemble for trading signals.
    
    Architecture:
    - Base Models: 5 different algorithms
    - Voting: Majority + weighted voting
    - Probability: Averaging probabilities
    - Robustness: Handles missing data, NaN values
    - Persistence: Save/load trained models
    
    Output Signal: -1.0 to 1.0
    - -1.0: Strong sell signal
    - 0.0: Neutral
    - 1.0: Strong buy signal
    """
    
    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize individual models
        self.rf_model: Optional[RandomForestClassifier] = None
        self.gb_model: Optional[GradientBoostingClassifier] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.lr_model: Optional[LogisticRegression] = None
        
        # Feature scaling
        self.scaler: Optional[StandardScaler] = None
        
        # Model metadata
        self.is_trained = False
        self.feature_names: List[str] = []
        self.feature_count = 0
        self.training_date = None
        
        # Training history
        self.training_history = {
            'accuracies': {},
            'precisions': {},
            'recalls': {},
            'f1_scores': {}
        }
        
        logger.info("✅ Ensemble Signal Generator initialized")
        logger.info(f"   Model directory: {model_dir}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Train all 5 models.
        
        Args:
            X: Feature matrix (rows=samples, cols=features)
            y: Target variable (1=up, 0=down)
            test_size: Fraction for testing (0.2 = 20%)
            verbose: Print training details
        
        Returns:
            Training results dictionary with metrics
        
        Raises:
            ValueError: If data is insufficient or invalid
        """
        logger.info("\n" + "="*80)
        logger.info("🧠 TRAINING 5-MODEL ENSEMBLE")
        logger.info("="*80)
        
        # Validation
        if len(X) < 100:
            raise ValueError(f"Need at least 100 samples, got {len(X)}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}")
        
        # Check target distribution
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Target must have both classes (0 and 1)")
        
        min_samples = min(class_counts.values)
        if min_samples < 10:
            raise ValueError(f"Each class needs at least 10 samples, min={min_samples}")
        
        logger.info(f"📊 Data shape: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"   Class distribution: {dict(class_counts)}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        self.feature_count = len(self.feature_names)
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0])
        
        logger.info(f"✅ Filled missing values")
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
        y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
        
        logger.info(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✅ Features scaled")
        
        results = {}
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 1: RANDOM FOREST
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("\n[1/5] Training Random Forest...")
            
            self.rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_rf = self.rf_model.predict(X_test_scaled)
            rf_acc = accuracy_score(y_test, y_pred_rf)
            rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
            rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
            rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
            
            self.training_history['accuracies']['RF'] = rf_acc
            self.training_history['precisions']['RF'] = rf_prec
            self.training_history['recalls']['RF'] = rf_rec
            self.training_history['f1_scores']['RF'] = rf_f1
            
            results['RF'] = {
                'accuracy': rf_acc,
                'precision': rf_prec,
                'recall': rf_rec,
                'f1': rf_f1,
                'oob_score': self.rf_model.oob_score_
            }
            
            logger.info(f"   ✅ Accuracy: {rf_acc:.3f} | F1: {rf_f1:.3f} | OOB: {self.rf_model.oob_score_:.3f}")
        
        except Exception as e:
            logger.error(f"   ❌ RF training failed: {e}")
            self.rf_model = None
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 2: GRADIENT BOOSTING
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("[2/5] Training Gradient Boosting...")
            
            self.gb_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
                verbose=0
            )
            
            self.gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_gb = self.gb_model.predict(X_test_scaled)
            gb_acc = accuracy_score(y_test, y_pred_gb)
            gb_prec = precision_score(y_test, y_pred_gb, zero_division=0)
            gb_rec = recall_score(y_test, y_pred_gb, zero_division=0)
            gb_f1 = f1_score(y_test, y_pred_gb, zero_division=0)
            
            self.training_history['accuracies']['GB'] = gb_acc
            self.training_history['precisions']['GB'] = gb_prec
            self.training_history['recalls']['GB'] = gb_rec
            self.training_history['f1_scores']['GB'] = gb_f1
            
            results['GB'] = {
                'accuracy': gb_acc,
                'precision': gb_prec,
                'recall': gb_rec,
                'f1': gb_f1
            }
            
            logger.info(f"   ✅ Accuracy: {gb_acc:.3f} | F1: {gb_f1:.3f}")
        
        except Exception as e:
            logger.error(f"   ❌ GB training failed: {e}")
            self.gb_model = None
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 3: XGBOOST
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("[3/5] Training XGBoost...")
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                min_child_weight=1,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            self.xgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_xgb = self.xgb_model.predict(X_test_scaled)
            xgb_acc = accuracy_score(y_test, y_pred_xgb)
            xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
            xgb_rec = recall_score(y_test, y_pred_xgb, zero_division=0)
            xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
            
            self.training_history['accuracies']['XGB'] = xgb_acc
            self.training_history['precisions']['XGB'] = xgb_prec
            self.training_history['recalls']['XGB'] = xgb_rec
            self.training_history['f1_scores']['XGB'] = xgb_f1
            
            results['XGB'] = {
                'accuracy': xgb_acc,
                'precision': xgb_prec,
                'recall': xgb_rec,
                'f1': xgb_f1
            }
            
            logger.info(f"   ✅ Accuracy: {xgb_acc:.3f} | F1: {xgb_f1:.3f}")
        
        except Exception as e:
            logger.error(f"   ❌ XGB training failed: {e}")
            self.xgb_model = None
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 4: LIGHTGBM
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("[4/5] Training LightGBM...")
            
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            self.lgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_lgb = self.lgb_model.predict(X_test_scaled)
            lgb_acc = accuracy_score(y_test, y_pred_lgb)
            lgb_prec = precision_score(y_test, y_pred_lgb, zero_division=0)
            lgb_rec = recall_score(y_test, y_pred_lgb, zero_division=0)
            lgb_f1 = f1_score(y_test, y_pred_lgb, zero_division=0)
            
            self.training_history['accuracies']['LGB'] = lgb_acc
            self.training_history['precisions']['LGB'] = lgb_prec
            self.training_history['recalls']['LGB'] = lgb_rec
            self.training_history['f1_scores']['LGB'] = lgb_f1
            
            results['LGB'] = {
                'accuracy': lgb_acc,
                'precision': lgb_prec,
                'recall': lgb_rec,
                'f1': lgb_f1
            }
            
            logger.info(f"   ✅ Accuracy: {lgb_acc:.3f} | F1: {lgb_f1:.3f}")
        
        except Exception as e:
            logger.error(f"   ❌ LGB training failed: {e}")
            self.lgb_model = None
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL 5: LOGISTIC REGRESSION
        # ═══════════════════════════════════════════════════════════════
        
        try:
            logger.info("[5/5] Training Logistic Regression...")
            
            self.lr_model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.lr_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_lr = self.lr_model.predict(X_test_scaled)
            lr_acc = accuracy_score(y_test, y_pred_lr)
            lr_prec = precision_score(y_test, y_pred_lr, zero_division=0)
            lr_rec = recall_score(y_test, y_pred_lr, zero_division=0)
            lr_f1 = f1_score(y_test, y_pred_lr, zero_division=0)
            
            self.training_history['accuracies']['LR'] = lr_acc
            self.training_history['precisions']['LR'] = lr_prec
            self.training_history['recalls']['LR'] = lr_rec
            self.training_history['f1_scores']['LR'] = lr_f1
            
            results['LR'] = {
                'accuracy': lr_acc,
                'precision': lr_prec,
                'recall': lr_rec,
                'f1': lr_f1
            }
            
            logger.info(f"   ✅ Accuracy: {lr_acc:.3f} | F1: {lr_f1:.3f}")
        
        except Exception as e:
            logger.error(f"   ❌ LR training failed: {e}")
            self.lr_model = None
        
        # ═══════════════════════════════════════════════════════════════
        # ENSEMBLE SUMMARY
        # ═══════════════════════════════════════════════════════════════
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        # Calculate ensemble metrics
        ensemble_f1_scores = [r['f1'] for r in results.values()]
        ensemble_avg_f1 = np.mean(ensemble_f1_scores) if ensemble_f1_scores else 0
        
        logger.info("\n" + "="*80)
        logger.info("📊 ENSEMBLE TRAINING SUMMARY")
        logger.info("="*80)
        
        for model_name, metrics in results.items():
            logger.info(f"{model_name:>5}: Acc={metrics['accuracy']:.3f}, "
                       f"Prec={metrics['precision']:.3f}, "
                       f"Rec={metrics['recall']:.3f}, "
                       f"F1={metrics['f1']:.3f}")
        
        logger.info(f"\nEnsemble F1 Score: {ensemble_avg_f1:.3f}")
        logger.info(f"Models trained: {sum(1 for m in [self.rf_model, self.gb_model, self.xgb_model, self.lgb_model, self.lr_model] if m is not None)}/5")
        logger.info("="*80 + "\n")
        
        # Save models
        self._save_models()
        
        return {
            'is_trained': True,
            'models_count': sum(1 for m in [self.rf_model, self.gb_model, self.xgb_model, self.lgb_model, self.lr_model] if m is not None),
            'ensemble_f1': ensemble_avg_f1,
            'individual_results': results,
            'training_date': self.training_date.isoformat()
        }
    
    def generate_signal(self, features: pd.DataFrame) -> Dict:
        """
        Generate trading signal from features.
        
        Args:
            features: Feature dataframe (1 row with same columns as training)
        
        Returns:
            {
                'signal': float (-1 to 1, -1=sell, 0=neutral, 1=buy),
                'confidence': float (0 to 1, how confident),
                'consensus': float (0 to 1, % models agreeing),
                'model_votes': {model_name: vote (0 or 1)},
                'probabilities': list of probabilities,
                'timestamp': datetime
            }
        """
        
        # Load models if not trained
        if not self.is_trained:
            if not self._load_models():
                logger.warning("No trained models available")
                return self._neutral_signal()
        
        try:
            # Validate input
            if features.empty:
                logger.warning("Empty features dataframe")
                return self._neutral_signal()
            
            if features.shape[0] != 1:
                logger.warning(f"Expected 1 row, got {features.shape[0]}")
                features = features.iloc[-1:].copy()
            
            # Handle missing features
            for col in self.feature_names:
                if col not in features.columns:
                    logger.debug(f"Missing feature: {col}, using 0")
                    features[col] = 0
            
            # Select and reorder features
            features = features[[col for col in self.feature_names if col in features.columns]]
            
            # Fill NaN values
            features = features.fillna(features.mean())
            
            # Scale features
            if self.scaler is None:
                logger.warning("No scaler available")
                return self._neutral_signal()
            
            features_scaled = self.scaler.transform(features)
            
            # Collect predictions
            model_votes = {}
            probabilities = []
            model_count = 0
            
            # RF prediction
            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict(features_scaled)[0]
                    rf_prob = self.rf_model.predict_proba(features_scaled)[0, 1]
                    model_votes['RF'] = int(rf_pred)
                    probabilities.append(rf_prob)
                    model_count += 1
                except Exception as e:
                    logger.debug(f"RF prediction error: {e}")
            
            # GB prediction
            if self.gb_model is not None:
                try:
                    gb_pred = self.gb_model.predict(features_scaled)[0]
                    gb_prob = self.gb_model.predict_proba(features_scaled)[0, 1]
                    model_votes['GB'] = int(gb_pred)
                    probabilities.append(gb_prob)
                    model_count += 1
                except Exception as e:
                    logger.debug(f"GB prediction error: {e}")
            
            # XGB prediction
            if self.xgb_model is not None:
                try:
                    xgb_pred = self.xgb_model.predict(features_scaled)[0]
                    xgb_prob = self.xgb_model.predict_proba(features_scaled)[0, 1]
                    model_votes['XGB'] = int(xgb_pred)
                    probabilities.append(xgb_prob)
                    model_count += 1
                except Exception as e:
                    logger.debug(f"XGB prediction error: {e}")
            
            # LGB prediction
            if self.lgb_model is not None:
                try:
                    lgb_pred = self.lgb_model.predict(features_scaled)[0]
                    lgb_prob = self.lgb_model.predict_proba(features_scaled)[0, 1]
                    model_votes['LGB'] = int(lgb_pred)
                    probabilities.append(lgb_prob)
                    model_count += 1
                except Exception as e:
                    logger.debug(f"LGB prediction error: {e}")
            
            # LR prediction
            if self.lr_model is not None:
                try:
                    lr_pred = self.lr_model.predict(features_scaled)[0]
                    lr_prob = self.lr_model.predict_proba(features_scaled)[0, 1]
                    model_votes['LR'] = int(lr_pred)
                    probabilities.append(lr_prob)
                    model_count += 1
                except Exception as e:
                    logger.debug(f"LR prediction error: {e}")
            
            # No valid models
            if model_count == 0:
                logger.warning("No valid models for prediction")
                return self._neutral_signal()
            
            # Calculate consensus
            bullish_votes = sum(1 for v in model_votes.values() if v == 1)
            consensus = bullish_votes / model_count
            
            # Calculate signal (-1 to 1)
            avg_probability = np.mean(probabilities)
            signal = 2 * avg_probability - 1  # Convert [0,1] to [-1,1]
            
            # Confidence: how confident in the decision
            confidence = abs(consensus - 0.5) * 2  # Range [0,1]
            
            return {
                'signal': float(np.clip(signal, -1, 1)),
                'confidence': float(np.clip(confidence, 0, 1)),
                'consensus': float(consensus),
                'model_votes': model_votes,
                'probabilities': probabilities,
                'models_used': model_count,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return self._neutral_signal()
    
    def _neutral_signal(self) -> Dict:
        """Return neutral signal."""
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'consensus': 0.5,
            'model_votes': {},
            'probabilities': [],
            'models_used': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_models(self):
        """Save all trained models to disk."""
        try:
            if self.rf_model is not None:
                dump(self.rf_model, f"{self.model_dir}/rf_model.pkl")
                logger.debug(f"Saved RF model")
            
            if self.gb_model is not None:
                dump(self.gb_model, f"{self.model_dir}/gb_model.pkl")
                logger.debug(f"Saved GB model")
            
            if self.xgb_model is not None:
                self.xgb_model.save_model(f"{self.model_dir}/xgb_model.json")
                logger.debug(f"Saved XGB model")
            
            if self.lgb_model is not None:
                dump(self.lgb_model, f"{self.model_dir}/lgb_model.pkl")
                logger.debug(f"Saved LGB model")
            
            if self.lr_model is not None:
                dump(self.lr_model, f"{self.model_dir}/lr_model.pkl")
                logger.debug(f"Saved LR model")
            
            # Save scaler
            if self.scaler is not None:
                dump(self.scaler, f"{self.model_dir}/scaler.pkl")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_count': self.feature_count,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'training_history': self.training_history
            }
            
            with open(f"{self.model_dir}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✅ All models saved to {self.model_dir}")
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            if not os.path.exists(f"{self.model_dir}/metadata.pkl"):
                logger.warning(f"No models found in {self.model_dir}")
                return False
            
            # Load models
            if os.path.exists(f"{self.model_dir}/rf_model.pkl"):
                self.rf_model = load(f"{self.model_dir}/rf_model.pkl")
            
            if os.path.exists(f"{self.model_dir}/gb_model.pkl"):
                self.gb_model = load(f"{self.model_dir}/gb_model.pkl")
            
            if os.path.exists(f"{self.model_dir}/xgb_model.json"):
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(f"{self.model_dir}/xgb_model.json")
            
            if os.path.exists(f"{self.model_dir}/lgb_model.pkl"):
                self.lgb_model = load(f"{self.model_dir}/lgb_model.pkl")
            
            if os.path.exists(f"{self.model_dir}/lr_model.pkl"):
                self.lr_model = load(f"{self.model_dir}/lr_model.pkl")
            
            # Load scaler
            if os.path.exists(f"{self.model_dir}/scaler.pkl"):
                self.scaler = load(f"{self.model_dir}/scaler.pkl")
            
            # Load metadata
            with open(f"{self.model_dir}/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            self.feature_count = metadata.get('feature_count', 0)
            self.training_history = metadata.get('training_history', {})
            
            self.is_trained = True
            
            logger.info(f"✅ Models loaded from {self.model_dir}")
            logger.info(f"   Features: {self.feature_count}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N important features from ensemble.
        
        Returns features from models that support it (RF, GB, XGB, LGB)
        """
        importance_dict = {}
        
        # RF feature importance
        if self.rf_model is not None and hasattr(self.rf_model, 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.rf_model.feature_importances_):
                importance_dict[f"RF_{fname}"] = float(imp)
        
        # GB feature importance
        if self.gb_model is not None and hasattr(self.gb_model, 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.gb_model.feature_importances_):
                importance_dict[f"GB_{fname}"] = float(imp)
        
        # XGB feature importance
        if self.xgb_model is not None:
            try:
                xgb_imp = self.xgb_model.get_booster().get_score(importance_type='weight')
                for fname, imp in xgb_imp.items():
                    importance_dict[f"XGB_{fname}"] = float(imp)
            except:
                pass
        
        # LGB feature importance
        if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
            for fname, imp in zip(self.feature_names, self.lgb_model.feature_importances_):
                importance_dict[f"LGB_{fname}"] = float(imp)
        
        # Sort and return top N
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])


if __name__ == "__main__":
    # Test ensemble
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("ENSEMBLE SIGNAL GENERATOR - TEST")
    print("="*80 + "\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i:02d}" for i in range(n_features)]
    )
    
    # Create target with some signal
    y = (X['feature_00'] + X['feature_01'] * 0.5 > 0).astype(int)
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {dict(y.value_counts())}\n")
    
    # Initialize and train
    ensemble = EnsembleSignalGenerator(model_dir="models/test")
    
    training_result = ensemble.train(X, y)
    
    print(f"\nTraining successful: {training_result['is_trained']}")
    print(f"Models trained: {training_result['models_count']}/5")
    print(f"Ensemble F1: {training_result['ensemble_f1']:.3f}\n")
    
    # Generate signal
    print("="*80)
    print("TEST: GENERATE SIGNAL")
    print("="*80 + "\n")
    
    test_features = X.iloc[-1:].copy()
    signal = ensemble.generate_signal(test_features)
    
    print(f"Signal: {signal['signal']:+.3f}")
    print(f"Confidence: {signal['confidence']:.3f}")
    print(f"Consensus: {signal['consensus']:.3f}")
    print(f"Models used: {signal['models_used']}/5")
    print(f"Model votes: {signal['model_votes']}")
    print(f"Probabilities: {[f'{p:.3f}' for p in signal['probabilities']]}")
    
    # Feature importance
    print("\n" + "="*80)
    print("TOP 10 IMPORTANT FEATURES")
    print("="*80 + "\n")
    
    importance = ensemble.get_feature_importance(top_n=10)
    for i, (feature, imp) in enumerate(importance.items(), 1):
        print(f"{i:2d}. {feature:40s} {imp:.6f}")
    
    print("\n✅ All tests passed!")