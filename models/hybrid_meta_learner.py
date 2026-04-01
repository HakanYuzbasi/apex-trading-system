"""
models/hybrid_meta_learner.py
================================================================================
PHASE 2: HYBRID META-LEARNING ARCHITECTURE
================================================================================

This module implements the Hybrid LSTM + Tree-Ensemble architecture defined
in the 90-Day Advanced Quantitative Roadmap.

Key Capabilities:
1. Spatial Processing: LightGBM / XGBoost process static bar features.
2. Temporal Processing: LSTM processes sequential lookback windows (e.g. 60 bars).
3. Meta-Labeling: Both pipelines output probabilistic confidence which is
   blended as a meta-label probability over the Base Signal.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Tree Ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Deep Learning
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# 1. DEEP TEMPORAL COMPONENT (LSTM)
# ------------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class TemporalLSTM(nn.Module):
        """Processes sequential market microstructure (L2 Imbalance, Volumes)."""
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x shape: (batch_size, sequence_length, features)
            lstm_out, _ = self.lstm(x)
            # Take the output of the final timestep
            last_timestep = lstm_out[:, -1, :]
            norm_out = self.batch_norm(last_timestep)
            hidden = self.relu(self.fc1(norm_out))
            prob = self.sigmoid(self.fc2(hidden))
            return prob


# ------------------------------------------------------------------------------
# 2. HYBRID META-LEARNER PIPELINE
# ------------------------------------------------------------------------------
class HybridMetaLearner:
    def __init__(self, seq_length: int = 30):
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        
        # Base Tree Estimators
        self.estimators = {}
        if XGBClassifier:
            self.estimators['xgb'] = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, 
                subsample=0.8, colsample_bytree=0.8, random_state=42
            )
        if LGBMClassifier:
            self.estimators['lgb'] = LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, 
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
        
        # Calibrated wrappers mapping output to physical win rate
        self.calibrated_trees = {}
        
        # LSTM Model
        self.lstm_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert flat time-series arrays into rolling 3D tensors for LSTM."""
        X_seq, Y, X_spatial = [], [], []
        for i in range(len(features) - self.seq_length):
            X_seq.append(features[i : i + self.seq_length])
            X_spatial.append(features[i + self.seq_length - 1]) # Most recent step for trees
            Y.append(labels[i + self.seq_length - 1])
        return np.array(X_seq), np.array(Y), np.array(X_spatial)

    def train_pipeline(self, X_raw: np.ndarray, y: np.ndarray):
        """
        Train the parallel hybrid architecture.
        X_raw is expected to contain all Phase 1 features (Cross-Asset, L2 Imbalance).
        """
        logger.info("Initializing Hybrid Training Pipeline...")
        
        # Normalize globally
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Create sequences for temporal models
        X_seq, Y, X_spatial = self._create_sequences(X_scaled, y)
        
        # 1. Train SPATIAL Ensembles (Trees)
        logger.info("Training Spatial Tree Ensembles with Isotonic Calibration...")
        cv = TimeSeriesSplit(n_splits=5)
        for name, clf in self.estimators.items():
            calibrated = CalibratedClassifierCV(clf, cv=cv, method='isotonic')
            calibrated.fit(X_spatial, Y)
            self.calibrated_trees[name] = calibrated
            logger.info(f"  -> {name.upper()} calibrated and fitted.")

        # 2. Train TEMPORAL Model (LSTM)
        if TORCH_AVAILABLE:
            logger.info(f"Training Temporal LSTM on {self.device}...")
            self.lstm_model = TemporalLSTM(input_dim=X_raw.shape[1]).to(self.device)
            self.lstm_model.train()
            
            # Prepare PyTorch Tensors
            tensor_x = torch.Tensor(X_seq).to(self.device)
            tensor_y = torch.Tensor(Y).view(-1, 1).to(self.device)
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=64, shuffle=False) # MUST be False for TS
            
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            for epoch in range(10): # Example fast loop
                epoch_loss = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    output = self.lstm_model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                logger.info(f"  -> LSTM Epoch {epoch+1}/10 Loss: {epoch_loss/len(loader):.4f}")

    def predict_meta_label(self, X_recent: np.ndarray) -> float:
        """
        Inference: Generates a blended Kelly probability.
        Accepts the recent window (seq_length, features) of data.
        """
        if len(X_recent) < self.seq_length:
            return 0.0
            
        # Ensure window is exact length
        window = X_recent[-self.seq_length:]
        window_sc = self.scaler.transform(window)
        
        # 1. Spatial Probability (Trees)
        latest_spatial = window_sc[-1].reshape(1, -1)
        tree_probs = []
        for name, clf in self.calibrated_trees.items():
            prob = clf.predict_proba(latest_spatial)[1]
            tree_probs.append(prob)
        
        avg_tree_prob = np.mean(tree_probs) if tree_probs else 0.5
        
        # 2. Temporal Probability (LSTM)
        lstm_prob = 0.5
        if self.lstm_model is not None and TORCH_AVAILABLE:
            self.lstm_model.eval()
            with torch.no_grad():
                tensor_seq = torch.Tensor(window_sc).unsqueeze(0).to(self.device) # Add batch dim
                lstm_prob = self.lstm_model(tensor_seq).item()
                
        # 3. Hybrid Blending (Meta-Ansatz)
        # Weighting can be optimized based on out-of-sample Brier score tracking
        hybrid_confidence = (0.7 * avg_tree_prob) + (0.3 * lstm_prob)
        
        return float(hybrid_confidence)

# ------------------------------------------------------------------------------
# 3. BACKTEST HOOKS & ICIR TRACKING
# ------------------------------------------------------------------------------
def evaluate_hybrid_metrics(hybrid_probs: np.ndarray, y_true: np.ndarray) -> Dict:
    """Calculate Information Coefficient and Brier Score for the Hybrid Output."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from scipy.stats import spearmanr
    
    ic, p_val = spearmanr(hybrid_probs, y_true)
    brier = brier_score_loss(y_true, hybrid_probs)
    auc = roc_auc_score(y_true, hybrid_probs)
    
    return {
        "Information Coefficient (IC)": round(ic, 4),
        "P-Value": round(p_val, 4),
        "Brier Score": round(brier, 4),
        "ROC-AUC": round(auc, 4),
        "Kelly Edge Quality": "STRONG" if ic > 0.15 else "WEAK"
    }

if __name__ == "__main__":
    logger.info("Hybrid Meta-Learner Module Ready for Phase 2 Integration.")
