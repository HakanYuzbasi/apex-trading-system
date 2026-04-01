"""
quant_system/models/lstm_model.py
================================================================================
LSTM Temporal Meta-Learner Layer
================================================================================
Outputs mean probability, epistemic variance, and scaled confidence bounds.
"""

import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
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
        
        self.fc_shared = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        
        self.head_prob = nn.Linear(32, 1)
        self.head_uncert = nn.Linear(32, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus() # Variance strictly > 0

    def forward(self, x):
        """Returns: (probability, variance, confidence)"""
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        norm_out = self.batch_norm(last_step)
        
        shared = self.relu(self.fc_shared(norm_out))
        
        prob = self.sigmoid(self.head_prob(shared))
        variance = self.softplus(self.head_uncert(shared))
        
        # UPGRADE 2: UNCERTAINTY -> CONFIDENCE PIPELINE
        confidence = 1.0 / (1.0 + variance)
        
        return prob, variance, confidence
