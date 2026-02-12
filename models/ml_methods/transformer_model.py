"""Lightweight Transformer regressor with scikit-learn compatible interface.

Requires PyTorch. If torch is not installed, importing this module will
raise ImportError and the __init__.py will set TRANSFORMER_AVAILABLE = False.
"""

import math
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class _TransformerNet(nn.Module):
    """Small transformer encoder → linear regression head."""

    def __init__(self, input_dim: int, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


class TransformerRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Transformer regressor.

    Lightweight architecture: 2 heads, 2 layers, dim=64.
    Treats each flat feature vector as a length-1 sequence by default.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 50,
                 batch_size: int = 256, patience: int = 8,
                 random_state: int = 42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self._net = None
        self._device = "cpu"
        self.feature_importances_ = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"

        X_t = self._to_sequences(X)
        y_t = torch.tensor(y, dtype=torch.float32)

        split = int(len(X_t) * 0.8)
        X_train, X_val = X_t[:split], X_t[split:]
        y_train, y_val = y_t[:split], y_t[split:]

        self._net = _TransformerNet(
            input_dim=X_t.shape[-1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_pred = self._net(X_val.to(self._device))
                val_loss = criterion(val_pred, y_val.to(self._device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
            self._net.to(self._device)

        n_features = X.shape[-1] if X.ndim > 1 else X.shape[1]
        self.feature_importances_ = np.ones(n_features) / n_features
        return self

    def predict(self, X):
        self._net.eval()
        X_t = self._to_sequences(X).to(self._device)
        with torch.no_grad():
            preds = self._net(X_t).cpu().numpy()
        return preds

    def _to_sequences(self, X) -> torch.Tensor:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        return torch.tensor(arr, dtype=torch.float32)
