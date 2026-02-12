"""LSTM regressor with scikit-learn compatible interface.

Requires PyTorch. If torch is not installed, importing this module will
raise ImportError and the __init__.py will set LSTM_AVAILABLE = False.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNet(nn.Module):
    """2-layer LSTM with linear output head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last timestep
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible LSTM regressor.

    Internally reshapes flat feature vectors into sequences by treating
    each sample as a length-1 sequence. For richer temporal modelling,
    pass pre-windowed data (n_samples, seq_len, n_features) to fit().
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 50, batch_size: int = 256,
                 patience: int = 8, random_state: int = 42):
        self.hidden_dim = hidden_dim
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

        # Train/val split (last 20% as validation)
        split = int(len(X_t) * 0.8)
        X_train, X_val = X_t[:split], X_t[split:]
        y_train, y_val = y_t[:split], y_t[split:]

        self._net = _LSTMNet(
            input_dim=X_t.shape[-1],
            hidden_dim=self.hidden_dim,
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

            # Validation
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

        # Uniform feature importances (LSTM doesn't provide them)
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
        """Convert (n, features) → (n, 1, features) tensor."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]  # (n, 1, features)
        return torch.tensor(arr, dtype=torch.float32)
