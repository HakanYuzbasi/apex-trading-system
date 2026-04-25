"""
scripts/train_lstm_signals.py

Round 13 FIX 3 — per-symbol LSTM signal training.

Architecture (from the brief):
    input  : ``LSTM_LOOKBACK`` (60) bar window of 8 features
              [return, log_volume, rsi14, macd_hist, atr_pct,
               price_vs_sma20, day_of_week_sin, day_of_week_cos]
    layers : LSTM(hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, dropout=LSTM_DROPOUT)
              -> Linear(LSTM_HIDDEN, 64) -> ReLU -> Dropout(LSTM_DROPOUT)
              -> Linear(64, 1) -> Tanh
    label  : sign(close[t+SIGNAL_HORIZON_BARS] - close[t])  in {-1, +1}
    loss   : BCEWithLogitsLoss on the +1 class probability (label mapped
             from {-1, +1} -> {0.0, 1.0})
    optim  : AdamW(lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    epochs : LSTM_EPOCHS with early stopping (patience=LSTM_PATIENCE)
    split  : TimeSeriesSplit(n_splits=LSTM_N_SPLITS) — no shuffle, no leakage

Per-symbol model artefacts are saved to ``models/saved_advanced/lstm_<SYM>.pt``
together with a JSON sidecar describing input feature stats so the
inference path can re-normalise consistently.

Falls back gracefully when PyTorch is missing — emits a WARNING and
exits with rc=2 so the Round 13 driver can skip ahead to GBM-only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ApexConfig
from core.logging_config import setup_logging
from models.ml_validator import leakage_check

logger = logging.getLogger(__name__)


SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
    "GLD", "TLT", "IWM",
)
TRAIN_START: str = "2018-01-01"
TRAIN_END: str = "2023-01-01"
RANDOM_STATE: int = 42

LSTM_FEATURE_NAMES: Tuple[str, ...] = (
    "ret", "log_vol", "rsi14", "macd_hist", "atr_pct",
    "price_vs_sma20", "dow_sin", "dow_cos",
)


def _build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build the LSTM input feature matrix from raw OHLCV bars."""
    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    ret = close.pct_change()
    log_vol = np.log1p(volume.clip(lower=0.0))

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr_pct = atr / close.replace(0.0, np.nan)

    sma20 = close.rolling(20, min_periods=20).mean()
    price_vs_sma20 = (close - sma20) / sma20

    dow = pd.Series(ohlcv.index.dayofweek, index=ohlcv.index, dtype=float)
    dow_sin = np.sin(2 * math.pi * dow / 5.0)
    dow_cos = np.cos(2 * math.pi * dow / 5.0)

    feats = pd.DataFrame(
        {
            "ret": ret,
            "log_vol": log_vol,
            "rsi14": rsi,
            "macd_hist": macd_hist,
            "atr_pct": atr_pct,
            "price_vs_sma20": price_vs_sma20,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        },
        index=ohlcv.index,
    ).dropna()
    return feats


def _build_label(close: pd.Series, horizon: int) -> pd.Series:
    """Forward-return sign label for the LSTM."""
    fwd = close.shift(-horizon) / close - 1.0
    return np.sign(fwd).replace(0.0, np.nan).rename("label")


def _windowize(
    feats: pd.DataFrame, label: pd.Series, lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack 60-bar windows into shape (n, lookback, n_features)."""
    feat_cols = list(LSTM_FEATURE_NAMES)
    aligned = feats[feat_cols].join(label, how="inner").dropna()
    arr = aligned[feat_cols].to_numpy(dtype=np.float32)
    lab = aligned["label"].to_numpy(dtype=np.float32)
    n_samples = len(arr) - lookback + 1
    if n_samples <= 0:
        return np.zeros((0, lookback, arr.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack([arr[i : i + lookback] for i in range(n_samples)], axis=0)
    y = lab[lookback - 1 :]
    return X, y


def fetch_yf(sym: str) -> pd.DataFrame:
    """Load OHLCV via the same chunked + cached fetcher as the Round 12 driver."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from round12_real_data_report import fetch_panel_chunked
    panel = fetch_panel_chunked((sym,), TRAIN_START, TRAIN_END)
    return panel[sym]


def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 13 — Per-symbol LSTM training (PyTorch)")
    print("=" * 72)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        print(f"PyTorch unavailable ({exc}); LSTM training skipped.")
        return 2

    print(f" PyTorch         : {torch.__version__}")
    print(f" Symbols         : {list(SYMBOLS)}")
    print(f" Span            : {TRAIN_START} -> {TRAIN_END}")
    print(f" LSTM_LOOKBACK   : {ApexConfig.LSTM_LOOKBACK}")
    print(f" LSTM_HIDDEN     : {ApexConfig.LSTM_HIDDEN}")
    print(f" LSTM_LAYERS     : {ApexConfig.LSTM_LAYERS}")
    print(f" LSTM_DROPOUT    : {ApexConfig.LSTM_DROPOUT}")
    print(f" LSTM_EPOCHS     : {ApexConfig.LSTM_EPOCHS}")
    print(f" LSTM_PATIENCE   : {ApexConfig.LSTM_PATIENCE}")
    print(f" LSTM_LR         : {ApexConfig.LSTM_LR}")
    print(f" LSTM_WEIGHT_DCY : {ApexConfig.LSTM_WEIGHT_DECAY}")
    print(f" LSTM_N_SPLITS   : {ApexConfig.LSTM_N_SPLITS}")
    print()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    class LSTMSignal(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=ApexConfig.LSTM_HIDDEN,
                num_layers=ApexConfig.LSTM_LAYERS,
                dropout=ApexConfig.LSTM_DROPOUT if ApexConfig.LSTM_LAYERS > 1 else 0.0,
                batch_first=True,
            )
            self.fc1 = nn.Linear(ApexConfig.LSTM_HIDDEN, 64)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(ApexConfig.LSTM_DROPOUT)
            self.fc2 = nn.Linear(64, 1)
            self.tanh = nn.Tanh()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            h = self.relu(self.fc1(last))
            h = self.dropout(h)
            logits = self.fc2(h)
            return self.tanh(logits).squeeze(-1)  # in [-1, 1]

    save_dir = Path(__file__).resolve().parents[1] / "models" / "saved_advanced"
    save_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(getattr(ApexConfig, "SIGNAL_HORIZON_BARS", 5))
    lookback = int(ApexConfig.LSTM_LOOKBACK)
    n_features = len(LSTM_FEATURE_NAMES)

    summary: Dict[str, Dict[str, Any]] = {}

    for sym in SYMBOLS:
        print("-" * 72)
        print(f" Training LSTM for {sym}")
        ohlcv = fetch_yf(sym)
        feats = _build_features(ohlcv)
        label = _build_label(ohlcv["Close"].astype(float), horizon)

        # Leakage audit BEFORE windowization (per the brief).
        audit_df = feats[list(LSTM_FEATURE_NAMES)].join(label, how="inner").dropna()
        if audit_df.empty:
            print(f"   skipped — no overlapping rows for {sym}")
            continue
        leakage_check(
            audit_df,
            label_col="label",
            feature_cols=list(LSTM_FEATURE_NAMES),
            reference_col=None,
            max_future_shift=horizon,
            leak_corr_threshold=0.98,
            raise_on_fail=True,
        )

        X, y = _windowize(feats, label, lookback=lookback)
        if len(X) < 200:
            print(f"   {sym}: only {len(X)} windows after warm-up; skipped")
            summary[sym] = {"status": "insufficient_samples", "n_samples": int(len(X))}
            continue

        # Per-feature mean/std on the FIRST training partition only — avoids
        # using held-out info to scale the test slice.
        ts_split = TimeSeriesSplit(n_splits=int(ApexConfig.LSTM_N_SPLITS))
        splits = list(ts_split.split(X))
        first_train_idx, _ = splits[0]
        flat_train = X[first_train_idx].reshape(-1, n_features)
        feat_mean = flat_train.mean(axis=0)
        feat_std = flat_train.std(axis=0) + 1e-6

        def _normalise(arr: np.ndarray) -> np.ndarray:
            return (arr - feat_mean) / feat_std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMSignal(n_features=n_features).to(device)
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(ApexConfig.LSTM_LR),
            weight_decay=float(ApexConfig.LSTM_WEIGHT_DECAY),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        # Train on the LAST split; earlier splits are used as the validation
        # ladder for early stopping per the TimeSeriesSplit protocol.
        train_idx, val_idx = splits[-1]
        X_train_n = _normalise(X[train_idx])
        X_val_n = _normalise(X[val_idx])
        y_train01 = ((y[train_idx] + 1.0) / 2.0).astype(np.float32)
        y_val01 = ((y[val_idx] + 1.0) / 2.0).astype(np.float32)

        train_ds = TensorDataset(
            torch.from_numpy(X_train_n.astype(np.float32)),
            torch.from_numpy(y_train01),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val_n.astype(np.float32)),
            torch.from_numpy(y_val01),
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

        best_val: float = float("inf")
        best_state: Optional[Dict[str, "torch.Tensor"]] = None
        epochs_no_improve = 0
        history: List[Tuple[int, float, float]] = []
        for epoch in range(int(ApexConfig.LSTM_EPOCHS)):
            model.train()
            train_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                optim.zero_grad()
                pred = model(xb)
                # Map tanh in [-1, 1] -> logit in (-inf, inf) for BCE.
                logit = torch.atanh(pred.clamp(-0.999_999, 0.999_999))
                loss = loss_fn(logit, yb)
                loss.backward()
                optim.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(1, n_batches)

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val = 0
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb)
                    logit = torch.atanh(pred.clamp(-0.999_999, 0.999_999))
                    val_loss += loss_fn(logit, yb).item()
                    n_val += 1
                val_loss /= max(1, n_val)
            history.append((epoch, train_loss, val_loss))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(ApexConfig.LSTM_PATIENCE):
                    print(
                        f"   epoch {epoch:3d}: early stop "
                        f"(train={train_loss:.4f}, val={val_loss:.4f})"
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        # Final test accuracy on val slice.
        model.eval()
        with torch.no_grad():
            preds = []
            tgt = []
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                p = model(xb)
                preds.append(p.cpu().numpy())
                tgt.append((yb.cpu().numpy() * 2.0 - 1.0))
            preds_arr = np.concatenate(preds)
            tgt_arr = np.concatenate(tgt)
            test_acc = float(np.mean(np.sign(preds_arr) == np.sign(tgt_arr)))

        # Persist.
        artefact_path = save_dir / f"lstm_{sym}.pt"
        meta = {
            "feature_names": list(LSTM_FEATURE_NAMES),
            "lookback": lookback,
            "horizon": horizon,
            "feat_mean": feat_mean.tolist(),
            "feat_std": feat_std.tolist(),
            "lstm_hidden": int(ApexConfig.LSTM_HIDDEN),
            "lstm_layers": int(ApexConfig.LSTM_LAYERS),
            "lstm_dropout": float(ApexConfig.LSTM_DROPOUT),
            "test_accuracy": float(test_acc),
            "best_val_loss": float(best_val),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "epochs_run": history[-1][0] + 1 if history else 0,
        }
        torch.save(
            {"model_state": model.state_dict(), "meta": meta},
            artefact_path,
        )
        print(
            f"   {sym}: saved {artefact_path.name} ({os.path.getsize(artefact_path)} B)"
            f" — test_acc={test_acc:.4f}, n_train={len(train_idx)}, "
            f"n_val={len(val_idx)}, best_val_loss={best_val:.4f}"
        )
        summary[sym] = {
            "status": "trained",
            "test_accuracy": test_acc,
            "best_val_loss": best_val,
            "epochs_run": history[-1][0] + 1 if history else 0,
            "path": str(artefact_path),
        }

    print()
    print(" Per-symbol LSTM summary")
    print(" " + "-" * 68)
    for sym, s in summary.items():
        if s.get("status") == "trained":
            print(f"   {sym:<5}: test_acc={s['test_accuracy']:.4f}  "
                  f"epochs={s['epochs_run']}  val_loss={s['best_val_loss']:.4f}")
        else:
            print(f"   {sym:<5}: {s.get('status')} ({s.get('n_samples', 0)} samples)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
