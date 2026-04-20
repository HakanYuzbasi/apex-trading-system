"""
scripts/train_baseline_models.py

Train a single baseline gradient-boosting classifier on real OHLCV data
from yfinance and save it under the three per-regime paths the backtester
and live trader route through
(:attr:`config.ApexConfig.ML_MODEL_PATH_TRENDING` / ``MEAN_REV`` /
``VOLATILE``). Using the same model across all three regime slots is a
deliberate starting point; per-regime tuning is a later optimisation and
out of scope for the Round 9 gap fix.

Pipeline:

1. Download daily OHLCV for ``AAPL`` + ``MSFT`` + ``GOOGL`` spanning
   ``2018-01-01 → 2022-12-31`` through ``yfinance``.
2. Build the 8-column feature matrix defined by
   :func:`backtesting.real_signal_adapter.compute_ml_features` — RSI(14),
   MACD(12/26/9), ATR(14), volume_ratio, price_vs_sma20 and
   price_vs_sma50.
3. Label each row ``+1`` when the forward-5-bar log return is positive
   and ``-1`` otherwise. Neutral bars (zero forward return) are dropped.
4. Concatenate across symbols, sort by timestamp, and run
   :func:`models.ml_validator.leakage_check` **before** ``fit()``. If the
   audit fails the script aborts with a non-zero exit code.
5. Train a :class:`sklearn.ensemble.GradientBoostingClassifier` on the
   first 80 % of the time-ordered data, evaluate on the held-out 20 %.
6. Save the fitted model plus training-baseline stats (mean + std of the
   training-set probability output) as a pickled dict to all three
   ``ApexConfig.ML_MODEL_PATH_*`` targets. The backtester's adapter
   (:class:`backtesting.real_signal_adapter.RealSignalAdapter`) expects
   the payload shape ``{"model": estimator, "baseline_stats": {...}}``
   for GAP-8B drift detection.

The script prints a reproducibility banner (random seed, data hash,
feature counts, class balance, metrics, saved paths) so the Round 9
report can inline the output verbatim.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

# Make the project root importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.real_signal_adapter import ML_FEATURE_NAMES, compute_ml_features
from config import ApexConfig
from core.logging_config import setup_logging
from models.ml_validator import leakage_check


SYMBOLS: Tuple[str, ...] = ("AAPL", "MSFT", "GOOGL")
TRAIN_START: str = "2018-01-01"
TRAIN_END: str = "2023-01-01"  # yfinance end is exclusive
FORWARD_HORIZON: int = 5
RANDOM_STATE: int = 42
TRAIN_FRACTION: float = 0.80


def fetch_yf_ohlcv(symbols: Tuple[str, ...], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV bars via yfinance.

    Args:
        symbols: Tickers to download.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).

    Returns:
        ``{symbol: DataFrame}``. Each frame has columns ``Open``, ``High``,
        ``Low``, ``Close``, ``Volume`` indexed by a tz-naive DatetimeIndex.

    Raises:
        RuntimeError: If every symbol returned zero bars.
    """
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False,
        )
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        panel[sym] = df
    if not panel:
        raise RuntimeError(f"yfinance returned no bars for any of {symbols}")
    return panel


def build_training_frame(
    panel: Dict[str, pd.DataFrame],
    *,
    horizon: int,
) -> pd.DataFrame:
    """
    Stitch per-symbol feature + label rows into a single time-ordered frame.

    Args:
        panel: ``{symbol: OHLCV DataFrame}``.
        horizon: Forward return horizon in bars.

    Returns:
        DataFrame indexed by timestamp with columns
        ``[*ML_FEATURE_NAMES, "label", "symbol"]``. Sorted by index so
        :func:`leakage_check` can validate monotonic time ordering.
    """
    frames: List[pd.DataFrame] = []
    for sym, ohlcv in panel.items():
        feats = compute_ml_features(ohlcv)
        close = ohlcv["Close"].astype(float)
        fwd_return = np.log(close.shift(-horizon) / close)
        label = np.sign(fwd_return).replace(0.0, np.nan)
        joined = feats.join(label.rename("label"), how="inner").dropna()
        joined = joined.assign(symbol=sym)
        frames.append(joined)
    combined = pd.concat(frames, axis=0).sort_index()
    combined["label"] = combined["label"].astype(int)
    return combined


def digest_panel(panel: Dict[str, pd.DataFrame]) -> str:
    """Deterministic SHA256 of the concatenated Close series (for audit)."""
    hasher = hashlib.sha256()
    for sym in sorted(panel.keys()):
        arr = panel[sym]["Close"].to_numpy().tobytes()
        hasher.update(sym.encode("utf-8"))
        hasher.update(arr)
    return hasher.hexdigest()[:16]


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
) -> GradientBoostingClassifier:
    """
    Fit a scikit-learn GradientBoostingClassifier.

    Args:
        X_train: Training feature frame (columns == :data:`ML_FEATURE_NAMES`).
        y_train: Training label Series in ``{-1, +1}``.

    Returns:
        Fitted :class:`GradientBoostingClassifier`.
    """
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    return model


def baseline_stats_from(
    model: GradientBoostingClassifier, X_train: pd.DataFrame,
) -> Dict[str, float]:
    """
    Record mean/std of the adapter's signed score on the training set.

    The adapter converts ``predict_proba`` into a ``[-1, +1]`` score via
    ``(p_up - 0.5) * 2``. Tracking its training distribution lets
    :class:`~backtesting.real_signal_adapter.RealSignalAdapter` reject
    production scores that drift too far from calibration (GAP-8B).

    Args:
        model: Fitted classifier.
        X_train: Training feature frame.

    Returns:
        Dict with ``mean`` and ``std`` of the signed score on the training
        set.
    """
    proba = model.predict_proba(X_train.to_numpy())
    classes = list(model.classes_)
    up_idx = classes.index(1) if 1 in classes else len(classes) - 1
    p_up = proba[:, up_idx]
    signed = np.clip((p_up - 0.5) * 2.0, -1.0, 1.0)
    return {
        "mean": float(np.mean(signed)),
        "std": float(np.std(signed, ddof=0)),
        "n_samples": int(len(signed)),
    }


def resolve_save_paths() -> List[str]:
    """
    Resolve the three per-regime save paths, using sensible defaults when
    the corresponding ApexConfig fields are empty.
    """
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "models" / "saved_advanced"
    default_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for attr, default_name in (
        ("ML_MODEL_PATH_TRENDING", "baseline_trending.pkl"),
        ("ML_MODEL_PATH_MEAN_REV", "baseline_mean_rev.pkl"),
        ("ML_MODEL_PATH_VOLATILE", "baseline_volatile.pkl"),
    ):
        configured = str(getattr(ApexConfig, attr, "") or "").strip()
        paths.append(configured if configured else str(default_dir / default_name))
    return paths


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    """Persist the training payload as a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 9 — Baseline ML model training")
    print("=" * 72)
    print(f" Symbols        : {list(SYMBOLS)}")
    print(f" Training span  : {TRAIN_START} → {TRAIN_END}")
    print(f" Horizon        : {FORWARD_HORIZON} bars (sign of fwd log-return)")
    print(f" Random seed    : {RANDOM_STATE}")
    print()

    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_ohlcv(SYMBOLS, TRAIN_START, TRAIN_END)
    for sym, df in panel.items():
        print(f"   {sym}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    print(f" Panel SHA-16  : {digest_panel(panel)}")
    print()

    print(" Building feature matrix ...")
    combined = build_training_frame(panel, horizon=FORWARD_HORIZON)
    feature_cols = list(ML_FEATURE_NAMES)
    print(f"   rows           : {len(combined)}")
    print(f"   features       : {feature_cols}")
    up_count = int((combined["label"] == 1).sum())
    down_count = int((combined["label"] == -1).sum())
    total = up_count + down_count
    up_ratio = up_count / total if total else 0.0
    print(f"   up labels      : {up_count} ({up_ratio:.3%})")
    print(f"   down labels    : {down_count} ({1.0 - up_ratio:.3%})")
    print()

    # ── Leakage audit — MUST run before fit() ────────────────────────────────
    print(" Running leakage_check (raise_on_fail=True) ...")
    audit = leakage_check(
        combined[feature_cols + ["label"]],
        label_col="label",
        feature_cols=feature_cols,
        reference_col=None,
        max_future_shift=FORWARD_HORIZON,
        leak_corr_threshold=0.98,
        raise_on_fail=True,
    )
    print(f"   leakage_check.ok   : {audit['ok']}")
    print(f"   leaky_features     : {audit['leaky_features'] or '{}'}")
    print()

    # ── Time-ordered train/test split ────────────────────────────────────────
    split_idx = int(len(combined) * TRAIN_FRACTION)
    train_df = combined.iloc[:split_idx]
    test_df = combined.iloc[split_idx:]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    print(f" Train rows     : {len(train_df)}  ({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f" Test  rows     : {len(test_df)}   ({test_df.index[0].date()} → {test_df.index[-1].date()})")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    print(" Training GradientBoostingClassifier ...")
    model = train_model(X_train, y_train)
    print(" Done.")
    print()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_train_pred = model.predict(X_train.to_numpy())
    y_test_pred = model.predict(X_test.to_numpy())
    proba_train = model.predict_proba(X_train.to_numpy())
    proba_test = model.predict_proba(X_test.to_numpy())
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_ll = log_loss(y_train, proba_train, labels=list(model.classes_))
    test_ll = log_loss(y_test, proba_test, labels=list(model.classes_))
    print(" Metrics")
    print(" " + "-" * 68)
    print(f"   train accuracy : {train_acc:.4f}   log-loss {train_ll:.4f}")
    print(f"   test  accuracy : {test_acc:.4f}   log-loss {test_ll:.4f}")
    print(f"   classes_       : {list(model.classes_)}")
    print()

    # ── Baseline stats for drift detection (GAP-8B) ──────────────────────────
    baseline = baseline_stats_from(model, X_train)
    print(" Baseline signed-score stats (training set)")
    print(" " + "-" * 68)
    print(f"   mean           : {baseline['mean']:.6f}")
    print(f"   std            : {baseline['std']:.6f}")
    print(f"   n_samples      : {baseline['n_samples']}")
    print()

    # ── Persist ───────────────────────────────────────────────────────────────
    payload: Dict[str, Any] = {
        "model": model,
        "baseline_stats": baseline,
        "feature_names": feature_cols,
        "training_metadata": {
            "symbols": list(SYMBOLS),
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "horizon": FORWARD_HORIZON,
            "random_state": RANDOM_STATE,
            "train_fraction": TRAIN_FRACTION,
            "panel_sha16": digest_panel(panel),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_log_loss": float(train_ll),
            "test_log_loss": float(test_ll),
        },
    }

    save_paths = resolve_save_paths()
    print(" Saving per-regime model copies")
    print(" " + "-" * 68)
    for p in save_paths:
        save_payload(p, payload)
        print(f"   -> {p}  ({os.path.getsize(p)} bytes)")
    print()

    print(" Round 9 training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
