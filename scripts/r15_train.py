"""
scripts/r15_train.py — Round 15 feature engineering + model training.

Round 15 applies four fixes to close the gaps identified in Round 14:

  FIX 1 (code — advanced_backtester.py line 1230)
        Confidence-proportional dynamic threshold formula:
          threshold = 0.30 + 0.40 * max(0, 0.55 − confidence)
        Replaces the binary step-function that always evaluated to 0.55 on
        every candidate bar with SGD signals, starving the entry gate. The
        new formula collapses to the floor (0.30) when confidence ≥ 0.55,
        letting high-confidence GBM signals through. Controlled by the new
        APEX_DYN_THRESH_CONF_PROP flag.

  FIX 2 (training)
        GBM primary classifier restored (SGD demoted to online-only).
        R14 showed the SGD swap dropped trade count from 73 to 10 on the
        2023 window. This run retrains GradientBoostingClassifier with
        improved hyperparameters (n_estimators=300, min_samples_leaf=5,
        max_features='sqrt') and saves to models/saved_advanced/r15_*.pkl.

  FIX 3 (code — real_signal_adapter.py)
        Extended feature set: 8 → 11 features. Three new features added
        to compute_ml_features / ML_FEATURE_NAMES:
          bb_pctb          Bollinger %B (20-period, 2σ)
          roc_10           10-bar rate of change
          price_vs_high60d position vs 60-day high/low range

  FIX 4 (training)
        Platt-calibrated GBMs using CalibratedClassifierCV(cv=5,
        method='isotonic'). Better probability estimates → more accurate
        confidence scores → confidence-proportional threshold (FIX 1)
        works as intended. Saved to models/saved_advanced/r15_cal_*.pkl.

Output
------
  models/saved_advanced/r15_trending.pkl    GBM basic (FIX 2+3)
  models/saved_advanced/r15_mean_rev.pkl
  models/saved_advanced/r15_volatile.pkl
  models/saved_advanced/r15_cal_trending.pkl  Calibrated GBM (FIX 4)
  models/saved_advanced/r15_cal_mean_rev.pkl
  models/saved_advanced/r15_cal_volatile.pkl
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.real_signal_adapter import ML_FEATURE_NAMES, compute_ml_features
from config import ApexConfig
from core.logging_config import setup_logging
from models.ml_validator import leakage_check

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
    "GLD", "TLT", "IWM",
)
TRAIN_START: str = "2018-01-01"
TRAIN_END: str = "2023-01-01"
FORWARD_HORIZON: int = 5
RANDOM_STATE: int = 42
TRAIN_FRACTION: float = 0.80

ADX_PERIOD: int = 14
ATR_PERIOD: int = 14
ADX_TRENDING_MIN: float = 25.0
ADX_MEAN_REV_MAX: float = 20.0
VOL_MAX: float = 0.015
MIN_REGIME_SAMPLES: int = 200

REGIME_TRENDING = "TRENDING"
REGIME_MEAN_REV = "MEAN_REV"
REGIME_VOLATILE = "VOLATILE"

SAVE_DIR = Path(__file__).resolve().parents[1] / "models" / "saved_advanced"


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest
# ─────────────────────────────────────────────────────────────────────────────

def fetch_panel(symbols: Tuple[str, ...], start: str, end: str) -> Dict[str, pd.DataFrame]:
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
        raise RuntimeError(f"No bars fetched for {symbols}")
    return panel


def _sha16(panel: Dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for sym in sorted(panel):
        h.update(sym.encode())
        h.update(panel[sym]["Close"].to_numpy().tobytes())
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, p: int) -> pd.Series:
    high, low, close = high.astype(float), low.astype(float), close.astype(float)
    plus_dm = high.diff().clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(1)
    tr_s = tr.rolling(p, min_periods=p).sum()
    pdi = 100.0 * (plus_dm.rolling(p, min_periods=p).sum() / tr_s)
    mdi = 100.0 * (minus_dm.rolling(p, min_periods=p).sum() / tr_s)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.rolling(p, min_periods=p).mean()


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, p: int) -> pd.Series:
    high, low, close = high.astype(float), low.astype(float), close.astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(1)
    return tr.ewm(alpha=1.0 / p, adjust=False).mean()


def _classify(adx: float, vol_ratio: float) -> str:
    if not np.isfinite(vol_ratio):
        return REGIME_TRENDING
    if vol_ratio >= VOL_MAX:
        return REGIME_VOLATILE
    if np.isfinite(adx) and adx > ADX_TRENDING_MIN:
        return REGIME_TRENDING
    if np.isfinite(adx) and adx < ADX_MEAN_REV_MAX and vol_ratio < VOL_MAX:
        return REGIME_MEAN_REV
    return REGIME_TRENDING


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_frame(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sym, ohlcv in panel.items():
        feats = compute_ml_features(ohlcv)
        close = ohlcv["Close"].astype(float)
        high = ohlcv["High"].astype(float)
        low = ohlcv["Low"].astype(float)

        adx = _compute_adx(high, low, close, ADX_PERIOD).rename("adx")
        atr = _compute_atr(high, low, close, ATR_PERIOD)
        vol_ratio = (atr / close).rename("vol_ratio")

        fwd = np.log(close.shift(-FORWARD_HORIZON) / close)
        label = np.sign(fwd).replace(0.0, np.nan).rename("label")

        joined = feats.join(adx).join(vol_ratio).join(label).dropna()
        joined = joined.assign(
            symbol=sym,
            regime=[_classify(a, v) for a, v in zip(joined["adx"], joined["vol_ratio"])],
        )
        frames.append(joined)
    df = pd.concat(frames, axis=0).sort_index()
    df["label"] = df["label"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gbm() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )


def _eval(model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(X) == 0:
        return float("nan"), float("nan")
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_",
                            getattr(getattr(model, "estimator", model), "classes_", [-1, 1])))
    acc = float(accuracy_score(y, pred))
    ll = float(log_loss(y, proba, labels=classes))
    return acc, ll


def _baseline_stats(model, X: np.ndarray) -> Dict[str, float]:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_",
                            getattr(getattr(model, "estimator", model), "classes_", [-1, 1])))
    up_idx = classes.index(1) if 1 in classes else -1
    p_up = proba[:, up_idx]
    signed = np.clip((p_up - 0.5) * 2.0, -1.0, 1.0)
    return {"mean": float(np.mean(signed)), "std": float(np.std(signed, ddof=0)), "n": int(len(signed))}


def train_regime(
    subset: pd.DataFrame,
    feat_cols: List[str],
    regime_name: str,
    fallback_model: Any,
    fallback_stats: Dict[str, float],
    *,
    calibrate: bool = False,
) -> Dict[str, Any]:
    if len(subset) < MIN_REGIME_SAMPLES:
        logger.warning("Regime %s has %d rows — using fallback", regime_name, len(subset))
        return {"model": fallback_model, "baseline_stats": fallback_stats,
                "feature_names": feat_cols,
                "training_metadata": {"regime": regime_name, "fell_back": True,
                                      "n": len(subset)}}

    leakage_check(
        subset[feat_cols + ["label"]],
        label_col="label", feature_cols=feat_cols,
        max_future_shift=FORWARD_HORIZON, leak_corr_threshold=0.98,
        raise_on_fail=True,
    )

    split = int(len(subset) * TRAIN_FRACTION)
    X_tr = subset.iloc[:split][feat_cols].to_numpy()
    y_tr = subset.iloc[:split]["label"].to_numpy()
    X_te = subset.iloc[split:][feat_cols].to_numpy()
    y_te = subset.iloc[split:]["label"].to_numpy()

    base = _make_gbm()
    base.fit(X_tr, y_tr)

    if calibrate:
        model = CalibratedClassifierCV(base, cv=5, method="isotonic")
        model.fit(X_tr, y_tr)
    else:
        model = base

    tr_acc, tr_ll = _eval(model, X_tr, y_tr)
    te_acc, te_ll = _eval(model, X_te, y_te)
    stats = _baseline_stats(model, X_tr)

    return {
        "model": model,
        "baseline_stats": stats,
        "feature_names": feat_cols,
        "training_metadata": {
            "regime": regime_name,
            "calibrated": calibrate,
            "fell_back": False,
            "n": len(subset),
            "train_rows": len(X_tr),
            "test_rows": len(X_te),
            "train_accuracy": tr_acc,
            "test_accuracy": te_acc,
            "train_log_loss": tr_ll,
            "test_log_loss": te_ll,
            "n_features": len(feat_cols),
            "feature_names": feat_cols,
            "random_state": RANDOM_STATE,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 15 — r15_train.py")
    print(" FIX 1: conf-proportional threshold  [applied to advanced_backtester.py]")
    print(" FIX 2: GBM primary classifier restored")
    print(" FIX 3: 11-feature set               [applied to real_signal_adapter.py]")
    print(" FIX 4: Platt-calibrated GBMs (isotonic CalibratedClassifierCV)")
    print("=" * 72)

    # ── Verify code fixes are in place ───────────────────────────────────────
    assert hasattr(ApexConfig, "DYN_THRESH_CONF_PROP"), \
        "FIX 1: DYN_THRESH_CONF_PROP missing from ApexConfig"
    assert len(ML_FEATURE_NAMES) == 11, \
        f"FIX 3: expected 11 features, got {len(ML_FEATURE_NAMES)}"
    print()
    print(f" FIX 1 verified: DYN_THRESH_CONF_PROP in ApexConfig")
    print(f" FIX 3 verified: ML_FEATURE_NAMES has {len(ML_FEATURE_NAMES)} features:")
    for fn in ML_FEATURE_NAMES:
        print(f"   - {fn}")
    print()

    # ── Fetch training data ───────────────────────────────────────────────────
    print(f" Fetching {len(SYMBOLS)}-symbol panel {TRAIN_START} → {TRAIN_END} ...")
    panel = fetch_panel(SYMBOLS, TRAIN_START, TRAIN_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    sha = _sha16(panel)
    print(f" Panel SHA-16: {sha}")
    print()

    # ── Build 11-feature matrix ───────────────────────────────────────────────
    print(" Building 11-feature matrix with regime labels ...")
    combined = build_frame(panel)
    feat_cols = list(ML_FEATURE_NAMES)
    total = len(combined)
    regime_counts = combined["regime"].value_counts().to_dict()
    for r in (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE):
        n = int(regime_counts.get(r, 0))
        print(f"   {r:<10}: {n:>6} rows ({n / total:.1%})")
    print(f"   TOTAL     : {total:>6} rows, {len(feat_cols)} features")
    print()

    # ── Full-dataset fallback ─────────────────────────────────────────────────
    print(" Training full-dataset fallback GBM ...")
    leakage_check(
        combined[feat_cols + ["label"]],
        label_col="label", feature_cols=feat_cols,
        max_future_shift=FORWARD_HORIZON, leak_corr_threshold=0.98,
        raise_on_fail=True,
    )
    split_full = int(total * TRAIN_FRACTION)
    X_ft = combined.iloc[:split_full][feat_cols].to_numpy()
    y_ft = combined.iloc[:split_full]["label"].to_numpy()
    X_fv = combined.iloc[split_full:][feat_cols].to_numpy()
    y_fv = combined.iloc[split_full:]["label"].to_numpy()
    full_gbm = _make_gbm()
    full_gbm.fit(X_ft, y_ft)
    full_stats = _baseline_stats(full_gbm, X_ft)
    fa_tr, _ = _eval(full_gbm, X_ft, y_ft)
    fa_te, _ = _eval(full_gbm, X_fv, y_fv)
    print(f"   train acc: {fa_tr:.4f}   test acc: {fa_te:.4f}")
    print(f"   baseline stats: mean={full_stats['mean']:.4f}  std={full_stats['std']:.4f}")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── FIX 2: Train per-regime GBMs ─────────────────────────────────────────
    print(" FIX 2 — Training per-regime GBMs (n_estimators=300, min_samples_leaf=5)")
    print("-" * 72)
    basic_models: Dict[str, Dict[str, Any]] = {}
    for regime in (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE):
        subset = combined[combined["regime"] == regime].copy()
        payload = train_regime(subset, feat_cols, regime, full_gbm, full_stats, calibrate=False)
        meta = payload["training_metadata"]
        if meta.get("fell_back"):
            print(f"   {regime:<10}: FALLBACK ({meta['n']} rows < {MIN_REGIME_SAMPLES})")
        else:
            print(f"   {regime:<10}: {meta['train_rows']} train / {meta['test_rows']} test   "
                  f"train_acc={meta['train_accuracy']:.4f}  test_acc={meta['test_accuracy']:.4f}")
        path = str(SAVE_DIR / f"r15_{regime.lower()}.pkl")
        payload["training_metadata"].update({"sha16": sha, "train_start": TRAIN_START, "train_end": TRAIN_END})
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"             saved → {path}  ({os.path.getsize(path):,} bytes)")
        basic_models[regime] = payload
    print()

    # ── FIX 4: Train calibrated GBMs ─────────────────────────────────────────
    print(" FIX 4 — Training calibrated GBMs (CalibratedClassifierCV isotonic cv=5)")
    print("-" * 72)
    for regime in (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE):
        subset = combined[combined["regime"] == regime].copy()
        payload = train_regime(subset, feat_cols, regime, full_gbm, full_stats, calibrate=True)
        meta = payload["training_metadata"]
        if meta.get("fell_back"):
            print(f"   {regime:<10}: FALLBACK")
        else:
            print(f"   {regime:<10}: train_acc={meta['train_accuracy']:.4f}  "
                  f"test_acc={meta['test_accuracy']:.4f}  (calibrated)")
        path = str(SAVE_DIR / f"r15_cal_{regime.lower()}.pkl")
        payload["training_metadata"].update({"sha16": sha, "train_start": TRAIN_START, "train_end": TRAIN_END})
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"             saved → {path}  ({os.path.getsize(path):,} bytes)")
    print()

    print(" Round 15 training complete.")
    print(f" Saved models in {SAVE_DIR}/")
    for fname in sorted(SAVE_DIR.iterdir()):
        if fname.suffix == ".pkl":
            print(f"   {fname.name:<35} {os.path.getsize(fname):>12,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
