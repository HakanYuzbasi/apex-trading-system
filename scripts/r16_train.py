"""scripts/r16_train.py — Round 16: sigmoid + raw GBM models with leakage_check.

FIX 1A: CalibratedClassifierCV(method='sigmoid', cv=5)
FIX 1B: raw GBM predict_proba  (no calibration wrapper)
Saves r16_sig_*.pkl and r16_raw_*.pkl to models/saved_advanced/.
"""
from __future__ import annotations

import os, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.calibration import CalibratedClassifierCV

from backtesting.real_signal_adapter import ML_FEATURE_NAMES
from scripts.r16_data import apply_synth_fallback
from core.logging_config import setup_logging
from models.ml_validator import leakage_check
from scripts.r15_train import (
    FORWARD_HORIZON, MIN_REGIME_SAMPLES, REGIME_MEAN_REV, REGIME_TRENDING,
    REGIME_VOLATILE, SAVE_DIR, SYMBOLS, TRAIN_END, TRAIN_FRACTION, TRAIN_START,
    _baseline_stats, _eval, _make_gbm, build_frame, fetch_panel,
)

REGIMES = (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE)
FEAT_COLS = list(ML_FEATURE_NAMES)


def _train_set(
    combined,
    full_gbm,
    full_stats,
    prefix: str,
    calibrate: bool,
    method: str | None,
) -> None:
    for regime in REGIMES:
        sub = combined[combined["regime"] == regime].copy()
        n = len(sub)
        if n < MIN_REGIME_SAMPLES:
            model, stats = full_gbm, full_stats
            print(f"  {prefix}_{regime.lower():<12}: FALLBACK  (n={n} < {MIN_REGIME_SAMPLES})")
        else:
            leakage_check(
                sub[FEAT_COLS + ["label"]], label_col="label",
                feature_cols=FEAT_COLS, max_future_shift=FORWARD_HORIZON,
                leak_corr_threshold=0.98, raise_on_fail=True,
            )
            sp = int(n * TRAIN_FRACTION)
            X_tr = sub.iloc[:sp][FEAT_COLS].to_numpy()
            y_tr = sub.iloc[:sp]["label"].to_numpy()
            X_te = sub.iloc[sp:][FEAT_COLS].to_numpy()
            y_te = sub.iloc[sp:]["label"].to_numpy()
            base = _make_gbm()
            base.fit(X_tr, y_tr)
            if calibrate:
                model = CalibratedClassifierCV(base, cv=5, method=method)
                model.fit(X_tr, y_tr)
            else:
                model = base
            stats = _baseline_stats(model, X_tr)
            tr_a, _ = _eval(model, X_tr, y_tr)
            te_a, _ = _eval(model, X_te, y_te)
            print(f"  {prefix}_{regime.lower():<12}: n={n}  "
                  f"train_acc={tr_a:.4f}  test_acc={te_a:.4f}")
        path = SAVE_DIR / f"{prefix}_{regime.lower()}.pkl"
        payload = {
            "model": model,
            "baseline_stats": stats,
            "feature_names": FEAT_COLS,
            "training_metadata": {"regime": regime, "method": method, "n": n,
                                  "calibrated": calibrate},
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    saved → {path}  ({os.path.getsize(path):,} bytes)")


def main() -> int:
    apply_synth_fallback()
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)
    print("=" * 72)
    print(" Round 16 — r16_train.py")
    print(" FIX 1A: CalibratedClassifierCV(method='sigmoid', cv=5)")
    print(" FIX 1B: raw GBM predict_proba  (no calibration)")
    print("=" * 72)
    assert len(ML_FEATURE_NAMES) == 11, f"expected 11 features, got {len(ML_FEATURE_NAMES)}"

    print(f" Fetching {len(SYMBOLS)}-symbol panel {TRAIN_START} → {TRAIN_END} ...")
    panel = fetch_panel(SYMBOLS, TRAIN_START, TRAIN_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")

    print(" Building feature matrix ...")
    combined = build_frame(panel)
    print(f"   total rows: {len(combined)}  features: {len(FEAT_COLS)}")

    print(" Running global leakage_check ...")
    leakage_check(
        combined[FEAT_COLS + ["label"]], label_col="label",
        feature_cols=FEAT_COLS, max_future_shift=FORWARD_HORIZON,
        leak_corr_threshold=0.98, raise_on_fail=True,
    )
    print("   PASSED")

    sp = int(len(combined) * TRAIN_FRACTION)
    full_gbm = _make_gbm()
    full_gbm.fit(combined.iloc[:sp][FEAT_COLS].to_numpy(),
                 combined.iloc[:sp]["label"].to_numpy())
    full_stats = _baseline_stats(full_gbm, combined.iloc[:sp][FEAT_COLS].to_numpy())
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n FIX 1A — sigmoid calibration")
    print("-" * 72)
    _train_set(combined, full_gbm, full_stats, "r16_sig", True, "sigmoid")

    print("\n FIX 1B — raw GBM (no calibration)")
    print("-" * 72)
    _train_set(combined, full_gbm, full_stats, "r16_raw", False, None)

    print("\n Round 16 training complete.")
    print("FILE COMPLETE: r16_train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
