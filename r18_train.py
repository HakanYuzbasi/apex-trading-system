"""r18_train.py — R18 training: model up to 2023-12-31, vol circuit breaker config.
Raises RuntimeError if yfinance fails. 8 features (R17 winner).
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from r17_train import UNIVERSE, build_features, load_sym, _fetch_vix

# ── Date ranges ───────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
TRAIN_END   = "2023-12-31"   # strict cutoff — no 2024+ lookahead
OOS_START   = "2024-01-01"
OOS_END     = "2025-12-31"
WF_START    = "2018-01-01"
WF_END      = "2024-12-31"
TRAIN_BARS  = 504             # ~2 years of business days
OOS_BARS    = 126             # best OOS window from R17
OUT_DIR     = Path("r18_artifacts")

# ── Fixed backtest params ─────────────────────────────────────────────────────
SIGNAL_THRESHOLD = 0.55
INITIAL_CAPITAL  = 100_000.0
KELLY_FRACTION   = 0.5

# ── FIX 1: Volatility circuit breaker config ──────────────────────────────────
VOL_CB_ENABLED     = True
VOL_CB_RATIO_SCALE = 2.0    # ATR(5)/ATR(60) > 2.0 → size × 0.25
VOL_CB_RATIO_HALT  = 3.0    # > 3.0 → no new entries
VOL_CB_RATIO_CLOSE = 4.0    # > 4.0 → close all positions immediately


def train_model(start: str = TRAIN_START, end: str = TRAIN_END) -> dict:
    """Train GBM on [start, end] with 8 features (no macro, no blackout)."""
    OUT_DIR.mkdir(exist_ok=True)
    spy_ret = load_sym("SPY", start, end)["Close"].pct_change(1)
    all_X, all_y = [], []
    for sym in UNIVERSE:
        df = load_sym(sym, start, end)
        X  = build_features(df, macro=False,
                             spy_ret=spy_ret.reindex(df.index, fill_value=0.0))
        y  = (df["Close"].pct_change(1).shift(-1) > 0).astype(int)
        X  = X.dropna(); y = y.reindex(X.index).dropna(); X = X.loc[y.index]
        all_X.append(X.assign(_s=sym).reset_index(drop=True))
        all_y.append(y.reset_index(drop=True))

    X_all = pd.concat(all_X, ignore_index=True).replace([np.inf,-np.inf], np.nan).dropna()
    y_all = pd.concat(all_y, ignore_index=True).reindex(X_all.index).dropna()
    X_all = X_all.loc[y_all.index].drop(columns=["_s"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    gbm.fit(Xs, y_all)

    feat_names = list(X_all.columns)
    top5 = [(f, float(v)) for f, v in
            sorted(zip(feat_names, gbm.feature_importances_), key=lambda x: -x[1])[:5]]
    art  = {"gbm": gbm, "scaler": scaler, "feat_names": feat_names,
            "top5": top5, "train_end": end}
    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(art, f)
    print(f"  Trained: samples={len(X_all)} feats={len(feat_names)} end={end}")
    return art


def load_model() -> dict:
    path = OUT_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError("Run r18_train.py first (model.pkl missing)")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train_model()
    print("FILE COMPLETE: r18_train.py")
