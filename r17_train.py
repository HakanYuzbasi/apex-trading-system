"""r17_train.py — R17 training: GBM+SGD, 8/11 features, earnings blackout."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

UNIVERSE    = ["SPY","QQQ","IWM","GLD","TLT","XLE","XLF","XLK","AAPL","MSFT","JPM","XOM"]
TRAIN_START = "2021-01-01"
TRAIN_END   = "2022-12-31"
OUT_DIR     = Path("r17_artifacts")

# FIX 1 — regime Kelly constants (imported by r17_backtest)
KELLY_FRACTION             = 0.5
KELLY_REGIME_SCALE_ON      = 1.0
KELLY_REGIME_SCALE_OFF     = 0.4
KELLY_REGIME_SCALE_HIGHVOL = 0.2
HIGH_VOL_ATR_THRESH        = 0.025


def _atr(h, l, c, n=20):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(c, n=14):
    d = c.diff(); u = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = u.rolling(n).mean() / dn.rolling(n).mean().replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

_vix_cache: dict = {}

def _fetch_vix(start: str, end: str) -> pd.Series:
    key = f"{start}_{end}"
    if key not in _vix_cache:
        raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            raise RuntimeError("yfinance failed: ^VIX returned empty data")
        _vix_cache[key] = (raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]).squeeze()
    return _vix_cache[key]


def build_features(df: pd.DataFrame, macro: bool,
                   spy_ret: pd.Series = None, vix: pd.Series = None) -> pd.DataFrame:
    """8 base features; macro=True adds spy_mom5, vix_level, vix_chg5 (11 total)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    r1   = c.pct_change(1)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    X = pd.DataFrame({
        "r1":        r1,
        "r5":        c.pct_change(5),
        "r20":       c.pct_change(20),
        "vol20":     r1.rolling(20).std() * np.sqrt(252),
        "rsi14":     _rsi(c),
        "atr_ratio": _atr(h, l, c) / c.replace(0, np.nan),
        "vol_ratio": v / v.rolling(20).mean().replace(0, np.nan),
        "macd":      (ema12 - ema26) / c.replace(0, np.nan),
    }, index=df.index)
    if macro:
        if spy_ret is None or vix is None:
            raise ValueError("macro=True requires spy_ret and vix")
        vix_r = vix.reindex(df.index, method="ffill")
        X["spy_mom5"]  = spy_ret.reindex(df.index, fill_value=0.0).rolling(5).mean()
        X["vix_level"] = vix_r / 30.0
        X["vix_chg5"]  = vix_r.pct_change(5)
    return X


def _earnings_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=index)
    for q in pd.date_range(index[0], index[-1], freq="Q"):
        bd = q - pd.offsets.BDay(1)
        for delta in range(-2, 3):
            d = bd + pd.offsets.BDay(delta)
            if d in mask.index:
                mask[d] = True
    return mask


def load_sym(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"yfinance failed: {symbol} returned empty data")
    return df


def main(macro: bool = True, blackout: bool = True,
         sgd_w: float = 0.1, start: str = TRAIN_START, end: str = TRAIN_END) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    spy_ret = load_sym("SPY", start, end)["Close"].pct_change(1)
    vix     = _fetch_vix(start, end) if macro else None
    all_X, all_y = [], []
    for sym in UNIVERSE:
        df = load_sym(sym, start, end)
        X  = build_features(df, macro=macro, spy_ret=spy_ret.reindex(df.index, fill_value=0.0), vix=vix)
        y  = (df["Close"].pct_change(1).shift(-1) > 0).astype(int)
        if blackout:
            X, y = X.loc[~_earnings_mask(df.index)], y.loc[~_earnings_mask(df.index)]
        X  = X.dropna(); y = y.reindex(X.index).dropna(); X = X.loc[y.index]
        all_X.append(X.assign(_sym=sym).reset_index(drop=True))
        all_y.append(y.reset_index(drop=True))

    X_all = pd.concat(all_X, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    y_all = pd.concat(all_y, ignore_index=True).reindex(X_all.index).dropna()
    X_all = X_all.loc[y_all.index].drop(columns=["_sym"])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gbm.fit(Xs, y_all)

    sgd = SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42)
    sgd.fit(Xs, y_all)

    feat_names = list(X_all.columns)
    fi_sorted = [(f, float(v)) for f, v in
                 sorted(zip(feat_names, gbm.feature_importances_), key=lambda x: -x[1])[:5]]

    artifacts = {"gbm": gbm, "sgd": sgd, "scaler": scaler,
                 "feat_names": feat_names, "macro": macro, "blackout": blackout,
                 "sgd_w": sgd_w, "lstm_fallback": 0.15, "top5_features": fi_sorted}
    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    json.dump({"macro": macro, "blackout": blackout, "sgd_w": sgd_w, "n_features": len(feat_names),
               "top5_features": fi_sorted, "train_samples": int(len(X_all))},
              open(OUT_DIR / "train_meta.json", "w"), indent=2)
    print(f"  feats={len(feat_names)} samples={len(X_all)} macro={macro} blackout={blackout} top5={fi_sorted}")
    return artifacts


if __name__ == "__main__":
    macro_flag = "--no-macro"    not in sys.argv
    black_flag = "--no-blackout" not in sys.argv
    sgd_arg    = float(sys.argv[sys.argv.index("--sgd")+1]) if "--sgd" in sys.argv else 0.1
    main(macro=macro_flag, blackout=black_flag, sgd_w=sgd_arg)
    print("FILE COMPLETE: r17_train.py")
