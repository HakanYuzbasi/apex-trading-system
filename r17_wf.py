"""r17_wf.py — R17 walk-forward: OOS window sweep A/B/C (FIX 3).
Raises RuntimeError if yfinance fails.
"""
import json
import numpy as np
import pandas as pd

from r17_train import UNIVERSE, OUT_DIR, build_features, load_sym, _fetch_vix

WF_START   = "2019-01-01"
WF_END     = "2024-12-31"
TRAIN_BARS = 504              # ~2 years of daily bars
OOS_CONFIGS = {"A": 63, "B": 126, "C": 189}
SIGNAL_THRESHOLD = 0.55
KELLY_FRACTION   = 0.5


def _fold_sharpe(X_tr, y_tr, X_oos, fwd_oos):
    """Train GBM on fold train, score on fold OOS. Returns (sharpe, compounded_ret)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    sc  = StandardScaler()
    Xts = sc.fit_transform(X_tr)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(Xts, y_tr)
    Xos   = sc.transform(X_oos)
    prob  = gbm.predict_proba(Xos)[:, 1]
    sig   = (prob > SIGNAL_THRESHOLD).astype(float)
    daily = (sig * fwd_oos * KELLY_FRACTION).dropna()
    if daily.empty or daily.std() == 0:
        return 0.0, 0.0
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
    comp   = float((1 + daily).prod() - 1) * 100
    return sharpe, comp


def walk_forward(oos_bars: int, all_data: dict, biz_days: pd.DatetimeIndex,
                 macro: bool, vix: pd.Series, spy_ret: pd.Series) -> dict:
    fold_sharpes, fold_comps = [], []
    i = TRAIN_BARS
    while i + oos_bars <= len(biz_days):
        tr_idx  = set(biz_days[max(0, i - TRAIN_BARS):i])
        oos_idx = set(biz_days[i:i + oos_bars])

        # Build cross-symbol train arrays
        Xtr_list, ytr_list = [], []
        for sym, df in all_data.items():
            sub = df.loc[df.index.isin(tr_idx)]
            if len(sub) < 50:
                continue
            vix_s  = vix.reindex(sub.index, method="ffill") if macro else None
            spy_rs = spy_ret.reindex(sub.index, fill_value=0.0)
            X = build_features(sub, macro=macro, spy_ret=spy_rs, vix=vix_s)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = (sub["Close"].pct_change(1).shift(-1) > 0).astype(int).reindex(X.index).dropna()
            X = X.loc[y.index]
            Xtr_list.append(X); ytr_list.append(y)
        if not Xtr_list:
            i += oos_bars; continue

        X_tr = pd.concat(Xtr_list, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        y_tr = pd.concat(ytr_list, ignore_index=True).reindex(X_tr.index).dropna()
        X_tr = X_tr.loc[y_tr.index]

        # Build cross-symbol OOS arrays
        Xos_list, fwd_list = [], []
        for sym, df in all_data.items():
            sub = df.loc[df.index.isin(oos_idx)]
            if len(sub) < 10:
                continue
            vix_s  = vix.reindex(sub.index, method="ffill") if macro else None
            spy_rs = spy_ret.reindex(sub.index, fill_value=0.0)
            X = build_features(sub, macro=macro, spy_ret=spy_rs, vix=vix_s)
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            fwd = sub["Close"].pct_change(1).shift(-1).reindex(X.index).reset_index(drop=True)
            Xos_list.append(X.reset_index(drop=True)); fwd_list.append(fwd)
        if not Xos_list:
            i += oos_bars; continue

        X_oos   = pd.concat(Xos_list, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        fwd_oos = pd.concat(fwd_list, ignore_index=True).reindex(X_oos.index).dropna()
        X_oos   = X_oos.loc[fwd_oos.index]

        sh, comp = _fold_sharpe(X_tr, y_tr, X_oos, fwd_oos)
        fold_sharpes.append(sh); fold_comps.append(comp)
        i += oos_bars

    if not fold_sharpes:
        return {"mean_sharpe": 0.0, "positive_folds": 0, "n_folds": 0,
                "worst_dd": 0.0, "compounded": 0.0}
    return {
        "mean_sharpe":    round(float(np.mean(fold_sharpes)), 3),
        "positive_folds": int(sum(s > 0 for s in fold_sharpes)),
        "n_folds":        len(fold_sharpes),
        "worst_dd":       round(float(min(fold_comps)), 2),
        "compounded":     round(float(sum(fold_comps)), 2),
    }


def main(macro: bool = True) -> dict:
    print("  Loading WF universe data...")
    all_data = {}
    for sym in UNIVERSE:
        all_data[sym] = load_sym(sym, WF_START, WF_END)
    spy_ret = all_data["SPY"]["Close"].pct_change(1)
    vix     = _fetch_vix(WF_START, WF_END) if macro else None
    biz_days = all_data["SPY"].index

    results = {}
    for cfg, oos_bars in OOS_CONFIGS.items():
        print(f"  OOS config {cfg} ({oos_bars} bars)...")
        results[cfg] = walk_forward(oos_bars, all_data, biz_days, macro, vix, spy_ret)
        r = results[cfg]
        print(f"    MeanSharpe={r['mean_sharpe']}  PosFolds={r['positive_folds']}/{r['n_folds']}"
              f"  WorstDD={r['worst_dd']}%  Compounded={r['compounded']}%")

    json.dump(results, open(OUT_DIR / "wf_results.json", "w"), indent=2)
    best = max(results, key=lambda k: results[k]["mean_sharpe"])
    print(f"  Best OOS config: {best} (bars={OOS_CONFIGS[best]})")
    return results


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    main()
    print("FILE COMPLETE: r17_wf.py")
