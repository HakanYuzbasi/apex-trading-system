"""r18_wf.py — R18 WF: 2020 fold CB analysis, 2024-25 OOS, smoke tests."""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from r17_train import UNIVERSE, build_features, load_sym
from r18_train import (OUT_DIR, WF_START, WF_END, OOS_START, OOS_END,
                        TRAIN_BARS, OOS_BARS, SIGNAL_THRESHOLD, KELLY_FRACTION, train_model)
from r18_backtest import VolCircuitBreaker, _atr_n, backtest

RESULTS_FILE = "backtest_results_round18.txt"


def _score_oos(X_tr, y_tr, oos_items: list, cb_enabled: bool) -> dict:
    sc = StandardScaler(); Xts = sc.fit_transform(X_tr)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=4,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(Xts, y_tr)
    cb = VolCircuitBreaker(enabled=cb_enabled); daily: list[float] = []
    for Xo, fwd, df_sub, dates in oos_items:
        prob = gbm.predict_proba(sc.transform(Xo))[:, 1]
        a5  = _atr_n(df_sub, 5).reindex(dates, method="ffill").fillna(0).values
        a60 = _atr_n(df_sub,60).reindex(dates, method="ffill").fillna(1).values
        for j, dt in enumerate(dates):
            mult, _ = cb.evaluate(float(a5[j]), float(a60[j]), dt)
            daily.append(float(prob[j]>SIGNAL_THRESHOLD)*KELLY_FRACTION*mult *
                         float(fwd.reindex([dt]).fillna(0).iloc[0]))
    dr = pd.Series(daily).replace([np.inf,-np.inf], np.nan).dropna()
    if dr.empty or dr.std() == 0: return {"sharpe":0.,"comp":0.,"dd":0.}
    eq = (1+dr).cumprod()
    return {"sharpe":round(float(dr.mean()/dr.std()*np.sqrt(252)),3),
            "comp":  round(float(eq.iloc[-1]-1)*100,2),
            "dd":    round(float(((eq-eq.cummax())/eq.cummax()).min())*100,2)}


def _build_xy(sym_data, idx_set, spy_ret):
    Xl, yl = [], []
    for sym, df in sym_data.items():
        sub = df.loc[df.index.isin(idx_set)]
        if len(sub) < 50: continue
        X = build_features(sub, macro=False, spy_ret=spy_ret.reindex(sub.index, fill_value=0.0))
        X = X.replace([np.inf,-np.inf], np.nan).dropna()
        y = (sub["Close"].pct_change(1).shift(-1)>0).astype(int).reindex(X.index).dropna()
        X = X.loc[y.index]; Xl.append(X.reset_index(drop=True)); yl.append(y.reset_index(drop=True))
    if not Xl: return None, None
    Xc = pd.concat(Xl,ignore_index=True).dropna(); yc = pd.concat(yl,ignore_index=True).reindex(Xc.index).dropna()
    return Xc.loc[yc.index], yc


def walk_forward(sym_data: dict, spy_ret: pd.Series) -> dict:
    biz = sym_data["SPY"].index
    folds_on, folds_off, f20_on, f20_off = [], [], None, None
    i = TRAIN_BARS
    while i + OOS_BARS <= len(biz):
        oos_days = biz[i:i+OOS_BARS]; oos_set = set(oos_days)
        is_2020  = (oos_days[0] <= pd.Timestamp("2020-05-01") and
                    oos_days[-1] >= pd.Timestamp("2020-02-01"))
        X_tr, y_tr = _build_xy(sym_data, set(biz[max(0,i-TRAIN_BARS):i]), spy_ret)
        if X_tr is None: i += OOS_BARS; continue
        oos_items = []
        for sym, df in sym_data.items():
            sub = df.loc[df.index.isin(oos_set)]
            if len(sub) < 10: continue
            Xo = build_features(sub,macro=False,spy_ret=spy_ret.reindex(sub.index,fill_value=0.0))
            Xo = Xo.replace([np.inf,-np.inf],np.nan).dropna()
            dates = sub.index[sub.index.isin(Xo.index)]
            oos_items.append((Xo, sub["Close"].pct_change(1).shift(-1),
                               df.loc[df.index.isin(oos_set)], dates))
        r_on  = _score_oos(X_tr, y_tr, oos_items, True)
        r_off = _score_oos(X_tr, y_tr, oos_items, False)
        folds_on.append(r_on); folds_off.append(r_off)
        if is_2020:
            f20_on, f20_off = r_on, r_off
            print(f"  2020 fold [{oos_days[0].date()}→{oos_days[-1].date()}]: "
                  f"DD_ON={r_on['dd']}%  DD_OFF={r_off['dd']}%  Sharpe_ON={r_on['sharpe']}")
        i += OOS_BARS
    def summ(fs):
        sh=[f["sharpe"] for f in fs]; dd=[f["dd"] for f in fs]
        return {"mean_sharpe":round(float(np.mean(sh)),3) if sh else 0,
                "positive":sum(s>0 for s in sh),"total":len(sh),
                "worst_dd":round(float(min(dd)),2) if dd else 0}
    return {"on":summ(folds_on),"off":summ(folds_off),"2020_on":f20_on,"2020_off":f20_off}


def run_smoke_tests() -> list[str]:
    res = []
    def chk(n, fn):
        try: fn(); res.append(f"  PASS  {n}")
        except Exception as e: res.append(f"  FAIL  {n}: {e}")
    chk("T1 r17_train imports",    lambda: __import__("r17_train"))
    chk("T2 r18_train imports",    lambda: __import__("r18_train"))
    chk("T3 r18_backtest imports", lambda: __import__("r18_backtest"))
    def t4():
        cb = VolCircuitBreaker()
        assert cb.evaluate(1.0, 2.0)[1] == "normal"
        assert cb.evaluate(2.5, 1.0)[1] == "scale"
        assert cb.evaluate(3.5, 1.0)[1] == "halt"
        assert cb.evaluate(4.5, 1.0)[1] == "close"
    chk("T4 VolCB mode thresholds (normal/scale/halt/close)", t4)
    def t5():
        mult, mode = VolCircuitBreaker().evaluate(2.5, 1.0)
        assert mode == "scale" and abs(mult - 0.25) < 1e-9
    chk("T5 VolCB scale mult=0.25", t5)
    def t6():
        cb = VolCircuitBreaker(); cb.evaluate(3.5, 1.0, "2020-03-12")
        assert len(cb.log) == 1 and cb.log[0]["to"] == "halt"
    chk("T6 VolCB logs regime transition", t6)
    def t7():
        from r18_train import VOL_CB_RATIO_SCALE, VOL_CB_RATIO_HALT, VOL_CB_RATIO_CLOSE
        assert VOL_CB_RATIO_SCALE == 2.0 and VOL_CB_RATIO_HALT == 3.0 and VOL_CB_RATIO_CLOSE == 4.0
    chk("T7 VOL_CB config constants", t7)
    return res


def main():
    OUT_DIR.mkdir(exist_ok=True)
    lines = ["=" * 60, "BACKTEST RESULTS ROUND 18", "=" * 60, ""]
    smoke = run_smoke_tests()
    for s in smoke: print(s)
    lines += ["SMOKE TESTS:", *smoke, f"Result: {sum('PASS' in s for s in smoke)}/7", ""]
    train_model()
    print("\n[WALK-FORWARD]")
    sym_data = {sym: load_sym(sym, WF_START, WF_END) for sym in UNIVERSE}
    wf = walk_forward(sym_data, sym_data["SPY"]["Close"].pct_change(1))
    lines += ["WALK-FORWARD (WF_START=2018, OOS=126 bars):",
              f"  CB ON:  MeanSharpe={wf['on']['mean_sharpe']}  PosFolds={wf['on']['positive']}/{wf['on']['total']}  WorstDD={wf['on']['worst_dd']}%",
              f"  CB OFF: MeanSharpe={wf['off']['mean_sharpe']}  PosFolds={wf['off']['positive']}/{wf['off']['total']}  WorstDD={wf['off']['worst_dd']}%"]
    if wf["2020_on"]:
        lines += ["", "2020 COVID Fold (March 2020 in OOS):",
                  f"  CB ON:  DD={wf['2020_on']['dd']}%  Sharpe={wf['2020_on']['sharpe']}",
                  f"  CB OFF: DD={wf['2020_off']['dd']}%  Sharpe={wf['2020_off']['sharpe']}"]
    print("[2024-2025 OOS]")
    oos = backtest(cb_enabled=True, start=OOS_START, end=OOS_END)
    print(f"  {oos['Return']}%  Sharpe={oos['Sharpe']}  MaxDD={oos['MaxDD']}%  Trades={oos['Trades']}")
    lines += ["", "2024-2025 OOS Forward Test (CB=ON):",
              f"  Return={oos['Return']}%  Sharpe={oos['Sharpe']}  MaxDD={oos['MaxDD']}%  Trades={oos['Trades']}",
              "  Monthly:"] + [f"    {ym}: {pnl:+.2f}" for ym, pnl in oos["monthly"].items()]
    def chk(c, lbl): return f"  {'PASS' if c else 'FAIL'}  {lbl}"
    f20 = wf["2020_on"]["dd"] if wf["2020_on"] else -99.0
    lines += ["", "TARGET CHECKS:",
              chk(f20 > -15,                       f"2020 fold DD >-15% CB ON  ({f20}%)"),
              chk(oos["Sharpe"] > 0.40,             f"2024-25 Sharpe >0.40      ({oos['Sharpe']})"),
              chk(wf["on"]["mean_sharpe"] > 0.30,   f"WF Mean Sharpe >0.30      ({wf['on']['mean_sharpe']})"),
              "", "=" * 60]
    output = "\n".join(lines); print("\n" + output)
    open(RESULTS_FILE, "w").write(output + "\n"); print(f"Saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main(); print("FILE COMPLETE: r18_wf.py")
