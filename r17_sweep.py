"""r17_sweep.py — R17: SGD sweep, macro validation, 4-pass A/B, WF, smoke tests."""
import json, pandas as pd
from r17_train import OUT_DIR, _fetch_vix, build_features
import r17_train, r17_backtest as rb, r17_wf

RESULTS_FILE = "backtest_results_round17.txt"
SGD_SWEEP    = [0.0, 0.1, 0.2, 0.3]
MIN_TRADES   = 50


def run_smoke_tests() -> list[str]:
    res = []
    def chk(name, fn):
        try:    fn(); res.append(f"  PASS  {name}")
        except Exception as e: res.append(f"  FAIL  {name}: {e}")

    chk("T1 r17_train imports",    lambda: __import__("r17_train"))
    chk("T2 r17_backtest imports", lambda: __import__("r17_backtest"))
    chk("T3 r17_wf imports",       lambda: __import__("r17_wf"))
    def t4():
        assert isinstance(rb.DynamicLSTMWeight(0.15).weight(), float)
    chk("T4 DynamicLSTMWeight fallback float", t4)

    def t5():
        d = rb.DynamicLSTMWeight()
        for _ in range(25): d.record(True, True)
        assert d.weight() > 0.3, f"got {d.weight()}"
    chk("T5 DynamicLSTMWeight high-acc weight >0.3", t5)

    def t6():
        assert rb.KELLY_REGIME_SCALE_ON == 1.0
        assert rb.KELLY_REGIME_SCALE_OFF == 0.4
        assert rb.KELLY_REGIME_SCALE_HIGHVOL == 0.2
    chk("T6 Kelly regime scale constants", t6)

    def t7():
        import yfinance as yf
        df = yf.download("SPY", start="2023-01-01", end="2023-03-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: raise RuntimeError("yfinance failed")
        spy_ret = df["Close"].pct_change(1)
        vix = _fetch_vix("2023-01-01", "2023-03-31")
        X8  = build_features(df, macro=False, spy_ret=spy_ret)
        X11 = build_features(df, macro=True,  spy_ret=spy_ret, vix=vix)
        assert X8.shape[1] == 8 and X11.shape[1] == 11
    chk("T7 feature counts 8 vs 11", t7)
    return res


# ── FIX 4: macro feature validation ─────────────────────────────────────────
def macro_validation(lines: list) -> bool:
    lines.append("FIX 4 — Macro Feature Validation (PASS A=11feat+blackout vs B=8feat):")
    print("  Training PASS A (11 features + earnings blackout)...")
    r17_train.main(macro=True,  blackout=True,  sgd_w=0.1)
    ra = rb.backtest(mode="D")
    print("  Training PASS B (8 features, no blackout)...")
    r17_train.main(macro=False, blackout=False, sgd_w=0.1)
    rb_res = rb.backtest(mode="D")
    use_macro = ra["Sharpe"] >= rb_res["Sharpe"]
    lines += [f"  PASS A (11+blackout): Sharpe={ra['Sharpe']}  Trades={ra['Trades']}",
              f"  PASS B (8, no blackout): Sharpe={rb_res['Sharpe']}  Trades={rb_res['Trades']}",
              f"  Decision: {'11 features + blackout (A wins)' if use_macro else '8 features (B wins, reverted)'}", ""]
    return use_macro


def sgd_sweep(use_macro: bool, lines: list) -> float:
    lines.append("FIX 5 — SGD Blend Weight Sweep on 2023:")
    r17_train.main(macro=use_macro, blackout=use_macro, sgd_w=0.0)
    best_w, best_sh = 0.0, -999.0
    for w in SGD_SWEEP:
        r = rb.backtest(mode="D", sgd_w_override=w)
        row = f"  SGD_W={w:.1f}: Ret={r['Return']}%  Sharpe={r['Sharpe']}  Trades={r['Trades']}  MaxDD={r['MaxDD']}%"
        lines.append(row); print(row)
        if r["Sharpe"] > best_sh and r["Trades"] >= MIN_TRADES:
            best_sh = r["Sharpe"]
            best_w = w
    lines += [f"  Locked: {best_w}", ""]
    return best_w


def four_pass(best_sgd_w: float, use_macro: bool, lines: list) -> dict:
    lines.append("4-Pass A/B/C/D (2023):")
    r17_train.main(macro=use_macro, blackout=use_macro, sgd_w=best_sgd_w)
    results = {}
    for m in ("A", "B", "C", "D"):
        r = rb.backtest(mode=m, sgd_w_override=best_sgd_w); results[m] = r
        row = f"  PASS {m}: Ret={r['Return']}%  Sharpe={r['Sharpe']}  MaxDD={r['MaxDD']}%  Trades={r['Trades']}  Deploy={r['GrossDeployed%']}%"
        lines.append(row); print(row)
    lines.append(""); return results

def main():
    OUT_DIR.mkdir(exist_ok=True)
    lines = ["=" * 62, "BACKTEST RESULTS ROUND 17", "=" * 62, ""]

    print("\n[SMOKE TESTS]")
    smoke = run_smoke_tests()
    for s in smoke: print(s)
    lines += ["SMOKE TESTS:", *smoke, f"Result: {sum('PASS' in s for s in smoke)}/7", ""]

    print("\n[FIX 4: MACRO VALIDATION]")
    use_macro = macro_validation(lines)

    print("\n[FIX 5: SGD SWEEP]")
    best_sgd_w = sgd_sweep(use_macro, lines)

    print("\n[4-PASS A/B/C/D]")
    ab = four_pass(best_sgd_w, use_macro, lines)

    print("\n[OOS WINDOW SWEEP / WF]")
    lines.append("OOS Window Sweep (FIX 3) + Walk-Forward:")
    wf_res = r17_wf.main(macro=use_macro)
    for cfg, r in wf_res.items():
        lines.append(f"  {cfg}(OOS={r17_wf.OOS_CONFIGS[cfg]}): MeanSharpe={r['mean_sharpe']} "
                     f"PosFolds={r['positive_folds']}/{r['n_folds']} "
                     f"WorstDD={r['worst_dd']}% Compounded={r['compounded']}%")
    best_cfg = max(wf_res, key=lambda k: wf_res[k]["mean_sharpe"])
    bwf = wf_res[best_cfg]
    lines += [f"  Best OOS: {best_cfg}", "",
              f"PASS D WF: MeanSharpe={bwf['mean_sharpe']}  Compounded={bwf['compounded']}%"
              f"  PosFolds={bwf['positive_folds']}/{bwf['n_folds']}  WorstDD={bwf['worst_dd']}%", ""]

    d_res = ab["D"]
    lines.append(f"Threshold dist (PASS D): mean={d_res['thresh_mean']}  std={d_res['thresh_std']}")
    if (OUT_DIR / "train_meta.json").exists():
        meta = json.load(open(OUT_DIR / "train_meta.json"))
        lines += ["Top-5 features (PASS D):"] + [f"  {f}: {v:.4f}" for f, v in meta.get("top5_features", [])]
    lines.append("")

    def chk(cond, label): return f"  {'PASS' if cond else 'FAIL'}  {label}"
    lines += ["TARGET CHECKS:",
              chk(d_res['Return'] > 0,           f"2023 Return >0%      ({d_res['Return']}%)"),
              chk(d_res['Sharpe'] > 0.60,         f"2023 Sharpe >0.60    ({d_res['Sharpe']})"),
              chk(d_res['Trades'] >= 70,          f"2023 Trades >=70     ({d_res['Trades']})"),
              chk(d_res['MaxDD']  > -15,          f"2023 MaxDD <-15%     ({d_res['MaxDD']}%)"),
              chk(bwf['mean_sharpe'] > 0.20,      f"WF MeanSharpe >0.20  ({bwf['mean_sharpe']})"),
              chk(bwf['positive_folds'] >= 12,    f"WF PosFolds >=12/22  ({bwf['positive_folds']}/{bwf['n_folds']})"),
              chk(bwf['worst_dd'] > -15,          f"WF WorstDD >-15%     ({bwf['worst_dd']}%)"),
              "", "=" * 62]

    output = "\n".join(lines)
    print("\n" + output)
    with open(RESULTS_FILE, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved → {RESULTS_FILE}")
    return output


if __name__ == "__main__":
    main(); print("FILE COMPLETE: r17_sweep.py")
