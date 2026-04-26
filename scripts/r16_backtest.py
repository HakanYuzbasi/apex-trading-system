"""Round 16: 2-pass A/B on 2023 — sigmoid vs raw GBM. Saves r16_best_config.json."""
import json, os, pickle, sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.real_signal_adapter import ML_FEATURE_NAMES, RealSignalAdapter, compute_ml_features
from config import ApexConfig
from core.logging_config import setup_logging
from models.ml_validator import leakage_check
from scripts.r15_backtest import _apply_r15_env
from scripts.r16_data import apply_synth_fallback
from scripts.r15_train import FORWARD_HORIZON, build_frame
from scripts.round12_real_data_report import (
    R12_BT_END, R12_BT_START, R12_FULL_END, R12_FULL_START, R12_SYMBOLS,
    _reload_apex_config, fetch_panel_chunked,
)
from scripts.round14_real_data_report import _apply_round14_env_overrides

_REPO = Path(__file__).resolve().parents[1]
_DIR  = _REPO / "models" / "saved_advanced"
CONFIG_PATH = _REPO / "r16_best_config.json"
FEAT_COLS = list(ML_FEATURE_NAMES)
_BASE: Dict[str, str] = {
    "APEX_KELLY_ENABLED": "true", "APEX_KELLY_MIN_SAMPLES": "10",
    "APEX_MAX_POSITION_PCT": "0.35", "APEX_PARTIAL_EXIT_ENABLED": "true",
    "APEX_PARTIAL_EXIT_R_STOP_MULT": "3.5", "APEX_CORR_THRESHOLD": "0.85",
    "APEX_MAX_CONCURRENT_POSITIONS": "10", "APEX_MACRO_REGIME_ENABLED": "true",
    "APEX_KELLY_WARM_START_ENABLED": "true", "APEX_SPREAD_BPS_DEFAULT": "3.0",
    "APEX_SPREAD_BPS_ETF": "1.0", "APEX_MARKET_IMPACT_MULT": "0.1",
    "APEX_PORTFOLIO_GROSS_LEVERAGE_MAX": "1.5", "APEX_SHORT_MARGIN_PCT": "0.50",
    "APEX_LSTM_ENABLED": "true", "APEX_SHORT_ONLY_IN_RISK_OFF": "true",
    "APEX_DYNAMIC_THRESHOLD_FLOOR": "0.30", "APEX_ONLINE_LEARNING_ENABLED": "false",
    "APEX_DYN_THRESH_CONF_PROP": "true",
}
def _env_for(pfx: str) -> Dict[str, str]:
    e = dict(_BASE)
    e["APEX_ML_MODEL_PATH_TRENDING"] = str(_DIR / f"{pfx}_trending.pkl")
    e["APEX_ML_MODEL_PATH_MEAN_REV"]  = str(_DIR / f"{pfx}_mean_rev.pkl")
    e["APEX_ML_MODEL_PATH_VOLATILE"]  = str(_DIR / f"{pfx}_volatile.pkl")
    return e
PASS_A_ENV = _env_for("r16_sig")
PASS_B_ENV = _env_for("r16_raw")
def _apply_r16_env(env: Dict[str, str]) -> None:
    _apply_r15_env(env)
    raw = env.get("APEX_ML_CONFIDENCE_THRESHOLD", "")
    if raw:
        ApexConfig.ML_CONFIDENCE_THRESHOLD = float(raw)
def _compute_p40(trending_pkl: str, panel: dict) -> float:
    with open(trending_pkl, "rb") as fh:
        payload = pickle.load(fh)
    model = payload["model"]
    sigs: List[float] = []
    for ohlcv in panel.values():
        feats = compute_ml_features(ohlcv)
        if feats.empty:
            continue
        X = feats[FEAT_COLS].to_numpy()
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [-1, 1]))
        up_idx = classes.index(1) if 1 in classes else -1
        sigs.extend(float(v) for v in np.abs((proba[:, up_idx] - 0.5) * 2.0))
    return float(np.percentile(sigs, 40)) if sigs else 0.13
def _run_pass(label: str, env: Dict[str, str], panel: dict) -> Dict[str, Any]:
    prev = {k: os.environ.get(k, "") for k in env}
    for k, v in env.items():
        os.environ[k] = v
    _reload_apex_config(env); _apply_round14_env_overrides(env); _apply_r16_env(env)
    try:
        adapter = RealSignalAdapter()
        bt = AdvancedBacktester(initial_capital=100_000)
        adapter.attach_panel(panel)
        res = bt.run_backtest(data=panel, signal_generator=adapter,
                              start_date=R12_BT_START, end_date=R12_BT_END,
                              position_size_usd=5_000, max_positions=10)
        sh  = float(res.get("sharpe_ratio") or 0.0)
        ret = float(res.get("total_return") or 0.0)
        dd  = float(res.get("max_drawdown") or 0.0)
        t   = int(res.get("total_trades") or 0)
        gd  = float(res.get("avg_gross_deployed") or 0.0)
        print(f" {label}")
        print(f"   Return={ret*100:+.3f}%  Sharpe={sh:.4f}  MaxDD={dd*100:.3f}%"
              f"  Trades={t}  GrossDep={gd*100:.1f}%")
        return res
    finally:
        for k, v in prev.items():
            if v: os.environ[k] = v
            else: os.environ.pop(k, None)
        _reload_apex_config({}); _apply_round14_env_overrides({}); _apply_r16_env({})
def main() -> int:
    apply_synth_fallback()
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)
    print("=" * 72)
    print(" Round 16 — r16_backtest.py  (2-pass A/B on 2023)")
    print("=" * 72)
    print(" Fetching panel ...")
    panel = fetch_panel_chunked(R12_SYMBOLS, R12_FULL_START, R12_FULL_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars")
    print("\n FIX 2 — leakage_check(raise_on_fail=True) ...")
    leakage_check(build_frame(panel)[FEAT_COLS + ["label"]], label_col="label",
                  feature_cols=FEAT_COLS, max_future_shift=FORWARD_HORIZON,
                  leak_corr_threshold=0.98, raise_on_fail=True)
    print("   PASSED\n")
    res_a = _run_pass("PASS A — sigmoid calibration",      PASS_A_ENV, panel)
    print()
    res_b = _run_pass("PASS B — raw GBM (no calibration)", PASS_B_ENV, panel)
    cands = [("sigmoid", PASS_A_ENV, res_a, str(_DIR / "r16_sig_trending.pkl")),
             ("raw",     PASS_B_ENV, res_b, str(_DIR / "r16_raw_trending.pkl"))]
    winner = None
    for name, env, res, pkl_path in cands:
        t  = int(res.get("total_trades") or 0)
        sh = float(res.get("sharpe_ratio") or 0.0)
        if t >= 60 and sh > 0.50:
            winner = (name, env, pkl_path)
            print(f"\n Winner: '{name}'  Trades={t} >= 60  Sharpe={sh:.4f} > 0.50")
            break
    if winner is None:
        def _score(c):
            sh = float(c[2].get("sharpe_ratio") or 0.0)
            t  = int(c[2].get("total_trades") or 0)
            return (t > 0, sh if np.isfinite(sh) else -99.0, t)
        best = max(cands, key=_score)
        winner = (best[0], best[1], best[3])
        sh_b = float(best[2].get("sharpe_ratio") or 0.0)
        t_b  = int(best[2].get("total_trades") or 0)
        print(f"\n No target met — best Sharpe: '{best[0]}'  Sharpe={sh_b:.4f}  Trades={t_b}")
    name, env, pkl_path = winner
    threshold = _compute_p40(pkl_path, panel)
    print(f" P40 ML_CONFIDENCE_THRESHOLD ({name}): {threshold:.4f}")
    env_saved = dict(env)
    env_saved["APEX_ML_CONFIDENCE_THRESHOLD"] = str(round(threshold, 6))
    CONFIG_PATH.write_text(json.dumps({"winner": name, "threshold": threshold,
                                       "env": env_saved}, indent=2))
    print(f" Config saved → {CONFIG_PATH}")
    print("\n" + "=" * 72)
    print(" 2023 summary  [Return | Sharpe | MaxDD | Trades | GrossDep%]")
    print("=" * 72)
    for tag, res in [("SIGMOID", res_a), ("RAW GBM", res_b)]:
        ret = float(res.get("total_return") or 0.0)
        sh  = float(res.get("sharpe_ratio") or 0.0)
        dd  = float(res.get("max_drawdown") or 0.0)
        t   = int(res.get("total_trades") or 0)
        gd  = float(res.get("avg_gross_deployed") or 0.0)
        print(f" {tag:<8}  {ret*100:+.3f}%  {sh:.4f}  {dd*100:.3f}%  {t}  {gd*100:.1f}%")
    print("\nFILE COMPLETE: r16_backtest.py")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
