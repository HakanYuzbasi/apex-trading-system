"""
scripts/r15_backtest.py — Round 15 four-pass A/B on 2023.

Passes build cumulatively to isolate the effect of each fix:

  PASS A  R14 PASS D baseline
          (LSTM on, RISK_OFF shorts, floor=0.30, online SGD — all R14 fixes)
          No ML model pkl files on disk → adapter falls back to momentum
          z-score.  DYN_THRESH_CONF_PROP=false (old formula).

  PASS B  + FIX 1 conf-proportional threshold
          Activates APEX_DYN_THRESH_CONF_PROP=true. Threshold now responds
          continuously to confidence.  Still momentum z-score signals.

  PASS C  + FIX 2+3 R15 basic GBMs (11-feature GradientBoosting)
          Points APEX_ML_MODEL_PATH_* at the r15_*.pkl files trained by
          r15_train.py.  Online learning disabled (GBMs have no partial_fit).
          DYN_THRESH_CONF_PROP=true.

  PASS D  + FIX 4 calibrated GBMs (CalibratedClassifierCV isotonic)
          Points APEX_ML_MODEL_PATH_* at the r15_cal_*.pkl files.
          Better probability estimates → threshold responds as intended.
          All four fixes active.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.real_signal_adapter import RealSignalAdapter
from config import ApexConfig
from core.logging_config import setup_logging

from scripts.round12_real_data_report import (
    R12_SYMBOLS, R12_FULL_START, R12_FULL_END, R12_BT_START, R12_BT_END,
    fetch_panel_chunked, wf_edge_stats, _reload_apex_config,
)
from scripts.round14_real_data_report import _apply_round14_env_overrides

# ─────────────────────────────────────────────────────────────────────────────
# Model paths (set by r15_train.py)
# ─────────────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parents[1]
_DIR = _REPO / "models" / "saved_advanced"

R15_BASIC_TRENDING  = str(_DIR / "r15_trending.pkl")
R15_BASIC_MEAN_REV  = str(_DIR / "r15_mean_rev.pkl")
R15_BASIC_VOLATILE  = str(_DIR / "r15_volatile.pkl")
R15_CAL_TRENDING    = str(_DIR / "r15_cal_trending.pkl")
R15_CAL_MEAN_REV    = str(_DIR / "r15_cal_mean_rev.pkl")
R15_CAL_VOLATILE    = str(_DIR / "r15_cal_volatile.pkl")


def _check_models(paths: List[str]) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"R15 models missing — run r15_train.py first:\n" +
            "\n".join(f"  {p}" for p in missing)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pass configs — additive from PASS A baseline
# ─────────────────────────────────────────────────────────────────────────────

# R14 PASS D base (all Round 14 fixes, no R15 fixes, no model paths)
PASS_A_ENV: Dict[str, str] = {
    "APEX_KELLY_ENABLED":                "true",
    "APEX_KELLY_MIN_SAMPLES":            "10",
    "APEX_MAX_POSITION_PCT":             "0.35",
    "APEX_PARTIAL_EXIT_ENABLED":         "true",
    "APEX_PARTIAL_EXIT_R_STOP_MULT":     "3.5",
    "APEX_CORR_THRESHOLD":               "0.85",
    "APEX_MAX_CONCURRENT_POSITIONS":     "10",
    "APEX_MACRO_REGIME_ENABLED":         "true",
    "APEX_KELLY_WARM_START_ENABLED":     "true",
    "APEX_SPREAD_BPS_DEFAULT":           "3.0",
    "APEX_SPREAD_BPS_ETF":               "1.0",
    "APEX_MARKET_IMPACT_MULT":           "0.1",
    "APEX_PORTFOLIO_GROSS_LEVERAGE_MAX": "1.5",
    "APEX_SHORT_MARGIN_PCT":             "0.50",
    "APEX_LSTM_ENABLED":                 "true",
    "APEX_SHORT_ONLY_IN_RISK_OFF":       "true",
    "APEX_DYNAMIC_THRESHOLD_FLOOR":      "0.30",
    "APEX_ONLINE_LEARNING_ENABLED":      "true",
    "APEX_DYN_THRESH_CONF_PROP":         "false",
    "APEX_ML_MODEL_PATH_TRENDING":       "",
    "APEX_ML_MODEL_PATH_MEAN_REV":       "",
    "APEX_ML_MODEL_PATH_VOLATILE":       "",
}

PASS_B_ENV: Dict[str, str] = dict(PASS_A_ENV)
PASS_B_ENV.update({
    "APEX_DYN_THRESH_CONF_PROP": "true",          # FIX 1
})

PASS_C_ENV: Dict[str, str] = dict(PASS_B_ENV)
PASS_C_ENV.update({
    "APEX_ML_MODEL_PATH_TRENDING": R15_BASIC_TRENDING,   # FIX 2+3
    "APEX_ML_MODEL_PATH_MEAN_REV": R15_BASIC_MEAN_REV,
    "APEX_ML_MODEL_PATH_VOLATILE": R15_BASIC_VOLATILE,
    "APEX_ONLINE_LEARNING_ENABLED": "false",              # GBMs no partial_fit
})

PASS_D_ENV: Dict[str, str] = dict(PASS_C_ENV)
PASS_D_ENV.update({
    "APEX_ML_MODEL_PATH_TRENDING": R15_CAL_TRENDING,     # FIX 4
    "APEX_ML_MODEL_PATH_MEAN_REV": R15_CAL_MEAN_REV,
    "APEX_ML_MODEL_PATH_VOLATILE": R15_CAL_VOLATILE,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.3f}%"


def _fmt_num(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.4f}"


def _apply_r15_env(env: Dict[str, str]) -> None:
    """Patch ApexConfig with R15-specific fields not covered by round12/14 reloaders."""
    raw = env.get("APEX_DYN_THRESH_CONF_PROP", "false")
    ApexConfig.DYN_THRESH_CONF_PROP = raw.lower() == "true"
    for attr, key in [
        ("ML_MODEL_PATH_TRENDING", "APEX_ML_MODEL_PATH_TRENDING"),
        ("ML_MODEL_PATH_MEAN_REV",  "APEX_ML_MODEL_PATH_MEAN_REV"),
        ("ML_MODEL_PATH_VOLATILE",  "APEX_ML_MODEL_PATH_VOLATILE"),
    ]:
        setattr(ApexConfig, attr, env.get(key, ""))


def _run_pass(
    label: str,
    panel: Dict[str, Any],
    env: Dict[str, str],
    max_positions: int = 10,
) -> Dict[str, Any]:
    prev_env: Dict[str, str] = {}
    for k, v in env.items():
        prev_env[k] = os.environ.get(k, "")
        os.environ[k] = v
    _reload_apex_config(env)
    _apply_round14_env_overrides(env)
    _apply_r15_env(env)
    try:
        adapter = RealSignalAdapter()
        bt = AdvancedBacktester(initial_capital=100_000)
        adapter.attach_panel(panel)
        res = bt.run_backtest(
            data=panel,
            signal_generator=adapter,
            start_date=R12_BT_START,
            end_date=R12_BT_END,
            position_size_usd=5_000,
            max_positions=max_positions,
        )
        trades: List[Dict[str, Any]] = bt.trades

        print("=" * 72)
        print(f" {label}")
        print("=" * 72)
        rows = [
            ("Total Return",              _fmt_pct(res.get("total_return"))),
            ("Sharpe Ratio",              _fmt_num(res.get("sharpe_ratio"))),
            ("Max Drawdown",              _fmt_pct(res.get("max_drawdown"))),
            ("Win Rate",                  _fmt_pct(res.get("win_rate"))),
            ("Profit Factor",             _fmt_num(res.get("profit_factor"))),
            ("Total Trades",              f"{int(res.get('total_trades') or 0)}"),
            ("Avg Capital Deployed (net)", _fmt_pct(res.get("avg_capital_deployed"))),
            ("Avg Gross Deployed",        _fmt_pct(res.get("avg_gross_deployed"))),
            ("RISK_ON fraction",          _fmt_num(res.get("risk_on_fraction"))),
        ]
        for lbl, val in rows:
            print(f"  {lbl:<30}: {val}")

        block = res.get("entry_block_reasons") or {}
        if block:
            print()
            print("  Entry-block breakdown")
            print("  " + "-" * 64)
            for k, v in sorted(block.items(), key=lambda kv: -int(kv[1])):
                print(f"    {k:<28}: {int(v)}")

        dyn = res.get("dynamic_threshold_stats") or {}
        if dyn.get("n", 0) > 0:
            print()
            print("  Dyn-threshold distribution (post-floor)")
            print("  " + "-" * 64)
            print(f"    n={int(dyn['n'])}  mean={_fmt_num(dyn.get('mean'))}  "
                  f"std={_fmt_num(dyn.get('std'))}  "
                  f"p10={_fmt_num(dyn.get('p10'))}  p90={_fmt_num(dyn.get('p90'))}")

        models_loaded = bool(getattr(ApexConfig, "ML_MODEL_PATH_TRENDING", ""))
        conf_prop = bool(getattr(ApexConfig, "DYN_THRESH_CONF_PROP", False))
        print()
        print(f"  DYN_THRESH_CONF_PROP  : {conf_prop}")
        print(f"  ML models loaded      : {models_loaded}")
        print()
        return res
    finally:
        for k, v in prev_env.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})
        _apply_round14_env_overrides({})
        _apply_r15_env({})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    _check_models([
        R15_BASIC_TRENDING, R15_BASIC_MEAN_REV, R15_BASIC_VOLATILE,
        R15_CAL_TRENDING,   R15_CAL_MEAN_REV,   R15_CAL_VOLATILE,
    ])

    print("=" * 72)
    print(" Round 15 — r15_backtest.py  (4-pass A/B on 2023)")
    print("=" * 72)
    print(f" Universe  : {list(R12_SYMBOLS)}")
    print(f" BT window : {R12_BT_START} → {R12_BT_END}")
    print(f" FIX 1     : DYN_THRESH_CONF_PROP=true (conf-proportional threshold)")
    print(f" FIX 2+3   : R15 basic GBMs (11-feature) at r15_*.pkl")
    print(f" FIX 4     : Calibrated GBMs at r15_cal_*.pkl")
    print()

    print(" Fetching 10-symbol panel ...")
    panel = fetch_panel_chunked(R12_SYMBOLS, R12_FULL_START, R12_FULL_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    results: Dict[str, Dict[str, Any]] = {}
    results["A"] = _run_pass("PASS A — R14 PASS D baseline (no R15 fixes)",
                             panel, PASS_A_ENV)
    results["B"] = _run_pass("PASS B — +FIX 1 conf-proportional threshold",
                             panel, PASS_B_ENV)
    results["C"] = _run_pass("PASS C — +FIX 2+3 R15 basic GBMs (11 features)",
                             panel, PASS_C_ENV)
    results["D"] = _run_pass("PASS D — +FIX 4 calibrated GBMs (all fixes)",
                             panel, PASS_D_ENV)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(" Summary — 2023 backtest")
    print("=" * 72)
    hdr = f" {'':6} {'TotRet':>9} {'Sharpe':>8} {'MaxDD':>9} {'Win%':>7} {'PF':>7} {'Trades':>7}"
    print(hdr)
    print(" " + "-" * 68)
    for key, lbl in [("A", "PASS A"), ("B", "PASS B"), ("C", "PASS C"), ("D", "PASS D")]:
        r = results[key]
        def _p(x: Any) -> str:
            try:
                return f"{float(x)*100:>8.3f}%"
            except Exception:
                return f"{'N/A':>9}"
        def _n(x: Any) -> str:
            try:
                v = float(x)
                return f"{v:>8.3f}" if np.isfinite(v) else f"{'N/A':>8}"
            except Exception:
                return f"{'N/A':>8}"
        print(f" {lbl:6}  {_p(r.get('total_return'))} "
              f"{_n(r.get('sharpe_ratio'))} "
              f"{_p(r.get('max_drawdown'))} "
              f"{_p(r.get('win_rate'))} "
              f"{_n(r.get('profit_factor'))} "
              f"{int(r.get('total_trades') or 0):>7d}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
