"""Round 16: WF 2020-2024 (22 folds) + smoke tests + report → backtest_results_round16.txt."""
import datetime, json, os, re, subprocess, sys
from io import StringIO
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.real_signal_adapter import RealSignalAdapter
from config import ApexConfig
from core.logging_config import setup_logging
from scripts.r16_backtest import CONFIG_PATH, _apply_r16_env
from scripts.r16_data import apply_synth_fallback
from scripts.round12_real_data_report import (
    R12_SYMBOLS, R12_FULL_START, _reload_apex_config, fetch_panel_chunked,
)
from scripts.round14_real_data_report import _apply_round14_env_overrides

_REPO = Path(__file__).resolve().parents[1]
WF_START, WF_END = "2020-01-01", "2024-12-31"


def _activate(env: dict) -> None:
    for k, v in env.items():
        os.environ[k] = v
    _reload_apex_config(env)
    _apply_round14_env_overrides(env)
    _apply_r16_env(env)


def run_wf(panel: dict) -> dict:
    is_n  = int(ApexConfig.WF_IS_BARS)
    oos_n = int(ApexConfig.WF_OOS_BARS)
    step_n = int(ApexConfig.WF_STEP_BARS)
    dates = pd.date_range(WF_START, WF_END, freq="D")
    adapter = RealSignalAdapter()
    adapter.attach_panel(panel)
    bt = AdvancedBacktester(initial_capital=100_000)
    folds, idx = [], 0
    while idx + is_n + oos_n <= len(dates):
        oos_s = dates[idx + is_n]
        oos_e = dates[min(idx + is_n + oos_n - 1, len(dates) - 1)]
        m = bt.run_backtest(data=panel, signal_generator=adapter,
                            start_date=str(oos_s.date()), end_date=str(oos_e.date()),
                            position_size_usd=5_000, max_positions=10)
        f = {"oos_start": str(oos_s.date()), "oos_end": str(oos_e.date()),
             "sharpe": float(m.get("sharpe_ratio") or 0.0),
             "ret":    float(m.get("total_return")  or 0.0),
             "dd":     float(m.get("max_drawdown")   or 0.0),
             "win":    float(m.get("win_rate")        or 0.0),
             "pf":     float(m.get("profit_factor")  or 0.0),
             "trades": int(m.get("total_trades")     or 0)}
        folds.append(f)
        n = len(folds)
        if n % 5 == 0 or n == 1:
            print(f"  [fold {n:>2}] {oos_s.date()} → {oos_e.date()}"
                  f"  sharpe={f['sharpe']:+.3f}  ret={f['ret']*100:+.2f}%")
        idx += step_n
    sharpes = [f["sharpe"] for f in folds if np.isfinite(f["sharpe"])]
    comp = 1.0
    for f in folds:
        comp *= 1.0 + f["ret"]
    return {
        "folds": folds,
        "mean_sharpe":       float(np.mean(sharpes)) if sharpes else float("nan"),
        "compounded_return": comp - 1.0,
        "worst_dd":          min(f["dd"] for f in folds) if folds else 0.0,
        "positive_folds":    sum(1 for f in folds if f["ret"] > 0),
        "n_folds":           len(folds),
    }


def run_smoke() -> tuple:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_integration_smoke.py", "-v"],
                       capture_output=True, text=True, cwd=str(_REPO))
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else 0
    return passed, failed, out

def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)
    buf = StringIO()
    def w(line: str = "") -> None:
        print(line); buf.write(line + "\n")
    apply_synth_fallback()
    config = json.loads(CONFIG_PATH.read_text())
    winner, threshold = config["winner"], config["threshold"]
    _activate(config["env"])
    w("=" * 75)
    w(f" Apex Trading System — Round 16 Report")
    w(f"   generated : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    w(f"   branch    : claude/fix-gbm-calibration-YOxJX")
    w("=" * 75)
    w(f" FIX 1  calibration={winner}  FIX 2  threshold(P40)={threshold:.4f}")
    w(f"        ML_MODEL_PATH_TRENDING={ApexConfig.ML_MODEL_PATH_TRENDING}")
    w()
    w(" Fetching 10-symbol panel 2020-01-01 → 2024-12-31 ...")
    panel = fetch_panel_chunked(R12_SYMBOLS, R12_FULL_START, "2025-01-01")
    for sym, df in panel.items():
        w(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    w()
    w(f" Walk-forward 2020-2024  IS={ApexConfig.WF_IS_BARS} OOS={ApexConfig.WF_OOS_BARS}"
      f" step={ApexConfig.WF_STEP_BARS}")
    w("-" * 75)
    agg   = run_wf(panel)
    folds = agg["folds"]
    w()
    w(f" {'#':>3}  {'OOS start':<12} {'OOS end':<12} {'Sharpe':>8} {'TotRet':>9}"
      f" {'MaxDD':>9} {'Win%':>7} {'PF':>7} {'Trades':>7}")
    w(" " + "-" * 70)
    for i, f in enumerate(folds, 1):
        pf_s = f"{f['pf']:>7.3f}" if np.isfinite(f["pf"]) and f["pf"] < 999 else "    inf"
        w(f" {i:>3}  {f['oos_start']:<12} {f['oos_end']:<12}"
          f" {f['sharpe']:>8.4f} {f['ret']*100:>8.3f}% {f['dd']*100:>8.3f}%"
          f" {f['win']*100:>6.2f}% {pf_s} {f['trades']:>7d}")
    w()
    ms, cr, wd, pf_n, nf = (agg["mean_sharpe"], agg["compounded_return"],
                             agg["worst_dd"], agg["positive_folds"], agg["n_folds"])
    w(" Walk-forward aggregate"); w(" " + "-" * 70)
    w(f" Folds run              : {nf}")
    w(f" Mean Sharpe            : {ms:.6f}")
    w(f" Compounded Return      : {cr*100:.4f}%")
    w(f" Worst Fold Drawdown    : {wd*100:.4f}%")
    w(f" Positive Folds         : {pf_n} / {nf}")
    w(f" Negative Folds         : {nf - pf_n} / {nf}")
    w()
    w(" Targets"); w(" " + "-" * 70)
    for desc, hit, val in [("Mean Sharpe > 0.15",      ms > 0.15,  f"{ms:.4f}"),
                            ("Positive folds >= 11/22", pf_n >= 11, f"{pf_n}/{nf}"),
                            ("Worst DD < -15%",         wd > -0.15, f"{wd*100:.2f}%")]:
        w(f"   {desc:<32} {'HIT' if hit else 'MISS':4}   actual={val}")
    w()
    w(" Smoke tests (7/7)"); w("-" * 75)
    passed, failed, smoke_raw = run_smoke()
    w(f" Result: {passed} passed, {failed} failed")
    for line in smoke_raw.splitlines():
        if any(k in line for k in ("PASSED", "FAILED", "ERROR", "passed", "failed")):
            w(f"   {line.strip()}")
    w()
    out_path = _REPO / "backtest_results_round16.txt"
    out_path.write_text(buf.getvalue())
    print(f"\n Saved → {out_path}")
    print("FILE COMPLETE: r16_wf.py")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
