"""
scripts/round13_real_data_report.py

Round 13 driver — three-pass A/B against the Round 12 PASS-4 baseline.

  PASS A : R12 baseline reproduction
           (no leverage cap, GBM-only, exec costs ON, macro ON,
           warm-start ON, partials ON, corr 0.85, kelly recalibrated)
  PASS B : FIX 1+2 only
           (gross-leverage cap 1.5x + Reg-T short margin + complete
           entry-block tagging, LSTM disabled)
  PASS C : FIX 1+2+3
           (adds the per-symbol LSTM ensemble at 0.6 weight)

Walk-forward 2020-2024 runs once with FIX 1+2+3 enabled to validate
the leverage-cap effect on per-fold drawdowns.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.real_signal_adapter import RealSignalAdapter
from config import ApexConfig
from core.logging_config import setup_logging

# Re-use the Round 12 fetcher + warm-start helpers.
from scripts.round12_real_data_report import (
    R12_SYMBOLS, R12_FULL_START, R12_FULL_END, R12_BT_START, R12_BT_END,
    fetch_panel_chunked, wf_edge_stats, _reload_apex_config,
)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "NaN"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NaN"
    return f"{v * 100:.4f}%"


def _fmt_num(x: Any) -> str:
    if x is None:
        return "NaN"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.6f}"


def _print_pass(label: str, res: Dict[str, Any], trades: List[Dict[str, Any]]) -> None:
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
        ("Avg Winner R",              _fmt_num(res.get("avg_winner_r"))),
        ("Avg Capital Deployed % (net)", _fmt_pct(res.get("avg_capital_deployed"))),
        ("Avg Gross Deployed %",      _fmt_pct(res.get("avg_gross_deployed"))),
        ("Avg Portfolio Correlation", _fmt_num(res.get("avg_portfolio_correlation"))),
        ("Avg Total Cost bps",        _fmt_num(res.get("avg_total_cost_bps"))),
        ("RISK_ON fraction",          _fmt_num(res.get("risk_on_fraction"))),
    ]
    for lab, val in rows:
        print(f" {lab:<32}: {val}")
    counts: Dict[str, int] = {}
    for t in trades:
        sym = t.get("symbol", "?")
        counts[sym] = counts.get(sym, 0) + 1
    if counts:
        print()
        print(" Trades per symbol")
        print(" " + "-" * 68)
        for sym in sorted(counts.keys()):
            print(f"   {sym:<10}: {counts[sym]}")
    block_reasons = res.get("entry_block_reasons") or {}
    if block_reasons:
        print()
        print(" Entry-block breakdown")
        print(" " + "-" * 68)
        for k, v in sorted(block_reasons.items(), key=lambda kv: -int(kv[1])):
            print(f"   {k:<28}: {int(v)}")
    print()


def _run_pass(
    label: str,
    panel: Dict[str, pd.DataFrame],
    env_overrides: Dict[str, str],
    *,
    max_positions: int,
    warm_start: Optional[Tuple[float, float, float, int]] = None,
) -> Dict[str, Any]:
    previous: Dict[str, str] = {}
    for k, v in env_overrides.items():
        previous[k] = os.environ.get(k, "")
        os.environ[k] = v
    _reload_apex_config(env_overrides)
    # Reload Round 13 fields directly (the R12 helper doesn't know them).
    ApexConfig.PORTFOLIO_GROSS_LEVERAGE_MAX = float(
        os.environ.get("APEX_PORTFOLIO_GROSS_LEVERAGE_MAX", "1.5")
    )
    ApexConfig.SHORT_MARGIN_PCT = float(
        os.environ.get("APEX_SHORT_MARGIN_PCT", "0.50")
    )
    ApexConfig.LSTM_ENABLED = (
        os.environ.get("APEX_LSTM_ENABLED", "true").lower() == "true"
    )
    ApexConfig.LSTM_ENSEMBLE_WEIGHT = float(
        os.environ.get("APEX_LSTM_ENSEMBLE_WEIGHT", "0.6")
    )
    try:
        adapter = RealSignalAdapter()
        if warm_start is not None and bool(getattr(ApexConfig, "KELLY_WARM_START_ENABLED", False)):
            wr, aw, al, _nf = warm_start
            adapter._aggregator.warm_start_source(
                "primary",
                win_rate=wr, avg_win=aw, avg_loss=al,
                n_samples=int(getattr(ApexConfig, "KELLY_MIN_SAMPLES", 10)),
            )
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
        _print_pass(label, res, bt.trades)
        rep = adapter.source_hit_report()
        print(" Adapter source hits")
        print(" " + "-" * 68)
        for k in ("ml", "pattern", "orb", "momentum_fallback",
                  "_entries_fired", "_ml_active",
                  "_lstm_active", "_lstm_hits"):
            print(f"   {k:<28}: {int(rep.get(k, 0))}")
        print()
        return res
    finally:
        for k, v in previous.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})


def _run_wf(
    panel: Dict[str, pd.DataFrame],
    env_overrides: Dict[str, str],
    label: str,
) -> Dict[str, Any]:
    previous: Dict[str, str] = {}
    for k, v in env_overrides.items():
        previous[k] = os.environ.get(k, "")
        os.environ[k] = v
    _reload_apex_config(env_overrides)
    ApexConfig.PORTFOLIO_GROSS_LEVERAGE_MAX = float(
        os.environ.get("APEX_PORTFOLIO_GROSS_LEVERAGE_MAX", "1.5")
    )
    ApexConfig.SHORT_MARGIN_PCT = float(
        os.environ.get("APEX_SHORT_MARGIN_PCT", "0.50")
    )
    ApexConfig.LSTM_ENABLED = (
        os.environ.get("APEX_LSTM_ENABLED", "true").lower() == "true"
    )
    ApexConfig.LSTM_ENSEMBLE_WEIGHT = float(
        os.environ.get("APEX_LSTM_ENSEMBLE_WEIGHT", "0.6")
    )
    try:
        is_n = int(ApexConfig.WF_IS_BARS)
        oos_n = int(ApexConfig.WF_OOS_BARS)
        step_n = int(ApexConfig.WF_STEP_BARS)
        print("=" * 72)
        print(f" {label}")
        print("=" * 72)
        print(f"   WF_IS_BARS  : {is_n}")
        print(f"   WF_OOS_BARS : {oos_n}")
        print(f"   WF_STEP_BARS: {step_n}")
        print(f"   symbols     : {list(panel.keys())}")
        print()
        adapter = RealSignalAdapter()
        bt = AdvancedBacktester(initial_capital=100_000)
        adapter.attach_panel(panel)
        wf_out = bt.run_walk_forward(
            data=panel,
            signal_generator=adapter,
            start_date=R12_FULL_START,
            end_date="2024-12-31",
            position_size_usd=5_000,
            max_positions=10,
        )
        folds = list(wf_out.get("folds", []))
        agg = dict(wf_out.get("aggregate", {}))
        print(f" Folds produced : {len(folds)}")
        print()
        print(" Per-fold OOS metrics")
        print(" " + "-" * 68)
        print(f" {'#':>3}  {'OOS start':<12} {'OOS end':<12} "
              f"{'Sharpe':>10} {'TotRet':>9} {'MaxDD':>9} "
              f"{'Win%':>7} {'PF':>7} {'Trades':>7}")
        print(" " + "-" * 68)
        for i, f in enumerate(folds, 1):
            sh = float(f.get("sharpe_ratio", float("nan")))
            sh_txt = "     NaN" if not np.isfinite(sh) else f"{sh:>10.4f}"
            print(
                f" {i:>3}  {f.get('oos_start',''):<12} {f.get('oos_end',''):<12} "
                f"{sh_txt} "
                f"{float(f.get('total_return', 0.0)) * 100:>8.3f}% "
                f"{float(f.get('max_drawdown', 0.0)) * 100:>8.3f}% "
                f"{float(f.get('win_rate', 0.0)) * 100:>6.2f}% "
                f"{float(f.get('profit_factor', 0.0)):>7.3f} "
                f"{int(f.get('total_trades', 0)):>7d}"
            )
        print()
        print(" Walk-forward aggregate")
        print(" " + "-" * 68)
        print(f" Folds run               : {int(agg.get('folds_run', 0))}")
        print(f" Mean Sharpe             : {_fmt_num(agg.get('mean_sharpe'))}")
        print(f" Median Sharpe           : {_fmt_num(agg.get('median_sharpe'))}")
        print(f" Compounded Return       : {_fmt_pct(agg.get('compounded_return'))}")
        print(f" Worst Fold Drawdown     : {_fmt_pct(agg.get('worst_fold_drawdown'))}")
        print(f" Positive Folds          : {int(agg.get('positive_folds', 0))} / {len(folds)}")
        print(f" Negative Folds          : {int(agg.get('negative_folds', 0))} / {len(folds)}")
        print(f" Insufficient-data Folds : {int(agg.get('insufficient_data_folds', 0))} / {len(folds)}")
        print()
        return wf_out
    finally:
        for k, v in previous.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})


PASS_A_ENV: Dict[str, str] = {
    "APEX_KELLY_ENABLED":               "true",
    "APEX_KELLY_MIN_SAMPLES":           "10",
    "APEX_MAX_POSITION_PCT":            "0.35",
    "APEX_PARTIAL_EXIT_ENABLED":        "true",
    "APEX_PARTIAL_EXIT_R_STOP_MULT":    "3.5",
    "APEX_CORR_THRESHOLD":              "0.85",
    "APEX_MAX_CONCURRENT_POSITIONS":    "10",
    "APEX_MACRO_REGIME_ENABLED":        "true",
    "APEX_KELLY_WARM_START_ENABLED":    "true",
    "APEX_SPREAD_BPS_DEFAULT":          "3.0",
    "APEX_SPREAD_BPS_ETF":              "1.0",
    "APEX_MARKET_IMPACT_MULT":          "0.1",
    "APEX_PORTFOLIO_GROSS_LEVERAGE_MAX": "100.0",  # effectively disabled
    "APEX_SHORT_MARGIN_PCT":            "0.0",     # legacy short-as-credit
    "APEX_LSTM_ENABLED":                "false",
    "APEX_LSTM_ENSEMBLE_WEIGHT":        "0.0",
}

PASS_B_ENV: Dict[str, str] = dict(PASS_A_ENV)
PASS_B_ENV.update({
    "APEX_PORTFOLIO_GROSS_LEVERAGE_MAX": "1.5",
    "APEX_SHORT_MARGIN_PCT":            "0.50",
    "APEX_LSTM_ENABLED":                "false",
    "APEX_LSTM_ENSEMBLE_WEIGHT":        "0.0",
})

PASS_C_ENV: Dict[str, str] = dict(PASS_B_ENV)
PASS_C_ENV.update({
    "APEX_LSTM_ENABLED":         "true",
    "APEX_LSTM_ENSEMBLE_WEIGHT": "0.6",
})


def main(*, run_bt: bool = True, run_wf: bool = True) -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 13 — leverage cap + complete block tagging + LSTM ensemble")
    print("=" * 72)
    print(f" PORTFOLIO_GROSS_LEVERAGE_MAX : {ApexConfig.PORTFOLIO_GROSS_LEVERAGE_MAX}")
    print(f" SHORT_MARGIN_PCT             : {ApexConfig.SHORT_MARGIN_PCT}")
    print(f" LSTM_ENABLED                 : {ApexConfig.LSTM_ENABLED}")
    print(f" LSTM_ENSEMBLE_WEIGHT         : {ApexConfig.LSTM_ENSEMBLE_WEIGHT}")
    print(f" Symbols (10)                 : {list(R12_SYMBOLS)}")
    print()

    print(" Fetching 10-symbol panel ...")
    panel = fetch_panel_chunked(R12_SYMBOLS, R12_FULL_START, R12_FULL_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    warm_stats: Optional[Tuple[float, float, float, int]] = None
    if run_wf:
        wf_out = _run_wf(
            panel, PASS_C_ENV,
            "Walk-forward — FIX 1+2+3 (leverage cap + LSTM, 10 symbols, OOS=126)",
        )
        warm_stats = wf_edge_stats(wf_out)
        print(f" Warm-start stats extracted: "
              f"win_rate={warm_stats[0]:.4f}, avg_win={warm_stats[1]:.4f}, "
              f"avg_loss={warm_stats[2]:.4f}, n_folds={warm_stats[3]}")
        print()

    if run_bt:
        _run_pass(
            "PASS A — R12 baseline reproduction (no leverage cap, GBM-only)",
            panel, PASS_A_ENV, max_positions=10, warm_start=warm_stats,
        )
        _run_pass(
            "PASS B — FIX 1+2 only (leverage cap + short margin + block tagging)",
            panel, PASS_B_ENV, max_positions=10, warm_start=warm_stats,
        )
        _run_pass(
            "PASS C — FIX 1+2+3 (+ per-symbol LSTM ensemble @ 0.6)",
            panel, PASS_C_ENV, max_positions=10, warm_start=warm_stats,
        )

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-bt", action="store_true")
    p.add_argument("--no-wf", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    rc = main(run_bt=not ns.no_bt, run_wf=not ns.no_wf)
    sys.exit(rc)
