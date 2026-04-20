"""
scripts/round9_real_data_report.py

Round 9 Addendum driver — replaces the single-purpose momentum z-score
generator used in Round 8 with the full
:class:`backtesting.real_signal_adapter.RealSignalAdapter` (ML + pattern
+ ORB + aggregator). The driver runs three passes over the real yfinance
OHLCV panel:

1. ``run_backtest`` (2023-01-01 → 2023-12-31) **before** any ML models are
   on disk — captured to demonstrate the rule-based floor.
2. ``run_backtest`` (same window) **after** the baseline ML models have
   been trained by :mod:`scripts.train_baseline_models` — the ML primary
   signal is now active.
3. ``run_walk_forward`` over the full 2020 → 2024 span using the
   NaN-safe Sharpe fix landed in :mod:`backtesting.advanced_backtester`.

In addition to the raw backtester metrics the driver inspects the
adapter's ``source_hit_report`` after each run so the final report can
inline an ORB/pattern/ML/momentum frequency table, plus it walks the
backtester's ``trades`` list to produce per-symbol trade counts.

This script is invoked by the Round 9 report generator
(:mod:`scripts.run_round9_report`) and also runnable directly for
interactive use.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.advanced_backtester import AdvancedBacktester
from backtesting.real_signal_adapter import RealSignalAdapter
from config import ApexConfig
from core.logging_config import setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yf_panel(
    symbols: List[str], start: str, end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV bars from yfinance.

    Args:
        symbols: Tickers.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).

    Returns:
        ``{symbol: DataFrame}``. Each frame has columns ``Open``,
        ``High``, ``Low``, ``Close``, ``Volume`` with a tz-naive
        DatetimeIndex.

    Raises:
        RuntimeError: If every symbol returned zero bars.
    """
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(
            start=start, end=end, interval="1d",
            auto_adjust=True, actions=False,
        )
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        panel[sym] = df
    if not panel:
        raise RuntimeError(f"yfinance returned zero bars for {symbols}")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN"
    return f"{x * 100:.4f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN"
    return f"{x:.6f}"


SINGLE_METRIC_KEYS: List[Tuple[str, str, Any]] = [
    ("Total Return", "total_return", fmt_pct),
    ("Sharpe Ratio", "sharpe_ratio", fmt_num),
    ("Max Drawdown", "max_drawdown", fmt_pct),
    ("Win Rate", "win_rate", fmt_pct),
    ("Profit Factor", "profit_factor", fmt_num),
    ("Total Trades", "total_trades", lambda x: f"{int(x)}"),
    ("Final Value", "final_value", lambda x: f"${float(x):,.2f}"),
]


def print_single_metrics(res: Dict[str, Any]) -> None:
    for label, key, fmt in SINGLE_METRIC_KEYS:
        val = res.get(key)
        if val is None:
            print(f" {label:<22}: <missing>")
            continue
        try:
            print(f" {label:<22}: {fmt(float(val))}")
        except (TypeError, ValueError):
            print(f" {label:<22}: {val!r}")


def print_trades_per_symbol(bt: AdvancedBacktester) -> None:
    trades = list(getattr(bt, "trades", []) or [])
    counts: Dict[str, int] = {}
    for t in trades:
        sym = t.get("symbol", "?")
        counts[sym] = counts.get(sym, 0) + 1
    print(" Trades per symbol")
    print(" " + "-" * 68)
    if not counts:
        print("   (no trades recorded)")
    else:
        for sym in sorted(counts.keys()):
            print(f"   {sym:<10} : {counts[sym]}")
    print()


def print_source_hits(adapter: RealSignalAdapter) -> None:
    report = adapter.source_hit_report()
    print(" Signal-source frequency (contributions to non-zero entries)")
    print(" " + "-" * 68)
    for key in ("ml", "pattern", "orb", "momentum_fallback"):
        print(f"   {key:<20} : {int(report.get(key, 0))}")
    print(f"   {'_entries_fired':<20} : {int(report.get('_entries_fired', 0))}")
    print(f"   {'_ml_active':<20} : {int(report.get('_ml_active', 0))}")
    print()


def run_single_backtest(
    label: str,
    panel: Dict[str, pd.DataFrame],
    adapter: RealSignalAdapter,
    start: str,
    end: str,
) -> Dict[str, Any]:
    print("=" * 72)
    print(f" {label}")
    print("=" * 72)
    bt = AdvancedBacktester(initial_capital=100_000)
    adapter.attach_panel(panel)
    res = bt.run_backtest(
        data=panel,
        signal_generator=adapter,
        start_date=start,
        end_date=end,
        position_size_usd=5_000,
        max_positions=3,
    )
    print_single_metrics(res)
    print()
    print_trades_per_symbol(bt)
    print_source_hits(adapter)
    return res


def run_walk_forward(
    panel: Dict[str, pd.DataFrame],
    adapter: RealSignalAdapter,
    start: str,
    end: str,
) -> Dict[str, Any]:
    is_n = int(ApexConfig.WF_IS_BARS)
    oos_n = int(ApexConfig.WF_OOS_BARS)
    step_n = int(ApexConfig.WF_STEP_BARS)

    print("=" * 72)
    print(f" Walk-forward (run_walk_forward) [NaN-safe Sharpe]")
    print("=" * 72)
    print(f"   WF_IS_BARS   : {is_n}")
    print(f"   WF_OOS_BARS  : {oos_n}")
    print(f"   WF_STEP_BARS : {step_n}")
    print()

    bt = AdvancedBacktester(initial_capital=100_000)
    adapter.attach_panel(panel)
    wf_out = bt.run_walk_forward(
        data=panel,
        signal_generator=adapter,
        start_date=start,
        end_date=end,
        position_size_usd=5_000,
        max_positions=3,
    )
    folds: List[Dict[str, Any]] = list(wf_out.get("folds", []))
    agg: Dict[str, Any] = dict(wf_out.get("aggregate", {}))

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
            f" {i:>3}  {f.get('oos_start', ''):<12} {f.get('oos_end', ''):<12} "
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
    for label, key, fmt in [
        ("Folds run", "folds_run", lambda x: f"{int(x)}"),
        ("Mean Sharpe", "mean_sharpe", fmt_num),
        ("Median Sharpe", "median_sharpe", fmt_num),
        ("Compounded Return", "compounded_return", fmt_pct),
        ("Worst Fold Drawdown", "worst_fold_drawdown", fmt_pct),
        ("Positive Folds", "positive_folds",
         lambda x: f"{int(x)} / {len(folds)}"),
        ("Negative Folds", "negative_folds",
         lambda x: f"{int(x)} / {len(folds)}"),
        ("Insufficient-data Folds", "insufficient_data_folds",
         lambda x: f"{int(x)} / {len(folds)}"),
    ]:
        val = agg.get(key)
        if val is None:
            print(f" {label:<24}: <missing>")
            continue
        try:
            print(f" {label:<24}: {fmt(val)}")
        except (TypeError, ValueError):
            print(f" {label:<24}: {val!r}")
    print()
    print_trades_per_symbol(bt)
    print_source_hits(adapter)
    return wf_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    *,
    run_pre_ml: bool = True,
    run_post_ml: bool = True,
    run_wf: bool = True,
) -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    full_start = "2020-01-01"
    full_end = "2025-01-01"
    bt_start = "2023-01-01"
    bt_end = "2023-12-31"

    print("=" * 72)
    print(" Round 9 Addendum — Real-data backtest (RealSignalAdapter)")
    print("=" * 72)
    print(f" Data source    : yfinance ({yf.__version__})")
    print(f" Symbols        : {symbols}")
    print(f" Data range     : {full_start} -> {full_end}")
    print(f" Signal stack   : RealSignalAdapter (ML + pattern + ORB + aggregator)")
    print(
        f" ML TRENDING path: {ApexConfig.ML_MODEL_PATH_TRENDING or '<unset>'}"
    )
    print(
        f" ML MEAN_REV path: {ApexConfig.ML_MODEL_PATH_MEAN_REV or '<unset>'}"
    )
    print(
        f" ML VOLATILE path: {ApexConfig.ML_MODEL_PATH_VOLATILE or '<unset>'}"
    )
    print()

    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_panel(symbols, full_start, full_end)
    for sym, df in panel.items():
        print(f"   {sym}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    if run_pre_ml:
        adapter_pre = RealSignalAdapter(enable_ml=False)
        run_single_backtest(
            label=(
                "1. run_backtest (2023-01-01 -> 2023-12-31) "
                "[pre-ML: rule-based primary]"
            ),
            panel=panel,
            adapter=adapter_pre,
            start=bt_start,
            end=bt_end,
        )

    if run_post_ml:
        adapter_post = RealSignalAdapter()
        # Require ML path to exist on disk for the post-ML pass so an
        # accidentally empty model directory doesn't silently reduce to the
        # pre-ML scenario.
        if not adapter_post.source_hit_report().get("_ml_active"):
            print("=" * 72)
            print(
                " 2. run_backtest [post-ML] SKIPPED — "
                "no trained ML model found on disk."
            )
            print("=" * 72)
            print()
        else:
            run_single_backtest(
                label=(
                    "2. run_backtest (2023-01-01 -> 2023-12-31) "
                    "[post-ML: baseline classifier active]"
                ),
                panel=panel,
                adapter=adapter_post,
                start=bt_start,
                end=bt_end,
            )

    if run_wf:
        adapter_wf = RealSignalAdapter()
        run_walk_forward(
            panel=panel,
            adapter=adapter_wf,
            start=full_start,
            end="2024-12-31",
        )

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-pre-ml", action="store_true")
    p.add_argument("--no-post-ml", action="store_true")
    p.add_argument("--no-wf", action="store_true")
    p.add_argument(
        "--buffered", action="store_true",
        help="Capture stdout and replay at end (used by the report generator)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    if ns.buffered:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(
                run_pre_ml=not ns.no_pre_ml,
                run_post_ml=not ns.no_post_ml,
                run_wf=not ns.no_wf,
            )
        sys.stdout.write(buf.getvalue())
    else:
        rc = main(
            run_pre_ml=not ns.no_pre_ml,
            run_post_ml=not ns.no_post_ml,
            run_wf=not ns.no_wf,
        )
    sys.exit(rc)
