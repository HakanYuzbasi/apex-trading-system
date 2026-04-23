"""
scripts/round10_real_data_report.py

Round 10 — Driver for the post-gap-fix real-data backtest. Extends the
Round 9 driver with:

* ``ApexConfig.ML_CONFIDENCE_THRESHOLD``-driven entry gate (PROBLEM 1).
* Per-regime ML classifiers selected by ADX(14) / ATR(14) (PROBLEM 2).
* A 7-symbol universe (AAPL, MSFT, GOOGL, SPY, QQQ, NVDA, AMZN) and a
  126-bar OOS window to lift walk-forward fold density (PROBLEM 3).

The driver executes three passes:

1. ``run_backtest`` 2023-01-01 → 2023-12-31 with the PRE-threshold gate
   (0.30) so the Round 9 baseline can be reproduced for comparison.
2. ``run_backtest`` 2023-01-01 → 2023-12-31 with the NEW calibrated
   threshold (``ApexConfig.ML_CONFIDENCE_THRESHOLD``) and regime-routed
   ML models active.
3. ``run_walk_forward`` 2020-01-01 → 2024-12-31 with the 126-bar OOS
   window over all 7 symbols, using the NaN-safe aggregate from Round 9.

Every pass reports per-symbol trade counts, signal-source frequency
(ml / pattern / orb / momentum_fallback) and, for the ML passes, the
per-regime hit breakdown emitted by
:meth:`backtesting.real_signal_adapter.RealSignalAdapter.regime_hit_report`.
"""
from __future__ import annotations

import argparse
import io
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


ROUND10_SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
)
ROUND10_FULL_START: str = "2020-01-01"
ROUND10_FULL_END: str = "2025-01-01"
ROUND10_BT_START: str = "2023-01-01"
ROUND10_BT_END: str = "2023-12-31"


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yf_panel(
    symbols: Tuple[str, ...], start: str, end: str,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV bars from yfinance."""
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
        raise RuntimeError(f"yfinance returned zero bars for {list(symbols)}")
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


def print_regime_hits(adapter: RealSignalAdapter) -> None:
    regime_report = adapter.regime_hit_report()
    print(" Per-regime ML hits (bars where ML returned a non-zero score)")
    print(" " + "-" * 68)
    if not regime_report:
        print("   (no ML hits recorded — ML inactive or zero signals)")
    else:
        for key in sorted(regime_report.keys()):
            print(f"   {key:<20} : {regime_report[key]}")
    loaded = adapter.loaded_regime_models()
    print(f"   loaded regimes       : {loaded}")
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
    print(f"   entry_threshold = {adapter._entry_threshold:.6f}")
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
    print_regime_hits(adapter)
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
    print(" Walk-forward (run_walk_forward) [NaN-safe Sharpe, 7 symbols]")
    print("=" * 72)
    print(f"   WF_IS_BARS   : {is_n}")
    print(f"   WF_OOS_BARS  : {oos_n}")
    print(f"   WF_STEP_BARS : {step_n}")
    print(f"   symbols      : {list(panel.keys())}")
    print()

    bt = AdvancedBacktester(initial_capital=100_000)
    adapter.attach_panel(panel)
    wf_out = bt.run_walk_forward(
        data=panel,
        signal_generator=adapter,
        start_date=start,
        end_date=end,
        position_size_usd=5_000,
        max_positions=5,
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
    print_regime_hits(adapter)
    return wf_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    *,
    run_pre_threshold: bool = True,
    run_post_threshold: bool = True,
    run_wf: bool = True,
) -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    configured_threshold = float(
        getattr(ApexConfig, "ML_CONFIDENCE_THRESHOLD", 0.19)
    )
    legacy_threshold = 0.30

    print("=" * 72)
    print(" Round 10 — Real-data backtest (regime-routed ML, calibrated gate)")
    print("=" * 72)
    print(f" Data source          : yfinance ({yf.__version__})")
    print(f" Symbols              : {list(ROUND10_SYMBOLS)}")
    print(f" Data range           : {ROUND10_FULL_START} -> {ROUND10_FULL_END}")
    print(f" Legacy threshold     : {legacy_threshold}")
    print(f" Configured threshold : {configured_threshold}")
    print(f" ML TRENDING path     : {ApexConfig.ML_MODEL_PATH_TRENDING or '<unset>'}")
    print(f" ML MEAN_REV path     : {ApexConfig.ML_MODEL_PATH_MEAN_REV or '<unset>'}")
    print(f" ML VOLATILE path     : {ApexConfig.ML_MODEL_PATH_VOLATILE or '<unset>'}")
    print()

    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_panel(ROUND10_SYMBOLS, ROUND10_FULL_START, ROUND10_FULL_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    if run_pre_threshold:
        adapter_pre = RealSignalAdapter(entry_threshold=legacy_threshold)
        run_single_backtest(
            label=(
                f"1. run_backtest ({ROUND10_BT_START} -> {ROUND10_BT_END}) "
                f"[PRE — legacy 0.30 gate, regime models active]"
            ),
            panel=panel,
            adapter=adapter_pre,
            start=ROUND10_BT_START,
            end=ROUND10_BT_END,
        )

    if run_post_threshold:
        adapter_post = RealSignalAdapter()  # uses ML_CONFIDENCE_THRESHOLD
        run_single_backtest(
            label=(
                f"2. run_backtest ({ROUND10_BT_START} -> {ROUND10_BT_END}) "
                f"[POST — calibrated {configured_threshold:.3f} gate]"
            ),
            panel=panel,
            adapter=adapter_post,
            start=ROUND10_BT_START,
            end=ROUND10_BT_END,
        )

    if run_wf:
        adapter_wf = RealSignalAdapter()
        run_walk_forward(
            panel=panel,
            adapter=adapter_wf,
            start=ROUND10_FULL_START,
            end="2024-12-31",
        )

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-pre", action="store_true")
    p.add_argument("--no-post", action="store_true")
    p.add_argument("--no-wf", action="store_true")
    p.add_argument("--buffered", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    if ns.buffered:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(
                run_pre_threshold=not ns.no_pre,
                run_post_threshold=not ns.no_post,
                run_wf=not ns.no_wf,
            )
        sys.stdout.write(buf.getvalue())
    else:
        rc = main(
            run_pre_threshold=not ns.no_pre,
            run_post_threshold=not ns.no_post,
            run_wf=not ns.no_wf,
        )
    sys.exit(rc)
