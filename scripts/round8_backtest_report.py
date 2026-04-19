"""
scripts/round8_backtest_report.py

Driver that reproduces Round 8 default-config backtest metrics.
Uses the same synthetic-data generation pattern as
``backtesting/advanced_backtester.py::__main__`` but extended to cover
the ``WF_IS_BARS + WF_OOS_BARS + N * WF_STEP_BARS`` span that
``run_walk_forward`` needs to produce multiple folds.

Outputs a deterministic, plain-text report that is captured into
``backtest_results_round8.txt``.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtesting.advanced_backtester import AdvancedBacktester
from config import ApexConfig
from core.logging_config import setup_logging


class MockSignalGenerator:
    """Deterministic stand-in signal generator matching the CLI's shape."""

    def generate_ml_signal(self, symbol: str, prices: Any) -> Dict[str, float]:
        return {"signal": float(np.random.randn() * 0.5)}


def build_synthetic_panel(start: str, end: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Matches the CLI generator in backtesting/advanced_backtester.py."""
    np.random.seed(42)
    dates = pd.date_range(start, end, freq="D")
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        close = 100 + np.random.randn(len(dates)).cumsum()
        data[symbol] = pd.DataFrame(
            {
                "Open": close + np.random.randn(len(dates)) * 0.5,
                "High": close + np.abs(np.random.randn(len(dates))),
                "Low": close - np.abs(np.random.randn(len(dates))),
                "Close": close,
                "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
            },
            index=dates,
        )
    return data


def fmt_pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def fmt_num(x: float) -> str:
    return f"{x:.6f}"


def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    symbols = ["AAPL", "MSFT", "GOOGL"]

    # ── 1. Single run_backtest() with default config ─────────────────────────
    print("═" * 72)
    print(" Round 8 — Default-config backtest (run_backtest)")
    print("═" * 72)
    print(f" Symbols               : {symbols}")
    print(f" Initial capital       : $100,000")
    print(f" Position size         : $5,000")
    print(f" Max concurrent pos.   : 3")
    print(f" Date range            : 2023-01-01 → 2023-12-31")
    print()

    data_single = build_synthetic_panel("2023-01-01", "2024-01-01", symbols)
    bt_single = AdvancedBacktester(initial_capital=100_000)
    res: Dict[str, Any] = bt_single.run_backtest(
        data=data_single,
        signal_generator=MockSignalGenerator(),
        start_date="2023-01-01",
        end_date="2023-12-31",
        position_size_usd=5_000,
        max_positions=3,
    )

    metrics_map = [
        ("Total Return", "total_return", fmt_pct),
        ("Sharpe Ratio", "sharpe_ratio", fmt_num),
        ("Max Drawdown", "max_drawdown", fmt_pct),
        ("Win Rate", "win_rate", fmt_pct),
        ("Profit Factor", "profit_factor", fmt_num),
        ("Total Trades", "total_trades", lambda x: f"{int(x)}"),
        ("Final Value", "final_value", lambda x: f"${float(x):,.2f}"),
    ]
    for label, key, fmt in metrics_map:
        val = res.get(key)
        if val is None:
            print(f" {label:<22}: <missing>")
            continue
        try:
            print(f" {label:<22}: {fmt(float(val))}")
        except (TypeError, ValueError):
            print(f" {label:<22}: {val!r}")
    print()

    # ── 2. run_walk_forward() with config defaults ───────────────────────────
    is_n = int(ApexConfig.WF_IS_BARS)
    oos_n = int(ApexConfig.WF_OOS_BARS)
    step_n = int(ApexConfig.WF_STEP_BARS)
    # Need at least is_n + oos_n + a few steps to produce multiple folds.
    span_days = is_n + oos_n + 3 * step_n + 5
    wf_start = pd.Timestamp("2020-01-01")
    wf_end = wf_start + pd.Timedelta(days=span_days - 1)

    print("═" * 72)
    print(" Round 8 — Walk-forward (run_walk_forward, defaults from config)")
    print("═" * 72)
    print(f" WF_IS_BARS            : {is_n}")
    print(f" WF_OOS_BARS           : {oos_n}")
    print(f" WF_STEP_BARS          : {step_n}")
    print(f" Date range            : {wf_start.date()} → {wf_end.date()}  "
          f"({span_days} days)")
    print()

    data_wf = build_synthetic_panel(
        str(wf_start.date()), str(wf_end.date()), symbols
    )
    bt_wf = AdvancedBacktester(initial_capital=100_000)
    wf_out: Dict[str, Any] = bt_wf.run_walk_forward(
        data=data_wf,
        signal_generator=MockSignalGenerator(),
        start_date=str(wf_start.date()),
        end_date=str(wf_end.date()),
        position_size_usd=5_000,
        max_positions=3,
    )

    folds: List[Dict[str, Any]] = list(wf_out.get("folds", []))
    agg: Dict[str, Any] = dict(wf_out.get("aggregate", {}))

    print(f" Folds produced        : {len(folds)}")
    print()
    print(" Per-fold OOS Sharpe")
    print(" " + "─" * 68)
    print(f" {'#':>3}  {'OOS start':<12} {'OOS end':<12} "
          f"{'Sharpe':>10} {'TotRet':>10} {'MaxDD':>10} "
          f"{'Win%':>8} {'PF':>8} {'Trades':>7}")
    print(" " + "─" * 68)
    for i, f in enumerate(folds, 1):
        print(
            f" {i:>3}  {f.get('oos_start', ''):<12} {f.get('oos_end', ''):<12} "
            f"{float(f.get('sharpe_ratio', 0.0)):>10.4f} "
            f"{float(f.get('total_return', 0.0)) * 100:>9.3f}% "
            f"{float(f.get('max_drawdown', 0.0)) * 100:>9.3f}% "
            f"{float(f.get('win_rate', 0.0)) * 100:>7.2f}% "
            f"{float(f.get('profit_factor', 0.0)):>8.3f} "
            f"{int(f.get('total_trades', 0)):>7d}"
        )
    print()

    print(" Walk-forward aggregate")
    print(" " + "─" * 68)
    for label, key, fmt in [
        ("Folds run", "folds_run", lambda x: f"{int(x)}"),
        ("Mean Sharpe", "mean_sharpe", fmt_num),
        ("Median Sharpe", "median_sharpe", fmt_num),
        ("Compounded Return", "compounded_return", fmt_pct),
        ("Worst Fold Drawdown", "worst_fold_drawdown", fmt_pct),
        ("Positive Folds", "positive_folds",
         lambda x: f"{int(x)} / {len(folds)}"),
    ]:
        val = agg.get(key)
        if val is None:
            print(f" {label:<22}: <missing>")
            continue
        try:
            print(f" {label:<22}: {fmt(val)}")
        except (TypeError, ValueError):
            print(f" {label:<22}: {val!r}")
    print()

    return 0


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main()
    out = buf.getvalue()
    print(out)
    sys.exit(rc)
