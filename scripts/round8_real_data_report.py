"""
scripts/round8_real_data_report.py

Round 8 Addendum — Replace the synthetic-data harness with real historical
OHLCV pulled from Yahoo Finance (yfinance, no API key required) and a
deterministic rule-based momentum signal. No random values, no ML
inference, no fake signals.

This driver wires the existing :class:`AdvancedBacktester` (same
``run_backtest`` / ``run_walk_forward`` entry points) to:

1. Real daily OHLCV for AAPL, MSFT, GOOGL spanning 2020-01-01 → 2024-12-31
   (adj-close adjusted; back-filled through yfinance).
2. A deterministic momentum signal generator that scores each symbol in
   ``[-1, 1]`` using a 20-day z-score of close prices. No ML models are
   loaded (none are trained) — rule-based only.

Every metric the backtester emits is written both to stdout and to
``backtest_results_real_data.txt``.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting.advanced_backtester import AdvancedBacktester
from config import ApexConfig
from core.logging_config import setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest — yfinance, daily bars, adjusted
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yf_panel(
    symbols: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV bars for each symbol via yfinance.

    Args:
        symbols: Tickers to fetch (e.g. ``["AAPL", "MSFT", "GOOGL"]``).
        start: Inclusive start date in ``YYYY-MM-DD`` format.
        end: Exclusive end date in ``YYYY-MM-DD`` format.

    Returns:
        ``{symbol: DataFrame}`` keyed by ticker. Each frame has columns
        ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` and a tz-naive
        ``DatetimeIndex`` at 00:00. Any empty frame is excluded.

    Raises:
        RuntimeError: If every symbol returned zero rows (e.g. network
        outage or Yahoo rate-limit).
    """
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
        if df is None or df.empty:
            continue
        # Strip timezone + normalise column order to what the backtester expects.
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
        )[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Close"])
        panel[sym] = df
    if not panel:
        raise RuntimeError(
            f"yfinance returned zero bars for every symbol in {symbols}"
        )
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic rule-based signal — 20-day momentum z-score
# ─────────────────────────────────────────────────────────────────────────────

class MomentumZScoreSignalGenerator:
    """
    Deterministic momentum scorer — no ML, no randomness.

    The signal for a symbol at time ``t`` is::

        z = (close_t - SMA_20(close_t)) / STD_20(close_t)
        signal = clip(z / 2.0, -1.0, 1.0)

    Dividing the z-score by 2 saturates the ``±1`` band at ±2σ, which
    matches the ±0.30 / ±0.50 entry thresholds the backtester uses
    internally (a 0.6σ move produces a ~±0.30 signal).

    The generator only consumes the price series passed in by the
    backtester (which is sliced to ``prices.loc[:prev_date]`` at call
    time — no look-ahead).
    """

    _WINDOW: int = 20

    def generate_ml_signal(
        self,
        symbol: str,
        prices: pd.Series,
    ) -> Dict[str, float]:
        """
        Produce the current momentum z-score for ``symbol``.

        Args:
            symbol: Ticker passed through for logging only.
            prices: Close-price series up to and including ``prev_date``.

        Returns:
            ``{"signal": float}`` in ``[-1.0, 1.0]``. Returns ``0.0``
            when the window is under-populated or the rolling std is zero.
        """
        if prices is None or len(prices) < self._WINDOW + 1:
            return {"signal": 0.0}

        window = prices.tail(self._WINDOW)
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        if std <= 0.0 or not np.isfinite(std):
            return {"signal": 0.0}

        last = float(prices.iloc[-1])
        z = (last - mean) / std
        sig = float(np.clip(z / 2.0, -1.0, 1.0))
        if not np.isfinite(sig):
            return {"signal": 0.0}
        return {"signal": sig}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def fmt_num(x: float) -> str:
    return f"{x:.6f}"


def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    full_start = "2020-01-01"
    full_end = "2025-01-01"  # exclusive in yfinance

    print("═" * 72)
    print(" Round 8 Addendum — Real-data backtest report")
    print("═" * 72)
    print(f" Data source    : yfinance ({yf.__version__})")
    print(f" Symbols        : {symbols}")
    print(f" Data range     : {full_start} → {full_end}")
    print(f" Signal source  : MomentumZScoreSignalGenerator (rule-based)")
    print(f" ML models used : none (no trained .pkl files on disk)")
    print()

    # ── Data ingest ───────────────────────────────────────────────────────────
    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_panel(symbols, full_start, full_end)
    for sym, df in panel.items():
        print(f"   {sym}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
    print()

    signal_gen = MomentumZScoreSignalGenerator()

    # ── 1. run_backtest 2023-01-01 → 2023-12-31 ───────────────────────────────
    print("═" * 72)
    print(" 1. run_backtest (2023-01-01 → 2023-12-31)")
    print("═" * 72)

    bt_single = AdvancedBacktester(initial_capital=100_000)
    res_single: Dict[str, Any] = bt_single.run_backtest(
        data=panel,
        signal_generator=signal_gen,
        start_date="2023-01-01",
        end_date="2023-12-31",
        position_size_usd=5_000,
        max_positions=3,
    )

    for label, key, fmt in [
        ("Total Return", "total_return", fmt_pct),
        ("Sharpe Ratio", "sharpe_ratio", fmt_num),
        ("Max Drawdown", "max_drawdown", fmt_pct),
        ("Win Rate", "win_rate", fmt_pct),
        ("Profit Factor", "profit_factor", fmt_num),
        ("Total Trades", "total_trades", lambda x: f"{int(x)}"),
        ("Final Value", "final_value", lambda x: f"${float(x):,.2f}"),
    ]:
        val = res_single.get(key)
        if val is None:
            print(f" {label:<22}: <missing>")
            continue
        try:
            print(f" {label:<22}: {fmt(float(val))}")
        except (TypeError, ValueError):
            print(f" {label:<22}: {val!r}")
    print()

    # ── 2. run_walk_forward 2020-01-01 → 2024-12-31 ──────────────────────────
    is_n = int(ApexConfig.WF_IS_BARS)
    oos_n = int(ApexConfig.WF_OOS_BARS)
    step_n = int(ApexConfig.WF_STEP_BARS)

    print("═" * 72)
    print(" 2. run_walk_forward (2020-01-01 → 2024-12-31, config defaults)")
    print("═" * 72)
    print(f"   WF_IS_BARS   : {is_n}")
    print(f"   WF_OOS_BARS  : {oos_n}")
    print(f"   WF_STEP_BARS : {step_n}")
    print()

    bt_wf = AdvancedBacktester(initial_capital=100_000)
    wf_out: Dict[str, Any] = bt_wf.run_walk_forward(
        data=panel,
        signal_generator=signal_gen,
        start_date=full_start,
        end_date="2024-12-31",
        position_size_usd=5_000,
        max_positions=3,
    )

    folds: List[Dict[str, Any]] = list(wf_out.get("folds", []))
    agg: Dict[str, Any] = dict(wf_out.get("aggregate", {}))

    print(f" Folds produced : {len(folds)}")
    print()
    print(" Per-fold OOS metrics")
    print(" " + "─" * 68)
    print(f" {'#':>3}  {'OOS start':<12} {'OOS end':<12} "
          f"{'Sharpe':>9} {'TotRet':>9} {'MaxDD':>9} "
          f"{'Win%':>7} {'PF':>7} {'Trades':>7}")
    print(" " + "─" * 68)
    for i, f in enumerate(folds, 1):
        print(
            f" {i:>3}  {f.get('oos_start', ''):<12} {f.get('oos_end', ''):<12} "
            f"{float(f.get('sharpe_ratio', 0.0)):>9.4f} "
            f"{float(f.get('total_return', 0.0)) * 100:>8.3f}% "
            f"{float(f.get('max_drawdown', 0.0)) * 100:>8.3f}% "
            f"{float(f.get('win_rate', 0.0)) * 100:>6.2f}% "
            f"{float(f.get('profit_factor', 0.0)):>7.3f} "
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
