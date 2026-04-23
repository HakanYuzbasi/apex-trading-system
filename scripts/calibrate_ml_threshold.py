"""
scripts/calibrate_ml_threshold.py

Produce a calibrated ``ML_CONFIDENCE_THRESHOLD`` default for the Round 10
gap-fix by replaying the backtest-window (2023-01-01 → 2023-12-31)
through :meth:`backtesting.real_signal_adapter.RealSignalAdapter._ml_primary`
with the entry gate disabled and collecting the distribution of the
non-zero ``|signal|`` scores. The script reports quantiles (P20, P40,
P50, P60, P80) and writes a machine-parseable line starting with
``CALIBRATION_P40=`` that the training / report driver reads verbatim.

The threshold is calibrated on the *same* symbols (AAPL, MSFT, GOOGL)
that the Round 9 backtest ran on; expanding the universe to 7 symbols
in Round 10 happens after calibration so this baseline remains
directly comparable to Round 9's 158-signal / 13-trade figure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.real_signal_adapter import RealSignalAdapter
from core.logging_config import setup_logging


CALIBRATION_SYMBOLS = ("AAPL", "MSFT", "GOOGL")
CALIBRATION_WINDOW_START = "2022-06-01"  # warm-up for SMA50 + ATR
CALIBRATION_WINDOW_END = "2024-01-01"    # yfinance end is exclusive
SCORE_WINDOW_START = pd.Timestamp("2023-01-01")
SCORE_WINDOW_END = pd.Timestamp("2023-12-31")


def fetch_panel(symbols, start: str, end: str) -> Dict[str, pd.DataFrame]:
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
        raise RuntimeError("yfinance returned no bars")
    return panel


def collect_primary_signal_distribution(
    panel: Dict[str, pd.DataFrame], adapter: RealSignalAdapter,
) -> List[float]:
    """
    Walk every bar in the scoring window per symbol and record
    ``|_ml_primary()|`` wherever ML returned a non-zero magnitude.
    """
    scores: List[float] = []
    for sym, df in panel.items():
        # Only score bars inside the reporting window. We still pass the
        # full warm-up history in the ``window`` argument so the indicator
        # features are populated.
        in_window_mask = (df.index >= SCORE_WINDOW_START) & (df.index <= SCORE_WINDOW_END)
        target_dates = df.index[in_window_mask]
        for ts in target_dates:
            window = df.loc[:ts]
            primary, conf = adapter._ml_primary(window)  # intentional private access
            mag = abs(float(primary))
            if mag > 0.0:
                scores.append(mag)
    return scores


def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 10 — ML confidence threshold calibration")
    print("=" * 72)
    print(f" Symbols : {list(CALIBRATION_SYMBOLS)}")
    print(
        f" Window  : {SCORE_WINDOW_START.date()} -> {SCORE_WINDOW_END.date()}"
    )
    print()

    panel = fetch_panel(
        CALIBRATION_SYMBOLS, CALIBRATION_WINDOW_START, CALIBRATION_WINDOW_END,
    )
    for sym, df in panel.items():
        print(f"   {sym}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    adapter = RealSignalAdapter(enable_ml=True)
    report = adapter.source_hit_report()
    if not report.get("_ml_active"):
        print("ERROR: ML model not loaded — set ApexConfig.ML_MODEL_PATH_* "
              "env vars or run scripts/train_baseline_models.py first.")
        return 2

    scores = collect_primary_signal_distribution(panel, adapter)
    if not scores:
        print("ERROR: zero non-trivial primary signals computed.")
        return 3

    arr = np.asarray(scores, dtype=float)
    quantiles = {
        "p10": float(np.quantile(arr, 0.10)),
        "p20": float(np.quantile(arr, 0.20)),
        "p40": float(np.quantile(arr, 0.40)),
        "p50": float(np.quantile(arr, 0.50)),
        "p60": float(np.quantile(arr, 0.60)),
        "p80": float(np.quantile(arr, 0.80)),
    }

    print(f" Collected : {len(arr)} non-zero |primary_signal| scores")
    print(f" mean      : {float(arr.mean()):.6f}")
    print(f" std       : {float(arr.std(ddof=0)):.6f}")
    for k, v in quantiles.items():
        print(f"   {k.upper():<4}    : {v:.6f}")
    print()

    # Machine-parseable line consumed by the report driver.
    print(f"CALIBRATION_P40={quantiles['p40']:.6f}")
    print(f"CALIBRATION_N={len(arr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
