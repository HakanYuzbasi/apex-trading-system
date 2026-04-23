"""
scripts/round11_real_data_report.py

Round 11 — Portfolio-optimisation validation driver. Reproduces the
Round 10 baseline on the same 7-symbol universe and then runs four
comparison passes, each toggling exactly one Round 11 fix to isolate
its contribution. Finally a combined "all fixes active" pass runs to
report the integrated effect.

The driver reports, for every pass:

* Total Return, Sharpe, Max Drawdown, Win Rate, Profit Factor
* Total Trades, Avg Winner R-multiple
* Avg Capital Deployed %, Avg Portfolio Correlation

and for the walk-forward pass the full 22-fold table with the NaN-safe
aggregate from Round 9 plus the Round 11 diagnostics.
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


R11_SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
)
R11_FULL_START: str = "2020-01-01"
R11_FULL_END: str = "2025-01-01"
R11_BT_START: str = "2023-01-01"
R11_BT_END: str = "2023-12-31"


def _disk_cache_path() -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_yf_panel(
    symbols: Tuple[str, ...], start: str, end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV bars from yfinance with per-symbol disk cache
    and exponential-backoff retries. The cache key is the tuple
    ``(symbol, start, end)`` so results are reproducible across passes.
    """
    cache_dir = _disk_cache_path()
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        cache_file = cache_dir / f"{sym}_{start}_{end}.parquet"
        if cache_file.is_file():
            try:
                df_cached = pd.read_parquet(cache_file)
                df_cached.index = pd.to_datetime(df_cached.index)
                panel[sym] = df_cached
                continue
            except Exception:
                pass

        last_exc: Exception = RuntimeError(
            f"yfinance returned zero bars for {sym}"
        )
        df: pd.DataFrame = pd.DataFrame()
        for attempt in range(5):
            try:
                df = yf.Ticker(sym).history(
                    start=start, end=end, interval="1d",
                    auto_adjust=True, actions=False,
                )
                if df is not None and not df.empty:
                    break
            except Exception as exc:
                last_exc = exc
            import time as _time
            _time.sleep(2 ** attempt)
        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance failed for {sym} after 5 retries: {last_exc}"
            )
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        panel[sym] = df
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass
    if not panel:
        raise RuntimeError(f"yfinance returned zero bars for {list(symbols)}")
    return panel


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


def print_metrics_block(label: str, res: Dict[str, Any], trades: List[Dict[str, Any]]) -> None:
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
        ("Avg Capital Deployed %",    _fmt_pct(res.get("avg_capital_deployed"))),
        ("Avg Portfolio Correlation", _fmt_num(res.get("avg_portfolio_correlation"))),
    ]
    for lab, val in rows:
        print(f" {lab:<28}: {val}")

    counts: Dict[str, int] = {}
    for t in trades:
        sym = t.get("symbol", "?")
        counts[sym] = counts.get(sym, 0) + 1
    if counts:
        print()
        print(" Trades per symbol")
        print(" " + "-" * 68)
        for sym in sorted(counts.keys()):
            print(f"   {sym:<10} : {counts[sym]}")
    print()


def run_single_pass(
    pass_label: str,
    panel: Dict[str, pd.DataFrame],
    adapter: RealSignalAdapter,
    env_overrides: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run a single backtest with the given environment overrides, then
    reset them so subsequent passes start clean.
    """
    previous: Dict[str, str] = {}
    for k, v in env_overrides.items():
        previous[k] = os.environ.get(k, "")
        os.environ[k] = v

    # Re-read ApexConfig values that are consulted at runtime by the
    # backtester by overriding the class attributes in-memory.
    _reload_apex_config(env_overrides)

    try:
        bt = AdvancedBacktester(initial_capital=100_000)
        adapter.attach_panel(panel)
        res = bt.run_backtest(
            data=panel,
            signal_generator=adapter,
            start_date=R11_BT_START,
            end_date=R11_BT_END,
            position_size_usd=5_000,
            max_positions=5,
        )
        print_metrics_block(pass_label, res, bt.trades)
    finally:
        for k, v in previous.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})
    return res


def _reload_apex_config(env_overrides: Dict[str, str]) -> None:
    """
    Apply/revert the Round 11 ApexConfig class attributes in-memory so
    each comparison pass sees exactly the toggles it names. This mirrors
    ``config.py`` defaults when ``env_overrides`` is empty.
    """
    def _env_bool(name: str, default: bool) -> bool:
        raw = env_overrides.get(name, os.environ.get(name))
        if raw is None:
            return default
        return raw.lower() == "true"

    def _env_float(name: str, default: float) -> float:
        raw = env_overrides.get(name, os.environ.get(name))
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _env_int(name: str, default: int) -> int:
        raw = env_overrides.get(name, os.environ.get(name))
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    ApexConfig.KELLY_ENABLED = _env_bool("APEX_KELLY_ENABLED", True)
    ApexConfig.KELLY_FRACTION_R11 = _env_float("APEX_KELLY_FRACTION_R11", 0.5)
    ApexConfig.KELLY_MIN_SAMPLES = _env_int("APEX_KELLY_MIN_SAMPLES", 30)
    ApexConfig.MIN_POSITION_USD = _env_float("APEX_MIN_POSITION_USD", 500.0)
    ApexConfig.MAX_POSITION_PCT = _env_float("APEX_MAX_POSITION_PCT", 0.15)
    ApexConfig.PARTIAL_EXIT_ENABLED = _env_bool("APEX_PARTIAL_EXIT_ENABLED", True)
    ApexConfig.PARTIAL_EXIT_R1 = _env_float("APEX_PARTIAL_EXIT_R1", 0.33)
    ApexConfig.PARTIAL_EXIT_R2 = _env_float("APEX_PARTIAL_EXIT_R2", 0.33)
    ApexConfig.PARTIAL_EXIT_ATR_MULT = _env_float("APEX_PARTIAL_EXIT_ATR_MULT", 2.0)
    ApexConfig.PARTIAL_EXIT_R_STOP_MULT = _env_float("APEX_PARTIAL_EXIT_R_STOP_MULT", 2.5)
    ApexConfig.CORR_THRESHOLD = _env_float("APEX_CORR_THRESHOLD", 0.70)
    ApexConfig.CORR_LOOKBACK_BARS = _env_int("APEX_CORR_LOOKBACK_BARS", 60)


def run_walk_forward(
    panel: Dict[str, pd.DataFrame],
    adapter: RealSignalAdapter,
    label: str,
) -> Dict[str, Any]:
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

    bt = AdvancedBacktester(initial_capital=100_000)
    adapter.attach_panel(panel)
    wf_out = bt.run_walk_forward(
        data=panel,
        signal_generator=adapter,
        start_date=R11_FULL_START,
        end_date="2024-12-31",
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


def main(*, run_bt: bool = True, run_wf_flag: bool = True) -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 11 — Real-data backtest (Kelly + partial exits + corr)")
    print("=" * 72)
    print(f" Data source          : yfinance ({yf.__version__})")
    print(f" Symbols (7)          : {list(R11_SYMBOLS)}")
    print(f" KELLY_ENABLED        : {ApexConfig.KELLY_ENABLED}")
    print(f" KELLY_FRACTION_R11   : {ApexConfig.KELLY_FRACTION_R11}")
    print(f" KELLY_MIN_SAMPLES    : {ApexConfig.KELLY_MIN_SAMPLES}")
    print(f" MIN_POSITION_USD     : {ApexConfig.MIN_POSITION_USD}")
    print(f" MAX_POSITION_PCT     : {ApexConfig.MAX_POSITION_PCT}")
    print(f" PARTIAL_EXIT_ENABLED : {ApexConfig.PARTIAL_EXIT_ENABLED}")
    print(f" PARTIAL_EXIT_R1      : {ApexConfig.PARTIAL_EXIT_R1}")
    print(f" PARTIAL_EXIT_R2      : {ApexConfig.PARTIAL_EXIT_R2}")
    print(f" CORR_THRESHOLD       : {ApexConfig.CORR_THRESHOLD}")
    print(f" CORR_LOOKBACK_BARS   : {ApexConfig.CORR_LOOKBACK_BARS}")
    print()

    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_panel(R11_SYMBOLS, R11_FULL_START, R11_FULL_END)
    for sym, df in panel.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    # NOTE on methodology: each A/B pass uses a fresh RealSignalAdapter so
    # the source_hits tally is not polluted across runs, and env vars are
    # restored after every pass.
    if run_bt:
        # ── Pass 0: Round 10 baseline reproduction (all Round 11 fixes off) ──
        run_single_pass(
            "PASS 0 — Round 10 baseline (Kelly off, partials off, corr off)",
            panel,
            RealSignalAdapter(),
            {
                "APEX_KELLY_ENABLED":        "false",
                "APEX_PARTIAL_EXIT_ENABLED": "false",
                "APEX_CORR_THRESHOLD":       "1.01",  # >1 disables penalty
            },
        )

        # ── Pass 1: Kelly only ───────────────────────────────────────────────
        run_single_pass(
            "PASS 1 — FIX 1 only (Kelly sizing ON, partials OFF, corr OFF)",
            panel,
            RealSignalAdapter(),
            {
                "APEX_KELLY_ENABLED":        "true",
                "APEX_PARTIAL_EXIT_ENABLED": "false",
                "APEX_CORR_THRESHOLD":       "1.01",
            },
        )

        # ── Pass 2: Partial exits only ───────────────────────────────────────
        run_single_pass(
            "PASS 2 — FIX 2 only (Kelly OFF, partials ON, corr OFF)",
            panel,
            RealSignalAdapter(),
            {
                "APEX_KELLY_ENABLED":        "false",
                "APEX_PARTIAL_EXIT_ENABLED": "true",
                "APEX_CORR_THRESHOLD":       "1.01",
            },
        )

        # ── Pass 3: Correlation only ─────────────────────────────────────────
        run_single_pass(
            "PASS 3 — FIX 3 only (Kelly OFF, partials OFF, corr ON)",
            panel,
            RealSignalAdapter(),
            {
                "APEX_KELLY_ENABLED":        "false",
                "APEX_PARTIAL_EXIT_ENABLED": "false",
                "APEX_CORR_THRESHOLD":       "0.70",
            },
        )

        # ── Pass 4: All fixes combined ──────────────────────────────────────
        run_single_pass(
            "PASS 4 — ALL FIXES (Kelly + partials + correlation gating)",
            panel,
            RealSignalAdapter(),
            {
                "APEX_KELLY_ENABLED":        "true",
                "APEX_PARTIAL_EXIT_ENABLED": "true",
                "APEX_CORR_THRESHOLD":       "0.70",
            },
        )

    if run_wf_flag:
        _reload_apex_config({
            "APEX_KELLY_ENABLED":        "true",
            "APEX_PARTIAL_EXIT_ENABLED": "true",
            "APEX_CORR_THRESHOLD":       "0.70",
        })
        run_walk_forward(
            panel,
            RealSignalAdapter(),
            "Walk-forward — ALL FIXES (2020-01-01 -> 2024-12-31, 7 symbols, OOS=126)",
        )

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-bt", action="store_true")
    p.add_argument("--no-wf", action="store_true")
    p.add_argument("--buffered", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    if ns.buffered:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(run_bt=not ns.no_bt, run_wf_flag=not ns.no_wf)
        sys.stdout.write(buf.getvalue())
    else:
        rc = main(run_bt=not ns.no_bt, run_wf_flag=not ns.no_wf)
    sys.exit(rc)
