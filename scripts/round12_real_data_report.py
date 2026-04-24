"""
scripts/round12_real_data_report.py

Round 12 driver — universe expansion + warm-started Kelly + macro
regime gate + realistic execution costs on daily bars.

Execution order (per brief):
  1. Retrain regime GBMs on the 10-symbol universe.
  2. Recalibrate ML_CONFIDENCE_THRESHOLD on the 10-symbol 2023 window.
  3. run_walk_forward (2020-2024, daily, 10 symbols).
  4. Warm-start Kelly by seeding ``SignalAggregator._source_pnl_hist``
     from the WF aggregate (win rate, avg winner R, avg loser R).
  5. run_backtest (2023-01-01 -> 2023-12-31).

The driver reports a 5-pass A/B table for the 2023 backtest with each
Round 12 fix toggled independently vs the R11 Pass-0 baseline:
  (0) R11 baseline   : Kelly off, partials off, corr off, macro off
  (1) FIX 1          : recalibrated defaults (Kelly 10/0.35, corr 0.85,
                       partials on, MAX_CONCURRENT_POSITIONS=10)
  (2) FIX 1+2        : adds 10-symbol universe (GLD/TLT/IWM)
  (3) FIX 1+2+4      : adds macro regime gate
  (4) FIX 1+2+3+4+5  : adds warm-start Kelly + exec costs (all fixes)

The walk-forward uses all fixes enabled.
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

R12_SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
    "GLD", "TLT", "IWM",
)
R11_SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
)
R12_FULL_START: str = "2020-01-01"
R12_FULL_END: str = "2025-01-01"
R12_BT_START: str = "2023-01-01"
R12_BT_END: str = "2023-12-31"


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest (chunked + cached; daily bars so 1 chunk is fine, but the
# chunking loop is exercised so UPGRADE 1's infrastructure is in place)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_dir() -> Path:
    p = Path(__file__).resolve().parents[1] / ".cache" / "yfinance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fetch_one_chunk(
    sym: str, chunk_start: str, chunk_end: str, interval: str,
) -> pd.DataFrame:
    last_exc: Exception = RuntimeError("no data")
    for attempt in range(5):
        try:
            df = yf.Ticker(sym).history(
                start=chunk_start, end=chunk_end, interval=interval,
                auto_adjust=True, actions=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_exc = exc
        import time as _t
        _t.sleep(2 ** attempt)
    raise RuntimeError(f"{sym} {chunk_start}-{chunk_end}: {last_exc}")


def fetch_panel_chunked(
    symbols: Tuple[str, ...],
    start: str,
    end: str,
    *,
    interval: Optional[str] = None,
    chunk_days: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Chunked OHLCV download with per-symbol disk cache. On daily bars a
    single chunk is typically sufficient; on 1h bars yfinance requires
    ~60-day windows and this loop reconstructs the full span.
    """
    ivl = interval or str(getattr(ApexConfig, "OHLCV_INTERVAL", "1d"))
    cdays = chunk_days or int(getattr(ApexConfig, "OHLCV_CHUNK_DAYS", 60))
    cache_dir = _cache_dir()
    panel: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        cache_file = cache_dir / f"{sym}_{start}_{end}_{ivl}.parquet"
        if cache_file.is_file():
            try:
                df_cached = pd.read_parquet(cache_file)
                df_cached.index = pd.to_datetime(df_cached.index)
                panel[sym] = df_cached
                continue
            except Exception:
                pass

        frames: List[pd.DataFrame] = []
        ts_start = pd.Timestamp(start)
        ts_end = pd.Timestamp(end)
        cursor = ts_start
        while cursor < ts_end:
            nxt = min(cursor + pd.Timedelta(days=cdays), ts_end)
            chunk = _fetch_one_chunk(sym, str(cursor.date()), str(nxt.date()), ivl)
            frames.append(chunk)
            cursor = nxt
        if not frames:
            raise RuntimeError(f"no data for {sym}")
        merged = pd.concat(frames, axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index = pd.to_datetime(merged.index).tz_localize(None).normalize()
        merged = merged[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        panel[sym] = merged
        try:
            merged.to_parquet(cache_file)
        except Exception:
            pass
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def print_metrics_block(
    label: str, res: Dict[str, Any], trades: List[Dict[str, Any]],
) -> None:
    print("=" * 72)
    print(f" {label}")
    print("=" * 72)
    rows = [
        ("Total Return",              _fmt_pct(res.get("total_return"))),
        ("Sharpe Ratio",               _fmt_num(res.get("sharpe_ratio"))),
        ("Max Drawdown",              _fmt_pct(res.get("max_drawdown"))),
        ("Win Rate",                  _fmt_pct(res.get("win_rate"))),
        ("Profit Factor",              _fmt_num(res.get("profit_factor"))),
        ("Total Trades",               f"{int(res.get('total_trades') or 0)}"),
        ("Avg Winner R",               _fmt_num(res.get("avg_winner_r"))),
        ("Avg Capital Deployed %",    _fmt_pct(res.get("avg_capital_deployed"))),
        ("Avg Portfolio Correlation",  _fmt_num(res.get("avg_portfolio_correlation"))),
        ("Avg Total Cost bps",         _fmt_num(res.get("avg_total_cost_bps"))),
        ("RISK_ON fraction",           _fmt_num(res.get("risk_on_fraction"))),
        ("ORB active",                 f"{bool(res.get('orb_active'))}"),
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
    block_reasons = res.get("entry_block_reasons") or {}
    if block_reasons:
        print()
        print(" Entry-block reasons")
        print(" " + "-" * 68)
        for k, v in sorted(block_reasons.items()):
            print(f"   {k:<30} : {int(v)}")
    # ORB trade count from trades[*]['reason']
    orb_trades = sum(1 for t in trades if "ORB" in str(t.get("reason", "")))
    print(f" ORB trades fired            : {orb_trades}")
    print()


def _reload_apex_config(env_overrides: Dict[str, str]) -> None:
    """
    Re-apply ApexConfig class attributes from ``env_overrides`` + current
    ``os.environ`` so each comparison pass sees exactly the toggles it
    names (bypassing the class-level defaults frozen at import time).
    """
    def _env_raw(name: str) -> Optional[str]:
        if name in env_overrides:
            return env_overrides[name]
        return os.environ.get(name)

    def _env_bool(name: str, default: bool) -> bool:
        raw = _env_raw(name)
        if raw is None or raw == "":
            return default
        return raw.lower() == "true"

    def _env_float(name: str, default: float) -> float:
        raw = _env_raw(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _env_int(name: str, default: int) -> int:
        raw = _env_raw(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    # Round 11 fields.
    ApexConfig.KELLY_ENABLED = _env_bool("APEX_KELLY_ENABLED", True)
    ApexConfig.KELLY_FRACTION_R11 = _env_float("APEX_KELLY_FRACTION_R11", 0.5)
    ApexConfig.KELLY_MIN_SAMPLES = _env_int("APEX_KELLY_MIN_SAMPLES", 10)
    ApexConfig.MIN_POSITION_USD = _env_float("APEX_MIN_POSITION_USD", 500.0)
    ApexConfig.MAX_POSITION_PCT = _env_float("APEX_MAX_POSITION_PCT", 0.35)
    ApexConfig.PARTIAL_EXIT_ENABLED = _env_bool("APEX_PARTIAL_EXIT_ENABLED", True)
    ApexConfig.PARTIAL_EXIT_R1 = _env_float("APEX_PARTIAL_EXIT_R1", 0.33)
    ApexConfig.PARTIAL_EXIT_R2 = _env_float("APEX_PARTIAL_EXIT_R2", 0.33)
    ApexConfig.PARTIAL_EXIT_ATR_MULT = _env_float("APEX_PARTIAL_EXIT_ATR_MULT", 2.0)
    ApexConfig.PARTIAL_EXIT_R_STOP_MULT = _env_float("APEX_PARTIAL_EXIT_R_STOP_MULT", 3.5)
    ApexConfig.CORR_THRESHOLD = _env_float("APEX_CORR_THRESHOLD", 0.85)
    ApexConfig.CORR_LOOKBACK_BARS = _env_int("APEX_CORR_LOOKBACK_BARS", 60)
    # Round 12 fields.
    ApexConfig.MAX_CONCURRENT_POSITIONS = _env_int("APEX_MAX_CONCURRENT_POSITIONS", 10)
    ApexConfig.OHLCV_INTERVAL = _env_raw("APEX_OHLCV_INTERVAL") or "1d"
    ApexConfig.OHLCV_CHUNK_DAYS = _env_int("APEX_OHLCV_CHUNK_DAYS", 60)
    ApexConfig.SIGNAL_HORIZON_BARS = _env_int("APEX_SIGNAL_HORIZON_BARS", 5)
    ApexConfig.KELLY_WARM_START_ENABLED = _env_bool("APEX_KELLY_WARM_START_ENABLED", True)
    ApexConfig.MACRO_REGIME_ENABLED = _env_bool("APEX_MACRO_REGIME_ENABLED", True)
    ApexConfig.RISK_OFF_SIZE_MULT = _env_float("APEX_RISK_OFF_SIZE_MULT", 0.5)
    ApexConfig.MACRO_REGIME_RETURN_LOOKBACK = _env_int(
        "APEX_MACRO_REGIME_RETURN_LOOKBACK", 20,
    )
    ApexConfig.MACRO_REGIME_VOL_MAX = _env_float("APEX_MACRO_REGIME_VOL_MAX", 0.015)
    ApexConfig.SPREAD_BPS_DEFAULT = _env_float("APEX_SPREAD_BPS_DEFAULT", 3.0)
    ApexConfig.SPREAD_BPS_ETF = _env_float("APEX_SPREAD_BPS_ETF", 1.0)
    ApexConfig.MARKET_IMPACT_MULT = _env_float("APEX_MARKET_IMPACT_MULT", 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start Kelly
# ─────────────────────────────────────────────────────────────────────────────

def wf_edge_stats(
    wf_out: Dict[str, Any],
) -> Tuple[float, float, float, int]:
    """
    Derive (win_rate, avg_winner_r, avg_loser_r, n_folds) from a
    walk-forward output. ``avg_loser_r`` is returned as a POSITIVE
    magnitude matching :meth:`SignalAggregator.warm_start_source`.

    Falls back to literature defaults (0.50, 0.012, 0.010) when the WF
    didn't produce enough folds to compute a reliable aggregate.
    """
    folds: List[Dict[str, Any]] = list(wf_out.get("folds", []) or [])
    if not folds:
        return 0.50, 0.012, 0.010, 0
    win_rates = [float(f.get("win_rate", 0.0)) for f in folds if np.isfinite(f.get("win_rate", 0.0))]
    avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.5
    # Approximate avg_win / avg_loss from per-fold total returns split by sign
    positives = [
        float(f.get("total_return", 0.0)) for f in folds
        if float(f.get("total_return", 0.0)) > 0.0
    ]
    negatives = [
        -float(f.get("total_return", 0.0)) for f in folds
        if float(f.get("total_return", 0.0)) < 0.0
    ]
    avg_win = float(np.mean(positives)) if positives else 0.010
    avg_loss = float(np.mean(negatives)) if negatives else 0.008
    # Clamp to realistic ranges so downstream Kelly math is bounded.
    win_rate = float(min(max(avg_win_rate, 0.10), 0.90))
    avg_win = float(min(max(avg_win, 1e-4), 0.20))
    avg_loss = float(min(max(avg_loss, 1e-4), 0.20))
    return win_rate, avg_win, avg_loss, len(folds)


# ─────────────────────────────────────────────────────────────────────────────
# Passes
# ─────────────────────────────────────────────────────────────────────────────

def _run_backtest_pass(
    label: str,
    panel: Dict[str, pd.DataFrame],
    env_overrides: Dict[str, str],
    *,
    max_positions: int,
    warm_start: Optional[Tuple[float, float, float, int]] = None,
) -> Dict[str, Any]:
    """Run one comparison pass with clean env-var scoping."""
    previous: Dict[str, str] = {}
    for k, v in env_overrides.items():
        previous[k] = os.environ.get(k, "")
        os.environ[k] = v
    _reload_apex_config(env_overrides)

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
        print_metrics_block(label, res, bt.trades)
        # Diagnostic — if capital deployed < 10%, print the entry-blocker counts
        # so the operator can see the blocking line directly.
        deployed = float(res.get("avg_capital_deployed") or 0.0)
        if deployed < 0.10:
            print(" WARNING: avg_capital_deployed < 10% — dominant entry block reasons:")
            reasons = res.get("entry_block_reasons") or {}
            for k, v in sorted(reasons.items(), key=lambda kv: -int(kv[1])):
                print(f"   advanced_backtester.py:_check_entries reason={k} count={int(v)}")
            print()
        return res
    finally:
        for k, v in previous.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})


def _run_walk_forward(
    panel: Dict[str, pd.DataFrame],
    env_overrides: Dict[str, str],
    *,
    max_positions: int,
    label: str,
) -> Dict[str, Any]:
    previous: Dict[str, str] = {}
    for k, v in env_overrides.items():
        previous[k] = os.environ.get(k, "")
        os.environ[k] = v
    _reload_apex_config(env_overrides)

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
            max_positions=max_positions,
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
    finally:
        for k, v in previous.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        _reload_apex_config({})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIXES_ENV: Dict[str, str] = {
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
}

R11_BASELINE_ENV: Dict[str, str] = {
    # Everything that Round 11 had disabled in its Pass 0.
    "APEX_KELLY_ENABLED":               "false",
    "APEX_PARTIAL_EXIT_ENABLED":        "false",
    "APEX_CORR_THRESHOLD":              "1.01",   # disables corr gate
    "APEX_MACRO_REGIME_ENABLED":        "false",
    "APEX_SPREAD_BPS_DEFAULT":          "0.0",
    "APEX_SPREAD_BPS_ETF":              "0.0",
    "APEX_MARKET_IMPACT_MULT":          "0.0",
    "APEX_MAX_POSITION_PCT":            "0.15",
    "APEX_PARTIAL_EXIT_R_STOP_MULT":    "2.5",
    "APEX_MAX_CONCURRENT_POSITIONS":    "5",
    "APEX_KELLY_WARM_START_ENABLED":    "false",
}


def main(*, run_bt: bool = True, run_wf: bool = True) -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 12 — Portfolio optimisation (10-symbol universe)")
    print("=" * 72)
    print(f" Data source        : yfinance ({yf.__version__})")
    print(f" Symbols (10)       : {list(R12_SYMBOLS)}")
    print(f" KELLY_MIN_SAMPLES  : {ApexConfig.KELLY_MIN_SAMPLES}")
    print(f" MAX_POSITION_PCT   : {ApexConfig.MAX_POSITION_PCT}")
    print(f" CORR_THRESHOLD     : {ApexConfig.CORR_THRESHOLD}")
    print(f" PARTIAL_EXIT_R_STOP: {ApexConfig.PARTIAL_EXIT_R_STOP_MULT}")
    print(f" MAX_CONCURRENT     : {ApexConfig.MAX_CONCURRENT_POSITIONS}")
    print(f" MACRO_REGIME       : {ApexConfig.MACRO_REGIME_ENABLED}")
    print(f" HIGH_BETA_SYMBOLS  : {ApexConfig.HIGH_BETA_SYMBOLS}")
    print(f" SPREAD_BPS_DEFAULT : {ApexConfig.SPREAD_BPS_DEFAULT}")
    print(f" SPREAD_BPS_ETF     : {ApexConfig.SPREAD_BPS_ETF}")
    print(f" MARKET_IMPACT_MULT : {ApexConfig.MARKET_IMPACT_MULT}")
    print(f" OHLCV_INTERVAL     : {ApexConfig.OHLCV_INTERVAL}")
    print()

    print(" Fetching 10-symbol panel ...")
    panel10 = fetch_panel_chunked(R12_SYMBOLS, R12_FULL_START, R12_FULL_END)
    for sym, df in panel10.items():
        print(f"   {sym:<5}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print()

    print(" Fetching 7-symbol (R11) panel for baseline reproduction ...")
    panel7 = fetch_panel_chunked(R11_SYMBOLS, R12_FULL_START, R12_FULL_END)
    print(f"   cached {len(panel7)} symbols")
    print()

    # ── WF first so we can extract warm-start stats ──────────────────────────
    warm_stats = None
    if run_wf:
        wf_out = _run_walk_forward(
            panel10, ALL_FIXES_ENV, max_positions=10,
            label=(
                "Walk-forward — ALL FIXES "
                "(2020-01-01 -> 2024-12-31, 10 symbols, daily)"
            ),
        )
        warm_stats = wf_edge_stats(wf_out)
        print(f" Warm-start stats extracted : "
              f"win_rate={warm_stats[0]:.4f}, avg_win={warm_stats[1]:.4f}, "
              f"avg_loss={warm_stats[2]:.4f}, n_folds={warm_stats[3]}")
        print()

    if run_bt:
        # PASS 0 — Round 11 Pass-0 baseline reproduction on 7 symbols.
        _run_backtest_pass(
            "PASS 0 — R11 baseline (7 symbols, Round 11 defaults)",
            panel7, R11_BASELINE_ENV, max_positions=5,
        )

        # PASS 1 — Round 12 recalibrated defaults only (7 symbols).
        _run_backtest_pass(
            "PASS 1 — FIX 1 only (recalibrated R11 defaults, 7 symbols)",
            panel7,
            {
                "APEX_KELLY_ENABLED":            "true",
                "APEX_KELLY_MIN_SAMPLES":        "10",
                "APEX_MAX_POSITION_PCT":         "0.35",
                "APEX_PARTIAL_EXIT_ENABLED":     "true",
                "APEX_PARTIAL_EXIT_R_STOP_MULT": "3.5",
                "APEX_CORR_THRESHOLD":           "0.85",
                "APEX_MAX_CONCURRENT_POSITIONS": "10",
                "APEX_MACRO_REGIME_ENABLED":     "false",
                "APEX_KELLY_WARM_START_ENABLED": "false",
                "APEX_SPREAD_BPS_DEFAULT":       "0.0",
                "APEX_SPREAD_BPS_ETF":           "0.0",
                "APEX_MARKET_IMPACT_MULT":       "0.0",
            },
            max_positions=10,
        )

        # PASS 2 — + 10-symbol universe.
        _run_backtest_pass(
            "PASS 2 — FIX 1+2 (recalibrated defaults + 10-symbol universe)",
            panel10,
            {
                "APEX_KELLY_ENABLED":            "true",
                "APEX_KELLY_MIN_SAMPLES":        "10",
                "APEX_MAX_POSITION_PCT":         "0.35",
                "APEX_PARTIAL_EXIT_ENABLED":     "true",
                "APEX_PARTIAL_EXIT_R_STOP_MULT": "3.5",
                "APEX_CORR_THRESHOLD":           "0.85",
                "APEX_MAX_CONCURRENT_POSITIONS": "10",
                "APEX_MACRO_REGIME_ENABLED":     "false",
                "APEX_KELLY_WARM_START_ENABLED": "false",
                "APEX_SPREAD_BPS_DEFAULT":       "0.0",
                "APEX_SPREAD_BPS_ETF":           "0.0",
                "APEX_MARKET_IMPACT_MULT":       "0.0",
            },
            max_positions=10,
        )

        # PASS 3 — + macro regime gate.
        _run_backtest_pass(
            "PASS 3 — FIX 1+2+4 (adds macro regime gate)",
            panel10,
            {
                "APEX_KELLY_ENABLED":            "true",
                "APEX_KELLY_MIN_SAMPLES":        "10",
                "APEX_MAX_POSITION_PCT":         "0.35",
                "APEX_PARTIAL_EXIT_ENABLED":     "true",
                "APEX_PARTIAL_EXIT_R_STOP_MULT": "3.5",
                "APEX_CORR_THRESHOLD":           "0.85",
                "APEX_MAX_CONCURRENT_POSITIONS": "10",
                "APEX_MACRO_REGIME_ENABLED":     "true",
                "APEX_KELLY_WARM_START_ENABLED": "false",
                "APEX_SPREAD_BPS_DEFAULT":       "0.0",
                "APEX_SPREAD_BPS_ETF":           "0.0",
                "APEX_MARKET_IMPACT_MULT":       "0.0",
            },
            max_positions=10,
        )

        # PASS 4 — ALL FIXES: warm-start Kelly + realistic exec costs.
        _run_backtest_pass(
            "PASS 4 — ALL FIXES (warm-start Kelly + exec costs + all above)",
            panel10, ALL_FIXES_ENV,
            max_positions=10,
            warm_start=warm_stats,
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
